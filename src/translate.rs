//! Gemini REST <-> OpenAI chat-completions translation.
//!
//! Kept separate from the HTTP layer so every shape can be unit-tested without
//! a server or a GPU. Everything here is pure: JSON in, JSON out.
//!
//! The two formats disagree in three places that matter, and each one has bitten
//! this rig already:
//!   * Google nests text in `contents[].parts[]`; OpenAI has flat `messages[]`.
//!   * A tool call is a `functionCall` *part* in Google and a `tool_calls` entry
//!     with JSON-string arguments in OpenAI.
//!   * A reasoning model streams its first tokens into `reasoning_content`, not
//!     `content` — waiting on `content` alone looks like a hung model.

use serde_json::{json, Map, Value};

/// Google `role` values are `user` / `model`; OpenAI uses `user` / `assistant`.
fn role_to_openai(role: &str) -> &str {
    match role {
        "model" => "assistant",
        other => other,
    }
}

/// Collect the text of every `text` part, ignoring parts we surface elsewhere.
fn parts_text(parts: &[Value]) -> String {
    parts
        .iter()
        .filter_map(|p| p.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("")
}

/// Gemini `GenerateContentRequest` -> OpenAI `/v1/chat/completions` body.
pub fn request_to_openai(model: &str, req: &Value, stream: bool) -> Value {
    let mut messages: Vec<Value> = Vec::new();

    // systemInstruction is a Content, not a string — it carries parts like any
    // other turn.
    if let Some(sys) = req.get("systemInstruction") {
        let text = sys
            .get("parts")
            .and_then(Value::as_array)
            .map(|p| parts_text(p))
            .unwrap_or_default();
        if !text.is_empty() {
            messages.push(json!({"role": "system", "content": text}));
        }
    }

    for content in req
        .get("contents")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[])
    {
        let role = content.get("role").and_then(Value::as_str).unwrap_or("user");
        let empty = Vec::new();
        let parts = content
            .get("parts")
            .and_then(Value::as_array)
            .unwrap_or(&empty);

        // A functionResponse part becomes its own OpenAI `tool` message; it
        // cannot ride along inside an assistant or user turn.
        let mut tool_msgs: Vec<Value> = Vec::new();
        let mut tool_calls: Vec<Value> = Vec::new();

        for part in parts {
            if let Some(fr) = part.get("functionResponse") {
                let name = fr.get("name").and_then(Value::as_str).unwrap_or("");
                let response = fr.get("response").cloned().unwrap_or(Value::Null);
                tool_msgs.push(json!({
                    "role": "tool",
                    // OpenAI keys the reply by call id; Gemini keys it by name,
                    // so the name is the only stable identifier we have.
                    "tool_call_id": name,
                    "content": response.to_string(),
                }));
            }
            if let Some(fc) = part.get("functionCall") {
                let name = fc.get("name").and_then(Value::as_str).unwrap_or("");
                let args = fc.get("args").cloned().unwrap_or_else(|| json!({}));
                tool_calls.push(json!({
                    "id": name,
                    "type": "function",
                    "function": {
                        "name": name,
                        // OpenAI wants the arguments as a JSON *string*.
                        "arguments": args.to_string(),
                    }
                }));
            }
        }

        let text = parts_text(parts);
        if !tool_calls.is_empty() {
            let mut msg = Map::new();
            msg.insert("role".into(), json!("assistant"));
            msg.insert(
                "content".into(),
                if text.is_empty() { Value::Null } else { json!(text) },
            );
            msg.insert("tool_calls".into(), json!(tool_calls));
            messages.push(Value::Object(msg));
        } else if !text.is_empty() {
            messages.push(json!({"role": role_to_openai(role), "content": text}));
        }
        messages.extend(tool_msgs);
    }

    let mut body = Map::new();
    body.insert("model".into(), json!(model));
    body.insert("messages".into(), json!(messages));
    body.insert("stream".into(), json!(stream));

    if let Some(cfg) = req.get("generationConfig") {
        for (from, to) in [
            ("temperature", "temperature"),
            ("topP", "top_p"),
            ("topK", "top_k"),
            ("maxOutputTokens", "max_tokens"),
            ("seed", "seed"),
        ] {
            if let Some(v) = cfg.get(from) {
                body.insert(to.into(), v.clone());
            }
        }
        if let Some(stops) = cfg.get("stopSequences") {
            body.insert("stop".into(), stops.clone());
        }
    }

    // Tool declarations: Gemini groups them under tools[].functionDeclarations.
    let mut tools: Vec<Value> = Vec::new();
    for t in req
        .get("tools")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[])
    {
        for decl in t
            .get("functionDeclarations")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or(&[])
        {
            let params = decl
                .get("parametersJsonSchema")
                .or_else(|| decl.get("parameters"))
                .cloned()
                .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
            tools.push(json!({
                "type": "function",
                "function": {
                    "name": decl.get("name").cloned().unwrap_or(Value::Null),
                    "description": decl.get("description").cloned().unwrap_or(json!("")),
                    "parameters": params,
                }
            }));
        }
    }
    if !tools.is_empty() {
        body.insert("tools".into(), json!(tools));
    }

    Value::Object(body)
}

/// Build a Gemini `Candidate` from an OpenAI assistant message.
fn candidate_from_message(msg: &Value, finish: Option<&str>) -> Value {
    let mut parts: Vec<Value> = Vec::new();

    // Reasoning first: a thinking model puts its chain here, and dropping it
    // silently is how a full response ends up looking empty.
    for key in ["reasoning_content", "reasoning"] {
        if let Some(text) = msg.get(key).and_then(Value::as_str) {
            if !text.is_empty() {
                parts.push(json!({"text": text, "thought": true}));
            }
        }
    }
    if let Some(text) = msg.get("content").and_then(Value::as_str) {
        if !text.is_empty() {
            parts.push(json!({"text": text}));
        }
    }
    for call in msg
        .get("tool_calls")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[])
    {
        let f = call.get("function").cloned().unwrap_or(Value::Null);
        let name = f.get("name").and_then(Value::as_str).unwrap_or("");
        // `arguments` is a JSON string. A weaker model sometimes emits invalid
        // JSON here; keep it as a raw string rather than dropping the call, so
        // the agent sees a bad-arguments error instead of silence.
        let args = f
            .get("arguments")
            .and_then(Value::as_str)
            .map(|s| serde_json::from_str::<Value>(s).unwrap_or_else(|_| json!({"_raw": s})))
            .unwrap_or_else(|| json!({}));
        parts.push(json!({"functionCall": {"name": name, "args": args}}));
    }

    json!({
        "content": {"role": "model", "parts": parts},
        "finishReason": match finish {
            Some("stop") | None => "STOP",
            Some("length") => "MAX_TOKENS",
            Some("tool_calls") => "STOP",
            Some(other) => other,
        },
        "index": 0,
    })
}

/// OpenAI completion -> Gemini `GenerateContentResponse`.
pub fn response_to_gemini(resp: &Value) -> Value {
    let choice = resp
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|c| c.first())
        .cloned()
        .unwrap_or(Value::Null);
    let msg = choice.get("message").cloned().unwrap_or(Value::Null);
    let finish = choice.get("finish_reason").and_then(Value::as_str);

    let usage = resp.get("usage").cloned().unwrap_or(Value::Null);
    json!({
        "candidates": [candidate_from_message(&msg, finish)],
        "usageMetadata": {
            "promptTokenCount": usage.get("prompt_tokens").cloned().unwrap_or(json!(0)),
            "candidatesTokenCount": usage.get("completion_tokens").cloned().unwrap_or(json!(0)),
            "totalTokenCount": usage.get("total_tokens").cloned().unwrap_or(json!(0)),
        },
        "modelVersion": resp.get("model").cloned().unwrap_or(Value::Null),
    })
}

/// One OpenAI SSE `delta` chunk -> one Gemini streaming response, or None when
/// the chunk carries nothing renderable (role-only openers, empty keepalives).
pub fn chunk_to_gemini(chunk: &Value) -> Option<Value> {
    let choice = chunk
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|c| c.first())?;
    let delta = choice.get("delta")?;
    let finish = choice.get("finish_reason").and_then(Value::as_str);

    let mut parts: Vec<Value> = Vec::new();
    for key in ["reasoning_content", "reasoning"] {
        if let Some(t) = delta.get(key).and_then(Value::as_str) {
            if !t.is_empty() {
                parts.push(json!({"text": t, "thought": true}));
            }
        }
    }
    if let Some(t) = delta.get("content").and_then(Value::as_str) {
        if !t.is_empty() {
            parts.push(json!({"text": t}));
        }
    }
    for call in delta
        .get("tool_calls")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[])
    {
        let f = call.get("function").cloned().unwrap_or(Value::Null);
        let name = f.get("name").and_then(Value::as_str).unwrap_or("");
        let args = f
            .get("arguments")
            .and_then(Value::as_str)
            .map(|s| serde_json::from_str::<Value>(s).unwrap_or_else(|_| json!({"_raw": s})))
            .unwrap_or_else(|| json!({}));
        parts.push(json!({"functionCall": {"name": name, "args": args}}));
    }

    if parts.is_empty() && finish.is_none() {
        return None;
    }

    let mut candidate = Map::new();
    candidate.insert("content".into(), json!({"role": "model", "parts": parts}));
    candidate.insert("index".into(), json!(0));
    if let Some(f) = finish {
        candidate.insert(
            "finishReason".into(),
            json!(match f {
                "length" => "MAX_TOKENS",
                _ => "STOP",
            }),
        );
    }
    Some(json!({"candidates": [Value::Object(candidate)]}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_turn_round_trips() {
        let req = json!({"contents": [{"role": "user", "parts": [{"text": "hi"}]}]});
        let out = request_to_openai("gemma", &req, false);
        assert_eq!(out["messages"][0]["role"], "user");
        assert_eq!(out["messages"][0]["content"], "hi");
    }

    #[test]
    fn model_role_becomes_assistant() {
        let req = json!({"contents": [{"role": "model", "parts": [{"text": "ok"}]}]});
        let out = request_to_openai("gemma", &req, false);
        assert_eq!(out["messages"][0]["role"], "assistant");
    }

    #[test]
    fn system_instruction_leads() {
        let req = json!({
            "systemInstruction": {"parts": [{"text": "be brief"}]},
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}]
        });
        let out = request_to_openai("gemma", &req, false);
        assert_eq!(out["messages"][0]["role"], "system");
        assert_eq!(out["messages"][0]["content"], "be brief");
    }

    #[test]
    fn function_call_becomes_tool_call_with_string_args() {
        let req = json!({"contents": [{"role": "model", "parts": [
            {"functionCall": {"name": "ls", "args": {"path": "/tmp"}}}
        ]}]});
        let out = request_to_openai("gemma", &req, false);
        let call = &out["messages"][0]["tool_calls"][0];
        assert_eq!(call["function"]["name"], "ls");
        // Must be a STRING, not an object — this is the classic mistranslation.
        assert!(call["function"]["arguments"].is_string());
    }

    #[test]
    fn function_response_becomes_its_own_tool_message() {
        let req = json!({"contents": [{"role": "user", "parts": [
            {"functionResponse": {"name": "ls", "response": {"out": "a\nb"}}}
        ]}]});
        let out = request_to_openai("gemma", &req, false);
        assert_eq!(out["messages"][0]["role"], "tool");
        assert_eq!(out["messages"][0]["tool_call_id"], "ls");
    }

    #[test]
    fn tool_declarations_translate() {
        let req = json!({"tools": [{"functionDeclarations": [
            {"name": "grep", "description": "search", "parametersJsonSchema": {"type": "object"}}
        ]}], "contents": []});
        let out = request_to_openai("gemma", &req, false);
        assert_eq!(out["tools"][0]["function"]["name"], "grep");
        assert_eq!(out["tools"][0]["type"], "function");
    }

    #[test]
    fn generation_config_maps_to_openai_names() {
        let req = json!({"contents": [], "generationConfig":
            {"temperature": 0.5, "maxOutputTokens": 128, "topP": 0.9}});
        let out = request_to_openai("gemma", &req, false);
        assert_eq!(out["temperature"], 0.5);
        assert_eq!(out["max_tokens"], 128);
        assert_eq!(out["top_p"], 0.9);
    }

    #[test]
    fn response_carries_text_and_usage() {
        let resp = json!({
            "choices": [{"message": {"role": "assistant", "content": "hello"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
        });
        let out = response_to_gemini(&resp);
        assert_eq!(out["candidates"][0]["content"]["parts"][0]["text"], "hello");
        assert_eq!(out["usageMetadata"]["totalTokenCount"], 4);
        assert_eq!(out["candidates"][0]["finishReason"], "STOP");
    }

    #[test]
    fn reasoning_content_is_kept_as_a_thought_part() {
        let resp = json!({"choices": [{"message":
            {"reasoning_content": "thinking...", "content": "answer"}}]});
        let out = response_to_gemini(&resp);
        let parts = &out["candidates"][0]["content"]["parts"];
        assert_eq!(parts[0]["thought"], true);
        assert_eq!(parts[1]["text"], "answer");
    }

    #[test]
    fn length_finish_becomes_max_tokens() {
        let resp = json!({"choices": [{"message": {"content": "x"},
                                       "finish_reason": "length"}]});
        assert_eq!(response_to_gemini(&resp)["candidates"][0]["finishReason"], "MAX_TOKENS");
    }

    #[test]
    fn bad_tool_arguments_survive_as_raw() {
        let resp = json!({"choices": [{"message": {"tool_calls": [
            {"function": {"name": "f", "arguments": "{not json"}}
        ]}}]});
        let out = response_to_gemini(&resp);
        let call = &out["candidates"][0]["content"]["parts"][0]["functionCall"];
        assert_eq!(call["name"], "f");
        assert_eq!(call["args"]["_raw"], "{not json");
    }

    #[test]
    fn stream_chunk_with_text() {
        let chunk = json!({"choices": [{"delta": {"content": "he"}}]});
        let out = chunk_to_gemini(&chunk).expect("should render");
        assert_eq!(out["candidates"][0]["content"]["parts"][0]["text"], "he");
    }

    #[test]
    fn stream_chunk_reasoning_first_token_counts() {
        let chunk = json!({"choices": [{"delta": {"reasoning_content": "hm"}}]});
        let out = chunk_to_gemini(&chunk).expect("reasoning is a real first token");
        assert_eq!(out["candidates"][0]["content"]["parts"][0]["thought"], true);
    }

    #[test]
    fn empty_role_only_chunk_is_dropped() {
        let chunk = json!({"choices": [{"delta": {"role": "assistant"}}]});
        assert!(chunk_to_gemini(&chunk).is_none());
    }

    #[test]
    fn final_chunk_with_finish_reason_is_kept() {
        let chunk = json!({"choices": [{"delta": {}, "finish_reason": "stop"}]});
        let out = chunk_to_gemini(&chunk).expect("finish must reach the client");
        assert_eq!(out["candidates"][0]["finishReason"], "STOP");
    }
}
