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

/// A Gemini media part -> the OpenAI content-part that carries the same bytes.
///
/// Gemini inlines media as base64 (`inlineData`) or references it by URI
/// (`fileData`); OpenAI takes a URL, and a `data:` URL covers both. vLLM keys
/// the part type off the modality, so the mime type decides between
/// `image_url` and `audio_url` — sending audio as `image_url` is rejected by
/// the server rather than silently ignored.
fn media_part(part: &Value) -> Option<Value> {
    let (mime, url) = if let Some(inline) = part.get("inlineData") {
        let mime = inline.get("mimeType").and_then(Value::as_str).unwrap_or("application/octet-stream");
        let data = inline.get("data").and_then(Value::as_str)?;
        (mime.to_string(), format!("data:{mime};base64,{data}"))
    } else if let Some(file) = part.get("fileData") {
        let mime = file.get("mimeType").and_then(Value::as_str).unwrap_or("");
        let uri = file.get("fileUri").and_then(Value::as_str)?;
        (mime.to_string(), uri.to_string())
    } else {
        return None;
    };

    Some(if mime.starts_with("audio/") {
        json!({"type": "audio_url", "audio_url": {"url": url}})
    } else if mime.starts_with("video/") {
        json!({"type": "video_url", "video_url": {"url": url}})
    } else {
        json!({"type": "image_url", "image_url": {"url": url}})
    })
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
        let mut media: Vec<Value> = Vec::new();

        for part in parts {
            if let Some(m) = media_part(part) {
                media.push(m);
            }
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
        } else if !media.is_empty() {
            // With attachments, `content` must be an array of typed parts; a
            // bare string has nowhere to put the image.
            let mut content: Vec<Value> = Vec::new();
            if !text.is_empty() {
                content.push(json!({"type": "text", "text": text}));
            }
            content.extend(media);
            messages.push(json!({"role": role_to_openai(role), "content": content}));
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

        // Structured output. The CLI does not only use this for user-facing
        // work: its next-speaker check, loop detection and chat compression all
        // ask for JSON and parse the reply. Without a response_format the model
        // answers in prose, JSON.parse throws, and those subsystems fail in
        // ways that look like unrelated bugs.
        let wants_json = cfg
            .get("responseMimeType")
            .and_then(Value::as_str)
            .map(|m| m.contains("json"))
            .unwrap_or(false);
        let schema = cfg.get("responseJsonSchema").or_else(|| cfg.get("responseSchema"));
        if let Some(schema) = schema {
            body.insert(
                "response_format".into(),
                json!({
                    "type": "json_schema",
                    "json_schema": {"name": "response", "schema": schema, "strict": true}
                }),
            );
        } else if wants_json {
            body.insert("response_format".into(), json!({"type": "json_object"}));
        }
    }

    // A streamed response carries no token counts unless asked; the CLI shows
    // them and budgets against them, so a stream without usage reads as zero.
    if stream {
        body.insert("stream_options".into(), json!({"include_usage": true}));
    }

    // toolConfig decides whether the model may call a tool, must call one, or
    // must not. Dropping it turns "must not" into "may", which is how a
    // summarisation call comes back as a tool call instead of a summary.
    if let Some(mode) = req
        .get("toolConfig")
        .and_then(|c| c.get("functionCallingConfig"))
        .and_then(|c| c.get("mode"))
        .and_then(Value::as_str)
    {
        let choice = match mode.to_ascii_uppercase().as_str() {
            "NONE" => Some(json!("none")),
            "ANY" => Some(json!("required")),
            "AUTO" => Some(json!("auto")),
            _ => None,
        };
        if let Some(choice) = choice {
            body.insert("tool_choice".into(), choice);
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

/// Accumulates tool calls across SSE chunks.
///
/// OpenAI streams a tool call in pieces: the first delta carries `index`, `id`
/// and `function.name`, and later deltas append fragments of
/// `function.arguments` with no name at all. Gemini has no equivalent — a
/// `functionCall` part is atomic. Translating each chunk independently emitted
/// one nameless call per fragment, which reached the CLI as `generic_tool` and
/// made every tool call fail.
#[derive(Default)]
pub struct ToolCallAccumulator {
    calls: Vec<(String, String)>, // (name, arguments-so-far), by index
}

impl ToolCallAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold one delta's tool_calls in. Returns nothing — completed calls are
    /// drained by `finish()` once the stream reports it is done.
    fn absorb(&mut self, delta: &Value) {
        for call in delta
            .get("tool_calls")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or(&[])
        {
            let idx = call.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
            while self.calls.len() <= idx {
                self.calls.push((String::new(), String::new()));
            }
            let f = call.get("function");
            if let Some(name) = f.and_then(|f| f.get("name")).and_then(Value::as_str) {
                if !name.is_empty() {
                    self.calls[idx].0 = name.to_string();
                }
            }
            if let Some(args) = f.and_then(|f| f.get("arguments")).and_then(Value::as_str) {
                self.calls[idx].1.push_str(args);
            }
        }
    }

    fn is_empty(&self) -> bool {
        self.calls.is_empty()
    }

    /// The assembled calls as Gemini `functionCall` parts.
    fn drain_parts(&mut self) -> Vec<Value> {
        self.calls
            .drain(..)
            .filter(|(name, _)| !name.is_empty())
            .map(|(name, args)| {
                let parsed = serde_json::from_str::<Value>(&args)
                    .unwrap_or_else(|_| json!({"_raw": args}));
                json!({"functionCall": {"name": name, "args": parsed}})
            })
            .collect()
    }
}

/// One OpenAI SSE `delta` chunk -> one Gemini streaming response, or None when
/// the chunk carries nothing renderable yet (role-only openers, keepalives, or
/// a tool-call fragment still being accumulated).
pub fn chunk_to_gemini_acc(chunk: &Value, acc: &mut ToolCallAccumulator) -> Option<Value> {
    let choice = match chunk.get("choices").and_then(Value::as_array).and_then(|c| c.first()) {
        Some(c) => c,
        // With include_usage, vLLM ends the stream with a choices-less chunk
        // carrying only the token counts. Dropping it is how a finished turn
        // reports zero tokens used.
        None => return usage_metadata(chunk).map(|u| json!({"usageMetadata": u})),
    };
    let delta = choice.get("delta")?;
    let finish = choice.get("finish_reason").and_then(Value::as_str);

    acc.absorb(delta);

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

    // Only once the model is done can the accumulated calls be emitted whole.
    if finish.is_some() && !acc.is_empty() {
        parts.extend(acc.drain_parts());
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
    let mut out = Map::new();
    out.insert("candidates".into(), json!([Value::Object(candidate)]));
    if let Some(u) = usage_metadata(chunk) {
        out.insert("usageMetadata".into(), u);
    }
    Some(Value::Object(out))
}

/// OpenAI `usage` -> Gemini `usageMetadata`, when the payload carries one.
fn usage_metadata(payload: &Value) -> Option<Value> {
    let usage = payload.get("usage")?;
    if usage.is_null() {
        return None;
    }
    Some(json!({
        "promptTokenCount": usage.get("prompt_tokens").cloned().unwrap_or(json!(0)),
        "candidatesTokenCount": usage.get("completion_tokens").cloned().unwrap_or(json!(0)),
        "totalTokenCount": usage.get("total_tokens").cloned().unwrap_or(json!(0)),
    }))
}

/// Stateless variant, kept for the non-accumulating tests.
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
    let mut out = Map::new();
    out.insert("candidates".into(), json!([Value::Object(candidate)]));
    if let Some(u) = usage_metadata(chunk) {
        out.insert("usageMetadata".into(), u);
    }
    Some(Value::Object(out))
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
    fn json_mime_type_requests_json_mode() {
        let req = json!({"contents": [], "generationConfig": {"responseMimeType": "application/json"}});
        let out = request_to_openai("m", &req, false);
        assert_eq!(out["response_format"]["type"], "json_object");
    }

    #[test]
    fn a_response_schema_becomes_guided_decoding() {
        let schema = json!({"type": "object", "properties": {"next": {"type": "string"}}});
        let req = json!({"contents": [], "generationConfig": {
            "responseMimeType": "application/json", "responseJsonSchema": schema}});
        let out = request_to_openai("m", &req, false);
        assert_eq!(out["response_format"]["type"], "json_schema");
        assert_eq!(out["response_format"]["json_schema"]["schema"], schema);
    }

    #[test]
    fn tool_config_mode_maps_to_tool_choice() {
        for (mode, expected) in [("NONE", "none"), ("ANY", "required"), ("AUTO", "auto")] {
            let req = json!({"contents": [],
                "toolConfig": {"functionCallingConfig": {"mode": mode}}});
            let out = request_to_openai("m", &req, false);
            assert_eq!(out["tool_choice"], expected, "mode {mode}");
        }
    }

    #[test]
    fn streaming_asks_for_token_counts() {
        let req = json!({"contents": []});
        assert!(request_to_openai("m", &req, false).get("stream_options").is_none());
        let s = request_to_openai("m", &req, true);
        assert_eq!(s["stream_options"]["include_usage"], true);
    }

    #[test]
    fn usage_only_final_chunk_is_forwarded_not_dropped() {
        let mut acc = ToolCallAccumulator::new();
        let chunk = json!({"choices": [],
            "usage": {"prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18}});
        let out = chunk_to_gemini_acc(&chunk, &mut acc).expect("usage chunk must survive");
        assert_eq!(out["usageMetadata"]["totalTokenCount"], 18);
        assert_eq!(out["usageMetadata"]["candidatesTokenCount"], 11);
    }

    #[test]
    fn inline_image_becomes_a_data_url_part() {
        let req = json!({"contents": [{"role": "user", "parts": [
            {"text": "what is this?"},
            {"inlineData": {"mimeType": "image/png", "data": "iVBORw0KGgo="}}]}]});
        let out = request_to_openai("m", &req, false);
        let content = &out["messages"][0]["content"];
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "image_url");
        assert_eq!(content[1]["image_url"]["url"], "data:image/png;base64,iVBORw0KGgo=");
    }

    #[test]
    fn audio_routes_to_audio_url_not_image_url() {
        let req = json!({"contents": [{"role": "user", "parts": [
            {"inlineData": {"mimeType": "audio/wav", "data": "UklGRg=="}}]}]});
        let out = request_to_openai("m", &req, false);
        assert_eq!(out["messages"][0]["content"][0]["type"], "audio_url");
    }

    #[test]
    fn file_uri_is_passed_through_as_a_url() {
        let req = json!({"contents": [{"role": "user", "parts": [
            {"fileData": {"mimeType": "image/jpeg", "fileUri": "https://x/y.jpg"}}]}]});
        let out = request_to_openai("m", &req, false);
        assert_eq!(out["messages"][0]["content"][0]["image_url"]["url"], "https://x/y.jpg");
    }

    #[test]
    fn text_only_turn_keeps_the_plain_string_form() {
        // vLLM accepts both, but the string form is what every other client
        // sends; changing it for every request would be a needless behaviour
        // change with its own failure modes.
        let req = json!({"contents": [{"role": "user", "parts": [{"text": "hi"}]}]});
        let out = request_to_openai("m", &req, false);
        assert_eq!(out["messages"][0]["content"], "hi");
    }

    #[test]
    fn streamed_tool_call_is_assembled_from_fragments() {
        // Exactly how vLLM streams one: name first, then arguments in pieces.
        let mut acc = ToolCallAccumulator::new();
        let opener = json!({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c1", "function": {"name": "read_file", "arguments": ""}}]}}]});
        assert!(chunk_to_gemini_acc(&opener, &mut acc).is_none(),
                "a name-only fragment must not be emitted on its own");

        for frag in ["{\"path\"", ": \"secret", ".txt\"}"] {
            let c = json!({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": frag}}]}}]});
            assert!(chunk_to_gemini_acc(&c, &mut acc).is_none());
        }

        let done = json!({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]});
        let out = chunk_to_gemini_acc(&done, &mut acc).expect("finish emits the call");
        let call = &out["candidates"][0]["content"]["parts"][0]["functionCall"];
        assert_eq!(call["name"], "read_file");
        assert_eq!(call["args"]["path"], "secret.txt");
    }

    #[test]
    fn two_parallel_streamed_tool_calls_stay_separate() {
        let mut acc = ToolCallAccumulator::new();
        let c = json!({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"name": "a", "arguments": "{}"}},
            {"index": 1, "function": {"name": "b", "arguments": "{}"}}]}}]});
        chunk_to_gemini_acc(&c, &mut acc);
        let done = json!({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]});
        let out = chunk_to_gemini_acc(&done, &mut acc).unwrap();
        let parts = out["candidates"][0]["content"]["parts"].as_array().unwrap().clone();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["functionCall"]["name"], "a");
        assert_eq!(parts[1]["functionCall"]["name"], "b");
    }

    #[test]
    fn streamed_text_still_flows_immediately() {
        let mut acc = ToolCallAccumulator::new();
        let c = json!({"choices": [{"delta": {"content": "hi"}}]});
        let out = chunk_to_gemini_acc(&c, &mut acc).expect("text is not buffered");
        assert_eq!(out["candidates"][0]["content"]["parts"][0]["text"], "hi");
    }

    #[test]
    fn final_chunk_with_finish_reason_is_kept() {
        let chunk = json!({"choices": [{"delta": {}, "finish_reason": "stop"}]});
        let out = chunk_to_gemini(&chunk).expect("finish must reach the client");
        assert_eq!(out["candidates"][0]["finishReason"], "STOP");
    }
}
