//! gemma-gateway — speak Gemini REST at the front, OpenAI at the back.
//!
//! Gemini CLI honours `GOOGLE_GEMINI_BASE_URL` (it has its own `AuthType::GATEWAY`),
//! so pointing it at this process is enough to drive a local vLLM-served Gemma
//! with *no changes to the CLI at all*. That matters more than it sounds: the
//! fork stays byte-identical to upstream, so `git pull` keeps working forever
//! and every feature they ship — hooks, MCP, subagents, checkpointing, plan
//! mode — arrives for free.
//!
//!     gemma-gateway --listen 127.0.0.1:8899 --upstream http://127.0.0.1:8891/v1
//!     GOOGLE_GEMINI_BASE_URL=http://127.0.0.1:8899 gemini
//!
//! Only the routes the CLI actually calls are implemented; anything else
//! returns 404 with a clear message rather than a confusing empty body.
//!
//! # Serving the CLI's own router
//!
//! Gemini CLI can put its routing classifier on a local model instead of
//! spending a full-size call on it every turn. Upstream that means installing
//! Google's LiteRT-LM binary and a Gemma 3 1B, but its client is just the GenAI
//! SDK pointed at a base URL:
//!
//!     new GoogleGenAI({ apiVersion: "v1beta", httpOptions: { baseUrl: host } })
//!         .models.generateContent({ model, contents, config: {
//!             responseMimeType: "application/json", temperature: 0,
//!             maxOutputTokens: 256 } })
//!
//! which is this gateway's wire format exactly, so no LiteRT install is needed:
//!
//!     "experimental": { "gemmaModelRouter": {
//!         "enabled": true,
//!         "autoStartServer": false,
//!         "classifier": { "host": "http://127.0.0.1:8899",
//!                         "model": "gemma3-1b-gpu-custom" } } }
//!
//! Three things there are load-bearing. `autoStartServer` defaults to true and
//! will try to launch a LiteRT binary that is not present. The model name is
//! checked against a hard-coded literal ("Only gemma3-1b-gpu-custom has been
//! tested"), so it has to be spelled that way — harmless, because `--model`
//! pins what is actually served and the requested name is discarded. And the
//! client JSON.parses the reply under a 10s timeout, so whatever sits behind
//! the gateway has to be quick and has to honour response_format.

mod translate;

use axum::{
    body::Body,
    extract::{Path, Query, State},
    http::{header::CONTENT_TYPE, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone)]
struct Gateway {
    upstream: String,
    model: Option<String>,
    /// Whatever vLLM turned out to be serving, remembered after the first ask.
    discovered: Arc<tokio::sync::RwLock<Option<String>>>,
    http: reqwest::Client,
}

impl Gateway {
    /// Which model to actually ask for.
    ///
    /// The CLI sends names like "gemini-2.5-pro" that mean nothing to vLLM, so
    /// something has to substitute a real one. An explicit `--model` wins, but
    /// pinning it is a trap in daily use: `gemma --boot` serves a name that
    /// follows the profile, so switching from quality to fast renames the model
    /// under a gateway that is still pinned to the old one, and every request
    /// 404s until someone restarts it.
    ///
    /// So with no pin, ask vLLM what it is serving and remember the answer.
    async fn resolve_model(&self, requested: &str) -> String {
        if let Some(pinned) = &self.model {
            return pinned.clone();
        }
        if let Some(known) = self.discovered.read().await.clone() {
            return known;
        }
        match self.first_served().await {
            Some(name) => {
                tracing::info!("upstream is serving {name}");
                *self.discovered.write().await = Some(name.clone());
                name
            }
            // Nothing to discover — pass the request through and let the
            // upstream error say so, rather than inventing a name.
            None => requested.to_string(),
        }
    }

    async fn first_served(&self) -> Option<String> {
        let url = format!("{}/models", self.upstream.trim_end_matches('/'));
        let body = self
            .http
            .get(&url)
            .send()
            .await
            .ok()?
            .json::<Value>()
            .await
            .ok()?;
        body.get("data")?
            .as_array()?
            .first()?
            .get("id")?
            .as_str()
            .map(str::to_string)
    }

    /// Forget the discovered name so the next request looks it up again. Called
    /// when upstream rejects a request: the usual cause is that the server was
    /// restarted onto a different profile.
    async fn forget_model(&self) {
        if self.model.is_none() {
            *self.discovered.write().await = None;
        }
    }

    /// POST a translated request upstream, retrying once against a freshly
    /// discovered model name if the first attempt is rejected.
    ///
    /// A rejection here almost always means one thing: vLLM was restarted onto
    /// a different profile, so the name cached at first contact no longer
    /// exists. `gem --12b` does exactly that, and without this the gateway kept
    /// asking for the model that used to be loaded until someone restarted it
    /// by hand. Rediscovering costs one extra request, once.
    ///
    /// Both callers go through here because the streaming path used to skip the
    /// status check entirely: vLLM answers an unknown model with 404 and a JSON
    /// body, which contains no `data:` lines, so the SSE loop forwarded nothing
    /// and returned a perfectly valid EMPTY 200 stream. The CLI reported "the
    /// model returned an empty response", which points at the model and not at
    /// the dead name in the request.
    async fn post_chat(
        &self,
        model: &str,
        body: &Value,
        stream: bool,
    ) -> Result<reqwest::Response, (StatusCode, String)> {
        let url = format!("{}/chat/completions", self.upstream.trim_end_matches('/'));
        let mut name = model.to_string();
        for attempt in 0..2 {
            let payload = translate::request_to_openai(&name, body, stream);
            tracing::debug!(openai = %payload, "outbound");
            let resp = self
                .http
                .post(&url)
                .json(&payload)
                .send()
                .await
                .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))?;
            if resp.status().is_success() {
                return Ok(resp);
            }
            let status = resp.status();
            let detail = resp.text().await.unwrap_or_default();
            tracing::warn!(%status, %detail, "upstream rejected the request");
            self.forget_model().await;
            let fresh = self.resolve_model(&name).await;
            // Same name back means the rejection was about something other than
            // a stale name, so retrying would just ask the same question twice.
            if attempt == 1 || fresh == name {
                return Err((StatusCode::BAD_GATEWAY, detail));
            }
            tracing::info!("retrying as {fresh}");
            name = fresh;
        }
        unreachable!("the loop returns on both attempts")
    }
}

/// `POST /v1beta/models/{model}:generateContent`
/// `POST /v1beta/models/{model}:streamGenerateContent`
/// `POST /v1beta/models/{model}:countTokens`
///
/// The SDK encodes the method after a colon, so one handler covers all three.
async fn model_action(
    State(gw): State<Arc<Gateway>>,
    Path(spec): Path<String>,
    Query(params): Query<HashMap<String, String>>,
    _headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let (model, action) = match spec.split_once(':') {
        Some((m, a)) => (m.to_string(), a.to_string()),
        None => (spec.clone(), "generateContent".to_string()),
    };
    let model = gw.resolve_model(&model).await;

    match action.as_str() {
        "countTokens" => count_tokens(gw, model, body).await,
        "streamGenerateContent" => {
            // The SDK asks for SSE explicitly; anything else means it wants the
            // legacy JSON-array form, which we do not implement.
            let sse = params.get("alt").map(|a| a == "sse").unwrap_or(false);
            stream_generate(gw, model, body, sse).await
        }
        "generateContent" => generate(gw, model, body).await,
        other => (
            StatusCode::NOT_FOUND,
            Json(json!({"error": {"message": format!("unsupported action: {other}")}})),
        )
            .into_response(),
    }
}

async fn generate(gw: Arc<Gateway>, model: String, body: Value) -> Response {
    tracing::debug!(gemini = %body, "inbound");
    let resp = match gw.post_chat(&model, &body, false).await {
        Ok(r) => r,
        Err((status, detail)) => {
            return (status, Json(json!({"error": {"message": detail}}))).into_response()
        }
    };
    match resp.json::<Value>().await {
        Ok(v) => {
            // A 200 carrying an error body: rarer than the 404, but it happens
            // for a malformed grammar, and it is still worth not translating.
            if v.get("error").is_some() {
                tracing::warn!(error = %v, "upstream returned an error");
                gw.forget_model().await;
                return (StatusCode::BAD_GATEWAY, Json(v)).into_response();
            }
            Json(translate::response_to_gemini(&v)).into_response()
        }
        Err(e) => upstream_error("decoding upstream response", e),
    }
}

/// Keep OpenAI-compatible clients on the same ingress as Gemini clients.
///
/// This also terminates client-side h2c upgrades before forwarding with
/// reqwest, which avoids Uvicorn treating an upgraded request body as empty.
async fn openai_chat(State(gw): State<Arc<Gateway>>, Json(mut body): Json<Value>) -> Response {
    let requested = body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let model = gw.resolve_model(requested).await;
    if let Some(object) = body.as_object_mut() {
        object.insert("model".into(), json!(model));
    }
    let url = format!("{}/chat/completions", gw.upstream.trim_end_matches('/'));
    match gw.http.post(&url).json(&body).send().await {
        Ok(resp) => {
            let status = resp.status();
            let content_type = resp.headers().get(CONTENT_TYPE).cloned();
            let mut response = Response::builder().status(status);
            if let Some(content_type) = content_type {
                response = response.header(CONTENT_TYPE, content_type);
            }
            response
                .body(Body::from_stream(resp.bytes_stream()))
                .expect("valid upstream response")
        }
        Err(e) => upstream_error("calling OpenAI-compatible upstream", e),
    }
}

async fn stream_generate(gw: Arc<Gateway>, model: String, body: Value, _sse: bool) -> Response {
    tracing::debug!(gemini = %body, "inbound (stream)");
    let resp = match gw.post_chat(&model, &body, true).await {
        Ok(r) => r,
        // Report the rejection instead of opening an empty stream over it.
        Err((status, detail)) => {
            return (status, Json(json!({"error": {"message": detail}}))).into_response()
        }
    };

    // Re-frame OpenAI SSE as Gemini SSE. Both use `data: <json>` lines, so the
    // work is per-chunk translation, not reframing the protocol. Chunks can
    // split mid-line, so hold a buffer across reads.
    let stream = async_stream::stream! {
        let mut bytes = resp.bytes_stream();
        let mut buf = String::new();
        // Tool calls arrive in fragments across chunks; hold them until the
        // model reports finish, then emit each as one whole functionCall.
        let mut acc = translate::ToolCallAccumulator::new();
        while let Some(next) = bytes.next().await {
            let chunk = match next {
                Ok(c) => c,
                Err(e) => { tracing::warn!(error = %e, "stream broke"); break; }
            };
            buf.push_str(&String::from_utf8_lossy(&chunk));
            while let Some(idx) = buf.find('\n') {
                let line = buf[..idx].trim().to_string();
                buf.drain(..=idx);
                let Some(data) = line.strip_prefix("data:") else { continue };
                let data = data.trim();
                if data.is_empty() { continue }
                if data == "[DONE]" {
                    // OpenAI terminates a stream with a literal [DONE] sentinel;
                    // Google's SSE does not — it just ends. Forwarding it makes
                    // the GenAI SDK try to JSON.parse the word "[DONE]" and
                    // crash the turn after the answer already arrived.
                    return;
                }
                let Ok(parsed) = serde_json::from_str::<Value>(data) else {
                    tracing::debug!(raw = %data, "unparseable chunk, skipped");
                    continue
                };
                if let Some(out) = translate::chunk_to_gemini_acc(&parsed, &mut acc) {
                    let framed = format!("data: {out}\n\n");
                    yield Ok::<_, std::io::Error>(axum::body::Bytes::from(framed));
                }
            }
        }
    };

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(Body::from_stream(stream))
        .unwrap()
}

/// vLLM exposes `/tokenize`, which is a real count rather than an estimate.
async fn count_tokens(gw: Arc<Gateway>, model: String, body: Value) -> Response {
    let payload = translate::request_to_openai(&model, &body, false);
    let base = gw.upstream.trim_end_matches("/v1").trim_end_matches('/');
    let url = format!("{base}/tokenize");

    let probe = json!({
        "model": model,
        "messages": payload.get("messages").cloned().unwrap_or(json!([])),
    });
    if let Ok(resp) = gw.http.post(&url).json(&probe).send().await {
        if let Ok(v) = resp.json::<Value>().await {
            if let Some(count) = v.get("count").and_then(Value::as_u64) {
                return Json(json!({"totalTokens": count})).into_response();
            }
        }
    }
    // Falling back to an estimate is better than failing the turn: the CLI uses
    // this for budgeting, not for correctness.
    let chars: usize = payload
        .get("messages")
        .and_then(Value::as_array)
        .map(|m| m.iter().map(|x| x.to_string().len()).sum())
        .unwrap_or(0);
    tracing::debug!("tokenize unavailable, estimating from {chars} chars");
    Json(json!({"totalTokens": chars / 4})).into_response()
}

/// `GET /v1beta/models` — the CLI lists models to validate its configuration.
async fn list_models(State(gw): State<Arc<Gateway>>) -> Response {
    let url = format!("{}/models", gw.upstream.trim_end_matches('/'));
    let ids: Vec<String> = match gw.http.get(&url).send().await {
        Ok(r) => r
            .json::<Value>()
            .await
            .ok()
            .and_then(|v| {
                v.get("data").and_then(Value::as_array).map(|d| {
                    d.iter()
                        .filter_map(|m| m.get("id").and_then(Value::as_str).map(str::to_string))
                        .collect()
                })
            })
            .unwrap_or_default(),
        Err(e) => {
            tracing::warn!(error = %e, "upstream model list unavailable");
            Vec::new()
        }
    };
    // Advertise the stock Gemini names alongside whatever is really loaded.
    //
    // The CLI's "auto" model mode does not send your turn straight to the
    // model. It first asks a small router model -- gemini-3.1-flash-lite --
    // which one should handle it. That name is not what vLLM is serving, so it
    // was missing from this list, and the CLI gave up before it ever sent the
    // request: four router calls billed at zero tokens, nothing in this
    // gateway's log at all, and a session that just sat there. The turn had not
    // failed, it had never started.
    //
    // Nothing is being faked that is not already true: resolve_model() rewrites
    // whatever name a request carries to the model actually loaded, so a call
    // for any of these is answered by the local Gemma. Impersonating the Gemini
    // API is this program's entire job -- listing only one name was the
    // inconsistency, not this.
    const ALIASES: &[&str] = &[
        "gemini-3.1-flash-lite",
        "gemini-3.1-flash",
        "gemini-3.1-pro",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ];
    let entry = |id: &str| {
        json!({
            "name": format!("models/{id}"),
            "displayName": id,
            "supportedGenerationMethods": ["generateContent", "streamGenerateContent", "countTokens"],
        })
    };
    // Real ones first: `gem` reads models[0] to decide what to pass as -m, so a
    // stock name at the front would pin the session to an alias and make every
    // log say "gemini-3.1-pro" for a Gemma.
    let mut models: Vec<Value> = ids.iter().map(|id| entry(id)).collect();
    models.extend(
        ALIASES
            .iter()
            .filter(|a| !ids.iter().any(|id| id == *a))
            .map(|a| entry(a)),
    );
    Json(json!({"models": models})).into_response()
}

fn upstream_error(context: &str, e: reqwest::Error) -> Response {
    tracing::error!(error = %e, "{context}");
    (
        StatusCode::BAD_GATEWAY,
        Json(json!({"error": {"message": format!("{context}: {e}"), "status": "UNAVAILABLE"}})),
    )
        .into_response()
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "gemma_gateway=info".into()),
        )
        .init();

    let mut listen = "127.0.0.1:8899".to_string();
    let mut upstream = "http://127.0.0.1:8891/v1".to_string();
    let mut model: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--listen" => listen = args.next().unwrap_or(listen),
            "--upstream" => upstream = args.next().unwrap_or(upstream),
            "--model" => model = args.next(),
            "--help" | "-h" => {
                println!(
                    "gemma-gateway --listen {listen} --upstream {upstream} [--model NAME]\n\n\
                     Without --model, the gateway asks the upstream what it is\n\
                     serving and follows it across profile switches.\n\n\
                     Then: GOOGLE_GEMINI_BASE_URL=http://{listen} gemini"
                );
                return;
            }
            other => eprintln!("ignoring unknown argument: {other}"),
        }
    }

    let gw = Arc::new(Gateway {
        upstream: upstream.clone(),
        model: model.clone(),
        discovered: Arc::new(tokio::sync::RwLock::new(None)),
        http: reqwest::Client::builder()
            // Long, because a local reasoning model can think for minutes and a
            // premature client timeout looks exactly like a hung server.
            .timeout(std::time::Duration::from_secs(1800))
            .build()
            .expect("http client"),
    });

    let app = Router::new()
        .route("/v1beta/models", get(list_models))
        .route("/v1beta/models/*spec", post(model_action))
        .route("/v1/models", get(list_models))
        .route("/v1/models/*spec", post(model_action))
        .route("/v1/chat/completions", post(openai_chat))
        .route("/health", get(|| async { "ok" }))
        .with_state(gw);

    tracing::info!("listening on {listen}, upstream {upstream}, model {model:?}");
    let listener = tokio::net::TcpListener::bind(&listen)
        .await
        .unwrap_or_else(|e| panic!("cannot bind {listen}: {e}"));
    axum::serve(listener, app).await.expect("server");
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;

    #[tokio::test]
    async fn openai_chat_forwards_json_through_the_gateway() {
        let upstream = Router::new().route(
            "/v1/chat/completions",
            post(|Json(body): Json<Value>| async move {
                Json(json!({"model": body["model"], "ok": true}))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, upstream).await.unwrap() });
        let gw = Arc::new(Gateway {
            upstream: format!("http://{address}/v1"),
            model: Some("served-model".into()),
            discovered: Arc::new(tokio::sync::RwLock::new(None)),
            http: reqwest::Client::new(),
        });

        let response = openai_chat(
            State(gw),
            Json(json!({"model": "requested-model", "messages": []})),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 1024).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&body).unwrap(),
            json!({"model": "served-model", "ok": true})
        );
        server.abort();
    }
}
