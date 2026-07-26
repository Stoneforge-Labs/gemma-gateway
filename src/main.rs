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

mod translate;

use axum::{
    body::Body,
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
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
    http: reqwest::Client,
}

impl Gateway {
    /// The model the CLI asked for, unless we were told to pin one. A pin is
    /// useful because the CLI's model names ("gemini-2.5-pro") mean nothing to
    /// vLLM, which serves exactly one.
    fn resolve_model(&self, requested: &str) -> String {
        self.model.clone().unwrap_or_else(|| requested.to_string())
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
    let model = gw.resolve_model(&model);

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
    let payload = translate::request_to_openai(&model, &body, false);
    let url = format!("{}/chat/completions", gw.upstream.trim_end_matches('/'));
    match gw.http.post(&url).json(&payload).send().await {
        Ok(resp) => match resp.json::<Value>().await {
            Ok(v) => {
                if v.get("error").is_some() {
                    tracing::warn!(error = %v, "upstream returned an error");
                    return (StatusCode::BAD_GATEWAY, Json(v)).into_response();
                }
                Json(translate::response_to_gemini(&v)).into_response()
            }
            Err(e) => upstream_error("decoding upstream response", e),
        },
        Err(e) => upstream_error("calling upstream", e),
    }
}

async fn stream_generate(gw: Arc<Gateway>, model: String, body: Value, _sse: bool) -> Response {
    let payload = translate::request_to_openai(&model, &body, true);
    let url = format!("{}/chat/completions", gw.upstream.trim_end_matches('/'));

    let resp = match gw.http.post(&url).json(&payload).send().await {
        Ok(r) => r,
        Err(e) => return upstream_error("opening upstream stream", e),
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
    let models: Vec<Value> = ids
        .iter()
        .map(|id| {
            json!({
                "name": format!("models/{id}"),
                "displayName": id,
                "supportedGenerationMethods": ["generateContent", "streamGenerateContent", "countTokens"],
            })
        })
        .collect();
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
        .route("/health", get(|| async { "ok" }))
        .with_state(gw);

    tracing::info!("listening on {listen}, upstream {upstream}, model {model:?}");
    let listener = tokio::net::TcpListener::bind(&listen)
        .await
        .unwrap_or_else(|e| panic!("cannot bind {listen}: {e}"));
    axum::serve(listener, app).await.expect("server");
}
