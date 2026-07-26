import json, urllib.request
G = "http://127.0.0.1:8899/v1beta/models/gemma-4-12b"

def post(action, body, stream=False):
    url = f"{G}:{action}" + ("?alt=sse" if stream else "")
    r = urllib.request.Request(url, data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(r, timeout=300)

# 1. structured output with a schema -- must come back as parseable JSON
schema = {"type": "object",
          "properties": {"capital": {"type": "string"}, "population_millions": {"type": "number"}},
          "required": ["capital", "population_millions"]}
r = json.load(post("generateContent", {
    "contents": [{"role": "user", "parts": [{"text": "Capital of France and its population."}]}],
    "generationConfig": {"responseMimeType": "application/json", "responseJsonSchema": schema,
                         "maxOutputTokens": 200}}))
text = "".join(p.get("text", "") for p in r["candidates"][0]["content"]["parts"]
               if not p.get("thought"))
print("1. JSON mode raw:", text.strip()[:120])
parsed = json.loads(text)          # throws if the model answered in prose
assert "capital" in parsed, parsed
print("   -> parsed OK, capital =", parsed["capital"])

# 2. tool_choice NONE must suppress the call even though a tool is offered
tools = [{"functionDeclarations": [{"name": "get_weather", "description": "weather",
          "parametersJsonSchema": {"type": "object", "properties": {"city": {"type": "string"}}}}]}]
r = json.load(post("generateContent", {
    "contents": [{"role": "user", "parts": [{"text": "What is the weather in Paris?"}]}],
    "tools": tools, "toolConfig": {"functionCallingConfig": {"mode": "NONE"}},
    "generationConfig": {"maxOutputTokens": 100}}))
parts = r["candidates"][0]["content"]["parts"]
assert not any("functionCall" in p for p in parts), f"NONE still called a tool: {parts}"
print("2. tool_choice NONE -> no tool call, as required")

# 3. streamed usage must arrive
resp = post("streamGenerateContent", {
    "contents": [{"role": "user", "parts": [{"text": "Count to three."}]}],
    "generationConfig": {"maxOutputTokens": 60}}, stream=True)
usage = None
for raw in resp:
    line = raw.decode(errors="replace").strip()
    if line.startswith("data:"):
        chunk = json.loads(line[5:])
        if "usageMetadata" in chunk:
            usage = chunk["usageMetadata"]
assert usage and usage["totalTokenCount"] > 0, f"no streamed usage: {usage}"
print("3. streamed usageMetadata:", usage)
print("\nALL THREE VERIFIED")
