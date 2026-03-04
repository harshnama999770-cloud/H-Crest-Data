import json
import requests

payload = {
    "model": "deepseek-coder-7b-instruct-v1.5",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in one line."}
    ],
    "temperature": 0.2,
    "max_tokens": 300,
    "stream": False
}

r = requests.post(
    "http://127.0.0.1:1234/chat/completions",
    headers={"Content-Type": "application/json"},
    data=json.dumps(payload),
    timeout=60
)

data = r.json()
reply = data["choices"][0]["message"]["content"]
print("LLM reply:", reply.strip())
