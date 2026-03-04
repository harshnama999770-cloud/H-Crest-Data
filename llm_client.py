# =========================================================
# llm_client.py
# Connects your project to LM Studio (DeepSeekCoder / any local LLM)
# SAFE + SIMPLE
# =========================================================

import requests
import json
import time


class LocalLLMClient:
    """
    Local LLM client for LM Studio (OpenAI-compatible API).

    Works with:
    - DeepSeekCoder 7B
    - Llama
    - Mistral
    - Any model running inside LM Studio

    LM Studio Server example:
      http://127.0.0.1:1234/chat/completions
    """

    def __init__(
        self,
        base_url="http://127.0.0.1:1234",
        model="deepseek-coder-7b-instruct-v1.5",
        timeout=30,
        max_retries=2,
        verbose=True,
    ):
        self.base_url = str(base_url).rstrip("/")
        self.model = str(model)
        self.timeout = int(timeout)
        self.max_retries = int(max_retries)
        self.verbose = bool(verbose)

    # -------------------------------------------------
    # Internal: safe POST
    # -------------------------------------------------
    def _post(self, endpoint: str, payload: dict):
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                r = requests.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(payload),
                    timeout=self.timeout,
                )

                if r.status_code != 200:
                    last_error = f"HTTP {r.status_code}: {r.text[:300]}"
                    if self.verbose:
                        print(f"[LLM] Error attempt {attempt}: {last_error}")
                    time.sleep(0.8)
                    continue

                return r.json()

            except Exception as e:
                last_error = str(e)
                if self.verbose:
                    print(f"[LLM] Connection failed attempt {attempt}: {last_error}")
                time.sleep(0.8)

        raise RuntimeError(f"LM Studio LLM request failed: {last_error}")

    # -------------------------------------------------
    # Main function: Chat Completion
    # -------------------------------------------------
    def chat(
    self,
    messages,
    temperature=0.1,
    max_tokens=512,
    timeout=None,
    **kwargs,  # ✅ future-proof
    ):
        """
        messages format:
        [
          {"role": "system", "content": "..."},
          {"role": "user", "content": "..."}
        ]
        """

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "stream": False,
        }

        data = self._post("/chat/completions", payload)

        try:
            text = data["choices"][0]["message"]["content"]
        except Exception:
            text = ""

        return str(text).strip()

    # -------------------------------------------------
    # Utility: Ask and force JSON output
    # -------------------------------------------------
    def ask_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature=0.0,
        max_tokens=700,
    ):
        """
        Asks the LLM and expects JSON output.
        If model returns extra text, we try to extract JSON safely.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        raw = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # -----------------------------
        # SAFE JSON extraction
        # -----------------------------
        raw = raw.strip()

        # Try direct JSON
        try:
            return json.loads(raw)
        except Exception:
            pass

        # Try extract first {...} block
        start = raw.find("{")
        end = raw.rfind("}")

        if start != -1 and end != -1 and end > start:
            block = raw[start : end + 1]
            try:
                return json.loads(block)
            except Exception:
                pass

        # If still fails, return raw text
        return {
            "error": "LLM did not return valid JSON",
            "raw": raw[:2000],
        }
