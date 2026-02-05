import json
import os
import urllib.request
import urllib.error


def _load_dotenv(dotenv_path: str) -> None:
    if not os.path.exists(dotenv_path):
        return
    try:
        with open(dotenv_path, "r") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        return


class LLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_sec: int | None = None,
    ) -> None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        _load_dotenv(os.path.join(project_root, ".env"))

        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
        self.base_url = (
            base_url
            or os.getenv("GEMINI_BASE_URL")
            or os.getenv("LLM_BASE_URL")
            or "https://generativelanguage.googleapis.com/v1beta"
        ).rstrip("/")
        self.model = model or os.getenv("GEMINI_MODEL") or os.getenv("LLM_MODEL") or "gemini-2.5-flash"
        self.timeout_sec = int(timeout_sec or os.getenv("LLM_TIMEOUT_SEC", "30"))

    def generate(
        self,
        user_text: str,
        system_text: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 256,
    ) -> str | None:
        if not self.api_key:
            print("[LLM] Missing GEMINI_API_KEY; skipping LLM call.")
            return None

        payload = {
            "contents": [{"role": "user", "parts": [{"text": user_text}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_text:
            payload["systemInstruction"] = {"parts": [{"text": system_text}]}

        url = f"{self.base_url}/models/{self.model}:generateContent"
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                body = response.read().decode("utf-8")
                data = json.loads(body)
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, IndexError, json.JSONDecodeError) as exc:
            print(f"[LLM] Request failed: {exc}")
            return None
