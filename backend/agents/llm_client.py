import os
import time
import requests
from dotenv import load_dotenv

# load environment variables from .env file at project root
load_dotenv()

# Groq free tier endpoint, OpenAI-compatible API
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_ID = "llama-3.1-8b-instant"

# retry config for 429 rate-limit and 503 responses
MAX_RETRIES = 5
BACKOFF_BASE = 2  # seconds, doubles each retry


class LLMError(Exception):
    # raised when all retries fail or API returns an unexpected error
    pass


def _get_headers() -> dict:
    # reads Groq API key from environment to avoid hardcoding credentials
    token = os.environ.get("GROQ_API_KEY", "")
    if not token:
        raise LLMError("GROQ_API_KEY environment variable is not set")
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def generate(prompt: str, max_new_tokens: int = 512) -> str:
    # sends prompt to Groq API using OpenAI-compatible chat format
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": 0.3,
    }

    headers = _get_headers()

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)

            if response.status_code in (429, 503):
                # rate limited or service unavailable, backoff and retry
                wait = BACKOFF_BASE ** attempt
                time.sleep(wait)
                continue

            if response.status_code != 200:
                raise LLMError(f"Groq API returned {response.status_code}: {response.text}")

            data = response.json()
            return data["choices"][0]["message"]["content"].strip()

        except requests.RequestException as e:
            if attempt == MAX_RETRIES - 1:
                raise LLMError(f"Request failed after {MAX_RETRIES} attempts: {e}") from e
            time.sleep(BACKOFF_BASE ** attempt)

    raise LLMError(f"Model unavailable after {MAX_RETRIES} retries")