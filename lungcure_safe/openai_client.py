"""OpenAI-compatible API client wrapper for SafeBench."""
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request

_client = None
_client_cache = {}
_key_pool = None
_key_pool_signature = None
_client_lock = threading.Lock()


def get_client(api_key=None, base_url=None):
    global _client
    if _client is not None:
        return _client
    try:
        import openai
    except ImportError as exc:
        raise RuntimeError("Install the 'openai' package to use API-backed inference.") from exc
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    base_url = base_url or os.environ.get("OPENAI_BASE_URL")
    init_kw = {"api_key": api_key}
    if base_url:
        init_kw["base_url"] = base_url
    init_kw["timeout"] = float(os.environ.get("OPENAI_TIMEOUT_SECONDS", "60"))
    init_kw["max_retries"] = 0
    _client = openai.OpenAI(**init_kw)
    return _client


def _new_client(api_key, base_url=None):
    try:
        import openai
    except ImportError as exc:
        raise RuntimeError("Install the 'openai' package to use API-backed inference.") from exc
    init_kw = {"api_key": api_key, "timeout": float(os.environ.get("OPENAI_TIMEOUT_SECONDS", "60")), "max_retries": 0}
    if base_url:
        init_kw["base_url"] = base_url
    return openai.OpenAI(**init_kw)


def get_client_for_key(api_key, base_url=None):
    cache_key = (api_key, base_url or "")
    with _client_lock:
        client = _client_cache.get(cache_key)
        if client is None:
            client = _new_client(api_key, base_url)
            _client_cache[cache_key] = client
        return client


class KeyPool:
    def __init__(self, api_keys):
        now = time.time()
        self.items = [
            {
                "key": key,
                "idx": idx,
                "cooldown_until": now,
                "fail_count": 0,
                "success_count": 0,
                "in_use": 0,
                "total_requests": 0,
                "total_429": 0,
            }
            for idx, key in enumerate(api_keys)
            if key
        ]
        if not self.items:
            raise RuntimeError("LUNGCURE_SAFE_KEY_POOL is enabled but no API keys were provided.")
        self.rr_index = 0
        self.lock = threading.Lock()

    def acquire_key(self):
        while True:
            with self.lock:
                now = time.time()
                n = len(self.items)
                for offset in range(n):
                    idx = (self.rr_index + offset) % n
                    item = self.items[idx]
                    if item["cooldown_until"] <= now:
                        item["in_use"] += 1
                        item["total_requests"] += 1
                        self.rr_index = (idx + 1) % n
                        return item
                next_ready = min(item["cooldown_until"] for item in self.items)
            time.sleep(max(0.2, next_ready - time.time()))

    def release_key(self, item, success=False):
        with self.lock:
            item["in_use"] = max(0, item["in_use"] - 1)
            if success:
                item["fail_count"] = 0
                item["success_count"] += 1

    def mark_rate_limited(self, item):
        with self.lock:
            item["total_429"] += 1
            item["fail_count"] += 1
            cooldown = 8 * (2 ** (item["fail_count"] - 1)) + random.uniform(0, 2)
            item["cooldown_until"] = time.time() + min(cooldown, 300)


def _split_api_keys(raw):
    return [part.strip().strip("'\"") for part in re.split(r"[\s,;]+", raw or "") if part.strip().strip("'\"")]


def _pool_keys(api_key=None):
    raw = os.environ.get("LUNGCURE_SAFE_API_KEYS")
    keys = _split_api_keys(raw)
    key_files = []
    if os.environ.get("LUNGCURE_SAFE_API_KEYS_FILE"):
        key_files.append(os.environ["LUNGCURE_SAFE_API_KEYS_FILE"])
    for key_file in key_files:
        try:
            with open(key_file, encoding="utf-8") as handle:
                keys.extend(_split_api_keys(handle.read()))
        except OSError:
            pass
    if api_key:
        keys.insert(0, api_key)
    elif os.environ.get("OPENAI_API_KEY"):
        keys.insert(0, os.environ["OPENAI_API_KEY"])
    deduped = []
    seen = set()
    for key in keys:
        if key and key not in seen:
            deduped.append(key)
            seen.add(key)
    return deduped


def get_key_pool(api_key=None):
    global _key_pool, _key_pool_signature
    keys = _pool_keys(api_key)
    signature = tuple(keys)
    with _client_lock:
        if _key_pool is None or _key_pool_signature != signature:
            _key_pool = KeyPool(keys)
            _key_pool_signature = signature
        return _key_pool


def _is_rate_limit_error(exc):
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    text = str(exc).lower()
    return "429" in text or "too many requests" in text or "rate limit" in text


def chat_completion(messages, model, api_key=None, base_url=None, temperature=0.0, max_tokens=384):
    api_style = os.environ.get("LUNGCURE_SAFE_API_STYLE", "chat").lower()
    if api_style == "messages":
        return messages_completion(messages, model, api_key=api_key, base_url=base_url, temperature=temperature, max_tokens=max_tokens)
    if api_style == "grokapi":
        return grokapi_completion(messages, model, base_url=base_url)
    create_kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if os.environ.get("LUNGCURE_SAFE_OMIT_TEMPERATURE") != "1":
        create_kwargs["temperature"] = temperature
    extra_body_json = os.environ.get("LUNGCURE_SAFE_EXTRA_BODY_JSON")
    if extra_body_json:
        try:
            create_kwargs["extra_body"] = json.loads(extra_body_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError("LUNGCURE_SAFE_EXTRA_BODY_JSON must be valid JSON.") from exc
    if os.environ.get("LUNGCURE_SAFE_KEY_POOL") == "1":
        attempts = max(1, int(os.environ.get("LUNGCURE_SAFE_KEY_POOL_ATTEMPTS", "8")))
        pool = get_key_pool(api_key)
        last_error = None
        for _ in range(attempts):
            item = pool.acquire_key()
            try:
                client = get_client_for_key(item["key"], base_url or os.environ.get("OPENAI_BASE_URL"))
                resp = client.chat.completions.create(**create_kwargs)
                pool.release_key(item, success=True)
                break
            except Exception as exc:
                last_error = exc
                if _is_rate_limit_error(exc):
                    pool.mark_rate_limited(item)
                    pool.release_key(item, success=False)
                    continue
                pool.release_key(item, success=False)
                raise
        else:
            raise last_error
    else:
        client = get_client(api_key, base_url)
        resp = client.chat.completions.create(**create_kwargs)
    message = resp.choices[0].message
    content = (
        getattr(message, "content", None)
        or getattr(message, "reasoning_content", None)
        or getattr(message, "reasoning", None)
    )
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        content = "\n".join(parts)
    if not content or not content.strip():
        raise RuntimeError("API returned an empty chat completion.")
    return content


def _messages_payload(messages, model, temperature=0.0, max_tokens=384):
    system_parts = []
    anthropic_messages = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "system":
            system_parts.append(str(content))
        else:
            anthropic_messages.append({"role": role, "content": str(content)})
    payload = {
        "model": model,
        "messages": anthropic_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if system_parts:
        payload["system"] = "\n\n".join(system_parts)
    return payload


def _extract_messages_text(resp_obj):
    parts = []
    for item in resp_obj.get("content") or []:
        if not isinstance(item, dict):
            parts.append(str(item))
            continue
        if item.get("type") == "text":
            parts.append(str(item.get("text") or ""))
        elif item.get("type") in {"thinking", "reasoning"}:
            # Some providers emit useful text only in a reasoning block when
            # the output is truncated. Keep it as a fallback below.
            parts.append(str(item.get("thinking") or item.get("text") or ""))
    return "\n".join(p for p in parts if p).strip()


def messages_completion(messages, model, api_key=None, base_url=None, temperature=0.0, max_tokens=384):
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "").rstrip("/")
    if not base_url:
        raise RuntimeError("OPENAI_BASE_URL is required for messages API style.")
    url = f"{base_url}/messages"
    payload = _messages_payload(messages, model, temperature=temperature, max_tokens=max_tokens)
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "anthropic-version": os.environ.get("ANTHROPIC_VERSION", "2023-06-01"),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=float(os.environ.get("OPENAI_TIMEOUT_SECONDS", "60"))) as response:
            resp_obj = json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"messages API HTTP {exc.code}: {body[:500]}") from exc
    content = _extract_messages_text(resp_obj)
    if not content:
        raise RuntimeError("API returned an empty messages completion.")
    return content


def grokapi_completion(messages, model, base_url=None):
    base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "").rstrip("/")
    if not base_url:
        raise RuntimeError("OPENAI_BASE_URL is required for grokapi API style.")
    url = f"{base_url}/ask"
    prompt_parts = []
    for message in messages:
        role = str(message.get("role") or "user").upper()
        content = str(message.get("content") or "")
        if content:
            prompt_parts.append(f"{role}:\n{content}")
    payload = {
        "message": "\n\n".join(prompt_parts),
        "model": model,
        "extra_data": None,
    }
    proxy = os.environ.get("GROKAPI_PROXY")
    if proxy:
        payload["proxy"] = proxy
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=float(os.environ.get("OPENAI_TIMEOUT_SECONDS", "180"))) as response:
            resp_obj = json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"grokapi HTTP {exc.code}: {body[:500]}") from exc
    content = resp_obj.get("response") or resp_obj.get("message") or resp_obj.get("content")
    if not content and isinstance(resp_obj.get("stream_response"), list):
        content = "".join(str(part) for part in resp_obj["stream_response"])
    if not content or not str(content).strip():
        raise RuntimeError(f"grokapi returned an empty response: {str(resp_obj)[:500]}")
    return str(content)


def reset_client():
    global _client, _client_cache, _key_pool, _key_pool_signature
    _client = None
    _client_cache = {}
    _key_pool = None
    _key_pool_signature = None
