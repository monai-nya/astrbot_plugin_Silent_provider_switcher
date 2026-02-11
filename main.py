from __future__ import annotations

import asyncio
import copy
import re
import time
import types
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, Iterable, List, Optional, Set

from astrbot.api import AstrBotConfig, logger as astr_logger
from astrbot.api.event import filter
from astrbot.api.star import Context, Star, register

try:
    from astrbot.core.provider.entities import LLMResponse, ProviderType
except ImportError:
    from astrbot.api.provider import LLMResponse

    ProviderType = None

_MAX_RECURSION_DEPTH = 10
_SENSITIVE_PATTERN = re.compile(
    r'(sk-|key-|token-|bearer\s+)[a-zA-Z0-9\-_]{4,}',
    re.IGNORECASE,
)

MAX_FALLBACK_ENTRIES = 3
_COOLDOWN_CLEANUP_THRESHOLD = 50
FAILOVER_STATUS_CODES = {
    401,
    402,
    403,
    408,
    409,
    429,
    500,
    502,
    503,
    504,
}
DEFAULT_ERROR_KEYWORDS = [
    "rate limit",
    "too many requests",
    "timeout",
    "timed out",
    "connection reset",
    "invalid api key",
    "authentication error",
]
PROVIDER_CONTAINER_ATTRS = (
    "providers",
    "_providers",
    "provider_map",
    "provider_dict",
)
CONTEXT_PROVIDER_ATTRS = (
    "providers",
    "_providers",
    "provider_manager",
    "provider_pool",
    "provider_registry",
    "llm_providers",
)


@dataclass
class _FallbackEntry:
    provider_id: str
    base_url: str = ""
    api_key: str = ""
    model: str = ""


@register(
    "silent_provider_switcher",
    "洛曦",
    "在 LLM 出错时静默切换到备用提供商。",
    "1.0.0",
)
class SilentProviderSwitcher(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._wrapped: weakref.WeakSet = weakref.WeakSet()
        self._wrapped_ids: Set[int] = set()
        self._cooldowns: Dict[str, float] = {}
        self._wrap_lock = asyncio.Lock()
        self._cooldown_lock = asyncio.Lock()
        self._provider_locks: Dict[str, asyncio.Lock] = {}

    async def initialize(self):
        await self._install_provider_failover()

    @filter.on_astrbot_loaded()
    async def on_bot_loaded(self):
        await self._install_provider_failover()

    def _is_enabled(self) -> bool:
        return bool(self.config.get("enabled", True))

    def _should_log_switch(self) -> bool:
        return bool(self.config.get("log_switch", True))

    def _should_log_errors(self) -> bool:
        return bool(self.config.get("log_errors", True))

    def _get_cooldown_seconds(self) -> int:
        raw = self.config.get("cooldown_seconds", 0)
        try:
            value = int(raw)
        except Exception:
            value = 0
        return max(value, 0)

    def _now(self) -> float:
        return time.monotonic()

    async def _is_in_cooldown(self, provider_id: str) -> bool:
        if not provider_id or provider_id == "unknown":
            return False
        cooldown = self._get_cooldown_seconds()
        if cooldown <= 0:
            return False
        async with self._cooldown_lock:
            ts = self._cooldowns.get(provider_id)
            if ts is None:
                return False
            if self._now() - ts < cooldown:
                return True
            self._cooldowns.pop(provider_id, None)
            return False

    async def _mark_cooldown(self, provider_id: str) -> None:
        if not provider_id or provider_id == "unknown":
            return
        cooldown = self._get_cooldown_seconds()
        if cooldown <= 0:
            return
        async with self._cooldown_lock:
            self._cooldowns[provider_id] = self._now()
            if len(self._cooldowns) > _COOLDOWN_CLEANUP_THRESHOLD:
                now = self._now()
                expired = [
                    k for k, v in self._cooldowns.items()
                    if now - v >= cooldown
                ]
                for k in expired:
                    self._cooldowns.pop(k, None)

    def _get_error_keywords(self) -> Iterable[str]:
        raw = str(self.config.get("error_keywords", "")).strip()
        if not raw:
            return []
        items = []
        for line in raw.replace(",", "\n").splitlines():
            line = line.strip()
            if line:
                items.append(line)
        return items

    def _get_fallback_entries(self) -> List[_FallbackEntry]:
        entries: List[_FallbackEntry] = []
        for idx in range(1, MAX_FALLBACK_ENTRIES + 1):
            provider_id = str(
                self.config.get(f"fallback_provider_id_{idx}", "")
            ).strip()
            if not provider_id:
                continue
            entries.append(
                _FallbackEntry(
                    provider_id=provider_id,
                    base_url=str(
                        self.config.get(f"fallback_base_url_{idx}", "")
                    ).strip(),
                    api_key=str(self.config.get(f"fallback_api_key_{idx}", "")).strip(),
                    model=str(self.config.get(f"fallback_model_{idx}", "")).strip(),
                )
            )
        if not entries:
            fallback_id = str(self.config.get("fallback_provider_id", "")).strip()
            if fallback_id:
                entries.append(
                    _FallbackEntry(
                        provider_id=fallback_id,
                        model=str(self.config.get("fallback_model", "")).strip(),
                    )
                )
        return entries

    def _get_provider_id(self, provider: Any) -> str:
        if not provider:
            return "unknown"
        if hasattr(provider, "provider_config"):
            try:
                value = provider.provider_config.get("id")
                if value:
                    return str(value)
            except Exception as exc:
                astr_logger.debug("读取 provider_config.id 失败: %s", exc)
        for attr in ("provider_id", "id", "name"):
            if hasattr(provider, attr):
                try:
                    value = getattr(provider, attr)
                    if value:
                        return str(value)
                except Exception as exc:
                    astr_logger.debug("读取 provider 属性 %s 失败: %s", attr, exc)
        return "unknown"

    def _get_all_providers(self) -> List[Any]:
        providers: List[Any] = []
        if hasattr(self.context, "get_all_providers"):
            try:
                providers = list(self.context.get_all_providers())
            except Exception as exc:
                astr_logger.exception("get_all_providers 调用失败: %s", exc)
        if not providers:
            astr_logger.warning("未能通过 get_all_providers 获取提供商，尝试属性探测。")
            for attr in CONTEXT_PROVIDER_ATTRS:
                if hasattr(self.context, attr):
                    try:
                        obj = getattr(self.context, attr)
                    except Exception as exc:
                        astr_logger.debug("读取 context.%s 失败: %s", attr, exc)
                        continue
                    providers = self._flatten_providers(obj)
                    if providers:
                        break
        if not providers:
            astr_logger.warning("未获取到任何提供商。")
        return providers

    def _filter_chat_providers(self, providers: List[Any]) -> List[Any]:
        if ProviderType is None:
            return providers
        filtered: List[Any] = []
        for provider in providers:
            try:
                if provider.meta().provider_type == ProviderType.CHAT_COMPLETION:
                    filtered.append(provider)
            except Exception as exc:
                astr_logger.debug("provider 类型判断失败: %s", exc)
                filtered.append(provider)
        return filtered

    def _get_all_chat_providers(self) -> List[Any]:
        return self._filter_chat_providers(self._get_all_providers())

    def _flatten_providers(
        self, obj: Any, _depth: int = 0, _seen: Optional[Set[int]] = None
    ) -> List[Any]:
        if not obj:
            return []
        if _depth > _MAX_RECURSION_DEPTH:
            astr_logger.warning(
                "_flatten_providers 递归深度超限 (%d)，停止递归。", _depth
            )
            return []
        if _seen is None:
            _seen = set()
        obj_id = id(obj)
        if obj_id in _seen:
            return []
        _seen.add(obj_id)
        if isinstance(obj, dict):
            return [v for v in obj.values() if v]
        if isinstance(obj, (list, tuple, set)):
            return [v for v in obj if v]
        for attr in PROVIDER_CONTAINER_ATTRS:
            if hasattr(obj, attr):
                try:
                    return self._flatten_providers(
                        getattr(obj, attr), _depth + 1, _seen
                    )
                except Exception as exc:
                    astr_logger.debug("读取 %s.%s 失败: %s", type(obj), attr, exc)
        for method in ("get_all_providers", "list_providers", "get_providers"):
            if hasattr(obj, method):
                try:
                    return self._flatten_providers(
                        getattr(obj, method)(), _depth + 1, _seen
                    )
                except Exception as exc:
                    astr_logger.debug("调用 %s.%s 失败: %s", type(obj), method, exc)
        return []

    def _should_failover_exception(self, exc: Exception) -> bool:
        code = getattr(exc, "status_code", None) or getattr(exc, "status", None)
        if isinstance(code, int) and code in FAILOVER_STATUS_CODES:
            return True
        message = str(exc).lower()
        keywords = [k.lower() for k in self._get_error_keywords()]
        if not keywords:
            keywords = list(DEFAULT_ERROR_KEYWORDS)
        else:
            keywords = keywords + list(DEFAULT_ERROR_KEYWORDS)
        return any(keyword in message for keyword in keywords)

    def _is_error_response(self, resp: Any) -> bool:
        if not resp:
            return False
        if isinstance(resp, LLMResponse):
            role = getattr(resp, "role", None)
            if role in {"err", "error"}:
                return True
            if getattr(resp, "error", None) is not None:
                return True
            if getattr(resp, "err", None) is not None:
                return True
        if getattr(resp, "error", None) is not None:
            return True
        if getattr(resp, "err", None) is not None:
            return True
        return False

    def _matches_error_keywords(self, resp: Any) -> bool:
        keywords = list(self._get_error_keywords())
        if not keywords or not resp:
            return False
        parts = []
        if isinstance(resp, str):
            parts.append(resp)
        parts.extend(
            [
                str(getattr(resp, "completion_text", "") or ""),
                str(getattr(resp, "error", "") or ""),
                str(getattr(resp, "err", "") or ""),
            ]
        )
        haystack = " ".join(parts).lower()
        return any(keyword.lower() in haystack for keyword in keywords)

    def _get_provider_lock(self, provider_id: str) -> asyncio.Lock:
        if provider_id not in self._provider_locks:
            self._provider_locks[provider_id] = asyncio.Lock()
        return self._provider_locks[provider_id]

    def _apply_provider_overrides(
        self, provider: Any, base_url: str, api_key: str
    ) -> List[tuple]:
        restores: List[tuple] = []
        if base_url:
            for attr in ("base_url", "api_base", "api_base_url", "endpoint", "host"):
                if hasattr(provider, attr):
                    old = getattr(provider, attr)
                    if old != base_url:
                        setattr(provider, attr, base_url)
                        restores.append(("attr", attr, old))
            cfg = getattr(provider, "config", None)
            if isinstance(cfg, dict):
                old = cfg.get("base_url")
                if old != base_url:
                    cfg["base_url"] = base_url
                    restores.append(("config", "base_url", old))
        if api_key:
            for attr in ("api_key", "key", "token", "access_token"):
                if hasattr(provider, attr):
                    old = getattr(provider, attr)
                    if old != api_key:
                        setattr(provider, attr, api_key)
                        restores.append(("attr", attr, old))
            cfg = getattr(provider, "config", None)
            if isinstance(cfg, dict):
                old = cfg.get("api_key")
                if old != api_key:
                    cfg["api_key"] = api_key
                    restores.append(("config", "api_key", old))
        return restores

    def _restore_provider_overrides(self, provider: Any, restores: List[tuple]) -> None:
        for kind, key, old in restores:
            if kind == "attr":
                if hasattr(provider, key):
                    setattr(provider, key, old)
            elif kind == "config":
                cfg = getattr(provider, "config", None)
                if isinstance(cfg, dict):
                    if old is None:
                        cfg.pop(key, None)
                    else:
                        cfg[key] = old

    @contextmanager
    def _provider_overrides(self, provider: Any, entry: Optional[_FallbackEntry]):
        restores: List[tuple] = []
        if entry:
            restores = self._apply_provider_overrides(
                provider, entry.base_url, entry.api_key
            )
        try:
            yield
        finally:
            self._restore_provider_overrides(provider, restores)

    def _sanitize_log_text(self, text: str) -> str:
        """Remove potential secrets from log output."""
        return _SENSITIVE_PATTERN.sub(r'\1***', text)

    async def _build_failover_plan(self, primary: Any):
        plan = []
        if primary:
            primary_id = self._get_provider_id(primary)
            if not await self._is_in_cooldown(primary_id):
                plan.append((primary, None, True))
        entries = self._get_fallback_entries()
        if not entries:
            return plan or ([(primary, None, True)] if primary else [])

        providers = {
            self._get_provider_id(p): p for p in self._get_all_chat_providers()
        }
        for entry in entries:
            if await self._is_in_cooldown(entry.provider_id):
                continue
            provider = providers.get(entry.provider_id)
            if provider is None and hasattr(self.context, "get_provider_by_id"):
                try:
                    provider = self.context.get_provider_by_id(entry.provider_id)
                except Exception:
                    provider = None
            if provider is None:
                astr_logger.warning(
                    "Fallback provider not found: %s", entry.provider_id
                )
                continue
            if provider is primary and not (
                entry.base_url or entry.api_key or entry.model
            ):
                continue
            plan.append((provider, entry, False))
        if not plan and primary:
            plan = [(primary, None, True)]
        return plan

    def _is_wrapped(self, provider: Any) -> bool:
        if provider in self._wrapped:
            return True
        return id(provider) in self._wrapped_ids

    def _mark_wrapped(self, provider: Any) -> None:
        try:
            self._wrapped.add(provider)
        except TypeError:
            self._wrapped_ids.add(id(provider))

    def _wrap_text_chat(self, provider: Any) -> None:
        if not hasattr(provider, "text_chat"):
            return
        if not hasattr(provider, "_silent_provider_switcher_original_text_chat"):
            provider._silent_provider_switcher_original_text_chat = provider.text_chat
        if getattr(provider, "_silent_provider_switcher_text_wrapped", False):
            return

        async def wrapper(p_self, *args, **kwargs):
            return await self._execute_with_failover(p_self, *args, **kwargs)

        provider.text_chat = types.MethodType(wrapper, provider)
        provider._silent_provider_switcher_text_wrapped = True

    def _wrap_text_chat_stream(self, provider: Any) -> None:
        if not hasattr(provider, "text_chat_stream"):
            return
        if not hasattr(provider, "_silent_provider_switcher_original_text_chat_stream"):
            provider._silent_provider_switcher_original_text_chat_stream = (
                provider.text_chat_stream
            )
        if getattr(provider, "_silent_provider_switcher_stream_wrapped", False):
            return

        _switcher = self
        _bound_provider = provider

        async def stream_wrapper(*args, **kwargs):
            async for chunk in _switcher._execute_stream_with_failover(
                _bound_provider, *args, **kwargs
            ):
                yield chunk

        provider.text_chat_stream = stream_wrapper
        provider._silent_provider_switcher_stream_wrapped = True

    async def _install_provider_failover(self):
        if not self._is_enabled():
            return
        providers = self._get_all_chat_providers()
        if not providers:
            return
        async with self._wrap_lock:
            for provider in providers:
                if self._is_wrapped(provider):
                    continue
                self._wrap_text_chat(provider)
                self._wrap_text_chat_stream(provider)
                self._mark_wrapped(provider)

    def _get_prompt_preview(self, args: tuple, kwargs: dict) -> str:
        raw = ""
        if args:
            raw = str(args[0])[:80]
        elif "prompt" in kwargs:
            raw = str(kwargs["prompt"])[:80]
        return self._sanitize_log_text(raw) if raw else ""

    def _log_timing(self, provider_id: str, elapsed: float, ok: bool) -> None:
        if not self._should_log_switch():
            return
        status = "ok" if ok else "error"
        astr_logger.info(
            "Provider %s finished (%s) in %.2f s",
            provider_id,
            status,
            elapsed,
        )

    def _log_exception(
        self, provider_id: str, exc: Exception, prompt_preview: str
    ) -> None:
        if not self._should_log_errors():
            return
        astr_logger.exception(
            "Provider %s raised %s: %s (prompt=%s)",
            provider_id,
            type(exc).__name__,
            exc,
            prompt_preview or "[empty]",
        )

    def _log_error_response(
        self, provider_id: str, resp: Any, prompt_preview: str
    ) -> None:
        if not self._should_log_errors():
            return
        err = getattr(resp, "error", None)
        err_msg = getattr(resp, "error_message", None) or getattr(resp, "err", None)
        text = getattr(resp, "completion_text", None)
        astr_logger.error(
            "Provider %s returned error response: error=%s err=%s text=%s (prompt=%s)",
            provider_id,
            err,
            err_msg,
            (text[:200] if isinstance(text, str) else text),
            prompt_preview or "[empty]",
        )

    def _safe_clone(
        self, obj: Any, memo: Optional[Dict[int, Any]] = None, _depth: int = 0
    ) -> Any:
        if _depth > _MAX_RECURSION_DEPTH:
            astr_logger.debug(
                "_safe_clone 递归深度超限 (%d)，返回原对象。", _depth
            )
            return obj
        if memo is None:
            memo = {}
        obj_id = id(obj)
        if obj_id in memo:
            return memo[obj_id]
        try:
            cloned = copy.deepcopy(obj)
            memo[obj_id] = cloned
            return cloned
        except Exception as exc:
            astr_logger.debug("deepcopy 失败，回退为安全拷贝: %s", exc)
            if isinstance(obj, dict):
                cloned_dict: Dict[Any, Any] = {}
                memo[obj_id] = cloned_dict
                for key, value in obj.items():
                    cloned_dict[
                        self._safe_clone(key, memo, _depth + 1)
                    ] = self._safe_clone(value, memo, _depth + 1)
                return cloned_dict
            if isinstance(obj, list):
                cloned_list: List[Any] = []
                memo[obj_id] = cloned_list
                cloned_list.extend(
                    self._safe_clone(value, memo, _depth + 1) for value in obj
                )
                return cloned_list
            if isinstance(obj, tuple):
                cloned_tuple = tuple(
                    self._safe_clone(value, memo, _depth + 1) for value in obj
                )
                memo[obj_id] = cloned_tuple
                return cloned_tuple
            if isinstance(obj, set):
                cloned_set = {
                    self._safe_clone(value, memo, _depth + 1) for value in obj
                }
                memo[obj_id] = cloned_set
                return cloned_set
            return obj

    def _peek_function_response_names(self, payload: Any) -> List[str]:
        names: List[str] = []
        try:
            if isinstance(payload, dict):
                contents = payload.get("contents")
                if isinstance(contents, list):
                    for content in contents:
                        if not isinstance(content, dict):
                            continue
                        parts = content.get("parts")
                        if not isinstance(parts, list):
                            continue
                        for part in parts:
                            if not isinstance(part, dict):
                                continue
                            func_resp = part.get("function_response")
                            if not isinstance(func_resp, dict):
                                continue
                            names.append(str(func_resp.get("name", "")))
                messages = payload.get("messages")
                if isinstance(messages, list):
                    for msg in messages:
                        if not isinstance(msg, dict):
                            continue
                        if msg.get("role") in {"tool", "function"}:
                            names.append(str(msg.get("name", "")))
        except Exception as exc:
            astr_logger.debug("函数名预览失败: %s", exc)
        return names

    def _sanitize_payload(
        self, call_args: Any, call_kwargs: Dict[str, Any], provider_id: str
    ) -> None:
        removed = 0
        preview_payload = call_kwargs if call_kwargs else {}
        if not preview_payload and call_args:
            first = call_args[0]
            if isinstance(first, dict):
                preview_payload = first

        names_before = self._peek_function_response_names(preview_payload)
        if names_before:
            astr_logger.debug(
                "failover payload preview provider=%s function_response.names=%s",
                provider_id,
                names_before,
            )

        def _is_empty_name(value: Any) -> bool:
            if value is None:
                return True
            if isinstance(value, str):
                return value.strip() == ""
            return False

        def _sanitize_contents(contents: list) -> None:
            nonlocal removed
            new_contents = []
            for content in contents:
                if not isinstance(content, dict):
                    new_contents.append(content)
                    continue
                parts = content.get("parts")
                if not isinstance(parts, list):
                    new_contents.append(content)
                    continue
                new_parts = []
                for part in parts:
                    if not isinstance(part, dict):
                        new_parts.append(part)
                        continue
                    func_resp = part.get("function_response")
                    if isinstance(func_resp, dict) and _is_empty_name(
                        func_resp.get("name")
                    ):
                        removed += 1
                        continue
                    new_parts.append(part)
                if new_parts:
                    content = dict(content)
                    content["parts"] = new_parts
                    new_contents.append(content)
                else:
                    removed += 1
            contents[:] = new_contents

        def _sanitize_messages(messages: list) -> None:
            nonlocal removed
            new_messages = []
            for msg in messages:
                if not isinstance(msg, dict):
                    new_messages.append(msg)
                    continue
                if msg.get("role") in {"tool", "function"} and _is_empty_name(
                    msg.get("name")
                ):
                    removed += 1
                    continue
                new_messages.append(msg)
            messages[:] = new_messages

        if isinstance(preview_payload, dict):
            contents = preview_payload.get("contents")
            if isinstance(contents, list):
                _sanitize_contents(contents)
            messages = preview_payload.get("messages")
            if isinstance(messages, list):
                _sanitize_messages(messages)

        if removed:
            astr_logger.warning(
                "清理空的 function_response.name: provider=%s removed=%d",
                provider_id,
                removed,
            )

    async def _execute_with_failover(self, primary, *args, **kwargs):
        plan = await self._build_failover_plan(primary)
        errors = []
        prompt_preview = self._get_prompt_preview(args, kwargs)

        for index, (provider, entry, is_primary) in enumerate(plan):
            provider_id = self._get_provider_id(provider)
            original_call = getattr(
                provider,
                "_silent_provider_switcher_original_text_chat",
                provider.text_chat,
            )
            call_args = self._safe_clone(args)
            call_kwargs = self._safe_clone(kwargs)
            if entry and entry.model:
                call_kwargs["model"] = entry.model
            self._sanitize_payload(call_args, call_kwargs, provider_id)
            provider_lock = self._get_provider_lock(provider_id)
            start = self._now()
            try:
                async with provider_lock:
                    with self._provider_overrides(provider, entry):
                        result = await original_call(*call_args, **call_kwargs)
                self._log_timing(provider_id, self._now() - start, True)
                if self._is_error_response(result) or self._matches_error_keywords(
                    result
                ):
                    self._log_error_response(provider_id, result, prompt_preview)
                    await self._mark_cooldown(provider_id)
                    if index < len(plan) - 1:
                        errors.append((provider_id, RuntimeError("LLM error response")))
                        continue
                if not is_primary and self._should_log_switch():
                    astr_logger.info(
                        "Switched to fallback provider: %s (prompt=%s)",
                        provider_id,
                        prompt_preview or "[empty]",
                    )
                return result
            except Exception as exc:
                self._log_timing(provider_id, self._now() - start, False)
                self._log_exception(provider_id, exc, prompt_preview)
                if self._should_failover_exception(exc):
                    await self._mark_cooldown(provider_id)
                    if index < len(plan) - 1:
                        errors.append((provider_id, exc))
                        continue
                raise

        if errors:
            raise errors[-1][1]
        raise RuntimeError("No available provider")

    async def _execute_stream_with_failover(
        self, primary, *args, **kwargs
    ) -> AsyncGenerator[Any, None]:
        plan = await self._build_failover_plan(primary)
        errors = []
        prompt_preview = self._get_prompt_preview(args, kwargs)

        for index, (provider, entry, is_primary) in enumerate(plan):
            provider_id = self._get_provider_id(provider)
            original_stream = getattr(
                provider,
                "_silent_provider_switcher_original_text_chat_stream",
                getattr(provider, "text_chat_stream", None),
            )
            original_call = getattr(
                provider,
                "_silent_provider_switcher_original_text_chat",
                getattr(provider, "text_chat", None),
            )

            if not original_stream and not original_call:
                errors.append((provider_id, RuntimeError("Missing chat method")))
                continue

            call_args = self._safe_clone(args)
            call_kwargs = self._safe_clone(kwargs)
            if entry and entry.model:
                call_kwargs["model"] = entry.model
            self._sanitize_payload(call_args, call_kwargs, provider_id)
            provider_lock = self._get_provider_lock(provider_id)

            emitted = False
            switch_to_next = False
            start = self._now()
            try:
                async with provider_lock:
                  with self._provider_overrides(provider, entry):
                    if original_stream:
                        async for chunk in original_stream(*call_args, **call_kwargs):
                            if not emitted:
                                if self._is_error_response(
                                    chunk
                                ) or self._matches_error_keywords(chunk):
                                    self._log_error_response(
                                        provider_id, chunk, prompt_preview
                                    )
                                    await self._mark_cooldown(provider_id)
                                    if index < len(plan) - 1:
                                        errors.append(
                                            (
                                                provider_id,
                                                RuntimeError("LLM error response"),
                                            )
                                        )
                                        switch_to_next = True
                                        break
                                    raise RuntimeError("LLM error response")
                                emitted = True
                                yield chunk
                            else:
                                yield chunk
                        if switch_to_next:
                            self._log_timing(provider_id, self._now() - start, False)
                            continue
                    else:
                        result = await original_call(*call_args, **call_kwargs)
                        if self._is_error_response(
                            result
                        ) or self._matches_error_keywords(result):
                            self._log_error_response(
                                provider_id, result, prompt_preview
                            )
                            await self._mark_cooldown(provider_id)
                            if index < len(plan) - 1:
                                errors.append(
                                    (provider_id, RuntimeError("LLM error response"))
                                )
                                self._log_timing(
                                    provider_id, self._now() - start, False
                                )
                                continue
                            raise RuntimeError("LLM error response")
                        emitted = True
                        yield result

                self._log_timing(provider_id, self._now() - start, True)
                if not is_primary and self._should_log_switch():
                    astr_logger.info(
                        "Switched to fallback provider(stream): %s (prompt=%s)",
                        provider_id,
                        prompt_preview or "[empty]",
                    )
                return
            except Exception as exc:
                self._log_timing(provider_id, self._now() - start, False)
                self._log_exception(provider_id, exc, prompt_preview)
                if emitted:
                    astr_logger.warning(
                        "流已开始输出，无法回退到下一个 provider: %s",
                        provider_id,
                    )
                    raise
                if self._should_failover_exception(exc):
                    await self._mark_cooldown(provider_id)
                    if index < len(plan) - 1:
                        errors.append((provider_id, exc))
                        continue
                raise

        if errors:
            raise errors[-1][1]
        raise RuntimeError("No available provider")
