"""供应商异常请求体的受控记录。"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.kernel.llm.payload.content import File, ReasoningText, Text
from src.kernel.llm.payload.tooling import ToolCall, ToolResult
from src.kernel.llm.roles import ROLE

from .dedupe import redact_sensitive_value

_RECORD_PATH = Path("data/agentic_chatter/provider_error_requests.jsonl")
_MAX_PAYLOADS = 32
_MAX_SYSTEM_CHARS = 20_000
_MAX_USER_CHARS = 30_000
_MAX_TOOL_CHARS = 12_000
_MAX_TOTAL_CHARS = 120_000
_PERSONA_FIELDS = {
    "nickname": "{{AGENTIC_NICKNAME}}",
    "alias_names": "{{AGENTIC_ALIAS_NAMES}}",
    "personality_core": "{{AGENTIC_PERSONALITY_CORE}}",
    "personality_side": "{{AGENTIC_PERSONALITY_SIDE}}",
    "identity": "{{AGENTIC_IDENTITY}}",
    "background_story": "{{AGENTIC_BACKGROUND_STORY}}",
    "reply_style": "{{AGENTIC_REPLY_STYLE}}",
}


def _truncate(text: str, limit: int, field: str, truncated_fields: list[str]) -> str:
    if len(text) <= limit:
        return text
    truncated_fields.append(field)
    return f"{text[: max(0, limit - 32)]}...[truncated:{limit}]"


def _json_text(value: Any) -> str:
    try:
        return json.dumps(redact_sensitive_value(value), ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return "[UNSERIALIZABLE]"


def _redact_system_text(text: str, persona_values: dict[str, Any]) -> str:
    result = text
    for field, replacement in _PERSONA_FIELDS.items():
        value = persona_values.get(field)
        if isinstance(value, (list, tuple)):
            candidates = [str(item) for item in value if str(item)]
        elif value is None:
            candidates = []
        else:
            candidates = [str(value)]
        for candidate in sorted(candidates, key=len, reverse=True):
            if candidate:
                result = result.replace(candidate, replacement)
    return result


def _serialize_file(content: File) -> dict[str, Any]:
    result: dict[str, Any] = {"type": type(content).__name__}
    mime_type = getattr(content, "mime_type", None)
    if mime_type:
        result["mime_type"] = str(mime_type)
    result["omitted"] = True
    return result


def _serialize_content(
    content: Any,
    *,
    role: ROLE,
    persona_values: dict[str, Any],
    truncated_fields: list[str],
) -> Any:
    if isinstance(content, ReasoningText):
        return {"type": "reasoning", "omitted": True}
    if isinstance(content, Text):
        limit = _MAX_SYSTEM_CHARS if role == ROLE.SYSTEM else _MAX_USER_CHARS
        text = content.text
        if role == ROLE.SYSTEM:
            text = _redact_system_text(text, persona_values)
        return _truncate(text, limit, f"{role.value}.text", truncated_fields)
    if isinstance(content, ToolCall):
        return {
            "type": "tool_call",
            "id": content.id,
            "name": content.name,
            "args": _truncate(
                _json_text(content.args),
                _MAX_TOOL_CHARS,
                "tool_call.args",
                truncated_fields,
            ),
        }
    if isinstance(content, ToolResult):
        value = _json_text(content.value)
        return {
            "type": "tool_result",
            "call_id": content.call_id,
            "name": content.name,
            "value": _truncate(
                value,
                _MAX_TOOL_CHARS,
                "tool_result.value",
                truncated_fields,
            ),
        }
    if isinstance(content, File):
        return _serialize_file(content)
    if isinstance(content, dict):
        return redact_sensitive_value(content)
    if isinstance(content, (str, int, float, bool)) or content is None:
        return content
    if role == ROLE.TOOL and hasattr(content, "to_schema"):
        try:
            return redact_sensitive_value(content.to_schema())
        except Exception:
            return {"type": "tool", "class": type(content).__name__, "omitted": True}
    return {"type": "unknown", "class": type(content).__name__, "omitted": True}


def serialize_payload(
    payload: Any,
    *,
    persona_values: dict[str, Any] | None = None,
    truncated_fields: list[str] | None = None,
) -> dict[str, Any]:
    """将单个 LLM payload 投影为安全 JSON 对象。"""
    persona_values = persona_values or {}
    truncated_fields = truncated_fields if truncated_fields is not None else []
    role = getattr(payload, "role", None)
    if not isinstance(role, ROLE):
        role = ROLE(str(role)) if str(role) in {item.value for item in ROLE} else ROLE.USER
    content = getattr(payload, "content", [])
    return {
        "role": role.value,
        "content": [
            _serialize_content(
                item,
                role=role,
                persona_values=persona_values,
                truncated_fields=truncated_fields,
            )
            for item in content
        ],
    }


def build_provider_error_request_record(
    payloads: Sequence[Any] | None,
    *,
    rule: str,
    stream_id: str,
    provider_error_chars: int,
    persona_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构建供应商异常请求体的受控记录。"""
    truncated_fields: list[str] = []
    items = list(payloads or [])
    payload_records = [
        serialize_payload(
            payload,
            persona_values=persona_values,
            truncated_fields=truncated_fields,
        )
        for payload in items[:_MAX_PAYLOADS]
    ]
    if len(items) > _MAX_PAYLOADS:
        truncated_fields.append("payloads")
    record: dict[str, Any] = {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "stream_id": _truncate(
            str(stream_id), 512, "stream_id", truncated_fields
        ),
        "provider_error_rule": _truncate(
            str(rule), 512, "provider_error_rule", truncated_fields
        ),
        "provider_error_chars": provider_error_chars,
        "payload_count": len(items),
        "payloads": payload_records,
        "truncated": bool(truncated_fields),
        "truncated_fields": sorted(set(truncated_fields)),
    }
    encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    if len(encoded) > _MAX_TOTAL_CHARS:
        total_fields = sorted(set(truncated_fields + ["total"]))
        kept_payloads: list[dict[str, Any]] = []
        for payload_record in payload_records:
            candidate = {
                **record,
                "payloads": [*kept_payloads, payload_record],
                "truncated": True,
                "truncated_fields": total_fields,
            }
            if (
                len(json.dumps(candidate, ensure_ascii=False, separators=(",", ":")))
                > _MAX_TOTAL_CHARS
            ):
                break
            kept_payloads.append(payload_record)
        record["payloads"] = kept_payloads
        record["truncated"] = True
        record["truncated_fields"] = total_fields
    return record


def _append_record_sync(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)


async def append_provider_error_request_record(record: dict[str, Any]) -> Path:
    """将记录追加到插件专属 JSONL 文件并返回路径。"""
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    await asyncio.to_thread(_append_record_sync, _RECORD_PATH, line)
    return _RECORD_PATH


__all__ = [
    "append_provider_error_request_record",
    "build_provider_error_request_record",
    "serialize_payload",
]

