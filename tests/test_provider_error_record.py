"""供应商异常请求体受控记录测试。"""

from __future__ import annotations

import json

import pytest

from src.kernel.llm import LLMPayload, ROLE, Text, ToolCall, ToolResult
from src.kernel.llm.payload.content import ReasoningText

from ..tooling import provider_error_record as record_module
from ..tooling.provider_error_record import (
    append_provider_error_request_record,
    build_provider_error_request_record,
)


class _SearchTool:
    @classmethod
    def to_schema(cls) -> dict:
        return {
            "function": {
                "name": "tool-search",
                "description": "检索资料",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            }
        }


def _payloads() -> list[LLMPayload]:
    return [
        LLMPayload(
            ROLE.SYSTEM,
            Text(
                "你是小狐。核心人格：热情。身份：测试助手。"
                "背景：来自测试群。回复风格：简洁。工具规则保留。"
            ),
        ),
        LLMPayload(ROLE.USER, Text("用户说小狐和热情都应该原样保留。")),
        LLMPayload(ROLE.TOOL, [_SearchTool]),
        LLMPayload(
            ROLE.ASSISTANT,
            [
                ReasoningText("不应记录的推理"),
                ToolCall(
                    id="call-search",
                    name="tool-search",
                    args={"query": "天气", "api_key": "secret-value"},
                ),
            ],
        ),
        LLMPayload(
            ROLE.TOOL_RESULT,
            ToolResult(
                value={"result": "晴天", "authorization": "hidden"},
                call_id="call-search",
                name="tool-search",
            ),
        ),
    ]


def test_request_record_redacts_system_persona_and_preserves_user_and_tools() -> None:
    record = build_provider_error_request_record(
        _payloads(),
        rule="google_prompt_policy_block",
        stream_id="stream-1",
        provider_error_chars=123,
        persona_values={
            "nickname": "小狐",
            "personality_core": "热情",
            "identity": "测试助手",
            "background_story": "来自测试群",
            "reply_style": "简洁",
        },
    )

    system = record["payloads"][0]["content"][0]
    user = record["payloads"][1]["content"][0]
    tool_call = record["payloads"][3]["content"][1]
    tool_result = record["payloads"][4]["content"][0]

    assert "{{AGENTIC_NICKNAME}}" in system
    assert "{{AGENTIC_PERSONALITY_CORE}}" in system
    assert "{{AGENTIC_IDENTITY}}" in system
    assert "{{AGENTIC_BACKGROUND_STORY}}" in system
    assert "{{AGENTIC_REPLY_STYLE}}" in system
    assert "工具规则保留" in system
    assert "小狐" not in system
    assert "用户说小狐和热情都应该原样保留。" == user
    assert tool_call["id"] == "call-search"
    assert tool_call["name"] == "tool-search"
    assert "secret-value" not in tool_call["args"]
    assert "[REDACTED]" in tool_call["args"]
    assert tool_result["call_id"] == "call-search"
    assert tool_result["name"] == "tool-search"
    assert "hidden" not in tool_result["value"]
    assert "[REDACTED]" in tool_result["value"]
    assert record["payloads"][3]["content"][0] == {
        "type": "reasoning",
        "omitted": True,
    }


def test_request_record_honors_total_size_limit(monkeypatch) -> None:
    """总记录超过上限时仅保留装得下的完整 payload。"""
    monkeypatch.setattr(record_module, "_MAX_TOTAL_CHARS", 900)
    record = build_provider_error_request_record(
        [LLMPayload(ROLE.USER, Text("内容" * 1_000))],
        rule="google_prompt_policy_block",
        stream_id="stream-1",
        provider_error_chars=123,
    )

    assert record["truncated"]
    assert "total" in record["truncated_fields"]
    assert not record["payloads"]
    assert len(json.dumps(record, ensure_ascii=False, separators=(",", ":"))) <= 900


@pytest.mark.asyncio
async def test_append_request_record_uses_jsonl(monkeypatch, tmp_path) -> None:
    target = tmp_path / "provider_error_requests.jsonl"
    monkeypatch.setattr(record_module, "_RECORD_PATH", target)
    record = build_provider_error_request_record(
        [],
        rule="google_prompt_policy_block",
        stream_id="stream-1",
        provider_error_chars=123,
    )

    path = await append_provider_error_request_record(record)
    await append_provider_error_request_record(record)

    assert path == target
    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["provider_error_rule"] == "google_prompt_policy_block"
