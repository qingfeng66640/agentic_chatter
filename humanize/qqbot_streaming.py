"""QQBot C2C 实时流式输出桥接。"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from src.app.plugin_system.api import service_api

from .segmenter import CleanReplyResult, clean_reply_text_with_metadata

DEFAULT_STREAMING_SERVICE = "qqbot_adapter:service:qqbot"
_THOUGHT_TAGS = ("think", "analysis", "reasoning")
_OPENING_THOUGHT_TAG = re.compile(
    r"<\s*(?P<tag>think|analysis|reasoning)\b[^>]*>",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class QQBotStreamingContext:
    """QQBot C2C 流式发送所需上下文。"""

    user_openid: str
    event_id: str = ""
    msg_id: str = ""


def extract_streaming_context(message: Any | None) -> QQBotStreamingContext | None:
    """从 QQ C2C 触发消息提取流式上下文。"""
    if message is None:
        return None
    if str(getattr(message, "platform", "") or "").lower() != "qq":
        return None
    if str(getattr(message, "chat_type", "") or "").lower() == "group":
        return None

    extra = getattr(message, "extra", None)
    extra = extra if isinstance(extra, dict) else {}
    user_openid = str(
        extra.get("qq_user_openid") or getattr(message, "sender_id", "") or ""
    ).strip()
    if not user_openid:
        return None

    event_type = str(extra.get("qq_event_type") or "").upper()
    if event_type == "INTERACTION_CREATE":
        event_id = str(extra.get("qq_event_id") or "").strip()
        return QQBotStreamingContext(user_openid, event_id=event_id) if event_id else None

    msg_id = str(getattr(message, "message_id", "") or "").strip()
    return QQBotStreamingContext(user_openid, msg_id=msg_id) if msg_id else None


def resolve_streaming_service(signature: str) -> Any | None:
    """按配置签名或唯一能力发现解析流式 Service。"""
    target = str(signature or "").strip()
    if target:
        try:
            service = service_api.get_service(target)
        except Exception:
            return None
        return service if callable(getattr(service, "start_streaming", None)) else None

    try:
        candidates = [
            candidate_signature
            for candidate_signature, service_cls in service_api.get_all_services().items()
            if callable(getattr(service_cls, "start_streaming", None))
        ]
    except Exception:
        return None
    if len(candidates) != 1:
        return None
    try:
        service = service_api.get_service(candidates[0])
    except Exception:
        return None
    return service if callable(getattr(service, "start_streaming", None)) else None


async def _maybe_await(value: Any) -> Any:
    """兼容同步测试替身和真实异步 controller。"""
    return await value if inspect.isawaitable(value) else value


@dataclass
class QQBotStreamingSession:
    """单次 LLM 响应对应的 QQBot 流式会话。"""

    context: QQBotStreamingContext
    service: Any
    initial_chars: int = 1
    update_min_chars: int = 1
    controller: Any | None = None
    raw_text: str = ""
    visible_text: str = ""
    last_submitted_text: str = ""
    pending: str = ""
    inside_think: bool = False
    start_failed: bool = False
    update_failed: bool = False
    ended: bool = False
    removed_thoughts: list[str] = field(default_factory=list)
    _thought_buffer: str = ""
    _active_tag: str | None = None

    @property
    def started(self) -> bool:
        """流式 controller 是否已成功启动。"""
        return self.controller is not None

    async def on_event(self, event: Any) -> None:
        """仅消费可见正文增量，忽略 reasoning 与工具事件。"""
        delta = getattr(event, "text_delta", None)
        if not isinstance(delta, str) or not delta:
            return
        self.raw_text += delta
        self._consume_visible_delta(delta)
        await self._flush_visible()

    def _consume_visible_delta(self, delta: str) -> None:
        """过滤可能跨 chunk 的思考标签和块内容。"""
        self.pending += delta
        output: list[str] = []
        while self.pending:
            lowered = self.pending.lower()
            if self.inside_think and self._active_tag:
                close = f"</{self._active_tag}>"
                index = lowered.find(close)
                if index >= 0:
                    self._thought_buffer += self.pending[:index]
                    self.pending = self.pending[index + len(close) :]
                    thought = self._thought_buffer.strip()
                    if thought:
                        self.removed_thoughts.append(thought)
                    self._thought_buffer = ""
                    self._active_tag = None
                    self.inside_think = False
                    continue
                keep = self._possible_marker_suffix(self.pending, close)
                self._thought_buffer += self.pending[:-keep] if keep else self.pending
                self.pending = self.pending[-keep:] if keep else ""
                break

            opening = self._find_opening_tag(lowered)
            if opening is not None:
                index, tag, marker_end = opening
                output.append(self.pending[:index])
                self.pending = self.pending[marker_end:]
                self._active_tag = tag
                self.inside_think = True
                continue

            keep = self._possible_opening_suffix(self.pending)
            output.append(self.pending[:-keep] if keep else self.pending)
            self.pending = self.pending[-keep:] if keep else ""
            break
        if output:
            self.visible_text += "".join(output)

    @staticmethod
    def _find_opening_tag(text: str) -> tuple[int, str, int] | None:
        """查找首个完整的思考开始标签，兼容标签属性与空白。"""
        match = _OPENING_THOUGHT_TAG.search(text)
        if match is None:
            return None
        return match.start(), str(match.group("tag")).lower(), match.end()

    @staticmethod
    def _possible_opening_suffix(text: str) -> int:
        """返回可能构成思考开始标签的尾部长度。"""
        lowered = text.lower()
        start = lowered.rfind("<")
        if start < 0:
            return 0
        suffix = lowered[start:]
        if ">" in suffix or len(suffix) > 128:
            return 0
        candidate = suffix[1:].lstrip()
        if not candidate:
            return len(suffix)
        for tag in _THOUGHT_TAGS:
            if tag.startswith(candidate):
                return len(suffix)
            if candidate.startswith(tag) and (
                len(candidate) == len(tag) or candidate[len(tag)].isspace()
            ):
                return len(suffix)
        return 0

    @staticmethod
    def _possible_marker_suffix(text: str, marker: str) -> int:
        """返回可能构成完整标签的尾部长度。"""
        lowered = text.lower()
        return max(
            (
                size
                for size in range(1, min(len(lowered), len(marker) - 1) + 1)
                if marker.startswith(lowered[-size:])
            ),
            default=0,
        )

    async def _flush_visible(self) -> None:
        """按完整正文前缀启动或更新 QQBot 流。"""
        current = self.visible_text
        if not current or self.start_failed or self.update_failed:
            return
        if not self.started:
            if len(current) < max(1, int(self.initial_chars)):
                return
            try:
                result = await self.service.start_streaming(
                    user_openid=self.context.user_openid,
                    initial_text=current,
                    event_id=self.context.event_id,
                    msg_id=self.context.msg_id,
                )
            except Exception:
                self.start_failed = True
                return
            controller = (
                result.get("controller")
                if isinstance(result, dict) and result.get("success")
                else None
            )
            if (
                controller is None
                or not callable(getattr(controller, "update", None))
                or not callable(getattr(controller, "end", None))
            ):
                self.start_failed = True
                return
            self.controller = controller
            self.last_submitted_text = current
            return

        if len(current) - len(self.last_submitted_text) < max(
            1, int(self.update_min_chars)
        ):
            return
        try:
            succeeded = await _maybe_await(self.controller.update(current))
            if succeeded is False:
                raise RuntimeError("controller.update 返回失败")
            self.last_submitted_text = current
        except Exception:
            self.update_failed = True

    async def finalize(self, final_raw_text: str) -> CleanReplyResult:
        """使用完整正文权威清洗并结束已启动的 QQBot 流。"""
        result = clean_reply_text_with_metadata(final_raw_text or self.raw_text)
        if not self.started or self.ended:
            return result
        self.ended = True
        final_text = result.text
        if not final_text.startswith(self.last_submitted_text):
            final_text = self.visible_text
        try:
            await _maybe_await(self.controller.end(final_text))
        except Exception:
            pass
        return result


def create_streaming_session(
    message: Any | None,
    *,
    enabled: bool,
    service_signature: str = DEFAULT_STREAMING_SERVICE,
    initial_chars: int = 1,
    update_min_chars: int = 1,
    service_resolver: Callable[[str], Any | None] = resolve_streaming_service,
) -> QQBotStreamingSession | None:
    """在配置、消息和 Service 均满足时创建流式会话。"""
    if not enabled:
        return None
    context = extract_streaming_context(message)
    if context is None:
        return None
    service = service_resolver(service_signature)
    if service is None:
        return None
    return QQBotStreamingSession(
        context=context,
        service=service,
        initial_chars=max(1, int(initial_chars)),
        update_min_chars=max(1, int(update_min_chars)),
    )


__all__ = [
    "DEFAULT_STREAMING_SERVICE",
    "QQBotStreamingContext",
    "QQBotStreamingSession",
    "create_streaming_session",
    "extract_streaming_context",
    "resolve_streaming_service",
]
