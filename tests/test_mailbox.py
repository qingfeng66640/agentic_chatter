"""连续输入 mailbox 测试。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

from types import SimpleNamespace

from ..pipeline import mailbox as mailbox_module
from ..pipeline.mailbox import (
    StreamMailbox,
    clear_mailbox_registry,
    get_stream_mailbox,
    message_key,
)


def _message(message_id: str = "", content: str = "") -> SimpleNamespace:
    """构造最小消息对象。"""
    return SimpleNamespace(
        message_id=message_id,
        stream_id="stream",
        time=123.0,
        sender_id="user",
        sender_name="测试用户",
        message_type="text",
        reply_to=None,
        content=content,
        processed_plain_text=content,
    )


async def test_merge_deduplicates_message_id_and_preserves_order() -> None:
    """合并快照按消息标识去重，并按首次出现顺序交付待处理消息。"""
    mailbox = StreamMailbox("stream")
    first = _message("first")
    duplicate = _message("first")
    second = _message("second")

    assert await mailbox.merge_snapshot([first, duplicate, second]) == 2
    owner = object()
    generation = await mailbox.try_acquire(owner)
    assert generation is not None
    claim = await mailbox.claim_pending(owner, generation)

    assert claim is not None
    assert claim.messages == (first, second)


async def test_message_without_id_has_stable_fallback_across_snapshots() -> None:
    """无消息标识的等价快照使用稳定键，合并时只保留一条。"""
    first = _message(content="同一条消息")
    second = _message(content="同一条消息")

    assert message_key(first) == message_key(second)

    mailbox = StreamMailbox("stream")
    assert await mailbox.merge_snapshot([first, second]) == 1


async def test_pending_merge_window_starts_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """待处理窗口从首条消息开始计时，后续消息不会重置等待时间。"""
    now = 100.0
    monkeypatch.setattr(mailbox_module.time, "monotonic", lambda: now)
    mailbox = StreamMailbox("stream")

    assert await mailbox.merge_snapshot([_message("first")]) == 1
    assert await mailbox.pending_state(5.0) == (1, 5.0)

    now = 102.0
    assert await mailbox.merge_snapshot([_message("second")]) == 1
    assert await mailbox.pending_state(5.0) == (2, 3.0)

    now = 106.0
    assert await mailbox.pending_state(5.0) == (2, 0.0)


async def test_claim_clears_pending_merge_window() -> None:
    """认领全部待处理消息后，待处理数量和合并等待时间均清零。"""
    mailbox = StreamMailbox("stream")
    owner = object()
    generation = await mailbox.try_acquire(owner)
    assert generation is not None
    await mailbox.merge_snapshot([_message("first")])
    claim = await mailbox.claim_pending(owner, generation)

    assert claim is not None
    assert await mailbox.pending_state(5.0) == (0, 0.0)


async def test_interruption_streak_resets_only_on_commit() -> None:
    """释放认领保留连续中断次数，只有成功提交才将次数清零。"""
    mailbox = StreamMailbox("stream")
    assert await mailbox.record_interruption() == 1
    assert await mailbox.record_interruption() == 2

    owner = object()
    generation = await mailbox.try_acquire(owner)
    assert generation is not None
    await mailbox.merge_snapshot([_message("first")])
    claim = await mailbox.claim_pending(owner, generation)
    assert claim is not None
    assert await mailbox.release_claim(claim)
    assert await mailbox.interruption_state() == 2

    retry = await mailbox.claim_pending(owner, generation)
    assert retry is not None
    assert await mailbox.commit_claim(retry)
    assert await mailbox.interruption_state() == 0


async def test_interruption_streak_is_isolated_by_stream() -> None:
    """一个聊天流的中断记录不影响另一个流的计数。"""
    first = StreamMailbox("first")
    second = StreamMailbox("second")

    await first.record_interruption()

    assert await first.interruption_state() == 1
    assert await second.interruption_state() == 0


async def test_release_restores_claim_before_new_pending() -> None:
    """释放旧认领后，旧消息应排在期间新到消息之前重新交付。"""
    mailbox = StreamMailbox("stream")
    old = _message("old")
    new = _message("new")
    owner = object()
    generation = await mailbox.try_acquire(owner)
    assert generation is not None
    await mailbox.merge_snapshot([old])
    claim = await mailbox.claim_pending(owner, generation)
    assert claim is not None

    await mailbox.merge_snapshot([old, new])
    assert await mailbox.release_claim(claim)
    retry = await mailbox.claim_pending(owner, generation)

    assert retry is not None
    assert retry.messages == (old, new)


async def test_commit_only_removes_claim_messages() -> None:
    """提交仅移除已认领消息，期间新增消息仍可被下一轮认领。"""
    mailbox = StreamMailbox("stream")
    old = _message("old")
    new = _message("new")
    owner = object()
    generation = await mailbox.try_acquire(owner)
    assert generation is not None
    await mailbox.merge_snapshot([old])
    claim = await mailbox.claim_pending(owner, generation)
    assert claim is not None
    await mailbox.merge_snapshot([old, new])

    assert await mailbox.commit_claim(claim)
    next_claim = await mailbox.claim_pending(owner, generation)

    assert next_claim is not None
    assert next_claim.messages == (new,)


async def test_commit_apply_failure_keeps_claim_releasable() -> None:
    """提交回调失败时不得清除 claim，仍可释放重试。"""
    mailbox = StreamMailbox("stream")
    message = _message("retry")
    owner = object()
    generation = await mailbox.try_acquire(owner)
    assert generation is not None
    await mailbox.merge_snapshot([message])
    claim = await mailbox.claim_pending(owner, generation)
    assert claim is not None

    def fail_apply() -> None:
        """模拟提交时应用回调失败。

        Raises:
            RuntimeError: 固定抛出提交应用失败错误。
        """
        raise RuntimeError("apply failed")

    try:
        await mailbox.commit_claim(claim, apply=fail_apply)
    except RuntimeError as exc:
        assert str(exc) == "apply failed"
    else:
        raise AssertionError("提交回调应抛出异常")

    assert await mailbox.release_claim(claim)
    retry = await mailbox.claim_pending(owner, generation)
    assert retry is not None
    assert retry.messages == (message,)


async def test_wrong_owner_or_generation_cannot_modify_claim() -> None:
    """错误持有者、过期代次或未释放认领都不能提前释放持有权。"""
    mailbox = StreamMailbox("stream")
    owner = object()
    generation = await mailbox.try_acquire(owner)
    assert generation is not None
    await mailbox.merge_snapshot([_message("one")])
    claim = await mailbox.claim_pending(owner, generation)
    assert claim is not None

    assert not await mailbox.release_owner(object(), generation)
    assert not await mailbox.release_owner(owner, generation + 1)
    assert not await mailbox.release_owner(owner, generation)
    assert await mailbox.release_claim(claim)
    assert await mailbox.release_owner(owner, generation)


async def test_busy_owner_blocks_same_stream_but_not_other_stream() -> None:
    """同一聊天流被占用时拒绝新持有者，但不阻塞其他流。"""
    first = StreamMailbox("first")
    second = StreamMailbox("second")
    owner = object()

    assert await first.try_acquire(owner) is not None
    assert await first.try_acquire(object()) is None
    assert await second.try_acquire(object()) is not None


def test_registry_isolated_by_stream() -> None:
    """注册表复用同一聊天流的邮箱，同时隔离不同流的实例。"""
    clear_mailbox_registry()

    assert get_stream_mailbox("a") is get_stream_mailbox("a")
    assert get_stream_mailbox("a") is not get_stream_mailbox("b")


def test_clear_registry_does_not_change_rehydrated_message_keys() -> None:
    """清空邮箱注册表不会改变重新构造的等价消息键。"""
    first = message_key(_message(content="消息"))
    clear_mailbox_registry()
    second = message_key(_message(content="消息"))

    assert first == second


async def test_flush_failure_can_release_claim_for_retry() -> None:
    """失败后的认领可释放并重新获取，且待重试消息保持不变。"""
    mailbox = StreamMailbox("stream")
    message = _message("retry")
    owner = object()
    generation = await mailbox.try_acquire(owner)
    assert generation is not None
    await mailbox.merge_snapshot([message])
    claim = await mailbox.claim_pending(owner, generation)
    assert claim is not None

    assert await mailbox.release_claim(claim)
    retry = await mailbox.claim_pending(owner, generation)

    assert retry is not None
    assert retry.messages == (message,)
