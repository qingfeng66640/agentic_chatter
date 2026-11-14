"""控制词模糊匹配与任务消歧工具函数的测试。"""

from __future__ import annotations

import pytest

from ..task_runtime.control import (
    classify_task_message,
    describe_tasks,
    match_task_ref,
    resolve_control_target,
)
from ..task_runtime.models import TaskState, TaskStatus, TaskType
from ..task_runtime.runtime import TaskRuntime


def _make_task(
    stream_id: str, goal: str, status: TaskStatus = TaskStatus.RUNNING
) -> TaskRuntime:
    """构造通用任务，并设置控制指令消歧需要的状态。

    Args:
        stream_id: 任务所属的聊天流标识。
        goal: 用于构造任务状态的用户目标。
        status: 支持暂停、等待用户和失败状态，其他值保留运行状态。

    Returns:
        已启动并完成所需状态切换的任务运行时。
    """
    runtime = TaskRuntime(
        TaskState(
            stream_id=stream_id,
            user_goal=goal,
            task_type=TaskType.GENERAL,
        )
    )
    runtime.start()
    if status == TaskStatus.PAUSED:
        runtime.pause()
    elif status == TaskStatus.WAITING_USER:
        runtime.set_waiting_user("需要确认")
    elif status == TaskStatus.FAILED:
        runtime.fail("失败", "原因")
    return runtime


@pytest.mark.parametrize(
    ("text", "command"),
    [
        ("取消任务", "cancel"),
        ("取消这个任务", "cancel"),
        ("停止任务", "cancel"),
        ("终止任务", "cancel"),
        ("放弃任务", "cancel"),
        ("别做了", "cancel"),
        ("不用做了", "cancel"),
        ("不用继续了", "cancel"),
        ("取消任务！", "cancel"),
        ("暂停任务", "pause"),
        ("任务暂停", "pause"),
        ("继续任务", "resume"),
        ("恢复任务", "resume"),
        ("任务状态", "status"),
        ("任务进度", "status"),
        ("任务列表", "status"),
        ("有什么任务", "status"),
        ("任务怎么样了", "status"),
    ],
)
def test_fuzzy_control_phrases(text: str, command: str) -> None:
    """常见控制短语应归类为控制消息并映射到对应指令。

    Args:
        text: 参数化提供的自然语言控制短语。
        command: 该短语应对应的规范化控制指令。
    """
    message = classify_task_message(text)
    assert message.kind.value == "control"
    assert message.text == command


@pytest.mark.parametrize(
    "text",
    [
        "取消订单",
        "帮我停止音乐",
        "取消",
        "继续说",
        "停一下我还没说完",
        "今天的天气怎么样",
        "这个任务很有趣",
        "任务很重",
    ],
)
def test_non_control_messages_stay_chat(text: str) -> None:
    """含相似词语但不构成任务控制指令的消息应保留为闲聊。"""
    message = classify_task_message(text)
    assert message.kind.value == "chat"


def test_exact_control_words_still_work() -> None:
    """明确的状态查询和带首尾空白的取消指令仍应被正确识别。"""
    assert classify_task_message("查看任务状态").text == "status"
    assert classify_task_message("  取消任务  ").text == "cancel"


def test_match_task_ref_by_ordinal() -> None:
    """带井号、号字或任务前缀的序号应定位任务，越界序号应无匹配。"""
    tasks = [_make_task("s", "任务一"), _make_task("s", "任务二")]
    assert match_task_ref("#2", tasks) is tasks[1]
    assert match_task_ref("2号 取消", tasks) is tasks[1]
    assert match_task_ref("任务1", tasks) is tasks[0]
    assert match_task_ref("#9", tasks) is None


def test_match_task_ref_by_id_prefix() -> None:
    """八位任务 ID 前缀应定位目标，而四位短前缀不应命中。"""
    tasks = [_make_task("s", "任务一"), _make_task("s", "任务二")]
    prefix = tasks[1].state.task_id[:8]
    assert match_task_ref(f"{prefix} 取消", tasks) is tasks[1]
    # 短前缀不命中
    assert match_task_ref(f"{tasks[1].state.task_id[:4]} 取消", tasks) is None


def test_match_task_ref_ignores_plain_text() -> None:
    """不包含任务标识的普通取消文本不应匹配具体任务。"""
    tasks = [_make_task("s", "任务一")]
    assert match_task_ref("取消任务", tasks) is None


def test_describe_tasks_renders_list() -> None:
    """任务列表应包含总数、运行与暂停状态、任务 ID 以及序号说明。"""
    tasks = [
        _make_task("s", "一个超过三十个字符长度的任务目标用来验证截断逻辑是否生效正确"),
        _make_task("s", "第二个", TaskStatus.PAUSED),
    ]
    text = describe_tasks(tasks)
    assert "共 2 个任务" in text
    assert "[运行中]" in text
    assert "[已暂停]" in text
    assert "任务ID=" in text
    assert "序号为当前排序" in text


def test_resolve_control_target_single_task_direct() -> None:
    """只有一个任务时，取消指令应直接选中它且不提示消歧。"""
    tasks = [_make_task("s", "唯一任务")]
    target, notice = resolve_control_target("cancel", "取消任务", tasks)
    assert target is tasks[0]
    assert notice == ""


def test_resolve_control_target_multi_task_disambiguation() -> None:
    """多个任务均可取消时，不带目标的指令应返回任务列表供消歧。"""
    tasks = [_make_task("s", "任务一"), _make_task("s", "任务二")]
    target, notice = resolve_control_target("cancel", "取消任务", tasks)
    assert target is None
    assert "共 2 个任务" in notice


def test_resolve_control_target_pause_narrows_to_running() -> None:
    """暂停指令应排除已暂停任务，直接选中唯一运行中的任务。"""
    running = _make_task("s", "运行中")
    paused = _make_task("s", "已暂停", TaskStatus.PAUSED)
    target, notice = resolve_control_target("pause", "暂停任务", [running, paused])
    assert target is running
    assert notice == ""


def test_resolve_control_target_resume_narrows_to_resumable() -> None:
    """继续指令应排除运行中的任务，直接选中唯一已暂停任务。"""
    running = _make_task("s", "运行中")
    paused = _make_task("s", "已暂停", TaskStatus.PAUSED)
    target, notice = resolve_control_target("resume", "继续任务", [running, paused])
    assert target is paused
    assert notice == ""


def test_resolve_control_target_status_needs_no_target() -> None:
    """状态查询无需选择单个任务，也无需产生消歧提示。"""
    tasks = [_make_task("s", "任务一"), _make_task("s", "任务二")]
    target, notice = resolve_control_target("status", "任务状态", tasks)
    assert target is None
    assert notice == ""


def test_resolve_control_target_empty_tasks() -> None:
    """没有候选任务时应返回空目标和空提示。"""
    target, notice = resolve_control_target("cancel", "取消任务", [])
    assert target is None
    assert notice == ""
