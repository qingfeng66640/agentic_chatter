"""私聊 Chatter 注册与模型选择测试。"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from src.core.components.types import ChatType
from src.kernel.llm import LLMContextManager

from .. import chatter as chatter_module
from ..chatter import (
    AgenticChatter,
    AgenticDiscussChatter,
    AgenticPrivateChatter,
)
from ..config import AgenticChatterConfig
from ..plugin import AgenticChatterPlugin


def _plugin_with_config(config: AgenticChatterConfig) -> AgenticChatterPlugin:
    plugin = object.__new__(AgenticChatterPlugin)
    plugin.config = config
    return plugin


def test_private_config_defaults_preserve_existing_behavior() -> None:
    config = AgenticChatterConfig()

    assert config.plugin.private_enabled
    assert config.plugin.private_model_name == ""


def test_components_follow_plugin_and_private_switches() -> None:
    config = AgenticChatterConfig()
    plugin = _plugin_with_config(config)

    components = plugin.get_components()
    assert AgenticChatter in components
    assert AgenticPrivateChatter in components
    assert AgenticDiscussChatter in components

    config.plugin.private_enabled = False
    components = plugin.get_components()
    assert AgenticChatter in components
    assert AgenticPrivateChatter not in components
    assert AgenticDiscussChatter in components

    config.plugin.enabled = False
    assert plugin.get_components() == []


def test_chatter_types_are_precise_and_unique() -> None:
    assert AgenticChatter.chat_type == ChatType.GROUP
    assert AgenticPrivateChatter.chat_type == ChatType.PRIVATE
    assert AgenticDiscussChatter.chat_type == ChatType.DISCUSS
    assert AgenticChatter.name == "agentic"
    assert AgenticPrivateChatter.name == "agentic_private"
    assert AgenticDiscussChatter.name == "agentic_discuss"


def test_private_act_request_falls_back_to_model_task(monkeypatch) -> None:
    config = AgenticChatterConfig()
    config.plugin.model_task = "custom_actor"
    config.plugin.private_model_name = ""
    chatter = AgenticPrivateChatter(stream_id="private-stream", plugin=object())
    create_request = Mock(return_value=object())
    monkeypatch.setattr(chatter, "create_request", create_request)

    result = chatter._create_act_request(config)

    assert result is create_request.return_value
    create_request.assert_called_once_with(
        task="custom_actor",
        request_name="agentic_private",
    )


def test_private_act_request_uses_named_model(monkeypatch) -> None:
    config = AgenticChatterConfig()
    config.plugin.private_model_name = "private-chat-model"
    chatter = AgenticPrivateChatter(stream_id="private-stream", plugin=object())
    model_set = [{"model_identifier": "provider/private"}]
    get_model = Mock(return_value=model_set)
    get_task = Mock(
        return_value=[{"temperature": 0.35, "max_tokens": 2048}]
    )
    monkeypatch.setattr(chatter_module.llm_api, "get_model_set_by_name", get_model)
    monkeypatch.setattr(chatter_module.llm_api, "get_model_set_by_task", get_task)

    request = chatter._create_act_request(config)

    get_task.assert_called_once_with("actor")
    get_model.assert_called_once_with(
        "private-chat-model",
        temperature=0.35,
        max_tokens=2048,
    )
    assert request.model_set == model_set
    assert request.request_name == "agentic_private"
    assert request.meta_data == {"stream_id": "private-stream"}
    assert isinstance(request.context_manager, LLMContextManager)


def test_group_act_request_ignores_private_model(monkeypatch) -> None:
    config = AgenticChatterConfig()
    config.plugin.model_task = "group_actor"
    config.plugin.private_model_name = "private-chat-model"
    chatter = AgenticChatter(stream_id="group-stream", plugin=object())
    create_request = Mock(return_value=SimpleNamespace())
    get_model = Mock()
    monkeypatch.setattr(chatter, "create_request", create_request)
    monkeypatch.setattr(chatter_module.llm_api, "get_model_set_by_name", get_model)

    result = chatter._create_act_request(config)

    assert result is create_request.return_value
    create_request.assert_called_once_with(
        task="group_actor",
        request_name="agentic",
    )
    get_model.assert_not_called()
