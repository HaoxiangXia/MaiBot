from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src.maisaka.runtime import MaisakaHeartFlowChatting
from src.maisaka.turn_trigger import scheduler as scheduler_module
from src.maisaka.turn_trigger.gates import DynamicReplyTurnGate, TurnGateResult
from src.maisaka.turn_trigger.scheduler import MessageTurnScheduler, TURN_TRIGGER_EVALUATE_HOOK
from src.maisaka.chat_loop_service import register_maisaka_hook_specs
from src.plugin_runtime.host.hook_spec_registry import HookSpecRegistry


def test_turn_trigger_evaluate_hook_spec_registered() -> None:
    registry = HookSpecRegistry()
    register_maisaka_hook_specs(registry)
    spec = registry.get_hook_spec(TURN_TRIGGER_EVALUATE_HOOK)
    assert spec is not None
    assert spec.allow_abort is True
    assert spec.allow_kwargs_mutation is True


def test_dynamic_reply_turn_gate_apply_external_probability() -> None:
    runtime = SimpleNamespace(
        log_prefix="[测试群聊]",
        chat_stream=SimpleNamespace(is_group_session=True),
        _chat_history=[],
    )
    gate = DynamicReplyTurnGate(runtime)
    high_res = gate.apply_external_probability(probability=0.95, frequency=0.5)
    assert high_res.should_trigger is True
    assert high_res.probability == pytest.approx(0.95)

    low_res = gate.apply_external_probability(probability=0.05, frequency=0.5, threshold_override=0.6)
    assert low_res.should_trigger is False
    assert low_res.threshold == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_scheduler_hook_probability_fusion_and_silent_consume(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        scheduler_module,
        "global_config",
        SimpleNamespace(
            bot=SimpleNamespace(nickname="麦麦"),
            chat=SimpleNamespace(reply_timing=SimpleNamespace(planner_message_steering=False)),
        ),
    )
    msg = SimpleNamespace(
        message_id="m1",
        processed_plain_text="今天吃什么？",
        is_at=False,
        is_mentioned=False,
        message_info=SimpleNamespace(user_info=SimpleNamespace(user_cardname="群友A", user_nickname="A", user_id="1001")),
    )
    consumed_records: list[int | None] = []

    async def fake_consume(count: int | None = None) -> int:
        consumed_records.append(count)
        runtime._last_processed_index = len(runtime.message_cache)
        return 1

    runtime = SimpleNamespace(
        session_id="group-1",
        _running=True,
        _agent_state="stop",
        _STATE_WAIT="wait",
        _message_turn_scheduled=False,
        chat_stream=SimpleNamespace(is_group_session=True),
        _get_pending_message_count=lambda: len(runtime.message_cache) - runtime._last_processed_index,
        _get_effective_reply_frequency=lambda: 0.4,
        _is_reply_frequency_silent=lambda: False,
        _has_forced_turn_trigger=lambda: False,
        _idle_backoff=SimpleNamespace(get_delay_seconds=lambda pending_count: 0.0),
        _get_message_quiet_period_remaining_seconds=lambda: 0.0,
        _cancel_wait_timeout_task=Mock(),
        _update_stage_status=Mock(),
        _defer_message_turn_check=Mock(),
        _enqueue_message_turn=Mock(),
        _consume_pending_messages_silently=fake_consume,
        message_cache=[msg],
        _last_processed_index=0,
        _chat_history=[],
        log_prefix="[test]",
    )

    scheduler = MessageTurnScheduler(runtime)
    scheduler._has_evaluate_hook_subscribers = lambda: True

    # Case 1: Hook returns decision="consume" -> silent read without triggering Planner
    fake_manager = SimpleNamespace(
        invoke_hook=AsyncMock(
            return_value=SimpleNamespace(
                aborted=False,
                kwargs={"decision": "consume", "consumed_count": 1, "detail": "Jev判背景闲聊"},
            )
        )
    )
    scheduler._get_runtime_manager = lambda: fake_manager

    base_result = TurnGateResult(
        decision="wait",
        detail="频率=0.400 动态门控: 概率=0.32 阈值=0.50",
        probability=0.32,
        threshold=0.50,
    )
    await scheduler._run_hook_evaluation(base_result=base_result, start_index=0, target_index=1)

    assert consumed_records == [1]
    assert runtime._last_processed_index == 1
    runtime._enqueue_message_turn.assert_not_called()

    # Case 2: Hook fuses probability above threshold -> triggers Planner
    msg2 = SimpleNamespace(
        message_id="m2",
        processed_plain_text="有人会配 NixOS 吗？",
        is_at=False,
        is_mentioned=False,
        message_info=SimpleNamespace(user_info=SimpleNamespace(user_cardname="群友B", user_nickname="B", user_id="1002")),
    )
    runtime.message_cache.append(msg2)
    fake_manager.invoke_hook = AsyncMock(
        return_value=SimpleNamespace(
            aborted=False,
            kwargs={"probability": 0.88, "threshold": 0.50, "detail": "Jev融合 p=0.88"},
        )
    )
    await scheduler._run_hook_evaluation(base_result=base_result, start_index=1, target_index=2)
    runtime._enqueue_message_turn.assert_called_once()


@pytest.mark.asyncio
async def test_runtime_consume_pending_messages_silently() -> None:
    runtime = object.__new__(MaisakaHeartFlowChatting)
    msg1 = SimpleNamespace(message_id="1")
    msg2 = SimpleNamespace(message_id="2")
    runtime.message_cache = [msg1, msg2]
    runtime._last_processed_index = 0
    runtime._cycle_counter = 1
    runtime.log_prefix = "[test]"
    runtime._fixed_message_debouncer = SimpleNamespace(consume=Mock())
    runtime._reasoning_engine = SimpleNamespace(
        _ingest_messages=AsyncMock(),
        _post_process_chat_history_after_cycle=AsyncMock(),
    )

    consumed = await runtime._consume_pending_messages_silently(count=2)
    assert consumed == 2
    assert runtime._last_processed_index == 2
    runtime._fixed_message_debouncer.consume.assert_called_once()
    runtime._reasoning_engine._ingest_messages.assert_awaited_once_with([msg1, msg2])
