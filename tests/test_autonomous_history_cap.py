"""Autonomous history cap — each wake is an independent observation, so the
autonomous session must not accumulate its own prior answers (few-shot pile-up
that collapsed per-wake thinking 2687→1271). Pure test of the cap helper."""

from lumos_node.chat import _cap_autonomous_history
from lumos_node.llm.lm_studio import ChatMessage


def _hist(turns: int) -> list[ChatMessage]:
    # `turns` user/assistant pairs, newest last.
    msgs: list[ChatMessage] = []
    for i in range(turns):
        msgs.append(ChatMessage(role="user", content=f"u{i}"))
        msgs.append(ChatMessage(role="assistant", content=f"a{i}"))
    return msgs


def test_keeps_only_last_n_turns():
    capped = _cap_autonomous_history(_hist(7), turns=1)
    assert len(capped) == 2                       # one turn = 2 messages
    assert capped[0].content == "u6" and capped[1].content == "a6"  # the newest


def test_zero_turns_is_fully_fresh():
    assert _cap_autonomous_history(_hist(7), turns=0) == []


def test_short_history_unchanged():
    h = _hist(2)
    capped = _cap_autonomous_history(h, turns=5)   # want 5, only have 2
    assert [m.content for m in capped] == [m.content for m in h]


def test_carry_two_turns():
    capped = _cap_autonomous_history(_hist(7), turns=2)
    assert [m.content for m in capped] == ["u5", "a5", "u6", "a6"]


def test_empty_history():
    assert _cap_autonomous_history([], turns=1) == []
    assert _cap_autonomous_history([], turns=0) == []
