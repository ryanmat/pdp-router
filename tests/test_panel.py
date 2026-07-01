# Description: Tests for the shared panel composer, chair synthesis, and JSONL writer.
# Description: Covers lineage diversity, anonymization, error paths, and inbox file shape.

from __future__ import annotations

import json
from dataclasses import dataclass
from unittest.mock import patch

import pytest

from pdp_router._lineage import classify_lineage
from pdp_router._panel import (
    ChairSynthResult,
    PanelMemberResponse,
    append_panel_transcript_jsonl,
    append_routing_decisions_jsonl,
    build_chair_prompt,
    compose_panel,
    synthesize_chair,
)


@dataclass
class _FakeCompletion:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    model: str = ""


def _member(model_id: str, text: str = "answer", error: str | None = None) -> PanelMemberResponse:
    return PanelMemberResponse(
        model_id=model_id,
        lineage=classify_lineage(model_id),
        text=text,
        input_tokens=10,
        output_tokens=20,
        estimated_cost_usd=0.0001,
        latency_ms=50.0,
        error=error,
    )


_POOL = [
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "meta/llama-4-scout-17b-16e-instruct-maas",
    "meta/llama-4-maverick-17b-128e-instruct-maas",
    "deepseek-chat",
]

_TRUST = {
    "claude-opus-4-7": 0.78,
    "claude-sonnet-4-6": 0.65,
    "claude-haiku-4-5-20251001": 0.55,
    "gemini-2.5-pro": 0.72,
    "gemini-2.5-flash": 0.58,
    "gemini-2.5-flash-lite": 0.52,
    "meta/llama-4-scout-17b-16e-instruct-maas": 0.48,
    "meta/llama-4-maverick-17b-128e-instruct-maas": 0.50,
    "deepseek-chat": 0.45,
}


def test_compose_picks_lineage_diverse_panel():
    picked = compose_panel(n=3, candidates=_POOL, trust_weights=_TRUST)
    lineages = {classify_lineage(m) for m in picked}
    assert len(picked) == 3
    assert len(lineages) == 3


def test_compose_picks_highest_trust_per_lineage():
    picked = compose_panel(n=3, candidates=_POOL, trust_weights=_TRUST)
    assert "claude-opus-4-7" in picked
    assert "gemini-2.5-pro" in picked
    assert "meta/llama-4-maverick-17b-128e-instruct-maas" in picked


def test_compose_anthropic_only_filter():
    picked = compose_panel(n=3, lineage_filter="anthropic", candidates=_POOL, trust_weights=_TRUST)
    assert all(m.startswith("claude-") for m in picked)
    assert len(picked) == 3


def test_compose_non_anthropic_filter():
    picked = compose_panel(
        n=3, lineage_filter="non_anthropic", candidates=_POOL, trust_weights=_TRUST
    )
    assert all(not m.startswith("claude-") for m in picked)


def test_compose_excludes_specified_models():
    picked = compose_panel(
        n=3,
        exclude_models=["claude-opus-4-7"],
        candidates=_POOL,
        trust_weights=_TRUST,
    )
    assert "claude-opus-4-7" not in picked
    assert any(m.startswith("claude-") for m in picked)


def test_compose_n_zero_returns_empty():
    assert compose_panel(n=0, candidates=_POOL, trust_weights=_TRUST) == []


def test_compose_pool_smaller_than_n_returns_all():
    pool = ["claude-opus-4-7", "gemini-2.5-pro"]
    picked = compose_panel(n=5, candidates=pool, trust_weights=_TRUST)
    assert set(picked) == {"claude-opus-4-7", "gemini-2.5-pro"}


def test_compose_fills_round_2_from_remaining_lineages():
    picked = compose_panel(n=5, candidates=_POOL, trust_weights=_TRUST)
    assert len(picked) == 5
    assert "claude-sonnet-4-6" in picked


def test_compose_no_trust_data_still_picks():
    picked = compose_panel(n=3, candidates=_POOL, trust_weights={})
    assert len(picked) == 3
    assert len({classify_lineage(m) for m in picked}) == 3


def test_compose_unknown_lineage_filter_raises():
    with pytest.raises(ValueError):
        compose_panel(n=3, lineage_filter="weird", candidates=_POOL)


def test_classify_lineage_openai_and_qwen():
    assert classify_lineage("openai/gpt-5.5") == "openai"
    assert classify_lineage("openai/gpt-5.5-20260423") == "openai"
    assert classify_lineage("qwen/qwen3.7-plus") == "qwen"


def test_compose_includes_openai_and_qwen_lineages():
    """The OpenRouter arms compose as distinct lineages, not collapsed to unknown."""
    pool = [
        "claude-opus-4-7",
        "gemini-2.5-pro",
        "openai/gpt-5.5",
        "qwen/qwen3.7-plus",
    ]
    picked = compose_panel(n=4, candidates=pool, trust_weights={})
    assert set(picked) == set(pool)
    assert {classify_lineage(m) for m in picked} == {
        "anthropic",
        "google",
        "openai",
        "qwen",
    }


def test_synthesize_chair_anonymizes_panelist_labels():
    captured: dict[str, str] = {}

    def fake_complete(system: str, user_msg: str, max_tokens: int) -> _FakeCompletion:
        captured["system"] = system
        captured["user"] = user_msg
        return _FakeCompletion(text="synthesized", input_tokens=50, output_tokens=80)

    members = [
        _member("claude-opus-4-7", text="opus answer"),
        _member("gemini-2.5-pro", text="gemini answer"),
        _member("meta/llama-4-scout-17b-16e-instruct-maas", text="llama answer"),
    ]

    result = synthesize_chair(
        user_prompt="why is the sky blue?",
        system="",
        panel_responses=members,
        chair_model_id="claude-sonnet-4-6",
        complete_fn=fake_complete,
    )

    assert result.error is None
    assert result.text == "synthesized"
    assert "PANELIST A:" in captured["user"]
    assert "PANELIST B:" in captured["user"]
    assert "PANELIST C:" in captured["user"]
    for m in members:
        assert m.model_id not in captured["user"]


def test_synthesize_chair_passes_user_system_through():
    captured: dict[str, str] = {}

    def fake_complete(system: str, user_msg: str, max_tokens: int) -> _FakeCompletion:
        captured["system"] = system
        return _FakeCompletion(text="ok")

    synthesize_chair(
        user_prompt="hi",
        system="You are concise.",
        panel_responses=[_member("claude-opus-4-7")],
        chair_model_id="claude-sonnet-4-6",
        complete_fn=fake_complete,
    )

    assert captured["system"].startswith("You are concise.")
    assert "You are the chair of a panel of" in captured["system"]


def test_synthesize_chair_returns_chair_only_tokens():
    """Chair token counts must be the chair-call tokens only, not panel sum."""

    def fake_complete(system: str, user_msg: str, max_tokens: int) -> _FakeCompletion:
        return _FakeCompletion(
            text="synth", input_tokens=33, output_tokens=44, estimated_cost_usd=0.0011
        )

    members = [_member("claude-opus-4-7"), _member("gemini-2.5-pro")]
    result = synthesize_chair(
        user_prompt="hi",
        system="",
        panel_responses=members,
        chair_model_id="claude-sonnet-4-6",
        complete_fn=fake_complete,
    )

    assert result.input_tokens == 33
    assert result.output_tokens == 44
    assert result.estimated_cost_usd == pytest.approx(0.0011)


def test_synthesize_chair_handles_complete_fn_exception():
    def boom(system: str, user_msg: str, max_tokens: int) -> _FakeCompletion:
        raise RuntimeError("upstream 503")

    result = synthesize_chair(
        user_prompt="hi",
        system="",
        panel_responses=[_member("claude-opus-4-7")],
        chair_model_id="claude-sonnet-4-6",
        complete_fn=boom,
    )

    assert isinstance(result, ChairSynthResult)
    assert result.text == ""
    assert result.error is not None
    assert "upstream 503" in result.error
    assert result.input_tokens == 0


def test_synthesize_chair_filters_empty_survivors():
    """All-errored panel returns no_panel_survivors without calling complete_fn."""

    def must_not_call(system: str, user_msg: str, max_tokens: int) -> _FakeCompletion:
        raise AssertionError("complete_fn should not run when no survivors")

    members = [
        _member("claude-opus-4-7", text="", error="timeout"),
        _member("gemini-2.5-pro", text="   ", error=None),
    ]
    result = synthesize_chair(
        user_prompt="hi",
        system="",
        panel_responses=members,
        chair_model_id="claude-sonnet-4-6",
        complete_fn=must_not_call,
    )

    assert result.error == "no_panel_survivors"
    assert result.text == ""


def test_jsonl_writer_round_trip(tmp_path):
    inbox = tmp_path / "inbox"
    rows = [
        {"alert_id": "chat-1", "model_selected": "claude-opus-4-7", "routing_mode": "cascade"},
        {"alert_id": "chat-1", "model_selected": "gemini-2.5-pro", "routing_mode": "panel"},
    ]

    append_routing_decisions_jsonl(inbox_dir=inbox, rows=rows)

    files = list(inbox.glob("proxy-*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == rows[0]
    assert json.loads(lines[1]) == rows[1]


def test_jsonl_writer_creates_dir(tmp_path):
    """Inbox dir is auto-created if missing."""
    inbox = tmp_path / "deep" / "nested" / "inbox"
    assert not inbox.exists()
    append_routing_decisions_jsonl(inbox_dir=inbox, rows=[{"alert_id": "a", "model_selected": "x"}])
    assert inbox.exists()
    assert len(list(inbox.glob("proxy-*.jsonl"))) == 1


def test_jsonl_writer_swallows_oserror(tmp_path):
    """OSError on mkdir or open is logged and swallowed; never raises."""
    inbox = tmp_path / "inbox"
    with patch("pathlib.Path.mkdir", side_effect=OSError("permission denied")):
        # Should NOT raise.
        append_routing_decisions_jsonl(
            inbox_dir=inbox, rows=[{"alert_id": "a", "model_selected": "x"}]
        )
    # Nothing got created.
    assert not inbox.exists() or not list(inbox.glob("proxy-*.jsonl"))


def test_jsonl_writer_no_rows_is_noop(tmp_path):
    inbox = tmp_path / "inbox"
    append_routing_decisions_jsonl(inbox_dir=inbox, rows=[])
    assert not inbox.exists()


# -- append_panel_transcript_jsonl (the chat-quality eval capture sidecar) --
# Reuses the module-level _member helper (signature: model_id, text, error).


def _write_transcript(out, **overrides):
    """append_panel_transcript_jsonl with sensible defaults; override per test."""
    kwargs = {
        "transcript_dir": out,
        "chat_request_id": "r",
        "surface": "nonstream",
        "prompt": "p",
        "messages": [{"role": "user", "content": "p"}],
        "system": "",
        "members": [_member("claude-opus-4-7", text="a")],
        "synthesis_text": "s",
        "synthesis_status": "complete",
        "chair_model": "claude-sonnet-4-6",
        "panel_score": 7,
        "score": 3,
    }
    kwargs.update(overrides)
    append_panel_transcript_jsonl(**kwargs)


def test_panel_transcript_round_trip(tmp_path):
    out = tmp_path / "transcripts"
    members = [
        _member("claude-opus-4-7", text="opus answer"),
        _member("gemini-2.5-pro", text="gemini answer"),
    ]
    # Multi-turn: the full conversation the members saw, plus the last-turn convenience key.
    convo = [
        {"role": "user", "content": "earlier turn"},
        {"role": "assistant", "content": "earlier reply"},
        {"role": "user", "content": "compare X vs Y"},
    ]
    _write_transcript(
        out,
        chat_request_id="req-1",
        prompt="compare X vs Y",
        messages=convo,
        system="be terse",
        members=members,
        synthesis_text="the synthesized answer",
        synthesis_status="complete",
        panel_score=8,
        score=4,
    )

    files = list(out.glob("panel-*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1, "one panel turn == one JSON object, members nested"
    rec = json.loads(lines[0])
    assert rec["chat_request_id"] == "req-1"
    assert rec["surface"] == "nonstream"
    assert rec["panel_score"] == 8
    assert rec["score"] == 4
    assert rec["chair_model"] == "claude-sonnet-4-6"
    assert rec["prompt"] == "compare X vs Y"
    assert rec["messages"] == convo, "full multi-turn history is what the members saw"
    assert rec["system"] == "be terse"
    assert rec["synthesis_text"] == "the synthesized answer"
    assert rec["synthesis_status"] == "complete"
    assert [m["model_id"] for m in rec["members"]] == ["claude-opus-4-7", "gemini-2.5-pro"]
    assert [m["text"] for m in rec["members"]] == ["opus answer", "gemini answer"]
    assert rec["members"][0]["lineage"] == "anthropic"
    assert rec["members"][0]["output_tokens"] == 20
    assert "ts" in rec


def test_panel_transcript_empty_synthesis_records_chair_failure(tmp_path):
    """An empty synthesis_text + chair_empty status is recorded as-is, not dropped."""
    out = tmp_path / "transcripts"
    _write_transcript(
        out,
        members=[_member("claude-opus-4-7", text="only answer")],
        synthesis_text="",
        synthesis_status="chair_empty",
    )
    rec = json.loads(next(out.glob("panel-*.jsonl")).read_text().splitlines()[0])
    assert rec["synthesis_text"] == ""
    assert rec["synthesis_status"] == "chair_empty"
    assert len(rec["members"]) == 1


def test_panel_transcript_creates_dir(tmp_path):
    out = tmp_path / "deep" / "nested" / "transcripts"
    assert not out.exists()
    _write_transcript(out)
    assert out.exists()
    assert len(list(out.glob("panel-*.jsonl"))) == 1


def test_panel_transcript_empty_members_noop(tmp_path):
    """No survivors -> nothing gradeable -> no file."""
    out = tmp_path / "transcripts"
    _write_transcript(out, members=[])
    assert not out.exists()


def test_panel_transcript_swallows_oserror_on_mkdir(tmp_path):
    """OSError on mkdir is logged and swallowed; never raises into the request."""
    out = tmp_path / "transcripts"
    with patch("pathlib.Path.mkdir", side_effect=OSError("permission denied")):
        _write_transcript(out)
    assert not out.exists() or not list(out.glob("panel-*.jsonl"))


def test_panel_transcript_swallows_oserror_on_write(tmp_path):
    """OSError on the open/write branch (mkdir succeeds first) is also swallowed."""
    out = tmp_path / "transcripts"
    with patch("pathlib.Path.open", side_effect=OSError("disk full")):
        _write_transcript(out)
    # mkdir may have created the dir, but no file was written and nothing raised.
    assert not list(out.glob("panel-*.jsonl"))


def test_panel_transcript_swallows_non_serializable(tmp_path):
    """A non-JSON-serializable field raises TypeError inside the try; the broadened
    except must swallow it so the fire-and-forget contract holds in the stream finally."""
    out = tmp_path / "transcripts"
    # object() is not JSON-serializable -> json.dumps raises TypeError, not OSError.
    _write_transcript(out, messages=[{"role": "user", "content": object()}])
    assert not list(out.glob("panel-*.jsonl"))


# -- build_chair_prompt (extracted helper shared by blocking + streaming chair) --


def test_build_chair_prompt_anonymizes_and_composes():
    survivors = [_member("claude-opus-4-7", "alpha"), _member("gemini-2.5-pro", "beta")]
    chair_system, chair_user = build_chair_prompt(
        user_prompt="Which is better?", system="Be terse.", survivors=survivors
    )
    # Anonymized A/B labels carry the survivor text in order.
    assert "PANELIST A:\nalpha" in chair_user
    assert "PANELIST B:\nbeta" in chair_user
    assert "USER PROMPT:\nWhich is better?" in chair_user
    # Model ids never leak to the chair (brand / positional bias mitigation).
    assert "claude-opus-4-7" not in chair_user
    assert "gemini-2.5-pro" not in chair_user
    # System composition: caller system first, then the chair instruction with N=2.
    assert chair_system.startswith("Be terse.")
    assert "chair of a panel of 2" in chair_system


def test_build_chair_prompt_no_caller_system():
    chair_system, _ = build_chair_prompt(
        user_prompt="Q", system="", survivors=[_member("deepseek-chat", "x")]
    )
    # With no caller system, only the chair instruction is present (no leading blank join).
    assert chair_system.startswith("You are the chair")


def test_synthesize_chair_uses_build_chair_prompt():
    """Behavioral parity: synthesize_chair builds the same anonymized prompt and
    passes it to complete_fn."""
    captured: dict = {}

    def _complete_fn(system, user, max_tokens):
        captured["system"] = system
        captured["user"] = user
        return _FakeCompletion(text="synth", input_tokens=3, output_tokens=4)

    result = synthesize_chair(
        user_prompt="Compare",
        system="Sys.",
        panel_responses=[_member("claude-opus-4-7", "a"), _member("deepseek-chat", "b")],
        chair_model_id="claude-sonnet-4-6",
        complete_fn=_complete_fn,
        max_tokens=128,
    )
    assert result.text == "synth"
    assert result.error is None
    expected_system, expected_user = build_chair_prompt(
        user_prompt="Compare",
        system="Sys.",
        survivors=[_member("claude-opus-4-7", "a"), _member("deepseek-chat", "b")],
    )
    assert captured["system"] == expected_system
    assert captured["user"] == expected_user
