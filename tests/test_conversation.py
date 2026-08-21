# Description: Unit tests for the per-conversation state cache (LRU + TTL + sweep throttle).
# Description: Pure in-memory tests; endpoint-level sticky-driver behavior lives in test_proxy.py.

from __future__ import annotations

from pdp_router import _conversation
from pdp_router._conversation import ConversationCache, ConversationState


class _Clock:
    """Controllable stand-in for time.monotonic."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _cache(monkeypatch, **kwargs) -> tuple[ConversationCache, _Clock]:
    clock = _Clock()
    monkeypatch.setattr(_conversation.time, "monotonic", clock)
    return ConversationCache(**kwargs), clock


class TestGetAndPeek:
    def test_get_creates_and_returns_the_same_state(self, monkeypatch) -> None:
        cache, _ = _cache(monkeypatch)
        state = cache.get("abc")
        assert isinstance(state, ConversationState)
        assert cache.get("abc") is state
        assert len(cache) == 1

    def test_peek_never_creates(self, monkeypatch) -> None:
        cache, _ = _cache(monkeypatch)
        assert cache.peek("missing") is None
        assert len(cache) == 0
        cache.get("abc")
        assert cache.peek("abc") is not None

    def test_fresh_state_defaults(self, monkeypatch) -> None:
        cache, _ = _cache(monkeypatch)
        state = cache.get("abc")
        assert state.driver is None
        assert state.spend_usd == 0.0
        assert state.budget_warned is False
        assert state.last_request_id is None
        assert state.turn_count == 0


class TestEviction:
    def test_overflow_evicts_the_oldest(self, monkeypatch) -> None:
        cache, _ = _cache(monkeypatch, max_entries=2)
        cache.get("a")
        cache.get("b")
        cache.get("c")
        assert cache.peek("a") is None
        assert cache.peek("b") is not None
        assert cache.peek("c") is not None

    def test_access_refreshes_lru_order(self, monkeypatch) -> None:
        cache, _ = _cache(monkeypatch, max_entries=2)
        cache.get("a")
        cache.get("b")
        cache.get("a")  # refresh: b is now oldest
        cache.get("c")
        assert cache.peek("a") is not None
        assert cache.peek("b") is None

    def test_max_entries_floor_of_one(self, monkeypatch) -> None:
        cache, _ = _cache(monkeypatch, max_entries=0)
        cache.get("a")
        cache.get("b")
        assert len(cache) == 1


class TestTtl:
    def test_expired_entry_is_swept(self, monkeypatch) -> None:
        cache, clock = _cache(monkeypatch, ttl_s=100.0)
        cache.get("a")
        clock.advance(101.0)
        cache.get("b")  # triggers a sweep
        assert cache.peek("a") is None

    def test_fresh_entry_survives_the_sweep(self, monkeypatch) -> None:
        cache, clock = _cache(monkeypatch, ttl_s=100.0)
        cache.get("a")
        clock.advance(50.0)
        cache.get("b")
        assert cache.peek("a") is not None

    def test_sweep_is_throttled(self, monkeypatch) -> None:
        # Two gets inside the throttle window: the second must not pay for a
        # sweep, so an entry that expired between them is still present.
        cache, clock = _cache(monkeypatch, ttl_s=3.0)
        cache.get("victim")
        clock.advance(1.0)
        cache.get("other")  # sweep runs here (first since construction)
        clock.advance(3.5)  # victim now expired, but only 3.5s since last sweep
        cache.get("third")
        assert cache.peek("victim") is not None  # throttle held the sweep
        clock.advance(2.0)  # 5.5s since last sweep: next get sweeps
        cache.get("fourth")
        assert cache.peek("victim") is None

    def test_expiry_keys_on_last_seen_not_creation(self, monkeypatch) -> None:
        cache, clock = _cache(monkeypatch, ttl_s=100.0)
        cache.get("a")
        clock.advance(90.0)
        cache.get("a")  # touch: last_seen refreshed
        clock.advance(90.0)
        cache.get("b")  # sweep at t=180; a last seen at t=90 -> fresh
        assert cache.peek("a") is not None
