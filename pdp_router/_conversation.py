# Description: Per-conversation in-memory state for the proxy: sticky tool driver,
# Description: spend accounting, and previous-turn lineage for implicit feedback.

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field

# Sweep at most this often, mirroring the trust/bandit caches' poll-throttle
# discipline: per-request overhead stays O(1) amortized regardless of rate.
_SWEEP_INTERVAL_S = 5.0


@dataclass
class ConversationState:
    """Mutable state for one conversation (keyed by first-user-turn digest).

    driver: the tool driver that served this conversation; the sticky pin
      reuses it on later turns so a multi-turn agent transcript keeps one
      provider prompt cache warm instead of paying a cold read per switch.
    spend_usd: accumulated estimated cost across turns. Soft accounting: it
      resets on service restart and a streaming turn whose provider reported
      no usage adds nothing.
    budget_warned: the one-shot budget-warning latch for the spend cap.
    last_request_id / last_model: the previous routed turn, so a next-turn
      implicit-feedback row can name the turn it grades.
    last_user_text_digest: digest of the previous latest user text (retry
      detection without retaining the text itself).
    memory_block: the memory block this conversation was pinned to (the date
      line alone when nothing surfaced); None until resolved. Mirrors the
      persisted pin so later turns skip the store read.
    """

    driver: str | None = None
    spend_usd: float = 0.0
    budget_warned: bool = False
    last_request_id: str | None = None
    last_user_text_digest: str | None = None
    last_model: str | None = None
    memory_block: str | None = None
    turn_count: int = 0
    last_seen: float = field(default_factory=time.monotonic)


class ConversationCache:
    """Bounded, TTL'd map of conversation digest -> ConversationState.

    LRU via OrderedDict: get() re-appends the entry, inserting past
    max_entries evicts the oldest. Because every access re-appends, ordering
    equals recency, so the expiry sweep can stop at the first fresh entry.
    Single-event-loop access only -- no method awaits, so no lock is needed.
    """

    def __init__(self, *, max_entries: int = 512, ttl_s: float = 7200.0) -> None:
        self._max_entries = max(1, max_entries)
        self._ttl_s = ttl_s
        self._entries: OrderedDict[str, ConversationState] = OrderedDict()
        self._last_sweep = float("-inf")

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: str) -> ConversationState:
        """Return the state for key, creating it if absent.

        Refreshes LRU position and last_seen; runs the throttled sweep.
        """
        self._sweep()
        state = self._entries.get(key)
        if state is None:
            state = ConversationState()
            self._entries[key] = state
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
        else:
            self._entries.move_to_end(key)
        state.last_seen = time.monotonic()
        return state

    def peek(self, key: str) -> ConversationState | None:
        """Return the state without creating, reordering, or sweeping."""
        return self._entries.get(key)

    def _sweep(self) -> None:
        now = time.monotonic()
        if now - self._last_sweep < _SWEEP_INTERVAL_S:
            return
        self._last_sweep = now
        cutoff = now - self._ttl_s
        for key in list(self._entries.keys()):
            if self._entries[key].last_seen >= cutoff:
                break
            del self._entries[key]
