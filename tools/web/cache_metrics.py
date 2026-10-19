"""Completed-turn cache accounting; engine done counters are the sole input."""
import threading


class CacheMetrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.prompt_tokens = self.reused_tokens = self.turns = 0
        self.last = None

    def record(self, done):
        prompt = int(done.get("prompt_tokens", 0))
        reused = int(done.get("reused_tokens", 0))
        with self.lock:
            self.prompt_tokens += prompt
            self.reused_tokens += reused
            self.turns += 1
            self.last = {"prompt_tokens": prompt, "reused_tokens": reused,
                         "prefill_tokens": done.get("prefill_tokens", 0),
                         "prefix_hit_rate": reused / prompt if prompt else 0.0,
                         "expert_decode_hit_rate": done.get("decode_hit_rate", 0.0)}
            return self._snapshot()

    def _snapshot(self):
        return {"last_turn": dict(self.last) if self.last is not None else None,
                "cumulative": {"turns": self.turns, "prompt_tokens": self.prompt_tokens,
                               "reused_tokens": self.reused_tokens,
                               "prefix_hit_rate": self.reused_tokens / self.prompt_tokens if self.prompt_tokens else 0.0}}

    def snapshot(self):
        with self.lock:
            return self._snapshot()
