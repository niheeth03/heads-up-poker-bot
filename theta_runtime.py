"""
theta_runtime.py - plays a strategy trained by training/train_theta.py.

    policy = ThetaPolicy("models/theta.npz")
    action = policy.act(obs, memory)       # an Action, or None if not covered

Computes Theta's card bucket exactly like training (same equity sample counts,
same features; tests/test_features.py checks the features match numba).
"""

from __future__ import annotations

import numpy as np

from cfr_common import build_eval5, build_tree, equity_np, hole_index, obs_key
from game import Action
from population_runtime import blocker_flag, draw_flag, made_group


def texture_code(board) -> int:
    ranks = [c >> 2 for c in board]
    paired = int(len(set(ranks)) < len(ranks))
    flushy = int(any(sum((c & 3) == s for c in board) >= 3 for s in range(4)))
    return paired + 2 * flushy


class ThetaPolicy:
    MIN_VISITS = 50.0
    PURIFY = 0.05

    def __init__(self, path: str):
        self.ok = False
        try:
            m = np.load(path)
            tree = build_tree()
            if int(m["n_dec"]) != tree["n_dec"]:
                return
            self.key_to_dec = tree["key_to_dec"]
            self.avg = m["avg_strategy"]
            self.visits = m["visits"]
            self.flags = [int(x) for x in m["flags"]]
            self.pre_bucket = m["pre_bucket"]
            self.pre_tercile = m["pre_tercile"]
            self.prev_thr = m["prev_thr"]
            self.thr = [m["thr"][s, :int(m["nthr"][s])] for s in range(4)]
            self.k_post = [int(k) for k in m["k_post"]]
            self.eval5 = build_eval5().astype(np.int64)
            self.rng = np.random.default_rng()
            self.key = "theta_eq_" + path
            self.ok = True
        except Exception:
            self.ok = False

    def _equity(self, obs, memory, street: int) -> float:
        """Equity on `street` (1 flop, 2 turn, 3 river) for this hand, cached."""
        cache = memory.get(self.key)
        if cache is None or cache[0] != obs.hand_index:
            cache = (obs.hand_index, {})
            memory[self.key] = cache
        if street not in cache[1]:
            board = tuple(obs.board[:street + 2])
            cache[1][street] = equity_np(obs.hole_cards, board, self.k_post[street], self.eval5, self.rng)
        return cache[1][street]

    def bucket_from(self, hole, board, eq, prev_level) -> int:
        """Postflop bucket from features (also used by tests)."""
        thr = self.thr[len(board) - 2]
        b = int(np.searchsorted(thr, eq))
        mult = len(thr) + 1
        if self.flags[0]:
            b += mult * made_group(hole, board)
            mult *= 4
        if self.flags[1]:
            b += mult * draw_flag(hole, board)
            mult *= 2
        if self.flags[2]:
            b += mult * blocker_flag(hole, board)
            mult *= 2
        if self.flags[3]:
            b += mult * texture_code(board)
            mult *= 4
        if self.flags[4]:
            b += mult * prev_level
        return b

    def bucket(self, obs, memory) -> int:
        street = int(obs.street)
        idx = hole_index(obs.hole_cards)
        if street == 0:
            return int(self.pre_bucket[idx])
        eq = self._equity(obs, memory, street)
        if street == 1:
            prev = int(self.pre_tercile[idx])
        else:
            prev = int(np.searchsorted(self.prev_thr[street - 1], self._equity(obs, memory, street - 1)))
        return self.bucket_from(tuple(obs.hole_cards), tuple(obs.board), eq, prev)

    def act(self, obs, memory: dict):
        if not self.ok:
            return None
        d = self.key_to_dec.get(obs_key(obs))
        if d is None:
            return None
        b = self.bucket(obs, memory)
        if b >= self.visits.shape[1] or self.visits[d, b] < self.MIN_VISITS:
            return None
        p = np.array(self.avg[d, b], dtype=np.float64)
        legal = np.zeros(4)
        for a in obs.legal_actions:
            legal[int(a)] = 1.0
        p *= legal
        p[p < self.PURIFY] = 0.0
        total = p.sum()
        if total <= 0:
            return None
        r = self.rng.random() * total
        acc = 0.0
        for a in range(4):
            acc += p[a]
            if p[a] > 0 and r < acc:
                return Action(a)
        return None
