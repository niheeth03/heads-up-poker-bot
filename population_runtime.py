"""
population_runtime.py - plays a strategy trained by training/train_population.py.

    policy = PopulationPolicy("models/gamma.npz")
    action = policy.act(obs, memory)      # an Action, or None if not covered

Card features here mirror the numba versions in train_population.py exactly
(tests/test_population.py checks this), so a hand lands in the same bucket
live as it did in training.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from cfr_common import build_eval5, build_tree, equity_np, hole_index, obs_key
from evaluator import get_table
from game import Action

_PAIRS = tuple(combinations(range(5), 2))


def made_group(hole, board) -> int:
    table = get_table()
    best = 0
    for i, j in _PAIRS:
        hm = (1 << hole[i]) | (1 << hole[j])
        for tri in combinations(board, 3):
            s = table[hm | (1 << tri[0]) | (1 << tri[1]) | (1 << tri[2])]
            if s > best:
                best = s
    cat = best >> 20
    if cat <= 1:
        return 0
    if cat <= 4:
        return 1
    if cat == 5:
        return 2
    return 3


def _in_window(r: int, w: int) -> bool:
    if w < 5:
        return w <= r <= w + 4
    return r == 8 or r <= 3


def draw_flag(hole, board) -> int:
    if len(board) >= 5:
        return 0
    for s in range(4):
        if sum((c & 3) == s for c in hole) >= 2 and sum((c & 3) == s for c in board) == 2:
            return 1
    for w in range(6):
        branks = {c >> 2 for c in board if _in_window(c >> 2, w)}
        for i, j in _PAIRS:
            ri, rj = hole[i] >> 2, hole[j] >> 2
            if ri == rj or not _in_window(ri, w) or not _in_window(rj, w):
                continue
            if len(branks - {ri, rj}) >= 2:
                return 1
    return 0


def blocker_flag(hole, board) -> int:
    for s in range(4):
        if sum((c & 3) == s for c in board) >= 2 and (32 + s) in hole:
            return 1
    return 0


class PopulationPolicy:
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
            self.thr = [m["thr"][s, :int(m["nthr"][s])] for s in range(4)]
            self.k_post = [int(k) for k in m["k_post"]]
            self.eval5 = build_eval5().astype(np.int64)
            self.rng = np.random.default_rng()
            self.cache_key = "pop_bucket_" + path
            self.ok = True
        except Exception:
            self.ok = False

    def bucket_from(self, hole, board, eq: float) -> int:
        """Bucket for a postflop street given its equity (shared by tests)."""
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
        return b

    def bucket(self, obs, memory: dict) -> int:
        street = int(obs.street)
        if street == 0:
            return int(self.pre_bucket[hole_index(obs.hole_cards)])
        c = memory.get(self.cache_key)
        if c is not None and c[0] == obs.hand_index and c[1] == street:
            return c[2]
        eq = equity_np(obs.hole_cards, obs.board, self.k_post[street], self.eval5, self.rng)
        b = self.bucket_from(tuple(obs.hole_cards), tuple(obs.board), eq)
        memory[self.cache_key] = (obs.hand_index, street, b)
        return b

    def probs(self, obs, memory: dict):
        """Trained action probabilities (length 4, legal-masked) or None."""
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
        return p if p.sum() > 0 else None

    def act(self, obs, memory: dict):
        p = self.probs(obs, memory)
        if p is None:
            return None
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
