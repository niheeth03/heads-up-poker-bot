"""
cfr_common.py - pieces shared by the CFR trainer (training/train_cfr.py) and
the live bot (agent.py). Only needs numpy at runtime.

1. The betting tree. Because stacks reset every hand and there are only four
   actions with a raise cap, the whole betting game is small. We enumerate it
   exactly with game.Game, merging identical betting states into one node, so
   the tree always agrees with the real rules.

2. Card abstraction ("buckets"). Hands are grouped by equity against a random
   hand: preflop from a precomputed table, postflop from a quick Monte Carlo
   estimate. Training and the live bot use the SAME estimator (same sample
   count), so the bucket a hand lands in has the same meaning in both.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from evaluator import NUM_CARDS, get_table
from game import Action, Deal, Game

# ---------------------------------------------------------------------------
# Betting tree
# ---------------------------------------------------------------------------

NODE_DECISION = 0
NODE_FOLD = 1
NODE_SHOWDOWN = 2


def state_key(street: int, is_button: bool, my_bet: int, opp_bet: int,
              my_stack: int, opp_stack: int, raises: int, min_raise_to: int,
              any_action_this_street: bool) -> tuple:
    """Everything that determines the rest of the betting game, as seen by
    the player to act. Two histories with the same key play identically."""
    return (int(street), bool(is_button), my_bet, opp_bet, my_stack, opp_stack,
            raises, min_raise_to - opp_bet, bool(any_action_this_street))


def obs_key(obs) -> tuple:
    """state_key for a live Observation."""
    street = int(obs.street)
    any_act = any(int(a.street) == street for a in obs.action_history)
    return state_key(street, obs.is_button, obs.my_bet, obs.opp_bet, obs.my_stack,
                     obs.opp_stack, obs.raises_this_street, obs.min_raise_to, any_act)


def _game_key(g: Game) -> tuple:
    seat = g.to_act
    street = int(g.street)
    any_act = any(int(r.street) == street for r in g.history)
    return state_key(street, seat == g.button_seat, g.bets[seat], g.bets[1 - seat],
                     g.stacks[seat], g.stacks[1 - seat], g.raises_this_street,
                     g._min_raise_to(seat), any_act)


def build_tree() -> dict:
    """Enumerate every betting state reachable in one hand.

    Players are labelled by role: 0 = button (small blind), 1 = big blind.
    Returns numpy arrays describing the tree plus {state_key: decision index}.
    """
    dummy = Deal(hole_cards=((0, 1, 2, 3, 4), (5, 6, 7, 8, 9)), board=(10, 11, 12, 13, 14))
    root = Game(dummy, button_seat=0)

    ntype, nplayer, nstreet, ndec = [], [], [], []
    child, contrib = [], []
    key_to_node: dict[tuple, int] = {}
    dec_keys: list[tuple] = []

    def new_node(t, p, s, c0, c1):
        ntype.append(t); nplayer.append(p); nstreet.append(s); ndec.append(-1)
        child.append([-1, -1, -1, -1]); contrib.append([c0, c1])
        return len(ntype) - 1

    stack = []

    def node_for(g: Game) -> int:
        if g.is_terminal:
            res = g.result()
            if res.went_to_showdown:
                return new_node(NODE_SHOWDOWN, -1, int(g.street), g.contributed[0], g.contributed[1])
            loser = 1 - res.winner_seat
            return new_node(NODE_FOLD, loser, int(g.street), g.contributed[0], g.contributed[1])
        k = _game_key(g)
        nid = key_to_node.get(k)
        if nid is None:
            nid = new_node(NODE_DECISION, g.to_act, int(g.street), g.contributed[0], g.contributed[1])
            ndec[nid] = len(dec_keys)
            dec_keys.append(k)
            key_to_node[k] = nid
            stack.append((nid, g))
        return nid

    node_for(root)
    while stack:
        nid, g = stack.pop()
        for a in g.legal_actions():
            g2 = g.clone()
            g2.apply(a)
            child[nid][int(a)] = node_for(g2)

    n_dec = len(dec_keys)
    dec_node = np.zeros(n_dec, dtype=np.int32)
    for nid, d in enumerate(ndec):
        if d >= 0:
            dec_node[d] = nid
    return {
        "ntype": np.array(ntype, dtype=np.int8),
        "nplayer": np.array(nplayer, dtype=np.int8),
        "nstreet": np.array(nstreet, dtype=np.int8),
        "ndec": np.array(ndec, dtype=np.int32),
        "child": np.array(child, dtype=np.int32),
        "contrib": np.array(contrib, dtype=np.int32),
        "dec_node": dec_node,
        "key_to_dec": {k: i for i, k in enumerate(dec_keys)},
        "n_dec": n_dec,
    }


# ---------------------------------------------------------------------------
# Fast vectorised evaluation (colex index -> score)
# ---------------------------------------------------------------------------

# BINOM[k, c] = C(c, k). The colex index of sorted cards a<b<c<d<e is
# C(a,1)+C(b,2)+C(c,3)+C(d,4)+C(e,5), a dense index in [0, C(36,5)).
BINOM = np.zeros((6, NUM_CARDS + 1), dtype=np.int64)
for _k in range(6):
    for _c in range(NUM_CARDS + 1):
        _v = 1
        for _i in range(_k):
            _v = _v * (_c - _i) // (_i + 1)
        BINOM[_k, _c] = _v if _c >= _k else 0
N_FIVE = int(BINOM[5, NUM_CARDS])  # 376,992


def build_eval5() -> np.ndarray:
    """Score of every 5-card hand, indexed by colex index."""
    table = get_table()
    combos = np.array(list(combinations(range(NUM_CARDS), 5)), dtype=np.int64)
    masks = np.zeros(len(combos), dtype=np.int64)
    for j in range(5):
        masks |= np.left_shift(np.int64(1), combos[:, j])
    scores = np.fromiter((table[int(m)] for m in masks), dtype=np.int32, count=len(masks))
    idx = colex5(combos)
    out = np.zeros(N_FIVE, dtype=np.int32)
    out[idx] = scores
    return out


def colex5(sorted_cards: np.ndarray) -> np.ndarray:
    """Colex index of sorted 5-card sets along the last axis."""
    s = sorted_cards
    return (BINOM[1, s[..., 0]] + BINOM[2, s[..., 1]] + BINOM[3, s[..., 2]]
            + BINOM[4, s[..., 3]] + BINOM[5, s[..., 4]])


_PAIR_IDX = np.array(list(combinations(range(5), 2)), dtype=np.int64)    # (10, 2)
_TRIP_IDX = np.array(list(combinations(range(5), 3)), dtype=np.int64)    # (10, 3)


def omaha_scores_np(holes: np.ndarray, boards: np.ndarray, eval5: np.ndarray) -> np.ndarray:
    """Omaha score (2 hole + 3 board) for K hands at once. holes, boards: (K, 5)."""
    k = holes.shape[0]
    hp = holes[:, _PAIR_IDX]                       # (K, 10, 2)
    bt = boards[:, _TRIP_IDX]                      # (K, 10, 3)
    five = np.empty((k, 10, 10, 5), dtype=np.int64)
    five[..., :2] = hp[:, :, None, :]
    five[..., 2:] = bt[:, None, :, :]
    five.sort(axis=-1)
    return eval5[colex5(five)].reshape(k, 100).max(axis=1)


def equity_np(hole, board, n_samples: int, eval5: np.ndarray, rng: np.random.Generator) -> float:
    """Monte Carlo equity vs a uniformly random opponent hand, rolling out the
    rest of the board. Same estimator as training's equity_nb."""
    dead = np.zeros(NUM_CARDS, dtype=bool)
    dead[list(hole)] = True
    dead[list(board)] = True
    deck = np.flatnonzero(~dead)
    need = 5 - len(board)
    picks = np.argsort(rng.random((n_samples, len(deck))), axis=1)[:, :need + 5]
    samp = deck[picks]
    opp = samp[:, :5]
    boards = np.empty((n_samples, 5), dtype=np.int64)
    boards[:, :len(board)] = board
    boards[:, len(board):] = samp[:, 5:]
    holes = np.broadcast_to(np.asarray(hole, dtype=np.int64), (n_samples, 5))
    if need == 0:
        my = omaha_scores_np(holes[:1], boards[:1], eval5)[0]
    else:
        my = omaha_scores_np(holes, boards, eval5)
    op = omaha_scores_np(opp, boards, eval5)
    return float(((my > op) + 0.5 * (my == op)).mean())


def hole_index(hole) -> int:
    """Colex index of a 5-card hole (any order)."""
    s = sorted(hole)
    return int(BINOM[1, s[0]] + BINOM[2, s[1]] + BINOM[3, s[2]] + BINOM[4, s[3]] + BINOM[5, s[4]])


def bucket_of(value: float, thresholds: np.ndarray) -> int:
    """Bucket = number of thresholds strictly below value."""
    return int(np.searchsorted(thresholds, value, side="left"))


ACTIONS = (Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT)
