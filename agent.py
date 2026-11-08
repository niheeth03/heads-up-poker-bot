"""
agent.py - YOUR poker bot. This one file is a complete web server.

Run it:
    python agent.py                  # listens on port 5008
    PORT=5010 python agent.py        # or pick a port

Play a match against a baseline (three windows):
    python agent.py                                 # your bot on port 5008
    python baseline-bots/baseline_random.py         # opponent on port 5009
    python judge_engine.py                          # judge: both halves (uses those two ports)

Check that the tournament can run it:
    python -m pytest -q              # tests in test_agent.py
    ./tools/check_build.sh           # builds and smoke-tests it like the tournament

TODO:
    1. Assign your agent a unique PARTICIPANT and AGENT_NAME below.
    2. Implement the three functions between "YOUR CODE HERE" and "END OF YOUR CODE".

Everything under "SERVER PLUMBING" talks to the judge. You normally never
need to change it, and if you break it the judge can't talk to your bot.
"""

import os
import time
import traceback

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from evaluator import card_rank, card_suit, cards_to_str, omaha_score, warmup  # handy helpers, see evaluator.py
from game import Action, AgentConfig, HandResult, Observation, Street, default_action

# Build omaha_score's lookup table now, while the server starts, instead of
# during your first decision (where it would eat into your time bank).
# Do any other slow setup (loading a model, ...) here too: the server must
# answer within 15 seconds of starting.
warmup()

# rename as your team/bot's name
PARTICIPANT = "team-alpha"
AGENT_NAME = "alphabot"


# ===== YOUR CODE HERE =====

from collections import Counter
from itertools import combinations
import random

import numpy as np

# ---------------------------------------------------------------------------
# 0. CFR self-play strategy (trained offline by training/train_cfr.py)
# ---------------------------------------------------------------------------
# The model maps (betting state, card bucket) -> probabilities over the four
# actions. If anything about it fails to load, or a state is missing, the bot
# falls back to the hand-written strategy further down.

_HERE = os.path.dirname(os.path.abspath(__file__))

# Precomputed lookup tables (tools/build_startup_tables.py). Loading them takes
# milliseconds; building them from scratch takes ~2-5 s on a slow single CPU.
# If the file is missing, everything below falls back to building them.
_STARTUP: dict | None = None
if not os.environ.get("AGENT_SKIP_STARTUP_TABLES"):
    try:
        _z = np.load(os.path.join(_HERE, "models", "startup_tables.npz"))
        if _z["pct"].shape == (376992,) and _z["eval5"].shape == (376992,):
            _STARTUP = {"pct": _z["pct"], "eval5": _z["eval5"].astype(np.int64)}
    except Exception:
        _STARTUP = None
_CFR_PATH = os.path.join(_HERE, "models", "cfr_strategy.npz")
_CFR: dict | None = None
_NP_RNG = np.random.default_rng()
_MIN_VISITS = 50.0       # trust a strategy entry only if it was trained enough
_PURIFY = 0.05           # drop actions with tiny probability (reduces noise)


def _load_cfr() -> None:
    global _CFR
    try:
        from cfr_common import build_eval5, build_tree
        m = np.load(_CFR_PATH)
        tree = build_tree()
        if int(m["n_dec"]) != tree["n_dec"]:
            return  # rules changed since training: tree no longer matches
        _CFR = {
            "key_to_dec": tree["key_to_dec"],
            "avg": m["avg_strategy"],
            "visits": m["visits"],
            "pre_bucket": m["pre_bucket"],
            "thr": (None, m["thr1"], m["thr2"], m["thr3"]),
            "k_post": [int(k) for k in m["k_post"]],
            "eval5": _STARTUP["eval5"] if _STARTUP is not None else build_eval5().astype(np.int64),
            "pre_eq": m["pre_eq"].astype(np.float64) / 65535.0,
        }
    except Exception:
        _CFR = None


_load_cfr()


def _cfr_bucket(obs: Observation, memory: dict) -> int:
    from cfr_common import equity_np, hole_index
    street = int(obs.street)
    if street == 0:
        return int(_CFR["pre_bucket"][hole_index(obs.hole_cards)])
    cache = memory.get("bucket_cache")
    if cache is not None and cache[0] == obs.hand_index and cache[1] == street:
        return cache[2]
    eq = equity_np(obs.hole_cards, obs.board, _CFR["k_post"][street], _CFR["eval5"], _NP_RNG)
    b = int(np.searchsorted(_CFR["thr"][street], eq))
    memory["bucket_cache"] = (obs.hand_index, street, b)
    return b



def _cfr_action(obs: Observation, memory: dict):
    """Sample an action from the trained strategy, or None if not covered."""
    if _CFR is None:
        return None
    from cfr_common import obs_key
    d = _CFR["key_to_dec"].get(obs_key(obs))
    if d is None:
        return None
    b = _cfr_bucket(obs, memory)
    if _CFR["visits"][d, b] < _MIN_VISITS:
        return None
    probs = np.array(_CFR["avg"][d, b], dtype=np.float64)
    legal = np.zeros(4)
    for a in obs.legal_actions:
        legal[int(a)] = 1.0
    probs *= legal
    probs[probs < _PURIFY] = 0.0
    total = probs.sum()
    if total <= 0:
        return None
    r = _NP_RNG.random() * total
    acc = 0.0
    for a in range(4):
        acc += probs[a]
        if probs[a] > 0 and r < acc:
            return Action(a)
    return None

# ---------------------------------------------------------------------------
# 1. Precomputed Hand Rating & Percentile Table (Built at startup)
# ---------------------------------------------------------------------------

_HAND_PERCENTILES: dict[int, float] = {}


def _compute_hand_raw_score(cards: tuple[int, ...]) -> int:
    """Rate 5 hole cards for Short Deck PLO5 structure."""
    ranks = [card_rank(c) for c in cards]
    suits = [card_suit(c) for c in cards]
    r_counts = Counter(ranks)
    s_counts = Counter(suits)
    score = 0

    # 1. Pairs & Trips (Omaha set potential)
    for r, count in r_counts.items():
        if count == 2:
            score += 16 + r * 3        # AA=40, KK=37 ... 66=16
        elif count == 3:
            score += 10 + r * 2        # Trips in hand is less flexible in Omaha
        elif count >= 4:
            score -= 12                # Quads in hand blocks own outs

    # 2. Suitedness (Omaha rule: exactly 2 used)
    # Having 2 of a suit is ideal. Having 3+ wastes flush outs.
    for s, count in s_counts.items():
        if count == 2:
            highest_r = max(r for r, suit in zip(ranks, suits) if suit == s)
            score += 12 + highest_r * 2
        elif count == 3:
            score += 4
        elif count >= 4:
            score -= 6

    # 3. High card ranks (top 2 matter most)
    sorted_r = sorted(ranks, reverse=True)
    score += sorted_r[0] * 4 + sorted_r[1] * 2

    # 4. Connectivity (Runs of consecutive ranks)
    u_ranks = sorted(set(ranks))
    for i in range(len(u_ranks)):
        for j in range(i + 1, len(u_ranks)):
            diff = u_ranks[j] - u_ranks[i]
            if diff == 1:
                score += 4
            elif diff == 2:
                score += 2

    # Ace wrap: A(8) connects with 6(0), 7(1), 8(2), 9(3)
    if 8 in u_ranks:
        for low in (0, 1, 2, 3):
            if low in u_ranks:
                score += 3

    return score


def _init_preflop_table() -> None:
    """Build percentile lookup for all 376,992 combinations (skipped when the
    precomputed table from models/startup_tables.npz is available)."""
    global _HAND_PERCENTILES
    if _HAND_PERCENTILES or _STARTUP is not None:
        return
    scored = []
    for h in combinations(range(36), 5):
        mask = (1 << h[0]) | (1 << h[1]) | (1 << h[2]) | (1 << h[3]) | (1 << h[4])
        raw = _compute_hand_raw_score(h)
        scored.append((raw, mask))
    scored.sort(key=lambda x: x[0])
    total = len(scored)
    for rank_idx, (_, mask) in enumerate(scored):
        _HAND_PERCENTILES[mask] = rank_idx / total


# Precompute lookup table immediately at server startup
_init_preflop_table()


def _get_hole_mask(cards: tuple[int, ...]) -> int:
    m = 0
    for c in cards:
        m |= (1 << c)
    return m


def _get_preflop_percentile(hole_cards: tuple[int, ...]) -> float:
    if _STARTUP is not None:
        from cfr_common import hole_index
        return float(_STARTUP["pct"][hole_index(hole_cards)])
    mask = _get_hole_mask(hole_cards)
    return _HAND_PERCENTILES.get(mask, 0.50)


# ---------------------------------------------------------------------------
# 2. Advanced Texture, Blockers, & Nut Analysis
# ---------------------------------------------------------------------------

def _analyze_board_and_blockers(hole_cards: tuple[int, ...], board: tuple[int, ...]) -> dict:
    """
    Evaluate Omaha board texture lethality, card removal, and nut blockers.
    In Short Deck (36 cards), holding the Ace of a suit when board has 3 of that
    suit blocks the opponent from holding the Nut Flush by over 60%.
    """
    if not board:
        return {"paired": False, "flush_suit": None, "has_nut_flush_blocker": False, "is_nut_flush": False}

    ranks_b = [card_rank(c) for c in board]
    suits_b = [card_suit(c) for c in board]
    ranks_h = [card_rank(c) for c in hole_cards]
    suits_h = [card_suit(c) for c in hole_cards]

    # Paired board check (Full House possible -> kills flushes and straights)
    r_counts_b = Counter(ranks_b)
    paired_board = any(cnt >= 2 for cnt in r_counts_b.values())

    # Flush texture: 3 or more of same suit on board
    s_counts_b = Counter(suits_b)
    flush_suit = None
    for s, cnt in s_counts_b.items():
        if cnt >= 3:
            flush_suit = s
            break

    has_nut_flush_blocker = False
    is_nut_flush = False

    if flush_suit is not None:
        # Do we hold the Ace of that flush suit? (card 32, 33, 34, 35)
        nut_flush_card = (8 << 2) | flush_suit
        has_nut_flush_blocker = (nut_flush_card in hole_cards)

        # Do we have an actual flush with that suit? (need at least 2 hole cards of that suit)
        hole_suit_cnt = sum(1 for s in suits_h if s == flush_suit)
        if hole_suit_cnt >= 2 and has_nut_flush_blocker:
            is_nut_flush = True

    return {
        "paired": paired_board,
        "flush_suit": flush_suit,
        "has_nut_flush_blocker": has_nut_flush_blocker,
        "is_nut_flush": is_nut_flush,
    }


# ---------------------------------------------------------------------------
# 3. Adaptive Bayesian Opponent Tracking
# ---------------------------------------------------------------------------

def _get_opp_profile(memory: dict) -> dict:
    """Infer opponent tendencies from observed action history."""
    hands = memory.get("hands_played", 0)
    raises = memory.get("opp_raises", 0)
    calls = memory.get("opp_calls", 0)
    folds = memory.get("opp_folds", 0)
    bets_faced = memory.get("opp_bets_faced", 0)

    total_actions = raises + calls
    aggression = (raises / total_actions) if total_actions > 0 else 0.35
    fold_rate = (folds / bets_faced) if bets_faced > 0 else 0.20

    is_maniac = (hands >= 6 and aggression > 0.65)
    is_calling_station = (hands >= 6 and aggression < 0.18 and fold_rate < 0.12)
    is_fold_heavy = (hands >= 6 and fold_rate > 0.45)

    return {
        "hands": hands,
        "aggression": aggression,
        "fold_rate": fold_rate,
        "is_maniac": is_maniac,
        "is_calling_station": is_calling_station,
        "is_fold_heavy": is_fold_heavy,
    }


# ---------------------------------------------------------------------------
# 4. Range-Aware Monte Carlo Simulation
# ---------------------------------------------------------------------------

# Vectorised tables for the fast equity path (built once at startup).
_FAST_EVAL5 = None       # colex index -> 5-card score
_PCT_BY_COLEX = None     # colex index -> preflop percentile (same as _HAND_PERCENTILES)


def _init_fast_equity() -> None:
    global _FAST_EVAL5, _PCT_BY_COLEX
    if _STARTUP is not None:
        _FAST_EVAL5 = _CFR["eval5"] if _CFR is not None else _STARTUP["eval5"]
        _PCT_BY_COLEX = _STARTUP["pct"]
        return
    try:
        from cfr_common import build_eval5, colex5
        combos = np.array(list(combinations(range(36), 5)), dtype=np.int64)
        pct = np.empty(len(combos))
        idx = colex5(combos)
        bits = np.left_shift(np.int64(1), combos).sum(axis=1)
        pct[idx] = [_HAND_PERCENTILES.get(int(m), 0.5) for m in bits]
        _FAST_EVAL5 = _CFR["eval5"] if _CFR is not None else build_eval5().astype(np.int64)
        _PCT_BY_COLEX = pct
    except Exception:
        _FAST_EVAL5 = _PCT_BY_COLEX = None


_init_fast_equity()


def _estimate_equity(
    hole_cards: tuple[int, ...],
    board: tuple[int, ...],
    opp_profile: dict,
    num_samples: int = 30,
) -> float:
    """Estimate win probability against opponent's likely range.

    Fast numpy version of _estimate_equity_slow with the same sampling rule:
    up to 3x num_samples random (opponent hand, runout) draws, opponent hands
    below the 35th percentile rejected when the opponent is aggressive, and
    the first num_samples accepted draws are scored."""
    if _FAST_EVAL5 is None or _PCT_BY_COLEX is None:
        return _estimate_equity_slow(hole_cards, board, opp_profile, num_samples)
    from cfr_common import colex5, omaha_scores_np
    filter_opp = (not opp_profile["is_maniac"]) and (opp_profile["aggression"] > 0.40)
    min_pct = 0.35 if filter_opp else 0.0
    dead = np.zeros(36, dtype=bool)
    dead[list(hole_cards)] = True
    dead[list(board)] = True
    deck = np.flatnonzero(~dead)
    need = 5 - len(board)
    draws = num_samples * 3 if min_pct > 0 else num_samples
    samp = deck[np.argsort(_NP_RNG.random((draws, len(deck))), axis=1)[:, :need + 5]]
    opp = np.sort(samp[:, :5], axis=1)
    if min_pct > 0:
        keep = _PCT_BY_COLEX[colex5(opp)] >= min_pct
        opp, samp = opp[keep][:num_samples], samp[keep][:num_samples]
    k = len(opp)
    if k == 0:
        return 0.50
    boards = np.empty((k, 5), dtype=np.int64)
    boards[:, :len(board)] = board
    boards[:, len(board):] = samp[:, 5:]
    holes = np.broadcast_to(np.asarray(hole_cards, dtype=np.int64), (k, 5))
    my = omaha_scores_np(holes, boards, _FAST_EVAL5)
    op = omaha_scores_np(opp, boards, _FAST_EVAL5)
    return float(((my > op) + 0.5 * (my == op)).mean())


def _estimate_equity_slow(
    hole_cards: tuple[int, ...],
    board: tuple[int, ...],
    opp_profile: dict,
    num_samples: int = 30,
) -> float:
    """Pure-Python reference version (used only if the fast tables failed)."""
    dead = set(hole_cards) | set(board)
    deck = [c for c in range(36) if c not in dead]
    needed_board = 5 - len(board)

    # Opponent range restriction
    filter_opp = (not opp_profile["is_maniac"]) and (opp_profile["aggression"] > 0.40)
    min_opp_percentile = 0.35 if filter_opp else 0.0

    wins = 0.0
    valid_samples = 0
    rng = random.Random()

    attempts = 0
    while valid_samples < num_samples and attempts < num_samples * 3:
        attempts += 1
        sampled = rng.sample(deck, needed_board + 5)
        full_board = list(board) + sampled[:needed_board]
        opp_hole = tuple(sorted(sampled[needed_board:]))

        if min_opp_percentile > 0.0:
            opp_pct = _get_preflop_percentile(opp_hole)
            if opp_pct < min_opp_percentile:
                continue

        valid_samples += 1
        my_score = omaha_score(hole_cards, full_board)
        opp_score = omaha_score(opp_hole, full_board)

        if my_score > opp_score:
            wins += 1.0
        elif my_score == opp_score:
            wins += 0.5

    if valid_samples == 0:
        return 0.50
    return wins / valid_samples


# ---------------------------------------------------------------------------
# 5. Strategic Decision Engines
# ---------------------------------------------------------------------------

def _decide_preflop(obs: Observation, opp_profile: dict) -> Action:
    pct = _get_preflop_percentile(obs.hole_cards)
    to_call = obs.to_call
    pot = obs.pot
    call_fraction = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.0

    # Positional Adjustment: Button acts last postflop (huge advantage)
    # Out of Position (Big Blind), keep pot smaller; In Position (Button), apply pressure.
    if opp_profile["is_maniac"]:
        # Maniac raises everything -> call down light, trap with premiums
        if pct >= 0.88 and Action.RAISE_POT in obs.legal_actions:
            return Action.RAISE_POT
        if pct >= 0.72 and Action.RAISE_HALF in obs.legal_actions:
            return Action.RAISE_HALF
        if pct >= 0.35:
            return Action.CHECK_CALL
        if to_call == 0:
            return Action.CHECK_CALL
        if Action.FOLD in obs.legal_actions:
            return Action.FOLD
        return Action.CHECK_CALL

    # In Position (Button) Strategy:
    # In Heads-Up, button acts last on every postflop street. Exploit positional leverage!
    if obs.is_button:
        if pct >= 0.72 and Action.RAISE_POT in obs.legal_actions:
            return Action.RAISE_POT
        if pct >= 0.48 and Action.RAISE_HALF in obs.legal_actions:
            return Action.RAISE_HALF
        # It only costs 1 chip to complete the SB on button: defend widely
        if pct >= 0.15 or to_call <= obs.big_blind:
            return Action.CHECK_CALL
        if Action.FOLD in obs.legal_actions:
            return Action.FOLD
        return Action.CHECK_CALL

    # Out of Position (Big Blind) Strategy: Pot Control
    # Re-raise only premium top-tier hands out of position
    if pct >= 0.88 and Action.RAISE_POT in obs.legal_actions and obs.raises_this_street < 2:
        return Action.RAISE_POT
    if pct >= 0.32:
        if to_call <= obs.big_blind * 2 or call_fraction <= 0.32:
            return Action.CHECK_CALL
        if Action.FOLD in obs.legal_actions:
            return Action.FOLD
        return Action.CHECK_CALL

    if to_call == 0:
        return Action.CHECK_CALL
    if Action.FOLD in obs.legal_actions:
        return Action.FOLD
    return Action.CHECK_CALL


def _decide_postflop(obs: Observation, opp_profile: dict) -> Action:
    to_call = obs.to_call
    pot = obs.pot
    call_fraction = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.0

    texture = _analyze_board_and_blockers(obs.hole_cards, obs.board)
    n_samples = 25 if obs.street != Street.RIVER else 35
    equity = _estimate_equity(obs.hole_cards, obs.board, opp_profile, num_samples=n_samples)

    # Cooler Avoidance: Discount non-nut flushes on flush boards or straights on paired boards
    if texture["flush_suit"] is not None and not texture["is_nut_flush"]:
        # If board has 3 of a suit and we don't have nut flush, discount equity when facing aggression
        if to_call > obs.big_blind * 4:
            equity *= 0.80

    if texture["paired"] and equity < 0.85:
        # On paired boards, full houses beat flushes & straights
        if to_call > obs.big_blind * 4:
            equity *= 0.85

    ev_call = equity * (pot + to_call) - to_call

    # Weapon 4: Speed-Trap Logic Against Aggressive Opponents
    # On Flop/Turn with monster hands (equity > 0.85), check/call to let them barrel into us
    if opp_profile["is_maniac"] or opp_profile["aggression"] > 0.60:
        if obs.street in (Street.FLOP, Street.TURN) and equity >= 0.85:
            # Trap by calling/checking instead of raising immediately
            return Action.CHECK_CALL
        if equity >= 0.65 and Action.RAISE_POT in obs.legal_actions:
            return Action.RAISE_POT
        if equity >= 0.35 or ev_call > 0 or to_call <= obs.big_blind:
            return Action.CHECK_CALL
        if Action.FOLD in obs.legal_actions:
            return Action.FOLD
        return Action.CHECK_CALL

    # Calling Station: Extract maximum value, zero bluffs
    if opp_profile["is_calling_station"]:
        if equity >= 0.72 and Action.RAISE_POT in obs.legal_actions:
            return Action.RAISE_POT
        if equity >= 0.58 and Action.RAISE_HALF in obs.legal_actions:
            return Action.RAISE_HALF
        if equity >= 0.35 or to_call == 0:
            return Action.CHECK_CALL
        if Action.FOLD in obs.legal_actions:
            return Action.FOLD
        return Action.CHECK_CALL

    # Weapon 2: Omaha Nut Flush Blocker Bluff
    # If board has 3 of a suit, we hold the Ace of that suit (blocking opponent from the nut flush),
    # to_call == 0, and opponent folds often: fire a calculated blocker raise!
    if (
        texture["flush_suit"] is not None
        and texture["has_nut_flush_blocker"]
        and not texture["is_nut_flush"]
        and to_call == 0
        and obs.raises_this_street == 0
        and (opp_profile["is_fold_heavy"] or obs.is_button)
    ):
        if Action.RAISE_HALF in obs.legal_actions:
            return Action.RAISE_HALF

    # Value Betting
    if equity >= 0.78 and Action.RAISE_POT in obs.legal_actions:
        return Action.RAISE_POT
    if equity >= 0.60 and Action.RAISE_HALF in obs.legal_actions and obs.raises_this_street < 2:
        return Action.RAISE_HALF

    # Positional Floating (In Position on Button)
    if obs.is_button and to_call <= obs.big_blind * 2 and equity >= 0.28:
        return Action.CHECK_CALL

    # Standard Calling
    if to_call == 0:
        return Action.CHECK_CALL
    if ev_call > 0.0 or equity >= (call_fraction * 0.95):
        return Action.CHECK_CALL

    if Action.FOLD in obs.legal_actions:
        return Action.FOLD
    return Action.CHECK_CALL


# ---------------------------------------------------------------------------
# The Three Tournament Functions
# ---------------------------------------------------------------------------

def new_half(config: AgentConfig, memory: dict) -> None:
    """Initialize half-match memory and opponent tracking."""
    memory["hands_played"] = 0
    memory["opp_raises"] = 0
    memory["opp_calls"] = 0
    memory["opp_folds"] = 0
    memory["opp_bets_faced"] = 0


# "hybrid": CFR strategy, switching to the exploitative heuristic once the
# opponent is clearly a maniac or a calling station, plus the pressure/induce
# adjustments below. "hybrid_plain" / "hybrid_pressure" / "hybrid_induce":
# hybrid with none / only one of those adjustments (for A/B tests).
# "cfr": always CFR (heuristic only where CFR has no answer). "heuristic": old bot.
BOT_MODE = os.environ.get("BOT_MODE", "hybrid")
_EXPLOIT_MIN_HANDS = 30
_PRESSURE_ON = BOT_MODE in ("hybrid", "hybrid_pressure")
# Induce mode is off by default: in A/B tests it did not beat plain play
# against heavy bluffers (avg about -0.2 bb/deal vs bot_beta over 3 runs).
_INDUCE_ON = BOT_MODE == "hybrid_induce"

# ---------------------------------------------------------------------------
# Opponent-adaptive adjustments on top of the CFR strategy
# ---------------------------------------------------------------------------
# Both only switch on when the opponent is CONFIDENTLY (99%, one-sided) outside
# the range our balanced CFR bots themselves show, so they never fire against
# a sound opponent. Calibration over 1000-hand halves:
#   fold rate when we bet postflop:   balanced CFR 0.46-0.51 | nit, fit_or_fold 0.56-0.58
#   bet rate when we check to them:   balanced CFR 0.58-0.67 | maniac, barreler,
#                                     always-raise, bot_beta 0.86-1.00
#   fold rate to our preflop raises:  balanced CFR 0.23-0.32 | nit, rock 0.56-0.62
_ADJ_MIN_SAMPLES = 40
_ADJ_Z = 2.33
_FOLD_BALANCED_MAX = 0.52    # pressure mode: they fold to bets more than this
_STAB_BALANCED_MAX = 0.70    # induce mode: they bet when checked to more than this
_PRE_FOLD_BALANCED_MAX = 0.36  # preflop pressure: they fold to our preflop raises more than this
_RAISES = (Action.RAISE_HALF, Action.RAISE_POT)


def _confidently_above(hits: int, n: int, ceiling: float) -> bool:
    if n < _ADJ_MIN_SAMPLES:
        return False
    r = hits / n
    return r - _ADJ_Z * (r * (1.0 - r) / n) ** 0.5 > ceiling


def _track_adjustment_stats(obs: Observation, start: int, memory: dict) -> None:
    """From the new part of the action history: how the opponent answered our
    postflop checks (bet = stab), our postflop bets and our preflop raises
    (non-fold replies; folds are counted in on_hand_end)."""
    hist = obs.action_history
    for i in range(max(start, 1), len(hist)):
        a, p = hist[i], hist[i - 1]
        if a.is_me or not p.is_me or a.street != p.street:
            continue
        if a.street == Street.PREFLOP:
            if p.action in _RAISES:
                memory["opp_pre_replies"] = memory.get("opp_pre_replies", 0) + 1
            continue
        if p.action == Action.CHECK_CALL and p.amount == 0:
            memory["opp_checked_to"] = memory.get("opp_checked_to", 0) + 1
            if a.action in _RAISES:
                memory["opp_stabs"] = memory.get("opp_stabs", 0) + 1
            if a.street == Street.FLOP:
                memory["opp_checked_to_flop"] = memory.get("opp_checked_to_flop", 0) + 1
                if a.action in _RAISES:
                    memory["opp_stabs_flop"] = memory.get("opp_stabs_flop", 0) + 1
        elif p.action in _RAISES:
            memory["opp_bet_replies"] = memory.get("opp_bet_replies", 0) + 1
            k = f"opp_bet_replies_{int(a.street)}"
            memory[k] = memory.get(k, 0) + 1
    for a in hist[start:]:   # did the opponent bet/raise on the river this hand?
        if not a.is_me and a.street == Street.RIVER and a.action in _RAISES:
            memory["opp_river_bet_hand"] = obs.hand_index


def _is_over_folder(memory: dict) -> bool:
    return _confidently_above(memory.get("opp_bet_folds", 0), memory.get("opp_bet_replies", 0),
                              _FOLD_BALANCED_MAX)


def _is_pre_folder(memory: dict) -> bool:
    return _confidently_above(memory.get("opp_pre_folds", 0), memory.get("opp_pre_replies", 0),
                              _PRE_FOLD_BALANCED_MAX)


def _is_stabber(memory: dict) -> bool:
    return _confidently_above(memory.get("opp_stabs", 0), memory.get("opp_checked_to", 0),
                              _STAB_BALANCED_MAX)


# Defend mode: against an opponent who bets the flop far more often than any
# balanced bot after we check (a "stealer"), fold less. Measured (bets on the
# flop after we check): balanced CFR bots 55-64%, Team-quad-j 89%.
_DEFEND_ON = BOT_MODE == "hybrid_defend"    # off by default until A/B tested
_FLOP_STAB_BALANCED_MAX = 0.75
_DEFEND_PRE_MARGIN = 0.08                   # preflop: equity must beat the price by this


def _is_flop_stealer(memory: dict) -> bool:
    return _confidently_above(memory.get("opp_stabs_flop", 0), memory.get("opp_checked_to_flop", 0),
                              _FLOP_STAB_BALANCED_MAX)


def _defend(obs: Observation, memory: dict, action: Action) -> Action:
    """Against a confirmed stealer, call instead of folding when the price is right."""
    if (not _DEFEND_ON or action != Action.FOLD or obs.to_call <= 0
            or obs.street == Street.RIVER or not _is_flop_stealer(memory)
            or Action.CHECK_CALL not in obs.legal_actions or _CFR is None):
        return action
    price = obs.to_call / (obs.pot + obs.to_call)
    if obs.street == Street.PREFLOP:
        if obs.raises_this_street > 2:
            return action
        from cfr_common import hole_index
        eq = float(_CFR["pre_eq"][hole_index(obs.hole_cards)])
        return Action.CHECK_CALL if eq >= price + _DEFEND_PRE_MARGIN else action
    from cfr_common import equity_np
    eq = equity_np(obs.hole_cards, obs.board, 64, _CFR["eval5"], _NP_RNG)
    return Action.CHECK_CALL if eq >= price else action


# ---------------------------------------------------------------------------
# Street-specific exploits (from hand logs vs the real field, 2026-11-08).
# Measured, balanced reference (our bot / byte_me) vs leaky teams:
#   fold to our RIVER bet:        0.50-0.54  | malki .89 Armaan .83 quad-j .79 ujjwal .73 harish .67
#   fold to our FLOP bet:         0.41-0.44  | EmptyDeck .12 PayneAM .16 AdityaVSM .24  (sticky callers)
#   their river bets that are weak: 0.16-0.27 | AdityaVSM .00 EmptyDeck .01 Armaan .04 (honest bettors)
# Each fires only when the opponent is CONFIDENTLY (99%) outside the balanced range.
# ---------------------------------------------------------------------------
# ON: passed live A/B 2026-11-08 (+0.19 bb/deal vs 10 competitors, +0.31 vs 25 practice bots,
# 0 losses, neutral vs the strongest teams; never fires vs byte_me).
_EXPLOIT2_ON = BOT_MODE in ("hybrid", "hybrid_x")
_STREET_FOLD_MAX = 0.62                   # turn/river fold-to-bet above this = over-folder
_STICKY_FLOP_FOLD_MIN = 0.30              # flop fold-to-bet below this = sticky caller
_HONEST_RIVER_WEAK_MIN = 0.10             # share of weak river bets below this = honest bettor


def _confidently_below(hits: int, n: int, floor: float) -> bool:
    if n < _ADJ_MIN_SAMPLES:
        return False
    r = (hits + 1) / (n + 2)              # +1/+2 so that 0 of n is not treated as certain
    return r + _ADJ_Z * (r * (1.0 - r) / n) ** 0.5 < floor


def _street_over_folder(memory: dict, street: int) -> bool:
    return _confidently_above(memory.get(f"opp_bet_folds_{street}", 0),
                              memory.get(f"opp_bet_replies_{street}", 0), _STREET_FOLD_MAX)


def _is_sticky_caller(memory: dict) -> bool:
    return _confidently_below(memory.get("opp_bet_folds_1", 0), memory.get("opp_bet_replies_1", 0),
                              _STICKY_FLOP_FOLD_MIN)


def _is_honest_river_bettor(memory: dict) -> bool:
    return _confidently_below(memory.get("opp_rbet_weak", 0), memory.get("opp_rbet_sd", 0),
                              _HONEST_RIVER_WEAK_MIN)


def _street_equity(obs: Observation, memory: dict) -> float:
    c = memory.get("x_eq")
    if c is not None and c[0] == obs.hand_index and c[1] == int(obs.street):
        return c[2]
    from cfr_common import equity_np
    eq = equity_np(obs.hole_cards, obs.board, 128, _CFR["eval5"], _NP_RNG)
    memory["x_eq"] = (obs.hand_index, int(obs.street), eq)
    return eq


def _street_exploit(obs: Observation, memory: dict, action: Action) -> Action:
    if not _EXPLOIT2_ON or _CFR is None or obs.street not in (Street.TURN, Street.RIVER):
        return action
    river = obs.street == Street.RIVER
    if obs.to_call == 0:
        can_bet = Action.RAISE_HALF in obs.legal_actions
        # 1. pressure: bet instead of checking against a turn/river over-folder
        if action == Action.CHECK_CALL and can_bet and _street_over_folder(memory, int(obs.street)):
            return Action.RAISE_HALF
        # 2. sticky caller: thin value bets, no river bluffs
        if river and _is_sticky_caller(memory):
            eq = _street_equity(obs, memory)
            if action == Action.CHECK_CALL and can_bet and eq >= 0.62:
                return Action.RAISE_HALF
            if action in _RAISES and eq < 0.35:
                return Action.CHECK_CALL
        return action
    # 3. honest river bettor: fold marginal hands to their river bet
    if river and action == Action.CHECK_CALL and Action.FOLD in obs.legal_actions and _is_honest_river_bettor(memory):
        if _street_equity(obs, memory) < 0.70:
            return Action.FOLD
    return action


def _adjust(obs: Observation, memory: dict, action: Action) -> Action:
    # Preflop pressure: against an opponent who folds too often to our raises,
    # raise instead of limping/checking/folding when it costs at most a blind.
    if obs.street == Street.PREFLOP:
        if (_PRESSURE_ON and action in (Action.CHECK_CALL, Action.FOLD)
                and obs.to_call <= obs.big_blind and Action.RAISE_HALF in obs.legal_actions
                and _is_pre_folder(memory)):
            return Action.RAISE_HALF
        return action
    if obs.to_call > 0:
        return action
    # Pressure: against an over-folder, bet (half pot) instead of checking.
    if (_PRESSURE_ON and action == Action.CHECK_CALL and Action.RAISE_HALF in obs.legal_actions
            and _is_over_folder(memory)):
        return Action.RAISE_HALF
    # Induce: against a heavy stabber, check first-in instead of betting and
    # let them bluff into us (the CFR strategy then check-raises / calls).
    first_to_act = not any(a.street == obs.street for a in obs.action_history)
    if _INDUCE_ON and action in _RAISES and first_to_act and _is_stabber(memory):
        return Action.CHECK_CALL
    return action


def _use_cfr(memory: dict) -> bool:
    if BOT_MODE == "heuristic":
        return False
    if BOT_MODE == "cfr":
        return True
    prof = _get_opp_profile(memory)
    if prof["hands"] < _EXPLOIT_MIN_HANDS:
        return True
    return not (prof["is_maniac"] or prof["is_calling_station"])


def _choose(obs: Observation, memory: dict) -> Action:
    # Update opponent action counts, counting each action only once
    # (action_history repeats the whole hand on every decision).
    seen_hand, seen_n = memory.get("seen", (-1, 0))
    start = seen_n if seen_hand == obs.hand_index else 0
    for act in obs.action_history[start:]:
        if not act.is_me:
            if act.action in (Action.RAISE_HALF, Action.RAISE_POT):
                memory["opp_raises"] = memory.get("opp_raises", 0) + 1
            elif act.action == Action.CHECK_CALL:
                memory["opp_calls"] = memory.get("opp_calls", 0) + 1
    _track_adjustment_stats(obs, start, memory)
    memory["seen"] = (obs.hand_index, len(obs.action_history))

    if _use_cfr(memory):
        try:
            action = _cfr_action(obs, memory)
            if action is not None:
                action = _adjust(obs, memory, action)
                action = _defend(obs, memory, action)
                action = _street_exploit(obs, memory, action)
        except Exception:
            action = None
        if action is not None and action in obs.legal_actions:
            return action

    opp_profile = _get_opp_profile(memory)
    if obs.street == Street.PREFLOP:
        return _decide_preflop(obs, opp_profile)
    return _decide_postflop(obs, opp_profile)


def choose_action(obs: Observation, memory: dict) -> Action:
    """CFR strategy with heuristic fallback; never raises, always legal."""
    try:
        action = _choose(obs, memory)
        if action not in obs.legal_actions:
            action = default_action(obs.to_call)
    except Exception:
        action = default_action(obs.to_call)
    if action in (Action.RAISE_HALF, Action.RAISE_POT):
        memory["raised_hand"] = obs.hand_index
    # Remember our latest action: if it was a postflop bet and the hand then
    # ends without showdown in our favour, the opponent folded to that bet.
    kind = None
    if action in _RAISES:
        kind = "pre" if obs.street == Street.PREFLOP else "post"
    memory["last_act"] = (obs.hand_index, kind, int(obs.street))
    return action


def on_hand_end(result: HandResult, memory: dict) -> None:
    """Record how often the opponent folds when we bet or raise."""
    try:
        memory["hands_played"] = memory.get("hands_played", 0) + 1
        if memory.get("raised_hand") == result.hand_index:
            memory["opp_bets_faced"] = memory.get("opp_bets_faced", 0) + 1
            if not result.went_to_showdown and result.winner == "me":
                memory["opp_folds"] = memory.get("opp_folds", 0) + 1
        last = memory.get("last_act")
        if (last and last[0] == result.hand_index and last[1]
                and not result.went_to_showdown and result.winner == "me"):
            key = "opp_pre" if last[1] == "pre" else "opp_bet"
            memory[key + "_replies"] = memory.get(key + "_replies", 0) + 1
            memory[key + "_folds"] = memory.get(key + "_folds", 0) + 1
            if last[1] == "post":   # per-street folds (1 flop, 2 turn, 3 river)
                for k in (f"opp_bet_replies_{last[2]}", f"opp_bet_folds_{last[2]}"):
                    memory[k] = memory.get(k, 0) + 1
        # Our last action was a postflop bet and the hand reached showdown: the
        # opponent CALLED it. Count that reply here, because we never get another
        # decision in which to see the call (otherwise river calls are invisible
        # and river fold rates look ~100%).
        if last and last[0] == result.hand_index and last[1] == "post" and result.went_to_showdown:
            k = f"opp_bet_replies_{last[2]}"
            memory[k] = memory.get(k, 0) + 1
        # How strong were the opponent's river bets that reached showdown?
        if (_EXPLOIT2_ON and _CFR is not None and result.went_to_showdown and result.opp_hole is not None
                and len(result.board) == 5 and memory.get("opp_river_bet_hand") == result.hand_index):
            from cfr_common import equity_np
            eq = equity_np(result.opp_hole, result.board, 128, _CFR["eval5"], _NP_RNG)
            memory["opp_rbet_sd"] = memory.get("opp_rbet_sd", 0) + 1
            memory["opp_rbet_weak"] = memory.get("opp_rbet_weak", 0) + (eq < 0.5)
    except Exception:
        pass


# ----- Theta strategy (rich card features, hardened training): replaces the CFR layer above.
from theta_runtime import ThetaPolicy  # noqa: E402

_THETA = ThetaPolicy(os.path.join(_HERE, "models", "theta.npz"))


_THETA_TIME_LIMIT = 5.0   # seconds of Theta thinking per half before falling back (bank is 8 s)
_hardened_cfr_action = _cfr_action  # the CFR layer above, using models/cfr_strategy.npz (Hardened)


def _cfr_action(obs: Observation, memory: dict):  # noqa: F811
    """Theta's strategy, with a time-bank safety net: if Theta's own thinking in
    this half passes _THETA_TIME_LIMIT, switch to the cheaper Hardened strategy."""
    if _THETA.ok and memory.get("theta_seconds", 0.0) < _THETA_TIME_LIMIT:
        t0 = time.perf_counter()
        try:
            return _THETA.act(obs, memory)
        finally:
            memory["theta_seconds"] = memory.get("theta_seconds", 0.0) + time.perf_counter() - t0
    return _hardened_cfr_action(obs, memory)

# ===== END OF YOUR CODE =====


# ===== SERVER PLUMBING (you normally don't need to edit below) =====
# The judge calls these routes:
#   GET  /           -> who you are (also used as the health check)
#   POST /new-half   -> start a fresh half: memory is wiped
#   POST /act        -> the judge sends any finished hands plus an Observation;
#                       on_hand_end() runs for each hand, then you answer an action
#   POST /hand-end   -> the judge sends the last finished hands of a half
#
# Why "async def"? The routes run on one thread and call your functions
# directly, so your functions never run twice at once. Your own functions
# stay normal "def" (don't make them async).
#
# Keep your per-half state in `memory`. In the tournament every half runs in
# a fresh server process, so everything starts over. When you test locally,
# though, one process can play both halves and only `memory` is wiped, NOT
# other global variables. Using `memory` makes your bot behave the same way
# in both places.

app = FastAPI()
MEMORY: dict = {}                                   # your memory for this half
STATUS = {"started": False}                         # has /new-half been called?
TOKEN = os.environ.get("AGENT_TOKEN") or None       # set by the tournament


async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
    """Check the token (and that the half has started) BEFORE reading the body.

    Checking first means a request without the right token costs your bot
    almost nothing, so nobody else can slow it down by sending junk.
    Returns (body, None) when the request is fine, or ({}, error response).
    """
    if TOKEN and request.headers.get("x-agent-token") != TOKEN:
        return {}, JSONResponse({"error": "bad or missing X-Agent-Token"}, status_code=401)
    if not STATUS["started"] and path != "/new-half":
        return {}, JSONResponse({"error": "call /new-half first"}, status_code=409)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return {}, JSONResponse({"error": "the request body must be a JSON object"}, status_code=400)
    return body, None


def _error_info(exc: Exception) -> dict:
    """Describe an exception from your code (call this inside `except`)."""
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _crash(exc: Exception, hand_end_errors: list | None = None) -> JSONResponse:
    """Report an exception from your code to the judge (with traceback)."""
    return JSONResponse({**_error_info(exc), "hand_end_errors": hand_end_errors or []},
                        status_code=500)


def _deliver_results(raw_results) -> list:
    """Call on_hand_end() for each finished hand the judge sent.
    Returns a list of error reports (one per hand that crashed)."""
    errors = []
    for raw in raw_results or []:
        try:
            on_hand_end(HandResult.from_dict(raw), MEMORY)
        except Exception as exc:
            errors.append(_error_info(exc))
    return errors


@app.get("/")
async def info():
    """Health check: tells the judge who this bot is."""
    return {"participant": PARTICIPANT, "agent_name": AGENT_NAME}


@app.post("/new-half")
async def route_new_half(request: Request):
    """Wipe memory and call your new_half(). The body is {"config": {...}}."""
    body, refused = await _read_request(request, "/new-half")
    if refused:
        return refused
    MEMORY.clear()
    STATUS["started"] = True
    try:
        new_half(AgentConfig.from_dict(body.get("config", {})), MEMORY)
    except Exception as exc:
        return _crash(exc)
    return {"ok": True}


@app.post("/act")
async def route_act(request: Request):
    """Deliver finished hands, then turn the JSON into an Observation, call
    choose_action(), and send the answer back."""
    body, refused = await _read_request(request, "/act")
    if refused:
        return refused
    hand_end_errors = _deliver_results(body.get("hand_results"))
    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return _crash(exc, hand_end_errors)
    # Send the action's name; anything strange is passed on for the judge to reject.
    if isinstance(action, Action):
        answer = action.name
    elif action is None or isinstance(action, (str, int, float, bool)):
        answer = action
    else:
        answer = repr(action)
    return {"action": answer, "hand_end_errors": hand_end_errors}


@app.post("/hand-end")
async def route_hand_end(request: Request):
    """Call on_hand_end() for the last finished hands of a half."""
    body, refused = await _read_request(request, "/hand-end")
    if refused:
        return refused
    return {"ok": True, "hand_end_errors": _deliver_results(body.get("results"))}


if __name__ == "__main__":
    # The tournament sets PORT as an environment variable, so each bot is compliant to our evaluation platform
    # Locally it defaults to 5008 for your own testing convenience
    port = int(os.environ.get("PORT", "5008"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
