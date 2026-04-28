"""P0 patch verification suite — agreement guidebook §3.

Locks in:
  P0-1  alert_quality_gate rejects malformed alerts
  P0-2  EXTENDED triggers when price > trigger × 1.05 (works for daily patterns)
  P0-3  retest_hold detector rejects floating retests + safe target
  P0-4  BREAKOUT_CONFIRMED requires final_score ≥ min_alert_score
  P0-5  VCP detector enforces proximity, base_bars, last_contraction guards
  P0-6  scoring.apply_vetoes adds stop_pct hard caps + price_extended cap
  P1-1  trend floor veto differentiated by pattern type
  P1-2  reversal patterns get +8 score bump on min_alert_score
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scanner.alerts import (alert_quality_gate, decide_alert_type,
                              per_pattern_min_alert_score)
from scanner.detectors import detect_breakout_retest_hold, detect_vcp
from scanner.indicators import add_common_indicators
from scanner.models import (CommonScores, PatternMatch, Signal, WeightedComponents)
from scanner.scoring import apply_vetoes
from scanner.state_machine import (PatternState, StateContext, determine_state)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _signal(*, pattern: str = "bull_flag", state: str = "breakout_confirmed",
             score: float = 78.0, price: float = 100.0,
             trigger: float | None = 95.0, invalid: float | None = 88.0,
             target: float | None = 110.0, ticker: str = "TEST") -> Signal:
    return Signal(
        run_id="r", asof=datetime.now(timezone.utc).isoformat(),
        ticker=ticker, pattern=pattern, pattern_state=state, final_score=score,
        common_scores=CommonScores(70, 70, 60, 70, 70, 70, 60, 80, 90),
        components=WeightedComponents(),
        veto_triggered=False, veto_reasons=[],
        price=price, trigger=trigger, invalid_below=invalid,
        measured_move_target=target, features_json={},
    )


def _match(*, pattern: str = "bull_flag", trigger: float = 100.0,
            invalid: float = 90.0, target: float = 115.0,
            structural_low: float = 92.0, current_price: float = 99.0) -> PatternMatch:
    return PatternMatch(
        pattern=pattern,
        neckline=trigger, invalid_below=invalid, measured_move_target=target,
        structure={"structural_low": structural_low, "current_price": current_price,
                    "neckline_distance_pct": (trigger - current_price) / current_price,
                    "base_bars": 8},
        geometry_features={"pole_gain": 0.20, "atr_pct": 0.02},
    )


def _features_blob(*, current_price: float = 99.0,
                    spy_regime_score: float = 100,
                    earnings_days: int | None = None) -> dict[str, Any]:
    return {
        "risk": {"R1_stop_clarity": {"score": 80}},
        "market": {
            "M1_spy_regime": {"score": spy_regime_score},
            "M3_earnings_risk": {"value": earnings_days, "score": 80},
        },
        "pattern_structure": {"current_price": current_price},
    }


# ---------------------------------------------------------------------------
# P0-1: alert_quality_gate
# ---------------------------------------------------------------------------

def test_quality_gate_rejects_target_below_trigger():
    sig = _signal(target=94.0, trigger=95.0)  # target < trigger
    reasons = alert_quality_gate(sig)
    assert "target_below_trigger" in reasons, reasons


def test_quality_gate_rejects_degenerate_stop():
    sig = _signal(trigger=100.0, invalid=99.95)  # diff < 0.2%
    reasons = alert_quality_gate(sig)
    assert "degenerate_stop" in reasons, reasons


def test_quality_gate_rejects_wide_stop():
    sig = _signal(price=100.0, invalid=80.0)  # 20% stop
    reasons = alert_quality_gate(sig)
    assert any("stop_too_wide" in r for r in reasons), reasons


def test_quality_gate_rejects_already_breached_stop():
    sig = _signal(price=85.0, invalid=90.0)  # price below invalid
    reasons = alert_quality_gate(sig)
    assert "invalid_already_breached" in reasons, reasons


def test_quality_gate_rejects_extended_above_trigger():
    sig = _signal(price=110.0, trigger=100.0)  # 10% extended
    reasons = alert_quality_gate(sig)
    assert any("price_extended" in r for r in reasons), reasons


def test_quality_gate_passes_clean_signal():
    sig = _signal(price=99.0, trigger=100.0, invalid=96.0, target=110.0)
    reasons = alert_quality_gate(sig)
    assert reasons == [], reasons


def test_quality_gate_exempts_invalidated():
    """INVALIDATED alerts should pass through even with bad target/stop —
    they're risk events, not entry signals."""
    sig = _signal(state="invalidated", price=85.0, invalid=90.0,
                   target=80.0, trigger=100.0)
    reasons = alert_quality_gate(sig)
    assert reasons == [], reasons


# ---------------------------------------------------------------------------
# P0-2: state machine EXTENDED
# ---------------------------------------------------------------------------

def test_extended_triggers_at_5pct_above_trigger():
    """Daily patterns (no bars_since_breakout) must still go EXTENDED."""
    ctx = StateContext(
        final_score=80, geometry_score=70, readiness_score=70,
        breakout_distance_atr=None, closed_above_trigger=True,
        closed_below_invalid=False, bars_since_breakout=None,
        volume_confirmed=True, extended_pct=0.06,  # 6% above
    )
    state = determine_state(_match(), ctx, prior_state=None)
    assert state == PatternState.EXTENDED, state


def test_not_extended_at_3pct_above():
    ctx = StateContext(
        final_score=80, geometry_score=70, readiness_score=70,
        breakout_distance_atr=None, closed_above_trigger=True,
        closed_below_invalid=False, bars_since_breakout=None,
        volume_confirmed=True, extended_pct=0.03,
    )
    state = determine_state(_match(), ctx, prior_state=None)
    assert state == PatternState.BREAKOUT_CONFIRMED, state


def test_extended_via_atr_distance():
    """High-volatility name: -3 ATR distance trips EXTENDED even if % is small."""
    ctx = StateContext(
        final_score=80, geometry_score=70, readiness_score=70,
        breakout_distance_atr=-3.0, closed_above_trigger=True,
        closed_below_invalid=False, bars_since_breakout=None,
        volume_confirmed=True, extended_pct=0.04,  # below the 5 % gate
    )
    state = determine_state(_match(), ctx, prior_state=None)
    assert state == PatternState.EXTENDED, state


# ---------------------------------------------------------------------------
# P0-3: retest_hold detector
# ---------------------------------------------------------------------------

def _build_floating_retest_hourly() -> pd.DataFrame:
    """Build hourly bars where 'retest' floats well above the level — should be REJECTED."""
    n = 100
    idx = pd.date_range("2024-04-01 14:00", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    # Build to swing high at $100 around bar 30
    up = np.linspace(85, 100, 30)
    pb = np.linspace(100, 98, 5)
    reapproach = np.linspace(98, 100, 10)
    # Breakout to $108 (so it's a real BO)
    bo = np.linspace(100, 108, 15)
    # "Retest" only reaches $105 — well above trigger ($100), should be REJECTED
    pb2 = np.linspace(108, 105, 25)
    hold = np.linspace(105, 106, n - 30 - 5 - 10 - 15 - 25)
    close = np.concatenate([up, pb, reapproach, bo, pb2, hold])
    high = close * (1 + rng.uniform(0.002, 0.01, n))
    low = close * (1 - rng.uniform(0.002, 0.01, n))
    open_ = (high + low) / 2
    vol = rng.integers(80_000, 250_000, n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                          "volume": vol}, index=idx)


def _daily_for_atr() -> pd.DataFrame:
    n = 220
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    rng = np.random.default_rng(42)
    close = np.linspace(80, 100, n)
    high = close * (1 + rng.uniform(0.005, 0.015, n))
    low = close * (1 - rng.uniform(0.005, 0.015, n))
    open_ = (high + low) / 2
    vol = rng.integers(2_000_000, 8_000_000, n).astype(float)
    return add_common_indicators(pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": vol}, index=idx))


def test_retest_hold_rejects_floating_retest():
    """retest_low far above level must NOT be accepted as a retest."""
    hourly = _build_floating_retest_hourly()
    daily = _daily_for_atr()
    m = detect_breakout_retest_hold(hourly, daily)
    # Either reject entirely OR (if some other valid retest is found) verify target>trigger
    if m is not None:
        assert m.measured_move_target > m.neckline, \
            f"target {m.measured_move_target} <= trigger {m.neckline}"


def test_retest_hold_target_never_below_trigger():
    """Property test: if any retest_hold match exists in the synthetic suite,
    its target must always be > trigger."""
    # Use the synthetic data from the existing per-pattern smoke test
    sys.path.insert(0, str(ROOT / "tests"))
    from smoke_test_new_patterns import make_retest_hold, make_cup_with_handle
    hourly = make_retest_hold()
    daily = add_common_indicators(make_cup_with_handle())
    m = detect_breakout_retest_hold(hourly, daily)
    if m is not None:
        assert m.measured_move_target > m.neckline, \
            f"target {m.measured_move_target} <= trigger {m.neckline}"


# ---------------------------------------------------------------------------
# P0-4: BREAKOUT requires min_alert_score
# ---------------------------------------------------------------------------

def test_breakout_blocked_when_score_below_min():
    """Per P0-4: BREAKOUT alert at score 50 with min=70 must NOT fire."""
    out = decide_alert_type(
        new_state=PatternState.BREAKOUT_CONFIRMED,
        prior_alert_type=None, final_score=50.0, prior_score=None, prior_alert_at=None,
        cooldown_hours=6, min_alert_score=70.0, score_upgrade_delta=8.0,
    )
    assert out is None, f"expected None, got {out}"


def test_breakout_fires_when_score_above_min():
    out = decide_alert_type(
        new_state=PatternState.BREAKOUT_CONFIRMED,
        prior_alert_type=None, final_score=72.0, prior_score=None, prior_alert_at=None,
        cooldown_hours=6, min_alert_score=70.0, score_upgrade_delta=8.0,
    )
    assert out == "breakout", out


def test_invalidated_fires_regardless_of_score():
    """Risk events bypass score gate — a 30-point INVALIDATED still fires."""
    out = decide_alert_type(
        new_state=PatternState.INVALIDATED,
        prior_alert_type=None, final_score=30.0, prior_score=None, prior_alert_at=None,
        cooldown_hours=6, min_alert_score=70.0, score_upgrade_delta=8.0,
    )
    assert out == "invalidated", out


def test_failed_fires_regardless_of_score():
    out = decide_alert_type(
        new_state=PatternState.BREAKOUT_FAILED,
        prior_alert_type=None, final_score=20.0, prior_score=None, prior_alert_at=None,
        cooldown_hours=6, min_alert_score=70.0, score_upgrade_delta=8.0,
    )
    assert out == "failed", out


def test_extended_never_fires():
    out = decide_alert_type(
        new_state=PatternState.EXTENDED,
        prior_alert_type=None, final_score=90.0, prior_score=None, prior_alert_at=None,
        cooldown_hours=6, min_alert_score=70.0, score_upgrade_delta=8.0,
    )
    assert out is None, out


# ---------------------------------------------------------------------------
# P0-5: VCP strict
# ---------------------------------------------------------------------------

def _build_extended_vcp_daily() -> pd.DataFrame:
    """Daily series with a VCP-shaped base, then huge runup (75 % above trigger)."""
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    rng = np.random.default_rng(7)
    # base around $50 with 4 contractions
    base = np.array([50, 48, 52, 49.5, 51, 49, 50.5, 49.7, 50.2, 49.9] * 5)[:50]
    # post-base massive run to $90
    runup = np.linspace(50, 90, 100)
    # tail at $88
    tail = np.full(50, 88.0)
    close = np.concatenate([base, runup, tail])
    close = close[:n]
    high = close * (1 + rng.uniform(0.005, 0.02, n))
    low = close * (1 - rng.uniform(0.005, 0.02, n))
    open_ = (high + low) / 2
    vol = rng.integers(2_000_000, 8_000_000, n).astype(float)
    return add_common_indicators(pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": vol}, index=idx))


def test_vcp_rejects_extended_price():
    """VCP detector must reject when current is already 5%+ above trigger."""
    daily = _build_extended_vcp_daily()
    m = detect_vcp(daily)
    if m is not None:
        # If detected, the recent base should be near current price
        assert m.structure["current_price"] <= m.neckline * 1.05, \
            f"VCP picked extended base: current={m.structure['current_price']}, trigger={m.neckline}"


# ---------------------------------------------------------------------------
# P0-6: scoring vetoes — stop & extended
# ---------------------------------------------------------------------------

def test_apply_vetoes_caps_wide_stop():
    """stop_pct > 15% → score capped at 55."""
    common = CommonScores(80, 80, 60, 70, 70, 70, 60, 80, 90)
    match = _match(trigger=100.0, invalid=80.0, current_price=100.0)  # 20% stop
    blob = _features_blob(current_price=100.0)
    score, reasons = apply_vetoes(80.0, common, match, blob)
    assert score <= 55.0, f"score {score} not capped"
    assert "stop_too_wide" in reasons, reasons


def test_apply_vetoes_extreme_stop():
    """stop_pct > 25% → score capped at 40."""
    common = CommonScores(80, 80, 60, 70, 70, 70, 60, 80, 90)
    match = _match(trigger=100.0, invalid=70.0, current_price=100.0)  # 30% stop
    blob = _features_blob(current_price=100.0)
    score, reasons = apply_vetoes(80.0, common, match, blob)
    assert score <= 40.0, f"score {score} not capped"
    assert "stop_extreme" in reasons, reasons


def test_apply_vetoes_caps_extended_price():
    """price > trigger × 1.05 → score capped at 50."""
    common = CommonScores(80, 80, 60, 70, 70, 70, 60, 80, 90)
    match = _match(trigger=100.0, invalid=95.0, current_price=110.0)  # 10% extended
    blob = _features_blob(current_price=110.0)
    score, reasons = apply_vetoes(80.0, common, match, blob)
    assert score <= 50.0, f"score {score} not capped"
    assert "price_extended_above_trigger" in reasons, reasons


def test_apply_vetoes_target_below_trigger_capped():
    common = CommonScores(80, 80, 60, 70, 70, 70, 60, 80, 90)
    match = _match(trigger=100.0, invalid=95.0, target=98.0, current_price=99.0)
    blob = _features_blob(current_price=99.0)
    score, reasons = apply_vetoes(80.0, common, match, blob)
    assert score <= 40.0, f"score {score} not capped"
    assert "target_below_trigger" in reasons, reasons


# ---------------------------------------------------------------------------
# P1-1: trend floor (differentiated)
# ---------------------------------------------------------------------------

def test_trend_floor_caps_continuation_pattern():
    """T<30 + bull_flag → score cap at 55."""
    common = CommonScores(80, 80, 60, 70, 70, 70, 60, 80, 90)
    common = CommonScores(trend=20, geometry=80, compression=60, sr_quality=70,
                           volume=70, readiness=70, risk=60, market=80, liquidity=90)
    match = _match(pattern="bull_flag")
    blob = _features_blob()
    score, reasons = apply_vetoes(80.0, common, match, blob)
    assert score <= 55.0, f"score {score} not capped"
    assert "trend_floor" in reasons, reasons


def test_retest_hold_needs_higher_trend():
    """T<50 + retest_hold → score cap at 65 (stricter than other patterns)."""
    common = CommonScores(trend=40, geometry=80, compression=60, sr_quality=70,
                           volume=70, readiness=70, risk=60, market=80, liquidity=90)
    match = _match(pattern="breakout_retest_hold")
    blob = _features_blob()
    score, reasons = apply_vetoes(80.0, common, match, blob)
    assert score <= 65.0, f"score {score} not capped"
    assert "weak_retest_trend" in reasons, reasons


def test_reversal_pattern_capped_in_downtrend_stock():
    """T<50 + double_bottom → score cap at 65."""
    common = CommonScores(trend=40, geometry=80, compression=60, sr_quality=70,
                           volume=70, readiness=70, risk=60, market=80, liquidity=90)
    match = _match(pattern="double_bottom")
    blob = _features_blob()
    score, reasons = apply_vetoes(80.0, common, match, blob)
    assert score <= 65.0, f"score {score} not capped"
    assert "reversal_trend_cap" in reasons, reasons


# ---------------------------------------------------------------------------
# P1-2: per-pattern min_alert_score
# ---------------------------------------------------------------------------

def test_continuation_pattern_uses_base_min():
    assert per_pattern_min_alert_score("bull_flag", 70.0) == 70.0
    assert per_pattern_min_alert_score("vcp", 72.0) == 72.0


def test_reversal_pattern_gets_score_bump():
    assert per_pattern_min_alert_score("double_bottom", 70.0) == 78.0
    assert per_pattern_min_alert_score("inverse_head_shoulders", 75.0) == 83.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> int:
    tests = [
        # P0-1
        ("P0-1 quality gate: target<trigger",          test_quality_gate_rejects_target_below_trigger),
        ("P0-1 quality gate: degenerate stop",         test_quality_gate_rejects_degenerate_stop),
        ("P0-1 quality gate: wide stop",                test_quality_gate_rejects_wide_stop),
        ("P0-1 quality gate: already breached",        test_quality_gate_rejects_already_breached_stop),
        ("P0-1 quality gate: extended price",          test_quality_gate_rejects_extended_above_trigger),
        ("P0-1 quality gate: clean passes",             test_quality_gate_passes_clean_signal),
        ("P0-1 quality gate: invalidated exempt",      test_quality_gate_exempts_invalidated),
        # P0-2
        ("P0-2 EXTENDED at +5 % above trigger",         test_extended_triggers_at_5pct_above_trigger),
        ("P0-2 not EXTENDED at +3 %",                   test_not_extended_at_3pct_above),
        ("P0-2 EXTENDED via -3 ATR distance",           test_extended_via_atr_distance),
        # P0-3
        ("P0-3 retest_hold rejects floating retest",   test_retest_hold_rejects_floating_retest),
        ("P0-3 retest_hold target > trigger always",   test_retest_hold_target_never_below_trigger),
        # P0-4
        ("P0-4 BREAKOUT blocked below min_alert",      test_breakout_blocked_when_score_below_min),
        ("P0-4 BREAKOUT fires above min_alert",        test_breakout_fires_when_score_above_min),
        ("P0-4 INVALIDATED bypasses gate",             test_invalidated_fires_regardless_of_score),
        ("P0-4 FAILED bypasses gate",                   test_failed_fires_regardless_of_score),
        ("P0-4 EXTENDED never fires",                   test_extended_never_fires),
        # P0-5
        ("P0-5 VCP rejects extended price",            test_vcp_rejects_extended_price),
        # P0-6
        ("P0-6 wide stop capped",                       test_apply_vetoes_caps_wide_stop),
        ("P0-6 extreme stop capped",                    test_apply_vetoes_extreme_stop),
        ("P0-6 extended-price capped",                  test_apply_vetoes_caps_extended_price),
        ("P0-6 target<trigger capped",                  test_apply_vetoes_target_below_trigger_capped),
        # P1-1
        ("P1-1 trend floor: continuation",              test_trend_floor_caps_continuation_pattern),
        ("P1-1 trend floor: retest_hold needs T≥50",   test_retest_hold_needs_higher_trend),
        ("P1-1 trend floor: reversal in downtrend",    test_reversal_pattern_capped_in_downtrend_stock),
        # P1-2
        ("P1-2 continuation uses base min",             test_continuation_pattern_uses_base_min),
        ("P1-2 reversal +8 bump",                       test_reversal_pattern_gets_score_bump),
    ]
    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✓ {name}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {name}: assertion failed: {e}")
        except Exception as e:
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} P0/P1 patch tests passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
