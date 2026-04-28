"""Combine common scores + pattern geometry + apply weight matrix + veto rules.

Per design §5.2 and §5.5.
"""
from __future__ import annotations

from typing import Any

from .models import CommonScores, PatternMatch, WeightedComponents


# Weight matrix per design §5.2 — keyed by pattern, values sum to 1.0 (rounded).
WEIGHT_MATRIX: dict[str, dict[str, float]] = {
    "double_bottom":          {"geometry": 0.25, "trend": 0.15, "compression": 0.08,
                               "sr_quality": 0.14, "volume": 0.13, "readiness": 0.08,
                               "risk": 0.06, "market": 0.06, "liquidity": 0.05},
    "ascending_triangle":     {"geometry": 0.28, "trend": 0.15, "compression": 0.08,
                               "sr_quality": 0.14, "volume": 0.12, "readiness": 0.08,
                               "risk": 0.04, "market": 0.06, "liquidity": 0.05},
    "bull_flag":              {"geometry": 0.30, "trend": 0.14, "compression": 0.09,
                               "sr_quality": 0.10, "volume": 0.14, "readiness": 0.09,
                               "risk": 0.03, "market": 0.06, "liquidity": 0.05},
    "high_tight_pullback":    {"geometry": 0.35, "trend": 0.12, "compression": 0.10,
                               "sr_quality": 0.08, "volume": 0.14, "readiness": 0.09,
                               "risk": 0.01, "market": 0.06, "liquidity": 0.05},
    "vcp":                    {"geometry": 0.25, "trend": 0.15, "compression": 0.12,
                               "sr_quality": 0.12, "volume": 0.13, "readiness": 0.07,
                               "risk": 0.05, "market": 0.06, "liquidity": 0.05},
    "cup_with_handle":        {"geometry": 0.30, "trend": 0.14, "compression": 0.08,
                               "sr_quality": 0.12, "volume": 0.13, "readiness": 0.08,
                               "risk": 0.05, "market": 0.05, "liquidity": 0.05},
    "inverse_head_shoulders": {"geometry": 0.28, "trend": 0.12, "compression": 0.08,
                               "sr_quality": 0.14, "volume": 0.14, "readiness": 0.08,
                               "risk": 0.06, "market": 0.05, "liquidity": 0.05},
    "base_on_base":           {"geometry": 0.25, "trend": 0.18, "compression": 0.10,
                               "sr_quality": 0.10, "volume": 0.12, "readiness": 0.08,
                               "risk": 0.05, "market": 0.07, "liquidity": 0.05},
    "tight_consolidation":    {"geometry": 0.20, "trend": 0.16, "compression": 0.18,
                               "sr_quality": 0.10, "volume": 0.13, "readiness": 0.08,
                               "risk": 0.05, "market": 0.05, "liquidity": 0.05},
    "breakout_retest_hold":   {"geometry": 0.22, "trend": 0.14, "compression": 0.06,
                               "sr_quality": 0.14, "volume": 0.13, "readiness": 0.18,
                               "risk": 0.05, "market": 0.05, "liquidity": 0.03},
}

# Patterns that are particularly trend-sensitive — extra penalty in downtrend market
TREND_SENSITIVE_PATTERNS = {"bull_flag", "high_tight_pullback", "vcp", "cup_with_handle",
                              "base_on_base", "tight_consolidation", "breakout_retest_hold",
                              "ascending_triangle"}

# Reversal patterns — looser trend floor (these *expect* weak prior trend by design)
REVERSAL_PATTERNS = {"double_bottom", "inverse_head_shoulders"}


def apply_weights(common: CommonScores, geometry_score: float, pattern: str) -> WeightedComponents:
    """Multiply each 0–100 score by its pattern weight; return components."""
    w = WEIGHT_MATRIX.get(pattern)
    if w is None:
        # default: even weights across all 9 dims
        w = {k: 1 / 9 for k in [
            "geometry", "trend", "compression", "sr_quality", "volume",
            "readiness", "risk", "market", "liquidity",
        ]}

    # Replace common.geometry with the *pattern-specific* geometry score
    return WeightedComponents(
        trend=common.trend * w["trend"],
        geometry=geometry_score * w["geometry"],
        compression=common.compression * w["compression"],
        sr_quality=common.sr_quality * w["sr_quality"],
        volume=common.volume * w["volume"],
        readiness=common.readiness * w["readiness"],
        risk=common.risk * w["risk"],
        market=common.market * w["market"],
        liquidity=common.liquidity * w["liquidity"],
    )


def apply_vetoes(raw_score: float, common: CommonScores, match: PatternMatch,
                  features_blob: dict[str, Any]) -> tuple[float, list[str]]:
    """Hard caps for known-bad situations. Returns (capped_score, reasons).

    Per agreement guidebook §3 P0-6 + P1-1, this layer adds:
      - stop_pct hard veto (>15 % cap, >25 % heavy cap)
      - trend floor veto for trend-sensitive patterns (T<30 cap)
      - extra-strict trend floor for retest_hold (T<50 cap)
      - reversal-pattern strict trend cap when used as breakout signal
      - price-extended cap (current > trigger × 1.05)
      - target<=trigger early reject (belt-and-suspenders; detector should also reject)
    """
    reasons: list[str] = []
    score = raw_score

    # ── Liquidity floor — if liquidity is weak, drop the signal entirely ──
    if common.liquidity < 25:
        reasons.append("liquidity_floor")
        score = 0.0

    # ── Risk clarity floor — if no clean stop, cap at 55 ──
    risk_blob = features_blob.get("risk", {})
    r1 = risk_blob.get("R1_stop_clarity", {}).get("score", 100)
    if match.invalid_below is None or r1 < 30:
        reasons.append("risk_unclear")
        score = min(score, 55.0)

    # ── P0-6: stop_pct hard veto ──
    # `risk` weight in the matrix is too small (≤6 %) for a 40 % stop to dent
    # final_score meaningfully. Replace with hard caps.
    price = float(features_blob.get("pattern_structure", {}).get("current_price", 0))
    if price > 0 and match.invalid_below:
        stop_pct = (price - match.invalid_below) / price * 100
        if stop_pct <= 0:
            reasons.append("invalid_already_breached")
            score = min(score, 40.0)
        elif stop_pct > 25:
            reasons.append("stop_extreme")
            score = min(score, 40.0)
        elif stop_pct > 15:
            reasons.append("stop_too_wide")
            score = min(score, 55.0)

    # ── P0: target ≤ trigger sanity check (belt-and-suspenders) ──
    if (match.measured_move_target is not None and match.neckline is not None
            and match.measured_move_target <= match.neckline):
        reasons.append("target_below_trigger")
        score = min(score, 40.0)

    # ── P0-2 redundant safety: price already extended above trigger ──
    if price > 0 and match.neckline and price > match.neckline * 1.05:
        reasons.append("price_extended_above_trigger")
        score = min(score, 50.0)

    # ── Earnings risk veto ──
    market_blob = features_blob.get("market", {})
    m3 = market_blob.get("M3_earnings_risk", {})
    earn_days = m3.get("value")
    if isinstance(earn_days, int) and earn_days <= 5:
        reasons.append("earnings_within_5d")
        score = min(score, 60.0)

    # ── Market regime veto (SPY) for trend-sensitive patterns ──
    m1 = market_blob.get("M1_spy_regime", {}).get("score", 100)
    if m1 < 30 and match.pattern in TREND_SENSITIVE_PATTERNS:
        reasons.append("market_downtrend")
        score = min(score, 65.0)

    # ── P1-1: per-stock trend floor veto (differentiated by pattern type) ──
    # SPY regime ≠ individual stock trend. A continuation pattern on a stock
    # whose own trend is dead has 70%+ historical false-breakout rate.
    if common.trend < 30 and match.pattern in TREND_SENSITIVE_PATTERNS:
        reasons.append("trend_floor")
        score = min(score, 55.0)

    # Retest Hold needs even stronger trend (it's a continuation-of-trend signal)
    if common.trend < 50 and match.pattern == "breakout_retest_hold":
        reasons.append("weak_retest_trend")
        score = min(score, 65.0)

    # Reversal patterns: cap when used as outright breakout in a downtrend stock
    if common.trend < 50 and match.pattern in REVERSAL_PATTERNS:
        reasons.append("reversal_trend_cap")
        score = min(score, 65.0)

    return float(round(score, 2)), reasons
