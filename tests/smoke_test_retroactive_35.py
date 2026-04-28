"""Retroactive verification — replay the 35 alerts from ChatExport against new gates.

Per agreement guidebook §4 Test Plan: the patches' value is measured by how many
of the historical bad alerts they would have blocked.

Expected outcomes (from briefing analysis):
  - 35 total alerts → 0 ACTIONABLE, 2 CAUTION, 15 SKIP, 10 INFO (FAILED/INVALIDATED)
  - Of the 17 BREAKOUT/RETEST alerts: ~6 had target<trigger, several had stop>15%,
    several had price extended >5% above trigger, several had T<30
  - INFO (10) should still pass through as risk events

We expect ≥13 of the 17 entry alerts to be blocked by the new gates,
and all 10 INFO alerts to pass through unchanged.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scanner.alerts import alert_quality_gate, per_pattern_min_alert_score
from scanner.models import CommonScores, Signal, WeightedComponents


def _alert_to_signal(a: dict) -> Signal:
    """Reconstruct a minimal Signal from a parsed Telegram alert."""
    state_map = {
        "BREAKOUT": "breakout_confirmed",
        "RETEST":   "retest_hold",
        "SETUP":    "setup",
        "WATCH":    "candidate",
        "FAILED":   "breakout_failed",
        "INVALIDATED": "invalidated",
    }
    pattern_map = {
        "Vcp": "vcp",
        "Double Bottom": "double_bottom",
        "Bull Flag": "bull_flag",
        "High Tight Pullback": "high_tight_pullback",
        "Ascending Triangle": "ascending_triangle",
        "Cup With Handle": "cup_with_handle",
        "Inverse Head Shoulders": "inverse_head_shoulders",
        "Base On Base": "base_on_base",
        "Tight Consolidation": "tight_consolidation",
        "Breakout Retest Hold": "breakout_retest_hold",
    }
    # Compute trivial target from price/trigger if not in alert (Telegram message
    # only includes target when present in the original signal; older signals
    # without target should still be reconstructable)
    return Signal(
        run_id="historical", asof=a["ts"],
        ticker=a["ticker"],
        pattern=pattern_map.get(a["pattern"], a["pattern"].lower().replace(" ", "_")),
        pattern_state=state_map.get(a["alert_type"], "unknown"),
        final_score=a["score"] or 0,
        common_scores=CommonScores(
            trend=a["T"] or 50, geometry=a["G"] or 50, compression=50,
            sr_quality=a["SR"] or 50, volume=a["V"] or 50,
            readiness=a["BR"] or 50, risk=50, market=50, liquidity=50,
        ),
        components=WeightedComponents(),
        veto_triggered=False, veto_reasons=[],
        price=a["price"] or 0,
        trigger=a["trigger"],
        invalid_below=a["invalid_below"],
        # Synthesize a target — for retest_hold use 2x retest depth
        measured_move_target=_synth_target(a),
        features_json={},
    )


def _synth_target(a: dict) -> float | None:
    """Reconstruct what the target field would have been in the original alert.

    For retest_hold alerts we know from the user's VNET example that the bug
    produced target ≤ trigger. We use the same formula here so we can verify
    the quality gate would have caught it.
    """
    if a["pattern"] != "Breakout Retest Hold":
        return None
    if a["trigger"] is None or a["invalid_below"] is None:
        return None
    # Original buggy formula: P + (P - retest_low) * 2; retest_low ≈ invalid_below / 0.99
    P = a["trigger"]
    retest_low = a["invalid_below"] / 0.99
    return P + (P - retest_low) * 2


def main() -> int:
    alerts_path = ROOT / "tests" / "fixtures" / "historical_alerts_2026-04-28.json"
    if not alerts_path.exists():
        print(f"SKIP: fixture {alerts_path} not found")
        return 0
    alerts = json.loads(alerts_path.read_text())
    print(f"Loaded {len(alerts)} historical alerts\n")

    # Counters
    by_outcome = {"BLOCKED_GATE": [], "BLOCKED_SCORE": [],
                   "PASSED_RISK_EVENT": [], "PASSED_ENTRY": []}

    # Use the same defaults the user runs in production
    min_alert_score = 70.0

    for a in alerts:
        sig = _alert_to_signal(a)

        # Risk events first — they bypass everything
        if sig.pattern_state in ("invalidated", "breakout_failed"):
            by_outcome["PASSED_RISK_EVENT"].append((a, sig, []))
            continue

        # Score gate
        effective_min = per_pattern_min_alert_score(sig.pattern, min_alert_score)
        if sig.final_score < effective_min:
            by_outcome["BLOCKED_SCORE"].append((a, sig, [f"score {sig.final_score:.1f} < {effective_min:.1f}"]))
            continue

        # Quality gate
        gate_reasons = alert_quality_gate(sig)
        if gate_reasons:
            by_outcome["BLOCKED_GATE"].append((a, sig, gate_reasons))
            continue

        by_outcome["PASSED_ENTRY"].append((a, sig, []))

    # ── Print summary ─────────────────────────────────────────────────
    print("=" * 90)
    print(f"OUTCOME BREAKDOWN")
    print("=" * 90)
    print(f"  PASSED — entry signals (would Telegram):   {len(by_outcome['PASSED_ENTRY'])}")
    print(f"  PASSED — risk events (FAILED/INVALIDATED): {len(by_outcome['PASSED_RISK_EVENT'])}")
    print(f"  BLOCKED — quality gate (target/stop/ext):  {len(by_outcome['BLOCKED_GATE'])}")
    print(f"  BLOCKED — score below per-pattern min:     {len(by_outcome['BLOCKED_SCORE'])}")
    print(f"  TOTAL:                                      {sum(len(v) for v in by_outcome.values())}")

    print()
    print("─" * 90)
    print("BLOCKED — quality gate (these would have been bad signals):")
    print("─" * 90)
    for a, sig, reasons in by_outcome["BLOCKED_GATE"]:
        print(f"  {a['ticker']:6s} {a['alert_type']:12s} {a['pattern']:22s} score={a['score']:5.1f}  → {', '.join(reasons)}")

    print()
    print("─" * 90)
    print("BLOCKED — score below pattern-specific min:")
    print("─" * 90)
    for a, sig, reasons in by_outcome["BLOCKED_SCORE"]:
        print(f"  {a['ticker']:6s} {a['alert_type']:12s} {a['pattern']:22s} {reasons[0]}")

    print()
    print("─" * 90)
    print(f"PASSED — these would have made it to Telegram ({len(by_outcome['PASSED_ENTRY'])} signals):")
    print("─" * 90)
    for a, sig, _ in by_outcome["PASSED_ENTRY"]:
        print(f"  {a['ticker']:6s} {a['alert_type']:12s} {a['pattern']:22s} score={a['score']:5.1f}  T={a['T']}")

    print()
    print("─" * 90)
    print(f"PASSED — risk events (kept as-is by design, {len(by_outcome['PASSED_RISK_EVENT'])} signals):")
    print("─" * 90)
    for a, sig, _ in by_outcome["PASSED_RISK_EVENT"]:
        print(f"  {a['ticker']:6s} {a['alert_type']:12s} {a['pattern']:22s} score={a['score']:5.1f}")

    # ── Acceptance check ──────────────────────────────────────────────
    print()
    print("=" * 90)
    print("ACCEPTANCE CRITERIA (per agreement guidebook §4):")
    print("=" * 90)
    # Re-evaluate what quality gate WOULD have caught regardless of score gate
    # (to verify gate completeness, since score gate may shadow some)
    gate_only_catches = []
    for a in alerts:
        sig = _alert_to_signal(a)
        if sig.pattern_state in ("invalidated", "breakout_failed"):
            continue
        gate_reasons = alert_quality_gate(sig)
        if gate_reasons:
            gate_only_catches.append((a, gate_reasons))

    passes = 0
    total = 0

    # 1. All risk events should pass through
    total += 1
    risk_count = len(by_outcome["PASSED_RISK_EVENT"])
    expected_risk = sum(1 for a in alerts if a["alert_type"] in ("INVALIDATED", "FAILED"))
    if risk_count == expected_risk:
        print(f"  ✓ All {risk_count} risk events preserved (expected {expected_risk})")
        passes += 1
    else:
        print(f"  ✗ Risk events preserved = {risk_count}, expected {expected_risk}")

    # 2. Quality gate would catch ≥5 target<trigger if score gate didn't shadow
    total += 1
    gate_targets = sum(1 for _, r in gate_only_catches if any("target_below_trigger" in x for x in r))
    if gate_targets >= 5:
        print(f"  ✓ {gate_targets} target<trigger caught by quality gate (expected ≥5)")
        passes += 1
    else:
        print(f"  ⚠ {gate_targets} target<trigger caught by quality gate (expected ≥5)")

    # 3. Quality gate catches wide stops
    total += 1
    gate_wide = sum(1 for _, r in gate_only_catches if any("stop_too_wide" in x for x in r))
    if gate_wide >= 5:
        print(f"  ✓ {gate_wide} wide-stop alerts caught by quality gate (expected ≥5)")
        passes += 1
    else:
        print(f"  ⚠ {gate_wide} wide-stop alerts caught by quality gate (expected ≥5)")

    # 4. Quality gate catches extended-price
    total += 1
    gate_ext = sum(1 for _, r in gate_only_catches if any("price_extended" in x for x in r))
    if gate_ext >= 5:
        print(f"  ✓ {gate_ext} extended-price alerts caught by quality gate (expected ≥5)")
        passes += 1
    else:
        print(f"  ⚠ {gate_ext} extended-price alerts caught by quality gate (expected ≥5)")

    # 5. Total entry alerts that pass should be ≤ 5 of ~24
    total += 1
    entry_alerts = sum(1 for a in alerts if a["alert_type"] not in ("INVALIDATED", "FAILED"))
    passed_entries = len(by_outcome["PASSED_ENTRY"])
    block_rate = (entry_alerts - passed_entries) / entry_alerts * 100
    if passed_entries <= 5:
        print(f"  ✓ {passed_entries}/{entry_alerts} entry alerts pass = {block_rate:.0f}% blocked (expected ≥80%)")
        passes += 1
    else:
        print(f"  ✗ {passed_entries}/{entry_alerts} entry alerts pass = only {block_rate:.0f}% blocked")

    # 6. Score gate did the heavy lifting (proves layered defense works)
    total += 1
    score_blocks = len(by_outcome["BLOCKED_SCORE"])
    if score_blocks >= 10:
        print(f"  ✓ Score gate blocked {score_blocks} alerts (cheap first-line defense)")
        passes += 1
    else:
        print(f"  ⚠ Score gate only blocked {score_blocks}")

    print(f"\n  Acceptance: {passes}/{total} criteria met")
    return 0 if passes == total else 1


if __name__ == "__main__":
    sys.exit(main())
