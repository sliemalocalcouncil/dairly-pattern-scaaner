"""Telegram alert dispatch + state file for dedup.

Alert decisions follow design §7.5: alert type derives from state transitions.
State file lives at result/_state.json and is committed by the GH Actions job.

State entry schema (per ticker × pattern):
    {
        "state":              "<PatternState value>",
        "last_alert_type":    "watch|setup|breakout|retest|failed|invalidated|None",
        "last_alert_score":   float | None,
        "last_alert_at":      "<ISO8601 UTC>" | None,
        # Active monitoring (review §P0-2): so the tracker can detect
        # invalidation/failure even if the detector stops returning a match.
        "trigger":            float | None,
        "invalid_below":      float | None,
        "monitored_until":    "<ISO8601 UTC>" | None,
    }
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from .config import Settings
from .models import AlertRecord, Signal
from .state_machine import PatternState

log = logging.getLogger(__name__)

STATE_FILE = "_state.json"


# ---------------------------------------------------------------------------
# State file (per ticker × pattern)
# ---------------------------------------------------------------------------

class AlertState:
    def __init__(self, settings: Settings):
        self.path: Path = settings.absolute_result_dir() / STATE_FILE
        self._data: dict[str, dict[str, Any]] = {}
        self._load()

    def _key(self, ticker: str, pattern: str) -> str:
        return f"{ticker.upper()}::{pattern}"

    def _load(self) -> None:
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception as e:
                log.warning("could not parse state file %s: %s", self.path, e)
                self._data = {}

    def save(self) -> None:
        """Atomic write — review §P2: never leave a partial state file on crash."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self._data, indent=2, sort_keys=True)
        tmp_fd, tmp_path = tempfile.mkstemp(prefix=".state_", suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp_path, self.path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def get(self, ticker: str, pattern: str) -> dict[str, Any]:
        return self._data.get(self._key(ticker, pattern), {})

    def update(self, ticker: str, pattern: str, **fields: Any) -> None:
        """Merge `fields` into the existing entry (preserving keys not given)."""
        key = self._key(ticker, pattern)
        existing = dict(self._data.get(key, {}))
        existing.update({k: v for k, v in fields.items() if v is not None or k in existing})
        self._data[key] = existing

    def delete(self, ticker: str, pattern: str) -> None:
        self._data.pop(self._key(ticker, pattern), None)

    def all_entries(self) -> list[tuple[str, str, dict[str, Any]]]:
        out: list[tuple[str, str, dict[str, Any]]] = []
        for key, entry in self._data.items():
            if "::" not in key:
                continue
            ticker, pattern = key.split("::", 1)
            out.append((ticker, pattern, entry))
        return out


# ---------------------------------------------------------------------------
# Alert decision
# ---------------------------------------------------------------------------

def decide_alert_type(new_state: PatternState, prior_alert_type: str | None,
                       final_score: float, prior_score: float | None,
                       prior_alert_at: str | None,
                       cooldown_hours: int, min_alert_score: float,
                       score_upgrade_delta: float) -> str | None:
    """Return alert_type to send, or None to suppress.

    Per agreement guidebook §3 P0-4: BREAKOUT_CONFIRMED, RETEST_HOLD, SETUP, WATCH
    all require final_score ≥ min_alert_score. Only INVALIDATED and BREAKOUT_FAILED
    are exempt (they're risk events, not entry signals).
    """

    # ── Terminal/risk events fire regardless of score (NOT entry signals) ──
    if new_state == PatternState.INVALIDATED:
        return None if prior_alert_type == "invalidated" else "invalidated"
    if new_state == PatternState.BREAKOUT_FAILED:
        return None if prior_alert_type == "failed" else "failed"

    # ── EXTENDED never alerts ──
    if new_state == PatternState.EXTENDED:
        return None

    # ── Score gate applies to ALL entry signals (P0-4 fix) ──
    if final_score < min_alert_score:
        return None

    if new_state == PatternState.BREAKOUT_CONFIRMED:
        if prior_alert_type != "breakout":
            return "breakout"
        # already alerted breakout; allow re-alert only on big upgrade
        if prior_score is not None and final_score - prior_score >= score_upgrade_delta:
            return "breakout"
        return None

    # RETEST_HOLD now fires its own alert with score gate applied
    if new_state == PatternState.RETEST_HOLD:
        if prior_alert_type == "retest":
            return None
        return "retest"

    # Setup / candidate
    if new_state == PatternState.SETUP:
        target_type = "setup"
    else:
        target_type = "watch"

    # Cooldown check
    if prior_alert_type == target_type and prior_alert_at:
        try:
            prior_dt = datetime.fromisoformat(prior_alert_at.replace("Z", "+00:00"))
            age_hours = (datetime.now(timezone.utc) - prior_dt).total_seconds() / 3600
            if age_hours < cooldown_hours:
                # allow only if score upgraded materially
                if prior_score is not None and final_score - prior_score >= score_upgrade_delta:
                    return target_type
                return None
        except Exception:
            pass

    return target_type


# ---------------------------------------------------------------------------
# P0-1: Alert quality gate — final guard before Telegram dispatch
# ---------------------------------------------------------------------------

def alert_quality_gate(signal: Signal, *, max_stop_pct: float = 15.0) -> list[str]:
    """Return list of reasons the signal should be REJECTED at the alert layer.

    This is intentionally redundant with detector / scoring vetoes. If a bad
    signal slipped through earlier layers (e.g., a new detector with a bug),
    this gate is the final stop before the user sees a malformed alert.

    Risk events (INVALIDATED / BREAKOUT_FAILED) are exempt from this gate
    because they're notifications, not entry signals.
    """
    reasons: list[str] = []

    # Risk events bypass the gate
    if signal.pattern_state in (PatternState.INVALIDATED.value,
                                  PatternState.BREAKOUT_FAILED.value):
        return reasons

    # 1. target ≤ trigger → math broken
    if (signal.trigger is not None and signal.measured_move_target is not None
            and signal.measured_move_target <= signal.trigger):
        reasons.append("target_below_trigger")

    # 2. degenerate stop (trigger ≈ invalid)
    if (signal.trigger is not None and signal.invalid_below is not None
            and signal.trigger > 0
            and abs(signal.trigger - signal.invalid_below) / signal.trigger < 0.002):
        reasons.append("degenerate_stop")

    # 3. stop already breached or absurdly wide
    if signal.invalid_below is not None and signal.price > 0:
        stop_pct = (signal.price - signal.invalid_below) / signal.price * 100
        if stop_pct <= 0:
            reasons.append("invalid_already_breached")
        elif stop_pct > max_stop_pct:
            reasons.append(f"stop_too_wide ({stop_pct:.1f}%)")

    # 4. price already extended above trigger
    if (signal.trigger is not None and signal.trigger > 0
            and signal.price > signal.trigger * 1.05):
        ext = (signal.price / signal.trigger - 1.0) * 100
        reasons.append(f"price_extended ({ext:.1f}% above trigger)")

    return reasons


# ---------------------------------------------------------------------------
# Telegram dispatch
# ---------------------------------------------------------------------------

EMOJI = {"watch": "👀", "setup": "🟡", "breakout": "🟢", "retest": "🔵",
         "failed": "🔴", "invalidated": "⚫"}


def format_message(signal: Signal, alert_type: str) -> str:
    e = EMOJI.get(alert_type, "•")
    pat = signal.pattern.replace("_", " ").title()
    lines = [
        f"{e} *{alert_type.upper()}* — `{signal.ticker}`",
        f"_{pat}_  ·  state: `{signal.pattern_state}`",
        f"score: *{signal.final_score:.1f}*  "
        f"(G:{signal.common_scores.geometry:.0f} T:{signal.common_scores.trend:.0f} "
        f"V:{signal.common_scores.volume:.0f} SR:{signal.common_scores.sr_quality:.0f} "
        f"BR:{signal.common_scores.readiness:.0f})",
        f"price: `${signal.price:.2f}`",
    ]
    if signal.trigger:
        lines.append(f"trigger: `${signal.trigger:.2f}`")
    if signal.invalid_below:
        lines.append(f"invalid: `${signal.invalid_below:.2f}` "
                     f"(_{(signal.price - signal.invalid_below) / signal.price * 100:.1f}% stop_)")
    if signal.measured_move_target:
        lines.append(f"target: `${signal.measured_move_target:.2f}`")
    if signal.veto_reasons:
        lines.append(f"⚠️ {', '.join(signal.veto_reasons)}")
    return "\n".join(lines)


def send_telegram(settings: Settings, text: str, parse_mode: str = "Markdown") -> bool:
    if settings.telegram_disabled:
        log.info("telegram disabled, would have sent: %s", text.replace("\n", " | ")[:120])
        return True
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        log.warning("telegram credentials missing; skipping send")
        return False
    url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
    payload = {
        "chat_id": settings.telegram_chat_id,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=15)
            if resp.status_code == 200:
                return True
            if resp.status_code == 429:
                retry = float(resp.json().get("parameters", {}).get("retry_after", 5))
                time.sleep(retry)
                continue
            log.warning("telegram %s: %s", resp.status_code, resp.text[:200])
        except Exception as e:
            log.warning("telegram error attempt=%d: %s", attempt + 1, e)
        time.sleep(2)
    return False


# ---------------------------------------------------------------------------
# Public entrypoints
# ---------------------------------------------------------------------------

def _trading_days_from_now(n: int) -> str:
    """ISO8601 UTC timestamp `n` trading days from now (close at 16:00 ET)."""
    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar("XNYS")
        start = datetime.now(timezone.utc).date()
        sched = nyse.schedule(start_date=start, end_date=start + __import__("datetime").timedelta(days=n * 2 + 14))
        if len(sched) >= n + 1:
            close_ts = sched.iloc[n]["market_close"]
            return pd.Timestamp(close_ts).tz_convert("UTC").isoformat()
    except Exception:
        pass
    # Fallback: rough timedelta
    return (datetime.now(timezone.utc) + __import__("datetime").timedelta(days=n + 2)).isoformat()


def update_signal_state(state: AlertState, signal: Signal, monitoring_days: int) -> None:
    """Persist a signal's state without sending an alert. Used for non-best signals
    so the active-signal tracker can still find them next cycle (review §P1)."""
    monitored_until = None
    new_pattern_state = signal.pattern_state
    if new_pattern_state in {PatternState.SETUP.value,
                              PatternState.BREAKOUT_CONFIRMED.value,
                              PatternState.RETEST_HOLD.value}:
        monitored_until = _trading_days_from_now(monitoring_days)
    state.update(
        signal.ticker, signal.pattern,
        state=new_pattern_state,
        trigger=signal.trigger,
        invalid_below=signal.invalid_below,
        monitored_until=monitored_until,
    )


# ---------------------------------------------------------------------------
# P1-2: per-pattern min_alert_score (2-tier — agreement guidebook §2 #10)
# ---------------------------------------------------------------------------

# Reversal patterns require a higher score because they fight the existing trend.
# Continuation patterns can use the user's MIN_ALERT_SCORE as-is.
# Phase v2.3 (after forward-return ≥30 cases) will fully separate per-pattern scores;
# this is the conservative interim split.
REVERSAL_MIN_ALERT_SCORE_BUMP = 8.0


def per_pattern_min_alert_score(pattern: str, base_min: float) -> float:
    if pattern in {"double_bottom", "inverse_head_shoulders"}:
        return base_min + REVERSAL_MIN_ALERT_SCORE_BUMP
    return base_min


def maybe_alert(settings: Settings, state: AlertState, signal: Signal) -> AlertRecord | None:
    """Decide + dispatch + persist state. Returns AlertRecord if sent, else None."""
    prior = state.get(signal.ticker, signal.pattern)

    # P1-2: pattern-aware threshold (reversal patterns get a +8 bump)
    effective_min = per_pattern_min_alert_score(signal.pattern, settings.min_alert_score)

    alert_type = decide_alert_type(
        new_state=PatternState(signal.pattern_state),
        prior_alert_type=prior.get("last_alert_type"),
        final_score=signal.final_score,
        prior_score=prior.get("last_alert_score"),
        prior_alert_at=prior.get("last_alert_at"),
        cooldown_hours=settings.cooldown_hours,
        min_alert_score=effective_min,
        score_upgrade_delta=settings.score_upgrade_delta,
    )

    now_iso = datetime.now(timezone.utc).isoformat()
    monitored_until = None
    if signal.pattern_state in {PatternState.SETUP.value,
                                  PatternState.BREAKOUT_CONFIRMED.value,
                                  PatternState.RETEST_HOLD.value}:
        monitored_until = _trading_days_from_now(settings.active_monitoring_trading_days)

    base_update = dict(
        state=signal.pattern_state,
        trigger=signal.trigger,
        invalid_below=signal.invalid_below,
        monitored_until=monitored_until,
    )

    if alert_type is None:
        # No alert, but still record the latest state for next cycle.
        state.update(signal.ticker, signal.pattern, **base_update)
        return None

    # P0-1: Final alert quality gate — last line of defense.
    # If a malformed signal (target<trigger, stop>15%, extended, etc.) somehow
    # made it through detection + scoring, refuse to send the Telegram message.
    # Risk events (FAILED / INVALIDATED) bypass this gate.
    gate_reasons = alert_quality_gate(signal)
    if gate_reasons:
        log.info("alert quality gate REJECTED %s %s (%s): %s",
                 signal.ticker, signal.pattern, alert_type, ", ".join(gate_reasons))
        # Persist state so cooldown still applies, but mark we did NOT alert.
        state.update(signal.ticker, signal.pattern, **base_update)
        return None

    text = format_message(signal, alert_type)
    if settings.dry_run:
        log.info("[DRY] would send: %s", text.replace("\n", " | ")[:200])
        sent = True
    else:
        sent = send_telegram(settings, text)

    if sent:
        state.update(
            signal.ticker, signal.pattern,
            **base_update,
            last_alert_type=alert_type,
            last_alert_score=signal.final_score,
            last_alert_at=now_iso,
        )
        return AlertRecord(
            run_id=signal.run_id,
            asof=signal.asof,
            ticker=signal.ticker,
            pattern=signal.pattern,
            pattern_state=signal.pattern_state,
            alert_type=alert_type,
            final_score=signal.final_score,
            price=signal.price,
            trigger=signal.trigger,
            invalid_below=signal.invalid_below,
            note=signal.note,
        )
    return None


# Lazy-import pandas only when the helper is used
import pandas as pd  # noqa: E402
