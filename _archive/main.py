"""
DNS Options Scanner - Main System
Clean, professional options trading system
"""

from datetime import datetime, time
from typing import Dict, List, Optional
import config
from dhan_api import get_dhan
from signal_detector import SignalDetector
from trade_validator import TradeValidator
from gap_predictor import GapPredictor
from accuracy_tracker import AccuracyTracker
from excel_audit_logger import ExcelAuditLogger
from vega_theta_engine import VegaThetaEngine


class DNSScanner:
    """
    Main DNS Options Scanner System
    
    Workflow:
    1. 9:15 AM - Capture baseline Greeks
    2. 10:00 AM - Generate signals and validate trades
    """
    
    def __init__(self):
        self.dhan = get_dhan()
        self.detectors = {}  # One detector per index
        self.validator = TradeValidator()
        self.accuracy_tracker = AccuracyTracker()
        self.excel_logger = ExcelAuditLogger()
        self.vega_engine = VegaThetaEngine()
        
        # Initialize detectors for each index
        for index in config.INDICES:
            self.detectors[index] = SignalDetector()
    
    def run_baseline_capture(self):
        """
        Step 1: Capture baseline Greeks (9:15-9:30 AM)
        Run this once at market open
        """
        
        print("\n" + "═" * 70)
        print("  🕐 DNS SCANNER - BASELINE CAPTURE")
        print("  Time:", datetime.now().strftime("%H:%M:%S"))
        print("═" * 70)
        
        # Check timing
        if not self._is_baseline_time():
            print("⚠️ Not baseline time. Run between 9:15-9:30 AM")
            print(f"   Current time: {datetime.now().strftime('%H:%M:%S')}")
            return
        
        for index in config.INDICES:
            print(f"\n📊 {index}")
            print("─" * 70)
            
            multi = self.dhan.get_multi_expiry_chain(index)
            if not multi:
                print(f"❌ Failed to fetch {index} multi-expiry data")
                continue

            chain = multi["current_chain"]
            next_chain = multi.get("next_chain")
            
            if not chain:
                print(f"❌ Failed to fetch {index} data")
                continue
            
            # Capture baseline
            detector = self.detectors[index]
            ok = detector.capture_baseline(chain)
            if ok and detector.baseline:
                self.excel_logger.log_baseline(index=index, chain=chain, baseline=detector.baseline)
            self.vega_engine.capture_baseline(index=index, current_chain=chain, next_chain=next_chain)
        
        print("\n" + "═" * 70)
        print("✅ BASELINE CAPTURE COMPLETE")
        print(f"⏰ Next: Run analysis at {config.ANALYSIS_TIME}")
        print("═" * 70)
    
    def run_analysis(self, run_phase: str = "analysis"):
        """
        Step 2: Generate signals and validate trades (10:00 AM)
        Compares current Greeks with baseline
        """
        
        print("\n" + "═" * 70)
        if run_phase == "midday":
            print("  🕧 DNS SCANNER - MIDDAY READING")
        else:
            print("  🚀 DNS SCANNER - SIGNAL GENERATION & TRADE VALIDATION")
        print("  Time:", datetime.now().strftime("%H:%M:%S"))
        print("═" * 70)
        
        # Try loading latest baseline from disk for any missing in-memory baseline
        for index in config.INDICES:
            detector = self.detectors[index]
            if detector.baseline is None:
                detector.load_latest_baseline(index)
            if index not in self.vega_engine.baseline:
                self.vega_engine.load_latest_baseline(index)

        # Check if baseline exists
        baseline_exists = all(
            index in self.vega_engine.baseline
            for index in config.INDICES
        )
        
        if not baseline_exists:
            print("❌ No baseline captured. Run baseline_capture first (9:15 AM)")
            return
        
        # Check timing warning for main analysis.
        if run_phase != "midday":
            current_time = datetime.now().time()
            if current_time > time(10, 30):
                print("⚠️ WARNING: Entering late (after 10:30 AM)")
                print("   Theta decay reduces edge - consider waiting for next day")
                print()
        
        for index in config.INDICES:
            self._analyze_index(index, run_phase=run_phase)
    
    def _analyze_index(self, index: str, run_phase: str = "analysis"):
        """
        Analyze a single index:
        1. Fetch current option chain
        2. Generate signal
        3. Validate trades
        """
        
        print(f"\n{'═' * 70}")
        print(f"  📊 {index} ANALYSIS")
        print("═" * 70)
        
        multi = self.dhan.get_multi_expiry_chain(index)
        if not multi:
            print(f"❌ Failed to fetch {index} multi-expiry data")
            return

        # Current expiry chain is used for reference display and audit.
        chain = multi["current_chain"]
        next_chain = multi.get("next_chain")
        
        if not chain:
            print(f"❌ Failed to fetch {index} data")
            return
        
        # Generate multi-expiry averaged signal
        signal = self.vega_engine.generate_signal(
            index=index,
            current_chain=chain,
            next_chain=next_chain,
            market_context={},
        )
        
        if not signal:
            print("❌ Could not generate signal")
            return
        
        # Display signal
        detector = self.detectors[index]
        detector_baseline = detector.baseline or {}
        signal["baseline_timestamp"] = detector_baseline.get("timestamp", "")
        signal["baseline_atm"] = detector_baseline.get("atm", "")
        signal["dual_context"] = self._build_dual_context(index=index, signal=signal)
        if self._apply_intraday_history_adjustment(signal):
            signal["dual_context"] = self._build_dual_context(index=index, signal=signal)
        signal["interpretation"] = self._compose_dual_interpretation(
            signal=signal,
            run_phase=run_phase,
        )
        self._display_signal(signal)
        if run_phase == "midday":
            self._display_shift_from_morning(index=index, current_signal=signal)

        signal_id = self.accuracy_tracker.record_prediction(signal)

        selected_trade = None

        if not signal.get("trade_allowed", True):
            self.excel_logger.log_analysis(
                index=index,
                chain=chain,
                signal=signal,
                signal_id=signal_id,
                confidence_threshold=config.MIN_CONFIDENCE,
                passed_confidence_gate=False,
                trade=None,
                run_phase=run_phase,
            )
            return

        # Hard confidence gate before trade recommendation
        if signal['confidence'] < config.MIN_CONFIDENCE:
            self.excel_logger.log_analysis(
                index=index,
                chain=chain,
                signal=signal,
                signal_id=signal_id,
                confidence_threshold=config.MIN_CONFIDENCE,
                passed_confidence_gate=False,
                trade=None,
                run_phase=run_phase,
            )
            return

        # Midday mode is reading-only. No trade recommendation.
        if run_phase != "midday":
            if bool(getattr(config, "LIVE_SIDEWAYS_ONLY", True)) and signal.get("direction") != "SIDEWAYS":
                self.excel_logger.log_analysis(
                    index=index,
                    chain=chain,
                    signal=signal,
                    signal_id=signal_id,
                    confidence_threshold=config.MIN_CONFIDENCE,
                    passed_confidence_gate=True,
                    trade=None,
                    run_phase=run_phase,
                )
                return

            if bool(getattr(config, "LIVE_REQUIRE_CALM_REGIME", True)) and signal.get("regime") != "CALM":
                self.excel_logger.log_analysis(
                    index=index,
                    chain=chain,
                    signal=signal,
                    signal_id=signal_id,
                    confidence_threshold=config.MIN_CONFIDENCE,
                    passed_confidence_gate=True,
                    trade=None,
                    run_phase=run_phase,
                )
                return

            if (
                signal.get("direction") == "SIDEWAYS"
                and bool(getattr(config, "ENABLE_VOL_SHORT_SAFETY_GATE", False))
            ):
                safe, reason = self._is_vol_short_safe(index=index, current_chain=chain)
                if not safe:
                    self.excel_logger.log_analysis(
                        index=index,
                        chain=chain,
                        signal=signal,
                        signal_id=signal_id,
                        confidence_threshold=config.MIN_CONFIDENCE,
                        passed_confidence_gate=True,
                        trade=None,
                        run_phase=run_phase,
                    )
                    return

            # Your rule: trade only on expiry where ATM straddle >= 1% of spot.
            (
                selected_chain,
                selected_expiry,
                straddle_ratio,
                selected_abs_delta,
                selected_delta_threshold,
                selected_dte,
            ) = self._select_straddle_expiry_chain(index)
            if not selected_chain:
                self.excel_logger.log_analysis(
                    index=index,
                    chain=chain,
                    signal=signal,
                    signal_id=signal_id,
                    confidence_threshold=config.MIN_CONFIDENCE,
                    passed_confidence_gate=True,
                    trade=None,
                    run_phase=run_phase,
                )
                return

            # Analysis-only mode: do not build/print trade setups here.
            # Keep selected_trade as None for logging compatibility.

        self.excel_logger.log_analysis(
            index=index,
            chain=chain,
            signal=signal,
            signal_id=signal_id,
            confidence_threshold=config.MIN_CONFIDENCE,
            passed_confidence_gate=True,
            trade=selected_trade,
            run_phase=run_phase,
        )
        
        print()

    def _is_vol_short_safe(self, index: str, current_chain: Dict) -> tuple[bool, str]:
        """
        Sideways can still be unsafe for short-vol if IV/premium expands abruptly.
        Compare baseline current-expiry ATM against current ATM snapshot.
        """
        baseline_pack = self.vega_engine.baseline.get(index, {})
        base_chain = baseline_pack.get("current") if isinstance(baseline_pack, dict) else None
        if not base_chain:
            return True, "baseline_missing_skip_gate"

        current_atm = int(current_chain.get("atm", 0) or 0)
        if current_atm <= 0:
            return True, "invalid_current_atm_skip_gate"

        current_row = next((s for s in current_chain.get("strikes", []) if int(s["strike"]) == current_atm), None)
        if not current_row:
            return True, "current_atm_row_missing_skip_gate"

        base_rows = {int(s["strike"]): s for s in base_chain.get("strikes", [])}
        base_row = base_rows.get(current_atm)
        if base_row is None:
            base_atm = int(base_chain.get("atm", 0) or 0)
            base_row = base_rows.get(base_atm)
        if not base_row:
            return True, "baseline_atm_row_missing_skip_gate"

        try:
            base_iv = (float(base_row["ce"]["iv"]) + float(base_row["pe"]["iv"])) / 2.0
            now_iv = (float(current_row["ce"]["iv"]) + float(current_row["pe"]["iv"])) / 2.0
            iv_jump = now_iv - base_iv

            base_straddle = float(base_row["ce"]["premium"]) + float(base_row["pe"]["premium"])
            now_straddle = float(current_row["ce"]["premium"]) + float(current_row["pe"]["premium"])
            expansion = ((now_straddle - base_straddle) / base_straddle) if base_straddle > 0 else 0.0
        except (KeyError, TypeError, ValueError):
            return True, "metric_parse_failed_skip_gate"

        if iv_jump > float(getattr(config, "MAX_ATM_IV_JUMP_POINTS", 2.0)):
            return False, f"ATM IV jump {iv_jump:.2f} > {config.MAX_ATM_IV_JUMP_POINTS:.2f}"
        if expansion > float(getattr(config, "MAX_ATM_STRADDLE_EXPANSION_PCT", 0.15)):
            return False, (
                f"ATM straddle expansion {expansion:.2%} > "
                f"{config.MAX_ATM_STRADDLE_EXPANSION_PCT:.2%}"
            )

        return True, "ok"

    def _build_dual_context(self, index: str, signal: Dict) -> Dict:
        """
        Build same-day intraday reference context without altering baseline.
        Uses all prior same-day readings (if available) to derive:
        - anchor from first reading
        - immediate shift from latest prior reading
        - consensus with all prior readings
        - sideways-specific range state labels
        """
        today = datetime.now().strftime("%Y-%m-%d")
        rows = self.accuracy_tracker._read_csv(self.accuracy_tracker.predictions_file)
        prior_rows = [
            row
            for row in rows
            if row.get("index") == index and str(row.get("timestamp", "")).startswith(today)
        ]

        if not prior_rows:
            return {
                "has_10am_reference": False,
                "has_intraday_reference": False,
                "status": "NO_10AM_REFERENCE",
                "range_state": "NO_REFERENCE",
                "prior_count": 0,
            }

        prior_rows.sort(key=lambda r: str(r.get("timestamp", "")))
        anchor = prior_rows[0]
        latest = prior_rows[-1]
        anchor_dir = anchor.get("pred_direction", "")
        latest_dir = latest.get("pred_direction", "")
        try:
            anchor_strength = float(anchor.get("strength", 0.0))
        except (TypeError, ValueError):
            anchor_strength = 0.0
        try:
            latest_strength = float(latest.get("strength", 0.0))
        except (TypeError, ValueError):
            latest_strength = 0.0

        now_dir = signal.get("direction", "")
        now_strength = float(signal.get("strength", 0.0))
        prior_count = len(prior_rows)

        # Short-term status: compare against latest prior reading.
        if now_dir != latest_dir:
            status = "FLIP"
        else:
            latest_abs = abs(latest_strength)
            now_abs = abs(now_strength)
            if latest_abs == 0:
                status = "UNCHANGED"
            elif now_abs < latest_abs * 0.8:
                status = "WEAKENING"
            elif now_abs > latest_abs * 1.2:
                status = "STRENGTHENING"
            else:
                status = "UNCHANGED"

        # Consensus with all prior readings.
        dir_counts: Dict[str, int] = {"BULLISH": 0, "BEARISH": 0, "SIDEWAYS": 0}
        for row in prior_rows:
            d = str(row.get("pred_direction", ""))
            if d in dir_counts:
                dir_counts[d] += 1

        same_as_now = dir_counts.get(str(now_dir), 0)
        agreement_ratio = (same_as_now / prior_count) if prior_count else 0.0

        prior_strengths: List[float] = []
        for row in prior_rows:
            try:
                prior_strengths.append(abs(float(row.get("strength", 0.0))))
            except (TypeError, ValueError):
                continue
        avg_prior_abs = (sum(prior_strengths) / len(prior_strengths)) if prior_strengths else 0.0
        now_abs = abs(now_strength)

        # Sideways-specific intraday regime labels.
        range_state = "NA"
        if now_dir == "SIDEWAYS":
            prior_sideways_ratio = (dir_counts["SIDEWAYS"] / prior_count) if prior_count else 0.0
            if prior_sideways_ratio < 0.60:
                range_state = "RANGE_BREAK_RISK"
            elif avg_prior_abs > 0 and now_abs < avg_prior_abs * 0.8:
                range_state = "RANGE_TIGHTENING"
            elif avg_prior_abs > 0 and now_abs > avg_prior_abs * 1.2:
                range_state = "RANGE_WIDENING"
            else:
                range_state = "RANGE_STABLE"


        return {
            "has_10am_reference": True,
            "has_intraday_reference": True,
            "anchor_timestamp": anchor.get("timestamp", ""),
            "anchor_direction": anchor_dir,
            "anchor_strength": round(anchor_strength, 2),
            "latest_timestamp": latest.get("timestamp", ""),
            "latest_direction": latest_dir,
            "latest_strength": round(latest_strength, 2),
            "prior_count": prior_count,
            "agreement_ratio": round(agreement_ratio, 4),
            "status": status,
            "range_state": range_state,
        }

    def _apply_intraday_history_adjustment(self, signal: Dict) -> bool:
        """
        Promote SIDEWAYS to directional when intraday history strongly supports
        continuation and baseline-to-now move is already directional.
        """
        if signal.get("direction") != "SIDEWAYS":
            return False

        dual = signal.get("dual_context", {}) or {}
        if not dual.get("has_intraday_reference"):
            return False

        try:
            prior_count = int(dual.get("prior_count", 0))
        except (TypeError, ValueError):
            prior_count = 0
        latest_dir = str(dual.get("latest_direction", ""))
        try:
            agreement_ratio = float(dual.get("agreement_ratio", 0.0))
        except (TypeError, ValueError):
            agreement_ratio = 0.0

        min_prior_reads = int(getattr(config, "INTRADAY_OVERRIDE_MIN_PRIOR_READS", 3))
        if prior_count < min_prior_reads:
            return False
        if latest_dir not in {"BULLISH", "BEARISH"}:
            return False
        # High sideways agreement means prior context is still range-like.
        max_sideways_agreement = float(
            getattr(config, "INTRADAY_OVERRIDE_MAX_SIDEWAYS_AGREEMENT", 0.40)
        )
        if agreement_ratio > max_sideways_agreement:
            return False

        raw = signal.get("analytics_features", {}).get("raw", {}) or {}
        spot_change_pct = float(signal.get("spot_change_pct", 0.0) or 0.0)
        flow_delta = float(raw.get("flow_delta_true", 0.0) or 0.0)
        flow_vega = float(raw.get("flow_vega_true", 0.0) or 0.0)

        move_threshold_pct = float(getattr(config, "INTRADAY_DIRECTIONAL_SPOT_MOVE_PCT", 0.35))
        flow_floor_ratio = float(getattr(config, "INTRADAY_OVERRIDE_FLOW_FLOOR_RATIO", 0.20))
        vega_floor_ratio = float(getattr(config, "INTRADAY_OVERRIDE_VEGA_FLOOR_RATIO", 0.20))
        directional_flow_floor = float(getattr(config, "RULE_DIRECTIONAL_FLOW_MIN", 500000.0)) * flow_floor_ratio
        directional_vega_floor = float(getattr(config, "RULE_DIRECTIONAL_VEGA_MIN", 2000000.0)) * vega_floor_ratio

        if latest_dir == "BULLISH":
            move_ok = spot_change_pct >= move_threshold_pct
            flow_ok = flow_delta >= directional_flow_floor or flow_vega >= directional_vega_floor
        else:
            move_ok = spot_change_pct <= -move_threshold_pct
            flow_ok = flow_delta <= -directional_flow_floor or flow_vega <= -directional_vega_floor

        if not (move_ok and flow_ok):
            return False

        signal["direction"] = latest_dir
        signal["directional_bias"] = "INTRADAY_CONTINUATION"
        signal["strategy_bias"] = "DIRECTIONAL"
        signal["intraday_override"] = True

        existing = str(signal.get("interpretation", "")).strip()
        note = (
            f"Intraday continuation override applied ({prior_count} prior reads, "
            f"latest={latest_dir}, spot_change_pct={spot_change_pct:+.2f}%)."
        )
        signal["interpretation"] = f"{existing} | {note}" if existing else note
        return True

    def _compose_dual_interpretation(self, signal: Dict, run_phase: str) -> str:
        """
        Final interpretation using both:
        1) Open baseline -> now shift (already in signal direction/strength)
        2) Intraday anchors -> now confirmation/weakening
        """
        base_text = signal.get("interpretation", "")
        dual = signal.get("dual_context", {})

        if not dual.get("has_intraday_reference"):
            return (
                f"{base_text} | Open-baseline signal active. "
                "No prior intraday reference yet, so this is baseline-only."
            )

        status = dual.get("status", "UNCHANGED")
        latest_dir = dual.get("latest_direction", "")
        latest_strength = float(dual.get("latest_strength", 0.0))
        now_dir = signal.get("direction", "")
        now_strength = float(signal.get("strength", 0.0))
        prior_count = int(dual.get("prior_count", 0))
        agreement_ratio = float(dual.get("agreement_ratio", 0.0))
        range_state = str(dual.get("range_state", "NA"))

        if now_dir == "SIDEWAYS":
            confirm_text = (
                f"Intraday references: {prior_count}. "
                f"Range state = {range_state}. "
                f"Agreement with prior reads = {agreement_ratio:.0%}."
            )
            return f"{base_text} | {confirm_text}" if run_phase == "analysis" else f"{base_text} | Midday check: {confirm_text}"

        if status == "FLIP":
            confirm_text = (
                f"Latest prior reference was {latest_dir} ({latest_strength:+.2f}), "
                f"now {now_dir} ({now_strength:+.2f}) -> momentum flip; reduce aggression."
            )
        elif status == "STRENGTHENING":
            confirm_text = (
                f"Latest prior reference {latest_dir} ({latest_strength:+.2f}) -> "
                f"now stronger ({now_strength:+.2f}); continuation confirmed. "
                f"Intraday agreement: {agreement_ratio:.0%} across {prior_count} prior reads."
            )
        elif status == "WEAKENING":
            confirm_text = (
                f"Latest prior reference {latest_dir} ({latest_strength:+.2f}) -> "
                f"now weaker ({now_strength:+.2f}); momentum fading. "
                f"Intraday agreement: {agreement_ratio:.0%} across {prior_count} prior reads."
            )
        else:
            confirm_text = (
                f"Latest prior reference {latest_dir} ({latest_strength:+.2f}) -> "
                f"now similar ({now_strength:+.2f}); structure stable. "
                f"Intraday agreement: {agreement_ratio:.0%} across {prior_count} prior reads."
            )

        if run_phase == "analysis":
            return f"{base_text} | {confirm_text}"
        return f"{base_text} | Midday check: {confirm_text}"

    def _display_shift_from_morning(self, index: str, current_signal: Dict):
        """Compare midday signal with latest same-day signal and show shift status."""
        today = datetime.now().strftime("%Y-%m-%d")
        prev = self.accuracy_tracker.get_latest_prediction(index=index, trade_date=today)
        if not prev:
            print("\nℹ️ No earlier same-day signal found for shift comparison.")
            return

        prev_dir = prev.get("pred_direction", "")
        try:
            prev_strength = float(prev.get("strength", 0))
        except (TypeError, ValueError):
            prev_strength = 0.0

        now_dir = current_signal["direction"]
        now_strength = float(current_signal["strength"])

        if prev_dir != now_dir:
            status = "FLIP"
        else:
            prev_abs = abs(prev_strength)
            now_abs = abs(now_strength)
            if prev_abs == 0:
                status = "UNCHANGED"
            elif now_abs < prev_abs * 0.8:
                status = "WEAKENING"
            elif now_abs > prev_abs * 1.2:
                status = "STRENGTHENING"
            else:
                status = "UNCHANGED"

        print("\n" + "-" * 70)
        print(f"  MIDDAY SHIFT CHECK: {status}")
        print(f"  Earlier: {prev_dir} ({prev_strength:+.2f})")
        print(f"  Current: {now_dir} ({now_strength:+.2f})")
        print("-" * 70)

    def _select_straddle_expiry_chain(self, index: str):
        """
        Select expiry for straddle using practical filters:
        1) ATM straddle premium must be >= configured ratio.
        2) Among qualified expiries, prefer ones passing DTE-aware delta neutrality.
        3) Then minimize |ATM straddle delta| for better neutrality.

        Returns:
            (chain, expiry, ratio, abs_delta, threshold, dte)
            or (None, None, 0.0, 0.0, 0.0, None)
        """
        expiries = self.dhan.get_expiry_list(index)
        if not expiries:
            return None, None, 0.0, 0.0, 0.0, None

        best = None

        for expiry in expiries[: config.MAX_EXPIRIES_TO_SCAN]:
            # Reuse chain data fetched earlier in this run to avoid API throttling.
            chain = self.dhan.get_option_chain_for_expiry(index, expiry, use_cache=True)
            if not chain:
                continue
            atm = chain["atm"]
            spot = chain["spot"]
            atm_row = next((s for s in chain["strikes"] if s["strike"] == atm), None)
            if not atm_row or spot <= 0:
                continue

            straddle = atm_row["ce"]["premium"] + atm_row["pe"]["premium"]
            ratio = straddle / spot
            if ratio < config.MIN_STRADDLE_PREMIUM_RATIO:
                continue

            combined_delta_per_contract = float(atm_row["ce"]["delta"]) + float(atm_row["pe"]["delta"])
            abs_delta = abs(combined_delta_per_contract)

            try:
                dte = max((datetime.strptime(expiry, "%Y-%m-%d").date() - datetime.now().date()).days, 0)
            except Exception:
                dte = None

            threshold = self.validator.get_delta_neutral_threshold(dte)
            is_within_threshold = abs_delta < threshold

            candidate = {
                "chain": chain,
                "expiry": expiry,
                "ratio": ratio,
                "abs_delta": abs_delta,
                "threshold": threshold,
                "dte": dte,
                "is_within_threshold": is_within_threshold,
            }

            if best is None:
                best = candidate
                continue

            # Rank: threshold-pass first, then lowest |delta|, then richer premium ratio.
            current_rank = (
                0 if candidate["is_within_threshold"] else 1,
                candidate["abs_delta"],
                -candidate["ratio"],
            )
            best_rank = (
                0 if best["is_within_threshold"] else 1,
                best["abs_delta"],
                -best["ratio"],
            )
            if current_rank < best_rank:
                best = candidate

        if not best:
            return None, None, 0.0, 0.0, 0.0, None

        return (
            best["chain"],
            best["expiry"],
            best["ratio"],
            best["abs_delta"],
            best["threshold"],
            best["dte"],
        )

    def print_accuracy_summary(self):
        stats = self.accuracy_tracker.summary()
        gap_stats = self.accuracy_tracker.gap_summary()
        print("\n" + "=" * 70)
        print("  ACCURACY SUMMARY")
        print("=" * 70)
        print(f"Total predictions: {stats['total_predictions']}")
        print(f"Labeled predictions: {stats['labeled_predictions']}")
        print(f"Exact accuracy: {stats['exact_accuracy']:.2%}")
        print(f"Structure accuracy: {stats['structure_accuracy']:.2%}")
        print(f"Directional accuracy: {stats['directional_accuracy']:.2%}")
        print("-" * 70)
        print(f"Total gap predictions: {gap_stats['total_gap_predictions']}")
        print(f"Labeled gap predictions: {gap_stats['labeled_gap_predictions']}")
        print(f"Gap exact accuracy: {gap_stats['gap_exact_accuracy']:.2%}")
        print("=" * 70)

    def _display_signal(self, signal: Dict):
        """Display signal in replay-style baseline/previous/current format."""

        def _to_float(value, default=None):
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def _fmt(value, decimals: int = 4, signed: bool = True, as_int: bool = False, suffix: str = "") -> str:
            if value is None:
                return "NA"
            if as_int:
                iv = int(round(value))
                text = f"{iv:+d}" if signed else f"{iv:d}"
            else:
                text = f"{value:+.{decimals}f}" if signed else f"{value:.{decimals}f}"
            return f"{text}{suffix}"

        def _fmt_diff(now_v, ref_v, decimals: int = 4, as_int: bool = False, suffix: str = "") -> str:
            if now_v is None or ref_v is None:
                return "NA"
            return _fmt(now_v - ref_v, decimals=decimals, signed=True, as_int=as_int, suffix=suffix)

        def _print_table(title: str, rows: List[Dict], first_ref_label: str, last_ref_label: str) -> None:
            print(f"\n{title}")
            print("-" * 122)
            print(
                f"{'Metric':<24} {'Baseline(09:20)':>16} {first_ref_label:>14} {'Current':>14} "
                f"{'Δ from 1st':>14} {'Δ from last':>14}"
            )
            print("-" * 122)
            for row in rows:
                print(
                    f"{row['metric']:<24} {row['baseline']:>16} {row['first']:>14} {row['now']:>14} "
                    f"{row['now_first']:>14} {row['now_prev']:>14}"
                )
            print("-" * 122)
            print(
                f"Δ from 1st = Current - {first_ref_label} | "
                f"Δ from last = Current - {last_ref_label}"
            )

        raw = signal.get("analytics_features", {}).get("raw", {}) or {}
        expiry_metrics = signal.get("expiry_metrics", {}) or {}
        current_metrics = expiry_metrics.get("current", {}) or {}
        current_debug = current_metrics.get("debug", {}) or {}
        next_metrics = expiry_metrics.get("next", {}) or {}
        next_debug = next_metrics.get("debug", {}) if isinstance(next_metrics, dict) else {}

        now_spot = float(signal.get("spot", 0.0) or 0.0)
        base_spot = float(signal.get("baseline_spot", 0.0) or 0.0)
        spot_change = now_spot - base_spot
        spot_change_pct = float(signal.get("spot_change_pct", 0.0) or 0.0)

        now_call_oi = float(raw.get("total_call_oi", 0.0) or 0.0)
        now_put_oi = float(raw.get("total_put_oi", 0.0) or 0.0)
        call_oi_change = float(raw.get("total_call_oi_change", 0.0) or 0.0)
        put_oi_change = float(raw.get("total_put_oi_change", 0.0) or 0.0)
        base_call_oi = now_call_oi - call_oi_change
        base_put_oi = now_put_oi - put_oi_change

        base_pos_delta = 0.0
        if current_debug:
            base_pos_delta = float(current_debug.get("pos_delta_baseline", 0.0) or 0.0)
            if next_debug:
                base_pos_delta = (
                    base_pos_delta + float(next_debug.get("pos_delta_baseline", 0.0) or 0.0)
                ) / 2.0

        base_total_oi = base_call_oi + base_put_oi
        base_imb = ((base_put_oi - base_call_oi) / base_total_oi) if base_total_oi > 0 else 0.0

        source_expiry = str(current_metrics.get("now_expiry", "") or "NA")
        if next_metrics and next_metrics.get("now_expiry"):
            source = f"live-multi-expiry({source_expiry}+{next_metrics.get('now_expiry')})"
        else:
            source = f"live({source_expiry})"

        today = datetime.now().strftime("%Y-%m-%d")
        first = self.accuracy_tracker.get_first_detailed_prediction(
            index=str(signal.get("index", "")),
            trade_date=today,
        )
        prev = self.accuracy_tracker.get_latest_detailed_prediction(
            index=str(signal.get("index", "")),
            trade_date=today,
        )
        first_label = "1st Read"
        prev_label = "Prev"
        first_spot = None
        first_flow_delta = None
        first_flow_vega = None
        first_gamma_shift = None
        first_gamma_skew_shift = None
        first_pos_delta = None
        first_spot_change_pct = None
        first_call_oi = None
        first_put_oi = None
        first_call_oi_change = None
        first_put_oi_change = None
        first_oi_imbalance = None
        first_call_iv_diff = None
        first_put_iv_diff = None
        first_total_iv_change = None
        prev_spot = None
        prev_flow_delta = None
        prev_flow_vega = None
        prev_gamma_shift = None
        prev_gamma_skew_shift = None
        prev_pos_delta = None
        prev_spot_change_pct = None
        prev_call_oi = None
        prev_put_oi = None
        prev_call_oi_change = None
        prev_put_oi_change = None
        prev_oi_imbalance = None
        prev_call_iv_diff = None
        prev_put_iv_diff = None
        prev_total_iv_change = None
        if first:
            first_ts = str(first.get("timestamp", ""))
            try:
                first_label = f"1st {datetime.strptime(first_ts, '%Y-%m-%d %H:%M:%S').strftime('%H:%M')}"
            except ValueError:
                first_label = "1st Read"
            first_spot = _to_float(first.get("spot"))
            first_flow_delta = _to_float(first.get("flow_delta_true"))
            first_flow_vega = _to_float(first.get("flow_vega_true"))
            first_gamma_shift = _to_float(first.get("gamma_shift_true"))
            first_gamma_skew_shift = _to_float(first.get("gamma_skew_shift_true"))
            first_pos_delta = _to_float(first.get("pos_delta_now"))
            first_spot_change_pct = _to_float(first.get("spot_change_pct"))
            first_call_oi = _to_float(first.get("total_call_oi"))
            first_put_oi = _to_float(first.get("total_put_oi"))
            first_call_oi_change = _to_float(first.get("total_call_oi_change"))
            first_put_oi_change = _to_float(first.get("total_put_oi_change"))
            first_oi_imbalance = _to_float(first.get("oi_imbalance"))
            first_call_iv_diff = _to_float(first.get("call_iv_diff"))
            first_put_iv_diff = _to_float(first.get("put_iv_diff"))
            first_total_iv_change = _to_float(first.get("total_iv_change"))
        if prev:
            prev_ts = str(prev.get("timestamp", ""))
            try:
                prev_label = f"Prev {datetime.strptime(prev_ts, '%Y-%m-%d %H:%M:%S').strftime('%H:%M')}"
            except ValueError:
                prev_label = "Prev"
            prev_spot = _to_float(prev.get("spot"))
            prev_flow_delta = _to_float(prev.get("flow_delta_true"))
            prev_flow_vega = _to_float(prev.get("flow_vega_true"))
            prev_gamma_shift = _to_float(prev.get("gamma_shift_true"))
            prev_gamma_skew_shift = _to_float(prev.get("gamma_skew_shift_true"))
            prev_pos_delta = _to_float(prev.get("pos_delta_now"))
            prev_spot_change_pct = _to_float(prev.get("spot_change_pct"))
            prev_call_oi = _to_float(prev.get("total_call_oi"))
            prev_put_oi = _to_float(prev.get("total_put_oi"))
            prev_call_oi_change = _to_float(prev.get("total_call_oi_change"))
            prev_put_oi_change = _to_float(prev.get("total_put_oi_change"))
            prev_oi_imbalance = _to_float(prev.get("oi_imbalance"))
            prev_call_iv_diff = _to_float(prev.get("call_iv_diff"))
            prev_put_iv_diff = _to_float(prev.get("put_iv_diff"))
            prev_total_iv_change = _to_float(prev.get("total_iv_change"))

        print("\n========================================================================")
        print(f"DATE: {datetime.now().strftime('%Y-%m-%d')}")
        print(f"INDEX: {signal.get('index', 'NA')}")
        print(f"SOURCE: {source}")
        print("========================================================================")

        now_label = datetime.now().strftime("%H:%M")
        print(f"\nTIME: {now_label}")
        print(f"Spot: {now_spot:.2f}")
        print(f"Vega State: {signal.get('vega_state', 'NA')}")
        print(f"IV State: {signal.get('iv_state', 'NA')}")
        print(f"Delta State: {signal.get('delta_state', 'NA')}")
        print(f"Gamma Stress: {signal.get('gamma_stress', 'NA')}")
        print(f"Confidence: {float(signal.get('confidence', 0.0)):.2f}")
        print(f"Regime: {signal.get('regime', 'NA')}")
        print(f"Direction: {signal.get('direction', 'NA')}")
        print(f"Strategy Bias: {signal.get('strategy_bias', 'NA')}")
        print(f"Interpretation: {signal.get('interpretation', '')}")
        print(f"Spot Change (09:20 -> {now_label}): {spot_change:+.2f} ({spot_change_pct:+.4f}%)")

        now_flow_delta = _to_float(raw.get("flow_delta_true"), 0.0)
        now_flow_vega = _to_float(raw.get("flow_vega_true"), 0.0)
        now_gamma_shift = _to_float(raw.get("gamma_shift_true"), 0.0)
        now_gamma_skew_shift = _to_float(raw.get("gamma_skew_shift_true"), 0.0)
        now_pos_delta = _to_float(raw.get("pos_delta_now"), 0.0)
        now_oi_imbalance = _to_float(raw.get("oi_imbalance"), 0.0)
        now_call_iv_diff = _to_float(raw.get("call_iv_diff"), 0.0)
        now_put_iv_diff = _to_float(raw.get("put_iv_diff"), 0.0)
        now_total_iv_change = _to_float(raw.get("total_iv_change"), 0.0)

        prev_iv_spread = (
            prev_call_iv_diff - prev_put_iv_diff
            if prev_call_iv_diff is not None and prev_put_iv_diff is not None
            else None
        )
        first_iv_spread = (
            first_call_iv_diff - first_put_iv_diff
            if first_call_iv_diff is not None and first_put_iv_diff is not None
            else None
        )
        now_iv_spread = now_call_iv_diff - now_put_iv_diff

        vega_delta_iv_rows = [
            {"metric": "flow_delta_true", "baseline": _fmt(0.0), "first": _fmt(first_flow_delta), "now": _fmt(now_flow_delta), "now_first": _fmt_diff(now_flow_delta, first_flow_delta), "now_prev": _fmt_diff(now_flow_delta, prev_flow_delta)},
            {"metric": "flow_vega_true", "baseline": _fmt(0.0), "first": _fmt(first_flow_vega), "now": _fmt(now_flow_vega), "now_first": _fmt_diff(now_flow_vega, first_flow_vega), "now_prev": _fmt_diff(now_flow_vega, prev_flow_vega)},
            {"metric": "call_iv_diff", "baseline": _fmt(0.0), "first": _fmt(first_call_iv_diff), "now": _fmt(now_call_iv_diff), "now_first": _fmt_diff(now_call_iv_diff, first_call_iv_diff), "now_prev": _fmt_diff(now_call_iv_diff, prev_call_iv_diff)},
            {"metric": "put_iv_diff", "baseline": _fmt(0.0), "first": _fmt(first_put_iv_diff), "now": _fmt(now_put_iv_diff), "now_first": _fmt_diff(now_put_iv_diff, first_put_iv_diff), "now_prev": _fmt_diff(now_put_iv_diff, prev_put_iv_diff)},
            {"metric": "total_iv_change", "baseline": _fmt(0.0), "first": _fmt(first_total_iv_change), "now": _fmt(now_total_iv_change), "now_first": _fmt_diff(now_total_iv_change, first_total_iv_change), "now_prev": _fmt_diff(now_total_iv_change, prev_total_iv_change)},
            {"metric": "iv_spread(c-p)", "baseline": _fmt(0.0), "first": _fmt(first_iv_spread), "now": _fmt(now_iv_spread), "now_first": _fmt_diff(now_iv_spread, first_iv_spread), "now_prev": _fmt_diff(now_iv_spread, prev_iv_spread)},
        ]
        _print_table("VEGA / DELTA / IV COMPARISON", vega_delta_iv_rows, first_label, prev_label)

        full_rows = [
            {"metric": "spot", "baseline": _fmt(base_spot, decimals=2, signed=False), "first": _fmt(first_spot, decimals=2, signed=False), "now": _fmt(now_spot, decimals=2, signed=False), "now_first": _fmt_diff(now_spot, first_spot, decimals=2), "now_prev": _fmt_diff(now_spot, prev_spot, decimals=2)},
            {"metric": "spot_change_pct", "baseline": _fmt(0.0, suffix="%"), "first": _fmt(first_spot_change_pct, suffix="%"), "now": _fmt(spot_change_pct, suffix="%"), "now_first": _fmt_diff(spot_change_pct, first_spot_change_pct, suffix="%"), "now_prev": _fmt_diff(spot_change_pct, prev_spot_change_pct, suffix="%")},
            {"metric": "gamma_shift_true", "baseline": _fmt(0.0), "first": _fmt(first_gamma_shift), "now": _fmt(now_gamma_shift), "now_first": _fmt_diff(now_gamma_shift, first_gamma_shift), "now_prev": _fmt_diff(now_gamma_shift, prev_gamma_shift)},
            {"metric": "gamma_skew_shift", "baseline": _fmt(0.0), "first": _fmt(first_gamma_skew_shift), "now": _fmt(now_gamma_skew_shift), "now_first": _fmt_diff(now_gamma_skew_shift, first_gamma_skew_shift), "now_prev": _fmt_diff(now_gamma_skew_shift, prev_gamma_skew_shift)},
            {"metric": "pos_delta_now", "baseline": _fmt(base_pos_delta), "first": _fmt(first_pos_delta), "now": _fmt(now_pos_delta), "now_first": _fmt_diff(now_pos_delta, first_pos_delta), "now_prev": _fmt_diff(now_pos_delta, prev_pos_delta)},
            {"metric": "total_call_oi", "baseline": _fmt(base_call_oi, signed=False, as_int=True), "first": _fmt(first_call_oi, signed=False, as_int=True), "now": _fmt(now_call_oi, signed=False, as_int=True), "now_first": _fmt_diff(now_call_oi, first_call_oi, as_int=True), "now_prev": _fmt_diff(now_call_oi, prev_call_oi, as_int=True)},
            {"metric": "total_put_oi", "baseline": _fmt(base_put_oi, signed=False, as_int=True), "first": _fmt(first_put_oi, signed=False, as_int=True), "now": _fmt(now_put_oi, signed=False, as_int=True), "now_first": _fmt_diff(now_put_oi, first_put_oi, as_int=True), "now_prev": _fmt_diff(now_put_oi, prev_put_oi, as_int=True)},
            {"metric": "call_oi_change", "baseline": _fmt(0.0, as_int=True), "first": _fmt(first_call_oi_change, as_int=True), "now": _fmt(call_oi_change, as_int=True), "now_first": _fmt_diff(call_oi_change, first_call_oi_change, as_int=True), "now_prev": _fmt_diff(call_oi_change, prev_call_oi_change, as_int=True)},
            {"metric": "put_oi_change", "baseline": _fmt(0.0, as_int=True), "first": _fmt(first_put_oi_change, as_int=True), "now": _fmt(put_oi_change, as_int=True), "now_first": _fmt_diff(put_oi_change, first_put_oi_change, as_int=True), "now_prev": _fmt_diff(put_oi_change, prev_put_oi_change, as_int=True)},
            {"metric": "oi_imbalance", "baseline": _fmt(base_imb), "first": _fmt(first_oi_imbalance), "now": _fmt(now_oi_imbalance), "now_first": _fmt_diff(now_oi_imbalance, first_oi_imbalance), "now_prev": _fmt_diff(now_oi_imbalance, prev_oi_imbalance)},
        ]
        _print_table("FULL METRICS COMPARISON", full_rows, first_label, prev_label)
    def _is_baseline_time(self) -> bool:
        """Check if current time is baseline capture time"""
        now = datetime.now().time()
        start = time(9, 15)
        end = time(9, 30)
        return start <= now <= end


def analyze_for_trade(index: str):
    """
    Complete analysis flow matching Mudit's methodology
    """

    # Step 1: Get multi-expiry option chain (current + next + monthly)
    print("\n📊 Fetching multi-expiry option chain...")
    api = get_dhan()
    multi_chain = api.get_multi_expiry_chain(index)

    if not multi_chain:
        print("❌ Cannot fetch option chain")
        return

    chain_for_calc = multi_chain["current_chain"] if "current_chain" in multi_chain else multi_chain

    # Step 2: Check 1% straddle premium threshold
    validator = TradeValidator()
    if not validator.validate_expiry_premium_threshold(chain_for_calc):
        print("\n⚠️ Current expiry rejected - switching to next expiry...")
        # Fetch next expiry and re-run
        return

    # Step 3: Calculate Vega/Theta decay (Mudit's method)
    detector = SignalDetector()
    detector.capture_baseline(chain_for_calc)
    vega_theta_decay = detector.calculate_vega_theta_decay(chain_for_calc)

    print(f"\n📈 Vega/Theta Decay:")
    print(f"   Put Vega: {vega_theta_decay['put_vega']:+.2f}")
    print(f"   Call Vega: {vega_theta_decay['call_vega']:+.2f}")
    print(f"   Put Theta: {vega_theta_decay['put_theta']:+.2f}")
    print(f"   Call Theta: {vega_theta_decay['call_theta']:+.2f}")

    # Step 4: Calculate Strength
    delta_flow = detector._calculate_delta_flow(chain_for_calc)
    strength = detector.calculate_strength(vega_theta_decay, delta_flow)

    print(f"\n💪 Strength: {strength:+.2f}")

    if strength > 10:
        print("   🟢 BULLISH")
    elif strength < -10:
        print("   🔴 BEARISH")
    else:
        print("   🟡 SIDEWAYS")

    # Step 5: Generate signal and validate trade
    signal = detector.generate_signal(chain_for_calc)
    if not signal:
        print("❌ Could not generate signal")
        return

    if signal['confidence'] < config.MIN_CONFIDENCE:
        return

    validator.validate_directional_trade(chain_for_calc, signal)

    # Step 6: Check if overnight gap prediction aligns
    predictor = GapPredictor()
    gap_pred = predictor.predict_gap(index)

    if gap_pred:
        overnight_safe = predictor.validate_overnight_position(
            gap_prediction=gap_pred,
            current_strength=strength,
            current_vega_theta=vega_theta_decay
        )

        if overnight_safe:
            print("\n✅ SAFE FOR OVERNIGHT POSITION")
        else:
            print("\n⚠️ CLOSE BEFORE 3:30 PM - Don't carry overnight")


def main():
    """Main entry point with command-line interface"""
    
    import sys
    
    scanner = DNSScanner()
    
    if len(sys.argv) < 2:
        print("\n" + "═" * 70)
        print("  📊 DNS OPTIONS SCANNER")
        print("═" * 70)
        print("\nUsage:")
        print("  python main.py baseline   → Capture baseline (run at 9:15 AM)")
        print("  python main.py analyze    → Generate signals (run at 10:00 AM)")
        print("  python main.py midday     → Midday reading refresh (11:30/13:30)")
        print("  python main.py accuracy   → Show tracked accuracy stats")
        print("\nWorkflow:")
        print("  1. At 9:15 AM: python main.py baseline")
        print("  2. At 10:00 AM: python main.py analyze")
        print("  3. Midday: python main.py midday")
        print("  4. After market close: run post_market_labeler.py")
        print("═" * 70)
        return
    
    command = sys.argv[1].lower()
    
    if command == 'baseline':
        scanner.run_baseline_capture()
    
    elif command == 'analyze':
        scanner.run_analysis()

    elif command == 'midday':
        scanner.run_analysis(run_phase="midday")

    elif command == 'accuracy':
        scanner.print_accuracy_summary()
    
    else:
        print(f"❌ Unknown command: {command}")
        print("Use: baseline, analyze, midday, or accuracy")


if __name__ == '__main__':
    main()
