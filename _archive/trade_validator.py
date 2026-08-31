"""
Trade Validator
Validates trades using DNS Edge Ratio methodology
Ensures theta/risk ratio is acceptable before recommending trades
"""

from datetime import datetime
from typing import Dict, List, Optional
import config
from greeks import GreeksCalculator
from greeks import RiskCalculator


class TradeValidator:
    """
    Validates trades based on Edge Ratio and risk metrics
    """

    def get_delta_neutral_threshold(self, days_to_expiry: Optional[int]) -> float:
        """
        Dynamic delta-neutral threshold by DTE.
        Falls back to MAX_DELTA_NEUTRAL if DTE is unavailable.
        """
        if days_to_expiry is None:
            return config.MAX_DELTA_NEUTRAL

        schedule = getattr(config, "DELTA_NEUTRAL_DTE_THRESHOLDS", None)
        if not schedule:
            return config.MAX_DELTA_NEUTRAL

        for max_dte, threshold in schedule:
            if days_to_expiry <= max_dte:
                return float(threshold)

        return config.MAX_DELTA_NEUTRAL

    def _days_to_expiry(self, expiry: Optional[str]) -> Optional[int]:
        if not expiry:
            return None
        try:
            exp = datetime.strptime(expiry, "%Y-%m-%d").date()
            return max((exp - datetime.now().date()).days, 0)
        except Exception:
            return None

    def validate_expiry_premium_threshold(self, option_chain: Dict) -> bool:
        """
        Check if ATM straddle premium >= 1% of spot
        If not, reject this expiry and suggest next expiry
        """

        atm = option_chain['atm']
        spot = option_chain['spot']

        # Find ATM straddle premium
        atm_strike = next((s for s in option_chain['strikes'] if s['strike'] == atm), None)

        if not atm_strike:
            return False

        straddle_premium = atm_strike['ce']['premium'] + atm_strike['pe']['premium']
        threshold = spot * 0.01  # 1% of spot

        print(f"\n📊 Straddle Premium Check:")
        print(f"   ATM {atm} Straddle: ₹{straddle_premium:.2f}")
        print(f"   Threshold (1% of spot): ₹{threshold:.2f}")

        if straddle_premium < threshold:
            print(f"   ❌ REJECT - Premium too low, switch to next expiry")
            return False
        else:
            print(f"   ✅ ACCEPT - Premium sufficient")
            return True
    
    def validate_directional_trade(
        self,
        option_chain: Dict,
        signal: Dict
    ) -> Optional[Dict]:
        """
        Validate a directional trade (selling puts for bullish, calls for bearish)
        
        Returns best strike to sell, or None if no good trades
        """
        
        direction = signal['direction']
        
        if direction == 'BULLISH':
            # Sell OTM puts
            candidates = self._get_otm_put_strikes(option_chain)
            option_type = 'pe'
        elif direction == 'BEARISH':
            # Sell OTM calls
            candidates = self._get_otm_call_strikes(option_chain)
            option_type = 'ce'
        else:
            return None  # Not directional
        
        # Validate each candidate
        validated_trades = []
        
        for strike_data in candidates:
            validation = self._validate_single_strike(
                strike_data=strike_data,
                option_type=option_type,
                spot=option_chain['spot'],
                index=option_chain['index']
            )
            
            if validation and validation['edge_ratio'] >= config.MIN_EDGE_RATIO:
                validated_trades.append(validation)
        
        if not validated_trades:
            return None
        
        # Return best trade (highest edge ratio)
        best_trade = max(validated_trades, key=lambda x: x['edge_ratio'])
        
        return {
            'trade_type': 'DIRECTIONAL',
            'direction': direction,
            'best_strike': best_trade,
            'all_candidates': validated_trades
        }
    
    def validate_delta_neutral_trade(
        self,
        option_chain: Dict,
        delta_threshold: Optional[float] = None,
    ) -> Optional[Dict]:
        """
        Validate delta-neutral trades (straddles/strangles)
        For sideways markets
        
        Returns best delta-neutral setup
        """
        
        atm = option_chain['atm']
        index = option_chain['index']
        spot = option_chain['spot']
        lot_size = config.LOT_SIZES[index]
        
        # Find ATM strike data
        atm_strike = None
        for s in option_chain['strikes']:
            if s['strike'] == atm:
                atm_strike = s
                break
        
        if not atm_strike:
            return None
        
        # Validate ATM straddle (sell both ATM call and put)
        ce_greeks = atm_strike['ce']
        pe_greeks = atm_strike['pe']
        
        # Combined position Greeks
        # Keep both per-contract and lot-scaled delta to avoid unit mismatch in neutrality check.
        combined_delta_per_contract = ce_greeks['delta'] + pe_greeks['delta']
        combined_delta = combined_delta_per_contract * lot_size
        combined_gamma = (ce_greeks['gamma'] + pe_greeks['gamma']) * lot_size
        combined_vega = (ce_greeks['vega'] + pe_greeks['vega']) * lot_size
        combined_theta = (ce_greeks['theta'] + pe_greeks['theta']) * lot_size
        combined_premium = (ce_greeks['premium'] + pe_greeks['premium']) * lot_size
        
        expiry = option_chain.get("expiry")
        days_to_expiry = self._days_to_expiry(expiry)
        applied_threshold = (
            float(delta_threshold)
            if delta_threshold is not None
            else self.get_delta_neutral_threshold(days_to_expiry)
        )

        # Check delta neutrality
        is_neutral = abs(combined_delta_per_contract) < applied_threshold

        if not is_neutral:
            print(
                "⚠️ Straddle not delta neutral: "
                f"delta_per_contract = {combined_delta_per_contract:.3f} "
                f"(threshold = {applied_threshold:.3f}, DTE = {days_to_expiry})"
            )
            return None
        
        # Calculate risks
        risks = RiskCalculator.calculate_risks(
            greeks={
                'delta': combined_delta_per_contract,
                'gamma': combined_gamma / lot_size,
                'vega': combined_vega / lot_size,
                'theta': combined_theta / lot_size,
            },
            lot_size=lot_size,
            spot=spot
        )
        
        edge_ratio = RiskCalculator.calculate_edge_ratio(
            theta=risks['theta_daily'],
            total_risk=risks['total_risk']
        )
        
        if edge_ratio < config.MIN_EDGE_RATIO:
            print(
                "⚠️ Straddle edge ratio too low: "
                f"{edge_ratio:.2f} < {config.MIN_EDGE_RATIO:.2f} "
                f"(theta_daily={risks['theta_daily']:.2f}, total_risk={risks['total_risk']:.2f})"
            )
            return None
        
        return {
            'trade_type': 'DELTA_NEUTRAL',
            'setup': 'ATM_STRADDLE',
            'strike': atm,
            'ce_premium': ce_greeks['premium'],
            'pe_premium': pe_greeks['premium'],
            'total_premium': combined_premium,
            'position_delta': combined_delta,
            'position_delta_per_contract': combined_delta_per_contract,
            'delta_neutral_threshold': applied_threshold,
            'days_to_expiry': days_to_expiry,
            'is_delta_neutral': is_neutral,
            'theta_daily': risks['theta_daily'],
            'delta_risk': risks['delta_risk'],
            'gamma_risk': risks['gamma_risk'],
            'vega_risk': risks['vega_risk'],
            'total_risk': risks['total_risk'],
            'edge_ratio': edge_ratio,
            'quality': self._get_quality_label(edge_ratio)
        }

    def validate_sideways_trade(
        self,
        option_chain: Dict,
        delta_threshold: Optional[float] = None,
        force_entry: Optional[bool] = None,
    ) -> Optional[Dict]:
        """
        Build sideways trades as per framework:
        1) Synthetic-center straddle
        2) Premium-width strangle around same synthetic center
        Recommend the best valid setup.
        """
        index = option_chain['index']
        if force_entry is None:
            force_entry = bool(getattr(config, "FORCE_SIDEWAYS_ENTRY", False))
        strikes = option_chain.get('strikes', [])
        if not strikes:
            return None

        strike_gap = config.STRIKE_GAPS[index]
        atm = int(option_chain['atm'])
        expiry = option_chain.get("expiry")
        dte = self._days_to_expiry(expiry)

        by_strike = {int(s['strike']): s for s in strikes}
        atm_row = by_strike.get(atm)
        if not atm_row:
            return None

        # Synthetic center from ATM put-call parity.
        synthetic = GreeksCalculator.calculate_synthetic_future(
            call_premium=float(atm_row['ce']['premium']),
            put_premium=float(atm_row['pe']['premium']),
            strike=atm,
            dte=dte or 1,
        )
        center = int(round(synthetic / strike_gap) * strike_gap)
        center_row = by_strike.get(center)

        if not center_row:
            nearest = min(by_strike.keys(), key=lambda k: abs(k - center))
            center = int(nearest)
            center_row = by_strike[center]

        straddle_premium = float(center_row['ce']['premium']) + float(center_row['pe']['premium'])
        width_points = int(round(straddle_premium / strike_gap) * strike_gap)
        width_points = max(width_points, strike_gap)

        candidates = []

        # Candidate 1: synthetic-center ATM straddle.
        straddle = self._evaluate_two_leg_short(
            option_chain=option_chain,
            ce_leg=center_row['ce'],
            pe_leg=center_row['pe'],
            setup='SYNTHETIC_ATM_STRADDLE',
            ce_strike=center,
            pe_strike=center,
            center_strike=center,
            delta_threshold=delta_threshold,
            force_entry=force_entry,
        )
        if straddle:
            candidates.append(straddle)

        # Candidate 2: premium-width strangle.
        pe_strike = center - width_points
        ce_strike = center + width_points
        pe_row = by_strike.get(pe_strike)
        ce_row = by_strike.get(ce_strike)
        if pe_row and ce_row:
            strangle = self._evaluate_two_leg_short(
                option_chain=option_chain,
                ce_leg=ce_row['ce'],
                pe_leg=pe_row['pe'],
                setup='PREMIUM_WIDTH_STRANGLE',
                ce_strike=ce_strike,
                pe_strike=pe_strike,
                center_strike=center,
                delta_threshold=delta_threshold,
                width_points=width_points,
                force_entry=force_entry,
            )
            if strangle:
                candidates.append(strangle)
        else:
            print(
                "⚠️ Premium-width strangle unavailable: "
                f"required strikes PE {pe_strike} / CE {ce_strike} not present"
            )

        if not candidates:
            return None

        # Prefer higher edge ratio; tie-breaker with lower |delta|.
        return max(
            candidates,
            key=lambda x: (x.get('edge_ratio', 0.0), -abs(x.get('position_delta_per_contract', 999.0))),
        )

    def _evaluate_two_leg_short(
        self,
        option_chain: Dict,
        ce_leg: Dict,
        pe_leg: Dict,
        setup: str,
        ce_strike: int,
        pe_strike: int,
        center_strike: int,
        delta_threshold: Optional[float] = None,
        width_points: Optional[int] = None,
        force_entry: bool = False,
    ) -> Optional[Dict]:
        """
        Common evaluator for short CE+PE structure (straddle or strangle).
        """
        index = option_chain['index']
        lot_size = config.LOT_SIZES[index]
        spot = option_chain['spot']
        expiry = option_chain.get("expiry")
        days_to_expiry = self._days_to_expiry(expiry)
        applied_threshold = (
            float(delta_threshold)
            if delta_threshold is not None
            else self.get_delta_neutral_threshold(days_to_expiry)
        )

        combined_delta_per_contract = float(ce_leg['delta']) + float(pe_leg['delta'])
        is_neutral = abs(combined_delta_per_contract) < applied_threshold
        if not is_neutral:
            print(
                f"⚠️ {setup} not delta neutral: "
                f"delta_per_contract = {combined_delta_per_contract:.3f} "
                f"(threshold = {applied_threshold:.3f}, DTE = {days_to_expiry})"
            )
            if not force_entry:
                return None

        greeks = {
            'delta': combined_delta_per_contract,
            'gamma': float(ce_leg['gamma']) + float(pe_leg['gamma']),
            'vega': float(ce_leg['vega']) + float(pe_leg['vega']),
            'theta': float(ce_leg['theta']) + float(pe_leg['theta']),
        }
        risks = RiskCalculator.calculate_risks(greeks=greeks, lot_size=lot_size, spot=spot)
        edge_ratio = RiskCalculator.calculate_edge_ratio(
            theta=risks['theta_daily'],
            total_risk=risks['total_risk'],
        )

        if edge_ratio < config.MIN_EDGE_RATIO:
            print(
                f"⚠️ {setup} edge ratio too low: {edge_ratio:.2f} < {config.MIN_EDGE_RATIO:.2f} "
                f"(theta_daily={risks['theta_daily']:.2f}, total_risk={risks['total_risk']:.2f})"
            )
            if not force_entry:
                return None

        return {
            'trade_type': 'DELTA_NEUTRAL',
            'setup': setup,
            'strike': center_strike,
            'ce_strike': ce_strike,
            'pe_strike': pe_strike,
            'width_points': width_points if width_points is not None else 0,
            'ce_premium': float(ce_leg['premium']),
            'pe_premium': float(pe_leg['premium']),
            'total_premium': (float(ce_leg['premium']) + float(pe_leg['premium'])) * lot_size,
            'position_delta': combined_delta_per_contract * lot_size,
            'position_delta_per_contract': combined_delta_per_contract,
            'delta_neutral_threshold': applied_threshold,
            'days_to_expiry': days_to_expiry,
            'is_delta_neutral': is_neutral,
            'force_entry_applied': bool(force_entry),
            'delta_check_passed': bool(is_neutral),
            'edge_check_passed': bool(edge_ratio >= config.MIN_EDGE_RATIO),
            'theta_daily': risks['theta_daily'],
            'delta_risk': risks['delta_risk'],
            'gamma_risk': risks['gamma_risk'],
            'vega_risk': risks['vega_risk'],
            'total_risk': risks['total_risk'],
            'edge_ratio': edge_ratio,
            'quality': self._get_quality_label(edge_ratio),
        }
    
    def _validate_single_strike(
        self,
        strike_data: Dict,
        option_type: str,
        spot: float,
        index: str
    ) -> Optional[Dict]:
        """
        Validate a single strike for selling
        """
        
        greeks = strike_data[option_type]
        strike = strike_data['strike']
        lot_size = config.LOT_SIZES[index]
        
        # Calculate risks
        risks = RiskCalculator.calculate_risks(
            greeks=greeks,
            lot_size=lot_size,
            spot=spot
        )
        
        # Calculate edge ratio
        edge_ratio = RiskCalculator.calculate_edge_ratio(
            theta=risks['theta_daily'],
            total_risk=risks['total_risk']
        )
        
        return {
            'strike': strike,
            'option_type': option_type.upper(),
            'premium': greeks['premium'],
            'premium_total': greeks['premium'] * lot_size,
            'delta': greeks['delta'],
            'position_delta': risks['position_delta'],
            'theta_daily': risks['theta_daily'],
            'delta_risk': risks['delta_risk'],
            'gamma_risk': risks['gamma_risk'],
            'vega_risk': risks['vega_risk'],
            'total_risk': risks['total_risk'],
            'edge_ratio': edge_ratio,
            'quality': self._get_quality_label(edge_ratio)
        }
    
    def _get_otm_put_strikes(self, option_chain: Dict) -> List[Dict]:
        """Get OTM put strikes (below ATM)"""
        atm = option_chain['atm']
        strike_gap = config.STRIKE_GAPS[option_chain['index']]
        
        # Get 3 strikes below ATM
        otm_puts = [
            s for s in option_chain['strikes']
            if s['strike'] < atm and s['strike'] >= atm - 3 * strike_gap
        ]
        
        return sorted(otm_puts, key=lambda x: x['strike'], reverse=True)[:3]
    
    def _get_otm_call_strikes(self, option_chain: Dict) -> List[Dict]:
        """Get OTM call strikes (above ATM)"""
        atm = option_chain['atm']
        strike_gap = config.STRIKE_GAPS[option_chain['index']]
        
        # Get 3 strikes above ATM
        otm_calls = [
            s for s in option_chain['strikes']
            if s['strike'] > atm and s['strike'] <= atm + 3 * strike_gap
        ]
        
        return sorted(otm_calls, key=lambda x: x['strike'])[:3]
    
    def _get_quality_label(self, edge_ratio: float) -> str:
        """Get quality label for edge ratio"""
        if edge_ratio >= config.MIN_EDGE_RATIO * 1.3:
            return 'EXCELLENT'
        elif edge_ratio >= config.MIN_EDGE_RATIO:
            return 'GOOD'
        else:
            return 'POOR'


def display_trade_recommendation(trade: Dict):
    """
    Display trade recommendation in clean format
    """
    
    print("\n" + "═" * 70)
    print("  💰 TRADE RECOMMENDATION")
    print("═" * 70)
    
    if trade['trade_type'] == 'DIRECTIONAL':
        print(f"\n🎯 Direction: {trade['direction']}")
        print(f"   Strategy: Sell {trade['best_strike']['option_type']}")
        
        best = trade['best_strike']
        print(f"\n✅ BEST STRIKE: {best['strike']} {best['option_type']}")
        print(f"   Premium: ₹{best['premium']:.2f} (Total: ₹{best['premium_total']:,.0f})")
        print(f"   Position Delta: {best['position_delta']:.2f}")
        
        print(f"\n💵 DAILY INCOME:")
        print(f"   Theta: ₹{best['theta_daily']:,.0f} per day")
        
        print(f"\n⚠️ RISKS:")
        print(f"   Delta Risk: ₹{best['delta_risk']:,.0f}")
        print(f"   Gamma Risk: ₹{best['gamma_risk']:,.0f}")
        print(f"   Vega Risk: ₹{best['vega_risk']:,.0f}")
        print(f"   Total Risk: ₹{best['total_risk']:,.0f}")
        
        print(f"\n📊 EDGE RATIO: {best['edge_ratio']:.2f}x ({best['quality']})")
        print(f"   You earn ₹{best['theta_daily']:,.0f} daily vs ₹{best['total_risk']:,.0f} risk")
        
        if best['edge_ratio'] >= config.MIN_EDGE_RATIO:
            print(f"\n✅ VERDICT: SAFE TO TRADE")
        else:
            print(f"\n❌ VERDICT: SKIP - Edge ratio too low")
    
    elif trade['trade_type'] == 'DELTA_NEUTRAL':
        setup = trade.get('setup', 'DELTA_NEUTRAL')
        print(f"\n🎯 Strategy: {setup}")
        print(f"   Market: SIDEWAYS (no clear direction)")
        print(
            f"\n✅ SETUP: Sell PE {trade.get('pe_strike', trade['strike'])} + "
            f"CE {trade.get('ce_strike', trade['strike'])}"
        )
        if trade.get('width_points', 0):
            print(f"   Width from center: ±{trade['width_points']}")
        print(f"   CE Premium: ₹{trade['ce_premium']:.2f}")
        print(f"   PE Premium: ₹{trade['pe_premium']:.2f}")
        print(f"   Total Premium: ₹{trade['total_premium']:,.0f}")
        print(f"   Position Delta: {trade['position_delta']:.2f} (✓ Neutral)")
        if trade.get('days_to_expiry') is not None:
            print(f"   DTE: {trade['days_to_expiry']}")
        if trade.get('delta_neutral_threshold') is not None:
            print(f"   Delta Threshold: {trade['delta_neutral_threshold']:.3f}")
        
        print(f"\n💵 DAILY INCOME:")
        print(f"   Theta: ₹{trade['theta_daily']:,.0f} per day")
        
        print(f"\n⚠️ RISKS:")
        print(f"   Delta Risk: ₹{trade['delta_risk']:,.0f}")
        print(f"   Gamma Risk: ₹{trade['gamma_risk']:,.0f}")
        print(f"   Vega Risk: ₹{trade['vega_risk']:,.0f}")
        print(f"   Total Risk: ₹{trade['total_risk']:,.0f}")
        
        print(f"\n📊 EDGE RATIO: {trade['edge_ratio']:.2f}x ({trade['quality']})")
        
        if trade['edge_ratio'] >= config.MIN_EDGE_RATIO:
            print(f"\n✅ VERDICT: SAFE TO TRADE")
        else:
            print(f"\n❌ VERDICT: SKIP - Edge ratio too low")
    
    print("═" * 70)
