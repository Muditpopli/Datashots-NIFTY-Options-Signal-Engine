"""
Overnight Gap Predictor
Predicts next day gap using Greeks analysis and option data
Uses 5 weighted indicators for directional bias
"""

from typing import Dict, Optional
import json
from datetime import datetime
from dhan_api import get_dhan
from accuracy_tracker import AccuracyTracker


class GapPredictor:
    """
    Predicts overnight gap direction using:
    1. Put-Call Ratio (PCR)
    2. IV Skew
    3. Net Delta
    4. Max Pain
    5. Vega Concentration
    """
    
    def __init__(self):
        self.dhan = get_dhan()
        self.accuracy_tracker = AccuracyTracker()
        
        # Weights for each indicator (total = 1.0)
        self.weights = {
            'pcr': 0.25,
            'iv_skew': 0.20,
            'net_delta': 0.25,
            'max_pain': 0.15,
            'vega_concentration': 0.15
        }
    
    def predict_gap(self, index: str) -> Optional[Dict]:
        """
        Predict overnight gap for index
        
        Returns:
        {
            'direction': 'BULLISH' | 'BEARISH' | 'NEUTRAL',
            'confidence': 0.0 to 1.0,
            'score': -100 to +100,
            'indicators': {...},
            'recommendation': 'GAP UP' | 'GAP DOWN' | 'FLAT OPEN'
        }
        """
        
        print(f"\n{'='*70}")
        print(f"  OVERNIGHT GAP PREDICTOR - {index}")
        print(f"{'='*70}")
        
        # Get option chain
        chain = self.dhan.get_option_chain(index, use_cache=False)
        
        if not chain:
            print("❌ Cannot fetch option chain")
            return None
        
        spot = chain['spot']
        atm = chain['atm']
        strikes = chain['strikes']
        
        print(f"\n📊 Current Market:")
        print(f"   Spot: {spot:.2f}")
        print(f"   ATM: {atm}")
        print(f"   Strikes: {len(strikes)}")
        
        # Calculate all indicators
        print(f"\n🔍 Analyzing Indicators...")
        
        indicators = {}
        
        # 1. Put-Call Ratio
        pcr_score, pcr_data = self._calculate_pcr(strikes, spot)
        indicators['pcr'] = pcr_data
        
        # 2. IV Skew
        iv_skew_score, iv_skew_data = self._calculate_iv_skew(strikes, atm)
        indicators['iv_skew'] = iv_skew_data
        
        # 3. Net Delta
        net_delta_score, net_delta_data = self._calculate_net_delta(strikes, spot)
        indicators['net_delta'] = net_delta_data
        
        # 4. Max Pain
        max_pain_score, max_pain_data = self._calculate_max_pain(strikes, spot)
        indicators['max_pain'] = max_pain_data
        
        # 5. Vega Concentration
        vega_score, vega_data = self._calculate_vega_concentration(strikes, spot)
        indicators['vega_concentration'] = vega_data
        
        # Calculate weighted score
        total_score = (
            pcr_score * self.weights['pcr'] +
            iv_skew_score * self.weights['iv_skew'] +
            net_delta_score * self.weights['net_delta'] +
            max_pain_score * self.weights['max_pain'] +
            vega_score * self.weights['vega_concentration']
        )
        
        # Determine direction
        if total_score > 20:
            direction = 'BULLISH'
            recommendation = 'GAP UP'
            confidence = min(abs(total_score) / 100, 1.0)
        elif total_score < -20:
            direction = 'BEARISH'
            recommendation = 'GAP DOWN'
            confidence = min(abs(total_score) / 100, 1.0)
        else:
            direction = 'NEUTRAL'
            recommendation = 'FLAT OPEN'
            confidence = 1.0 - (abs(total_score) / 20)
        
        result = {
            'index': index,
            'spot': spot,
            'direction': direction,
            'recommendation': recommendation,
            'score': total_score,
            'confidence': confidence,
            'indicators': indicators,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Display results
        self._display_results(result)
        
        # Save to file
        self._save_prediction(result)
        gap_id = self.accuracy_tracker.record_gap_prediction(result)
        print(f"🆔 Gap ID: {gap_id}")
        
        return result
    
    def _calculate_pcr(self, strikes, spot):
        """
        Put-Call Ratio Analysis
        PCR > 1.0 = More puts = Bullish (put writing)
        PCR < 1.0 = More calls = Bearish (call writing)
        """
        
        total_put_oi = 0
        total_call_oi = 0
        
        # Focus on strikes within 5% of spot
        for strike_data in strikes:
            strike = strike_data['strike']
            distance = abs(strike - spot) / spot
            
            if distance < 0.05:  # Within 5%
                total_put_oi += strike_data['pe']['oi']
                total_call_oi += strike_data['ce']['oi']
        
        pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 1.0
        
        # Score: PCR > 1.2 = Strong Bullish, PCR < 0.8 = Strong Bearish
        if pcr > 1.2:
            score = min((pcr - 1.0) * 50, 50)  # Max +50
        elif pcr < 0.8:
            score = max((pcr - 1.0) * 50, -50)  # Max -50
        else:
            score = (pcr - 1.0) * 25
        
        return score, {
            'pcr': round(pcr, 2),
            'put_oi': total_put_oi,
            'call_oi': total_call_oi,
            'score': round(score, 1),
            'signal': 'Bullish' if score > 10 else 'Bearish' if score < -10 else 'Neutral'
        }
    
    def _calculate_iv_skew(self, strikes, atm):
        """
        IV Skew Analysis
        OTM Put IV > OTM Call IV = Fear = Bullish (buying puts for protection)
        OTM Call IV > OTM Put IV = Greed = Bearish (buying calls for upside)
        """
        
        # Get OTM put (below ATM)
        otm_puts = [s for s in strikes if s['strike'] < atm]
        otm_put = otm_puts[-3] if len(otm_puts) >= 3 else otm_puts[-1] if otm_puts else None
        
        # Get OTM call (above ATM)
        otm_calls = [s for s in strikes if s['strike'] > atm]
        otm_call = otm_calls[2] if len(otm_calls) >= 3 else otm_calls[0] if otm_calls else None
        
        if not otm_put or not otm_call:
            return 0, {'skew': 0, 'signal': 'No data'}
        
        put_iv = otm_put['pe']['iv']
        call_iv = otm_call['ce']['iv']
        
        skew = put_iv - call_iv
        
        # Skew > 2% = Bullish, Skew < -2% = Bearish
        score = skew * 10  # Max ±50
        score = max(min(score, 50), -50)
        
        return score, {
            'put_iv': round(put_iv, 2),
            'call_iv': round(call_iv, 2),
            'skew': round(skew, 2),
            'score': round(score, 1),
            'signal': 'Bullish' if skew > 1 else 'Bearish' if skew < -1 else 'Neutral'
        }
    
    def _calculate_net_delta(self, strikes, spot):
        """
        Net Delta Analysis
        Net positive delta = Calls being bought = Bullish
        Net negative delta = Puts being bought = Bearish
        """
        
        total_call_delta = 0
        total_put_delta = 0
        
        for strike_data in strikes:
            strike = strike_data['strike']
            distance = abs(strike - spot) / spot
            
            if distance < 0.05:  # Within 5%
                ce_delta = strike_data['ce']['delta']
                pe_delta = strike_data['pe']['delta']
                ce_oi = strike_data['ce']['oi']
                pe_oi = strike_data['pe']['oi']
                
                # Weight by OI
                total_call_delta += ce_delta * ce_oi
                total_put_delta += pe_delta * pe_oi
        
        net_delta = total_call_delta + total_put_delta  # Put delta is negative
        
        # Normalize
        total_oi = sum(s['ce']['oi'] + s['pe']['oi'] for s in strikes)
        normalized_delta = net_delta / total_oi if total_oi > 0 else 0
        
        # Score based on normalized delta
        score = normalized_delta * 100000  # Scale up
        score = max(min(score, 50), -50)
        
        return score, {
            'net_delta': round(normalized_delta, 6),
            'score': round(score, 1),
            'signal': 'Bullish' if score > 10 else 'Bearish' if score < -10 else 'Neutral'
        }
    
    def _calculate_max_pain(self, strikes, spot):
        """
        Max Pain Analysis
        Spot > Max Pain = Likely to fall (pain for call writers)
        Spot < Max Pain = Likely to rise (pain for put writers)
        """
        
        max_pain = None
        min_pain_value = float('inf')
        
        for strike_data in strikes:
            strike = strike_data['strike']
            
            # Calculate total pain at this strike
            total_pain = 0
            
            for s in strikes:
                s_strike = s['strike']
                ce_oi = s['ce']['oi']
                pe_oi = s['pe']['oi']
                
                # ITM call pain
                if s_strike < strike:
                    total_pain += (strike - s_strike) * ce_oi
                
                # ITM put pain
                if s_strike > strike:
                    total_pain += (s_strike - strike) * pe_oi
            
            if total_pain < min_pain_value:
                min_pain_value = total_pain
                max_pain = strike
        
        if not max_pain:
            return 0, {'max_pain': 0, 'signal': 'No data'}
        
        # Distance from max pain
        distance = spot - max_pain
        distance_pct = (distance / spot) * 100
        
        # Score: Spot above max pain = Bearish, below = Bullish
        score = -distance_pct * 5  # Inverted
        score = max(min(score, 50), -50)
        
        return score, {
            'max_pain': max_pain,
            'current_spot': spot,
            'distance': round(distance, 2),
            'distance_pct': round(distance_pct, 2),
            'score': round(score, 1),
            'signal': 'Bullish' if distance < 0 else 'Bearish' if distance > 0 else 'Neutral'
        }
    
    def _calculate_vega_concentration(self, strikes, spot):
        """
        Vega Concentration Analysis
        More vega in OTM calls = Expecting upside volatility = Bullish
        More vega in OTM puts = Expecting downside volatility = Bearish
        """
        
        otm_call_vega = 0
        otm_put_vega = 0
        
        for strike_data in strikes:
            strike = strike_data['strike']
            
            # OTM calls (above spot)
            if strike > spot:
                ce_vega = strike_data['ce']['vega']
                ce_oi = strike_data['ce']['oi']
                otm_call_vega += ce_vega * ce_oi
            
            # OTM puts (below spot)
            if strike < spot:
                pe_vega = strike_data['pe']['vega']
                pe_oi = strike_data['pe']['oi']
                otm_put_vega += pe_vega * pe_oi
        
        total_vega = otm_call_vega + otm_put_vega
        
        if total_vega == 0:
            return 0, {'vega_ratio': 0, 'signal': 'No data'}
        
        vega_ratio = otm_call_vega / total_vega
        
        # Ratio > 0.55 = Bullish, < 0.45 = Bearish
        score = (vega_ratio - 0.5) * 200
        score = max(min(score, 50), -50)
        
        return score, {
            'otm_call_vega': round(otm_call_vega, 2),
            'otm_put_vega': round(otm_put_vega, 2),
            'vega_ratio': round(vega_ratio, 3),
            'score': round(score, 1),
            'signal': 'Bullish' if vega_ratio > 0.52 else 'Bearish' if vega_ratio < 0.48 else 'Neutral'
        }

    def validate_overnight_position(
        self,
        gap_prediction: Dict,
        current_strength: float,
        current_vega_theta: Dict
    ) -> bool:
        """
        Only go overnight directional if:
        1. Gap prediction is strong
        2. Current intraday strength CONFIRMS it
        3. Greeks (Vega/Theta) SUPPORT the direction
        """

        gap_direction = gap_prediction['direction']
        gap_confidence = gap_prediction['confidence']

        # Check 1: Gap confidence must be high
        if gap_confidence < 0.7:
            print("❌ Gap prediction confidence too low for overnight")
            return False

        # Check 2: Intraday strength must CONFIRM gap direction
        if gap_direction == 'BULLISH' and current_strength < 5:
            print("❌ Intraday strength doesn't confirm bullish gap")
            return False

        if gap_direction == 'BEARISH' and current_strength > -5:
            print("❌ Intraday strength doesn't confirm bearish gap")
            return False

        # Check 3: Greeks must STILL support the move at 2:30 PM
        if gap_direction == 'BULLISH':
            if current_vega_theta['put_vega'] < 0:  # Put vega increasing = bearish sign
                print("❌ Put vega increasing, contradicts bullish gap")
                return False

        print("✅ ALL CHECKS PASSED - Safe for overnight position")
        return True
    
    def _display_results(self, result):
        """Display prediction results"""
        
        print(f"\n{'='*70}")
        print(f"  OVERNIGHT GAP PREDICTION")
        print(f"{'='*70}")
        
        direction = result['direction']
        score = result['score']
        confidence = result['confidence']
        
        # Direction emoji
        if direction == 'BULLISH':
            emoji = '🟢'
        elif direction == 'BEARISH':
            emoji = '🔴'
        else:
            emoji = '🟡'
        
        print(f"\n{emoji} PREDICTION: {direction}")
        print(f"   Score: {score:+.1f}")
        print(f"   Confidence: {confidence*100:.0f}%")
        print(f"\n💡 RECOMMENDATION: {result['recommendation']}")
        
        print(f"\n📊 INDICATOR BREAKDOWN:")
        print(f"{'-'*70}")
        
        indicators = result['indicators']
        
        print(f"\n1. Put-Call Ratio (PCR)")
        pcr = indicators['pcr']
        print(f"   PCR: {pcr['pcr']:.2f} | Score: {pcr['score']:+.1f} | {pcr['signal']}")
        
        print(f"\n2. IV Skew")
        iv = indicators['iv_skew']
        print(f"   Skew: {iv['skew']:+.2f}% | Score: {iv['score']:+.1f} | {iv['signal']}")
        
        print(f"\n3. Net Delta")
        delta = indicators['net_delta']
        print(f"   Net Delta: {delta['net_delta']:.6f} | Score: {delta['score']:+.1f} | {delta['signal']}")
        
        print(f"\n4. Max Pain")
        pain = indicators['max_pain']
        print(f"   Max Pain: {pain['max_pain']} | Spot: {pain['current_spot']:.2f}")
        print(f"   Distance: {pain['distance']:+.2f} ({pain['distance_pct']:+.2f}%) | Score: {pain['score']:+.1f}")
        
        print(f"\n5. Vega Concentration")
        vega = indicators['vega_concentration']
        print(f"   Vega Ratio: {vega['vega_ratio']:.3f} | Score: {vega['score']:+.1f} | {vega['signal']}")
        
        print(f"\n{'='*70}")
        print(f"  TRADING STRATEGY")
        print(f"{'='*70}")
        
        if direction == 'BULLISH':
            print(f"\n✅ Expected GAP UP")
            print(f"   • Consider selling OTM puts at open")
            print(f"   • Wait for pullback before buying calls")
            print(f"   • Avoid selling calls at open")
        elif direction == 'BEARISH':
            print(f"\n✅ Expected GAP DOWN")
            print(f"   • Consider selling OTM calls at open")
            print(f"   • Wait for bounce before buying puts")
            print(f"   • Avoid selling puts at open")
        else:
            print(f"\n✅ Expected FLAT OPEN")
            print(f"   • Wait for direction confirmation")
            print(f"   • Consider delta-neutral strategies")
            print(f"   • Don't force directional trades")
        
        print(f"\n⚠️  Confidence: {confidence*100:.0f}% - Trade accordingly!")
        print(f"{'='*70}\n")
    
    def _save_prediction(self, result):
        """Save prediction to file"""
        
        import os
        
        # Create directory
        os.makedirs('data/predictions', exist_ok=True)
        
        # Filename with timestamp
        filename = f"data/predictions/gap_prediction_{result['index']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"💾 Prediction saved: {filename}")


# CLI
if __name__ == '__main__':
    import sys
    
    predictor = GapPredictor()
    
    # Get index from command line or default to NIFTY
    index = sys.argv[1] if len(sys.argv) > 1 else 'NIFTY'
    
    # Run prediction
    result = predictor.predict_gap(index)
    
    if result:
        print(f"\n✅ Prediction complete!")
        print(f"   Use this for tomorrow's gap trading strategy")
