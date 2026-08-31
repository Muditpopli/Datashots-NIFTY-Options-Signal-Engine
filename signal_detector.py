"""
Signal Detector
Compares baseline (9:15 AM) Greeks with current (10:00 AM) Greeks
Generates directional signals: BULLISH, BEARISH, or SIDEWAYS
"""

from typing import Dict, Optional
from datetime import datetime
import json
import os
import config


class SignalDetector:
    """
    Detects market direction by analyzing Greeks flow
    """
    
    def __init__(self):
        self.baseline = None
        self.baseline_time = None
    
    def capture_baseline(self, option_chain: Dict) -> bool:
        """
        Capture baseline Greeks snapshot
        
        Args:
            option_chain: Full option chain data from Dhan API
            
        Returns:
            True if successful
        """
        
        if not option_chain:
            print("❌ No option chain data to save as baseline")
            return False
        
        self.baseline = {
            'index': option_chain['index'],
            'spot': option_chain['spot'],
            'atm': option_chain['atm'],
            'strikes': option_chain['strikes'],
            'timestamp': datetime.now().isoformat()
        }
        
        self.baseline_time = datetime.now()
        
        # Save to file
        self._save_baseline_to_file()
        
        print(f"✅ Baseline captured for {option_chain['index']}")
        print(f"   Spot: {option_chain['spot']:.2f} | ATM: {option_chain['atm']}")
        print(f"   Strikes: {len(option_chain['strikes'])}")
        
        return True
    
    def generate_signal(self, current_chain: Dict) -> Optional[Dict]:
        """
        Generate trading signal by comparing current with baseline
        
        Returns signal dict with:
            - direction: 'BULLISH', 'BEARISH', or 'SIDEWAYS'
            - strength: Signal strength (-20 to +20)
            - confidence: 0 to 1
            - interpretation: What to trade
        """
        
        if not self.baseline:
            print("❌ No baseline captured. Run baseline capture first.")
            return None
        
        if self.baseline['index'] != current_chain['index']:
            print("❌ Index mismatch between baseline and current")
            return None
        
        # Calculate Greeks flow
        delta_flow = self._calculate_delta_flow(current_chain)
        vega_flow = self._calculate_vega_flow(current_chain)
        gamma_shift = self._calculate_gamma_shift(current_chain)
        
        # Combined strength score
        strength = (
            delta_flow * 0.5 +    # 50% weight to delta
            vega_flow * 0.3 +     # 30% weight to vega
            gamma_shift * 0.2     # 20% weight to gamma
        )
        
        # Determine direction
        if strength > config.BULLISH_THRESHOLD:
            direction = 'BULLISH'
            interpretation = 'Sell OTM Puts (collect premium on upside)'
            confidence = min(abs(strength) / 20.0, 1.0)
        elif strength < config.BEARISH_THRESHOLD:
            direction = 'BEARISH'
            interpretation = 'Sell OTM Calls (collect premium on downside)'
            confidence = min(abs(strength) / 20.0, 1.0)
        else:
            direction = 'SIDEWAYS'
            interpretation = 'Sell Delta-Neutral Straddle/Strangle (range-bound)'
            confidence = 0.7  # Medium confidence for sideways
        
        signal = {
            'index': current_chain['index'],
            'direction': direction,
            'strength': round(strength, 2),
            'confidence': round(confidence, 2),
            'interpretation': interpretation,
            'spot': current_chain['spot'],
            'atm': current_chain['atm'],
            'baseline_spot': self.baseline['spot'],
            'spot_change': round(current_chain['spot'] - self.baseline['spot'], 2),
            'spot_change_pct': round(
                (current_chain['spot'] - self.baseline['spot']) / self.baseline['spot'] * 100, 2
            ),
            'components': {
                'delta_flow': round(delta_flow, 2),
                'vega_flow': round(vega_flow, 2),
                'gamma_shift': round(gamma_shift, 2)
            }
        }
        
        return signal
    
    def _calculate_delta_flow(self, current_chain: Dict) -> float:
        """
        Calculate net delta flow (call delta - put delta)
        Positive = Call buying pressure (bullish)
        Negative = Put buying pressure (bearish)
        """
        
        baseline_strikes = {s['strike']: s for s in self.baseline['strikes']}
        
        total_call_delta_now = 0
        total_put_delta_now = 0
        total_call_delta_base = 0
        total_put_delta_base = 0
        
        atm = current_chain['atm']
        strike_gap = config.STRIKE_GAPS[current_chain['index']]
        
        # Analyze strikes near ATM (±3 strikes)
        for strike_data in current_chain['strikes']:
            strike = strike_data['strike']
            
            # Only look at near-ATM strikes
            if abs(strike - atm) > 3 * strike_gap:
                continue
            
            # Current deltas
            total_call_delta_now += abs(strike_data['ce']['delta'])
            total_put_delta_now += abs(strike_data['pe']['delta'])
            
            # Baseline deltas
            if strike in baseline_strikes:
                base = baseline_strikes[strike]
                total_call_delta_base += abs(base['ce']['delta'])
                total_put_delta_base += abs(base['pe']['delta'])
        
        # Calculate flow (change from baseline)
        call_flow = total_call_delta_now - total_call_delta_base
        put_flow = total_put_delta_now - total_put_delta_base
        
        # Net flow (positive = bullish, negative = bearish)
        net_flow = call_flow - put_flow
        
        return net_flow * 10  # Scale for visibility
    
    def _calculate_vega_flow(self, current_chain: Dict) -> float:
        """
        Calculate vega flow
        Rising call vega = bullish expectations
        Rising put vega = bearish expectations
        """
        
        baseline_strikes = {s['strike']: s for s in self.baseline['strikes']}
        
        call_vega_change = 0
        put_vega_change = 0
        
        atm = current_chain['atm']
        strike_gap = config.STRIKE_GAPS[current_chain['index']]
        
        for strike_data in current_chain['strikes']:
            strike = strike_data['strike']
            
            if abs(strike - atm) > 3 * strike_gap:
                continue
            
            if strike in baseline_strikes:
                base = baseline_strikes[strike]
                
                call_vega_change += (
                    strike_data['ce']['vega'] - base['ce']['vega']
                )
                put_vega_change += (
                    strike_data['pe']['vega'] - base['pe']['vega']
                )
        
        # If call vega rising faster = bullish
        # If put vega rising faster = bearish
        vega_flow = call_vega_change - put_vega_change
        
        return vega_flow * 5  # Scale
    
    def _calculate_gamma_shift(self, current_chain: Dict) -> float:
        """
        Calculate where gamma is concentrating
        High OTM call gamma = bullish
        High OTM put gamma = bearish
        """
        
        atm = current_chain['atm']
        strike_gap = config.STRIKE_GAPS[current_chain['index']]
        
        otm_call_gamma = 0
        otm_put_gamma = 0
        
        for strike_data in current_chain['strikes']:
            strike = strike_data['strike']
            
            # OTM calls (above ATM)
            if strike > atm:
                otm_call_gamma += strike_data['ce']['gamma']
            
            # OTM puts (below ATM)
            if strike < atm:
                otm_put_gamma += strike_data['pe']['gamma']
        
        # If call gamma > put gamma = expecting upside
        gamma_bias = otm_call_gamma - otm_put_gamma
        
        return gamma_bias * 1000  # Scale

    def calculate_vega_theta_decay(self, current_chain: Dict) -> Dict:
        """
        Calculate Vega and Theta decay from baseline
        POSITIVE = Decay (good for selling)
        NEGATIVE = Increase (avoid selling)
        """

        baseline_strikes = {s['strike']: s for s in self.baseline['strikes']}

        # Sum ATM to OTM strikes (like Mudit does)
        atm = current_chain['atm']
        strike_gap = config.STRIKE_GAPS[current_chain['index']]

        nifty_put_vega_change = 0
        nifty_call_vega_change = 0
        nifty_put_theta_change = 0
        nifty_call_theta_change = 0

        for strike_data in current_chain['strikes']:
            strike = strike_data['strike']

            # ATM to OTM only
            if strike < atm:  # PUT side
                if strike in baseline_strikes:
                    base = baseline_strikes[strike]
                    # Opening vega - Current vega = Decay
                    nifty_put_vega_change += (base['pe']['vega'] - strike_data['pe']['vega'])
                    nifty_put_theta_change += (base['pe']['theta'] - strike_data['pe']['theta'])

            elif strike >= atm:  # CALL side
                if strike in baseline_strikes:
                    base = baseline_strikes[strike]
                    nifty_call_vega_change += (base['ce']['vega'] - strike_data['ce']['vega'])
                    nifty_call_theta_change += (base['ce']['theta'] - strike_data['ce']['theta'])

        return {
            'put_vega': round(nifty_put_vega_change, 2),
            'call_vega': round(nifty_call_vega_change, 2),
            'put_theta': round(nifty_put_theta_change, 2),
            'call_theta': round(nifty_call_theta_change, 2)
        }

    def calculate_strength(self, vega_theta_decay: Dict, delta_flow: float) -> float:
        """
        Mudit's Strength Indicator
        Combines Vega, Theta, and Delta changes
        """

        put_vega = vega_theta_decay['put_vega']
        call_vega = vega_theta_decay['call_vega']
        put_theta = vega_theta_decay['put_theta']
        call_theta = vega_theta_decay['call_theta']

        # If call vega/theta decaying MORE = Bullish
        # If put vega/theta decaying MORE = Bearish
        vega_component = call_vega - put_vega  # Positive = call decaying more (bullish)
        theta_component = call_theta - put_theta
        delta_component = delta_flow

        # Weighted combination (adjust weights as needed)
        strength = (
            vega_component * 0.5 +
            theta_component * 0.3 +
            delta_component * 0.2
        )

        return round(strength, 2)
    
    def _save_baseline_to_file(self):
        """Save baseline to JSON file"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{config.BASELINE_DIR}/baseline_{self.baseline['index']}_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(self.baseline, f, indent=2)
            
            print(f"💾 Baseline saved: {filename}")
        except Exception as e:
            print(f"⚠️ Could not save baseline: {e}")

    def load_latest_baseline(self, index: str) -> bool:
        """
        Load most recent baseline for the given index from disk.

        Returns:
            True if baseline loaded successfully, else False
        """
        try:
            pattern = f"baseline_{index}_"
            candidates = []

            for name in os.listdir(config.BASELINE_DIR):
                if name.startswith(pattern) and name.endswith(".json"):
                    full_path = os.path.join(config.BASELINE_DIR, name)
                    candidates.append(full_path)

            if not candidates:
                return False

            latest_file = max(candidates, key=os.path.getmtime)

            with open(latest_file, "r") as f:
                data = json.load(f)

            required_keys = {"index", "spot", "atm", "strikes", "timestamp"}
            if not isinstance(data, dict) or not required_keys.issubset(data.keys()):
                print(f"⚠️ Invalid baseline format: {latest_file}")
                return False

            if data["index"] != index:
                print(f"⚠️ Baseline index mismatch in {latest_file}")
                return False

            self.baseline = data
            ts = data.get("timestamp")
            self.baseline_time = datetime.fromisoformat(ts) if ts else None

            print(f"📂 Loaded baseline: {latest_file}")
            return True
        except Exception as e:
            print(f"⚠️ Could not load baseline for {index}: {e}")
            return False
