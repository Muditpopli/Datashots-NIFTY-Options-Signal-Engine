"""
Greeks Calculator - Black-Scholes Model
Clean implementation for accurate option pricing and Greeks
"""

import math
from scipy.stats import norm
from typing import Dict
import config


class GreeksCalculator:
    """
    Black-Scholes Greeks calculator
    All Greeks are calculated for a SINGLE contract (multiply by lot size separately)
    """
    
    @staticmethod
    def calculate_greeks(
        spot: float,
        strike: float,
        dte: int,
        iv: float,
        option_type: str = 'CE'
    ) -> Dict[str, float]:
        """
        Calculate all Greeks for an option
        
        Args:
            spot: Current spot price
            strike: Strike price
            dte: Days to expiry
            iv: Implied volatility (as decimal, e.g., 0.18 for 18%)
            option_type: 'CE' or 'PE'
            
        Returns:
            Dict with premium, delta, gamma, theta, vega, rho
        """
        
        # Convert inputs
        S = spot
        K = strike
        T = max(dte / 365.0, 0.003)  # Min 1 day
        sigma = max(iv, 0.05)  # Min 5% IV
        r = config.RISK_FREE_RATE
        q = config.DIVIDEND_YIELD
        
        # Calculate d1 and d2
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        # Standard normal CDF and PDF
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        n_d1 = norm.pdf(d1)  # PDF for gamma/vega
        
        is_call = (option_type.upper() == 'CE')
        
        if is_call:
            # Call option
            premium = S * math.exp(-q * T) * N_d1 - K * math.exp(-r * T) * N_d2
            delta = math.exp(-q * T) * N_d1
            theta_annual = (
                -S * n_d1 * sigma * math.exp(-q * T) / (2 * math.sqrt(T))
                - r * K * math.exp(-r * T) * N_d2
                + q * S * math.exp(-q * T) * N_d1
            )
            rho = K * T * math.exp(-r * T) * N_d2 / 100  # Per 1% change
        else:
            # Put option
            premium = K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)
            delta = -math.exp(-q * T) * norm.cdf(-d1)
            theta_annual = (
                -S * n_d1 * sigma * math.exp(-q * T) / (2 * math.sqrt(T))
                + r * K * math.exp(-r * T) * norm.cdf(-d2)
                - q * S * math.exp(-q * T) * norm.cdf(-d1)
            )
            rho = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100
        
        # Gamma and Vega (same for calls and puts)
        gamma = n_d1 * math.exp(-q * T) / (S * sigma * math.sqrt(T))
        vega = S * n_d1 * math.sqrt(T) * math.exp(-q * T) / 100  # Per 1% IV change
        
        # Convert to daily values
        theta_daily = theta_annual / 365.0
        
        return {
            'premium': premium,
            'delta': delta,
            'gamma': gamma,
            'theta': theta_daily,  # Daily theta
            'vega': vega,  # Per 1% IV
            'rho': rho,
            'iv': iv * 100  # As percentage
        }
    
    @staticmethod
    def calculate_synthetic_future(
        call_premium: float,
        put_premium: float,
        strike: float,
        dte: int
    ) -> float:
        """
        Calculate synthetic future price using put-call parity
        Synthetic Future = Strike + (Call Premium - Put Premium) * e^(r*T)
        
        This gives you the "true" forward price implied by option prices
        Use this to find TRUE ATM strike
        """
        r = config.RISK_FREE_RATE
        T = dte / 365.0
        
        premium_diff = call_premium - put_premium
        synthetic = strike + premium_diff * math.exp(r * T)
        
        return synthetic


class RiskCalculator:
    """
    Calculate position risks and Edge Ratio
    """
    
    @staticmethod
    def calculate_risks(
        greeks: Dict[str, float],
        lot_size: int,
        spot: float
    ) -> Dict[str, float]:
        """
        Calculate all risk components for a position
        
        Returns dict with:
            - delta_risk: Risk from adverse price move
            - gamma_risk: Risk from gamma exposure
            - vega_risk: Risk from IV increase
            - total_risk: Sum of all risks with buffers
        """
        
        # Position Greeks (for full lot)
        delta = greeks['delta'] * lot_size
        gamma = greeks['gamma'] * lot_size
        vega = greeks['vega'] * lot_size
        theta = greeks['theta'] * lot_size
        
        # Risk calculations
        
        # Delta Risk: Adverse move of 2%
        adverse_move = spot * config.DELTA_RISK_MOVE
        delta_risk = abs(delta * adverse_move) * config.DELTA_RISK_BUFFER
        
        # Gamma Risk: Convexity risk from 2% move
        gamma_move = spot * config.GAMMA_RISK_MOVE
        gamma_risk = 0.5 * abs(gamma) * (gamma_move ** 2) * config.GAMMA_RISK_BUFFER
        
        # Vega Risk: 1% IV increase
        vega_risk = abs(vega) * config.VEGA_RISK_IV_CHANGE * 100 * config.VEGA_RISK_BUFFER
        
        # Total risk
        total_risk = delta_risk + gamma_risk + vega_risk
        
        return {
            'theta_daily': theta,
            'delta_risk': delta_risk,
            'gamma_risk': gamma_risk,
            'vega_risk': vega_risk,
            'total_risk': max(total_risk, 100),  # Min risk floor
            'position_delta': delta
        }
    
    @staticmethod
    def calculate_edge_ratio(theta: float, total_risk: float) -> float:
        """
        Calculate Edge Ratio = Theta / Total Risk
        
        Your income per unit of risk
        Must be > 0.75 to trade
        """
        if total_risk <= 0:
            return 0.0
        
        return abs(theta) / total_risk
    
    @staticmethod
    def is_delta_neutral(position_delta: float) -> bool:
        """
        Check if position is delta neutral
        Position delta should be < 0.15 for neutral trades
        """
        return abs(position_delta) < config.MAX_DELTA_NEUTRAL
