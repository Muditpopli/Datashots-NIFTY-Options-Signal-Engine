"""
DNS Scanner Test Suite
Validates all components without requiring live market data
"""

import sys
from greeks import GreeksCalculator, RiskCalculator
import config


def test_greeks_calculator():
    """Test Black-Scholes Greeks calculation"""
    
    print("\n" + "═" * 70)
    print("  TEST 1: Greeks Calculator")
    print("═" * 70)
    
    # Test parameters
    spot = 23500
    strike = 23500  # ATM
    dte = 7
    iv = 0.18
    
    print(f"\nTest Case: NIFTY ATM Option")
    print(f"  Spot: {spot} | Strike: {strike} | DTE: {dte} | IV: {iv*100}%")
    
    # Calculate CE Greeks
    ce = GreeksCalculator.calculate_greeks(spot, strike, dte, iv, 'CE')
    
    print(f"\nCall Option (CE):")
    print(f"  Premium: ₹{ce['premium']:.2f}")
    print(f"  Delta: {ce['delta']:.3f} (0 to 1 for calls)")
    print(f"  Gamma: {ce['gamma']:.5f} (curvature)")
    print(f"  Theta: ₹{ce['theta']:.2f} daily")
    print(f"  Vega: ₹{ce['vega']:.2f} per 1% IV")
    
    # Calculate PE Greeks
    pe = GreeksCalculator.calculate_greeks(spot, strike, dte, iv, 'PE')
    
    print(f"\nPut Option (PE):")
    print(f"  Premium: ₹{pe['premium']:.2f}")
    print(f"  Delta: {pe['delta']:.3f} (-1 to 0 for puts)")
    print(f"  Gamma: {pe['gamma']:.5f}")
    print(f"  Theta: ₹{pe['theta']:.2f} daily")
    print(f"  Vega: ₹{pe['vega']:.2f} per 1% IV")
    
    # Validate
    assert 0 <= ce['delta'] <= 1, "Call delta out of range"
    assert -1 <= pe['delta'] <= 0, "Put delta out of range"
    assert ce['gamma'] == pe['gamma'], "Gamma should be same for calls and puts"
    assert ce['vega'] == pe['vega'], "Vega should be same for calls and puts"
    
    print("\n✅ Greeks Calculator: PASSED")
    return True


def test_risk_calculator():
    """Test risk calculations and edge ratio"""
    
    print("\n" + "═" * 70)
    print("  TEST 2: Risk Calculator & Edge Ratio")
    print("═" * 70)
    
    # Sample Greeks for a short put
    greeks = {
        'delta': -0.45,
        'gamma': 0.00065,
        'theta': 16.5,
        'vega': 9.7
    }
    
    lot_size = 50
    spot = 23500
    
    print(f"\nTest Case: Short NIFTY 23450 PE")
    print(f"  Delta: {greeks['delta']}")
    print(f"  Gamma: {greeks['gamma']}")
    print(f"  Theta: ₹{greeks['theta']}")
    print(f"  Vega: ₹{greeks['vega']}")
    print(f"  Lot Size: {lot_size}")
    
    # Calculate risks
    risks = RiskCalculator.calculate_risks(greeks, lot_size, spot)
    
    print(f"\nRisk Assessment (1 lot):")
    print(f"  Theta Daily: ₹{risks['theta_daily']:,.0f}")
    print(f"  Delta Risk (2% move): ₹{risks['delta_risk']:,.0f}")
    print(f"  Gamma Risk (2% move): ₹{risks['gamma_risk']:,.0f}")
    print(f"  Vega Risk (1% IV): ₹{risks['vega_risk']:,.0f}")
    print(f"  Total Risk: ₹{risks['total_risk']:,.0f}")
    
    # Edge ratio
    edge = RiskCalculator.calculate_edge_ratio(
        risks['theta_daily'], 
        risks['total_risk']
    )
    
    print(f"\nEdge Ratio: {edge:.2f}x")
    
    if edge >= config.MIN_EDGE_RATIO:
        print(f"✅ GOOD TRADE (edge > {config.MIN_EDGE_RATIO})")
    else:
        print(f"❌ POOR TRADE (edge < {config.MIN_EDGE_RATIO})")
    
    print(f"\nInterpretation:")
    print(f"  You earn ₹{risks['theta_daily']:,.0f} per day")
    print(f"  You risk ₹{risks['total_risk']:,.0f} in worst case")
    print(f"  Edge = {edge:.2f}x means {edge*100:.0f}¢ income per ₹1 risk")
    
    assert risks['theta_daily'] > 0, "Theta should be positive for short options"
    assert risks['total_risk'] > 0, "Total risk should be positive"
    assert 0 <= edge <= 2, "Edge ratio should be reasonable"
    
    print("\n✅ Risk Calculator: PASSED")
    return True


def test_delta_neutrality():
    """Test delta neutrality check"""
    
    print("\n" + "═" * 70)
    print("  TEST 3: Delta Neutrality Check")
    print("═" * 70)
    
    print(f"\nMax Delta Neutral Threshold: {config.MAX_DELTA_NEUTRAL}")
    
    test_cases = [
        (0.05, True, "Very neutral"),
        (0.12, True, "Slightly biased but OK"),
        (0.14, True, "At edge of neutral"),
        (0.18, False, "Too directional"),
        (-0.08, True, "Neutral (negative)"),
        (-0.22, False, "Too bearish")
    ]
    
    print("\nTest Cases:")
    for delta, should_be_neutral, description in test_cases:
        is_neutral = RiskCalculator.is_delta_neutral(delta)
        status = "✓" if is_neutral == should_be_neutral else "✗"
        print(f"  {status} Delta: {delta:+.2f} → {'Neutral' if is_neutral else 'Directional'} ({description})")
        
        assert is_neutral == should_be_neutral, f"Delta neutrality check failed for {delta}"
    
    print("\n✅ Delta Neutrality: PASSED")
    return True


def test_synthetic_future():
    """Test synthetic future calculation"""
    
    print("\n" + "═" * 70)
    print("  TEST 4: Synthetic Future (ATM Calculation)")
    print("═" * 70)
    
    # Test case: ATM straddle
    strike = 23500
    call_premium = 195.5
    put_premium = 188.2
    dte = 7
    
    print(f"\nTest Case: ATM Straddle")
    print(f"  Strike: {strike}")
    print(f"  Call Premium: ₹{call_premium}")
    print(f"  Put Premium: ₹{put_premium}")
    print(f"  DTE: {dte} days")
    
    synthetic = GreeksCalculator.calculate_synthetic_future(
        call_premium, put_premium, strike, dte
    )
    
    print(f"\nSynthetic Future: {synthetic:.2f}")
    print(f"Premium Difference: ₹{call_premium - put_premium:.2f}")
    
    # Synthetic should be close to strike + premium difference
    expected_approx = strike + (call_premium - put_premium)
    print(f"Expected (approx): {expected_approx:.2f}")
    
    # Should be within 1% of strike for ATM
    assert abs(synthetic - strike) / strike < 0.01, "Synthetic too far from strike"
    
    print("\n✅ Synthetic Future: PASSED")
    return True


def test_configuration():
    """Test configuration values"""
    
    print("\n" + "═" * 70)
    print("  TEST 5: Configuration")
    print("═" * 70)
    
    print(f"\nMarket Settings:")
    print(f"  Indices: {config.INDICES}")
    print(f"  Strike Gaps: {config.STRIKE_GAPS}")
    print(f"  Lot Sizes: {config.LOT_SIZES}")
    
    print(f"\nSignal Thresholds:")
    print(f"  Bullish: > {config.BULLISH_THRESHOLD}")
    print(f"  Bearish: < {config.BEARISH_THRESHOLD}")
    print(f"  Min Confidence: {config.MIN_CONFIDENCE}")
    
    print(f"\nEdge Ratio Settings:")
    print(f"  Minimum Edge: {config.MIN_EDGE_RATIO}x")
    print(f"  Delta Risk Buffer: {config.DELTA_RISK_BUFFER}x")
    print(f"  Gamma Risk Buffer: {config.GAMMA_RISK_BUFFER}x")
    print(f"  Vega Risk Buffer: {config.VEGA_RISK_BUFFER}x")
    
    print(f"\nDelta Neutrality:")
    print(f"  Max Delta: {config.MAX_DELTA_NEUTRAL}")
    
    # Validate
    assert config.MIN_EDGE_RATIO > 0, "Edge ratio should be positive"
    assert config.BULLISH_THRESHOLD > 0, "Bullish threshold should be positive"
    assert config.BEARISH_THRESHOLD < 0, "Bearish threshold should be negative"
    assert 0 < config.MAX_DELTA_NEUTRAL < 1, "Max delta neutral should be between 0 and 1"
    
    print("\n✅ Configuration: PASSED")
    return True


def run_all_tests():
    """Run complete test suite"""
    
    print("\n" + "═" * 70)
    print("  🧪 DNS SCANNER - TEST SUITE")
    print("═" * 70)
    print("\nValidating all components...")
    
    tests = [
        ("Greeks Calculator", test_greeks_calculator),
        ("Risk Calculator", test_risk_calculator),
        ("Delta Neutrality", test_delta_neutrality),
        ("Synthetic Future", test_synthetic_future),
        ("Configuration", test_configuration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed, None))
        except Exception as e:
            results.append((test_name, False, str(e)))
    
    # Summary
    print("\n" + "═" * 70)
    print("  📊 TEST RESULTS")
    print("═" * 70)
    
    passed_count = sum(1 for _, passed, _ in results if passed)
    total_count = len(results)
    
    for test_name, passed, error in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")
        if error:
            print(f"         Error: {error}")
    
    print("\n" + "─" * 70)
    print(f"  {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n✅ ALL TESTS PASSED!")
        print("\nSystem is ready to use.")
        print("Next steps:")
        print("  1. Add Dhan credentials to .env")
        print("  2. Run 'python main.py baseline' at 9:15 AM")
        print("  3. Run 'python main.py analyze' at 10:00 AM")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("Please check errors above")
    
    print("═" * 70)
    
    return passed_count == total_count


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
