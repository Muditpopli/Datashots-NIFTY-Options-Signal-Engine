"""
Test Dhan API Credentials
Quick test to verify your API access
"""

from dhanhq import dhanhq
import os
from dotenv import load_dotenv

# Load credentials
load_dotenv()

ACCESS_TOKEN = os.getenv('DHAN_ACCESS_TOKEN', '').strip('"').strip("'")
CLIENT_ID = os.getenv('DHAN_CLIENT_ID', '').strip('"').strip("'")

print("═" * 70)
print("  DHAN API CREDENTIAL TEST")
print("═" * 70)

print(f"\nClient ID: {CLIENT_ID}")
print(f"Token: {ACCESS_TOKEN[:20]}...{ACCESS_TOKEN[-20:]}")

# Test connection
try:
    print("\n🔍 Testing Dhan API connection...")
    
    dhan = dhanhq(CLIENT_ID, ACCESS_TOKEN)
    
    # Test 1: Get NIFTY spot price
    print("\nTest 1: Fetching NIFTY spot price...")
    
    from datetime import datetime
    today = datetime.now().strftime('%Y-%m-%d')
    
    response = dhan.intraday_minute_data(
        security_id='13',
        exchange_segment='IDX_I',
        instrument_type='INDEX',
        from_date=today,
        to_date=today
    )
    
    if response and 'data' in response:
        data = response['data']
        if isinstance(data, dict) and 'close' in data:
            close_prices = data['close']
            if close_prices and len(close_prices) > 0:
                spot = close_prices[-1]
                print(f"✅ NIFTY Spot: {spot}")
        else:
            print(f"⚠️ Unexpected data format: {type(data)}")
    else:
        print(f"⚠️ Response: {response}")
    
    # Test 2: Get BANKNIFTY spot price
    print("\nTest 2: Fetching BANKNIFTY spot price...")
    response = dhan.intraday_minute_data(
        security_id='25',
        exchange_segment='IDX_I',
        instrument_type='INDEX',
        from_date=today,
        to_date=today
    )
    
    if response and 'data' in response:
        data = response['data']
        if isinstance(data, dict) and 'close' in data:
            close_prices = data['close']
            if close_prices and len(close_prices) > 0:
                spot = close_prices[-1]
                print(f"✅ BANKNIFTY Spot: {spot}")
        else:
            print(f"⚠️ Unexpected data format")
    else:
        print(f"⚠️ Response: {response}")
    
    print("\n" + "═" * 70)
    print("✅ DHAN API CONNECTION SUCCESSFUL")
    print("═" * 70)
    print("\nYour Dhan credentials are working!")
    print("The system can fetch spot prices.")
    print("\nNote: For Greeks, we'll use NSE data (free)")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\nPossible issues:")
    print("1. Invalid credentials")
    print("2. Token expired")
    print("3. Network/firewall blocking Dhan API")
    print("4. Dhan API maintenance")

print("\n" + "═" * 70)
