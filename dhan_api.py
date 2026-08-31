"""
FINAL DHAN API - Using Real option_chain() Method
FULLY AUTOMATED - NO SCREENSHOTS!
"""

from dhanhq import dhanhq
from typing import Dict, List, Optional
import config
from datetime import datetime, timedelta


class FinalDhanAPI:
    """
    Uses Dhan's actual option_chain() method
    FULLY AUTOMATED!
    """

    def __init__(self):
        token = config.DHAN_ACCESS_TOKEN.strip('"').strip("'")
        client = config.DHAN_CLIENT_ID.strip('"').strip("'")
        self.dhan = dhanhq(client, token)
        self.cache = {}
        self.quiet_logs = bool(getattr(config, "QUIET_API_LOGS", True))

    def _info(self, msg: str) -> None:
        if not self.quiet_logs:
            print(msg)

    def _warn(self, msg: str) -> None:
        if not self.quiet_logs:
            print(msg)

    def get_spot_price(self, index: str) -> Optional[float]:
        """Get spot price from Dhan with resilient fallbacks."""
        security_ids = {"NIFTY": 13, "BANKNIFTY": 25}
        security_id = str(security_ids[index])

        def _to_float(value) -> Optional[float]:
            try:
                num = float(value)
                return num if num > 0 else None
            except (TypeError, ValueError):
                return None

        def _extract_price(payload) -> Optional[float]:
            if payload is None:
                return None

            if isinstance(payload, (int, float, str)):
                return _to_float(payload)

            if isinstance(payload, list):
                for item in reversed(payload):
                    price = _extract_price(item)
                    if price:
                        return price
                return None

            if isinstance(payload, dict):
                # Prefer direct spot fields first.
                for key in ("last_price", "ltp", "close", "price"):
                    if key in payload:
                        val = payload[key]
                        if isinstance(val, list):
                            for x in reversed(val):
                                price = _to_float(x)
                                if price:
                                    return price
                        else:
                            price = _to_float(val)
                            if price:
                                return price

                # Fallback: walk nested structures.
                for value in payload.values():
                    price = _extract_price(value)
                    if price:
                        return price

            return None

        try:
            today = datetime.now().strftime("%Y-%m-%d")

            response = self.dhan.intraday_minute_data(
                security_id=security_id,
                exchange_segment="IDX_I",
                instrument_type="INDEX",
                from_date=today,
                to_date=today,
            )
            price = _extract_price(response.get("data") if isinstance(response, dict) else None)
            if price:
                return price
        except Exception as e:
            self._warn(f"[WARN] Spot via intraday failed: {e}")

        try:
            response = self.dhan.ticker_data({"IDX_I": [int(security_id)]})
            price = _extract_price(response.get("data") if isinstance(response, dict) else None)
            if price:
                return price
        except Exception as e:
            self._warn(f"[WARN] Spot via ticker failed: {e}")

        try:
            response = self.dhan.quote_data({"IDX_I": [int(security_id)]})
            price = _extract_price(response.get("data") if isinstance(response, dict) else None)
            if price:
                return price
        except Exception as e:
            self._warn(f"[WARN] Spot via quote failed: {e}")

        return None

    def get_current_expiry(self, index: str) -> Optional[str]:
        """
        Get current weekly expiry from Dhan
        Skip today's expiry if after 3:30 PM
        """
        expiries = self.get_expiry_list(index)
        if not expiries:
            self._warn("[WARN] Using fallback expiry...")
            today = datetime.now()
            days_ahead = 3 - today.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            next_thursday = today + timedelta(days=days_ahead)
            return next_thursday.strftime("%Y-%m-%d")

        today = datetime.now()
        cutoff = today.replace(hour=15, minute=30, second=0, microsecond=0)

        for expiry in expiries:
            expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
            if expiry_date > today.date():
                self._info(f"[OK] Using future expiry: {expiry}")
                return expiry
            if expiry_date == today.date() and today <= cutoff:
                self._info(f"[OK] Using today's expiry: {expiry}")
                return expiry

        self._warn(f"[WARN] Using first available expiry: {expiries[0]}")
        return expiries[0]

    def get_expiry_list(self, index: str) -> List[str]:
        """Fetch sorted expiry list from Dhan."""
        try:
            security_ids = {"NIFTY": 13, "BANKNIFTY": 25}
            under_security_id = security_ids[index]

            self._info("[INFO] Fetching expiry list from Dhan...")
            response = self.dhan.expiry_list(
                under_security_id=under_security_id,
                under_exchange_segment="IDX_I",
            )

            if not response or "data" not in response:
                return []

            data = response["data"]
            expiries = data.get("data") if isinstance(data, dict) and "data" in data else data

            if not isinstance(expiries, list):
                return []

            expiries = sorted(expiries)
            self._info(f"[INFO] Available expiries: {expiries[:5]}")
            return expiries
        except Exception as e:
            print(f"[ERR] Error fetching expiry list: {e}")
            return []

    def get_next_expiry(self, index: str) -> Optional[str]:
        """Get next expiry after current expiry."""
        expiries = self.get_expiry_list(index)
        if len(expiries) < 2:
            return None
        current = self.get_current_expiry(index)
        if not current:
            return expiries[1]
        try:
            idx = expiries.index(current)
            return expiries[idx + 1] if idx + 1 < len(expiries) else None
        except ValueError:
            return expiries[1]

    def get_option_chain(self, index: str, use_cache: bool = True) -> Optional[Dict]:
        """
        Get option chain using Dhan's option_chain() method
        FULLY AUTOMATED!
        """
        expiry = self.get_current_expiry(index)
        if not expiry:
            return None
        return self.get_option_chain_for_expiry(index=index, expiry=expiry, use_cache=use_cache)

    def get_option_chain_for_expiry(
        self,
        index: str,
        expiry: str,
        use_cache: bool = True,
    ) -> Optional[Dict]:
        """Fetch option chain for a specific expiry."""
        cache_key = f"{index}:{expiry}"
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]

        try:
            self._info(f"\n[INFO] Fetching {index} option chain from Dhan API...")

            spot = self.get_spot_price(index)
            if spot is None:
                print("[ERR] Cannot fetch spot")
                return None
            self._info(f"[OK] Spot: {spot:.2f}")
            self._info(f"[INFO] Expiry: {expiry}")

            security_ids = {"NIFTY": 13, "BANKNIFTY": 25}
            under_security_id = security_ids[index]

            self._info("[INFO] Fetching option chain...")
            response = self.dhan.option_chain(
                under_security_id=under_security_id,
                under_exchange_segment="IDX_I",
                expiry=expiry,
            )

            if not response or "data" not in response:
                print(f"[ERR] No data from Dhan: {response}")
                return None

            # Surface nested API status early (e.g., throttling/temporary failures).
            nested = response.get("data")
            if isinstance(nested, dict):
                nested_status = str(nested.get("status", "")).lower()
                if nested_status and nested_status != "success":
                    nested_data = nested.get("data")
                    if isinstance(nested_data, dict):
                        if "805" in nested_data:
                            print(f"[ERR] Dhan throttling on option chain: {nested_data.get('805')}")
                        else:
                            print(f"[ERR] Dhan returned non-success status for option chain: {nested_data}")
                    else:
                        print(f"[ERR] Dhan returned non-success status for option chain: {nested}")
                    return None

            self._info("[OK] Option chain received from Dhan")

            strikes = self._parse_dhan_option_chain(response, spot, index)
            if not strikes:
                print("[ERR] Failed to parse option chain")
                return None

            strike_gap = config.STRIKE_GAPS[index]
            atm = int(round(spot / strike_gap) * strike_gap)

            result = {
                "index": index,
                "spot": spot,
                "atm": atm,
                "expiry": expiry,
                "strikes": strikes,
            }

            self.cache[cache_key] = result
            self._info(f"[OK] Option chain ready: {len(strikes)} strikes")
            return result
        except Exception as e:
            print(f"[ERR] Error: {e}")
            import traceback

            traceback.print_exc()
            return None

    def get_multi_expiry_chain(self, index: str) -> Optional[Dict]:
        """
        Compatibility method for flows expecting multi-expiry data.
        Returns current and next expiry chains in strict real-data mode.
        """
        expiries = self.get_expiry_list(index)
        if not expiries:
            return None

        today = datetime.now()
        cutoff = today.replace(hour=15, minute=30, second=0, microsecond=0)
        current_expiry = None
        current_idx = None

        for idx, expiry in enumerate(expiries):
            expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
            if expiry_date > today.date():
                current_expiry = expiry
                current_idx = idx
                break
            if expiry_date == today.date() and today <= cutoff:
                current_expiry = expiry
                current_idx = idx
                break

        if current_expiry is None:
            current_expiry = expiries[0]
            current_idx = 0

        next_expiry = (
            expiries[current_idx + 1]
            if current_idx is not None and current_idx + 1 < len(expiries)
            else None
        )
        if not current_expiry:
            return None

        current_chain = self.get_option_chain_for_expiry(
            index=index, expiry=current_expiry, use_cache=False
        )
        next_chain = (
            self.get_option_chain_for_expiry(index=index, expiry=next_expiry, use_cache=False)
            if next_expiry
            else None
        )

        if not current_chain:
            return None

        return {
            "index": index,
            "spot": current_chain["spot"],
            "current_expiry": current_expiry,
            "next_expiry": next_expiry,
            "current_chain": current_chain,
            "next_chain": next_chain,
        }

    def _extract_oc(self, chain_data) -> Optional[Dict]:
        """
        Normalize Dhan payload and return oc dict if present.

        Supported shapes:
        1) {'data': {'oc': {...}}}
        2) {'data': {'data': {'oc': {...}}}}
        3) {'oc': {...}}
        """
        if not isinstance(chain_data, dict):
            return None

        def _walk(payload):
            if isinstance(payload, dict):
                oc = payload.get("oc")
                if isinstance(oc, dict):
                    return oc
                for value in payload.values():
                    found = _walk(value)
                    if isinstance(found, dict):
                        return found
            elif isinstance(payload, list):
                for value in payload:
                    found = _walk(value)
                    if isinstance(found, dict):
                        return found
            return None

        # Recursively search full payload for an `oc` dict.
        found = _walk(chain_data)
        if isinstance(found, dict):
            return found

        return None

    def _parse_dhan_option_chain(self, chain_data, spot: float, index: str) -> List[Dict]:
        """
        Parse Dhan's option chain response.
        """
        strikes = []
        skipped_strikes = 0
        reject_reasons = {}

        def _log_reject(reason: str, strike: int, leg_name: str, extra: str = "") -> None:
            reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
            if config.VERBOSE_REJECT_LOGS:
                suffix = f" ({extra})" if extra else ""
                print(f"[WARN] {strike} {leg_name}: {reason}{suffix}")

        def _parse_leg(leg_data: Dict, leg_name: str, strike: int) -> Optional[Dict]:
            """
            Parse one option leg (CE/PE) with strict real-data validation.
            Rejects leg if any required market field/greek is missing.
            """
            if not isinstance(leg_data, dict):
                _log_reject("leg payload missing", strike, leg_name)
                return None

            greeks = leg_data.get("greeks")
            if not isinstance(greeks, dict):
                _log_reject("greeks missing", strike, leg_name)
                return None

            required_fields = [
                ("last_price", leg_data.get("last_price")),
                ("implied_volatility", leg_data.get("implied_volatility")),
                ("oi", leg_data.get("oi")),
                ("delta", greeks.get("delta")),
                ("gamma", greeks.get("gamma")),
                ("theta", greeks.get("theta")),
                ("vega", greeks.get("vega")),
            ]

            missing = [name for name, value in required_fields if value is None]
            if missing:
                _log_reject("missing required fields", strike, leg_name, ",".join(missing))
                return None

            try:
                premium = float(leg_data["last_price"])
                iv = float(leg_data["implied_volatility"])
                delta = float(greeks["delta"])
                gamma = float(greeks["gamma"])
                theta = float(greeks["theta"])
                vega = float(greeks["vega"])
                volume = float(leg_data.get("volume", 0))
                oi = float(leg_data["oi"])
            except (TypeError, ValueError) as e:
                _log_reject("non-numeric field", strike, leg_name, str(e))
                return None

            # Strict mode: do not accept legs without live tradable premium or IV.
            if premium <= 0:
                _log_reject("invalid premium", strike, leg_name, str(premium))
                return None
            if iv <= 0:
                _log_reject("invalid IV", strike, leg_name, str(iv))
                return None

            return {
                "premium": premium,
                "iv": iv,
                "delta": delta,
                "gamma": gamma,
                "theta": theta,
                "vega": vega,
                "volume": volume,
                "oi": oi,
            }

        try:
            oc = self._extract_oc(chain_data)
            if not oc:
                keys_lvl1 = list(chain_data.keys()) if isinstance(chain_data, dict) else []
                keys_lvl2 = []
                if isinstance(chain_data, dict) and isinstance(chain_data.get("data"), dict):
                    keys_lvl2 = list(chain_data["data"].keys())
                print(f"[ERR] No 'oc' key. top_keys={keys_lvl1}, data_keys={keys_lvl2}")
                return []

            for strike_str, strike_data in oc.items():
                try:
                    strike = int(float(strike_str))

                    ce_data = strike_data.get("ce", {})
                    pe_data = strike_data.get("pe", {})
                    ce = _parse_leg(ce_data, "CE", strike)
                    pe = _parse_leg(pe_data, "PE", strike)

                    # Strict real-data mode: include only fully valid CE+PE strikes.
                    if ce and pe:
                        strikes.append(
                            {
                                "strike": strike,
                                "ce": ce,
                                "pe": pe,
                            }
                        )
                    else:
                        skipped_strikes += 1

                except Exception as e:
                    self._warn(f"[WARN] Error parsing strike {strike_str}: {e}")
                    continue

            if skipped_strikes > 0:
                self._warn(f"[WARN] Skipped {skipped_strikes} strikes due to missing/invalid real Greeks")
                if reject_reasons:
                    summary = ", ".join(
                        f"{reason}={count}" for reason, count in sorted(reject_reasons.items())
                    )
                    self._warn(f"[WARN] Reject summary: {summary}")

            strikes.sort(key=lambda x: x["strike"])
            return strikes

        except Exception as e:
            print(f"[ERR] Error in _parse_dhan_option_chain: {e}")
            import traceback

            traceback.print_exc()
            return []

    def _extract_strike_data(self, item, strike_price=None) -> Optional[Dict]:
        """Not needed anymore - using new parser"""
        return None


# Singleton
_instance = None


def get_dhan() -> FinalDhanAPI:
    global _instance
    if _instance is None:
        _instance = FinalDhanAPI()
    return _instance


# Test
if __name__ == "__main__":
    print("=" * 70)
    print("  DHAN API - AUTOMATED OPTION CHAIN")
    print("=" * 70)

    api = FinalDhanAPI()

    chain = api.get_option_chain("NIFTY", use_cache=False)

    if chain:
        print("\n[OK] SUCCESS!")
        print(f"   Spot: {chain['spot']:.2f}")
        print(f"   ATM: {chain['atm']}")
        print(f"   Strikes: {len(chain['strikes'])}")

        atm_strikes = [s for s in chain["strikes"] if s["strike"] == chain["atm"]]
        if atm_strikes:
            atm = atm_strikes[0]
            print(f"\n   ATM {chain['atm']}:")
            print(f"   CE Premium: {atm['ce']['premium']:.2f}")
            print(f"   CE IV: {atm['ce']['iv']:.2f}%")
            print(f"   PE Premium: {atm['pe']['premium']:.2f}")
            print(f"   PE IV: {atm['pe']['iv']:.2f}%")

        print("\n" + "=" * 70)
        print("System ready for trading!, Happy Trading 😊")
        print("=" * 70)
    else:
        print("\n[ERR] Failed - check error messages above")
