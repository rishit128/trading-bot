"""Read-only check of Angel One SmartAPI market data against Yahoo, before the bot depends on it.

Logs in, downloads daily candles (and today's 5-minute candles) for a few NSE stocks, compares them with Yahoo, logs out.
It calls only the login, candle and logout endpoints: nothing here can place, change or cancel an order.

Run it yourself in a terminal (it asks for your login, nothing is saved):
    venv\\Scripts\\python scripts\\check_angel_data.py [SYMBOL ...]
The API key comes from ANGEL_API_KEY if set, else it is asked for. The PIN is hidden as you type; the TOTP is the 6-digit
code from your authenticator app (not the secret behind it).
"""
import getpass
import os
import sys
from datetime import datetime, timedelta

import httpx
import pandas as pd
import yfinance as yf

BASE = "https://apiconnect.angelone.in"
SCRIP_MASTER = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
DEFAULT_SYMBOLS = ["RELIANCE", "TCS", "HDFCBANK", "SONACOMS", "MBAPL"]


def headers(api_key: str, jwt: str | None = None) -> dict[str, str]:
    h = {"Content-Type": "application/json", "Accept": "application/json", "X-UserType": "USER", "X-SourceID": "WEB",
         "X-ClientLocalIP": "127.0.0.1", "X-ClientPublicIP": "127.0.0.1", "X-MACAddress": "00:00:00:00:00:00",
         "X-PrivateKey": api_key}
    if jwt:
        h["Authorization"] = f"Bearer {jwt}"
    return h


def tokens(symbols: list[str]) -> dict[str, str]:
    rows = httpx.get(SCRIP_MASTER, timeout=120).json()
    wanted = {f"{s}-EQ": s for s in symbols}
    return {wanted[r["symbol"]]: r["token"] for r in rows if r["exch_seg"] == "NSE" and r["symbol"] in wanted}


def candles(client: httpx.Client, api_key: str, jwt: str, token: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
    body = {"exchange": "NSE", "symboltoken": token, "interval": interval,
            "fromdate": start.strftime("%Y-%m-%d %H:%M"), "todate": end.strftime("%Y-%m-%d %H:%M")}
    r = client.post(f"{BASE}/rest/secure/angelbroking/historical/v1/getCandleData", json=body, headers=headers(api_key, jwt))
    payload = r.json()
    if not payload.get("status") or not payload.get("data"):
        raise RuntimeError(f"{payload.get('errorcode')} {payload.get('message')}")
    df = pd.DataFrame(payload["data"], columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


def compare(symbol: str, angel: pd.DataFrame) -> str:
    yahoo = yf.Ticker(f"{symbol}.NS").history(period="3mo", interval="1d", auto_adjust=False)
    if yahoo.empty:
        return "Yahoo returned nothing"
    a = angel.copy()
    a.index = a.index.date
    y = yahoo.copy()
    y.index = y.index.date
    both = a.join(y[["Close", "Volume"]], how="inner")
    if both.empty:
        return "no common dates"
    gap = ((both["close"] - both["Close"]).abs() / both["Close"]).max()
    vol = both["volume"].sum() / max(both["Volume"].sum(), 1)
    only_angel, only_yahoo = len(set(a.index) - set(y.index)), len(set(y.index) - set(a.index))
    return (f"{len(both)} common days, largest close gap {gap:.2%}, volume Angel/Yahoo {vol:.0%}, "
            f"days only in Angel {only_angel}, only in Yahoo {only_yahoo}")


def main() -> None:
    symbols = [s.upper() for s in sys.argv[1:]] or DEFAULT_SYMBOLS
    api_key = os.environ.get("ANGEL_API_KEY") or input("SmartAPI API key: ").strip()
    client_code = input("Angel One client ID: ").strip()
    pin = getpass.getpass("PIN (hidden): ").strip()
    totp = input("6-digit code from your authenticator app: ").strip()

    found = tokens(symbols)
    missing = sorted(set(symbols) - set(found))
    if missing:
        print(f"not in Angel's NSE list: {missing}")

    with httpx.Client(timeout=30) as client:
        login = client.post(f"{BASE}/rest/auth/angelbroking/user/v1/loginByPassword",
                            json={"clientcode": client_code, "password": pin, "totp": totp}, headers=headers(api_key)).json()
        del pin
        if not login.get("status"):
            sys.exit(f"login failed: {login.get('errorcode')} {login.get('message')}")
        jwt = login["data"]["jwtToken"]
        print("login OK\n")
        try:
            end = datetime.now()
            for symbol, token in found.items():
                try:
                    daily = candles(client, api_key, jwt, token, "ONE_DAY", end - timedelta(days=90), end)
                    print(f"{symbol}: daily {len(daily)} bars, last {daily.index[-1].date()} close {daily['close'].iloc[-1]:,.2f}")
                    print(f"   vs Yahoo: {compare(symbol, daily)}")
                    intraday = candles(client, api_key, jwt, token, "FIVE_MINUTE", end - timedelta(days=5), end)
                    print(f"   5-minute: {len(intraday)} bars, last {intraday.index[-1]} close {intraday['close'].iloc[-1]:,.2f}")
                except Exception as e:  # one stock failing should not hide the others
                    print(f"{symbol}: FAILED {type(e).__name__}: {e}")
        finally:
            client.post(f"{BASE}/rest/secure/angelbroking/user/v1/logout", json={"clientcode": client_code},
                        headers=headers(api_key, jwt))
            print("\nlogged out")


if __name__ == "__main__":
    main()
