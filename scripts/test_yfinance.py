"""Verify yfinance can fetch market data (fallback data source behind Alpaca)."""
import sys

import yfinance as yf

def main():
    df = yf.download("AAPL", period="1mo", progress=False)

    if df.empty:
        print("FAIL: yfinance returned no data for AAPL")
        sys.exit(1)

    # yfinance returns MultiIndex columns (Price, Ticker) even for a single symbol.
    latest_close = float(df["Close"].iloc[-1].item())
    print(f"OK: fetched {len(df)} rows for AAPL, latest close=${latest_close:.2f}")
    print("\nAll yfinance checks passed.")

if __name__ == "__main__":
    main()
