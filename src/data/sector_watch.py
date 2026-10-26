"""The oil and gas stocks the swing bot always considers (2026-10-23, at the owner's request).

The screener keeps only the 15 best 12-1 month momentum stocks of those that pass its filters, so an oil or gas stock that
passes every filter (liquid, calm enough, price above its 50- and 200-day averages, RSI under 70) but ranks lower is never
looked at. Stocks on this list that pass the screener are added after its top 15. They keep every filter, go through the
same AI and risk engine and the same stop, and their decisions are tagged `+oilgas`. A stock in a downtrend is not added:
that would be a bet on the sector, which a ten-year comparison does not support (STRATEGY.md, 2026-10-23)."""

OIL_GAS = frozenset(
    {
        "RELIANCE", "ONGC", "IOC", "BPCL", "HINDPETRO", "OIL", "GAIL", "PETRONET", "MGL", "IGL", "GUJGASLTD", "GSPL",
        "CASTROLIND", "CHENNPETRO", "MRPL", "GULFOILLUB", "HINDOILEXP", "ATGL", "DEEPINDS", "SELAN",
    }
)  # fmt: skip
