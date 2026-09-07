"""Long-history return proxies from free academic sources.

The binding constraint on regime-conditional estimation is not months, it is
CYCLES. ETF history alone gives two recessions; that is anecdote, not estimation.

This module supplies the pre-ETF history:

* **Equity** - the Ken French data library's market factor, back to 1926. A genuine
  total return (``Mkt-RF + RF``), free, no key, and the standard series in the
  academic literature. Takes equity from 2 recessions to roughly 15.
* **Gold** - gold futures from 2000. Honest limitation: no free bullion series with
  1970s history survives (FRED's LBMA fix was withdrawn over licensing), so
  gold-specific results rest on a short sample and are labelled as such. The
  inflation-hedge ROLE is still covered over the long run by commodities, whose
  PPI-based proxy reaches back to 1913.
"""

from __future__ import annotations

import io
import logging
import zipfile

import pandas as pd

from . import cache

log = logging.getLogger(__name__)

FF_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_CSV.zip"
)


def fetch_french_market_return(refresh: bool = False) -> pd.Series | None:
    """US equity market TOTAL return, monthly, from 1926-07.

    Ken French publishes the market factor as an excess return, so the total
    return is ``Mkt-RF + RF``. Values in the file are percentages.
    """
    key = "equity_french_market"

    if not refresh:
        cached = cache.load_series(key, subdir="assets")
        if cached is not None:
            return cached

    try:
        import requests

        resp = requests.get(FF_URL, timeout=90)
        resp.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            text = z.read(z.namelist()[0]).decode("latin-1")

        rows = []
        for line in text.split("\n"):
            parts = [p.strip() for p in line.split(",")]
            # monthly rows are keyed YYYYMM; the file later switches to annual
            if len(parts) >= 5 and parts[0].isdigit() and len(parts[0]) == 6:
                rows.append(parts[:5])
            elif rows and parts[0] and not (parts[0].isdigit() and len(parts[0]) == 6):
                break

        if not rows:
            log.warning("Ken French file parsed to zero monthly rows")
            return None

        df = pd.DataFrame(rows, columns=["ym", "MktRF", "SMB", "HML", "RF"])
        df["MktRF"] = df["MktRF"].astype(float) / 100.0
        df["RF"] = df["RF"].astype(float) / 100.0
        df["date"] = pd.to_datetime(df["ym"], format="%Y%m") + pd.offsets.MonthEnd(0)

        mkt = (df["MktRF"] + df["RF"]).set_axis(df["date"]).sort_index()
        mkt.name = "equity_proxy_return"
        mkt = mkt[mkt.abs() < 1.0]  # guard against sentinel values like -99.99

        cache.save_series(key, mkt, subdir="assets")
        log.info("Ken French market return: %d months from %s",
                 len(mkt), mkt.index.min().date())
        return mkt

    except Exception as exc:  # noqa: BLE001
        log.warning("Ken French fetch failed: %s", exc)
        return cache.load_series(key, subdir="assets")


def fetch_gold_futures(refresh: bool = False) -> pd.Series | None:
    """Gold futures monthly returns, from 2000.

    Limitation, stated plainly: this does not reach the 1970s, so gold's behaviour
    in the stagflation regime that most motivates holding it is NOT in sample.
    Gold-conditional statistics are reported with that caveat attached.
    """
    key = "gold_futures"

    if not refresh:
        cached = cache.load_series(key, subdir="assets")
        if cached is not None:
            return cached

    try:
        import yfinance as yf

        df = yf.download(
            "GC=F", start="1990-01-01", interval="1mo",
            auto_adjust=True, progress=False,
        )
        if df is None or df.empty:
            return cache.load_series(key, subdir="assets")

        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        s = close.dropna()
        s.index = pd.DatetimeIndex(s.index) + pd.offsets.MonthEnd(0)
        s = s[~s.index.duplicated(keep="last")].sort_index()

        ret = s.pct_change().dropna().rename("gold_proxy_return")
        cache.save_series(key, ret, subdir="assets")
        log.info("gold futures: %d months from %s", len(ret), ret.index.min().date())
        return ret

    except Exception as exc:  # noqa: BLE001
        log.warning("gold futures fetch failed: %s", exc)
        return cache.load_series(key, subdir="assets")


def long_history_proxies(refresh: bool = False) -> dict[str, pd.Series]:
    """All externally-sourced long-history proxy RETURN series, by asset key."""
    out: dict[str, pd.Series] = {}

    equity = fetch_french_market_return(refresh=refresh)
    if equity is not None:
        out["equity"] = equity

    gold = fetch_gold_futures(refresh=refresh)
    if gold is not None:
        out["gold"] = gold

    return out
