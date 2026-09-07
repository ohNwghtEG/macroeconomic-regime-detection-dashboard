"""Asset return data, with long-history splicing.

THE SAMPLE-SIZE PROBLEM
-----------------------
SPY starts 1993, TLT 2002, GLD 2004, DBC 2006. The four-asset panel therefore
spans ~19 years containing **two** recessions (2008, 2020). Estimating four
regimes x four assets from two recessions is not estimation, it is anecdote.

Each asset splices a long-history total-return proxy behind its ETF, taking the
usable sample back to the 1960s-70s and roughly eight cycles. The ETF is used from
its inception onward; the proxy fills the history behind it.

TOTAL RETURN, NOT PRICE
Dividends are ~2%/yr on SPY and a much larger share of the total on TLT. A
price-only backtest understates both and can invert the conclusion outright, so
every series here is a total return.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import asset_config
from . import cache

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# yfinance
# ---------------------------------------------------------------------------

def fetch_etf(ticker: str, refresh: bool = False) -> pd.Series | None:
    """Monthly total-return index for one ETF.

    ``auto_adjust=True`` gives dividend- and split-adjusted prices, so month-over-
    month changes are total returns.

    yfinance breaks regularly (rate limits, API churn), so every success is cached
    and the cache is preferred - the repo must stay runnable offline.
    """
    cache_key = f"etf_{ticker}"

    if not refresh:
        cached = cache.load_series(cache_key, subdir="assets")
        if cached is not None:
            return cached

    try:
        import yfinance as yf

        df = yf.download(
            ticker,
            start="1990-01-01",
            auto_adjust=True,       # total return, not price
            progress=False,
            interval="1mo",
        )
        if df is None or df.empty:
            log.warning("yfinance returned nothing for %s", ticker)
            return cache.load_series(cache_key, subdir="assets")

        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        s = close.dropna()
        s.index = pd.DatetimeIndex(s.index) + pd.offsets.MonthEnd(0)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        s.name = ticker

        cache.save_series(cache_key, s, subdir="assets")
        log.info("fetched %s: %d months from %s",
                 ticker, len(s), s.index.min().date())
        return s

    except Exception as exc:  # noqa: BLE001
        log.warning("yfinance failed for %s: %s", ticker, exc)
        return cache.load_series(cache_key, subdir="assets")


# ---------------------------------------------------------------------------
# Long-history proxies
# ---------------------------------------------------------------------------

def bond_total_return_from_yield(
    yields: pd.Series,
    duration: float = 8.5,
) -> pd.Series:
    """Reconstruct a bond total return from a yield series.

    Monthly return decomposes into carry plus a duration-driven price change::

        r_t ~= y_{t-1}/12  -  D * (y_t - y_{t-1})

    Approximate - it holds duration fixed and ignores convexity and roll-down - but
    far better than discarding fifty years of bond history, which is the only
    alternative given TLT starts in 2002. The approximation is documented as a
    limitation rather than buried.
    """
    y = yields.dropna().sort_index() / 100.0
    carry = y.shift(1) / 12.0
    price = -duration * y.diff()
    return (carry + price).dropna().rename("bond_proxy_return")


def yield_accrual_return(yields: pd.Series) -> pd.Series:
    """Cash return: accrue an annualised yield monthly."""
    y = yields.dropna().sort_index() / 100.0
    return (y.shift(1) / 12.0).dropna().rename("cash_return")


def index_to_returns(index: pd.Series) -> pd.Series:
    """Monthly simple returns from a total-return index."""
    return index.dropna().sort_index().pct_change().dropna()


def splice(
    recent: pd.Series,
    historical: pd.Series,
    switch_date: str | pd.Timestamp,
) -> pd.Series:
    """Join a historical proxy to a live series at ``switch_date``.

    Both inputs are RETURN series, spliced on returns rather than on levels: index
    levels have arbitrary bases, and joining those would inject a spurious jump at
    the seam.
    """
    switch = pd.Timestamp(switch_date)
    hist = historical[historical.index < switch]
    rec = recent[recent.index >= switch]
    out = pd.concat([hist, rec]).sort_index()
    return out[~out.index.duplicated(keep="last")]


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_asset_returns(
    fred_raw: dict[str, pd.Series],
    refresh: bool = False,
    use_long_history: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the monthly total-return panel, splicing proxies behind each ETF.

    Returns ``(returns, provenance)`` - the second frame records which source each
    asset used and from when, so the seams are auditable rather than invisible.
    """
    cfg = asset_config()
    returns: dict[str, pd.Series] = {}
    provenance: list[dict] = []

    monthly_fred = {
        k: v.dropna().resample("ME").last() for k, v in fred_raw.items()
    }

    external: dict[str, pd.Series] = {}
    if use_long_history:
        from .long_history import long_history_proxies

        external = long_history_proxies(refresh=refresh)

    for spec in cfg["assets"]:
        key = spec["key"]
        ticker = spec.get("etf")
        etf_start = pd.Timestamp(spec["etf_start"])

        etf_returns = None
        if ticker:
            px = fetch_etf(ticker, refresh=refresh)
            if px is not None and len(px) > 12:
                etf_returns = index_to_returns(px)

        # external long-history proxies (Ken French equity, gold futures) take
        # precedence over the FRED-derived ones where both exist
        proxy_returns = external.get(key)
        if proxy_returns is None:
            proxy_returns = _build_proxy(spec, monthly_fred)

        if etf_returns is not None and proxy_returns is not None:
            series = splice(etf_returns, proxy_returns, etf_start)
            src_name = "external" if key in external else "FRED-derived"
            source = f"{ticker} from {etf_start.date()}; {src_name} proxy before"
        elif etf_returns is not None:
            series = etf_returns
            source = f"{ticker} only (no proxy available)"
        elif proxy_returns is not None:
            series = proxy_returns
            source = "proxy only (ETF unavailable)"
        else:
            log.error("no data at all for asset %s", key)
            continue

        series = series.sort_index()
        returns[key] = series.rename(key)
        provenance.append(
            {
                "asset": key,
                "name": spec["name"],
                "role": spec.get("role", ""),
                "source": source,
                "start": series.index.min().date(),
                "end": series.index.max().date(),
                "months": len(series),
            }
        )

    df = pd.DataFrame(returns).sort_index()
    return df, pd.DataFrame(provenance)


def _build_proxy(spec: dict, monthly_fred: dict[str, pd.Series]) -> pd.Series | None:
    """Construct the pre-ETF proxy return series for one asset."""
    proxy = spec.get("proxy") or {}
    kind = proxy.get("kind")

    if kind == "bond_total_return_from_yield":
        ys = proxy.get("yield_series", "GS10")
        if ys in monthly_fred:
            return bond_total_return_from_yield(
                monthly_fred[ys], duration=float(proxy.get("duration", 8.5))
            )

    elif kind == "fred_yield_accrual":
        ys = proxy.get("series")
        if ys and ys in monthly_fred:
            return yield_accrual_return(monthly_fred[ys])

    elif kind == "fred_price":
        ys = proxy.get("series")
        if ys and ys in monthly_fred:
            return index_to_returns(monthly_fred[ys])

    elif kind == "fred_total_return":
        # equity pre-1993: no free long total-return series survives on FRED, so
        # this falls back to the ETF alone unless a bundled CSV is supplied.
        return None

    return None


def forward_returns(returns: pd.DataFrame, horizons=(1, 3, 12)) -> dict[int, pd.DataFrame]:
    """Forward cumulative returns over each horizon.

    Used ONLY for the descriptive conditional-performance tables, never as a
    trading signal - by construction these contain the future.

    Note the overlap: 12-month forward returns sampled monthly overlap by 11
    months, so their standard errors must be Newey-West corrected. Naive t-stats
    on these are inflated by roughly sqrt(12).
    """
    out = {}
    for h in horizons:
        fwd = (1 + returns).rolling(h).apply(np.prod, raw=True).shift(-h) - 1
        out[h] = fwd
    return out
