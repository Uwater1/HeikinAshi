#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nasdaq_alpha_scan.py

Goal:
1) Pull ALL NASDAQ-listed tickers (official NASDAQ Trader symbol directory)
2) Download 1-year price data from Yahoo Finance via yfinance
3) Compute each stock's 1Y total return and "alpha" vs a chosen benchmark (default: QQQ)
4) (Optional but recommended) Only for a shortlist (top/bottom by alpha), fetch basic fundamentals / financial-statement-derived features
5) Save CSV outputs you can use for factor research / alpha model prototyping

This script is for EDUCATIONAL / RESEARCH use. Not investment advice.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"


# -----------------------------
# Utilities
# -----------------------------
def chunked(seq: Sequence[str], n: int) -> Iterable[List[str]]:
    """Yield successive n-sized chunks from seq."""
    for i in range(0, len(seq), n):
        yield list(seq[i : i + n])


def safe_float(x) -> float:
    try:
        if x is None:
            return float("nan")
        if isinstance(x, (float, int, np.floating, np.integer)):
            return float(x)
        # strings like '1,234'
        if isinstance(x, str):
            x = x.replace(",", "").strip()
        return float(x)
    except Exception:
        return float("nan")


def today_ymd() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d")


def ensure_out_dir(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# -----------------------------
# 1) Universe: NASDAQ tickers
# -----------------------------
def fetch_nasdaq_tickers(include_etfs: bool = False) -> List[str]:
    """
    Pull NASDAQ-listed symbols from NASDAQ Trader directory.

    Columns typically include:
    Symbol | Security Name | Market Category | Test Issue | Financial Status | Round Lot Size | ETF | NextShares
    """
    df = pd.read_csv(NASDAQ_LISTED_URL, sep="|", dtype=str)

    # The file ends with a metadata row like: "File Creation Time: ..."
    # We drop any rows where Symbol contains 'File Creation Time'
    df = df[df["Symbol"].notna()]
    df = df[~df["Symbol"].str.contains("File Creation Time", na=False)]

    # Filter out "Test Issue" tickers (usually not real tradable tickers)
    if "Test Issue" in df.columns:
        df = df[df["Test Issue"].fillna("N") == "N"]

    # Filter ETFs unless user explicitly wants them
    if not include_etfs and "ETF" in df.columns:
        df = df[df["ETF"].fillna("N") == "N"]

    tickers = df["Symbol"].astype(str).str.strip().tolist()

    # Yahoo Finance uses '-' instead of '.' for share classes
    tickers = [t.replace(".", "-") for t in tickers if t]

    # De-duplicate + stable ordering
    tickers = sorted(set(tickers))
    return tickers


# -----------------------------
# 2) Prices: download + returns
# -----------------------------
@dataclass
class PriceDownloadResult:
    adj_close: pd.DataFrame  # index=date, columns=tickers
    volume: pd.DataFrame     # index=date, columns=tickers


def download_prices(
    tickers: List[str],
    period: str = "1y",
    interval: str = "1d",
    batch_size: int = 200,
    pause: float = 1.0,
    threads: bool = True,
) -> PriceDownloadResult:
    """
    Download price data for many tickers in batches.
    Uses Adj Close for total-return calculation.
    """
    adj_parts: List[pd.DataFrame] = []
    vol_parts: List[pd.DataFrame] = []

    for b, batch in enumerate(chunked(tickers, batch_size), start=1):
        # yfinance accepts list[str] or a space-separated string
        try:
            data = yf.download(
                tickers=batch,
                period=period,
                interval=interval,
                auto_adjust=False,
                group_by="column",
                progress=False,
                threads=threads,
            )
        except Exception as e:
            print(f"[WARN] Batch {b} download failed ({len(batch)} tickers): {e}", file=sys.stderr)
            # fall back to single-ticker download to salvage data
            for t in batch:
                try:
                    d = yf.download(
                        tickers=t,
                        period=period,
                        interval=interval,
                        auto_adjust=False,
                        group_by="column",
                        progress=False,
                        threads=False,
                    )
                    if "Adj Close" in d.columns and "Volume" in d.columns:
                        adj_parts.append(d[["Adj Close"]].rename(columns={"Adj Close": t}))
                        vol_parts.append(d[["Volume"]].rename(columns={"Volume": t}))
                except Exception:
                    continue
            time.sleep(pause)
            continue

        # MultiIndex columns when multiple tickers
        if isinstance(data.columns, pd.MultiIndex):
            if ("Adj Close" in data.columns.get_level_values(0)) and ("Volume" in data.columns.get_level_values(0)):
                adj = data["Adj Close"].copy()
                vol = data["Volume"].copy()
            else:
                # sometimes yahoo returns 'Close' only; handle gracefully
                adj = data.get("Adj Close", pd.DataFrame(index=data.index))
                vol = data.get("Volume", pd.DataFrame(index=data.index))
        else:
            # single ticker case
            if "Adj Close" in data.columns:
                adj = data[["Adj Close"]].rename(columns={"Adj Close": batch[0]})
            else:
                adj = pd.DataFrame(index=data.index)
            if "Volume" in data.columns:
                vol = data[["Volume"]].rename(columns={"Volume": batch[0]})
            else:
                vol = pd.DataFrame(index=data.index)

        if not adj.empty:
            adj_parts.append(adj)
        if not vol.empty:
            vol_parts.append(vol)

        # polite pause to reduce the chance of rate limiting
        time.sleep(pause)

    if not adj_parts:
        raise RuntimeError("No price data downloaded. Check network, Yahoo access, or reduce universe size.")

    adj_all = pd.concat(adj_parts, axis=1)
    vol_all = pd.concat(vol_parts, axis=1) if vol_parts else pd.DataFrame(index=adj_all.index)

    # Remove duplicate columns if any (can happen if salvage + batch overlap)
    adj_all = adj_all.loc[:, ~adj_all.columns.duplicated()]
    vol_all = vol_all.loc[:, ~vol_all.columns.duplicated()]

    # Sort columns
    adj_all = adj_all.reindex(sorted(adj_all.columns), axis=1)
    vol_all = vol_all.reindex(sorted(vol_all.columns), axis=1)

    return PriceDownloadResult(adj_close=adj_all, volume=vol_all)


def total_return_from_series(s) -> float:
    """Total return using first/last valid values.

    Note: yfinance sometimes returns a 1-column DataFrame (often due to MultiIndex columns).
    This helper accepts either a Series or DataFrame and returns (last/first - 1).
    """
    # If a DataFrame was provided, squeeze to a single Series.
    if isinstance(s, pd.DataFrame):
        if s.shape[1] == 0:
            return float("nan")
        # Prefer the first (or only) column.
        s = s.iloc[:, 0]

    # Ensure numeric
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 2:
        return float("nan")

    first = float(s.iloc[0])
    last = float(s.iloc[-1])

    if (not np.isfinite(first)) or (not np.isfinite(last)) or first <= 0 or last <= 0:
        return float("nan")

    return float(last / first - 1.0)



def compute_returns_table(
    adj_close: pd.DataFrame,
    volume: Optional[pd.DataFrame] = None,
    min_obs: int = 200,
    min_avg_volume: float = 0.0,
) -> pd.DataFrame:
    """
    Build a per-ticker table with:
      - obs: number of non-NaN days
      - avg_volume: mean volume (if provided)
      - ret_1y: total return over available period
    Then filter by min_obs and min_avg_volume.
    """
    rows = []
    for t in adj_close.columns:
        s = adj_close[t]
        obs = int(s.notna().sum())
        ret = total_return_from_series(s)
        avg_vol = float("nan")
        if volume is not None and not volume.empty and t in volume.columns:
            avg_vol = float(pd.to_numeric(volume[t], errors="coerce").dropna().mean())
        rows.append({"ticker": t, "obs": obs, "avg_volume": avg_vol, "ret_1y": ret})

    df = pd.DataFrame(rows).set_index("ticker")

    # filters
    df = df[df["obs"] >= int(min_obs)]
    if min_avg_volume > 0 and "avg_volume" in df.columns:
        df = df[df["avg_volume"].fillna(0.0) >= float(min_avg_volume)]

    return df.sort_values("ret_1y", ascending=False)


# -----------------------------
# 3) Fundamentals / financial statements (shortlist only)
# -----------------------------
def _pick_line_item(df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
    """
    Pick a financial-statement line item from a yfinance DataFrame.
    df: index=line items, columns=period end dates
    Returns a Series across columns (dates) for the first matching candidate.
    """
    if df is None or df.empty:
        return None

    # exact match first
    for name in candidates:
        if name in df.index:
            return df.loc[name]

    # case-insensitive / whitespace-insensitive match
    norm_index = {str(i).replace(" ", "").lower(): i for i in df.index}
    for name in candidates:
        key = name.replace(" ", "").lower()
        if key in norm_index:
            return df.loc[norm_index[key]]

    return None


def compute_ttm_from_quarters(series: pd.Series, n_quarters: int = 4) -> float:
    """
    Given a Series of quarterly values indexed by quarter end dates (columns in yfinance),
    compute a simple TTM sum of the last n_quarters.
    """
    if series is None:
        return float("nan")
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < n_quarters:
        return float("nan")
    # yfinance columns are usually newest->oldest; but not guaranteed
    s = s.sort_index()
    last = s.iloc[-n_quarters:]
    return float(last.sum())


def fetch_fundamental_features(symbol: str, pause: float = 0.5) -> Dict[str, object]:
    """
    Fetch a small, robust set of fundamentals + TTM features from yfinance.
    This WILL be slow if you run it for thousands of tickers.
    Recommended: only run for shortlisted tickers.
    """
    features: Dict[str, object] = {"ticker": symbol}

    try:
        tk = yf.Ticker(symbol)
    except Exception as e:
        features["error"] = f"Ticker init failed: {e}"
        return features

    # 1) Info dict (valuation + margins + growth + etc.)
    info = {}
    try:
        info = tk.info or {}
    except Exception:
        info = {}

    # Pull commonly useful numeric fields (many may be missing)
    info_fields = [
        "marketCap",
        "enterpriseValue",
        "trailingPE",
        "forwardPE",
        "priceToSalesTrailing12Months",
        "priceToBook",
        "beta",
        "profitMargins",
        "grossMargins",
        "operatingMargins",
        "ebitdaMargins",
        "revenueGrowth",
        "earningsGrowth",
        "freeCashflow",
        "operatingCashflow",
        "totalCash",
        "totalDebt",
        "debtToEquity",
        "returnOnEquity",
        "returnOnAssets",
        "currentRatio",
        "quickRatio",
        "52WeekChange",
    ]
    for k in info_fields:
        features[k] = safe_float(info.get(k))

    # categorical fields (for later grouping / one-hot)
    for k in ["sector", "industry", "country", "exchange"]:
        v = info.get(k)
        features[k] = v if isinstance(v, str) else None

    # 2) Quarterly statements for simple TTM metrics
    q_fin = pd.DataFrame()
    q_cf = pd.DataFrame()
    q_bs = pd.DataFrame()
    try:
        q_fin = tk.quarterly_financials
    except Exception:
        q_fin = pd.DataFrame()
    try:
        q_cf = tk.quarterly_cashflow
    except Exception:
        q_cf = pd.DataFrame()
    try:
        q_bs = tk.quarterly_balance_sheet
    except Exception:
        q_bs = pd.DataFrame()

    # Revenue / Net income (TTM)
    rev = _pick_line_item(q_fin, ["Total Revenue", "TotalRevenue"])
    ni = _pick_line_item(q_fin, ["Net Income", "NetIncome"])
    op_inc = _pick_line_item(q_fin, ["Operating Income", "OperatingIncome"])
    gp = _pick_line_item(q_fin, ["Gross Profit", "GrossProfit"])

    features["ttm_revenue"] = compute_ttm_from_quarters(rev)
    features["ttm_net_income"] = compute_ttm_from_quarters(ni)
    features["ttm_operating_income"] = compute_ttm_from_quarters(op_inc)
    features["ttm_gross_profit"] = compute_ttm_from_quarters(gp)

    # Cashflow: Free cashflow / Operating cashflow (TTM)
    ocf = _pick_line_item(q_cf, ["Total Cash From Operating Activities", "Operating Cash Flow", "TotalCashFromOperatingActivities"])
    capex = _pick_line_item(q_cf, ["Capital Expenditures", "CapitalExpenditures"])
    features["ttm_operating_cf"] = compute_ttm_from_quarters(ocf)
    capex_ttm = compute_ttm_from_quarters(capex)
    features["ttm_capex"] = capex_ttm
    if not math.isnan(features["ttm_operating_cf"]) and not math.isnan(capex_ttm):
        # capex is usually negative; free cash flow approx ocf + capex
        features["ttm_free_cf_est"] = float(features["ttm_operating_cf"] + capex_ttm)
    else:
        features["ttm_free_cf_est"] = float("nan")

    # Balance sheet: use most recent quarter values
    # Note: for balance sheet, "TTM sum" doesn't make sense; use last value
    def last_value(series: Optional[pd.Series]) -> float:
        if series is None:
            return float("nan")
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return float("nan")
        s = s.sort_index()
        return float(s.iloc[-1])

    total_assets = _pick_line_item(q_bs, ["Total Assets", "TotalAssets"])
    total_liab = _pick_line_item(q_bs, ["Total Liab", "TotalLiab", "Total Liabilities", "TotalLiabilitiesNetMinorityInterest"])
    total_equity = _pick_line_item(q_bs, ["Total Stockholder Equity", "TotalStockholderEquity", "Total Equity Gross Minority Interest"])

    features["bs_total_assets"] = last_value(total_assets)
    features["bs_total_liabilities"] = last_value(total_liab)
    features["bs_total_equity"] = last_value(total_equity)

    # Derived ratios
    if features["ttm_revenue"] and not math.isnan(features["ttm_revenue"]) and features["marketCap"] and not math.isnan(features["marketCap"]):
        features["sales_yield_est"] = float(features["ttm_revenue"] / features["marketCap"])  # inverse of P/S-ish
    else:
        features["sales_yield_est"] = float("nan")

    time.sleep(pause)
    return features


def fetch_fundamentals_for_list(
    tickers: List[str],
    out_jsonl: Path,
    pause: float = 0.5,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Fetch fundamentals for a list of tickers and write JSONL as we go (crash-safe).
    If overwrite=False and file exists, we'll resume and only fetch missing tickers.
    """
    done: Dict[str, bool] = {}
    if out_jsonl.exists() and not overwrite:
        try:
            with out_jsonl.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        t = obj.get("ticker")
                        if isinstance(t, str):
                            done[t] = True
                    except Exception:
                        continue
        except Exception:
            pass

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    mode = "w" if overwrite else "a"
    with out_jsonl.open(mode, encoding="utf-8") as f:
        for i, t in enumerate(tickers, start=1):
            if done.get(t):
                continue
            feat = fetch_fundamental_features(t, pause=pause)
            f.write(json.dumps(feat, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"[INFO] fundamentals progress: {i}/{len(tickers)}")

    # Load JSONL to DataFrame
    records = []
    with out_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    df = pd.DataFrame(records).set_index("ticker")
    return df


# -----------------------------
# Main pipeline
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="NASDAQ 1Y alpha scan using yfinance.")
    parser.add_argument("--period", default="1y", help="Price lookback period for yfinance, e.g. 1y, 6mo, 2y")
    parser.add_argument("--interval", default="1d", help="Price interval, e.g. 1d, 1h, 30m (minute data has limits)")
    parser.add_argument("--benchmark", default="QQQ", help="Benchmark ticker, e.g. QQQ, SPY, ^IXIC")
    parser.add_argument("--include_etfs", action="store_true", help="Include ETFs in NASDAQ universe")
    parser.add_argument("--max_tickers", type=int, default=0, help="Debug: limit number of tickers processed (0 = no limit)")
    parser.add_argument("--batch_size", type=int, default=200, help="Batch size for yf.download")
    parser.add_argument("--pause", type=float, default=1.0, help="Pause (seconds) between batches to reduce rate limiting")
    parser.add_argument("--threads", action="store_true", help="Enable yfinance threaded downloads (may be faster, may trigger limits)")

    parser.add_argument("--min_obs", type=int, default=200, help="Min non-NaN price observations to keep a ticker")
    parser.add_argument("--min_avg_volume", type=float, default=200000.0, help="Min average daily volume filter (0 disables)")

    parser.add_argument("--top_n", type=int, default=200, help="How many top-alpha tickers to fetch fundamentals for")
    parser.add_argument("--bottom_n", type=int, default=200, help="How many bottom-alpha tickers to fetch fundamentals for")
    parser.add_argument("--fund_pause", type=float, default=0.6, help="Pause between fundamentals calls (seconds)")
    parser.add_argument("--no_fundamentals", action="store_true", help="Skip fundamentals step (fastest)")
    parser.add_argument("--out_dir", default="out", help="Output directory")
    parser.add_argument("--overwrite_fund_jsonl", action="store_true", help="Overwrite fundamentals JSONL instead of resume")

    args = parser.parse_args()

    out_dir = ensure_out_dir(Path(args.out_dir))

    print("[INFO] Fetching NASDAQ tickers ...")
    tickers = fetch_nasdaq_tickers(include_etfs=args.include_etfs)
    print(f"[INFO] Universe size (NASDAQ): {len(tickers)}")

    if args.max_tickers and args.max_tickers > 0:
        tickers = tickers[: args.max_tickers]
        print(f"[INFO] Limiting to first {len(tickers)} tickers (debug)")

    # Add benchmark
    bench = args.benchmark
    print(f"[INFO] Downloading prices for benchmark: {bench}")
    bench_prices = yf.download(
        tickers=bench,
        period=args.period,
        interval=args.interval,
        auto_adjust=False,
        group_by="column",
        progress=False,
        threads=False,
    )
    if "Adj Close" not in bench_prices.columns:
        raise RuntimeError(f"Benchmark {bench} has no Adj Close data from yfinance.")
    bench_ret = total_return_from_series(bench_prices["Adj Close"])
    print(f"[INFO] Benchmark {bench} return over {args.period}: {bench_ret:.2%}")

    # Download prices for all tickers
    print("[INFO] Downloading NASDAQ prices in batches ... (this can be heavy)")
    prices = download_prices(
        tickers=tickers,
        period=args.period,
        interval=args.interval,
        batch_size=args.batch_size,
        pause=args.pause,
        threads=args.threads,
    )

    # Compute per-ticker returns table
    print("[INFO] Computing returns table + filtering universe ...")
    ret_table = compute_returns_table(
        prices.adj_close,
        volume=prices.volume,
        min_obs=args.min_obs,
        min_avg_volume=args.min_avg_volume,
    )

    # Alpha vs benchmark
    # Also compute "relative return" (ratio form)
    ret_table["alpha_diff"] = ret_table["ret_1y"] - bench_ret
    ret_table["alpha_ratio"] = (1.0 + ret_table["ret_1y"]) / (1.0 + bench_ret) - 1.0

    # Save ranking
    rank_path = out_dir / f"nasdaq_alpha_ranking_{today_ymd()}.csv"
    ret_table.sort_values("alpha_ratio", ascending=False).to_csv(rank_path)
    print(f"[INFO] Saved alpha ranking: {rank_path}")

    if args.no_fundamentals:
        print("[INFO] --no_fundamentals set. Done.")
        return

    # Shortlist tickers
    top_n = max(0, int(args.top_n))
    bottom_n = max(0, int(args.bottom_n))

    shortlist = []
    if top_n > 0:
        shortlist += list(ret_table.sort_values("alpha_ratio", ascending=False).head(top_n).index)
    if bottom_n > 0:
        shortlist += list(ret_table.sort_values("alpha_ratio", ascending=True).head(bottom_n).index)

    shortlist = sorted(set(shortlist))
    print(f"[INFO] Shortlist size for fundamentals: {len(shortlist)} (top_n={top_n}, bottom_n={bottom_n})")

    fund_jsonl = out_dir / "fundamentals.jsonl"
    print("[INFO] Fetching fundamentals for shortlist (resume-safe) ...")
    fund_df = fetch_fundamentals_for_list(
        shortlist,
        out_jsonl=fund_jsonl,
        pause=args.fund_pause,
        overwrite=args.overwrite_fund_jsonl,
    )

    # Merge dataset
    dataset = ret_table.join(fund_df, how="left")
    dataset["label_outperform"] = (dataset["alpha_ratio"] > 0).astype(int)

    dataset_path = out_dir / f"nasdaq_dataset_{today_ymd()}.csv"
    dataset.to_csv(dataset_path)
    print(f"[INFO] Saved merged dataset: {dataset_path}")

    # Quick summary: top outperformers with some fundamentals
    preview_cols = [
        "ret_1y",
        "alpha_ratio",
        "marketCap",
        "trailingPE",
        "profitMargins",
        "revenueGrowth",
        "ttm_revenue",
        "ttm_net_income",
        "ttm_free_cf_est",
        "sector",
    ]
    existing = [c for c in preview_cols if c in dataset.columns]
    preview = dataset.sort_values("alpha_ratio", ascending=False).head(20)[existing]
    preview_path = out_dir / f"top20_preview_{today_ymd()}.csv"
    preview.to_csv(preview_path)
    print(f"[INFO] Saved top-20 preview: {preview_path}")

    print("[DONE] You now have:")
    print(f"  - Full alpha ranking: {rank_path}")
    print(f"  - Fundamentals JSONL (shortlist): {fund_jsonl}")
    print(f"  - Merged dataset: {dataset_path}")
    print(f"  - Top-20 preview: {preview_path}")


if __name__ == "__main__":
    main()
