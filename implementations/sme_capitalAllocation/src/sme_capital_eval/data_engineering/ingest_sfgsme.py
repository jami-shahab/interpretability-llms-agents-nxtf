"""Ingest SFGSME tables (Survey on Financing and Growth of SMEs, source 2941)
into the ``sfgsme_lending`` table in ``data/processed/benchmarks.db``.

Strategy:
1. Scrape the StatsCan search page for source 2941 to enumerate all table PIDs
2. Download each table via the WDS API (same approach as ingest_statscan.py)
3. Filter to lending-relevant columns and insert to SQLite

Usage:
    uv run ... -m sme_capital_eval.data_engineering.ingest_sfgsme
"""

import io
import re
import sqlite3
import time
import zipfile
from pathlib import Path
from typing import List

import pandas as pd
import requests
from bs4 import BeautifulSoup


# parents[3] = implementations/sme_capitalAllocation/
_ROOT = Path(__file__).parents[3]
_RAW_DIR = _ROOT / "data" / "raw" / "sfgsme"
_DB_PATH = _ROOT / "data" / "processed" / "benchmarks.db"

WDS_BASE = "https://www150.statcan.gc.ca/t1/wds/rest"
SOURCE_PAGE = "https://www150.statcan.gc.ca/n1/en/type/data?sourcecode=2941"

# Columns to look for in SFGSME tables
LENDING_KEYWORDS = ["loan", "financing", "approval", "debt", "amount", "requested", "approved", "credit"]


def _enumerate_table_pids() -> List[str]:
    """Scrape source 2941 search page to extract all 8-digit table PIDs."""
    cached = _RAW_DIR / "table_pids.txt"
    if cached.exists():
        pids = cached.read_text().splitlines()
        print(f"  Using cached PIDs ({len(pids)} tables)")
        return pids

    print(f"  Fetching: {SOURCE_PAGE}")
    resp = requests.get(SOURCE_PAGE, timeout=60)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")
    # StatsCan table links contain pid= parameter
    pids = set()
    for a in soup.find_all("a", href=True):
        m = re.search(r"pid=(\d{7,8})", a["href"])
        if m:
            pids.add(m.group(1).zfill(8))

    pids = sorted(pids)
    if not pids:
        print("  WARNING: No PIDs found. Trying fallback regex on raw HTML.")
        pids = list(set(re.findall(r"pid=(\d{7,8})", resp.text)))
        pids = sorted(set(p.zfill(8) for p in pids))

    print(f"  Found {len(pids)} table PIDs")
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    cached.write_text("\n".join(pids))
    return pids


def _download_table_csv(pid: str) -> Path | None:
    """Download one SFGSME table, return path to extracted CSV or None on failure."""
    table_dir = _RAW_DIR / pid
    existing = list(table_dir.glob("*.csv")) if table_dir.exists() else []
    if existing:
        return existing[0]

    url = f"{WDS_BASE}/getFullTableDownloadCSV/{pid}/en"
    try:
        meta_resp = requests.get(url, timeout=30)
        meta_resp.raise_for_status()
        zip_url = meta_resp.json().get("object", "")
        if not zip_url:
            return None

        zip_resp = requests.get(zip_url, timeout=60)
        zip_resp.raise_for_status()

        table_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(zip_resp.content)) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv") and "MetaData" not in n]
            if not csv_names:
                return None
            zf.extract(csv_names[0], table_dir)
            return table_dir / csv_names[0]
    except Exception as exc:
        print(f"    Skipping {pid}: {exc}")
        return None


def _is_lending_table(df: pd.DataFrame) -> bool:
    """Return True if the table contains lending-relevant columns."""
    combined = " ".join(str(c).lower() for c in df.columns)
    return any(kw in combined for kw in LENDING_KEYWORDS)


def _extract_rows(df: pd.DataFrame, pid: str) -> pd.DataFrame | None:
    """Extract and normalise lending rows from a SFGSME table."""
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    value_col = next((c for c in df.columns if c == "value"), None)
    if value_col is None:
        return None

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])
    if df.empty:
        return None

    industry_col = next((c for c in df.columns if "industry" in c or "naics" in c or "sector" in c), None)
    measure_col = next((c for c in df.columns if "measure" in c or "type" in c or "variable" in c), None)
    year_col = next((c for c in df.columns if "ref_date" in c or "year" in c), None)

    out = pd.DataFrame({
        "source_pid": pid,
        "ref_year": df[year_col].astype(str).str[:4] if year_col else "unknown",
        "industry": df[industry_col].astype(str) if industry_col else "all",
        "measure": df[measure_col].astype(str) if measure_col else "value",
        "value": df[value_col],
    })
    return out


def run() -> None:
    """Enumerate, download, filter, and ingest all SFGSME tables."""
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    pids = _enumerate_table_pids()
    print(f"\nDownloading up to {len(pids)} SFGSME tables…")

    all_rows = []
    for i, pid in enumerate(pids, 1):
        print(f"  [{i}/{len(pids)}] pid={pid} …", end=" ")
        csv_path = _download_table_csv(pid)
        if csv_path is None or not csv_path.exists():
            print("skipped (no CSV)")
            time.sleep(0.5)
            continue

        try:
            df = pd.read_csv(csv_path, encoding="latin-1", low_memory=False, nrows=5000)
            if not _is_lending_table(df):
                print("skipped (not lending)")
                continue
            rows = _extract_rows(df, pid)
            if rows is not None and not rows.empty:
                all_rows.append(rows)
                print(f"{len(rows)} rows")
            else:
                print("no usable rows")
        except Exception as exc:
            print(f"error: {exc}")
        time.sleep(0.3)  # polite to StatsCan servers

    if not all_rows:
        print("\nWARNING: No SFGSME lending rows collected. Check network access.")
        return

    combined = pd.concat(all_rows, ignore_index=True)
    print(f"\nTotal rows collected: {len(combined)}")

    with sqlite3.connect(_DB_PATH) as conn:
        combined.to_sql("sfgsme_lending", conn, if_exists="replace", index=False)

    print(f"SFGSME ingestion complete → {_DB_PATH} (sfgsme_lending table)")


if __name__ == "__main__":
    run()
