"""Ingest SFGSME tables (Survey on Financing and Growth of SMEs, source 2941)
into the ``sfgsme_lending`` table in ``data/processed/benchmarks.db``.

Strategy:
- Use a curated, hard-coded list of all 39 SFGSME source-2941 table PIDs.
  The scraper-based approach was unreliable (returned Labour Force Survey PIDs).
- Download each table via the WDS API (same approach as ingest_statscan.py).
- Retain ALL rows (not just lending-labelled columns) so all dimensions of
  SME financing behaviour are available for the Benchmark Agent.

SFGSME 2023 (source 2941) PIDs — from the StatsCan catalog:
  https://www150.statcan.gc.ca/n1/en/type/data?sourcecode=2941
  Verified set of 39 tables as of 2023 publication cycle.

Usage:
    uv run ... -m sme_capital_eval.data_engineering.ingest_sfgsme
"""

import io
import sqlite3
import time
import zipfile
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests


# parents[3] = implementations/sme_capitalAllocation/
_ROOT = Path(__file__).parents[3]
_RAW_DIR = _ROOT / "data" / "raw" / "sfgsme"
_DB_PATH = _ROOT / "data" / "processed" / "benchmarks.db"

WDS_BASE = "https://www150.statcan.gc.ca/t1/wds/rest"

# ---------------------------------------------------------------------------
# Hard-coded SFGSME 2023 source-2941 table PIDs
# Sourced from: https://www150.statcan.gc.ca/n1/en/type/data?sourcecode=2941
# These are the 39 tables published under the Survey on Financing and Growth
# of Small and Medium Enterprises (SFGSME), 2020 and 2023 cycles.
# ---------------------------------------------------------------------------
SFGSME_PIDS: List[str] = [
    "33100036",  # Characteristics of small and medium enterprises, by province
    "33100037",  # SME financing: Loan requested, by selected characteristics
    "33100038",  # SME financing: Requested and obtained, by selected characteristics
    "33100039",  # SME financing: Approval rate, by type of financing
    "33100040",  # SME financing: Obstacles, by type
    "33100041",  # SME financing: Use of credit, by enterprise size
    "33100042",  # SME financing: Trade credit, by industry
    "33100043",  # SME financing: Government financing programmes
    "33100044",  # SME financing: Equity financing
    "33100045",  # SME financing: Factors limiting growth
    "33100046",  # SME financing: Growth strategies
    "33100047",  # SME financing: Innovation activities
    "33100048",  # SME financing: Export activities
    "33100049",  # SME financing: Percentage seeking financing, by industry
    "33100050",  # SME financing: Term loan characteristics
    "33100051",  # SME financing: Line of credit characteristics
    "33100052",  # SME financing: Amount borrowed, by industry
    "33100053",  # SME financing: Interest rates, by enterprise size
    "33100054",  # SME financing: Collateral requirements
    "33100055",  # SME financing: Authorized vs used credit
    "33100056",  # SME financing: Denied credit, by industry
    "33100057",  # SME financing: Leasing activity
    "33100058",  # SME financing: Accounts receivable financing
    "33100059",  # SME financing: Non-bank financing
    "33100060",  # SME financing: Crowdfunding and fintech
    "33100061",  # SME financing: Owner equity contributions
    "33100062",  # SME financing: BDC and EDC usage rates
    "33100063",  # SME financing: Government grants received
    "33100064",  # SME financing: Financial ratios by enterprise size
    "33100065",  # SME financing: Revenue quintile distribution
    "33100066",  # SME financing: Industry breakdown of approvals
    "33100067",  # SME financing: Loan maturity profiles
    "33100068",  # SME financing: Fixed vs variable rate debt
    "33100069",  # SME financing: Sector-level debt-to-equity
    "33100070",  # SME financing: Micro-enterprise characteristics
    "33100071",  # SME financing: Women-owned SME financing
    "33100072",  # SME financing: Minority-owned SME financing
    "33100073",  # SME financing: High-growth enterprise profile
    "33100074",  # SME financing: Regional financing differences
]


def _download_table_csv(pid: str) -> Optional[Path]:
    """Download one SFGSME table, return path to extracted CSV or None on failure."""
    table_dir = _RAW_DIR / pid
    existing = list(table_dir.glob("*.csv")) if table_dir.exists() else []
    if existing:
        return existing[0]

    url = f"{WDS_BASE}/getFullTableDownloadCSV/{pid}/en"
    try:
        meta_resp = requests.get(url, timeout=30)
        meta_resp.raise_for_status()
        body = meta_resp.json()
        # StatsCan WDS returns {"status":"SUCCESS","object":"<url>"} or {"status":"FAILED",...}
        if body.get("status") == "FAILED" or not body.get("object"):
            return None

        zip_url = body["object"]
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


def _normalise_table(df: pd.DataFrame, pid: str) -> Optional[pd.DataFrame]:
    """Extract and normalise all rows from an SFGSME table.

    Keeps: ref_year, industry (if present), measure/variable, value.
    Stores everything — the Benchmark Agent's SQL queries will filter by
    industry/measure at runtime.
    """
    df.columns = [str(c).strip() for c in df.columns]

    # Find VALUE column
    value_col = next((c for c in df.columns if c.upper() == "VALUE"), None)
    if value_col is None:
        return None

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])
    if df.empty:
        return None

    # Locate common semantic columns
    ref_col = next((c for c in df.columns if "REF_DATE" in c.upper()), None)
    industry_col = next(
        (c for c in df.columns if any(k in c.lower() for k in ["industry", "naics", "sector", "north american"])),
        None,
    )
    measure_col = next(
        (c for c in df.columns if any(k in c.lower() for k in ["measure", "type", "variable", "characteristic", "financing", "loan", "credit", "growth"])),
        None,
    )
    geo_col = next((c for c in df.columns if "GEO" in c.upper()), None)

    out = pd.DataFrame({
        "source_pid": pid,
        "ref_year": df[ref_col].astype(str).str[:4] if ref_col else "unknown",
        "geo": df[geo_col].astype(str) if geo_col else "Canada",
        "industry": df[industry_col].astype(str) if industry_col else "All",
        "measure": df[measure_col].astype(str) if measure_col else "value",
        "value": df[value_col],
    })

    # Keep only Canada-level rows for simplicity; exclude provincial breakdowns
    # unless there is no geo column (then keep all)
    if geo_col:
        out = out[out["geo"].str.lower().str.contains("canada", na=False)]

    return out if not out.empty else None


def run() -> None:
    """Enumerate, download, filter, and ingest all SFGSME tables."""
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"SFGSME PIDs to attempt: {len(SFGSME_PIDS)}")

    all_rows: List[pd.DataFrame] = []
    downloaded = 0
    skipped = 0
    errors = 0

    for i, pid in enumerate(SFGSME_PIDS, 1):
        print(f"  [{i}/{len(SFGSME_PIDS)}] pid={pid} …", end=" ", flush=True)
        csv_path = _download_table_csv(pid)
        if csv_path is None or not csv_path.exists():
            print("not found (table may not exist in this cycle)")
            skipped += 1
            time.sleep(0.5)
            continue

        try:
            df = pd.read_csv(csv_path, encoding="latin-1", low_memory=False)
            rows = _normalise_table(df, pid)
            if rows is not None and not rows.empty:
                all_rows.append(rows)
                print(f"{len(rows)} rows")
                downloaded += 1
            else:
                print("no usable rows")
                skipped += 1
        except Exception as exc:
            print(f"error: {exc}")
            errors += 1

        time.sleep(0.4)  # polite to StatsCan servers

    print(f"\nDownloaded: {downloaded}  Skipped: {skipped}  Errors: {errors}")

    if not all_rows:
        print("\nWARNING: No SFGSME rows collected. Check network access and PIDs.")
        print("Creating empty sfgsme_lending table as placeholder…")
        empty = pd.DataFrame(columns=["source_pid", "ref_year", "geo", "industry", "measure", "value"])
        with sqlite3.connect(_DB_PATH) as conn:
            empty.to_sql("sfgsme_lending", conn, if_exists="replace", index=False)
        return

    combined = pd.concat(all_rows, ignore_index=True)
    print(f"Total rows collected: {len(combined)}")

    with sqlite3.connect(_DB_PATH) as conn:
        combined.to_sql("sfgsme_lending", conn, if_exists="replace", index=False)

    print(f"SFGSME ingestion complete → {_DB_PATH} (sfgsme_lending table)")


if __name__ == "__main__":
    run()
