"""Ingest StatsCan tables 34-10-0035-01 (Capex) and 33-10-0500-01 (Financial Ratios)
into a SQLite database at ``data/processed/benchmarks.db``.

Uses the StatsCan Web Data Service (WDS) API:
  GET /t1/wds/rest/getFullTableDownloadCSV/{pid}/en
  → JSON response with {"status":"SUCCESS","object": "<zip_url>"}
  → Download zip → extract CSV → clean → insert to SQLite.

Usage:
    uv run ... -m sme_capital_eval.data_engineering.ingest_statscan
"""

import io
import json
import sqlite3
import zipfile
from pathlib import Path

import pandas as pd
import requests


# parents[3] = implementations/sme_capitalAllocation/
_ROOT = Path(__file__).parents[3]
_RAW_DIR = _ROOT / "data" / "raw"
_DB_PATH = _ROOT / "data" / "processed" / "benchmarks.db"

WDS_BASE = "https://www150.statcan.gc.ca/t1/wds/rest"

TABLES = {
    "34100035": {
        "name": "capex",
        "raw_dir": _RAW_DIR / "statscan_34100035",
    },
    "33100500": {
        "name": "ratios",
        "raw_dir": _RAW_DIR / "statscan_33100500",
    },
}


def _download_table(pid: str, raw_dir: Path) -> Path:
    """Download a StatsCan full-table ZIP and return the extracted CSV path."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_candidates = list(raw_dir.glob("*.csv"))
    if csv_candidates:
        print(f"    Using cached CSV: {csv_candidates[0].name}")
        return csv_candidates[0]

    url = f"{WDS_BASE}/getFullTableDownloadCSV/{pid}/en"
    print(f"    Calling WDS API: {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    meta = resp.json()

    zip_url = meta.get("object", "")
    if not zip_url:
        raise RuntimeError(f"WDS API did not return a zip URL for pid={pid}: {meta}")

    print(f"    Downloading zip: {zip_url}")
    zip_resp = requests.get(zip_url, timeout=120)
    zip_resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(zip_resp.content)) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv") and "MetaData" not in n]
        if not csv_names:
            raise RuntimeError(f"No data CSV found in zip for pid={pid}")
        csv_name = csv_names[0]
        zf.extract(csv_name, raw_dir)
        print(f"    Extracted: {csv_name}")
        return raw_dir / csv_name


def _ingest_capex(csv_path: Path, conn: sqlite3.Connection) -> None:
    """Clean and insert capex data (34-10-0035-01) into capex_benchmarks table."""
    print(f"    Reading capex CSV ({csv_path.stat().st_size // 1024} KB)…")
    df = pd.read_csv(csv_path, encoding="latin-1", low_memory=False)

    # Standardise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Keep only recent data
    year_col = next((c for c in df.columns if "ref_date" in c or "year" in c), None)
    if year_col:
        df[year_col] = pd.to_numeric(df[year_col].astype(str).str[:4], errors="coerce")
        df = df[df[year_col] >= 2018]

    # Identify NAICS, value columns
    naics_col = next((c for c in df.columns if "naics" in c or "industry" in c), None)
    value_col = next((c for c in df.columns if c == "value"), None)
    asset_col = next((c for c in df.columns if "asset" in c or "type" in c), None)

    if not all([naics_col, value_col]):
        print(f"    WARNING: Could not identify required columns in {csv_path.name}. Skipping.")
        return

    # Filter numeric values only
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col, naics_col])

    # Aggregate to median per NAICS + asset_type + year
    group_cols = [c for c in [year_col, naics_col, asset_col] if c]
    agg = df.groupby(group_cols)[value_col].median().reset_index()
    agg.columns = list(group_cols) + ["median_capex_cad"]

    # Normalise for DB
    out = pd.DataFrame({
        "ref_year": agg[year_col] if year_col else 0,
        "naics": agg[naics_col].astype(str).str[:6],
        "asset_type": agg[asset_col].astype(str) if asset_col else "unknown",
        "median_capex_cad": agg["median_capex_cad"],
    })

    out.to_sql("capex_benchmarks", conn, if_exists="replace", index=False)
    print(f"    capex_benchmarks: {len(out)} rows inserted")


def _ingest_ratios(csv_path: Path, conn: sqlite3.Connection) -> None:
    """Clean and insert financial ratio data (33-10-0500-01) into financial_ratios table."""
    print(f"    Reading ratios CSV ({csv_path.stat().st_size // 1024} KB)…")
    df = pd.read_csv(csv_path, encoding="latin-1", low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    year_col = next((c for c in df.columns if "ref_date" in c), None)
    naics_col = next((c for c in df.columns if "naics" in c or "industry" in c), None)
    measure_col = next((c for c in df.columns if "measure" in c or "indicator" in c or "variable" in c), None)
    value_col = next((c for c in df.columns if c == "value"), None)
    size_col = next((c for c in df.columns if "size" in c or "employment" in c), None)

    if not all([naics_col, value_col]):
        print(f"    WARNING: Could not identify required columns. Skipping ratios.")
        return

    if year_col:
        df[year_col] = pd.to_numeric(df[year_col].astype(str).str[:4], errors="coerce")
        df = df[df[year_col] >= 2018]

    # Filter to SME size class
    if size_col:
        df = df[df[size_col].astype(str).str.contains("Small|Medium|1 to 499|SME", case=False, na=False)]

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col, naics_col])

    group_cols = [c for c in [year_col, naics_col, measure_col] if c]
    if not group_cols:
        return

    agg = df.groupby(group_cols)[value_col].median().reset_index()

    out = pd.DataFrame({
        "ref_year": agg[year_col] if year_col else 0,
        "naics": agg[naics_col].astype(str).str[:6],
        "measure": agg[measure_col].astype(str) if measure_col else "unknown",
        "median_value": agg[value_col],
    })

    out.to_sql("financial_ratios", conn, if_exists="replace", index=False)
    print(f"    financial_ratios: {len(out)} rows inserted")


def run() -> None:
    """Main ingestion entry point."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(_DB_PATH) as conn:
        for pid, meta in TABLES.items():
            print(f"\nProcessing table {pid} ({meta['name']})…")
            try:
                csv_path = _download_table(pid, meta["raw_dir"])
                if meta["name"] == "capex":
                    _ingest_capex(csv_path, conn)
                else:
                    _ingest_ratios(csv_path, conn)
            except Exception as exc:
                print(f"  ERROR: {exc}")

    print(f"\nStatsCan ingestion complete → {_DB_PATH}")


if __name__ == "__main__":
    run()
