"""Ingest StatsCan tables 34-10-0035-01 (Capex) and 33-10-0500-01 (Financial Ratios)
into a SQLite database at ``data/processed/benchmarks.db``.

Uses the StatsCan Web Data Service (WDS) API:
  GET /t1/wds/rest/getFullTableDownloadCSV/{pid}/en
  → JSON response with {"status":"SUCCESS","object": "<zip_url>"}
  → Download zip → extract CSV → clean → insert to SQLite.

Column names confirmed by inspection of the downloaded CSVs:
  34-10-0035-01:
    NAICS : 'North American Industry Classification System (NAICS)'
    type  : 'Capital and repair expenditures'
    value : 'VALUE' (millions CAD)
    date  : 'REF_DATE' (year, BOM-encoded as first col)

  33-10-0500-01:
    NAICS   : 'North American Industry Classification System (NAICS)'
    measure : 'Balance sheet and income statement components, selected financial ratios'
    value   : 'VALUE' (millions CAD)
    date    : 'REF_DATE' (first col)

Usage:
    uv run ... -m sme_capital_eval.data_engineering.ingest_statscan
"""

import io
import re
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

# Exact column names confirmed from CSV inspection
_NAICS_COL = "North American Industry Classification System (NAICS)"
_CAPEX_TYPE_COL = "Capital and repair expenditures"
_RATIO_MEASURE_COL = (
    "Balance sheet and income statement components, selected financial ratios"
)
_VALUE_COL = "VALUE"

# Financial ratio measures relevant to the Benchmark Agent
# (debt coverage, leverage, liquidity, profitability)
_RELEVANT_RATIO_MEASURES = {
    "Asset-to-equity ratio",
    "Debt-to-equity ratio",
    "Current ratio",
    "Return on equity",
    "Return on assets",
    "Net profit margin",
    "Total debt",
    "Total assets",
    "Cash and deposits",
    "Fixed assets, net value",
    "Inventory",
    "Income or loss before income taxes",
    "Income or loss after income taxes",
    "Labour expenses, total",
    "Depreciation, depletion and amortization",
    "Interest expense, mortgages",
    "Interest expense, debt securities",
    "Asset current total",
    "Dividend payout ratio",
}


def _extract_naics_code(label: str) -> str:
    """Extract the numeric NAICS code from a bracket-notation label.

    E.g. 'Food manufacturing [311]'     → '311'
         'Manufacturing [31-33]'         → '31'   (takes leading digits)
         'All Industries'                → ''      (no code → skip)
    """
    m = re.search(r"\[([0-9][0-9A-Za-z\-]*)\]", str(label))
    if not m:
        return ""
    raw = m.group(1)
    # For ranges like '31-33', keep only the leading segment
    return raw.split("-")[0]


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


def _get_ref_date_col(df: pd.DataFrame) -> str | None:
    """Return the REF_DATE column name, handling the UTF-8 BOM prefix."""
    for col in df.columns:
        if "REF_DATE" in col.upper():
            return col
    return None


def _ingest_capex(csv_path: Path, conn: sqlite3.Connection) -> None:
    """Clean and insert capex data (34-10-0035-01) into capex_benchmarks table.

    Stores only 'Capital expenditures' rows (not Repair) to keep values
    directly comparable to a project's proposed capex.
    NAICS codes are extracted from bracket notation and stored as numeric strings.
    """
    print(f"    Reading capex CSV ({csv_path.stat().st_size // 1024} KB)…")
    df = pd.read_csv(csv_path, encoding="latin-1", low_memory=False)

    # Locate the REF_DATE column (may have BOM prefix on first col)
    ref_col = _get_ref_date_col(df)
    if ref_col:
        df["_year"] = pd.to_numeric(df[ref_col].astype(str).str[:4], errors="coerce")
        df = df[df["_year"] >= 2018]
    else:
        df["_year"] = 0

    # Keep only Capital expenditures (not Repair)
    if _CAPEX_TYPE_COL in df.columns:
        df = df[df[_CAPEX_TYPE_COL] == "Capital expenditures"]

    if _NAICS_COL not in df.columns:
        print(f"    WARNING: NAICS column not found. Columns: {df.columns.tolist()}")
        return

    if _VALUE_COL not in df.columns:
        print(f"    WARNING: VALUE column not found.")
        return

    # Parse NAICS codes from bracket notation
    df["_naics_code"] = df[_NAICS_COL].apply(_extract_naics_code)
    df = df[df["_naics_code"] != ""]  # drop 'All Industries' and un-coded rows

    # Parse values
    df["_value"] = pd.to_numeric(df[_VALUE_COL], errors="coerce")
    df = df.dropna(subset=["_value", "_naics_code"])

    # Asset type (capex category)
    asset_col = _CAPEX_TYPE_COL if _CAPEX_TYPE_COL in df.columns else None

    group_cols = ["_year", "_naics_code"]
    if asset_col:
        df["_asset_type"] = df[asset_col]
        group_cols.append("_asset_type")

    agg = df.groupby(group_cols)["_value"].median().reset_index()

    out = pd.DataFrame({
        "ref_year": agg["_year"].astype(int),
        "naics": agg["_naics_code"].astype(str),
        "asset_type": agg["_asset_type"].astype(str) if "_asset_type" in agg.columns else "Capital expenditures",
        "median_capex_millions_cad": agg["_value"],
    })

    out.to_sql("capex_benchmarks", conn, if_exists="replace", index=False)
    print(f"    capex_benchmarks: {len(out)} rows inserted")
    print(f"    Sample NAICS codes: {sorted(out['naics'].unique())[:10]}")


def _ingest_ratios(csv_path: Path, conn: sqlite3.Connection) -> None:
    """Clean and insert financial ratio data (33-10-0500-01) into financial_ratios table.

    Filters to the subset of measures relevant to the Benchmark Agent
    (leverage, liquidity, profitability). NAICS codes parsed from brackets.
    """
    print(f"    Reading ratios CSV ({csv_path.stat().st_size // 1024} KB)…")
    df = pd.read_csv(csv_path, encoding="latin-1", low_memory=False)

    ref_col = _get_ref_date_col(df)
    if ref_col:
        df["_year"] = pd.to_numeric(df[ref_col].astype(str).str[:4], errors="coerce")
        df = df[df["_year"] >= 2018]
    else:
        df["_year"] = 0

    if _NAICS_COL not in df.columns:
        print(f"    WARNING: NAICS column not found in ratios CSV.")
        return
    if _RATIO_MEASURE_COL not in df.columns:
        print(f"    WARNING: Measure column not found in ratios CSV.")
        return
    if _VALUE_COL not in df.columns:
        print(f"    WARNING: VALUE column not found.")
        return

    # Parse NAICS codes
    df["_naics_code"] = df[_NAICS_COL].apply(_extract_naics_code)
    df = df[df["_naics_code"] != ""]

    # Filter to relevant measures only (keeps the DB lean and queries fast)
    df = df[df[_RATIO_MEASURE_COL].isin(_RELEVANT_RATIO_MEASURES)]

    df["_value"] = pd.to_numeric(df[_VALUE_COL], errors="coerce")
    df = df.dropna(subset=["_value", "_naics_code"])

    if df.empty:
        print("    WARNING: No rows remain after filtering. Check measure names.")
        return

    group_cols = ["_year", "_naics_code", _RATIO_MEASURE_COL]
    agg = df.groupby(group_cols)["_value"].median().reset_index()

    out = pd.DataFrame({
        "ref_year": agg["_year"].astype(int),
        "naics": agg["_naics_code"].astype(str),
        "measure": agg[_RATIO_MEASURE_COL].astype(str),
        "median_value_millions_cad": agg["_value"],
    })

    out.to_sql("financial_ratios", conn, if_exists="replace", index=False)
    print(f"    financial_ratios: {len(out)} rows inserted")
    print(f"    Measures stored: {sorted(out['measure'].unique())}")


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
