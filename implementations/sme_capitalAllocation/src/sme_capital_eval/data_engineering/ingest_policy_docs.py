"""Extract and write NXTFrontier policy documents to plain-text files.

Extracts SAMP and Treasury Policy from the agentdevelopmentplanning.md doc
and writes them as standalone .txt files to ``data/processed/policy_docs/``.

Usage:
    uv run ... -m sme_capital_eval.data_engineering.ingest_policy_docs
"""

from pathlib import Path


# parents[3] = implementations/sme_capitalAllocation/
_ROOT = Path(__file__).parents[3]
_OUT_DIR = _ROOT / "data" / "processed" / "policy_docs"

# Source path for the development planning doc
_SOURCE_DOC = _ROOT / "docs" / "agentdevelopmentplanning.md"

# Policy text constants — extracted directly from agentdevelopmentplanning.md
# so the scripts are self-contained and don't require parsing the Markdown.

SAMP_TEXT = """NXTFrontier Fund — Strategic Asset Management Plan (SAMP)
Aligned with ISO 55000:2014 Asset Management Standard

ASSET LIFECYCLE POLICY
1. All capital assets must follow a documented lifecycle plan from acquisition to disposal.
2. Deferred maintenance is prohibited as a funding mechanism. Assets requiring repair
   must be addressed before new capital expansion is approved.
3. Modular or temporary construction is not permitted for assets classified as
   'A_Fixed_Infrastructure' without explicit Board approval.
4. Asset archetype classification (A_Fixed_Infrastructure, B_Mobile_Fleet, C_Technology,
   D_Human_Capital) determines the applicable maintenance schedule and depreciation regime.

FLEET MANAGEMENT (Archetype B — Mobile Fleet)
5. Fleet expansion requires demonstrated utilization of existing fleet >= 45% on a
   trailing 12-month basis prior to acquisition approval.
6. Conflicting utilization data across company documents must be escalated to the IC
   before any approval is issued.
7. All fleet acquisitions must include a maintenance reserve allocation of 8% of asset
   value per year in operating budget.

CAPITAL WORKS (Archetype A — Fixed Infrastructure)
8. Capital projects exceeding $5,000,000 CAD require a minimum 6-month permitting and
   environmental review buffer before construction commencement.
9. Project timelines must include realistic regulatory milestones. Environmental review
   bypass is not permitted.

APPROVAL THRESHOLDS
10. Department-level approval: up to $500,000 CAD.
11. Investment Committee (IC) approval: $500,001 to $2,000,000 CAD.
12. Board of Directors approval: above $2,000,000 CAD.
13. Any project breaching the IC threshold ($2M) must include a formal IC memorandum
    and independent technical review.
"""

TREASURY_POLICY_TEXT = """NXTFrontier Fund — Treasury & Capital Allocation Policy
Effective: Q1 2024 | Review: Annually

DEBT MANAGEMENT
1. Maximum consolidated Debt-to-EBITDA ratio: 2.5x (post-transaction).
2. All leveraged transactions must be modeled at stressed EBITDA (-20%) to verify
   covenant headroom.
3. Sub-investment grade financing (mezzanine / subordinate debt) requires IC approval
   regardless of transaction size.
4. Senior secured debt from any single lender: maximum $1,150,000 CAD per transaction
   unless combined with BDC or EDC sub-debt tranche.

LIQUIDITY REQUIREMENTS
5. Minimum unrestricted cash floor: $250,000 CAD at all times (post-transaction).
6. Working capital ratio must remain >= 1.2x post-transaction.
7. Any transaction that depletes cash below the $250,000 floor is automatically declined
   unless the Board provides an explicit waiver in advance.

EQUITY & GOVERNMENT PROGRAMMES
8. Government grant funding (NRC-IRAP, CDAP, SR&ED) is encouraged but must not be
   relied upon until a grant commitment letter has been received.
9. EDC Export Development financing is available for export-eligible projects;
   requires EDC eligibility assessment before inclusion in the financing model.
10. Owner equity contributions from cash are permitted but subject to the liquidity floor.
    Equity contributions that breach the floor must be converted to equipment lease or
    vendor financing structures.

CAPITAL STRUCTURE STANDARDS
11. Preferred financing stack (in priority order):
    a. Operating lease or equipment finance (lowest balance-sheet impact)
    b. EDC / BDC backed senior debt
    c. Senior bank debt (single draw limit: $1,150,000 without sub-debt)
    d. Mezzanine / subordinated debt (IC approval required)
    e. Owner cash equity (subject to liquidity floor)

12. Projects relying exclusively on owner cash equity for the full capex amount
    must demonstrate post-transaction cash >= $250,000 or convert to lease financing.
"""


def run() -> None:
    """Write policy documents to processed directory."""
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    docs = {
        "samp.txt": SAMP_TEXT,
        "treasury_policy.txt": TREASURY_POLICY_TEXT,
    }

    for filename, content in docs.items():
        path = _OUT_DIR / filename
        path.write_text(content)
        print(f"  Written: {path} ({len(content)} chars)")

    print(f"\nPolicy docs ready at: {_OUT_DIR}")


if __name__ == "__main__":
    run()
