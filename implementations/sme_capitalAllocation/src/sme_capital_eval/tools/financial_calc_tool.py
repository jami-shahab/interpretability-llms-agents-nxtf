"""Financial calculation tool for the Sponsor and Governance agents.

All functions are deterministic Python — no LLM calls. The CrewAI BaseTool
wrapper exposes them to agents via tool-use.
"""

import json
from typing import Any, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pure calculation functions (Single Responsibility)
# ---------------------------------------------------------------------------


def post_transaction_leverage(current_debt: float, new_debt: float, ebitda: float) -> float:
    """Compute Debt-to-EBITDA after adding new debt.

    Returns
    -------
    float
        Post-transaction leverage ratio; ``inf`` if EBITDA is zero.
    """
    if ebitda <= 0:
        return float("inf")
    return (current_debt + new_debt) / ebitda


def post_transaction_cash(
    current_cash: float,
    equity_contribution: float,
    total_capex: float,
) -> float:
    """Compute remaining cash after equity funding a portion of capex.

    Parameters
    ----------
    current_cash : float
        Cash balance before transaction.
    equity_contribution : float
        The amount of capex funded from cash / owner equity.
    total_capex : float
        Total capital expenditure (for reference; not subtracted directly).

    Returns
    -------
    float
        Remaining cash after equity outlay.
    """
    return current_cash - equity_contribution


def revenue_multiplier(uplift: float, capex: float) -> float:
    """Compute revenue uplift / capex ratio.

    Returns
    -------
    float
        Multiplier; ``0.0`` if capex is zero.
    """
    if capex <= 0:
        return 0.0
    return uplift / capex


def payback_years(capex: float, annual_uplift: float) -> float:
    """Compute simple payback period in years.

    Returns
    -------
    float
        Payback years; ``inf`` if annual_uplift is zero or negative.
    """
    if annual_uplift <= 0:
        return float("inf")
    return capex / annual_uplift


def unlevered_irr_approx(capex: float, annual_uplift: float, years: float) -> float:
    """Approximate unlevered IRR using simplified annuity formula.

    This is a lightweight proxy (not DCF), suitable for screening-level
    decisions in the Sponsor Agent.

    Returns
    -------
    float
        Approximate IRR as a decimal (e.g. ``0.18`` for 18%).
    """
    if capex <= 0 or years <= 0:
        return 0.0
    # Simple ROI / years as first-order approximation
    total_return = annual_uplift * years
    return (total_return - capex) / (capex * years)


# ---------------------------------------------------------------------------
# CrewAI Tool schema
# ---------------------------------------------------------------------------


class FinancialCalcInput(BaseModel):
    """Input schema for FinancialCalcTool."""

    operation: str = Field(
        description=(
            "Which calculation to run. One of: "
            "'post_leverage', 'post_cash', 'revenue_multiplier', "
            "'payback_years', 'irr_approx'."
        )
    )
    params: dict = Field(description="Named parameters matching the chosen operation.")


class FinancialCalcTool(BaseTool):
    """Deterministic financial calculator for capital allocation decisions.

    Performs exact arithmetic so agents never have to compute numbers
    in their reasoning tokens — eliminating a major source of LLM error.
    """

    name: str = "financial_calc_tool"
    description: str = (
        "Run a deterministic financial calculation. "
        "Operations: post_leverage, post_cash, revenue_multiplier, payback_years, irr_approx."
    )
    args_schema: Type[BaseModel] = FinancialCalcInput

    def _run(self, operation: str, params: Any) -> str:  # noqa: PLR0911
        """Execute the requested financial operation.

        Parameters
        ----------
        operation : str
            Name of the calculation to perform.
        params : dict
            Keyword arguments for the operation.

        Returns
        -------
        str
            JSON string with the computed result or an error message.
        """
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except Exception:
                return json.dumps({"error": "params must be a JSON object"})

        try:
            if operation == "post_leverage":
                result = post_transaction_leverage(
                    float(params["current_debt"]),
                    float(params["new_debt"]),
                    float(params["ebitda"]),
                )
            elif operation == "post_cash":
                result = post_transaction_cash(
                    float(params["current_cash"]),
                    float(params["equity_contribution"]),
                    float(params.get("total_capex", 0)),
                )
            elif operation == "revenue_multiplier":
                result = revenue_multiplier(
                    float(params["uplift"]),
                    float(params["capex"]),
                )
            elif operation == "payback_years":
                result = payback_years(
                    float(params["capex"]),
                    float(params["annual_uplift"]),
                )
            elif operation == "irr_approx":
                result = unlevered_irr_approx(
                    float(params["capex"]),
                    float(params["annual_uplift"]),
                    float(params["years"]),
                )
            else:
                return json.dumps({"error": f"Unknown operation: {operation!r}"})
        except KeyError as exc:
            return json.dumps({"error": f"Missing parameter: {exc}"})
        except Exception as exc:
            return json.dumps({"error": str(exc)})

        return json.dumps({"operation": operation, "result": round(result, 6)})
