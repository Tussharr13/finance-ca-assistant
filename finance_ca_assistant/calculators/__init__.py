"""Deterministic calculators. Numbers in answers come from here, never from an LLM."""

from finance_ca_assistant.calculators.income_tax import (
    CalcStep,
    RegimeComparison,
    RegimeResult,
    SalaryTaxInput,
    UnsupportedTaxYear,
    compare_regimes,
)

__all__ = [
    "CalcStep",
    "RegimeComparison",
    "RegimeResult",
    "SalaryTaxInput",
    "UnsupportedTaxYear",
    "compare_regimes",
]
