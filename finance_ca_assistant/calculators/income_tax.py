"""Old vs new regime income-tax calculation for a salaried individual.

Every figure is produced by code from an effective-dated rule table and is
recorded as a step with the provision it relies on, so an answer can cite the
calculation instead of asking an LLM to do arithmetic.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields
from typing import Dict, List, Optional, Tuple

from finance_ca_assistant.calculators.tax_rules import RULES_BY_YEAR, RegimeRules, Slabs, TaxYearRules

CALC_VERSION = "salary_regime_v1"

# Items this calculator does not model. Answers must surface these rather
# than imply the computed figure is a complete tax position.
NOT_MODELLED: Tuple[str, ...] = (
    "Special-rate income such as capital gains (Sections 111A, 112, 112A) and winnings",
    "Savings and deposit interest deductions (Sections 80TTA, 80TTB)",
    "Other Chapter VI-A deductions (for example 80E, 80EEA, 80G, 80GG)",
    "Salary exemptions other than HRA (for example LTA, perquisite valuation)",
    "Let-out property, carried-forward losses, and agricultural income",
    "The combined Rs 7.5 lakh cap on employer PF, NPS and superannuation contributions",
)


class UnsupportedTaxYear(ValueError):
    """Raised when no verified rule table exists for the requested tax year."""


@dataclass(frozen=True)
class SalaryTaxInput:
    """Annual figures in rupees for one financial year.

    ``gross_salary`` includes basic, DA, HRA, allowances and any employer NPS
    contribution. ``basic_da`` is basic plus DA that forms part of retirement
    benefits, which is the salary base for HRA and Section 80CCD(2).
    """

    tax_year: str
    gross_salary: float
    basic_da: float
    age: int
    resident: bool = True
    hra_received: float = 0.0
    rent_paid: float = 0.0
    metro: bool = False
    professional_tax: float = 0.0
    other_income: float = 0.0
    deduction_80c: float = 0.0
    health_premium_self_family: float = 0.0
    health_premium_parents: float = 0.0
    parents_senior: bool = False
    nps_self_80ccd_1b: float = 0.0
    employer_nps_80ccd_2: float = 0.0
    home_loan_interest_self_occupied: float = 0.0

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, float) or (isinstance(value, int) and not isinstance(value, bool)):
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"{item.name} must be a finite, non-negative number")
        if not isinstance(self.age, int) or isinstance(self.age, bool) or self.age > 130:
            raise ValueError("age must be a whole number between 0 and 130")
        if self.basic_da > self.gross_salary:
            raise ValueError("basic_da cannot exceed gross_salary")


@dataclass(frozen=True)
class CalcStep:
    label: str
    amount: int
    source: str


@dataclass(frozen=True)
class RegimeResult:
    regime: str
    taxable_income: int
    tax_on_income: int
    rebate: int
    surcharge: int
    cess: int
    total_tax: int
    steps: List[CalcStep] = field(default_factory=list)


@dataclass(frozen=True)
class RegimeComparison:
    run_id: str
    tax_year: str
    law: str
    inputs: Dict[str, object]
    old: RegimeResult
    new: RegimeResult
    lower_regime: str  # "old", "new" or "equal"
    saving: int
    not_modelled: Tuple[str, ...] = NOT_MODELLED


def compare_regimes(inputs: SalaryTaxInput) -> RegimeComparison:
    """Compute tax under both regimes for the inputs' tax year."""

    rules = RULES_BY_YEAR.get(inputs.tax_year)
    if rules is None:
        raise UnsupportedTaxYear(
            f"No verified rules for {inputs.tax_year}; supported: {sorted(RULES_BY_YEAR)}"
        )
    old = _compute(inputs, rules, rules.old)
    new = _compute(inputs, rules, rules.new)
    if old.total_tax == new.total_tax:
        lower = "equal"
    else:
        lower = "old" if old.total_tax < new.total_tax else "new"
    payload = {"inputs": _canonical(asdict(inputs)), "rules": rules.tax_year, "calc_version": CALC_VERSION}
    run_id = "calc_" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    return RegimeComparison(
        run_id=run_id,
        tax_year=rules.tax_year,
        law=rules.law,
        inputs=asdict(inputs),
        old=old,
        new=new,
        lower_regime=lower,
        saving=abs(old.total_tax - new.total_tax),
    )


def _compute(inp: SalaryTaxInput, year: TaxYearRules, regime: RegimeRules) -> RegimeResult:
    steps: List[CalcStep] = []

    def step(label: str, amount: float, source: str) -> None:
        steps.append(CalcStep(label, round(amount), source))

    step("Gross salary", inp.gross_salary, "Section 17(1)")

    hra_exempt = 0.0
    if regime.allows_hra_exemption and inp.hra_received > 0 and inp.rent_paid > 0:
        pct = year.hra_metro_pct if inp.metro else year.hra_non_metro_pct
        hra_exempt = max(
            0.0,
            min(
                inp.hra_received,
                inp.rent_paid - year.hra_rent_excess_pct * inp.basic_da,
                pct * inp.basic_da,
            ),
        )
        step("Less: HRA exemption", -hra_exempt, year.sources["hra"])

    standard = min(regime.standard_deduction, inp.gross_salary - hra_exempt)
    step("Less: standard deduction", -standard, regime.sources["standard_deduction"])

    prof_tax = 0.0
    if regime.allows_professional_tax and inp.professional_tax > 0:
        prof_tax = min(inp.professional_tax, year.limit_professional_tax)
        step("Less: professional tax", -prof_tax, year.sources["professional_tax"])

    salary_income = max(0.0, inp.gross_salary - hra_exempt - standard - prof_tax)
    step("Income from salary", salary_income, "Section 15")

    house_loss = 0.0
    if regime.allows_self_occupied_interest and inp.home_loan_interest_self_occupied > 0:
        house_loss = min(inp.home_loan_interest_self_occupied, year.limit_self_occupied_interest)
        step("Less: interest on self-occupied home loan", -house_loss, year.sources["self_occupied_interest"])

    if inp.other_income > 0:
        step("Add: income from other sources", inp.other_income, "Section 56")

    gross_total = max(0.0, salary_income - house_loss + inp.other_income)
    step("Gross total income", gross_total, "Section 80B(5)")

    deductions = 0.0
    if regime.allows_chapter_via:
        d80c = min(inp.deduction_80c, year.limit_80c)
        self_limit = year.limit_80d_senior if inp.age >= 60 else year.limit_80d_below_60
        parent_limit = year.limit_80d_senior if inp.parents_senior else year.limit_80d_below_60
        d80d = min(inp.health_premium_self_family, self_limit) + min(inp.health_premium_parents, parent_limit)
        d80ccd_1b = min(inp.nps_self_80ccd_1b, year.limit_80ccd_1b)
        for label, amount, source in (
            ("Less: Section 80C", d80c, year.sources["80c"]),
            ("Less: Section 80D", d80d, year.sources["80d"]),
            ("Less: Section 80CCD(1B)", d80ccd_1b, year.sources["80ccd_1b"]),
        ):
            if amount > 0:
                step(label, -amount, source)
                deductions += amount
    d80ccd_2 = min(inp.employer_nps_80ccd_2, regime.employer_nps_limit_pct * inp.basic_da)
    if d80ccd_2 > 0:
        step("Less: Section 80CCD(2) employer NPS", -d80ccd_2, year.sources["80ccd_2"])
        deductions += d80ccd_2
    deductions = min(deductions, gross_total)

    taxable = _round_to_ten(gross_total - deductions)
    step("Total income (rounded)", taxable, year.sources["rounding"])

    slabs = _slabs_for(inp, regime)
    tax = _slab_tax(taxable, slabs)
    step("Tax at slab rates", tax, regime.sources["slabs"])

    rebate = _rebate(inp, regime, taxable, tax)
    if rebate > 0:
        step("Less: rebate", -rebate, regime.sources["rebate"])
    tax_after_rebate = tax - rebate

    surcharge = _surcharge(taxable, tax_after_rebate, regime, slabs)
    if surcharge > 0:
        step("Add: surcharge (with marginal relief)", surcharge, regime.sources["surcharge"])

    cess = (tax_after_rebate + surcharge) * year.cess_rate
    if cess > 0:
        step("Add: health and education cess", cess, year.sources["cess"])

    total = _round_to_ten(tax_after_rebate + surcharge + cess)
    step("Total tax payable (rounded)", total, year.sources["rounding"])

    return RegimeResult(
        regime=regime.name,
        taxable_income=taxable,
        tax_on_income=round(tax),
        rebate=round(rebate),
        surcharge=round(surcharge),
        cess=round(cess),
        total_tax=total,
        steps=steps,
    )


def _canonical(values: Dict[str, object]) -> Dict[str, object]:
    """Make equal figures hash equally: 2000000, 2000000.0 and 2000000.0000001 match."""

    return {
        key: round(float(value), 2) if isinstance(value, (int, float)) and not isinstance(value, bool) else value
        for key, value in values.items()
    }


def _slabs_for(inp: SalaryTaxInput, regime: RegimeRules) -> Slabs:
    # Higher basic exemption for seniors applies to residents only, and the
    # new regime has a single slab table for every age.
    if inp.resident and inp.age >= 80 and regime.slabs_80_plus:
        return regime.slabs_80_plus
    if inp.resident and inp.age >= 60 and regime.slabs_60_to_79:
        return regime.slabs_60_to_79
    return regime.slabs_below_60


def _slab_tax(income: float, slabs: Slabs) -> float:
    tax, lower = 0.0, 0
    for upper, rate in slabs:
        if upper is None or income <= upper:
            return tax + (income - lower) * rate
        tax += (upper - lower) * rate
        lower = upper
    return tax


def _rebate(inp: SalaryTaxInput, regime: RegimeRules, income: int, tax: float) -> float:
    if not inp.resident:
        return 0.0
    if income <= regime.rebate_income_limit:
        return min(tax, regime.rebate_max)
    if regime.rebate_marginal_relief:
        # Tax may not exceed the income above the rebate limit.
        excess = income - regime.rebate_income_limit
        if tax > excess:
            return tax - excess
    return 0.0


def _surcharge(income: int, tax: float, regime: RegimeRules, slabs: Slabs) -> float:
    applicable: Optional[Tuple[int, float]] = None
    previous_rate = 0.0
    for index, band in enumerate(regime.surcharge):
        if income > band[0]:
            applicable = band
            previous_rate = regime.surcharge[index - 1][1] if index else 0.0
    if applicable is None:
        return 0.0
    threshold, rate = applicable
    # Marginal relief: tax plus surcharge may not exceed the tax at the
    # threshold plus the income earned above it.
    ceiling = _slab_tax(threshold, slabs) * (1 + previous_rate) + (income - threshold)
    return max(0.0, min(tax * rate, ceiling - tax))


def _round_to_ten(amount: float) -> int:
    """Round half up to the nearest ten rupees."""

    return int(math.floor(max(0.0, amount) / 10 + 0.5)) * 10
