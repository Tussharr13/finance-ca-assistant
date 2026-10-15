"""Effective-dated income-tax rule tables for resident and non-resident individuals.

Each tax year is a separate, immutable table. Add a new year by adding a new
table after verifying it against the Finance Act for that year; never edit a
published year in place except to correct an error.

Only FY 2025-26 (AY 2026-27, Income-tax Act 1961 as amended by Finance Act
2025) is included. From 1 April 2026 the Income-tax Act 2025 applies and uses
different section numbers; its tables must be added and verified separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# (upper limit of slab in rupees or None for "and above", rate)
Slabs = Tuple[Tuple[Optional[int], float], ...]
# (income threshold above which the rate applies, rate), ascending
SurchargeBands = Tuple[Tuple[int, float], ...]


@dataclass(frozen=True)
class RegimeRules:
    name: str
    slabs_below_60: Slabs
    slabs_60_to_79: Slabs
    slabs_80_plus: Slabs
    standard_deduction: int
    rebate_income_limit: int
    rebate_max: int
    rebate_marginal_relief: bool
    surcharge: SurchargeBands
    allows_hra_exemption: bool
    allows_professional_tax: bool
    allows_chapter_via: bool  # 80C, 80D, 80CCD(1B)
    allows_self_occupied_interest: bool
    employer_nps_limit_pct: float  # 80CCD(2), % of basic + DA
    sources: Dict[str, str]


@dataclass(frozen=True)
class TaxYearRules:
    tax_year: str
    law: str
    old: RegimeRules
    new: RegimeRules
    cess_rate: float
    limit_80c: int
    limit_80d_below_60: int
    limit_80d_senior: int
    limit_80ccd_1b: int
    limit_self_occupied_interest: int
    limit_professional_tax: int
    hra_metro_pct: float
    hra_non_metro_pct: float
    hra_rent_excess_pct: float
    sources: Dict[str, str]


_OLD_SLABS_BELOW_60: Slabs = ((250_000, 0.0), (500_000, 0.05), (1_000_000, 0.20), (None, 0.30))

FY_2025_26 = TaxYearRules(
    tax_year="FY2025-26",
    law="Income-tax Act, 1961 as amended by the Finance Act, 2025",
    old=RegimeRules(
        name="old",
        slabs_below_60=_OLD_SLABS_BELOW_60,
        slabs_60_to_79=((300_000, 0.0), (500_000, 0.05), (1_000_000, 0.20), (None, 0.30)),
        slabs_80_plus=((500_000, 0.0), (1_000_000, 0.20), (None, 0.30)),
        standard_deduction=50_000,
        rebate_income_limit=500_000,
        rebate_max=12_500,
        rebate_marginal_relief=False,
        surcharge=((5_000_000, 0.10), (10_000_000, 0.15), (20_000_000, 0.25), (50_000_000, 0.37)),
        allows_hra_exemption=True,
        allows_professional_tax=True,
        allows_chapter_via=True,
        allows_self_occupied_interest=True,
        employer_nps_limit_pct=0.10,
        sources={
            "slabs": "Finance Act 2025, First Schedule, Part III, Paragraph A",
            "standard_deduction": "Section 16(ia)",
            "rebate": "Section 87A",
            "surcharge": "Finance Act 2025, First Schedule, Part III",
        },
    ),
    new=RegimeRules(
        name="new",
        # Section 115BAC applies the same slabs at every age.
        slabs_below_60=(
            (400_000, 0.0), (800_000, 0.05), (1_200_000, 0.10), (1_600_000, 0.15),
            (2_000_000, 0.20), (2_400_000, 0.25), (None, 0.30),
        ),
        slabs_60_to_79=(),
        slabs_80_plus=(),
        standard_deduction=75_000,
        rebate_income_limit=1_200_000,
        rebate_max=60_000,
        rebate_marginal_relief=True,
        surcharge=((5_000_000, 0.10), (10_000_000, 0.15), (20_000_000, 0.25)),
        allows_hra_exemption=False,
        allows_professional_tax=False,
        allows_chapter_via=False,
        allows_self_occupied_interest=False,
        employer_nps_limit_pct=0.14,
        sources={
            "slabs": "Section 115BAC(1A)",
            "standard_deduction": "Section 16(ia) read with Section 115BAC",
            "rebate": "Section 87A, first and second provisos",
            "surcharge": "Section 115BAC; Finance Act 2025, First Schedule (capped at 25%)",
        },
    ),
    cess_rate=0.04,
    limit_80c=150_000,
    limit_80d_below_60=25_000,
    limit_80d_senior=50_000,
    limit_80ccd_1b=50_000,
    limit_self_occupied_interest=200_000,
    limit_professional_tax=2_500,
    hra_metro_pct=0.50,
    hra_non_metro_pct=0.40,
    hra_rent_excess_pct=0.10,
    sources={
        "cess": "Finance Act 2025, Health and Education Cess",
        "80c": "Section 80C, read with Section 80CCE",
        "80d": "Section 80D",
        "80ccd_1b": "Section 80CCD(1B)",
        "80ccd_2": "Section 80CCD(2)",
        "self_occupied_interest": "Section 24(b)",
        "professional_tax": "Section 16(iii); Article 276(2) of the Constitution",
        "hra": "Section 10(13A) read with Rule 2A",
        "rounding": "Sections 288A and 288B",
    },
)

RULES_BY_YEAR: Dict[str, TaxYearRules] = {FY_2025_26.tax_year: FY_2025_26}
