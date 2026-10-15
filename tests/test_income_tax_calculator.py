import pytest

from finance_ca_assistant.calculators import SalaryTaxInput, UnsupportedTaxYear, compare_regimes

FY = "FY2025-26"


def salary(**overrides):
    values = {"tax_year": FY, "gross_salary": 2_000_000, "basic_da": 1_000_000, "age": 30}
    values.update(overrides)
    return SalaryTaxInput(**values)


def test_new_regime_income_up_to_12_lakh_pays_nothing():
    # 12,75,000 - 75,000 standard deduction = 12,00,000; slab tax 60,000 fully rebated.
    result = compare_regimes(salary(gross_salary=1_275_000, basic_da=600_000)).new
    assert result.taxable_income == 1_200_000
    assert result.tax_on_income == 60_000
    assert result.rebate == 60_000
    assert result.total_tax == 0


def test_new_regime_rebate_marginal_relief_just_above_12_lakh():
    # Total income 12,10,000: slab tax 61,500, capped at 10,000 above the limit, plus 4% cess.
    result = compare_regimes(salary(gross_salary=1_285_000, basic_da=600_000)).new
    assert result.taxable_income == 1_210_000
    assert result.rebate == 51_500
    assert result.total_tax == 10_400


def test_twenty_lakh_salary_with_hra_and_deductions():
    comparison = compare_regimes(
        salary(
            hra_received=400_000,
            rent_paid=360_000,
            metro=True,
            professional_tax=2_500,
            deduction_80c=150_000,
            health_premium_self_family=25_000,
        )
    )
    # Old: HRA exempt = least of 4,00,000 / 3,60,000 - 1,00,000 / 5,00,000 = 2,60,000.
    # 20,00,000 - 2,60,000 - 50,000 - 2,500 - 1,75,000 = 15,12,500 -> tax 2,66,250 + cess.
    assert comparison.old.taxable_income == 1_512_500
    assert comparison.old.total_tax == 276_900
    # New: 20,00,000 - 75,000 = 19,25,000 -> tax 1,85,000 + cess.
    assert comparison.new.taxable_income == 1_925_000
    assert comparison.new.total_tax == 192_400
    assert comparison.lower_regime == "new"
    assert comparison.saving == 84_500


def test_old_regime_rebate_up_to_5_lakh():
    result = compare_regimes(salary(gross_salary=550_000, basic_da=300_000)).old
    assert result.taxable_income == 500_000
    assert result.total_tax == 0


def test_resident_senior_gets_higher_old_regime_exemption_but_non_resident_does_not():
    senior = compare_regimes(salary(gross_salary=450_000, basic_da=200_000, age=65)).old
    assert senior.tax_on_income == 5_000  # 4,00,000 at 60-79 slabs
    assert senior.total_tax == 0

    nri = compare_regimes(salary(gross_salary=1_275_000, basic_da=600_000, resident=False)).new
    assert nri.rebate == 0
    assert nri.total_tax == 62_400


def test_surcharge_marginal_relief_above_50_lakh():
    # Total income 50,10,000: tax 10,83,000; surcharge capped at 7,000 by marginal relief.
    result = compare_regimes(salary(gross_salary=5_085_000, basic_da=2_500_000)).new
    assert result.taxable_income == 5_010_000
    assert result.surcharge == 7_000
    assert result.total_tax == 1_133_600


def test_deduction_caps_and_regime_restrictions():
    comparison = compare_regimes(
        salary(
            age=62,
            deduction_80c=300_000,
            health_premium_self_family=70_000,
            health_premium_parents=40_000,
            nps_self_80ccd_1b=90_000,
            employer_nps_80ccd_2=200_000,
            home_loan_interest_self_occupied=350_000,
        )
    )
    old_steps = {step.label: step.amount for step in comparison.old.steps}
    assert old_steps["Less: Section 80C"] == -150_000
    assert old_steps["Less: Section 80D"] == -75_000  # 50,000 senior self + 25,000 parents
    assert old_steps["Less: Section 80CCD(1B)"] == -50_000
    assert old_steps["Less: Section 80CCD(2) employer NPS"] == -100_000  # 10% of basic + DA
    assert old_steps["Less: interest on self-occupied home loan"] == -200_000

    new_steps = {step.label: step.amount for step in comparison.new.steps}
    assert new_steps["Less: Section 80CCD(2) employer NPS"] == -140_000  # 14% of basic + DA
    assert "Less: Section 80C" not in new_steps
    assert "Less: HRA exemption" not in new_steps


def test_every_step_cites_a_source_and_run_id_is_stable():
    first = compare_regimes(salary())
    assert all(step.source for step in first.old.steps + first.new.steps)
    assert first.run_id == compare_regimes(salary()).run_id
    assert first.run_id != compare_regimes(salary(other_income=10)).run_id
    assert first.not_modelled


def test_unverified_tax_year_is_refused():
    with pytest.raises(UnsupportedTaxYear):
        compare_regimes(salary(tax_year="FY2026-27"))


@pytest.mark.parametrize(
    "overrides",
    [{"gross_salary": -1}, {"basic_da": 3_000_000}, {"age": 200}, {"rent_paid": float("nan")}],
)
def test_invalid_inputs_are_rejected(overrides):
    with pytest.raises(ValueError):
        salary(**overrides)
