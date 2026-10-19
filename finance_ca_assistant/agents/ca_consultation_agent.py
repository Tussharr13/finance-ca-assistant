"""Conversational CA consultation agent built on top of RAG retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from finance_ca_assistant.calculators import RegimeComparison, SalaryTaxInput, UnsupportedTaxYear, compare_regimes
from finance_ca_assistant.calculators.tax_rules import RULES_BY_YEAR
from finance_ca_assistant.generation.llm_factory import EchoLLM
from finance_ca_assistant.generation.typed_claims import (
    CLAIMS_SYSTEM_PROMPT,
    build_claims_prompt,
    build_salary_evidence,
    render_claims,
    validate_claims,
)
from finance_ca_assistant.retrieval.result_formatter import format_retrieval_results, source_payloads

# Used when the client does not name a year. Planning for a later year must
# wait until that year's rules are verified (see decisionsForCA.md D-12).
LATEST_VERIFIED_TAX_YEAR = max(RULES_BY_YEAR)


@dataclass(frozen=True)
class ClientProfile:
    """Facts needed before a CA-style personal tax planning answer is reliable."""

    annual_salary_ctc: Optional[float] = None
    annual_basic_da: Optional[float] = None
    salary_currency: str = "INR"
    tax_year: Optional[str] = None
    age: Optional[int] = None
    residential_status: Optional[str] = None
    tax_regime_preference: Optional[str] = None
    city: Optional[str] = None
    monthly_rent: Optional[float] = None
    monthly_hra: Optional[float] = None
    deductions_investments: Optional[str] = None
    health_insurance: Optional[str] = None
    home_loan: Optional[str] = None
    other_income: Optional[str] = None
    employer_tds_or_form16: Optional[str] = None
    goals: Optional[str] = None
    # Annual rupee amounts the calculator needs. Zero means "has none".
    deduction_80c_amount: Optional[float] = None
    health_premium_self_family: Optional[float] = None
    health_premium_parents: Optional[float] = None
    parents_senior: Optional[bool] = None
    nps_self_amount: Optional[float] = None
    home_loan_interest: Optional[float] = None
    other_income_amount: Optional[float] = None
    professional_tax: Optional[float] = None

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "ClientProfile":
        """Build a profile from user/session state."""

        if not data:
            return cls()
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in data.items() if key in allowed})

    def with_updates(self, updates: Mapping[str, Any]) -> "ClientProfile":
        """Return a copy with non-empty inferred updates."""

        values = dict(self.__dict__)
        for key, value in updates.items():
            if key in values and value not in (None, ""):
                values[key] = value
        return ClientProfile(**values)

    def as_dict(self) -> Dict[str, Any]:
        """Return the profile as a serializable dictionary."""

        return dict(self.__dict__)


@dataclass(frozen=True)
class AgentResponse:
    """One conversational agent turn."""

    status: str
    message: str
    questions: List[str] = field(default_factory=list)
    profile_updates: Dict[str, Any] = field(default_factory=dict)
    retrieved_sources: str = ""
    # The same retrieval results as structured entries, for API clients.
    sources: List[Dict[str, Any]] = field(default_factory=list)
    answer: Optional[Dict[str, Any]] = None
    calculation: Optional[RegimeComparison] = None
    # Typed-claims explanation diagnostics: accepted/rejected claims or an error.
    explanation: Optional[Dict[str, Any]] = None


class CAConsultationAgent:
    """Agentic conversational wrapper around the CA RAG pipeline.

    The agent first gathers client facts. For salary tax planning it then runs
    the deterministic regime calculator and builds the reply from its output,
    so every figure comes from code and cites the provision behind it. Client
    facts never go into retrieval. When a real LLM is configured, it receives
    the facts and the calculation to write a typed-claims explanation; this is
    only acceptable for self-hosted models until redaction exists (D-23).
    """

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline

    def respond(
        self,
        user_message: str,
        profile: Optional[Mapping[str, Any] | ClientProfile] = None,
        history: Optional[Sequence[Mapping[str, str]]] = None,
        top_k: int = 6,
    ) -> AgentResponse:
        """Respond conversationally to a client advisory message."""

        selected_profile = profile if isinstance(profile, ClientProfile) else ClientProfile.from_mapping(profile)
        consultation_in_progress = selected_profile != ClientProfile()
        updates = _infer_profile_updates(user_message)
        selected_profile = selected_profile.with_updates(updates)
        topic = _classify_topic(user_message)
        # A message that only answers earlier questions ("rent 25k, no other
        # income") continues the consultation instead of starting a new topic.
        if topic == "general" and consultation_in_progress and updates:
            topic = "salary_tax_planning"

        if topic == "salary_tax_planning":
            return self._salary_tax_planning_turn(
                user_message=user_message,
                profile=selected_profile,
                profile_updates=updates,
                top_k=top_k,
            )

        answer = self.pipeline.answer(user_message, top_k=top_k)
        return AgentResponse(
            status=str(answer.answer.get("status", "abstained")),
            message=str(answer.answer.get("answer") or ""),
            profile_updates=updates,
            retrieved_sources=format_retrieval_results(answer.retrieved),
            sources=source_payloads(answer.retrieved),
            answer=answer.answer,
        )

    def _salary_tax_planning_turn(
        self,
        user_message: str,
        profile: ClientProfile,
        profile_updates: Dict[str, Any],
        top_k: int,
    ) -> AgentResponse:
        missing_fields = _missing_salary_tax_fields(profile)
        # A fixed query: client facts never go into retrieval.
        retrieval_query = (
            "salary tax planning legal deductions old regime new regime HRA 80C 80D NPS "
            "income tax compliance documentation Form 16"
        )
        retrieved = self.pipeline.retriever.retrieve(retrieval_query, top_k=top_k)
        retrieved_sources = format_retrieval_results(retrieved)
        sources = source_payloads(retrieved)

        if missing_fields:
            questions = _questions_for_missing_fields(missing_fields)
            message = _build_fact_gathering_message(profile, questions)
            return AgentResponse(
                status="needs_info",
                message=message,
                questions=questions,
                profile_updates=profile_updates,
                retrieved_sources=retrieved_sources,
                sources=sources,
            )

        tax_input, assumptions = _salary_tax_input(profile)
        try:
            comparison = compare_regimes(tax_input)
        except UnsupportedTaxYear:
            return AgentResponse(
                status="unsupported_year",
                message=(
                    f"I can't calculate {tax_input.tax_year} yet: its tax rules have not been verified in this tool. "
                    f"I can calculate {LATEST_VERIFIED_TAX_YEAR} instead, or a CA can advise on {tax_input.tax_year}."
                ),
                profile_updates=profile_updates,
                retrieved_sources=retrieved_sources,
                sources=sources,
            )
        message = _build_calculation_message(comparison, assumptions, profile)
        explanation = self._explain_calculation(user_message, profile, comparison, [r.chunk for r in retrieved])
        if explanation and explanation["accepted"]:
            message = f"Why this result:\n{explanation['rendered']}\n\n{message}"
        return AgentResponse(
            status="calculated",
            message=message,
            profile_updates=profile_updates,
            retrieved_sources=retrieved_sources,
            sources=sources,
            calculation=comparison,
            explanation=explanation,
        )

    def _explain_calculation(
        self,
        user_message: str,
        profile: ClientProfile,
        comparison: RegimeComparison,
        chunks: Sequence[Mapping[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Ask the LLM to explain the result as typed claims; keep only verified ones.

        The calculator table is always shown, so any failure here only removes
        the explanation. Returns None when no real LLM is configured.
        """

        llm = getattr(getattr(self.pipeline, "answer_generator", None), "llm", None)
        if llm is None or isinstance(llm, EchoLLM):
            return None
        evidence = build_salary_evidence(profile.as_dict(), comparison, chunks)
        try:
            raw = llm.generate(CLAIMS_SYSTEM_PROMPT, build_claims_prompt(user_message, evidence), max_tokens=700)
        except Exception as error:  # A model failure must not lose the calculation.
            return {"accepted": [], "rejected": [], "error": f"generation_failed:{type(error).__name__}", "rendered": ""}
        result = validate_claims(raw, evidence, lower_regime=comparison.lower_regime)
        return {
            "accepted": result.accepted,
            "rejected": result.rejected,
            "error": result.error,
            "rendered": render_claims(result.accepted, evidence),
        }


def _classify_topic(message: str) -> str:
    text = message.lower()
    salary_terms = {"salary", "ctc", "earning", "income"}
    planning_terms = {"save tax", "tax saving", "deduction", "investment", "legal", "old regime", "new regime"}
    if any(term in text for term in salary_terms) and any(term in text for term in planning_terms):
        return "salary_tax_planning"
    if "tax" in text and any(term in text for term in {"save", "plan", "invest", "deduction", "regime"}):
        return "salary_tax_planning"
    return "general"


def _infer_profile_updates(message: str) -> Dict[str, Any]:
    text = message.lower().replace("’", "'")
    updates: Dict[str, Any] = {}

    salary = _amount_after(_SALARY_LABEL, text)
    if salary is not None:
        amount, period = salary
        updates["annual_salary_ctc"] = amount * 12 if period == "month" else amount

    age_match = re.search(r"\b(?:age|aged)\s*(?:is|:|=|-)?\s*(\d{2})\b", text) or re.search(
        r"\b(\d{2})\s*(?:years?|yrs?)\s*old\b", text
    )
    if age_match:
        updates["age"] = int(age_match.group(1))

    if "both regime" in text or "both regimes" in text or "compare both" in text:
        updates["tax_regime_preference"] = "compare"
    elif "old regime" in text:
        updates["tax_regime_preference"] = "old"
    elif "new regime" in text:
        updates["tax_regime_preference"] = "new"

    # Check the qualified statuses first: "resident" is a substring of both.
    if re.search(r"\bnot ordinarily resident\b|\brnor\b", text):
        updates["residential_status"] = "not-ordinarily-resident"
    elif re.search(r"\bnon[- ]?resident\b|\bnri\b|\bnot (?:an? )?(?:indian )?resident\b", text):
        updates["residential_status"] = "non-resident"
    elif re.search(r"\bresident\b", text):
        updates["residential_status"] = "resident"

    metro = next((city for city in _METRO_CITIES if re.search(rf"\b{city}\b", text)), None)
    if re.search(r"\bnon[- ]?metro\b", text):
        updates["city"] = "non-metro"
    elif metro:
        updates["city"] = metro
    elif re.search(r"\bmetro\b", text):
        updates["city"] = "metro"
    else:
        city_match = re.search(
            r"\b(?:live|living|based|stay|staying)\s+in\s+([a-z]+(?: [a-z]+){0,2}?)"
            r"(?=[,.;!?]|\s+(?:and|with|for|since|but|near|where|on|at)\b|$)",
            text,
        )
        if city_match and city_match.group(1).split()[0] not in _NON_CITY_WORDS:
            updates["city"] = city_match.group(1)

    if re.search(_NEGATION + r"rent\b", text):
        updates["monthly_rent"] = 0.0
    else:
        rent = _amount_after(r"\brent\b", text)
        if rent is not None:
            amount, period = rent
            updates["monthly_rent"] = amount / 12 if period == "year" else amount

    if re.search(_NEGATION + r"hra\b", text):
        updates["monthly_hra"] = 0.0
    else:
        hra = _amount_after(r"\bhra\b", text)
        if hra is not None:
            amount, period = hra
            updates["monthly_hra"] = amount / 12 if period == "year" else amount

    if re.search(_NEGATION + r"(?:investments?|deductions?)\b", text):
        updates["deductions_investments"] = "none"
    else:
        found = [label for label, pattern in _DEDUCTION_TERMS if re.search(pattern, text)]
        if found:
            updates["deductions_investments"] = ", ".join(found)

    if re.search(_NEGATION + r"(?:(?:health|medical)\s+insurance|mediclaim)\b", text):
        updates["health_insurance"] = "none"
    elif re.search(r"\b(?:health|medical)\s+insurance\b|\bmediclaim\b|\b80d\b", text):
        updates["health_insurance"] = "yes"

    if re.search(_NEGATION + r"other\s+income\b", text):
        updates["other_income"] = "none"
    else:
        found = [label for label, pattern in _OTHER_INCOME_TERMS if re.search(pattern, text)]
        if found:
            updates["other_income"] = ", ".join(found)

    basic = _amount_after(_BASIC_LABEL, text)
    if basic is not None:
        updates["annual_basic_da"] = _annual(basic)

    year_match = re.search(r"\b(?:fy|financial year)?\s*(20\d\d)\s*-\s*(\d\d)\b", text)
    if year_match:
        updates["tax_year"] = f"FY{year_match.group(1)}-{year_match.group(2)}"

    # Amounts for the calculator. "None" answers above count as zero.
    if updates.get("deductions_investments") == "none":
        updates["deduction_80c_amount"] = 0.0
    else:
        amount = _amount_after(r"\b(?:section\s+)?80c\b(?:\s+(?:investments?|deductions?|total))?", text)
        if amount is not None:
            updates["deduction_80c_amount"] = _annual(amount)

    if updates.get("health_insurance") == "none":
        updates["health_premium_self_family"] = 0.0
        updates["health_premium_parents"] = 0.0
    else:
        parents = _amount_after(_PARENTS_PREMIUM_LABEL, text)
        if parents is not None:
            updates["health_premium_parents"] = _annual(parents)
        own = _amount_after(_SELF_PREMIUM_LABEL, text)
        if own is not None:
            updates["health_premium_self_family"] = _annual(own)
    if re.search(r"\bsenior citizen parents?\b|\bparents? (?:are|is) (?:senior|above 60|over 60)\b", text):
        updates["parents_senior"] = True

    nps = _amount_after(r"\bnps\b(?:\s+contribution)?", text)
    if nps is not None:
        updates["nps_self_amount"] = _annual(nps)

    loan_interest = _amount_after(r"\b(?:home loan interest|interest on (?:my )?home loan)\b", text)
    if loan_interest is not None:
        updates["home_loan_interest"] = _annual(loan_interest)

    if updates.get("other_income") == "none":
        updates["other_income_amount"] = 0.0
    else:
        amounts = [_amount_after(label, text) for label in _OTHER_INCOME_AMOUNT_LABELS]
        found = [_annual(amount) for amount in amounts if amount is not None]
        if found:
            updates["other_income_amount"] = sum(found)

    if re.search(_NEGATION + r"(?:form\s*-?\s*16|salary slips?)\b", text):
        updates["employer_tds_or_form16"] = "not available"
    elif re.search(r"\bform\s*-?\s*16\b|\bsalary slips?\b", text):
        updates["employer_tds_or_form16"] = "available"

    return updates


# Cities treated as metro for HRA exemption purposes.
_METRO_CITIES = ("mumbai", "delhi", "kolkata", "chennai")

# Words that start a "live in ..." phrase that is not a city ("live in a rented flat").
_NON_CITY_WORDS = frozenset(
    {"a", "an", "the", "my", "our", "his", "her", "their", "rented", "own", "pg", "hostel",
     "house", "flat", "apartment", "india"}
)

# Prefix meaning "the client does not have/pay X"; append the thing being negated.
_NEGATION = (
    r"\b(?:no|without|(?:do not|don'?t|does not|doesn'?t|not|never)\s+(?:have|pay|get|receive|own)|"
    r"(?:haven'?t|hasn'?t)(?:\s+got)?|not\s+paying|not\s+getting)\s+(?:any\s+)?(?:an?\s+)?"
)

# Words that, right before "income", mean the income is not salary.
_NON_SALARY_INCOME_WORDS = (
    "other", "rental", "interest", "freelance", "freelancing", "dividend", "business",
    "professional", "consulting", "side", "passive", "gains", "property", "agricultural",
)
_SALARY_LABEL = (
    r"\b(?:salary|ctc|earning|"
    + "".join(f"(?<!{word} )" for word in _NON_SALARY_INCOME_WORDS)
    + r"income)"
)

_BASIC_LABEL = r"\bbasic(?:\s+(?:salary|pay))?(?:\s*(?:\+|and|plus)\s*da)?\b"
_PARENTS_PREMIUM_LABEL = (
    r"\bparents'?\s*(?:(?:health|medical)\s+)?(?:insurance|premium|mediclaim)(?:\s+premium)?\b"
)
_SELF_PREMIUM_LABEL = (
    r"(?<!parents )(?<!parents' )\b(?:(?:health|medical)\s+insurance(?:\s+premium)?|mediclaim|80d)\b"
)
_OTHER_INCOME_AMOUNT_LABELS = (
    r"\bother\s+income\b",
    r"(?<!loan )\binterest\s+income\b",
    r"\bfreelanc\w*\s+income\b",
    r"\brental\s+income\b",
)

_DEDUCTION_TERMS = (
    ("EPF", r"\bepf\b|\bprovident fund\b"),
    ("PPF", r"\bppf\b"),
    ("ELSS", r"\belss\b"),
    ("NPS", r"\bnps\b"),
    ("80C", r"\b80c\b"),
    ("life insurance", r"\blife insurance\b|\blic\b"),
    ("tuition fees", r"\btuition\b"),
    ("home loan", r"\bhome loan\b"),
    ("donations", r"\bdonations?\b|\b80g\b"),
)

_OTHER_INCOME_TERMS = (
    # Loan interest is a deduction, not income.
    ("interest", r"(?<!loan )(?<!housing )(?<!emi )\binterest\b(?!\s+(?:paid|on\s+(?:my\s+)?(?:home\s+|housing\s+)?loan))"),
    ("capital gains", r"\bcapital gains?\b"),
    ("freelancing", r"\bfreelanc"),
    ("rental income", r"\brental income\b"),
    ("RSUs/ESOPs", r"\brsus?\b|\besops?\b"),
)

_AMOUNT_UNITS = {
    "lpa": 100000.0, "lakh": 100000.0, "lakhs": 100000.0, "lac": 100000.0, "lacs": 100000.0, "l": 100000.0,
    "cr": 10000000.0, "crore": 10000000.0, "crores": 10000000.0,
    "k": 1000.0, "thousand": 1000.0,
}

# Filler allowed between a label and its amount. Excludes "," and "." so the
# match cannot run across clauses ("salary, 32 years old").
_AMOUNT_GAP = (
    r"(?:[\s:=\-₹]|\brs\.?|\b(?:is|of|about|around|approx|roughly|inr|my|i|get|paid|pay|"
    r"received|receive|amount|was|annual|annually|monthly|yearly|per|month|year|package)\b)*?"
)

_MONTHLY = re.compile(r"\bmonthly\b|\bper\s+month\b|\bmonth\b|\bpm\b|/\s*month\b|\ba\s+month\b|\bp\.?m\.?(?!\w)")
_YEARLY = re.compile(
    r"\b(?:annual|annually|yearly|year|per\s+annum|pa|p\.a\.?|a\s+year|lpa)\b|/\s*(?:year|yr|annum)\b"
)


def _amount_after(label_pattern: str, text: str) -> Optional[Tuple[float, Optional[str]]]:
    """Parse an amount that follows a label, separated only by filler words.

    Returns ``(amount, period)`` where period is "month", "year" or None when
    the message does not say. Accepts Indian and Western digit grouping
    ("12,00,000", "1,200,000"). The filler whitelist stops "salary slips and
    Form 16" from reading as 16.
    """

    match = re.search(
        label_pattern + "(" + _AMOUNT_GAP + r")(\d[\d,]*(?:\.\d+)?)\s*(lpa|lakhs?|lacs?|crores?|cr|k|thousand|l)?\b",
        text,
    )
    if not match:
        return None
    gap, digits, unit = match.group(1), match.group(2), match.group(3) or ""
    amount = float(digits.replace(",", "")) * _AMOUNT_UNITS.get(unit, 1.0)
    # The period can be the word right before the label ("monthly salary"), in
    # the gap, or just after the amount ("1 lakh per month"). Stay within the
    # clause so a neighbouring "20 lpa" or "per month" does not leak in.
    word_before = re.search(r"\b(monthly|yearly|annual|annually)\s*$", text[:match.start()])
    following = re.split(r"[,.;!?]", text[match.end():match.end() + 16], maxsplit=1)[0]
    around = " ".join([word_before.group(1) if word_before else "", gap, unit, following])
    if _MONTHLY.search(around):
        period: Optional[str] = "month"
    elif _YEARLY.search(around):
        period = "year"
    else:
        period = None
    return amount, period


def _annual(parsed: Tuple[float, Optional[str]]) -> float:
    """Annualise an amount whose period defaults to yearly."""

    amount, period = parsed
    return amount * 12 if period == "month" else amount


def _missing_salary_tax_fields(profile: ClientProfile) -> List[str]:
    """Facts the regime calculation still needs, in the order to ask them.

    Both regimes are always calculated, so a regime preference is not needed.
    Amount questions are asked only when the client said they have the item.
    """

    def missing(name: str) -> bool:
        return getattr(profile, name) in (None, "")

    needed = [
        name for name in ("annual_salary_ctc", "annual_basic_da", "age", "residential_status", "monthly_rent")
        if missing(name)
    ]
    if (
        not missing("annual_basic_da")
        and not missing("annual_salary_ctc")
        and profile.annual_basic_da > profile.annual_salary_ctc
    ):
        needed.append("annual_basic_da")
    if profile.monthly_rent:
        needed += [name for name in ("city", "monthly_hra") if missing(name)]
    for name in ("deductions_investments", "health_insurance", "other_income"):
        if missing(name):
            needed.append(name)

    deductions = (profile.deductions_investments or "").lower()
    if deductions and deductions != "none" and missing("deduction_80c_amount"):
        needed.append("deduction_80c_amount")
    if "nps" in deductions and missing("nps_self_amount"):
        needed.append("nps_self_amount")
    if "home loan" in deductions and missing("home_loan_interest"):
        needed.append("home_loan_interest")
    if (
        profile.health_insurance == "yes"
        and missing("health_premium_self_family")
        and missing("health_premium_parents")
    ):
        needed.append("health_premium_self_family")
    if profile.other_income and profile.other_income != "none" and missing("other_income_amount"):
        needed.append("other_income_amount")
    return list(dict.fromkeys(needed))


def _questions_for_missing_fields(missing_fields: Sequence[str]) -> List[str]:
    question_bank = {
        "annual_salary_ctc": "What is your annual gross salary (CTC)?",
        "annual_basic_da": "What is your basic salary plus DA per year? It must not exceed your gross salary.",
        "age": "What is your age?",
        "residential_status": "Are you resident, non-resident, or not ordinarily resident for Indian income-tax purposes?",
        "tax_regime_preference": "Are you currently opting for the new regime, old regime, or should we compare both?",
        "city": "Which city do you live in for HRA purposes: metro or non-metro?",
        "monthly_rent": "Do you pay rent? If yes, what is the monthly rent and do you have rent receipts/PAN of landlord where needed?",
        "monthly_hra": "How much HRA does your salary structure show per month?",
        "deductions_investments": "What deductions or investments do you already have: EPF, PPF, ELSS, life insurance, tuition fees, NPS, donations, etc.?",
        "health_insurance": "Do you pay medical insurance premium for self, family, or parents?",
        "other_income": "Do you have other income such as interest, capital gains, freelancing, rent, or RSUs/ESOPs?",
        "employer_tds_or_form16": "Do you have latest salary slips, Form 16, and current TDS/projection from employer?",
        "deduction_80c_amount": "How much do you invest under Section 80C per year in total (EPF, PPF, ELSS, life insurance, tuition fees, principal on home loan)?",
        "nps_self_amount": "How much do you contribute to NPS yourself per year?",
        "home_loan_interest": "How much home-loan interest do you pay per year on the house you live in?",
        "health_premium_self_family": "What is the yearly health insurance premium for yourself and family, and separately for your parents (and are they 60 or older)?",
        "other_income_amount": "Roughly how much other income (interest, freelancing, rent) do you get per year?",
    }
    return [question_bank[field_name] for field_name in missing_fields[:7]]


def _build_fact_gathering_message(profile: ClientProfile, questions: Sequence[str]) -> str:
    known = {
        key: value
        for key, value in profile.as_dict().items()
        if value not in (None, "")
    }
    known_text = ", ".join(f"{key}={value}" for key, value in known.items()) or "none yet"
    planning_levers = [
        "compare old vs new regime before choosing",
        "evaluate HRA/rent documentation if applicable",
        "map eligible deductions such as 80C-style investments, NPS, medical insurance, and home-loan items where supported",
        "verify other income and TDS so there is no under-reporting",
        "keep evidence: salary slips, Form 16, rent receipts, investment proofs, insurance receipts, interest certificates",
    ]
    return (
        "I can work like a CA consultation, but I need facts before giving a reliable tax-saving plan.\n\n"
        f"Known facts: {known_text}.\n\n"
        "Initial legal planning areas to evaluate:\n"
        + "\n".join(f"- {item}" for item in planning_levers)
        + "\n\nAnswer these first:\n"
        + "\n".join(f"{index}. {question}" for index, question in enumerate(questions, start=1))
    )


def _salary_tax_input(profile: ClientProfile) -> Tuple[SalaryTaxInput, List[str]]:
    """Map a complete profile to calculator input, listing every assumption made."""

    year = profile.tax_year or LATEST_VERIFIED_TAX_YEAR
    assumptions = [
        "Gross salary is taken as your stated CTC. Employer PF and gratuity in CTC are not taxable salary, "
        "so the real tax may be lower.",
    ]
    if not profile.tax_year:
        assumptions.append(f"Calculated under {year} rules, the latest year verified in this tool.")
    city = (profile.city or "").lower()
    metro = city in _METRO_CITIES or city == "metro"
    if profile.monthly_rent:
        assumptions.append(
            f"HRA exemption uses {'metro (50%)' if metro else 'non-metro (40%)'} limits for {(profile.city or '').title()}."
        )
    if profile.health_premium_parents and profile.parents_senior is None:
        assumptions.append("Parents are assumed to be under 60 for the Section 80D limit.")
    if profile.professional_tax is None:
        assumptions.append("No professional tax is assumed.")
    status = profile.residential_status or "resident"
    tax_input = SalaryTaxInput(
        tax_year=year,
        gross_salary=float(profile.annual_salary_ctc or 0),
        basic_da=float(profile.annual_basic_da or 0),
        age=int(profile.age or 0),
        # RNOR is still a resident for the rebate and senior slabs.
        resident=status in {"resident", "not-ordinarily-resident"},
        hra_received=float(profile.monthly_hra or 0) * 12,
        rent_paid=float(profile.monthly_rent or 0) * 12,
        metro=metro,
        professional_tax=float(profile.professional_tax or 0),
        other_income=float(profile.other_income_amount or 0),
        deduction_80c=float(profile.deduction_80c_amount or 0),
        health_premium_self_family=float(profile.health_premium_self_family or 0),
        health_premium_parents=float(profile.health_premium_parents or 0),
        parents_senior=bool(profile.parents_senior),
        nps_self_80ccd_1b=float(profile.nps_self_amount or 0),
        home_loan_interest_self_occupied=float(profile.home_loan_interest or 0),
    )
    return tax_input, assumptions


def _inr(amount: float) -> str:
    """Format rupees with Indian digit grouping: 1234567 -> Rs 12,34,567."""

    rounded = int(round(amount))
    sign, digits = ("-" if rounded < 0 else ""), str(abs(rounded))
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        head = ",".join([head[max(0, i - 2):i] for i in range(len(head), 0, -2)][::-1])
        digits = f"{head},{tail}"
    return f"{sign}\u20b9{digits}"


def _build_calculation_message(comparison: RegimeComparison, assumptions: Sequence[str], profile: ClientProfile) -> str:
    old, new = comparison.old, comparison.new
    rows = [
        ("Taxable income", old.taxable_income, new.taxable_income),
        ("Tax at slab rates", old.tax_on_income, new.tax_on_income),
        ("Rebate (87A)", -old.rebate, -new.rebate),
        ("Surcharge", old.surcharge, new.surcharge),
        ("Cess (4%)", old.cess, new.cess),
        ("Total tax", old.total_tax, new.total_tax),
    ]
    table = "\n".join(f"| {label} | {_inr(a)} | {_inr(b)} |" for label, a, b in rows)
    if comparison.lower_regime == "equal":
        verdict = "Both regimes give the same tax."
    else:
        verdict = (
            f"The {comparison.lower_regime} regime is lower by {_inr(comparison.saving)} for {comparison.tax_year}."
        )
    if profile.tax_regime_preference in {"old", "new"} and profile.tax_regime_preference != comparison.lower_regime \
            and comparison.lower_regime != "equal":
        verdict += f" You mentioned the {profile.tax_regime_preference} regime; it costs more on these figures."

    def deduction_lines(result_steps) -> List[str]:
        return [
            f"- {step.label.removeprefix('Less: ')}: {_inr(-step.amount)} ({step.source})"
            for step in result_steps
            if step.label.startswith("Less:")
        ]

    old_lines = deduction_lines(old.steps) or ["- none"]
    new_lines = deduction_lines(new.steps) or ["- none"]
    return "\n".join(
        [
            f"Old vs new regime, {comparison.tax_year} ({comparison.law}). Calculation {comparison.run_id}.",
            "",
            "| | Old regime | New regime |",
            "|---|---|---|",
            table,
            "",
            verdict,
            "",
            "Deductions applied, old regime:",
            *old_lines,
            "Deductions applied, new regime:",
            *new_lines,
            "",
            "Assumptions:",
            *[f"- {item}" for item in assumptions],
            "",
            "Not included in this calculation:",
            *[f"- {item}" for item in comparison.not_modelled],
            "",
            "Keep as evidence: Form 16, salary slips, rent receipts (and landlord PAN if rent exceeds "
            "\u20b91 lakh a year), investment and insurance premium receipts, interest certificates.",
            "This is a calculation to support your decision, not filing advice. Confirm with a CA before you file.",
        ]
    )
