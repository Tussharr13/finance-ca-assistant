"""Conversational CA consultation agent built on top of RAG retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from finance_ca_assistant.retrieval.result_formatter import format_retrieval_results


@dataclass(frozen=True)
class ClientProfile:
    """Facts needed before a CA-style personal tax planning answer is reliable."""

    annual_salary_ctc: Optional[float] = None
    salary_currency: str = "INR"
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
    answer: Optional[Dict[str, Any]] = None


class CAConsultationAgent:
    """Agentic conversational wrapper around the CA RAG pipeline.

    The agent first gathers client facts, then uses RAG for grounded document
    context. It is intentionally conservative: it can suggest legal planning
    areas to evaluate, but it does not calculate liability or recommend a final
    tax position until required facts are available.
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
        updates = _infer_profile_updates(user_message)
        selected_profile = selected_profile.with_updates(updates)
        topic = _classify_topic(user_message)

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
        retrieval_query = (
            "salary tax planning legal deductions old regime new regime HRA 80C 80D NPS "
            "income tax compliance documentation Form 16"
        )
        retrieved = self.pipeline.retriever.retrieve(retrieval_query, top_k=top_k)
        retrieved_sources = format_retrieval_results(retrieved)

        if missing_fields:
            questions = _questions_for_missing_fields(missing_fields)
            message = _build_fact_gathering_message(profile, questions)
            return AgentResponse(
                status="needs_info",
                message=message,
                questions=questions,
                profile_updates=profile_updates,
                retrieved_sources=retrieved_sources,
            )

        grounded_question = _build_salary_tax_question(user_message, profile)
        answer = self.pipeline.answer(grounded_question, top_k=top_k)
        message = _build_advisory_message(answer.answer, profile)
        return AgentResponse(
            status=str(answer.answer.get("status", "abstained")),
            message=message,
            profile_updates=profile_updates,
            retrieved_sources=format_retrieval_results(answer.retrieved),
            answer=answer.answer,
        )


def _classify_topic(message: str) -> str:
    text = message.lower()
    salary_terms = {"salary", "ctc", "earning", "income"}
    planning_terms = {"save tax", "tax saving", "deduction", "investment", "legal", "old regime", "new regime"}
    if any(term in text for term in salary_terms) and any(term in text for term in planning_terms):
        return "salary_tax_planning"
    if "tax" in text and any(term in text for term in {"save", "planning", "invest", "deduction"}):
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


def _missing_salary_tax_fields(profile: ClientProfile) -> List[str]:
    required = [
        "annual_salary_ctc",
        "age",
        "residential_status",
        "tax_regime_preference",
        "city",
        "monthly_rent",
        "monthly_hra",
        "deductions_investments",
        "health_insurance",
        "other_income",
        "employer_tds_or_form16",
    ]
    return [field_name for field_name in required if getattr(profile, field_name) in (None, "")]


def _questions_for_missing_fields(missing_fields: Sequence[str]) -> List[str]:
    question_bank = {
        "annual_salary_ctc": "What is your annual CTC/gross salary, and what is the basic salary component?",
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


def _build_salary_tax_question(user_message: str, profile: ClientProfile) -> str:
    return (
        "Act as an Indian CA. Build a legal salary tax planning consultation using retrieved documents where available. "
        "Compare old and new regime considerations, identify deduction/documentation areas, red flags, and next steps. "
        f"Client facts: {profile.as_dict()}. User request: {user_message}"
    )


def _build_advisory_message(answer: Dict[str, Any], profile: ClientProfile) -> str:
    return (
        f"{answer.get('answer', '')}\n\n"
        f"CA analysis: {answer.get('ca_analysis', '')}\n\n"
        f"Risks/red flags: {', '.join(answer.get('risks') or [])}\n\n"
        f"Missing info: {answer.get('missing_info', '')}\n\n"
        f"Next steps: {', '.join(answer.get('next_steps') or [])}\n\n"
        f"Profile used: {profile.as_dict()}"
    ).strip()
