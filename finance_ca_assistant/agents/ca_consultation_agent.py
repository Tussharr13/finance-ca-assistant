"""Conversational CA consultation agent built on top of RAG retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

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
    hra_received: Optional[float] = None
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
            status="answered",
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
            status="answered",
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
    text = message.lower()
    updates: Dict[str, Any] = {}

    salary_match = re.search(
        r"(?:salary|ctc|earning|income)[^\d]{0,20}(\d+(?:\.\d+)?)\s*(lpa|lakhs?|lacs?|cr|crore|k|thousand)?",
        text,
    )
    if salary_match:
        amount = float(salary_match.group(1))
        unit = salary_match.group(2) or ""
        multiplier = 1.0
        if unit in {"lpa", "lakh", "lakhs", "lac", "lacs"}:
            multiplier = 100000.0
        elif unit in {"cr", "crore"}:
            multiplier = 10000000.0
        elif unit in {"k", "thousand"}:
            multiplier = 1000.0
        updates["annual_salary_ctc"] = amount * multiplier

    age_match = re.search(r"(?:age|aged)\s*(?:is)?\s*(\d{2})", text)
    if age_match:
        updates["age"] = int(age_match.group(1))

    if "both regime" in text or "both regimes" in text or "compare both" in text:
        updates["tax_regime_preference"] = "compare"
    elif "old regime" in text:
        updates["tax_regime_preference"] = "old"
    elif "new regime" in text:
        updates["tax_regime_preference"] = "new"

    if "resident" in text:
        updates["residential_status"] = "resident"
    elif "nri" in text or "non-resident" in text:
        updates["residential_status"] = "non-resident"

    if "rent" in text or "hra" in text:
        updates["city"] = updates.get("city") or None

    return updates


def _missing_salary_tax_fields(profile: ClientProfile) -> List[str]:
    required = [
        "annual_salary_ctc",
        "age",
        "residential_status",
        "tax_regime_preference",
        "city",
        "monthly_rent",
        "hra_received",
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
        "hra_received": "How much HRA does your salary structure show?",
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
