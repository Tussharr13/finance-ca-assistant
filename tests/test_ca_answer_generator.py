from finance_ca_assistant.generation.ca_answer_generator import CAAnswerGenerator
from finance_ca_assistant.generation.llm_factory import LLMClient


class JsonLLM(LLMClient):
    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 900) -> str:
        return """
<think>internal reasoning</think>
{
  "answer": "Clause 44 requires GST-related expenditure reporting.",
  "ca_analysis": "Use the cited Form 3CD context.",
  "risks": ["Do not report without supporting ledgers."],
  "missing_info": "GST expense breakup.",
  "next_steps": ["Collect ledger", "Review Form 3CD clause 44"],
  "disclaimer": "Educational guidance only."
}
"""


def test_generator_uses_parsed_llm_json_as_answer(sample_chunks):
    generator = CAAnswerGenerator(llm=JsonLLM())

    answer = generator.generate("What does clause 44 require?", sample_chunks)

    assert answer["answer"] == "Clause 44 requires GST-related expenditure reporting."
    assert answer["ca_analysis"] == "Use the cited Form 3CD context."
    assert answer["risks"] == ["Do not report without supporting ledgers."]
    assert answer["diagnostics"]["parsed_llm_json"] is True
    assert answer["basis"]["primary_sources"][0]["document"] == sample_chunks[0]["source"]
