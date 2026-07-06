"""Run a local end-to-end pipeline smoke test without downloading PDFs."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finance_ca_assistant.pipeline import RAGPipeline
from finance_ca_assistant.retrieval.result_formatter import format_retrieval_results


SAMPLE_CHUNKS = [
    {
        "id": "tax-44",
        "source": "ICAI_Tax_Audit_44AB_Guidance_Note_2023.pdf",
        "page": 221,
        "text": "Clause 44 of Form 3CD requires reporting expenditure relating to GST.",
        "document_type": "form_3cd",
    },
    {
        "id": "sa-500",
        "source": "ICAI_SA_500_Audit_Evidence.pdf",
        "page": 9,
        "text": "SA 500 explains that audit evidence must be sufficient and appropriate.",
        "document_type": "audit_standard",
    },
]


def main() -> None:
    pipeline = RAGPipeline.from_chunks(SAMPLE_CHUNKS)
    response = pipeline.answer("What does Form 3CD clause 44 require?", top_k=2)
    print(format_retrieval_results(response.retrieved))
    print(response.answer["answer"])
    artifacts = pipeline.save_artifacts("/tmp/finance_ca_assistant_smoke")
    print("Saved artifacts:", artifacts)


if __name__ == "__main__":
    main()

