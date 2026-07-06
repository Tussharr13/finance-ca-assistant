import pytest


@pytest.fixture
def sample_chunks():
    return [
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

