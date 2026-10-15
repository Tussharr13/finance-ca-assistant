# Citation enforcement and confidence

Implemented: 2026-09-27. Policy: `exact_extract_v1`.

## Acceptance contract

The model returns JSON with 1–10 claims. Each claim contains string fields
`text`, `quote`, and `chunk_id`. The ID must identify exactly one retrieved
chunk with a nonblank source and a positive integer page number.

Both text and quote must equal the complete chunk text after whitespace
normalization. This deliberately rejects paraphrases and cropped excerpts:
removing “No” from “No deduction is allowed” must not pass validation.
The generation prompt includes the chunk IDs and complete text.

One invalid claim blocks the entire answer. Missing claims, free-form output,
malformed JSON, unknown or duplicate IDs, and invalid page metadata cause
abstention. The legacy lexical-overlap method remains available for diagnostics
and compatibility; it does not authorize generated answers.

Only validated extracts enter `answer`, `claims`, and `basis.primary_sources`.
Source/page attribution is taken from retrieved metadata. Other model fields,
including analysis, risks, next steps, confidence, and raw output, are never
forwarded. Auxiliary response fields contain fixed explanatory text.

## Response states

| Field | Accepted extracts | Rejected or disabled generation |
|---|---|---|
| `status` | `evidence_only` | `abstained` |
| `basis.evidence_status` | `exact_quotes_verified` | `unverified` |
| `basis.confidence` | `not_assessed` | `not_assessed` |
| `basis.primary_sources` | Full quotes and source/page/ID | Empty |

The default EchoLLM abstains. Retrieval results remain available separately in
`PipelineResponse.retrieved`; they are not automatically presented as validated
answer citations. Consultation propagates the answer status. Missing-fact
consultations continue to return `needs_info`.

## What this does not prove

Quote matching establishes attribution to retrieved text. It does not establish
truth, relevance, legal currency, applicability, completeness, source authenticity,
or the absence of conflicting evidence. Even a complete chunk can omit context
from surrounding pages. No numerical or high/medium/low confidence is assigned.
Adding more chunks cannot increase confidence.

This first policy sacrifices free-form synthesis for an inspectable acceptance
rule. A future synthesis mode needs its own evaluated support checks, explicit
freshness/applicability rules, and financial calculation validation. Full chunks
also consume more output tokens; truncated model JSON abstains rather than being
partially recovered.

## Verification and rollout

Regression tests cover valid attribution, unchecked model-field suppression,
malformed responses, mixed valid/invalid claims, altered text and numbers,
unknown/duplicate IDs, invalid pages, negation cropping, and default abstention.
Backend changes precede notebook integration; no live Kaggle or real-model
acceptance rate has been measured for this contract.
