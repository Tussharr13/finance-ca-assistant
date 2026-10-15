"""Retrieval fine-tuning: synthetic questions, page-level splits, scoring, training."""

from finance_ca_assistant.finetuning.dataset import QueryPair, mine_hard_negatives, split_pairs
from finance_ca_assistant.finetuning.embedding_trainer import (
    adopt_finetuned_model,
    find_finetuned_model,
    train_embedding_model,
)
from finance_ca_assistant.finetuning.query_generation import generate_queries, load_query_records
from finance_ca_assistant.finetuning.retrieval_metrics import dense_search_fn, evaluate_pairs

__all__ = [
    "QueryPair",
    "adopt_finetuned_model",
    "dense_search_fn",
    "evaluate_pairs",
    "find_finetuned_model",
    "generate_queries",
    "load_query_records",
    "mine_hard_negatives",
    "split_pairs",
    "train_embedding_model",
]
