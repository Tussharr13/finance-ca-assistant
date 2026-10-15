"""Fine-tune a sentence-transformers embedding model on CA question triplets."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional

FINETUNED_MARKER = "finance_ca_finetune.json"


def train_embedding_model(
    base_model: str,
    triplets: List[Dict[str, str]],
    output_dir: str | Path,
    epochs: int = 2,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    device: Optional[str] = None,
    metadata: Optional[Dict[str, object]] = None,
) -> Path:
    """Train with MultipleNegativesRankingLoss and save to ``output_dir``.

    Every other positive in a batch also serves as a negative, so batches must
    not repeat a passage; the no-duplicates sampler enforces that.
    """

    if not triplets:
        raise ValueError("No training triplets")
    try:
        from datasets import Dataset
        from sentence_transformers import (
            SentenceTransformer,
            SentenceTransformerTrainer,
            SentenceTransformerTrainingArguments,
        )
        from sentence_transformers.losses import MultipleNegativesRankingLoss
        from sentence_transformers.training_args import BatchSamplers
    except ImportError as exc:
        raise RuntimeError("Fine-tuning needs sentence-transformers>=3 and datasets") from exc

    output = Path(output_dir)
    model = SentenceTransformer(base_model, device=device)
    dataset = Dataset.from_list(triplets)
    use_fp16 = bool(device and device.startswith("cuda"))
    args = SentenceTransformerTrainingArguments(
        output_dir=str(output / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        fp16=use_fp16,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="no",
        logging_steps=20,
        report_to="none",
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        loss=MultipleNegativesRankingLoss(model),
    )
    trainer.train()
    model.save(str(output))
    shutil.rmtree(output / "checkpoints", ignore_errors=True)
    # Not adopted until it beats the current model; see adopt_finetuned_model.
    marker = {"base_model": base_model, "triplets": len(triplets), "epochs": epochs, "adopted": False, **(metadata or {})}
    (output / FINETUNED_MARKER).write_text(json.dumps(marker, indent=2), encoding="utf-8")
    return output


def adopt_finetuned_model(candidate_dir: str | Path, target_dir: str | Path) -> Path:
    """Mark a candidate as adopted and move it to ``target_dir``, replacing any previous model."""

    candidate, target = Path(candidate_dir), Path(target_dir)
    marker_path = candidate / FINETUNED_MARKER
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["adopted"] = True
    marker_path.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(candidate), str(target))
    return target


def find_finetuned_model(roots: Iterable[str | Path]) -> Optional[Path]:
    """Return the first adopted fine-tuned model directory under ``roots``.

    Candidates that were trained but not adopted are ignored, so an
    interrupted or losing run is never loaded by mistake.
    """

    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for marker_path in sorted(root_path.rglob(FINETUNED_MARKER)):
            try:
                adopted = json.loads(marker_path.read_text(encoding="utf-8")).get("adopted") is True
            except (OSError, ValueError):
                continue
            if adopted:
                return marker_path.parent
    return None
