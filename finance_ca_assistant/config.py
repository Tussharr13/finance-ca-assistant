"""Configuration loading for local, Kaggle, and API deployments."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union


DEFAULT_SOURCE_URLS = {
    "CGST_Rules_2017_CBIC": "https://cbic-gst.gov.in/pdf/cgst-rules-01july2017.pdf",
    "ICAI_Tax_Audit_44AB_Guidance_Note_2023": "https://resource.cdn.icai.org/75812dtc61332.pdf",
    "ICAI_AS_1_Accounting_Policies": "https://resource.cdn.icai.org/89095asb-aps2918-as1.pdf",
    "ICAI_AS_2_Inventories": "https://resource.cdn.icai.org/89096asb-aps2918-as2.pdf",
    "ICAI_AS_9_Revenue_Recognition": "https://resource.cdn.icai.org/89101asb-aps2918-as9.pdf",
    "ICAI_AS_10_PPE": "https://resource.cdn.icai.org/89102asb-aps2918-as10.pdf",
    "ICAI_AS_22_Taxes_on_Income": "https://resource.cdn.icai.org/89115asb-aps2918-as22.pdf",
    "ICAI_AS_29_Provisions_Contingencies": "https://resource.cdn.icai.org/89123asb-aps2918-as29.pdf",
    "ICAI_Ind_AS_Overview_Revised_2023": "https://resource.cdn.icai.org/75317asb60889.pdf",
    "ICAI_SA_200_Overall_Objectives": "https://resource.cdn.icai.org/18132sa200_rev.pdf",
    "ICAI_SA_500_Audit_Evidence": "https://resource.cdn.icai.org/15576sa500revised.pdf",
}

SMOKE_SOURCE_URLS = {
    "ICAI_AS_1_Accounting_Policies": DEFAULT_SOURCE_URLS["ICAI_AS_1_Accounting_Policies"],
    "ICAI_AS_2_Inventories": DEFAULT_SOURCE_URLS["ICAI_AS_2_Inventories"],
}


@dataclass(frozen=True)
class AppConfig:
    """Runtime configuration for the CA RAG system."""

    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    indices_dir: Path = Path("data/indices")
    manifest_path: Path = Path("data/pdfs_manifest.json")
    clause_index_path: Path = Path("data/clause_index.json")
    chunks_path: Path = Path("data/processed/chunks.jsonl")
    embedding_model: str = "BAAI/bge-m3"
    chat_model: str = "Qwen/Qwen3-4B"
    retrieval_top_k: int = 6
    source_urls: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_SOURCE_URLS))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AppConfig":
        """Create config from a dictionary, coercing path-like fields."""

        path_fields = {
            "data_dir",
            "raw_dir",
            "processed_dir",
            "indices_dir",
            "manifest_path",
            "clause_index_path",
            "chunks_path",
        }
        values: Dict[str, Any] = {}
        for key, value in data.items():
            values[key] = Path(value) if key in path_fields else value
        return cls(**values)


def load_config(path: Optional[Union[str, Path]] = None) -> AppConfig:
    """Load JSON or YAML configuration and overlay environment settings."""

    config_data: Dict[str, Any] = {}
    if path:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        config_data.update(_load_config_file(config_path))

    if os.getenv("HF_EMBED_MODEL"):
        config_data["embedding_model"] = os.environ["HF_EMBED_MODEL"]
    if os.getenv("HF_CHAT_MODEL"):
        config_data["chat_model"] = os.environ["HF_CHAT_MODEL"]

    return AppConfig.from_mapping(config_data) if config_data else AppConfig()


def ensure_directories(config: AppConfig) -> None:
    """Create configured data directories if missing."""

    for path in [config.data_dir, config.raw_dir, config.processed_dir, config.indices_dir]:
        path.mkdir(parents=True, exist_ok=True)


def _load_config_file(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("Install pyyaml to read YAML config files") from exc
        with path.open("r", encoding="utf-8") as handle:
            return dict(yaml.safe_load(handle) or {})

    with path.open("r", encoding="utf-8") as handle:
        return dict(json.load(handle))
