"""Configuration loading for local, Kaggle, and API deployments."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union


@dataclass(frozen=True)
class SourceDefinition:
    """Versioned metadata for one authoritative knowledge source."""

    source_id: str
    title: str
    url: str
    authority: str
    document_type: str
    version: str
    jurisdiction: str = "India"
    publication_date: Optional[str] = None
    effective_from: Optional[str] = None
    effective_to: Optional[str] = None
    financial_year: Optional[str] = None
    assessment_year: Optional[str] = None
    supersedes: Tuple[str, ...] = ()
    expected_sha256: Optional[str] = None


def _source(
    source_id: str,
    title: str,
    url: str,
    authority: str,
    document_type: str,
    version: str,
) -> SourceDefinition:
    return SourceDefinition(
        source_id=source_id,
        title=title,
        url=url,
        authority=authority,
        document_type=document_type,
        version=version,
    )


DEFAULT_SOURCE_CATALOG = {
    source.source_id: source
    for source in [
        _source("CGST_Rules_2017_CBIC", "Central Goods and Services Tax Rules", "https://cbic-gst.gov.in/pdf/cgst-rules-01july2017.pdf", "CBIC", "gst_rules", "2017-source"),
        _source("ICAI_Tax_Audit_44AB_Guidance_Note_2023", "Guidance Note on Tax Audit under Section 44AB", "https://resource.cdn.icai.org/75812dtc61332.pdf", "ICAI", "form_3cd", "2023"),
        _source("ICAI_AS_1_Accounting_Policies", "AS 1 Disclosure of Accounting Policies", "https://resource.cdn.icai.org/89095asb-aps2918-as1.pdf", "ICAI", "accounting_standard", "repository-current"),
        _source("ICAI_AS_2_Inventories", "AS 2 Valuation of Inventories", "https://resource.cdn.icai.org/89096asb-aps2918-as2.pdf", "ICAI", "accounting_standard", "repository-current"),
        _source("ICAI_AS_9_Revenue_Recognition", "AS 9 Revenue Recognition", "https://resource.cdn.icai.org/89101asb-aps2918-as9.pdf", "ICAI", "accounting_standard", "repository-current"),
        _source("ICAI_AS_10_PPE", "AS 10 Property, Plant and Equipment", "https://resource.cdn.icai.org/89102asb-aps2918-as10.pdf", "ICAI", "accounting_standard", "repository-current"),
        _source("ICAI_AS_22_Taxes_on_Income", "AS 22 Accounting for Taxes on Income", "https://resource.cdn.icai.org/89115asb-aps2918-as22.pdf", "ICAI", "accounting_standard", "repository-current"),
        _source("ICAI_AS_29_Provisions_Contingencies", "AS 29 Provisions, Contingent Liabilities and Contingent Assets", "https://resource.cdn.icai.org/89123asb-aps2918-as29.pdf", "ICAI", "accounting_standard", "repository-current"),
        _source("ICAI_Ind_AS_Overview_Revised_2023", "Indian Accounting Standards Overview", "https://resource.cdn.icai.org/75317asb60889.pdf", "ICAI", "accounting_standard", "2023"),
        _source("ICAI_SA_200_Overall_Objectives", "SA 200 Overall Objectives of the Independent Auditor", "https://resource.cdn.icai.org/18132sa200_rev.pdf", "ICAI", "audit_standard", "revised"),
        _source("ICAI_SA_500_Audit_Evidence", "SA 500 Audit Evidence", "https://resource.cdn.icai.org/15576sa500revised.pdf", "ICAI", "audit_standard", "revised"),
    ]
}

DEFAULT_SOURCE_URLS = {
    source_id: source.url for source_id, source in DEFAULT_SOURCE_CATALOG.items()
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
