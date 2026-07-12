import json
from pathlib import Path


def _load_notebook_cells():
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "finance_ca_assistant_kaggle_mvp.ipynb"
    )
    return json.loads(notebook_path.read_text(encoding="utf-8"))["cells"]


def test_kaggle_section_1_purges_stale_package_modules():
    cells = _load_notebook_cells()
    section_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in cells
        if cell.get("cell_type") == "code"
    )

    assert "del sys.modules[module_name]" in section_source
    assert "importlib.invalidate_caches()" in section_source
    assert '"rev-parse", "--short", "HEAD"' in section_source
    assert "inspect.signature(build_knowledge_base_from_sources)" in section_source


def test_kaggle_section_3_reuses_cache_and_has_safe_limits():
    cells = _load_notebook_cells()
    start = next(
        index
        for index, cell in enumerate(cells)
        if "".join(cell.get("source", [])).startswith("## 3.")
    )
    end = next(
        index
        for index, cell in enumerate(cells[start + 1 :], start + 1)
        if "".join(cell.get("source", [])).startswith("## 4.")
    )
    section_source = "\n".join(
        "".join(cell.get("source", [])) for cell in cells[start:end]
    )

    assert "SOURCE_LIMIT = 3" in section_source
    assert "MAX_PAGES_PER_PDF = 30" in section_source
    assert "MAX_CHUNKS_PER_SOURCE = 300" in section_source
    assert 'PDF_BACKEND = "pymupdf"' in section_source
    assert "REBUILD_KB = False" in section_source
    assert "build_artifacts=False" in section_source


def test_kaggle_section_4_is_staged_and_safe_by_default():
    cells = _load_notebook_cells()

    start = next(
        index
        for index, cell in enumerate(cells)
        if "".join(cell.get("source", [])).startswith("## 4. Build Pipeline")
    )
    end = next(
        index
        for index, cell in enumerate(cells[start + 1 :], start + 1)
        if "".join(cell.get("source", [])).startswith("## 5.")
    )
    section_cells = cells[start:end]
    section_source = "\n".join("".join(cell.get("source", [])) for cell in section_cells)

    assert "USE_HF_EMBEDDINGS = True" not in section_source
    assert "USE_HF_RERANKER = True" not in section_source
    assert "USE_HF_LLM = True" not in section_source
    assert "ENABLE_HF_EMBEDDINGS = GPU_AVAILABLE" in section_source
    assert "ENABLE_HF_RERANKER = False" in section_source
    assert "ENABLE_HF_LLM = False" in section_source
    assert "MAX_CHUNKS_PER_SOURCE = 300" in section_source
    assert "BAAI/bge-small-en-v1.5" in section_source
    assert "Qwen/Qwen3-0.6B" in section_source
    assert "load_optional_provider" in section_source

    for offset, cell in enumerate(section_cells):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"section4_cell_{offset}", "exec")
