"""Robust PDF downloader with cache fallback and checksum validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from finance_ca_assistant.ingestion.source_registry import compute_sha256
from finance_ca_assistant.logger import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class DownloadResult:
    """Result of a PDF download or cache fallback."""

    url: str
    path: Path
    sha256: str
    from_cache: bool


class PDFDownloader:
    """Download PDFs into a raw-data directory.

    A caller may pass a custom ``session`` object with a ``get`` method for
    tests or enterprise HTTP clients. If download fails and a cached file
    exists, the downloader returns the cache instead of breaking a pipeline run.
    """

    def __init__(self, raw_dir: str | Path = "data/raw", session: Optional[object] = None) -> None:
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.session = session

    def download(
        self,
        url: str,
        target_name: Optional[str] = None,
        expected_sha256: Optional[str] = None,
        timeout: int = 30,
        use_cache_on_failure: bool = True,
        prefer_cache: bool = False,
    ) -> DownloadResult:
        """Download a PDF and validate its hash when provided."""

        filename = target_name or _filename_from_url(url)
        path = self.raw_dir / filename
        partial_path = path.with_suffix(f"{path.suffix}.part")

        if prefer_cache and path.exists():
            sha256 = compute_sha256(path)
            if expected_sha256 and sha256 != expected_sha256:
                raise ValueError(
                    f"SHA256 mismatch for cached {url}: expected {expected_sha256}, got {sha256}"
                )
            return DownloadResult(url=url, path=path, sha256=sha256, from_cache=True)

        try:
            self._fetch_to_path(url, partial_path, timeout=timeout)
            with partial_path.open("rb") as handle:
                pdf_header = handle.read(4)
            if pdf_header != b"%PDF":
                logger.warning("Downloaded content for %s does not start with PDF header", url)
            sha256 = compute_sha256(partial_path)
            if expected_sha256 and sha256 != expected_sha256:
                raise ValueError(
                    f"SHA256 mismatch for {url}: expected {expected_sha256}, got {sha256}"
                )
            partial_path.replace(path)
            return DownloadResult(url=url, path=path, sha256=sha256, from_cache=False)
        except Exception as error:
            partial_path.unlink(missing_ok=True)
            if use_cache_on_failure and path.exists():
                logger.warning("Download failed for %s; using cached %s: %s", url, path, error)
                return DownloadResult(
                    url=url,
                    path=path,
                    sha256=compute_sha256(path),
                    from_cache=True,
                )
            raise

    def _fetch_to_path(self, url: str, path: Path, timeout: int) -> None:
        """Stream a response to disk without retaining the full PDF in RAM."""

        session = self.session
        if session is None:
            import requests

            session = requests
        try:
            response = session.get(url, timeout=timeout, stream=True)
        except TypeError:
            response = session.get(url, timeout=timeout)

        try:
            response.raise_for_status()
            with path.open("wb") as handle:
                if hasattr(response, "iter_content"):
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                else:
                    handle.write(bytes(response.content))
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()


def _filename_from_url(url: str) -> str:
    filename = url.rstrip("/").split("/")[-1] or "source.pdf"
    return filename if filename.lower().endswith(".pdf") else f"{filename}.pdf"
