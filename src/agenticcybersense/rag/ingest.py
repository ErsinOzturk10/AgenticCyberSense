"""RAG ingestion helpers (placeholders)."""

from pathlib import Path

from agenticcybersense.logging_utils import get_logger
from agenticcybersense.settings import settings

logger = get_logger(__name__)


def ingest_documents(documents_dir: str | None = None) -> int:
    """Ingest PDF documents into the vector store.

    Args:
        documents_dir: Directory containing PDF files. Defaults to settings.

    Returns:
        Number of documents ingested.

    """
    documents_dir = documents_dir or settings.pdf_documents_dir
    docs_path = Path(documents_dir)

    if not docs_path.exists():
        logger.warning("Documents directory does not exist: %s", documents_dir)
        docs_path.mkdir(parents=True, exist_ok=True)
        return 0

    pdf_files = list(docs_path.glob("*.pdf"))
    logger.info("Found %d PDF files to ingest", len(pdf_files))

    # Placeholder: PDF loading/chunking not implemented in this stub.

    for pdf_file in pdf_files:
        _ingest_single_pdf(pdf_file)

    return len(pdf_files)


def _ingest_single_pdf(pdf_path: Path) -> None:
    """Ingest a single PDF file.

    Args:
        pdf_path: Path to the PDF file.

    """
    logger.info("Ingesting PDF: %s", pdf_path.name)
    # Placeholder ingestion: implement actual loader integration when needed.


def clear_vector_store() -> None:
    """Clear all documents from the vector store."""
    logger.info("Clearing vector store (placeholder)")
    # Placeholder: implement ChromaDB collection clearing in integration.
