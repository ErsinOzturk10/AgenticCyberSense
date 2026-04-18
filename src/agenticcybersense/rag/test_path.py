"""Test script to verify PDF paths and counts in the RAG data directory."""

from agenticcybersense.settings import settings

DATA_PATH = settings.pdf_docs_dir

pdfs = list(DATA_PATH.glob("*.pdf"))
