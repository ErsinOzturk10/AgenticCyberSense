"""Test script to verify PDF paths and counts in the RAG data directory."""

from pathlib import Path

DATA_PATH = Path("/Users/merveatay/Projekte/AgenticAI/AgenticCyberSense.v2/src/agenticcybersense/data")

pdfs = list(DATA_PATH.glob("*.pdf"))
