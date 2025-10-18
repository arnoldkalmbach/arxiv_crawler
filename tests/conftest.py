"""
Shared test fixtures and configuration for arxiv_crawler tests.
"""

import json
import pytest
from pathlib import Path


@pytest.fixture
def test_data_dir():
    """Return path to test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_pdf_path(test_data_dir):
    """
    Return path to a sample PDF file for testing.
    """
    pdf_path = test_data_dir / "sample_paper.pdf"
    if not pdf_path.exists():
        pytest.skip(f"Sample PDF not found at {pdf_path}. Please add test PDF files.")
    return str(pdf_path)


@pytest.fixture
def sample_paper_expected(test_data_dir):
    """
    Load expected output for sample_paper.pdf.

    To regenerate: uv run python tests/generate_fixtures.py
    """
    expected_path = test_data_dir / "sample_paper_expected.json"
    if not expected_path.exists():
        pytest.skip(f"Expected output not found at {expected_path}. Run: uv run python tests/generate_fixtures.py")

    with open(expected_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def grobid_url():
    """Default Grobid server URL for testing."""
    return "http://localhost:8070"
