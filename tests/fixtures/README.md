# Test Fixtures

This directory contains test data files for the arxiv_crawler test suite.

## Files

### `sample_paper.pdf`
A real arXiv paper used for integration testing. The paper should have:
- Multiple citations in the bibliography
- Citation references in the text body
- Ideally some citations with arXiv IDs

### `sample_paper_expected.json`
Golden fixture containing the expected output from processing `sample_paper.pdf`. This file is used for regression testing to ensure citation extraction behavior remains consistent.

**To regenerate:** `uv run python tests/generate_fixtures.py`
