# arxiv_crawler

A monorepo containing tools for crawling arXiv papers and training citation embedding models.

## Repository Structure

This repository is organized as a **uv workspace** with two packages:

```
arxiv_crawler/           # Root workspace directory
├── pyproject.toml       # Workspace configuration
├── arxiv_crawler/       # Paper crawler package
│   ├── pyproject.toml   # Crawler dependencies (minimal)
│   ├── src/arxiv_crawler/
│   └── scripts/
└── arxiv_search/        # Search & embedding package  
    ├── pyproject.toml   # Search dependencies (includes arxiv-crawler + ML libs)
    ├── src/arxiv_search/
    ├── browser/         # Web UI for browsing papers
    └── scripts/
```

### Packages

| Package | Description |
|---------|-------------|
| **arxiv-crawler** | Crawls arXiv papers, parses PDFs via Grobid, extracts citations and metadata |
| **arxiv-search** | Trains citation embedding models, provides a web UI for searching papers |

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- Docker (for Grobid server)

### Installation

From the repository root:

```bash
# Install all dependencies (both packages)
uv sync

# Or install with dev dependencies
uv sync --dev
```

### Running the Crawler

```bash
# Start Grobid server for PDF parsing
docker run --rm --gpus all --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.8.2-full

# Initialize the seed list of arXiv IDs (or manually create data/initial_arxiv_ids.json)
uv run --package arxiv-crawler scripts/initialize_list.py

# Run the crawler
uv run --package arxiv-crawler scripts/main.py
```

### Running the Search/Browser

```bash
# Start the web UI (run from arxiv_search/ directory)
cd arxiv_search
uv run uvicorn browser.app:app --reload --port 8000
```

## Development

### Installing Dev Dependencies

```bash
uv sync --dev
```

### Running Tests

```bash
# Run all tests for arxiv-crawler
uv run --package arxiv-crawler pytest arxiv_crawler/tests/ -v

# Run only unit tests (no Grobid server required)
uv run --package arxiv-crawler pytest arxiv_crawler/tests/ -v -m "not integration"
```

### Linting and Formatting

```bash
# Lint and format arxiv_crawler
uv run --package arxiv-crawler ruff check --fix arxiv_crawler/
uv run --package arxiv-crawler ruff format arxiv_crawler/

# Lint and format arxiv_search
uv run --package arxiv-search ruff check --fix arxiv_search/
uv run --package arxiv-search ruff format arxiv_search/
```

## Workspace Notes

This repo uses a **uv workspace** to manage local dependencies between packages:

- `arxiv-search` depends on `arxiv-crawler` (resolved from workspace, not PyPI)
- Each package maintains its own dependency list
- A single lockfile (`uv.lock`) at the repo root ensures reproducible builds
- Running `uv sync` from the root installs both packages in editable mode

To work with a specific package:

```bash
# Run a command in the context of a specific package
uv run --package arxiv-crawler <command>
uv run --package arxiv-search <command>
```

## License

See [LICENSE](LICENSE) for details.
