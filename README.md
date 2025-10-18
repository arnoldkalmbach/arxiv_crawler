# arxiv_crawler
Create a dataset of arxiv papers by recursively finding and downloading links in References.

## Usage
```
# Either manually create data/initial_arxiv_ids.json or run
uv run scripts/initialize_list.py

# Start grobid server to help parse references
docker run --rm --gpus all --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.8.2-full

# Then run
uv run scripts/main.py
```

## Development

### Setup
Install dependencies including dev tools:
```bash
uv sync --dev
```

### Running Tests
To run tests, use `uv` to ensure the correct Python interpreter and environment:
```bash
# Run all tests
uv run pytest tests/ -v

# Run only unit tests (no Grobid server required)
uv run pytest tests/ -v -m "not integration"
```

### Linting and Formatting
```bash
uv run ruff check --fix .
uv run ruff format .
```