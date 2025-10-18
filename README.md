# arxiv_crawler
Create a dataset of arxiv papers by recursively finding and downloading links in References.

## Usage
```
# Either manually create data/initial_arxiv_ids.json
# or run
uv run scripts/initialize_list.py

# Then run
uv run scripts/main.py
```

## Development
Run linter and formatter
```bash
uv run ruff check --fix .
uv run ruff format .
```