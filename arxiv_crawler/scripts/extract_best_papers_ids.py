"""Quick script to extract arxiv IDs from best_papers.md and merge with initial list."""

import json
import re
from pathlib import Path

# Pattern to match arxiv URLs like https://arxiv.org/abs/2503.11651
arxiv_url_pattern = r"(?:https?://)?(?:www\.)?arxiv\.org/abs/(\d{4}\.\d{4,}|\d{7})"

if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data"
    best_papers_path = data_dir / "best_papers.md"
    initial_ids_path = data_dir / "initial_arxiv_ids.json"
    output_path = data_dir / "best_papers_arxiv_ids.json"

    # Load initial arxiv IDs
    with open(initial_ids_path) as f:
        initial_ids = json.load(f)
    print(f"Loaded {len(initial_ids)} initial arxiv IDs")

    # Extract arxiv IDs from best_papers.md
    content = best_papers_path.read_text(encoding="utf-8")
    best_paper_ids = re.findall(arxiv_url_pattern, content)
    print(f"Found {len(best_paper_ids)} arxiv URLs in best_papers.md")

    # Merge: initial_ids first, then best_paper_ids, deduplicating
    seen = set()
    merged_ids = []
    for arxiv_id in initial_ids + best_paper_ids:
        if arxiv_id not in seen:
            seen.add(arxiv_id)
            merged_ids.append(arxiv_id)

    print(f"Merged to {len(merged_ids)} unique IDs")

    with open(output_path, "w") as f:
        json.dump(merged_ids, f, indent=2)

    print(f"Saved to {output_path}")
