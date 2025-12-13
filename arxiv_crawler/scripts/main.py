import argparse
import json
from arxiv_crawler.crawler import ArxivCrawler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arxiv citation crawler")
    parser.add_argument(
        "--seed-ids-file",
        type=str,
        default="data/v2/initial_arxiv_ids.json",
        help="Path to JSON file with seed arxiv IDs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/v2",
        help="Output directory for crawled data",
    )
    parser.add_argument(
        "--grobid-url",
        type=str,
        default="http://localhost:8070",
        help="GROBID server URL",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=20000,
        help="Maximum number of papers to crawl",
    )
    parser.add_argument(
        "--rate-limit-delay",
        type=float,
        default=3.0,
        help="Seconds between arxiv API calls",
    )
    parser.add_argument(
        "--priority",
        type=lambda s: [x.strip() for x in s.split(",")],
        default=["num_citations", "depth"],
        help="Priority for selecting next paper, e.g., 'num_citations,depth' or 'depth,num_citations'",
    )
    args = parser.parse_args()

    # Load seed papers
    with open(args.seed_ids_file, "r") as f:
        seed_arxiv_ids = json.load(f)

    print(f"Loaded {len(seed_arxiv_ids)} seed papers from {args.seed_ids_file}")

    # Create crawler and run
    crawler = ArxivCrawler(
        output_dir=args.output_dir,
        grobid_url=args.grobid_url,
        max_papers=args.max_papers,
        rate_limit_delay=args.rate_limit_delay,
        priority=("depth", "num_citations"),
    )

    crawler.crawl(seed_arxiv_ids=seed_arxiv_ids)
