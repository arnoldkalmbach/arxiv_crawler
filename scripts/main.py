import json
from arxiv_crawler.crawler import ArxivCrawler


if __name__ == "__main__":
    # Configuration
    CONFIG = {
        "seed_ids_file": "../data/initial_arxiv_ids.json",
        "output_dir": "../data",
        "grobid_url": "http://localhost:8070",
        "max_papers": 100,
        "rate_limit_delay": 3.0,  # seconds between arxiv API calls
    }

    # Load seed papers
    with open(CONFIG["seed_ids_file"], "r") as f:
        seed_arxiv_ids = json.load(f)

    print(f"Loaded {len(seed_arxiv_ids)} seed papers from {CONFIG['seed_ids_file']}")

    # Create crawler and run
    crawler = ArxivCrawler(
        output_dir=CONFIG["output_dir"],
        grobid_url=CONFIG["grobid_url"],
        max_papers=CONFIG["max_papers"],
        rate_limit_delay=CONFIG["rate_limit_delay"],
    )

    crawler.crawl(seed_arxiv_ids=seed_arxiv_ids)
