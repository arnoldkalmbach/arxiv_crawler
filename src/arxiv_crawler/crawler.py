import requests
import json
import tempfile
import gzip
from datetime import datetime
from pathlib import Path
from typing import Optional
import time

from tqdm.auto import tqdm

from arxiv_crawler.arxiv_util import (
    get_arxiv_metadata,
    normalize_arxiv_id,
    CitationExtractor,
)
from arxiv_crawler.models import ProcessedPaper, ProcessedCitation


class ArxivCrawler:
    """
    Arxiv crawler that discovers papers through citations.

    Papers are written incrementally to a JSONL file for efficient appending.
    To convert to parquet later: pl.read_ndjson('papers.jsonl').write_parquet('papers.parquet')
    """

    def __init__(
        self,
        output_dir: str = "../data",
        grobid_url: str = "http://localhost:8070",
        max_papers: int = 100,
        rate_limit_delay: float = 3.0,  # seconds between arxiv API calls
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.grobid_url = grobid_url
        self.max_papers = max_papers
        self.rate_limit_delay = rate_limit_delay

        # File paths
        self.papers_file: Path = self.output_dir / "papers.jsonl"
        self.state_file: Path = self.output_dir / "crawler_state.json"
        self.xml_dir: Path = self.output_dir / "xml_docs"
        self.xml_dir.mkdir(parents=True, exist_ok=True)

        # Citation extractor
        self.citation_extractor = CitationExtractor(grobid_url=grobid_url)

        # State tracking
        self.processed_ids: set[str] = set()  # Successfully processed arxiv ids
        self.failed_ids: set[str] = set()  # Failed to process arxiv ids (don't retry)
        self.queued_ids: dict[str, tuple[int, int]] = {}  # arxiv_id -> (num citations seen so far, depth)

        self._load_state()

    def _load_state(self):
        """Load previous crawler state if it exists."""
        if self.state_file.exists():
            print(f"Loading previous state from {self.state_file}")
            with open(self.state_file, "r") as f:
                state = json.load(f)
                self.processed_ids = set(state.get("processed_ids", []))
                self.failed_ids = set(state.get("failed_ids", []))
                self.queued_ids = state.get("queued_ids", {})
                print(f"  - Processed: {len(self.processed_ids)} papers")
                print(f"  - Failed: {len(self.failed_ids)} papers")
                print(f"  - Queued: {len(self.queued_ids)} papers")

    def _save_state(self):
        """Save current crawler state."""
        state = {
            "processed_ids": list(self.processed_ids),
            "failed_ids": list(self.failed_ids),
            "queued_ids": self.queued_ids,
            "last_updated": datetime.now().isoformat(),
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _process_paper(self, arxiv_id: str, depth: int) -> Optional[ProcessedPaper]:
        """Process a single paper: fetch metadata, text, and citations."""

        print(f"\n[Depth {depth}] Processing: {arxiv_id}")

        try:
            # Fetch metadata
            print("  - Fetching metadata...")
            time.sleep(self.rate_limit_delay)  # Rate limiting
            metadata_list = get_arxiv_metadata([arxiv_id])

            if not metadata_list:
                print(f"  ✗ No metadata found for {arxiv_id}")
                return None

            metadata = metadata_list[0]

            # Download PDF once to temp file for both text extraction and Grobid
            print("  - Downloading PDF...")
            response = requests.get(metadata["pdf_url"], timeout=30)
            response.raise_for_status()

            # Write to temp file - we need delete=False because we must close the file
            # before PDFMiner and Grobid can open it, but we still need it to exist
            with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as tmp_file:
                tmp_file.write(response.content)
                tmp_pdf_path = Path(tmp_file.name)

            try:
                # Extract citations and XML using Grobid from the PDF
                print("  - Extracting citations and XML with Grobid...")
                citations, xml_content = self.citation_extractor.process_paper(str(tmp_pdf_path))
            finally:
                # Clean up temp file that was created with delete=False
                tmp_pdf_path.unlink(missing_ok=True)

            # Save XML to compressed file
            xml_filename = f"{arxiv_id.replace('/', '_')}.xml.gz"
            xml_file_path = self.xml_dir / xml_filename
            print("  - Saving compressed XML...")
            with gzip.open(xml_file_path, "wb") as gz_file:
                gz_file.write(xml_content)

            # Process citations and update queue
            discovered_arxiv_ids = []
            processed_citations = []

            for citation in citations.values():
                # Convert Citation to ProcessedCitation
                processed_cit = ProcessedCitation(
                    citation_id=citation.citation_id,
                    authors=citation.details.authors,
                    title=citation.details.title,
                    year=citation.details.year,
                    venue=citation.details.venue,
                    arxiv_id=citation.details.arxiv_id,
                    reference_contexts=citation.references,
                    num_references=len(citation.references),
                )
                processed_citations.append(processed_cit)

                # If citation has arxiv ID, track it
                if citation.details.arxiv_id:
                    cited_id = normalize_arxiv_id(citation.details.arxiv_id)
                    discovered_arxiv_ids.append(cited_id)
                    if cited_id not in self.processed_ids and cited_id not in self.failed_ids:
                        if cited_id not in self.queued_ids:
                            self.queued_ids[cited_id] = (1, depth + 1)
                        else:
                            _, original_depth = self.queued_ids[cited_id]
                            self.queued_ids[cited_id] = (self.queued_ids[cited_id][0] + 1, original_depth)

            print(f"  ✓ Found {len(citations)} citations, {len(discovered_arxiv_ids)} with arxiv IDs")

            # Create ProcessedPaper object
            result = ProcessedPaper(
                arxiv_id=arxiv_id,
                title=metadata["title"],
                authors=metadata["authors"],
                abstract=metadata["abstract"],
                categories=metadata["categories"],
                published=metadata["published"],
                pdf_url=metadata["pdf_url"],
                arxiv_url=metadata["arxiv_url"],
                xml_file_path=str(xml_file_path.relative_to(self.output_dir)),
                citations=processed_citations,
                num_citations=len(citations),
                num_arxiv_citations=len(discovered_arxiv_ids),
                depth=depth,
                processing_timestamp=datetime.now().isoformat(),
            )

            return result

        except Exception as e:
            print(f"  ✗ Error processing {arxiv_id}: {str(e)}")
            import traceback

            traceback.print_exc()
            return None

    def _append_result(self, paper_data: ProcessedPaper):
        """Append paper data to JSONL file (efficient incremental writes)."""
        # Append a single JSON line to the file
        with open(self.papers_file, "a") as f:
            # Use Pydantic's model_dump to convert to dict, then serialize to JSON
            f.write(paper_data.model_dump_json() + "\n")

    def crawl(self, seed_arxiv_ids: list[str]):
        """
        Main crawl loop.

        Args:
            seed_arxiv_ids: Initial arxiv IDs to start crawling from
        """
        # Initialize queue with seed papers
        for arxiv_id in seed_arxiv_ids:
            arxiv_id = normalize_arxiv_id(arxiv_id)
            if arxiv_id in self.processed_ids or arxiv_id in self.failed_ids:
                print(f"  - Skipping {arxiv_id} because it has already been processed or failed")
            elif arxiv_id in self.queued_ids:
                print(f"  - Skipping {arxiv_id} because it is already in the queue")
            else:
                self.queued_ids[arxiv_id] = (0, 0)

        print(f"\n{'=' * 60}")
        print(f"Starting crawl with {len(seed_arxiv_ids)} seed papers")
        print(f"Max papers: {self.max_papers}")
        print(f"Grobid URL: {self.grobid_url}")
        print(f"Output: {self.papers_file}")
        print(f"{'=' * 60}\n")

        papers_processed = 0

        with tqdm(total=self.max_papers, desc="Crawling papers") as pbar:
            pbar.update(len(self.processed_ids))

            while self.queued_ids and papers_processed < self.max_papers:
                # Pop highest priority paper: max num citations, min depth
                arxiv_id, (num_citations, depth) = max(self.queued_ids.items(), key=lambda x: (x[1][0], -x[1][1]))
                del self.queued_ids[arxiv_id]

                # Process paper
                paper_data = self._process_paper(arxiv_id, depth)

                if paper_data:
                    # Success: save to JSONL and mark as processed
                    self._append_result(paper_data)
                    self.processed_ids.add(arxiv_id)
                    papers_processed += 1
                    pbar.update(1)
                    print(f"  ✓ Successfully processed ({papers_processed}/{self.max_papers})")
                else:
                    # Failure: mark as failed so we don't retry
                    self.failed_ids.add(arxiv_id)
                    print("  ✗ Marked as failed (won't retry)")

                # Save state after each paper
                self._save_state()

                # Show queue status
                print(
                    f"  Queue size: {len(self.queued_ids)} | Processed: {len(self.processed_ids)} | Failed: {len(self.failed_ids)}"
                )

        print(f"\n{'=' * 60}")
        print("Crawl complete!")
        print(f"  - Processed: {len(self.processed_ids)} papers")
        print(f"  - Failed: {len(self.failed_ids)} papers")
        print(f"  - Remaining in queue: {len(self.queued_ids)}")
        print(f"  - Output: {self.papers_file}")
        print(f"{'=' * 60}\n")
