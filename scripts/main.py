import requests
from io import BytesIO
import json

import polars as pl
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
from tqdm.auto import tqdm

from arxiv_crawler.arxiv_util import get_arxiv_metadata

# from arxiv_crawler.arxiv_util import CitationExtractor
# bib = CitationExtractor().process_paper("/home/arnold/Downloads/2501.03575v1.pdf")
# pl.from_dicts(list(bib.values())).unnest('details')


def extract_arxiv_text_simple(arxiv_url: str) -> str | None:
    """
    Extract full text from an arXiv PDF URL using PDFMiner.

    Args:
        arxiv_url: URL to the arXiv PDF

    Returns:
        Extracted text as string, or None if extraction fails
    """
    try:
        # Download PDF
        response = requests.get(arxiv_url)
        if response.status_code != 200:
            print(f"Failed to download PDF: {response.status_code}")
            return None

        # Extract text using PDFMiner
        pdf_file = BytesIO(response.content)
        laparams = LAParams(
            line_margin=0.5,
            word_margin=0.1,
            char_margin=2.0,
            boxes_flow=0.5,
        )

        text = extract_text(pdf_file, laparams=laparams)
        return text.strip()

    except Exception as e:
        print(f"Error extracting text: {str(e)}")
        return None


if __name__ == "__main__":
    with open("../data/initial_arxiv_ids.json", "r") as f:
        arxiv_ids = json.load(f)
    arxiv_metadata = get_arxiv_metadata(arxiv_ids)
    pl.from_dicts(arxiv_metadata).write_parquet("../data/arxiv_metadata.parquet")

    fulltexts = []
    for url in tqdm(arxiv_metadata["pdf_url"].to_list()):
        fulltexts.append(extract_arxiv_text_simple(url))

    arxiv_metadata.with_columns(pl.Series("text", fulltexts)).write_parquet("../data/papers.parquet")
