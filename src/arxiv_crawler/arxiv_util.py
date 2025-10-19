from typing import Any
import re
import requests
import feedparser
import tempfile
from io import BytesIO
from lxml import etree
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams

from arxiv_crawler.models import Citation, CitationDetails


arxiv_url_pattern = r"(?:https?://)?(?:www\.)?arxiv\.org/abs/(?:\d{4}\.\d{4,}|\d{7})"
arxiv_id_pattern = r"arXiv.*?(\d{4}\.\d{4,}|\d{7})"


def normalize_arxiv_id(arxiv_id: str) -> str:
    """Normalize arxiv ID by removing version suffix (v1, v2, etc.)."""
    if not arxiv_id:
        return arxiv_id
    # Remove version suffix like "v1", "v2"
    if "v" in arxiv_id and arxiv_id.split("v")[-1].isdigit():
        return arxiv_id.rsplit("v", 1)[0]
    return arxiv_id


def extract_arxiv_text_simple(arxiv_url_or_path: str) -> str | None:
    """
    Extract full text from an arXiv PDF URL or file path using PDFMiner.

    Args:
        arxiv_url_or_path: URL to the arXiv PDF or path to a local PDF file

    Returns:
        Extracted text as string, or None if extraction fails
    """
    try:
        laparams = LAParams(
            line_margin=0.5,
            word_margin=0.1,
            char_margin=2.0,
            boxes_flow=0.5,
        )

        # Check if it's a URL or a file path
        if arxiv_url_or_path.startswith("http://") or arxiv_url_or_path.startswith("https://"):
            # Download PDF
            response = requests.get(arxiv_url_or_path, timeout=30)
            if response.status_code != 200:
                print(f"Failed to download PDF: {response.status_code}")
                return None
            pdf_file = BytesIO(response.content)
            text = extract_text(pdf_file, laparams=laparams)
        else:
            # Extract from local file
            text = extract_text(arxiv_url_or_path, laparams=laparams)

        return text.strip()

    except Exception as e:
        print(f"Error extracting text: {str(e)}")
        return None


def get_arxiv_metadata(arxiv_ids: list[str]) -> list[dict[str, Any]]:
    unique_arxiv_ids = list(set(arxiv_ids))

    base_url = "http://export.arxiv.org/api/query"
    response = requests.get(base_url, params={"id_list": ",".join(unique_arxiv_ids), "max_results": 1000})
    feed = feedparser.parse(response.text)

    arxiv_metadata = []
    for paper in feed["entries"]:
        arxiv_metadata.append(
            {
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "abstract": paper.summary,
                "categories": [tag["term"] for tag in paper.tags] if "tags" in paper else [],
                "published": paper.published,
                "pdf_url": [link["href"] for link in paper.links if link["type"] == "application/pdf"][0],
                "arxiv_url": paper.link,
            }
        )

    return arxiv_metadata


class CitationExtractor:
    def __init__(self, grobid_url: str = "http://localhost:8070"):
        """Initialize the citation extractor with Grobid server URL."""
        self.grobid_url = grobid_url.rstrip("/")
        self.ns = {"tei": "http://www.tei-c.org/ns/1.0"}

    def _download_pdf(self, url: str) -> str:
        """Download PDF from arXiv URL."""
        response = requests.get(url)
        response.raise_for_status()

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        with temp_file as f:
            f.write(response.content)
        return temp_file.name

    def _get_text(self, elem) -> str:
        """Get all text content from an element."""
        return " ".join(elem.xpath(".//text()")).strip()

    def _get_sentence_context(self, ref_elem) -> str:
        """Extract the sentence containing a citation reference."""
        # Try to find parent sentence element
        sentence = ref_elem.xpath("./ancestor::tei:s", namespaces=self.ns)
        if sentence:
            return self._get_text(sentence[0])

        # Fallback to paragraph and try to find the relevant sentence
        paragraph = ref_elem.xpath("./ancestor::tei:p", namespaces=self.ns)
        if paragraph:
            para_text = self._get_text(paragraph[0])
            ref_text = self._get_text(ref_elem)

            # Simple sentence splitting (could be improved)
            sentences = para_text.split(". ")
            for sent in sentences:
                if ref_text in sent:
                    return sent.strip()

        return None

    def _extract_arxiv_id(self, bib_elem, venue_list) -> str | None:
        """
        Extract arXiv ID from bibliography entry using multiple strategies.

        Handles various formats:
        - Explicit: <idno type="arXiv">arXiv:1902.05509</idno>
        - CoRR: <idno>CoRR, abs/2004.10934</idno>
        - URLs: http://arxiv.org/abs/2004.10934
        - Preprint: arXiv preprint arXiv:2004.10934
        """
        # Strategy 1: Explicit arXiv identifier
        arxiv_idno = bib_elem.xpath('.//tei:idno[@type="arXiv"]/text()', namespaces=self.ns)
        if arxiv_idno:
            text = arxiv_idno[0]
            # Handle "arXiv:XXXX.XXXXX" or "arXiv XXXX.XXXXX" format
            match = re.search(r"arXiv:?\s*(\d{4}\.\d{4,5})", text, re.IGNORECASE)
            if match:
                return match.group(1)

        # Strategy 2: Check all idno elements for various patterns
        for idno in bib_elem.xpath(".//tei:idno/text()", namespaces=self.ns):
            # CoRR format: "CoRR, abs/2004.10934" or just "abs/2004.10934"
            match = re.search(r"abs/(\d{4}\.\d{4,5})", idno)
            if match:
                return match.group(1)

            # URL format: http://arxiv.org/abs/2004.10934 or https://arxiv.org/pdf/2004.10934.pdf
            match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", idno, re.IGNORECASE)
            if match:
                return match.group(1)

        # Strategy 3: Check ptr elements (URLs)
        for ptr in bib_elem.xpath(".//tei:ptr/@target", namespaces=self.ns):
            match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", ptr, re.IGNORECASE)
            if match:
                return match.group(1)

        # Strategy 4: Check venue/title for CoRR or arXiv patterns
        if venue_list:
            venue_text = venue_list[0]
            # CoRR format in venue
            match = re.search(r"abs/(\d{4}\.\d{4,5})", venue_text)
            if match:
                return match.group(1)

            # "arXiv preprint" pattern - check surrounding text
            if "arxiv" in venue_text.lower():
                # Get all text from the bib element to search more broadly
                all_text = " ".join(bib_elem.xpath(".//text()"))
                match = re.search(r"arXiv\s+preprint\s+arXiv:?(\d{4}\.\d{4,5})", all_text, re.IGNORECASE)
                if match:
                    return match.group(1)

        return None

    def process_paper(self, pdf_path: str) -> dict[str, Citation]:
        """
        Process paper and extract citations with their referring sentences.

        Returns:
            Dictionary mapping citation IDs to Citation objects
        """

        with open(pdf_path, "rb") as pdf_file:
            response = requests.post(
                f"{self.grobid_url}/api/processFulltextDocument",
                files={"input": pdf_file},
            )
            response.raise_for_status()

        # Parse XML
        parser = etree.XMLParser(recover=True)
        root = etree.fromstring(response.content, parser)

        citations: dict[str, Citation] = {}
        # Track references temporarily in sets
        references_sets: dict[str, set[str]] = {}

        # Process bibliography entries
        for bib in root.xpath("//tei:listBibl/tei:biblStruct", namespaces=self.ns):
            citation_id = bib.get("{http://www.w3.org/XML/1998/namespace}id")
            if not citation_id:
                continue

            # Extract authors
            authors = []
            for author in bib.xpath(".//tei:author/tei:persName", namespaces=self.ns):
                forename = author.xpath("./tei:forename/text()", namespaces=self.ns)
                surname = author.xpath("./tei:surname/text()", namespaces=self.ns)
                name_parts = []
                if forename:
                    name_parts.extend(forename)
                if surname:
                    name_parts.extend(surname)
                if name_parts:
                    authors.append(" ".join(name_parts))

            # Extract other details
            title = bib.xpath('.//tei:title[@level="a"]/text()', namespaces=self.ns)
            if not title:  # Try other title types if article title not found
                title = bib.xpath(".//tei:title/text()", namespaces=self.ns)

            year = bib.xpath('.//tei:date[@type="published"]/@when', namespaces=self.ns)
            venue = bib.xpath(".//tei:monogr/tei:title/text()", namespaces=self.ns)

            # Extract arXiv ID using multiple strategies
            arxiv_id = self._extract_arxiv_id(bib, venue)

            # Create Citation object
            details = CitationDetails(
                authors=authors,
                title=title[0] if title else None,
                year=year[0] if year else None,
                venue=venue[0] if venue else None,
                arxiv_id=arxiv_id,
            )

            citations[citation_id] = Citation(
                citation_id=citation_id,
                details=details,
                references=[],  # Will be populated below
            )
            references_sets[citation_id] = set()

        # Find citation references in the text
        for ref in root.xpath('//tei:ref[@type="bibr"]', namespaces=self.ns):
            target = ref.get("target", "").lstrip("#")
            if target in citations:
                sentence = self._get_sentence_context(ref)
                if sentence:
                    references_sets[target].add(sentence)

        # Convert sets to lists in Citation objects
        for citation_id, citation in citations.items():
            citation.references = list(references_sets[citation_id])

        return citations
