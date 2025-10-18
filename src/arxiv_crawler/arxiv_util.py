from typing import Any
import requests
import feedparser
import tempfile
from lxml import etree


arxiv_url_pattern = r"(?:https?://)?(?:www\.)?arxiv\.org/abs/(?:\d{4}\.\d{4,}|\d{7})"
arxiv_id_pattern = r"arXiv.*?(\d{4}\.\d{4,}|\d{7})"


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

    def process_paper(self, pdf_path: str) -> dict[str, dict]:
        """
        Process paper and extract citations with their referring sentences.

        Returns:
            dictionary mapping citation IDs to citation details and references
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

        citations = {}

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

            arxiv_id = bib.xpath('.//tei:idno[@type="arXiv"]/text()', namespaces=self.ns)
            if arxiv_id and arxiv_id[0].startswith("arXiv:"):
                arxiv_id = arxiv_id[0].replace("arXiv:", "")
            else:
                arxiv_id = None

            citations[citation_id] = {
                "details": {
                    "authors": authors,
                    "title": title[0] if title else None,
                    "year": year[0] if year else None,
                    "venue": venue[0] if venue else None,
                    "arxiv_id": arxiv_id,
                },
                "references": set(),
            }

            # Find citation references in the text
            for ref in root.xpath('//tei:ref[@type="bibr"]', namespaces=self.ns):
                target = ref.get("target", "").lstrip("#")
                if target in citations:
                    sentence = self._get_sentence_context(ref)
                    if sentence:
                        citations[target]["references"].add(sentence)

        for target in citations:
            citations[target]["references"] = list(citations[target]["references"])
        return citations
