"""
TEI XML parser for GROBID output.

Parses TEI XML files from GROBID's processFulltextDocument endpoint
and extracts structured content for display.
"""

import gzip
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# TEI XML namespace
TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


@dataclass
class BibEntry:
    """A bibliography entry from the TEI document."""

    id: str
    title: str
    authors: list[str]
    year: str
    arxiv_id: str | None = None


@dataclass
class TocEntry:
    """A table of contents entry."""

    id: str
    num: str
    title: str


@dataclass
class ParsedTeiDocument:
    """Parsed content from a TEI XML document."""

    title: str
    authors: list[str]
    date: str
    abstract_html: str
    body_html: str
    ack_html: str
    references_html: str
    bibliography: dict[str, BibEntry]
    toc: list[TocEntry]


def escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def get_element_text(el: ET.Element) -> str:
    """Get all text content from an element, including nested elements."""
    return "".join(el.itertext())


def parse_tei_xml(
    xml_path: Path,
    paper_url_builder: Callable[[str], str] | None = None,
) -> ParsedTeiDocument:
    """
    Parse a GROBID TEI XML file and extract structured content.

    Args:
        xml_path: Path to the .xml.gz file
        paper_url_builder: Optional function to build URLs for paper links.
                          Takes arxiv_id, returns URL string.
                          If None, no links are generated for papers.

    Returns:
        ParsedTeiDocument with extracted content
    """
    with gzip.open(xml_path, "rt", encoding="utf-8") as f:
        tree = ET.parse(f)
    root = tree.getroot()

    # Extract metadata from header
    title = ""
    authors: list[str] = []
    date = ""

    # Title
    title_el = root.find(".//tei:titleStmt/tei:title", TEI_NS)
    if title_el is not None and title_el.text:
        title = title_el.text.strip()

    # Authors
    for author_el in root.findall(".//tei:sourceDesc//tei:author", TEI_NS):
        persname = author_el.find("tei:persName", TEI_NS)
        if persname is not None:
            forename = persname.find("tei:forename", TEI_NS)
            surname = persname.find("tei:surname", TEI_NS)
            name_parts = []
            if forename is not None and forename.text:
                name_parts.append(forename.text)
            if surname is not None and surname.text:
                name_parts.append(surname.text)
            if name_parts:
                authors.append(" ".join(name_parts))

    # Date
    date_el = root.find(".//tei:publicationStmt/tei:date[@type='published']", TEI_NS)
    if date_el is not None and date_el.text:
        date = date_el.text.strip()

    # Extract bibliography for citation popups
    bibliography: dict[str, BibEntry] = {}
    for bibl in root.findall(".//tei:listBibl/tei:biblStruct", TEI_NS):
        bib_id = bibl.get("{http://www.w3.org/XML/1998/namespace}id", "")
        if bib_id:
            bib_title_el = bibl.find(".//tei:title", TEI_NS)
            bib_title = bib_title_el.text.strip() if bib_title_el is not None and bib_title_el.text else "Unknown"

            bib_authors: list[str] = []
            for author_el in bibl.findall(".//tei:author/tei:persName", TEI_NS):
                forename = author_el.find("tei:forename", TEI_NS)
                surname = author_el.find("tei:surname", TEI_NS)
                name_parts = []
                if forename is not None and forename.text:
                    name_parts.append(forename.text)
                if surname is not None and surname.text:
                    name_parts.append(surname.text)
                if name_parts:
                    bib_authors.append(" ".join(name_parts))

            bib_date_el = bibl.find(".//tei:date", TEI_NS)
            bib_year = ""
            if bib_date_el is not None:
                bib_year = bib_date_el.get("when", "")[:4] if bib_date_el.get("when") else (bib_date_el.text or "")[:4]

            # Check for arXiv ID
            arxiv_id = None
            for idno in bibl.findall(".//tei:idno", TEI_NS):
                if idno.get("type") == "arXiv" and idno.text:
                    # Extract just the ID from "arXiv:XXXX.XXXXX..."
                    match = re.search(r"(\d{4}\.\d{4,5})", idno.text)
                    if match:
                        arxiv_id = match.group(1)
                    break

            bibliography[bib_id] = BibEntry(
                id=bib_id,
                title=bib_title,
                authors=bib_authors,
                year=bib_year,
                arxiv_id=arxiv_id,
            )

    # Convert bibliography to dict format for HTML builders
    bib_dict = {k: {"title": v.title, "authors": v.authors, "year": v.year, "arxiv_id": v.arxiv_id} for k, v in bibliography.items()}

    # Abstract
    abstract_html = ""
    abstract_div = root.find(".//tei:profileDesc/tei:abstract/tei:div", TEI_NS)
    if abstract_div is not None:
        abstract_html = _tei_div_to_html(abstract_div, bib_dict, paper_url_builder)

    # Extract body content
    body_html = ""
    body_el = root.find(".//tei:body", TEI_NS)
    if body_el is not None:
        body_html = _tei_body_to_html(body_el, bib_dict, paper_url_builder)

    # Extract acknowledgements
    ack_html = ""
    ack_el = root.find(".//tei:back/tei:div[@type='acknowledgement']", TEI_NS)
    if ack_el is not None:
        ack_html = _tei_div_to_html(ack_el, bib_dict, paper_url_builder)

    # Build references section
    references_html = _build_references_html(bib_dict, paper_url_builder)

    # Build table of contents from sections
    toc: list[TocEntry] = []
    for div in body_el.findall("tei:div", TEI_NS) if body_el is not None else []:
        head = div.find("tei:head", TEI_NS)
        if head is not None:
            section_num = head.get("n", "")
            section_title = head.text or ""
            section_id = f"section-{section_num}" if section_num else f"section-{len(toc)}"
            toc.append(TocEntry(id=section_id, num=section_num, title=section_title))

    return ParsedTeiDocument(
        title=title,
        authors=authors,
        date=date,
        abstract_html=abstract_html,
        body_html=body_html,
        ack_html=ack_html,
        references_html=references_html,
        bibliography=bibliography,
        toc=toc,
    )


def _tei_body_to_html(
    body_el: ET.Element,
    bibliography: dict,
    paper_url_builder: Callable[[str], str] | None,
) -> str:
    """Convert TEI body element to HTML."""
    html_parts = []
    for div in body_el.findall("tei:div", TEI_NS):
        html_parts.append(_tei_div_to_html(div, bibliography, paper_url_builder))
    return "\n".join(html_parts)


def _tei_div_to_html(
    div_el: ET.Element,
    bibliography: dict,
    paper_url_builder: Callable[[str], str] | None,
    depth: int = 2,
) -> str:
    """Convert a TEI div element to HTML."""
    html_parts = []

    # Section heading
    head = div_el.find("tei:head", TEI_NS)
    if head is not None:
        section_num = head.get("n", "")
        section_title = head.text or ""
        section_id = f"section-{section_num}" if section_num else f"section-{id(div_el)}"
        heading_tag = f"h{min(depth, 6)}"
        num_span = f'<span class="section-num">{section_num}</span> ' if section_num else ""
        html_parts.append(f'<{heading_tag} id="{section_id}">{num_span}{escape_html(section_title)}</{heading_tag}>')

    # Process child elements
    for child in div_el:
        tag = child.tag.replace(f"{{{TEI_NS['tei']}}}", "")

        if tag == "head":
            continue  # Already processed
        elif tag == "p":
            html_parts.append(_tei_p_to_html(child, bibliography, paper_url_builder))
        elif tag == "div":
            html_parts.append(_tei_div_to_html(child, bibliography, paper_url_builder, depth + 1))
        elif tag == "figure":
            html_parts.append(_tei_figure_to_html(child))
        elif tag == "formula":
            html_parts.append(_tei_formula_to_html(child))

    return "\n".join(html_parts)


def _tei_p_to_html(
    p_el: ET.Element,
    bibliography: dict,
    paper_url_builder: Callable[[str], str] | None,
) -> str:
    """Convert a TEI paragraph to HTML with inline citations."""
    html_parts: list[str] = []

    def process_element(el: ET.Element, include_text: bool = True):
        """Recursively process an element and its children."""
        if include_text and el.text:
            html_parts.append(escape_html(el.text))

        for child in el:
            child_tag = child.tag.replace(f"{{{TEI_NS['tei']}}}", "")

            if child_tag == "ref":
                ref_type = child.get("type", "")
                target = child.get("target", "").lstrip("#")
                ref_text = get_element_text(child)

                if ref_type == "bibr" and target in bibliography:
                    bib = bibliography[target]
                    # Create citation with popup
                    authors_str = ", ".join(bib["authors"][:2])
                    if len(bib["authors"]) > 2:
                        authors_str += " et al."
                    popup_content = f"{bib['title']}"
                    if authors_str:
                        popup_content += f" — {authors_str}"
                    if bib["year"]:
                        popup_content += f" ({bib['year']})"

                    # Link to paper if we have an arXiv ID and a URL builder
                    if bib.get("arxiv_id") and paper_url_builder:
                        url = paper_url_builder(bib["arxiv_id"])
                        html_parts.append(
                            f'<a href="{url}" class="citation" '
                            f'data-ref-id="{target}" data-popup="{escape_html(popup_content)}">'
                            f"{escape_html(ref_text)}</a>"
                        )
                    else:
                        html_parts.append(
                            f'<span class="citation" data-ref-id="{target}" '
                            f'data-popup="{escape_html(popup_content)}">{escape_html(ref_text)}</span>'
                        )
                elif ref_type == "figure" or ref_type == "table":
                    html_parts.append(f'<span class="figure-ref">{escape_html(ref_text)}</span>')
                else:
                    html_parts.append(escape_html(ref_text))
            elif child_tag == "formula":
                html_parts.append(_tei_formula_to_html(child, inline=True))
            else:
                # Recursively process other elements
                process_element(child, include_text=True)

            if child.tail:
                html_parts.append(escape_html(child.tail))

    process_element(p_el)
    return f'<p>{"".join(html_parts)}</p>'


def _tei_figure_to_html(fig_el: ET.Element) -> str:
    """Convert a TEI figure element to HTML."""
    label_el = fig_el.find("tei:label", TEI_NS)
    desc_el = fig_el.find("tei:figDesc", TEI_NS)
    head_el = fig_el.find("tei:head", TEI_NS)

    label = label_el.text if label_el is not None and label_el.text else ""
    desc = desc_el.text if desc_el is not None and desc_el.text else ""
    head = head_el.text if head_el is not None and head_el.text else ""

    # Check if it's a table
    table_el = fig_el.find("tei:table", TEI_NS)
    if table_el is not None:
        table_html = _tei_table_to_html(table_el)
        caption = f"<strong>{label}</strong>" if label else ""
        if head:
            caption += f": {escape_html(head)}"
        return f"""<figure class="table-figure">
            {table_html}
            <figcaption>{caption}</figcaption>
        </figure>"""

    # Regular figure (no image available from GROBID)
    caption_parts = []
    if label:
        caption_parts.append(f"<strong>{escape_html(label)}</strong>")
    if head:
        caption_parts.append(escape_html(head))

    return f"""<figure class="paper-figure">
        <div class="figure-placeholder">[Figure]</div>
        <figcaption>{": ".join(caption_parts)}</figcaption>
        {f'<p class="figure-desc">{escape_html(desc)}</p>' if desc else ''}
    </figure>"""


def _tei_table_to_html(table_el: ET.Element) -> str:
    """Convert a TEI table element to HTML."""
    rows_html = []
    for row in table_el.findall("tei:row", TEI_NS):
        cells_html = []
        for cell in row.findall("tei:cell", TEI_NS):
            cell_text = get_element_text(cell)
            cells_html.append(f"<td>{escape_html(cell_text)}</td>")
        rows_html.append(f"<tr>{''.join(cells_html)}</tr>")

    return f'<table class="paper-table">{"".join(rows_html)}</table>'


def _tei_formula_to_html(formula_el: ET.Element, inline: bool = False) -> str:
    """Convert a TEI formula element to HTML (for MathJax rendering)."""
    formula_text = get_element_text(formula_el)
    formula_text = formula_text.strip()

    if inline:
        return f'<span class="math-inline">\\({formula_text}\\)</span>'
    else:
        return f'<div class="math-block">\\[{formula_text}\\]</div>'


def _build_references_html(
    bibliography: dict,
    paper_url_builder: Callable[[str], str] | None,
) -> str:
    """Build HTML for the references section."""
    if not bibliography:
        return ""

    html_parts = ['<ol class="references-list">']
    for bib_id, bib in bibliography.items():
        authors_str = ", ".join(bib["authors"])
        year_str = f" ({bib['year']})" if bib["year"] else ""

        if bib.get("arxiv_id") and paper_url_builder:
            url = paper_url_builder(bib["arxiv_id"])
            title_html = f'<a href="{url}">{escape_html(bib["title"])}</a>'
        else:
            title_html = escape_html(bib["title"])

        html_parts.append(
            f'<li id="ref-{bib_id}">'
            f'<span class="ref-authors">{escape_html(authors_str)}</span>{year_str}. '
            f'<span class="ref-title">{title_html}</span>'
            f"</li>"
        )
    html_parts.append("</ol>")

    return "\n".join(html_parts)

