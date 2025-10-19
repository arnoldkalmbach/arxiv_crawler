from pathlib import Path
import re
import requests
import json

import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import convert_to_unicode

from arxiv_crawler.arxiv_util import arxiv_url_pattern, arxiv_id_pattern


def parse_bibtex_from_markdown(content: str) -> list[dict[str, str]]:
    """
    Extract and parse BibTeX entries from markdown latex code blocks.
    Returns a list of dictionaries containing the parsed fields.
    """
    parser = BibTexParser()
    parser.customization = convert_to_unicode
    entries = []

    # Pattern to match code blocks with latex language identifier
    # Handles both ``` and ````
    latex_block_pattern = r"```+\s*latex\n(.*?)\n```+"

    # Find all latex code blocks
    latex_blocks = re.finditer(latex_block_pattern, content, re.DOTALL)

    for block in latex_blocks:
        block_content = block.group(1).strip()
        try:
            bib_database = bibtexparser.loads(block_content, parser)
            if bib_database.entries:
                entries.extend(bib_database.entries)
        except Exception as e:
            print(f"Failed to parse latex block: {str(e)}")
            print(f"Block content:\n{block_content}")

    return entries


def search_markdown_files(directory: str | Path) -> list[dict[str, str]]:
    """
    Recursively search directory for .md files and parse all BibTeX entries.
    Returns a list of all parsed entries across all files.

    Args:
        directory: Path-like object pointing to the search directory
    """
    directory = Path(directory)
    all_entries = []

    # Recursively find all .md files
    for md_file in directory.rglob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8")
            entries = parse_bibtex_from_markdown(content)
            if entries:
                # Add file path to each entry for reference
                for entry in entries:
                    entry["model_name"] = str(md_file.parent.name)
                all_entries.extend(entries)
        except Exception as e:
            print(f"Error processing {md_file}: {str(e)}")

    return all_entries


if __name__ == "__main__":
    mmdet_citations = search_markdown_files("/home/arnold/Projects/mmdetection")
    mmyolo_citations = search_markdown_files("/home/arnold/Projects/mmyolo")

    arxiv_ids = []
    for citation in mmdet_citations + mmyolo_citations:
        try:
            journal = citation["journal"]
            match = re.findall(arxiv_id_pattern, journal)
            if match:
                arxiv_ids.append(match[0])
        except KeyError:
            continue

    readme_url = "https://raw.githubusercontent.com/huggingface/pytorch-image-models/main/README.md"
    readme_content = requests.get(readme_url).text
    raw_urls = re.findall(arxiv_url_pattern, readme_content)

    readme_url = "https://raw.githubusercontent.com/mlfoundations/open_clip/63fbfd857c83af619e6a0a8344635ebb4a151a96/docs/PRETRAINED.md"
    readme_content = requests.get(readme_url).text
    raw_urls += re.findall(arxiv_url_pattern, readme_content)

    arxiv_ids.extend([url.split("/")[-1] for url in raw_urls])

    with open("data/initial_arxiv_ids.json", "w") as f:
        json.dump(arxiv_ids, f)
