#!/usr/bin/env python3
"""
Generate golden fixtures for CitationExtractor tests.

Run this script when you add a new sample_paper.pdf or want to update the expected output:
    uv run python tests/generate_fixtures.py

This will create/update the golden fixture files in tests/fixtures/
"""

import json
import re
import sys
from pathlib import Path
from difflib import SequenceMatcher

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arxiv_crawler.arxiv_util import CitationExtractor, get_arxiv_metadata


def similarity(a: str, b: str) -> float:
    """Calculate string similarity (0-1)."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def validate_arxiv_ids(citations: dict) -> dict:
    """
    Validate extracted arXiv IDs by querying the arXiv API.

    Returns:
        Dictionary with validation statistics and details
    """
    # Collect all arXiv IDs
    arxiv_ids = []
    id_to_citation = {}

    for cid, citation in citations.items():
        arxiv_id = citation.details.arxiv_id
        if arxiv_id:
            arxiv_ids.append(arxiv_id)
            id_to_citation[arxiv_id] = (cid, citation)

    if not arxiv_ids:
        return {"total_extracted": 0, "validated": 0, "valid_ids": [], "invalid_ids": [], "metadata_matches": []}

    print(f"\n🔍 Validating {len(arxiv_ids)} arXiv IDs via API...")

    try:
        # Query arXiv API
        api_results = get_arxiv_metadata(arxiv_ids)

        # Create lookup by ID (extract from URL)
        api_by_id = {}
        for result in api_results:
            # Extract ID from arxiv_url (e.g., http://arxiv.org/abs/1902.05509v1 -> 1902.05509)
            arxiv_url = result.get("arxiv_url", "")
            if "arxiv.org/abs/" in arxiv_url:
                id_with_version = arxiv_url.split("/abs/")[-1]
                # Remove version suffix (v1, v2, etc.)
                id_part = re.sub(r"v\d+$", "", id_with_version)
                api_by_id[id_part] = result

        valid_ids = []
        invalid_ids = []
        metadata_matches = []

        for arxiv_id in arxiv_ids:
            cid, citation = id_to_citation[arxiv_id]
            extracted = citation.details

            if arxiv_id in api_by_id:
                valid_ids.append(arxiv_id)
                api_data = api_by_id[arxiv_id]

                # Compare metadata
                title_sim = similarity(extracted.title or "", api_data.get("title", ""))

                # Check author overlap
                extracted_authors = set(extracted.authors)
                api_authors = set(api_data.get("authors", []))
                author_overlap = len(extracted_authors & api_authors) if extracted_authors and api_authors else 0

                # Extract year from published date (YYYY-MM-DD)
                api_year = api_data.get("published", "")[:4] if api_data.get("published") else None
                extracted_year = extracted.year[:4] if extracted.year else None
                year_match = api_year == extracted_year if api_year and extracted_year else None

                metadata_matches.append(
                    {
                        "arxiv_id": arxiv_id,
                        "citation_id": cid,
                        "title_similarity": round(title_sim, 2),
                        "author_overlap": author_overlap,
                        "year_match": year_match,
                        "extracted_title": extracted.title or "",
                        "extracted_authors": extracted.authors,
                        "extracted_year": extracted.year or "",
                        "extracted_venue": extracted.venue or "",
                        "api_title": api_data.get("title", ""),
                        "api_authors": api_data.get("authors", []),
                        "api_year": api_year,
                        "api_url": api_data.get("arxiv_url", ""),
                    }
                )
            else:
                invalid_ids.append(arxiv_id)

        return {
            "total_extracted": len(arxiv_ids),
            "validated": len(valid_ids),
            "valid_ids": valid_ids,
            "invalid_ids": invalid_ids,
            "metadata_matches": metadata_matches,
        }

    except Exception as e:
        print(f"⚠️  Error validating arXiv IDs: {e}")
        return {
            "total_extracted": len(arxiv_ids),
            "validated": 0,
            "valid_ids": [],
            "invalid_ids": [],
            "metadata_matches": [],
            "error": str(e),
        }


def generate_validation_report(validation_results: dict) -> str:
    """Generate a comprehensive validation report with all details as a string."""
    if not validation_results["total_extracted"]:
        return "\n📋 No arXiv IDs to validate\n"

    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("📋 ARXIV ID VALIDATION REPORT")
    lines.append("=" * 80)

    total = validation_results["total_extracted"]
    valid = validation_results["validated"]
    invalid = len(validation_results["invalid_ids"])

    # Summary statistics
    lines.append("\n📊 SUMMARY")
    lines.append("-" * 80)
    lines.append(f"✅ Valid IDs:   {valid}/{total} ({valid / total * 100:.1f}%)")
    if invalid > 0:
        lines.append(f"❌ Invalid IDs: {invalid}")

    # Metadata quality overview
    matches = validation_results["metadata_matches"]
    if matches:
        high_title_sim = sum(1 for m in matches if m["title_similarity"] >= 0.8)
        has_author_overlap = sum(1 for m in matches if m["author_overlap"] > 0)
        year_matches = [m for m in matches if m["year_match"] is not None]
        year_correct = sum(1 for m in year_matches if m["year_match"]) if year_matches else 0

        lines.append("\n📈 Metadata Quality:")
        lines.append(
            f"   • Title similarity ≥80%: {high_title_sim}/{len(matches)} ({high_title_sim / len(matches) * 100:.1f}%)"
        )
        lines.append(
            f"   • Author overlap:        {has_author_overlap}/{len(matches)} ({has_author_overlap / len(matches) * 100:.1f}%)"
        )
        if year_matches:
            lines.append(
                f"   • Year match:            {year_correct}/{len(year_matches)} ({year_correct / len(year_matches) * 100:.1f}%)"
            )

    # Show all invalid IDs first
    if invalid > 0:
        lines.append("\n" + "=" * 80)
        lines.append("❌ INVALID ARXIV IDs")
        lines.append("=" * 80)
        for invalid_id in validation_results["invalid_ids"]:
            lines.append(f"   • {invalid_id}")

    # Detailed comparison for all valid IDs
    if matches:
        lines.append("\n" + "=" * 80)
        lines.append("✅ DETAILED VALIDATION RESULTS (All Valid IDs)")
        lines.append("=" * 80)

        for i, match in enumerate(matches, 1):
            lines.append(f"\n{i}. Citation {match['citation_id']}: arXiv:{match['arxiv_id']}")
            lines.append("-" * 80)

            # Quality indicators
            quality_marks = []
            if match["title_similarity"] >= 0.9:
                quality_marks.append("📗 Excellent title match")
            elif match["title_similarity"] >= 0.8:
                quality_marks.append("📙 Good title match")
            else:
                quality_marks.append("📕 Weak title match")

            if match["author_overlap"] >= 3:
                quality_marks.append("👥 Strong author match")
            elif match["author_overlap"] > 0:
                quality_marks.append("👤 Partial author match")
            else:
                quality_marks.append("❓ No author match")

            if match["year_match"] is True:
                quality_marks.append("📅 Year matches")
            elif match["year_match"] is False:
                quality_marks.append("⚠️  Year mismatch")

            lines.append(f"   Quality: {' | '.join(quality_marks)}")
            lines.append(
                f"   Scores:  Title={match['title_similarity']:.2f} | Authors={match['author_overlap']} | Year={match['year_match']}"
            )

            lines.append("\n   📄 Title:")
            lines.append(f"      Extracted: {match['extracted_title']}")
            lines.append(f"      API:       {match['api_title']}")

            lines.append("\n   👥 Authors:")
            lines.append(
                f"      Extracted ({len(match['extracted_authors'])}): {', '.join(match['extracted_authors']) if match['extracted_authors'] else 'None'}"
            )
            lines.append(
                f"      API ({len(match['api_authors'])}):       {', '.join(match['api_authors']) if match['api_authors'] else 'None'}"
            )

            lines.append("\n   📅 Year & Venue:")
            lines.append(f"      Extracted: {match['extracted_year']} | Venue: {match['extracted_venue']}")
            lines.append(f"      API:       {match['api_year']}")

            lines.append(f"\n   🔗 arXiv URL: {match['api_url']}")

    lines.append("\n" + "=" * 80 + "\n")

    return "\n".join(lines)


def generate_citation_fixture(pdf_path: Path, output_path: Path, grobid_url: str = "http://localhost:8070"):
    """
    Generate a golden fixture from a PDF file.

    Args:
        pdf_path: Path to the PDF file to process
        output_path: Path where the fixture JSON will be saved
        grobid_url: URL of the Grobid server
    """
    print(f"Processing {pdf_path.name}...")

    try:
        extractor = CitationExtractor(grobid_url=grobid_url)
        citations = extractor.process_paper(str(pdf_path))
    except Exception as e:
        print(f"Error processing PDF: {e}")
        print("Make sure Grobid server is running at http://localhost:8070")
        sys.exit(1)

    # Print statistics
    print(f"\nExtracted {len(citations)} citations")
    cited_count = sum(1 for c in citations.values() if len(c.references) > 0)
    print(f"  - {cited_count} cited in text")
    print(f"  - {len(citations) - cited_count} only in bibliography")

    arxiv_count = sum(1 for c in citations.values() if c.details.arxiv_id is not None)
    print(f"  - {arxiv_count} have arXiv IDs")

    # Validate arXiv IDs
    validation_results = validate_arxiv_ids(citations)
    validation_report = generate_validation_report(validation_results)

    # Print the report to console
    print(validation_report)

    # Save fixture JSON (convert Citation objects to dicts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        # Convert Citation objects to dict format for JSON serialization
        citations_dict = {}
        for cit_id, citation in citations.items():
            citations_dict[cit_id] = {"details": citation.details.model_dump(), "references": citation.references}
        json.dump(citations_dict, f, indent=2, ensure_ascii=False)

    # Save validation report to file
    validation_report_path = output_path.with_suffix(".validation.txt")
    with open(validation_report_path, "w", encoding="utf-8") as f:
        f.write(validation_report)

    print(f"Golden fixture saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"\nValidation report saved to: {validation_report_path}")
    print(f"File size: {validation_report_path.stat().st_size / 1024:.1f} KB")


def main():
    """Generate all fixtures."""
    fixtures_dir = Path(__file__).parent / "fixtures"

    # Check for sample_paper.pdf
    sample_pdf = fixtures_dir / "sample_paper.pdf"
    if not sample_pdf.exists():
        print(f"Error: {sample_pdf} not found")
        print("Please add a sample PDF before generating fixtures")
        sys.exit(1)

    # Generate fixture for sample_paper.pdf
    output_json = fixtures_dir / "sample_paper_expected.json"
    generate_citation_fixture(sample_pdf, output_json)


if __name__ == "__main__":
    main()
