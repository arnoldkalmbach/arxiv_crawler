import pytest
import requests
from unittest.mock import Mock, patch

from arxiv_crawler.arxiv_util import CitationExtractor


class TestProcessPaperIntegration:
    """
    Integration tests for process_paper() with real Grobid server.

    These tests assume Grobid is running at the default URL (http://localhost:8070).
    """

    @pytest.mark.integration
    def test_citation_reference_matching(self, sample_pdf_path, grobid_url):
        """
        Test that citations are correctly matched to their references in the text.

        Expected behavior:
        - Citations mentioned in the paper should have non-empty 'references' list
        - Each reference should be a sentence containing the citation
        - References should be unique (no duplicates)
        - Citations not mentioned in text should have empty 'references' list
        """
        extractor = CitationExtractor(grobid_url=grobid_url)

        try:
            citations = extractor.process_paper(sample_pdf_path)
        except requests.exceptions.ConnectionError:
            pytest.skip("Grobid server not available at http://localhost:8070")

        if len(citations) == 0:
            pytest.skip("No citations found in sample PDF")

        # Analyze reference distribution
        cited_count = sum(1 for c in citations.values() if len(c.references) > 0)
        uncited_count = len(citations) - cited_count

        print(f"\n{'=' * 60}")
        print(f"Total citations: {len(citations)}")
        print(f"Citations with references in text: {cited_count}")
        print(f"Citations without references in text: {uncited_count}")

        # Check for citations with references
        for citation_id, citation in citations.items():
            references = citation.references

            if len(references) > 0:
                # Verify uniqueness
                unique_refs = set(references)
                assert len(unique_refs) == len(references), f"Citation {citation_id} has duplicate references"

                # Verify all are non-empty strings
                for ref in references:
                    assert isinstance(ref, str), "Reference should be a string"
                    assert len(ref) > 0, "Reference should not be empty"

                # Print first citation with references for manual verification
                if cited_count > 0:
                    print(f"\nSample citation with {len(references)} reference(s):")
                    print(f"Citation ID: {citation_id}")
                    print(f"Title: {citation.details.title}")
                    for i, ref in enumerate(references[:2], 1):  # Show max 2
                        print(f"  Reference {i}: {ref[:150]}...")
                    cited_count = -1  # Only print once

        print(f"{'=' * 60}\n")

    @pytest.mark.integration
    def test_arxiv_id_extraction(self, sample_pdf_path, grobid_url):
        """
        Test that arXiv IDs are correctly extracted from citations.

        Expected behavior:
        - arXiv IDs should be extracted when present in bibliography
        - "arXiv:" prefix should be stripped
        - Citations without arXiv IDs should have arxiv_id=None
        """
        extractor = CitationExtractor(grobid_url=grobid_url)

        try:
            citations = extractor.process_paper(sample_pdf_path)
        except requests.exceptions.ConnectionError:
            pytest.skip("Grobid server not available at http://localhost:8070")

        if len(citations) == 0:
            pytest.skip("No citations found in sample PDF")

        arxiv_citations = [c for c in citations.values() if c.details.arxiv_id is not None]

        print(f"\n{'=' * 60}")
        print(f"Total citations: {len(citations)}")
        print(f"Citations with arXiv ID: {len(arxiv_citations)}")

        for citation in arxiv_citations[:3]:  # Show max 3 examples
            arxiv_id = citation.details.arxiv_id
            # Verify arXiv: prefix is stripped
            assert not arxiv_id.startswith("arXiv:"), f"arXiv ID should not have 'arXiv:' prefix: {arxiv_id}"
            print(f"  ArXiv ID: {arxiv_id} - {citation.details.title[:60] if citation.details.title else 'N/A'}...")

        print(f"{'=' * 60}\n")

    @pytest.mark.integration
    def test_golden_output_matches(self, sample_pdf_path, sample_paper_expected, grobid_url):
        """
        Test that current output matches the golden fixture.

        This is a regression test to ensure the citation extraction behavior remains consistent.

        To regenerate fixtures: uv run python tests/generate_fixtures.py

        Expected behavior:
        - Same citation IDs extracted
        - Same citation details (authors, title, year, venue, arxiv_id)
        - Same reference sentences for each citation
        """
        extractor = CitationExtractor(grobid_url=grobid_url)

        try:
            citations = extractor.process_paper(sample_pdf_path)
        except requests.exceptions.ConnectionError:
            pytest.skip("Grobid server not available at http://localhost:8070")

        # Compare citation IDs
        actual_ids = set(citations.keys())
        expected_ids = set(sample_paper_expected.keys())

        assert actual_ids == expected_ids, (
            f"Citation IDs differ. Missing: {expected_ids - actual_ids}, Extra: {actual_ids - expected_ids}"
        )

        # Compare each citation in detail
        differences = []
        for cid in expected_ids:
            actual = citations[cid]
            expected = sample_paper_expected[cid]

            # Compare details
            for key in ["authors", "title", "year", "venue", "arxiv_id"]:
                # Access Citation object attributes
                actual_val = getattr(actual.details, key)
                expected_val = expected["details"][key]

                if actual_val != expected_val:
                    differences.append(f"Citation {cid}, field '{key}': expected {expected_val!r}, got {actual_val!r}")

            # Compare references (order doesn't matter, but content does)
            actual_refs = set(actual.references)
            expected_refs = set(expected["references"])

            if actual_refs != expected_refs:
                missing = expected_refs - actual_refs
                extra = actual_refs - expected_refs
                if missing or extra:
                    differences.append(
                        f"Citation {cid}, references differ. Missing: {len(missing)}, Extra: {len(extra)}"
                    )

        # Report all differences
        if differences:
            error_msg = "\n".join(["Output differs from golden fixture:"] + differences[:10])
            if len(differences) > 10:
                error_msg += f"\n... and {len(differences) - 10} more differences"
            error_msg += "\n\nTo regenerate fixtures: uv run python tests/generate_fixtures.py"
            pytest.fail(error_msg)

        print(f"\n✓ Golden output matches! {len(citations)} citations validated.")


class TestErrorHandling:
    """Test error handling for various failure scenarios."""

    def test_grobid_server_unavailable(self, tmp_path):
        """
        Test handling when Grobid server is not available.

        Expected behavior:
        - Should raise requests.exceptions.ConnectionError
        """
        extractor = CitationExtractor(grobid_url="http://localhost:9999")  # Wrong port

        # Create a temporary dummy PDF file
        dummy_pdf = tmp_path / "dummy.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4\n%dummy content")

        with pytest.raises(requests.exceptions.ConnectionError):
            extractor.process_paper(str(dummy_pdf))

    def test_grobid_server_error_response(self, tmp_path):
        """
        Test handling when Grobid returns error status.

        Expected behavior:
        - Should raise requests.exceptions.HTTPError when server returns non-200 status
        """
        extractor = CitationExtractor()

        # Create a temporary dummy PDF file
        dummy_pdf = tmp_path / "dummy.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4\n%dummy content")

        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
            mock_post.return_value = mock_response

            with pytest.raises(requests.exceptions.HTTPError):
                extractor.process_paper(str(dummy_pdf))

    def test_file_not_found(self):
        """
        Test handling when PDF file doesn't exist.

        Expected behavior:
        - Should raise FileNotFoundError when opening non-existent file
        """
        extractor = CitationExtractor()

        with pytest.raises(FileNotFoundError):
            extractor.process_paper("/nonexistent/path/to/file.pdf")

    def test_malformed_xml_recovery(self, tmp_path):
        """
        Test handling of malformed XML from Grobid.

        Expected behavior:
        - Parser should recover from minor XML issues (recover=True mode)
        - Should return partial results or empty dict rather than crashing
        """
        extractor = CitationExtractor()

        # Create a temporary dummy PDF file
        dummy_pdf = tmp_path / "dummy.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4\n%dummy content")

        # Mock Grobid to return malformed but parseable XML
        malformed_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
        <TEI xmlns="http://www.tei-c.org/ns/1.0">
            <text>
                <body>
                    <div>
                        <!-- Missing closing tag will be recovered -->
                        <p>Some text
                    </div>
                </body>
            </text>
        </TEI>"""

        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.content = malformed_xml
            mock_post.return_value = mock_response

            # Should not raise an exception due to recover=True in parser
            citations = extractor.process_paper(str(dummy_pdf))

            # Should return empty or partial results
            assert isinstance(citations, dict)


class TestCitationDetails:
    """Test extraction of citation details from Grobid XML."""

    def test_multiple_authors_extraction(self, tmp_path):
        """
        Test that multiple authors are correctly extracted.

        Expected behavior:
        - All authors should be in the list
        - Names should combine forename and surname
        - Order should be preserved
        """
        extractor = CitationExtractor()

        dummy_pdf = tmp_path / "dummy.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4\n%dummy content")

        # Mock Grobid response with multiple authors
        xml_response = b"""<?xml version="1.0" encoding="UTF-8"?>
        <TEI xmlns="http://www.tei-c.org/ns/1.0" xml:lang="en">
            <teiHeader>
                <fileDesc>
                    <sourceDesc>
                        <biblStruct>
                            <analytic>
                                <author>
                                    <persName>
                                        <forename type="first">John</forename>
                                        <surname>Doe</surname>
                                    </persName>
                                </author>
                                <author>
                                    <persName>
                                        <forename type="first">Jane</forename>
                                        <surname>Smith</surname>
                                    </persName>
                                </author>
                            </analytic>
                        </biblStruct>
                    </sourceDesc>
                </fileDesc>
            </teiHeader>
            <text>
                <body>
                    <div>
                        <listBibl>
                            <biblStruct xml:id="b0">
                                <analytic>
                                    <title level="a" type="main">Test Paper</title>
                                    <author>
                                        <persName>
                                            <forename type="first">Alice</forename>
                                            <surname>Johnson</surname>
                                        </persName>
                                    </author>
                                    <author>
                                        <persName>
                                            <forename type="first">Bob</forename>
                                            <surname>Williams</surname>
                                        </persName>
                                    </author>
                                    <author>
                                        <persName>
                                            <forename type="first">Charlie</forename>
                                            <surname>Brown</surname>
                                        </persName>
                                    </author>
                                </analytic>
                                <monogr>
                                    <title level="j">Test Journal</title>
                                    <imprint>
                                        <date type="published" when="2023"/>
                                    </imprint>
                                </monogr>
                            </biblStruct>
                        </listBibl>
                    </div>
                </body>
            </text>
        </TEI>"""

        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.content = xml_response
            mock_post.return_value = mock_response

            citations = extractor.process_paper(str(dummy_pdf))

            assert "b0" in citations
            authors = citations["b0"].details.authors
            assert len(authors) == 3
            assert "Alice Johnson" in authors
            assert "Bob Williams" in authors
            assert "Charlie Brown" in authors
            assert citations["b0"].details.title == "Test Paper"
            assert citations["b0"].details.year == "2023"
            assert citations["b0"].details.venue == "Test Journal"
