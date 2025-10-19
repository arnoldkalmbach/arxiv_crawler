"""Data models for the arxiv crawler."""

from typing import Optional
from pydantic import BaseModel, Field


class CitationDetails(BaseModel):
    """Details about a cited paper."""

    authors: list[str] = Field(default_factory=list)
    title: Optional[str] = None
    year: Optional[str] = None
    venue: Optional[str] = None
    arxiv_id: Optional[str] = None


class Citation(BaseModel):
    """A citation with its details and reference contexts."""

    citation_id: str
    details: CitationDetails
    references: list[str] = Field(default_factory=list, description="Sentences that reference this citation")


class ProcessedCitation(BaseModel):
    """Citation data as processed for storage (flattened version)."""

    citation_id: str
    authors: list[str] = Field(default_factory=list)
    title: Optional[str] = None
    year: Optional[str] = None
    venue: Optional[str] = None
    arxiv_id: Optional[str] = None
    reference_contexts: list[str] = Field(default_factory=list)
    num_references: int = 0


class ProcessedPaper(BaseModel):
    """Complete data for a processed paper."""

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    published: str
    pdf_url: str
    arxiv_url: str
    full_text: str
    citations: list[ProcessedCitation]
    num_citations: int
    num_arxiv_citations: int
    depth: int
    processing_timestamp: str

    class Config:
        # Allow extra fields for future extensibility
        extra = "allow"
