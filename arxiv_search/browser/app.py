"""
Browser interface for exploring the arXiv papers dataset.

Run with:
    cd arxiv_search && uv run uvicorn browser.app:app --reload --port 8000
"""

import json
from datetime import datetime
from pathlib import Path

import polars as pl
from arxiv_crawler import parse_tei_xml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from markupsafe import Markup

# Paths
BROWSER_DIR = Path(__file__).parent
DATA_DIR = BROWSER_DIR.parent / "data"
# PAPERS_FILE = DATA_DIR / "papers.jsonl"
CRAWLER_STATE_FILE = BROWSER_DIR.parent.parent / "arxiv_crawler" / "data" / "v2" / "crawler_state.json"
PAPERS_FILE = BROWSER_DIR.parent.parent / "arxiv_crawler" / "data" / "v2" / "papers.jsonl"
XML_DOCS_DIR = BROWSER_DIR.parent.parent / "arxiv_crawler" / "data" / "v2" / "xml_docs"
# PAPERS_FILE = DATA_DIR / "papers.jsonl"

# Initialize FastAPI app
app = FastAPI(title="arXiv Browser")

# Setup Jinja2 templates
templates = Jinja2Templates(directory=BROWSER_DIR / "templates")


def load_papers() -> pl.DataFrame:
    """Load papers from JSONL file."""
    print(f"Loading papers from {PAPERS_FILE}...")
    papers = pl.read_ndjson(PAPERS_FILE)
    print(f"Loaded {len(papers)} papers")
    return papers


def build_arxiv_id_index(papers: pl.DataFrame) -> dict[str, dict]:
    """Build a lookup from arxiv_id to paper data."""
    index = {}
    for row in papers.iter_rows(named=True):
        index[row["arxiv_id"]] = row
    return index


def build_cited_by_index(papers: pl.DataFrame) -> dict[str, list[str]]:
    """Build a reverse index: cited_arxiv_id -> list of citer_arxiv_ids."""
    cited_by: dict[str, list[str]] = {}
    for row in papers.iter_rows(named=True):
        citer_id = row["arxiv_id"]
        citations = row.get("citations") or []
        for citation in citations:
            cited_id = citation.get("arxiv_id")
            if cited_id:
                if cited_id not in cited_by:
                    cited_by[cited_id] = []
                cited_by[cited_id].append(citer_id)
    return cited_by


# Load data at startup
papers_df: pl.DataFrame = None  # type: ignore
arxiv_id_index: dict[str, dict] = {}
cited_by_index: dict[str, list[str]] = {}


@app.on_event("startup")
async def startup_event():
    global papers_df, arxiv_id_index, cited_by_index
    papers_df = load_papers()
    arxiv_id_index = build_arxiv_id_index(papers_df)
    cited_by_index = build_cited_by_index(papers_df)
    print(f"Built index with {len(arxiv_id_index)} papers, {len(cited_by_index)} cited papers")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request, page: int = 1):
    """Home page with papers sorted by citation count."""
    per_page = 25

    # Build list of papers with their citation counts
    papers_with_counts = []
    for arxiv_id, paper in arxiv_id_index.items():
        cite_count = len(cited_by_index.get(arxiv_id, []))
        papers_with_counts.append(
            {
                "arxiv_id": arxiv_id,
                "title": paper.get("title", "Unknown"),
                "abstract": paper.get("abstract", ""),
                "published": paper.get("published"),
                "categories": paper.get("categories", []),
                "cited_by_count": cite_count,
            }
        )

    # Sort by citation count (descending)
    papers_with_counts.sort(key=lambda x: x["cited_by_count"], reverse=True)

    # Pagination
    total_papers = len(papers_with_counts)
    total_pages = (total_papers + per_page - 1) // per_page
    page = max(1, min(page, total_pages))  # Clamp page number
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page

    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "num_papers": total_papers,
            "papers": papers_with_counts[start_idx:end_idx],
            "page": page,
            "total_pages": total_pages,
            "per_page": per_page,
        },
    )


@app.get("/search", response_class=HTMLResponse)
async def search(request: Request, q: str = ""):
    """Search papers by keyword in title or abstract."""
    if not q.strip():
        return templates.TemplateResponse(
            "search.html",
            {"request": request, "query": q, "papers": [], "num_results": 0},
        )

    # Case-insensitive search in title and abstract
    query_lower = q.lower()
    results = papers_df.filter(
        pl.col("title").str.to_lowercase().str.contains(query_lower, literal=True)
        | pl.col("abstract").str.to_lowercase().str.contains(query_lower, literal=True)
        | (
            pl.col("authors")
            .list.eval(pl.element().str.to_lowercase().str.contains(query_lower, literal=True))
            .list.any()
        )
    )

    # Convert to list of dicts for template and add citation counts
    papers_list = results.select(["arxiv_id", "title", "abstract", "published", "categories"]).to_dicts()
    for paper in papers_list:
        paper["cited_by_count"] = len(cited_by_index.get(paper["arxiv_id"], []))

    return templates.TemplateResponse(
        "search.html",
        {
            "request": request,
            "query": q,
            "papers": papers_list[:100],  # Limit to 100 results
            "num_results": len(results),
        },
    )


@app.get("/paper/{arxiv_id:path}/fulltext", response_class=HTMLResponse)
async def paper_fulltext(request: Request, arxiv_id: str):
    """Display full-text view of a paper from GROBID TEI XML."""
    # Check if paper exists in our index
    paper = arxiv_id_index.get(arxiv_id)

    # Find the XML file
    xml_file = XML_DOCS_DIR / f"{arxiv_id}.xml.gz"
    if not xml_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Full-text XML not available for paper {arxiv_id}",
        )

    # URL builder for paper links
    def paper_url_builder(aid: str) -> str:
        return f"/paper/{aid}"

    # Parse the TEI XML
    try:
        parsed = parse_tei_xml(xml_file, paper_url_builder=paper_url_builder)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error parsing XML for {arxiv_id}: {str(e)}",
        )

    # Convert toc entries to dicts for template
    toc_dicts = [{"id": t.id, "num": t.num, "title": t.title} for t in parsed.toc]

    return templates.TemplateResponse(
        "fulltext.html",
        {
            "request": request,
            "arxiv_id": arxiv_id,
            "paper": paper,
            "parsed": parsed,
            "title": parsed.title or (paper.get("title") if paper else arxiv_id),
            "authors": parsed.authors or (paper.get("authors", []) if paper else []),
            "date": parsed.date,
            "abstract_html": Markup(parsed.abstract_html),
            "body_html": Markup(parsed.body_html),
            "ack_html": Markup(parsed.ack_html),
            "references_html": Markup(parsed.references_html),
            "toc": toc_dicts,
            "bibliography": parsed.bibliography,
        },
    )


@app.get("/paper/{arxiv_id:path}", response_class=HTMLResponse)
async def paper_detail(request: Request, arxiv_id: str):
    """Display details for a single paper."""
    paper = arxiv_id_index.get(arxiv_id)
    if not paper:
        raise HTTPException(status_code=404, detail=f"Paper {arxiv_id} not found")

    # Process citations - separate those in our dataset from external ones
    citations = paper.get("citations") or []
    internal_citations = []
    external_citations = []

    for cit in citations:
        cit_arxiv_id = cit.get("arxiv_id")
        if cit_arxiv_id and cit_arxiv_id in arxiv_id_index:
            # This citation is in our dataset - include both dataset info and citation metadata
            dataset_paper = arxiv_id_index[cit_arxiv_id]
            internal_citations.append(
                {
                    "arxiv_id": cit_arxiv_id,
                    "title": dataset_paper.get("title", cit.get("title", "Unknown")),
                    "authors": dataset_paper.get("authors", cit.get("authors", [])),
                    "published": dataset_paper.get("published"),
                    "categories": dataset_paper.get("categories", []),
                    "reference_contexts": cit.get("reference_contexts", []),
                    "in_dataset": True,
                }
            )
        else:
            # External citation
            external_citations.append(
                {
                    "arxiv_id": cit_arxiv_id,
                    "title": cit.get("title", "Unknown"),
                    "authors": cit.get("authors", []),
                    "year": cit.get("year"),
                    "venue": cit.get("venue"),
                    "reference_contexts": cit.get("reference_contexts", []),
                    "in_dataset": False,
                }
            )

    # Get papers that cite this one
    citers = cited_by_index.get(arxiv_id, [])
    citing_papers = []
    for citer_id in citers:
        citer = arxiv_id_index.get(citer_id)
        if citer:
            # Find the reference context where this paper is cited
            reference_contexts = []
            citer_citations = citer.get("citations") or []
            for cit in citer_citations:
                if cit.get("arxiv_id") == arxiv_id:
                    reference_contexts = cit.get("reference_contexts", [])
                    break
            citing_papers.append(
                {
                    "arxiv_id": citer_id,
                    "title": citer.get("title", "Unknown"),
                    "reference_contexts": reference_contexts,
                }
            )

    # Get total citation count - how many papers cite this one (from num_citations field if available)
    # This includes citations from papers not in our dataset
    total_cited_by = paper.get("num_citations", len(citing_papers))

    return templates.TemplateResponse(
        "paper.html",
        {
            "request": request,
            "paper": paper,
            "internal_citations": internal_citations,
            "external_citations": external_citations,
            "citing_papers": citing_papers,
            "total_cited_by": total_cited_by,
        },
    )


@app.get("/crawler-status", response_class=HTMLResponse)
async def crawler_status(
    request: Request,
    sort: str = "priority",
    show_processed: bool = False,
    filter: str = "all",
):
    """Display the status of the crawler state file."""
    if not CRAWLER_STATE_FILE.exists():
        raise HTTPException(status_code=404, detail="Crawler state file not found")

    with open(CRAWLER_STATE_FILE) as f:
        state = json.load(f)

    processed_ids = state.get("processed_ids", [])
    failed_ids = state.get("failed_ids", [])
    queued_ids = state.get("queued_ids", {})
    last_updated = state.get("last_updated", "Unknown")

    # Parse last_updated for display
    try:
        last_updated_dt = datetime.fromisoformat(last_updated)
        last_updated_display = last_updated_dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        last_updated_display = str(last_updated)

    # Build combined list: processed papers (in_dataset=True) + queued papers (in_dataset=False)
    processed_set = set(processed_ids)
    all_papers = []
    
    # Add processed papers (no priority/depth info available after processing)
    for arxiv_id in processed_ids:
        paper_info = arxiv_id_index.get(arxiv_id, {})
        all_papers.append({
            "arxiv_id": arxiv_id,
            "priority": None,
            "depth": None,
            "title": paper_info.get("title"),
            "in_dataset": True,
        })
    
    # Add queued papers
    for arxiv_id, priority_data in queued_ids.items():
        priority = priority_data[0] if isinstance(priority_data, list) and len(priority_data) > 0 else 0
        depth = priority_data[1] if isinstance(priority_data, list) and len(priority_data) > 1 else 0
        paper_info = arxiv_id_index.get(arxiv_id, {})
        all_papers.append({
            "arxiv_id": arxiv_id,
            "priority": priority,
            "depth": depth,
            "title": paper_info.get("title"),
            "in_dataset": False,
        })

    # Apply filter
    if filter == "in_dataset":
        queued_papers = [p for p in all_papers if p["in_dataset"]]
    elif filter == "pending":
        queued_papers = [p for p in all_papers if not p["in_dataset"]]
    else:
        queued_papers = all_papers

    # Count for filter badges
    in_dataset_count = len(processed_ids)
    pending_count = len(queued_ids)

    # Sort queued papers (handle None values for processed papers)
    def sort_key_priority(x):
        p = x["priority"] if x["priority"] is not None else -1
        return (-p, x["arxiv_id"])
    
    def sort_key_depth(x):
        d = x["depth"] if x["depth"] is not None else -1
        p = x["priority"] if x["priority"] is not None else -1
        return (d, -p, x["arxiv_id"])
    
    if sort == "priority":
        queued_papers.sort(key=sort_key_priority)
    elif sort == "depth":
        queued_papers.sort(key=sort_key_depth)
    elif sort == "id":
        queued_papers.sort(key=lambda x: x["arxiv_id"])

    # Build processed papers list with dataset info
    processed_papers = []
    if show_processed:
        for arxiv_id in processed_ids:
            paper_info = arxiv_id_index.get(arxiv_id, {})
            processed_papers.append({
                "arxiv_id": arxiv_id,
                "title": paper_info.get("title"),
                "in_dataset": arxiv_id in arxiv_id_index,
            })

    return templates.TemplateResponse(
        "crawler_status.html",
        {
            "request": request,
            "processed_count": len(processed_ids),
            "failed_count": len(failed_ids),
            "queued_count": len(queued_ids),
            "in_dataset_count": in_dataset_count,
            "pending_count": pending_count,
            "last_updated": last_updated_display,
            "queued_papers": queued_papers,
            "processed_papers": processed_papers,
            "failed_ids": failed_ids,
            "sort": sort,
            "filter": filter,
            "show_processed": show_processed,
        },
    )
