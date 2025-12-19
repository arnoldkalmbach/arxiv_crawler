"""
Browser interface for exploring the arXiv papers dataset.

Run with:
    cd arxiv_search && uv run uvicorn browser.app:app --reload --port 8000
"""

import json
from datetime import datetime
from pathlib import Path

import polars as pl
from arxiv_crawler.tei_parser import parse_tei_xml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from markupsafe import Markup
from omegaconf import OmegaConf
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from arxiv_search.inference import RectFlowVectorInference
from arxiv_search.model import load_rectflow_model
from arxiv_search.search import ContextualSearch

# Paths
BROWSER_DIR = Path(__file__).parent
DATA_DIR = BROWSER_DIR.parent / "data"
# PAPERS_FILE = DATA_DIR / "papers.jsonl"
CRAWLER_STATE_FILE = BROWSER_DIR.parent.parent / "arxiv_crawler" / "data" / "v2" / "crawler_state.json"
PAPERS_FILE = BROWSER_DIR.parent.parent / "arxiv_crawler" / "data" / "v2" / "papers.jsonl"
XML_DOCS_DIR = BROWSER_DIR.parent.parent / "arxiv_crawler" / "data" / "v2" / "xml_docs"
# PAPERS_FILE = DATA_DIR / "papers.jsonl"

# Semantic search / inference paths - RectFlow model
RECTFLOW_RUN_DIR = (
    BROWSER_DIR.parent
    / "runs_rectflow"
    / "rectflow_DiT_blocks4_heads4_lr2e-04_bs128_timewuniform_timedistlognormal_mlpr6.0_20251218_120347"
)
RECTFLOW_CHECKPOINT = RECTFLOW_RUN_DIR / "checkpoints" / "model_50000.pth"
RECTFLOW_CONFIG_FILE = RECTFLOW_RUN_DIR / "config.yaml"

PAPER_EMBEDDINGS_FILE = DATA_DIR / "v2" / "paper_embeddings.parquet"
# Force CPU for inference (GPU compatibility/memory issues)
DEVICE = "cpu"

# Load rectflow config - contains all settings including model architecture
# Note: The velocity field checkpoint contains conditioning model weights,
# so we don't need a separate conditioning checkpoint for inference
rectflow_cfg = OmegaConf.load(RECTFLOW_CONFIG_FILE)
BASEMODEL_NAME = rectflow_cfg.data.basemodel_name
INFERENCE_MAX_LENGTH = rectflow_cfg.data.max_length
INFERENCE_PAD_TO_MULTIPLE_OF = rectflow_cfg.data.pad_to_multiple_of

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
contextual_search: ContextualSearch | None = None


@app.on_event("startup")
async def startup_event():
    # TODO: Need to reload this occasionally to keep the index up to date with the latest papers.
    global papers_df, arxiv_id_index, cited_by_index, contextual_search
    papers_df = load_papers()
    arxiv_id_index = build_arxiv_id_index(papers_df)
    cited_by_index = build_cited_by_index(papers_df)
    print(f"Built index with {len(arxiv_id_index)} papers, {len(cited_by_index)} cited papers")

    # Initialize semantic search inference with RectFlow model
    print(f"Loading RectFlow semantic search models on {DEVICE}...")
    print(f"  RectFlow checkpoint: {RECTFLOW_CHECKPOINT}")
    print(f"  Paper embeddings: {PAPER_EMBEDDINGS_FILE}")

    general_model = SentenceTransformer(BASEMODEL_NAME, device=DEVICE)

    # Load RectFlow model
    # The velocity field checkpoint contains both velocity field and conditioning model weights,
    # so we just need the architecture from rectflow_cfg.model (no separate conditioning checkpoint needed)
    rectified_flow = load_rectflow_model(
        velocity_field_checkpoint=str(RECTFLOW_CHECKPOINT),
        rectflow_config=rectflow_cfg.rectflow,
        model_config=rectflow_cfg.model,
        max_length=INFERENCE_MAX_LENGTH,
        device=DEVICE,
        conditioning_checkpoint=None,  # Weights come from velocity_field_checkpoint
    )

    vector_inference = RectFlowVectorInference(rectified_flow, num_steps=50)
    contextual_search = ContextualSearch(
        general_model=general_model,
        vector_inference=vector_inference,
        max_length=INFERENCE_MAX_LENGTH,
        pad_to_multiple_of=INFERENCE_PAD_TO_MULTIPLE_OF,
        device=DEVICE,
    )
    contextual_search.build_index(PAPER_EMBEDDINGS_FILE)
    print(f"Semantic search ready with {len(contextual_search.paper_embeddings)} paper embeddings")


# Pydantic models for API
class SemanticSearchRequest(BaseModel):
    arxiv_id: str
    selected_text: str
    top_k: int = 5
    num_query_variants: int = 20
    candidates_per_variant: int = 3
    lambda_: float = 0.6
    aspect_decay: float = 0.5
    aspect_threshold: float = 0.95


@app.post("/api/semantic-search")
async def semantic_search(request: SemanticSearchRequest):
    """
    Find papers semantically similar to the selected text in the context of a paper.

    Uses the paper's title+abstract as general context and the selected text as task context.
    Returns top-k matching papers from the dataset, excluding the context paper itself.
    Each result is labeled as "existing" (already cited) or "proposed" (not cited but relevant).
    """
    # Get the paper to build general context
    paper = arxiv_id_index.get(request.arxiv_id)
    if not paper:
        raise HTTPException(status_code=404, detail=f"Paper {request.arxiv_id} not found")

    # Build context in Specter format: "{title}[SEP]{abstract}"
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    general_context = f"{title}[SEP]{abstract}"
    task_context = request.selected_text

    print(f"[semantic-search] arxiv_id={request.arxiv_id}")
    print(f"[semantic-search] selected_text={task_context[:100]}...")

    # Get set of arxiv_ids that this paper already cites
    citations = paper.get("citations") or []
    cited_arxiv_ids = {cit.get("arxiv_id") for cit in citations if cit.get("arxiv_id")}
    print(f"[semantic-search] Paper cites {len(cited_arxiv_ids)} papers with arxiv_ids")

    # Overfetch by 1 to allow filtering out the context paper
    search_contexts = [(general_context, task_context)]
    matches_df = contextual_search.get_diverse_matches(
        search_contexts,
        num_results=request.top_k + 1,
        num_query_variants=request.num_query_variants,
        candidates_per_variant=request.candidates_per_variant,
        lambda_=request.lambda_,
        aspect_decay=request.aspect_decay,
        aspect_threshold=request.aspect_threshold,
    )

    # Filter to first query's results and build response
    query_matches = matches_df.filter(matches_df["query_index"] == 0)

    results = []
    for row in query_matches.iter_rows(named=True):
        match_arxiv_id = row.get("arxiv_id", "")

        # Skip the context paper itself
        if match_arxiv_id == request.arxiv_id:
            continue

        # Look up full paper info from our index
        match_paper = arxiv_id_index.get(match_arxiv_id, {})

        # Determine if this is an existing citation or a proposed one
        citation_type = "existing" if match_arxiv_id in cited_arxiv_ids else "proposed"

        results.append(
            {
                "arxiv_id": match_arxiv_id,
                "title": match_paper.get("title", row.get("title", "Unknown")),
                "abstract": match_paper.get("abstract", ""),
                "distance": float(row.get("score", row.get("distance", 0.0))),
                "citation_type": citation_type,
            }
        )

    # Limit to top_k after filtering
    results = results[: request.top_k]

    print(f"[semantic-search] Found {len(results)} matches")
    return JSONResponse(content={"matches": results})


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

    return templates.TemplateResponse(
        "paper.html",
        {
            "request": request,
            "paper": paper,
            "internal_citations": internal_citations,
            "external_citations": external_citations,
            "citing_papers": citing_papers,
        },
    )


@app.get("/crawler-status", response_class=HTMLResponse)
async def crawler_status(
    request: Request,
    queued_sort: str = "priority",
    dataset_sort: str = "cited_by",
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

    # Build in-dataset papers list with citation counts
    in_dataset_papers = []
    for arxiv_id in processed_ids:
        paper_info = arxiv_id_index.get(arxiv_id, {})
        if paper_info:
            # Count citations
            citations = paper_info.get("citations") or []
            internal_citations = sum(1 for c in citations if c.get("arxiv_id") and c.get("arxiv_id") in arxiv_id_index)
            external_citations = len(citations) - internal_citations
            cited_by_count = len(cited_by_index.get(arxiv_id, []))

            in_dataset_papers.append(
                {
                    "arxiv_id": arxiv_id,
                    "title": paper_info.get("title"),
                    "internal_citations": internal_citations,
                    "external_citations": external_citations,
                    "cited_by": cited_by_count,
                }
            )

    # Sort in-dataset papers (default "crawled" keeps original order from processed_ids)
    if dataset_sort == "cited_by":
        in_dataset_papers.sort(key=lambda x: (-x["cited_by"], x["arxiv_id"]))
    elif dataset_sort == "internal":
        in_dataset_papers.sort(key=lambda x: (-x["internal_citations"], x["arxiv_id"]))
    elif dataset_sort == "external":
        in_dataset_papers.sort(key=lambda x: (-x["external_citations"], x["arxiv_id"]))
    elif dataset_sort == "id":
        in_dataset_papers.sort(key=lambda x: x["arxiv_id"])
    elif dataset_sort == "crawled":
        pass  # Already in crawled order from iteration over processed_ids

    # Build queued papers list (pending only)
    queued_papers = []
    for arxiv_id, priority_data in queued_ids.items():
        priority = priority_data[0] if isinstance(priority_data, list) and len(priority_data) > 0 else 0
        depth = priority_data[1] if isinstance(priority_data, list) and len(priority_data) > 1 else 0
        paper_info = arxiv_id_index.get(arxiv_id, {})
        queued_papers.append(
            {
                "arxiv_id": arxiv_id,
                "priority": priority,
                "depth": depth,
                "title": paper_info.get("title"),
            }
        )

    # Sort queued papers
    if queued_sort == "priority":
        queued_papers.sort(key=lambda x: (-x["priority"], x["arxiv_id"]))
    elif queued_sort == "depth":
        queued_papers.sort(key=lambda x: (x["depth"], -x["priority"], x["arxiv_id"]))
    elif queued_sort == "id":
        queued_papers.sort(key=lambda x: x["arxiv_id"])

    return templates.TemplateResponse(
        "crawler_status.html",
        {
            "request": request,
            "processed_count": len(processed_ids),
            "failed_count": len(failed_ids),
            "queued_count": len(queued_ids),
            "last_updated": last_updated_display,
            "in_dataset_papers": in_dataset_papers,
            "queued_papers": queued_papers,
            "failed_ids": failed_ids,
            "queued_sort": queued_sort,
            "dataset_sort": dataset_sort,
        },
    )
