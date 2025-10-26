"""
Build embeddings dataset from arxiv papers.

This script processes arxiv papers and citations from arxiv_crawler.
It finds the citations that we've crawled and generates:
    - sentence embeddings for each paper (title + abstract)
    - full embeddings (sentence + token) for the context in which each citation appears

Inputs:
    - papers.jsonl: JSONL file containing papers with fields:
        - arxiv_id: unique paper identifier
        - title: paper title
        - abstract: paper abstract
        - citations: list of cited papers with reference contexts
        - published: publication date

Outputs:
    - citations.jsonl: Top level citations mapping arxiv id to reference context
    - paper_embeddings.parquet: Embeddings for each paper (title + abstract)
    - citation_embeddings_{batch}.parquet: Embeddings for citation contexts in batches
"""

from dotenv import load_dotenv

load_dotenv("../.env")  # Disable XET

import argparse
from pathlib import Path
from typing import Dict

import polars as pl
import sentence_transformers


def load_papers(papers_path: str, date_format: str = "%Y-%m-%d") -> pl.DataFrame:
    """
    Load papers from JSONL file and parse dates.

    Args:
        papers_path: Path to the papers JSONL file
        date_format: Format string for parsing publication dates

    Returns:
        DataFrame with loaded papers
    """
    papers = pl.read_ndjson(papers_path).with_columns(pl.col("published").str.to_date(format=date_format, exact=False))
    return papers


def create_arxiv_id_to_context(papers: pl.DataFrame) -> Dict[str, str]:
    """
    Create mapping from arxiv_id to context string (title[SEP]abstract).

    Args:
        papers: DataFrame containing papers with arxiv_id, title, and abstract columns

    Returns:
        Dictionary mapping arxiv_id to formatted context string
    """

    # "{title}[SEP]{abstract}" is the context used by Specter
    # See https://github.com/huggingface/sentence-transformers/blob/master/examples/sentence_transformer/applications/semantic-search/semantic_search_publications.py
    arxiv_id_to_context = {
        arxiv_id: f"{title}[SEP]{abstract}"
        for arxiv_id, title, abstract in papers[["arxiv_id", "title", "abstract"]].iter_rows()
    }
    return arxiv_id_to_context


def process_citations(papers: pl.DataFrame, valid_arxiv_ids: set) -> pl.DataFrame:
    """
    Extract and process citations from papers.

    Args:
        papers: DataFrame containing papers with citations
        valid_arxiv_ids: Set of valid arxiv IDs to filter citations

    Returns:
        DataFrame with processed citations including reference contexts and IDs
    """
    citations = (
        papers.explode("citations")
        .filter(pl.col("citations").struct["arxiv_id"].is_in(list(valid_arxiv_ids)))
        .select(["arxiv_id", "citations"])
        .rename({"arxiv_id": "citer_arxiv_id"})
        .unnest("citations")
        .rename({"arxiv_id": "cited_arxiv_id"})
        .select(["citer_arxiv_id", "cited_arxiv_id", "reference_contexts"])
        .explode("reference_contexts")
        .with_row_index()
        .with_columns(pl.col("reference_contexts").replace(None, ""))
        .with_columns(pl.col("reference_contexts").hash().cast(pl.Binary).bin.encode("base64").alias("reference_id"))
    )
    return citations


def generate_paper_embeddings(
    papers: pl.DataFrame,
    arxiv_id_to_context: Dict[str, str],
    sentence_encoder: sentence_transformers.SentenceTransformer,
) -> pl.DataFrame:
    """
    Generate embeddings for papers using title and abstract.

    Args:
        papers: DataFrame containing papers
        arxiv_id_to_context: Mapping from arxiv_id to context string
        sentence_encoder: SentenceTransformer model for encoding

    Returns:
        DataFrame with arxiv_id and sentence_embedding columns
    """
    print("Generating paper embeddings...")
    embeddings = sentence_encoder.encode(
        list(arxiv_id_to_context.values()),
        output_value="sentence_embedding",
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    embedding_size = embeddings.shape[1]

    paper_embeddings = pl.DataFrame(
        [
            papers["arxiv_id"],
            pl.Series(
                values=embeddings, name="sentence_embedding", dtype=pl.Array(pl.Float32, shape=(embedding_size,))
            ),
        ]
    )

    return paper_embeddings


def generate_citation_embeddings_batch(
    citations: pl.DataFrame,
    sentence_encoder: sentence_transformers.SentenceTransformer,
    batch_size: int,
    output_dir: Path,
):
    """
    Generate embeddings for citation contexts in batches and save to disk.

    Args:
        citations: DataFrame containing citation reference contexts
        sentence_encoder: SentenceTransformer model for encoding
        batch_size: Number of citations to process per batch
        output_dir: Directory to save batch embedding files
    """
    print(f"Generating citation embeddings in batches of {batch_size}...")

    # Get embedding size from first batch
    embedding_size = sentence_encoder.get_sentence_embedding_dimension()

    for batch_start in range(0, len(citations), batch_size):
        print(f"Processing batch starting at {batch_start}...")
        citations_batch = citations[batch_start : batch_start + batch_size]

        embeddings = sentence_encoder.encode(
            citations_batch["reference_contexts"].to_list(),
            show_progress_bar=True,
            output_value=None,
            convert_to_numpy=True,
        )

        schema = {
            "input_ids": pl.List(pl.Int32),
            "token_type_ids": pl.List(pl.Int32),
            "attention_mask": pl.List(pl.Int32),
            "token_embeddings": pl.List(pl.Array(pl.Float32, shape=(embedding_size,))),
            "sentence_embedding": pl.Array(pl.Float32, shape=(embedding_size,)),
        }

        citation_embeddings = pl.DataFrame(
            [
                pl.Series(
                    values=[embeddings[i][k].squeeze().cpu().numpy() for i in range(len(embeddings))],
                    name=k,
                    dtype=dtype,
                )
                for k, dtype in schema.items()
            ]
            + [citations_batch["reference_id"]]
        )

        output_path = output_dir / f"citation_embeddings_{batch_start}.arrow"
        citation_embeddings.write_ipc(str(output_path))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build embeddings dataset from arxiv papers", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input-papers", type=str, default="../../data/papers.jsonl", help="Path to input papers JSONL file"
    )

    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save output files")

    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/allenai-specter",
        help="Name of the sentence transformer model to use",
    )

    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"], help="Device to run the model on"
    )

    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for processing citation embeddings")

    parser.add_argument("--date-format", type=str, default="%Y-%m-%d", help="Date format for parsing publication dates")

    return parser.parse_args()


def main():
    """Main function to orchestrate the embedding generation process."""
    args = parse_args()

    # Convert paths
    input_papers = args.input_papers
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    MODEL_NAME = args.model_name
    DEVICE = args.device
    BATCH_SIZE = args.batch_size
    DATE_FORMAT = args.date_format

    print(f"Loading model: {MODEL_NAME} on {DEVICE}")
    sentence_encoder = sentence_transformers.SentenceTransformer(MODEL_NAME, device=DEVICE)

    print(f"Loading papers from {input_papers}")
    papers = load_papers(input_papers, DATE_FORMAT)
    print(f"Loaded {len(papers)} papers")

    print("Creating arxiv_id to context mapping...")
    arxiv_id_to_context = create_arxiv_id_to_context(papers)

    print("Processing citations...")
    citations = process_citations(papers, set(arxiv_id_to_context.keys()))
    print(f"Processed {len(citations)} citation contexts")

    citations_path = output_dir / "citations.jsonl"
    print(f"Saving citations to {citations_path}")
    citations.write_ndjson(str(citations_path))

    paper_embeddings = generate_paper_embeddings(papers, arxiv_id_to_context, sentence_encoder)
    paper_embeddings_path = output_dir / "paper_embeddings.parquet"
    print(f"Saving paper embeddings to {paper_embeddings_path}")
    paper_embeddings.write_parquet(str(paper_embeddings_path))

    # Generate and save citation embeddings in batches
    generate_citation_embeddings_batch(citations, sentence_encoder, BATCH_SIZE, output_dir)


if __name__ == "__main__":
    main()
