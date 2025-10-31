import argparse
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo, CommitOperationAdd

# Load environment variables (for HF_TOKEN)
load_dotenv()


def parse_args():
    """Parse command line arguments."""
    # Get script directory to make default paths relative to script location
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent  # arxiv_search/
    crawler_root = project_root.parent / "arxiv_crawler"  # ../arxiv_crawler

    parser = argparse.ArgumentParser(
        description="Upload arXiv dataset to HuggingFace Hub",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset_name", type=str, help="HuggingFace dataset name (e.g., username/dataset-name)")
    parser.add_argument(
        "--include-xml",
        action="store_true",
        help="Include XML source documents in the upload (optional, can be large)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(project_root / "data"),
        help="Path to data directory containing embeddings",
    )
    parser.add_argument(
        "--papers-file",
        type=str,
        default=str(crawler_root / "data" / "papers.jsonl"),
        help="Path to papers.jsonl file",
    )
    parser.add_argument(
        "--xml-docs-dir",
        type=str,
        default=str(crawler_root / "data" / "xml_docs"),
        help="Path to XML documents directory",
    )
    parser.add_argument(
        "--dataset-card",
        type=str,
        default=str(project_root / "DATASET_CARD.md"),
        help="Path to dataset card template file",
    )
    return parser.parse_args()


def load_dataset_card(dataset_name: str, include_xml: bool, card_template_path: Path) -> str:
    """
    Load and populate the dataset card template.

    Args:
        dataset_name: HuggingFace dataset name
        include_xml: Whether XML files are included in upload
        card_template_path: Path to the dataset card template markdown file

    Returns:
        Populated dataset card content
    """
    # Read template
    card_content = card_template_path.read_text()

    # Prepare optional XML section
    xml_section = (
        """### Optional Files
- `xml_docs/*.xml.gz`: Source XML documents from arXiv (for reproducibility/inspection)

#### Download XML Files
```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="{dataset_name}",
    repo_type="dataset",
    allow_patterns=["xml_docs/*.xml.gz"]
)
```
"""
        if include_xml
        else ""
    )

    # Substitute placeholders
    card_content = card_content.replace("{DATASET_NAME}", dataset_name)
    card_content = card_content.replace("{OPTIONAL_XML_SECTION}", xml_section)

    return card_content


def main():
    """Main function to upload dataset to HuggingFace Hub."""
    args = parse_args()

    # Convert paths
    data_dir = Path(args.data_dir)
    papers_file = Path(args.papers_file)
    xml_docs_dir = Path(args.xml_docs_dir)
    card_template_path = Path(args.dataset_card)

    print(f"Uploading dataset: {args.dataset_name}")
    print(f"Data directory: {data_dir}")
    print(f"Include XML: {args.include_xml}")
    print(f"Dataset card template: {card_template_path}")

    # Create repository
    print("Creating/checking repository...")
    create_repo(repo_id=args.dataset_name, repo_type="dataset", exist_ok=True)

    # Collect files to upload
    operations = []

    # Paper embeddings (shared)
    paper_embeddings_file = data_dir / "paper_embeddings.parquet"
    if paper_embeddings_file.exists():
        print(f"Adding paper embeddings: {paper_embeddings_file}")
        operations.append(
            CommitOperationAdd(
                path_in_repo="paper_embeddings.parquet",
                path_or_fileobj=str(paper_embeddings_file),
            )
        )
    else:
        print(f"Warning: {paper_embeddings_file} not found, skipping")

    # Source papers
    if papers_file.exists():
        print(f"Adding papers metadata: {papers_file}")
        operations.append(CommitOperationAdd(path_in_repo="papers.jsonl", path_or_fileobj=str(papers_file)))
    else:
        print(f"Warning: {papers_file} not found, skipping")

    # Train split
    train_dir = data_dir / "train"
    if train_dir.exists():
        train_citations = train_dir / "citations.jsonl"
        if train_citations.exists():
            print(f"Adding train citations: {train_citations}")
            operations.append(
                CommitOperationAdd(path_in_repo="train/citations.jsonl", path_or_fileobj=str(train_citations))
            )

        # Support both .arrow and .parquet for citation embeddings
        parquet_files = sorted(train_dir.glob("citation_embeddings_*.parquet"))
        arrow_files = sorted(train_dir.glob("citation_embeddings_*.arrow"))
        citation_files = parquet_files if parquet_files else arrow_files

        print(f"Adding {len(citation_files)} train citation embedding files")
        for embed_file in citation_files:
            operations.append(
                CommitOperationAdd(path_in_repo=f"train/{embed_file.name}", path_or_fileobj=str(embed_file))
            )
    else:
        print(f"Warning: {train_dir} not found, skipping train split")

    # Test split
    test_dir = data_dir / "test"
    if test_dir.exists():
        test_citations = test_dir / "citations.jsonl"
        if test_citations.exists():
            print(f"Adding test citations: {test_citations}")
            operations.append(
                CommitOperationAdd(path_in_repo="test/citations.jsonl", path_or_fileobj=str(test_citations))
            )

        # Support both .arrow and .parquet for citation embeddings
        parquet_files = sorted(test_dir.glob("citation_embeddings_*.parquet"))
        arrow_files = sorted(test_dir.glob("citation_embeddings_*.arrow"))
        citation_files = parquet_files if parquet_files else arrow_files

        print(f"Adding {len(citation_files)} test citation embedding files")
        for embed_file in citation_files:
            operations.append(
                CommitOperationAdd(path_in_repo=f"test/{embed_file.name}", path_or_fileobj=str(embed_file))
            )
    else:
        print(f"Warning: {test_dir} not found, skipping test split")

    # XML source documents (optional)
    if args.include_xml and xml_docs_dir.exists():
        print(f"Including XML documents from {xml_docs_dir}")
        xml_files = list(xml_docs_dir.rglob("*.xml.gz"))
        print(f"Found {len(xml_files)} XML files")
        for xml_file in sorted(xml_files):
            relative_path = xml_file.relative_to(xml_docs_dir.parent)
            operations.append(CommitOperationAdd(path_in_repo=str(relative_path), path_or_fileobj=str(xml_file)))
    elif args.include_xml:
        print(f"Warning: --include-xml specified but {xml_docs_dir} not found")

    # Upload
    print(f"\nUploading {len(operations)} files...")
    api = HfApi()
    api.create_commit(
        repo_id=args.dataset_name,
        repo_type="dataset",
        operations=operations,
        commit_message=f"Upload arXiv citation embeddings dataset ({len(operations)} files)",
    )

    # Create and upload dataset card
    print("Creating dataset card...")
    from huggingface_hub import DatasetCard

    if not card_template_path.exists():
        print(f"Warning: Dataset card template not found at {card_template_path}, skipping card upload")
    else:
        card_content = load_dataset_card(args.dataset_name, args.include_xml, card_template_path)
        card = DatasetCard(card_content)
        card.push_to_hub(args.dataset_name, repo_type="dataset")
        print("Dataset card uploaded")

    print(f"\n✓ Successfully uploaded to: https://huggingface.co/datasets/{args.dataset_name}")


if __name__ == "__main__":
    main()
