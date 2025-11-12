"""
Precompute embeddings for all paragraphs (judgments) and articles (legal acts).

This module creates a unified, persistent representation of all text nodes
that can be reused for various GNN training, evaluation, and analysis tasks.
"""

import json
import pickle
from pathlib import Path
from typing import Any
from datetime import datetime
import pandas as pd  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer  # type: ignore
from tqdm import tqdm  # type: ignore


class EmbeddingPreprocessor:
    """Preprocess and encode all paragraphs and articles with metadata."""

    def __init__(
        self,
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        mask_token: str = "[MASK]",
    ):
        self.encoder_name = encoder_name
        self.batch_size = batch_size
        self.mask_token = mask_token
        self.encoder = SentenceTransformer(encoder_name)

    def process_judgments(
        self,
        judgments_path: str,
        par_to_par_path: str,
        output_dir: str,
    ) -> None:
        """
        Process all judgment paragraphs and create unified index with both document and query embeddings.

        Creates:
        - paragraph_embeddings_doc.npy: Pre-computed embeddings for documents (cited paragraphs)
        - paragraph_embeddings_query.npy: Pre-computed masked embeddings for queries (citing paragraphs)
        - paragraph_metadata.pkl: Metadata for each paragraph
        - paragraph_index.json: Human-readable index mapping
        """
        print("Loading judgments...")
        with open(judgments_path) as f:
            judgments = json.load(f)

        # Load par-to-par data to get masked TEXT_FROM
        print("Loading par-to-par citations...")
        df = pd.read_csv(par_to_par_path)

        # Create mapping from paragraph ID to TEXT_FROM (masked text)
        text_from_map: dict[str, str] = {}
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing TEXT_FROM"):
            par_id = f"par:{row['CELEX_FROM']}:{row['NUMBER_FROM']}"
            text_from = str(row["TEXT_FROM"])
            # Use the first occurrence (they should all be the same for a given paragraph)
            if par_id not in text_from_map:
                text_from_map[par_id] = text_from

        print(f"Found masked text for {len(text_from_map)} citing paragraphs")

        # Collect all paragraphs with metadata
        paragraphs_data = []
        for celex, judgment in tqdm(judgments.items(), desc="Processing judgments"):
            meta = judgment.get("meta", {})
            date_str = meta.get("date")

            # Parse date
            date = None
            if date_str:
                try:
                    date = datetime.strptime(date_str, "%Y-%m-%d")
                except:
                    pass

            # Extract paragraphs
            for par_num, text in judgment.get("paragraphs", {}).items():
                par_id = f"par:{celex}:{par_num}"
                paragraphs_data.append(
                    {
                        "id": par_id,
                        "type": "paragraph",
                        "text": text,
                        "text_from": text_from_map.get(
                            par_id
                        ),  # Masked text if available
                        "celex": celex,
                        "paragraph_number": int(par_num),
                        "date": date.isoformat() if date else None,
                        "year": date.year if date else None,
                        "meta": meta,
                    }
                )

        print(f"Found {len(paragraphs_data)} paragraphs")

        # Encode document embeddings (full text)
        print("Encoding document embeddings (full text)...")
        doc_texts = [p["text"] for p in paragraphs_data]
        doc_embeddings = self.encoder.encode(
            doc_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        # Encode query embeddings (masked text from TEXT_FROM)
        print("Encoding query embeddings (masked TEXT_FROM)...")
        query_texts = []
        paragraphs_with_masked_text = 0
        for p in paragraphs_data:
            if p["text_from"] is not None:
                # Use masked text from par-to-par
                query_texts.append(p["text_from"])
                paragraphs_with_masked_text += 1
            else:
                # Fallback: use mask token for paragraphs that never cite
                query_texts.append(self.mask_token)

        print(
            f"Using TEXT_FROM for {paragraphs_with_masked_text}/{len(paragraphs_data)} paragraphs"
        )

        query_embeddings = self.encoder.encode(
            query_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        # Save outputs
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("Saving paragraph embeddings...")
        np.save(output_path / "paragraph_embeddings_doc.npy", doc_embeddings)
        np.save(output_path / "paragraph_embeddings_query.npy", query_embeddings)

        print("Saving paragraph metadata...")
        # Create simplified metadata (without text to reduce size)
        metadata = [
            {k: v for k, v in p.items() if k not in ["text", "text_from"]}
            for p in paragraphs_data
        ]
        with open(output_path / "paragraph_metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        # Create human-readable index
        print("Saving paragraph index...")
        index = {
            "encoder_name": self.encoder_name,
            "num_paragraphs": len(paragraphs_data),
            "embedding_dim": doc_embeddings.shape[1],
            "date_range": self._get_date_range(paragraphs_data),
            "sample_ids": [p["id"] for p in paragraphs_data[:10]],
        }
        with open(output_path / "paragraph_index.json", "w") as f:
            json.dump(index, f, indent=2)

        print(
            f"✓ Saved {len(paragraphs_data)} paragraph embeddings (doc + query) to {output_path}"
        )

    def process_legal_acts(
        self,
        legal_acts_path: str,
        output_dir: str,
    ) -> None:
        """
        Process all legal act articles and create unified index.

        Creates:
        - article_embeddings.npy: Pre-computed embeddings
        - article_metadata.pkl: Metadata for each article
        - article_index.json: Human-readable index mapping
        """
        print("Loading legal acts...")
        with open(legal_acts_path) as f:
            legal_acts = json.load(f)

        # Collect all articles with metadata
        articles_data = []
        for celex, act in tqdm(legal_acts.items(), desc="Processing legal acts"):
            for article in act.get("articles", []):
                # Get article number and text
                art_num = article.get("number", "")
                text = article.get("text", "")

                if not text:  # Skip empty articles
                    continue

                articles_data.append(
                    {
                        "id": f"art:{celex}:{art_num}",
                        "type": "article",
                        "text": text,
                        "celex": celex,
                        "article_number": art_num,
                        "title": act.get("title", ""),
                        "notes": article.get("notes", []),
                        "references": article.get("references", []),
                    }
                )

        print(f"Found {len(articles_data)} articles")

        # Encode all texts
        print("Encoding articles...")
        texts = [a["text"] for a in articles_data]
        embeddings = self.encoder.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        # Save outputs
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("Saving article embeddings...")
        np.save(output_path / "article_embeddings.npy", embeddings)

        print("Saving article metadata...")
        # Create simplified metadata (without text to reduce size)
        metadata = [{k: v for k, v in a.items() if k != "text"} for a in articles_data]
        with open(output_path / "article_metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        # Create human-readable index
        print("Saving article index...")
        index = {
            "encoder_name": self.encoder_name,
            "num_articles": len(articles_data),
            "embedding_dim": embeddings.shape[1],
            "sample_ids": [a["id"] for a in articles_data[:10]],
        }
        with open(output_path / "article_index.json", "w") as f:
            json.dump(index, f, indent=2)

        print(f"✓ Saved {len(articles_data)} article embeddings to {output_path}")

    def process_citations(
        self,
        judgments_path: str,
        par_to_par_path: str,
        output_dir: str,
    ) -> None:
        """
        Process citation relationships between paragraphs.

        Creates:
        - citations.pkl: List of (source_id, target_id) citation edges
        """
        print("Loading citation data...")
        with open(judgments_path) as f:
            judgments = json.load(f)

        df = pd.read_csv(par_to_par_path)

        # Extract citation edges
        citations = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing citations"):
            source_id = f"par:{row['CELEX_FROM']}:{row['NUMBER_FROM']}"
            target_id = f"par:{row['CELEX_TO']}:{row['NUMBER_TO']}"
            citations.append((source_id, target_id))

        print(f"Found {len(citations)} paragraph-to-paragraph citation edges")

        # Extract paragraph-to-article citation edges
        article_citations = []
        for celex_from, judgment in tqdm(
            judgments.items(), total=len(judgments), desc="Processing article citations"
        ):
            meta = judgment.get("meta", {})
            for reference in meta.get("references", []):
                if reference.get("unit") != "paragraphs":
                    continue

                if reference.get("cited_unit") != "articles":
                    continue
                for from_par in reference.get("from", []):
                    source_id = f"par:{celex_from}:{from_par}"
                    for to_art in reference.get("to", []):
                        target_id = f"art:{reference.get('target')}:{to_art}"
                        article_citations.append((source_id, target_id))

        print(f"Found {len(article_citations)} paragraph-to-article citation edges")

        citations.extend(article_citations)

        # Save citations
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with open(output_path / "citations.pkl", "wb") as f:
            pickle.dump(citations, f)

        print(f"✓ Saved {len(citations)} citation edges to {output_path}")

    def _get_date_range(
        self, paragraphs_data: list[dict[str, Any]]
    ) -> dict[str, str | None]:
        """Get min and max dates from paragraphs."""
        dates: list[str] = [
            str(p["date"]) for p in paragraphs_data if p.get("date") is not None
        ]
        if not dates:
            return {"min": None, "max": None}
        return {"min": min(dates), "max": max(dates)}


def main():
    """Main preprocessing pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Precompute embeddings for paragraphs and articles"
    )
    parser.add_argument(
        "--judgments",
        type=str,
        default="data/judgments_cleaned.json",
        help="Path to judgments JSON file",
    )
    parser.add_argument(
        "--legal-acts",
        type=str,
        default="data/legal_acts.json",
        help="Path to legal acts JSON file",
    )
    parser.add_argument(
        "--par-to-par",
        type=str,
        default="data/par-to-par-cleaned.csv",
        help="Path to paragraph-to-paragraph citations CSV file",
    )
    parser.add_argument(
        "--citations",
        type=str,
        default="data/par-to-par-og.csv",
        help="Path to paragraph-to-paragraph citations CSV file (for citation edges)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/preprocessed",
        help="Output directory for preprocessed data",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Name of sentence transformer encoder",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding",
    )
    args = parser.parse_args()

    preprocessor = EmbeddingPreprocessor(
        encoder_name=args.encoder,
        batch_size=args.batch_size,
    )

    # Process paragraphs
    print("\n" + "=" * 80)
    print("PROCESSING PARAGRAPHS")
    print("=" * 80)
    preprocessor.process_judgments(args.judgments, args.par_to_par, args.output_dir)

    # Process articles
    print("\n" + "=" * 80)
    print("PROCESSING ARTICLES")
    print("=" * 80)
    preprocessor.process_legal_acts(args.legal_acts, args.output_dir)

    # Process citations
    print("\n" + "=" * 80)
    print("PROCESSING CITATIONS")
    print("=" * 80)
    preprocessor.process_citations(args.judgments, args.citations, args.output_dir)

    print("\n" + "=" * 80)
    print("✓ PREPROCESSING COMPLETE")
    print("=" * 80)
    print(f"All data saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
