import pandas as pd  # type: ignore
import json
from pydantic import BaseModel, Field


class Document(BaseModel):
    docno: str = Field(..., description="Document Number")
    text: str = Field(..., description="Text")


def get_document_id(celex_id: str, para_num: int) -> str:
    return f"{celex_id}::{para_num}"


def load_candidate_documents(
    cutoff_date: str, use_all_paragraphs: bool = False
) -> list[Document]:
    """Load candidate documents (either all paragraphs or just target paragraphs)."""
    cutoff = pd.Timestamp(cutoff_date)

    if use_all_paragraphs:
        return _load_all_paragraphs_before_cutoff(cutoff)
    else:
        return _load_target_paragraphs_before_cutoff(cutoff)


def _load_all_paragraphs_before_cutoff(cutoff: pd.Timestamp) -> list[Document]:
    """Load all paragraphs from judgments_cleaned.json before the cutoff date."""
    with open("data/judgments_cleaned.json", "r") as f:
        judgments = json.load(f)

    # Extract all paragraphs with their metadata
    all_paragraphs = []
    for judgment in judgments:
        celex_id = judgment["celex_id"]
        date = judgment.get("meta", {}).get("date", "")
        if date:
            try:
                judgment_date = pd.to_datetime(date)
                if judgment_date < cutoff:
                    for para_num, para_text in judgment["paragraphs"].items():
                        para_id = get_document_id(celex_id, para_num)
                        all_paragraphs.append(
                            Document(
                                docno=para_id,
                                text=para_text,
                            )
                        )
            except:
                # Skip judgments with invalid dates
                continue

    print(
        f"Loaded {len(all_paragraphs)} paragraphs from all judgments before cutoff date"
    )
    return all_paragraphs


def _load_target_paragraphs_before_cutoff(
    cutoff: pd.Timestamp,
) -> list[Document]:
    """Load only target paragraphs from par-to-par dataset before cutoff date."""

    df = pd.read_csv("data/par-to-par.csv").dropna()

    docs = (
        df[["CELEX_TO", "NUMBER_TO", "DATE_TO", "TEXT_TO", "TITLE_TO"]]
        .drop_duplicates(["CELEX_TO", "NUMBER_TO"])
        .copy()
    )
    return [
        Document(
            docno=get_document_id(doc["CELEX_TO"], doc["NUMBER_TO"]),
            text=doc["TEXT_TO"],
        )
        for doc in docs.to_dict(orient="records")
        if pd.to_datetime(doc["DATE_TO"]) < cutoff
    ]


def load_queries(df: pd.DataFrame, cutoff_date: str) -> pd.DataFrame:
    """Load queries (source paragraphs on or after cutoff date)."""
    cutoff = pd.Timestamp(cutoff_date)

    queries = (
        df[
            [
                "CELEX_FROM",
                "NUMBER_FROM",
                "DATE_FROM",
                "TEXT_FROM",
                "TITLE_FROM",
                "FROM_ID",
            ]
        ]
        .drop_duplicates("FROM_ID")
        .copy()
    )
    queries.rename(
        columns={
            "CELEX_FROM": "CELEX",
            "NUMBER_FROM": "PARA_NO",
            "DATE_FROM": "DATE",
            "TEXT_FROM": "TEXT",
            "TITLE_FROM": "TITLE",
            "FROM_ID": "QID",
        },
        inplace=True,
    )
    queries["DATE"] = pd.to_datetime(queries["DATE"])
    queries = queries.loc[queries["DATE"] >= cutoff].copy()

    # Create PyTerrier queries dataframe with required fields
    queries_df = pd.DataFrame(
        {
            "qid": queries["QID"],
            "query": queries["TEXT"],
        }
    )
    return queries_df


def load_relevance_judgments(df: pd.DataFrame, cutoff_date: str) -> pd.DataFrame:
    """Load relevance judgments (qrels) for queries after cutoff and documents before cutoff."""
    cutoff = pd.Timestamp(cutoff_date)

    # Filter to include only queries after cutoff and documents before cutoff
    df_filtered = df.loc[df["DATE_FROM"] >= cutoff].copy()

    qrels_df = pd.DataFrame(
        {
            "qid": df_filtered["FROM_ID"],
            "docno": df_filtered["TO_ID"],
            "label": 1,  # Binary relevance (1 = relevant)
        }
    )

    # Remove duplicate qid-docno pairs (if any)
    qrels_df = qrels_df.drop_duplicates(subset=["qid", "docno"])
    return qrels_df


def load_documents_with_context(cutoff_date: str) -> list[Document]:
    """Load documents with context (previous and next paragraphs) for BM25 context retriever."""
    cutoff = pd.Timestamp(cutoff_date)

    with open("data/judgments_cleaned.json", "r") as f:
        judgments = json.load(f)

    # Create lookup for faster access
    judgments_lookup = {judgment["celex_id"]: judgment for judgment in judgments}

    all_paragraphs = []
    for judgment in judgments:
        celex_id = judgment["celex_id"]
        date = judgment.get("meta", {}).get("date", "")
        if date:
            try:
                judgment_date = pd.to_datetime(date)
                if judgment_date < cutoff:
                    paragraphs = judgment["paragraphs"]
                    for para_num_str, para_text in paragraphs.items():
                        para_num = int(para_num_str)
                        para_id = get_document_id(celex_id, para_num)

                        # Get context paragraphs
                        prev_para = paragraphs.get(str(para_num - 1))
                        next_para = paragraphs.get(str(para_num + 1))

                        # Concatenate with context using [SEP] token
                        parts = []
                        if prev_para:
                            parts.append(prev_para)
                        parts.append(para_text)
                        if next_para:
                            parts.append(next_para)

                        enhanced_text = " [SEP] ".join(parts)

                        all_paragraphs.append(
                            Document(
                                docno=para_id,
                                text=enhanced_text,
                            )
                        )
            except:
                # Skip judgments with invalid dates
                continue

    print(
        f"Loaded {len(all_paragraphs)} paragraphs with context from judgments before cutoff date"
    )
    return all_paragraphs


def load_and_prepare_data(
    csv_path: str, cutoff_date: str, use_all_paragraphs: bool = False
) -> tuple[list[Document], pd.DataFrame, pd.DataFrame]:
    """Load and prepare data for evaluation using the specified mode."""
    # Load and preprocess the main dataset
    df = pd.read_csv(csv_path)
    df["DATE_FROM"] = pd.to_datetime(df["DATE_FROM"])
    df["DATE_TO"] = pd.to_datetime(df["DATE_TO"])
    df["FROM_ID"] = df["CELEX_FROM"].astype(str) + "::" + df["NUMBER_FROM"].astype(str)
    df["TO_ID"] = df["CELEX_TO"].astype(str) + "::" + df["NUMBER_TO"].astype(str)

    # Load candidate documents
    docs = load_candidate_documents(cutoff_date, use_all_paragraphs)
    # Load queries
    queries_df = load_queries(df, cutoff_date)

    # Load relevance judgments
    qrels_df = load_relevance_judgments(df, cutoff_date)

    print(f"Queries: {len(queries_df)}")
    print(f"Qrels (relevance judgments): {len(qrels_df)}")
    print(f"Unique queries with judgments: {qrels_df['qid'].nunique()}")

    return docs, queries_df, qrels_df
