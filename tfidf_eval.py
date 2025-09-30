from typing import Any

import pandas as pd  # type: ignore
from tqdm import tqdm  # type: ignore
from langchain_core.documents import Document
from retrievers import TFIDFRetriever, BaseRetriever, BM25Retriever, preprocess_utils
from nltk.corpus import stopwords  # type: ignore
import nltk

nltk.download("stopwords")


def build_candidate_pool(df: pd.DataFrame, cutoff_date: pd.Timestamp) -> list[Document]:
    """Build the candidate pool of unique target paragraphs strictly before a cutoff date."""
    cands = (
        df[["CELEX_TO", "NUMBER_TO", "DATE_TO", "TEXT_TO", "TITLE_TO", "TO_ID"]]
        .drop_duplicates("TO_ID")
        .copy()
    )
    cands.rename(
        columns={
            "CELEX_TO": "CELEX",
            "NUMBER_TO": "PARA_NO",
            "DATE_TO": "DATE",
            "TEXT_TO": "TEXT",
            "TITLE_TO": "TITLE",
        },
        inplace=True,
    )
    cands["DATE"] = pd.to_datetime(cands["DATE"])
    cands = cands.loc[cands["DATE"] < cutoff_date].copy()
    return [
        Document(
            page_content=row["TEXT"],
            metadata={"id": row["TO_ID"], "celex": row["CELEX"], "date": row["DATE"]},
        )
        for _, row in cands.reset_index(drop=True).iterrows()
    ]


def build_queries(df: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    """Build the unique query set from FROM-side paragraphs."""
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
    queries = queries.loc[queries["DATE"] >= cutoff_date].copy()
    return queries.reset_index(drop=True)


def build_rel_map(df: pd.DataFrame) -> dict[str, set[str]]:
    """Ground truth mapping from query id to the set of relevant target ids."""
    return df.groupby("FROM_ID")["TO_ID"].apply(lambda s: set(s.astype(str))).to_dict()


def average_precision_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    """Compute Average Precision at k for a ranked list."""
    if not relevant:
        return 0.0
    hits = 0
    precisions: list[float] = []
    for i, rid in enumerate(ranked_ids[:k], start=1):
        if rid in relevant:
            hits += 1
            precisions.append(hits / i)
    return (sum(precisions) / len(relevant)) if precisions else 0.0


def eval_retriever(
    retriever: BaseRetriever,
    queries: pd.DataFrame,
    rel_map: dict[str, set[str]],
    k_list: list[int],
    cutoff_date: pd.Timestamp,
    cands: list[Document],
) -> pd.DataFrame:
    cutoff_date = pd.to_datetime(cutoff_date)

    # Allowed targets (strictly pre-cutoff)
    allowed_ids: set[str] = set(doc.metadata["id"] for doc in cands)

    eval_mask = queries["DATE"] >= cutoff_date
    q_eval = queries.loc[eval_mask].copy()

    # Build the list of evaluable queries (must have at least one allowed relevant)
    eval_records: list[dict[str, Any]] = []
    for _, q in q_eval.iterrows():
        qid: str = q["QID"]
        relevant_all: set[str] = rel_map.get(qid, set())
        relevant_allowed: set[str] = {rid for rid in relevant_all if rid in allowed_ids}
        if not relevant_allowed:
            continue
        eval_records.append(
            {
                "qid": qid,
                "qtext": str(q["TEXT"]),
                "qcelex": str(q["CELEX"]),
                "qdate": q["DATE"],
                "relevant_allowed": relevant_allowed,
            }
        )

    if not eval_records:
        return pd.DataFrame(
            {
                "k": k_list,
                "Precision@k": 0.0,
                "Recall@k": 0.0,
                "Avg #Relevant": 0.0,
                "Queries evaluated": 0,
            }
        )

    # Batch retrieve a full ranking for each query
    q_texts = [r["qtext"] for r in eval_records]
    max_k = max(k_list) if len(k_list) > 0 else 0
    batch_ranked_docs = retriever.get_relevant_documents_batch(q_texts, k=max_k)

    rows: list[dict[str, Any]] = []
    num_used = len(eval_records)
    for rec, ranked_docs in tqdm(
        zip(eval_records, batch_ranked_docs), desc="Evaluating", total=len(eval_records)
    ):
        ranked_ids_raw = [doc.metadata["id"] for doc in ranked_docs]
        ranked_ids = [rid for rid in ranked_ids_raw if rid in allowed_ids]
        relevant_allowed = rec["relevant_allowed"]

        for k in k_list:
            top_k_ids = ranked_ids[:k]
            hits = sum(1 for rid in top_k_ids if rid in relevant_allowed)
            prec = hits / k
            rec_k = hits / len(relevant_allowed)
            rows.append(
                {
                    "QID": rec["qid"],
                    "k": k,
                    "Precision@k": prec,
                    "Recall@k": rec_k,
                    "num_relevant": len(relevant_allowed),
                    "query_date": rec["qdate"],
                    "query_celex": rec["qcelex"],
                }
            )

    df_rows = pd.DataFrame(rows)
    summary = (
        df_rows.groupby("k")
        .agg(
            {
                "Precision@k": "mean",
                "Recall@k": "mean",
                "num_relevant": "mean",
            }
        )
        .reset_index()
        .rename(
            columns={
                "num_relevant": "Avg #Relevant",
            }
        )
    )
    summary["Queries evaluated"] = num_used
    return summary


def main() -> None:
    """Run TF-IDF retrieval evaluation with temporal cutoff and report summary metrics."""

    cutoff_date = pd.Timestamp("2018-01-01")  # pre-2018 candidates only
    k_list = [10, 50, 100]
    sample: int | None = None  # e.g., 500 for a quick run

    df = pd.read_csv("data/clean_data.csv")

    # Normalize types / IDs
    df["DATE_FROM"] = pd.to_datetime(df["DATE_FROM"])
    df["DATE_TO"] = pd.to_datetime(df["DATE_TO"])
    df["FROM_ID"] = df["CELEX_FROM"].astype(str) + "::" + df["NUMBER_FROM"].astype(str)
    df["TO_ID"] = df["CELEX_TO"].astype(str) + "::" + df["NUMBER_TO"].astype(str)

    # Build pools
    cands = build_candidate_pool(df, cutoff_date=cutoff_date)  # strictly pre-cutoff
    queries = build_queries(df, cutoff_date=cutoff_date)
    print(f"Candidates: {len(cands)}, Queries: {len(queries)}")

    if sample and sample > 0:
        mask_cut = queries["DATE"] >= cutoff_date
        qcut = queries.loc[mask_cut]
        queries = pd.concat(
            [
                queries.loc[~mask_cut],
                qcut.sample(n=min(sample, len(qcut)), random_state=42),
            ]
        ).reset_index(drop=True)
        print(f"Downsampled post-cutoff queries to {queries.loc[mask_cut].shape[0]}")

    rel_map = build_rel_map(df)

    retriever = BM25Retriever(
        documents=cands,
        preprocess=preprocess_utils.compose(
            preprocess_utils.lowercase(),
            preprocess_utils.remove_punctuation(),
            preprocess_utils.stopword_filter(stopwords=set(stopwords.words("english"))),
        ),
    )

    summary = eval_retriever(
        retriever=retriever,
        queries=queries,
        rel_map=rel_map,
        k_list=k_list,
        cutoff_date=cutoff_date,
        cands=cands,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
