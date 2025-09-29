import pandas as pd
from tqdm import tqdm
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import TFIDFRetriever
from langchain_core.documents import Document


def build_candidate_pool(df: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    """
    Returns unique TO paragraphs. If cutoff_date is provided, only keep candidates strictly before it.
    """
    cands = (
        df[["CELEX_TO", "NUMBER_TO", "DATE_TO", "TEXT_TO", "TITLE_TO", "to_id"]]
        .drop_duplicates("to_id")
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
    return cands.reset_index(drop=True)


def build_queries(df: pd.DataFrame) -> pd.DataFrame:
    queries = (
        df[
            [
                "CELEX_FROM",
                "NUMBER_FROM",
                "DATE_FROM",
                "TEXT_FROM",
                "TITLE_FROM",
                "from_id",
            ]
        ]
        .drop_duplicates("from_id")
        .copy()
    )
    queries.rename(
        columns={
            "CELEX_FROM": "CELEX",
            "NUMBER_FROM": "PARA_NO",
            "DATE_FROM": "DATE",
            "TEXT_FROM": "TEXT",
            "TITLE_FROM": "TITLE",
            "from_id": "QID",
        },
        inplace=True,
    )
    queries["DATE"] = pd.to_datetime(queries["DATE"])
    return queries.reset_index(drop=True)


def build_rel_map(df: pd.DataFrame) -> dict[str, set[str]]:
    """Ground truth: FROM para -> set(TO para ids)."""
    return df.groupby("from_id")["to_id"].apply(lambda s: set(s.astype(str))).to_dict()


def average_precision_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = 0
    precisions = []
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
    cands: pd.DataFrame,
) -> pd.DataFrame:
    """
    - Evaluates only queries ON/AFTER the cutoff_date (typical temporal eval setup).
    - Candidate pool is STRICTLY pre-cutoff (already enforced in `cands`).
    - Citations (ground truth) to post-cutoff targets are ignored automatically because they’re not in the pool.
    - Also excludes same-CELEX targets to prevent trivial self-matches.
    """
    cutoff_date = pd.to_datetime(cutoff_date)

    # Allowed targets (strictly pre-cutoff)
    allowed_ids: set[str] = set(cands["to_id"].astype(str))

    eval_mask = queries["DATE"] >= cutoff_date
    q_eval = queries.loc[eval_mask].copy()

    rows = []
    num_used = 0

    for _, q in tqdm(q_eval.iterrows(), total=len(q_eval)):
        qid: str = q["QID"]
        qcelex: str = q["CELEX"]
        qtext: str = q["TEXT"]

        relevant_all: set[str] = rel_map.get(qid, set())
        relevant_allowed: set[str] = {rid for rid in relevant_all if rid in allowed_ids}

        if not relevant_allowed:
            # No usable ground truth under the pre-cutoff constraint → skip
            continue

        num_used += 1

        ranked_docs = retriever.invoke(qtext)
        ranked_ids = [doc.metadata["id"] for doc in ranked_docs]

        for k in k_list:
            hits = sum(1 for rid in ranked_ids if rid in relevant_allowed)
            prec = hits / k
            rec = hits / len(relevant_allowed)
            ap = average_precision_at_k(ranked_ids, relevant_allowed, k)
            rows.append(
                {
                    "QID": qid,
                    "k": k,
                    "Precision@k": prec,
                    "Recall@k": rec,
                    "AP@k": ap,
                    "num_relevant": len(relevant_allowed),
                    "query_date": q["DATE"],
                    "query_celex": qcelex,
                }
            )

    if not rows:
        return pd.DataFrame(
            [],
            columns=[
                "k",
                "Precision@k",
                "Recall@k",
                "MAP@k",
                "Avg #Relevant",
                "Queries evaluated",
            ],
        )

    df_rows = pd.DataFrame(rows)
    summary = (
        df_rows.groupby("k")
        .agg(
            {
                "Precision@k": "mean",
                "Recall@k": "mean",
                "AP@k": "mean",
                "num_relevant": "mean",
            }
        )
        .reset_index()
        .rename(columns={"AP@k": "MAP@k", "num_relevant": "Avg #Relevant"})
    )
    summary["Queries evaluated"] = num_used
    return summary


def main():
    cutoff_date = pd.Timestamp("2018-01-01")  # pre-2018 candidates only
    k_list = [10, 50, 100]
    sample: Optional[int] = None  # e.g., 500 for a quick run

    df = pd.read_csv("data/clean_data.csv")

    # Normalize types / IDs
    df["DATE_FROM"] = pd.to_datetime(df["DATE_FROM"])
    df["DATE_TO"] = pd.to_datetime(df["DATE_TO"])
    df["from_id"] = df["CELEX_FROM"].astype(str) + "::" + df["NUMBER_FROM"].astype(str)
    df["to_id"] = df["CELEX_TO"].astype(str) + "::" + df["NUMBER_TO"].astype(str)

    # Build pools
    cands = build_candidate_pool(df, cutoff_date=cutoff_date)  # STRICTLY pre-2018
    queries = build_queries(df)  # all queries
    if sample and sample > 0:
        mask_cut = queries["DATE"] >= cutoff_date
        qcut = queries.loc[mask_cut]
        queries = pd.concat(
            [
                queries.loc[~mask_cut],
                qcut.sample(n=min(sample, len(qcut)), random_state=42),
            ]
        ).reset_index(drop=True)

    rel_map = build_rel_map(df)

    retriever = TFIDFRetriever.from_documents(
        documents=[
            Document(
                page_content=text, metadata={"id": id, "celex": celex, "date": date}
            )
            for text, id, celex, date in zip(
                cands["TEXT"].tolist(),
                cands["to_id"].tolist(),
                cands["CELEX"].astype(str).tolist(),
                pd.to_datetime(cands["DATE"]).tolist(),
            )
        ]
    )
    retriever.k = max(k_list)

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
