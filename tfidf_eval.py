import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from tqdm import tqdm


class TimeFilteredTFIDFRetriever:
    def __init__(
        self,
        texts: list[str],
        ids: list[str],
        celex: list[str],
        dates: list[pd.Timestamp],
        lowercase=True,
    ):
        self.texts = [t if isinstance(t, str) else "" for t in texts]
        self.ids = np.array(ids)
        self.celex = np.array(celex)
        self.dates = np.array(pd.to_datetime(dates))
        self.vectorizer = TfidfVectorizer(lowercase=lowercase)
        self.X = self.vectorizer.fit_transform(self.texts)

    def _mask_for_query(self, q_celex: str, q_date: pd.Timestamp) -> np.ndarray:
        return (self.dates <= q_date) & (self.celex != q_celex)

    def get_topk(
        self, query_text: str, q_celex: str, q_date: pd.Timestamp, k: int = 10
    ) -> list[str]:
        qv = self.vectorizer.transform(
            [query_text if isinstance(query_text, str) else ""]
        )
        scores = linear_kernel(qv, self.X).ravel()
        mask = self._mask_for_query(q_celex, q_date)
        scores[~mask] = -1.0
        order = np.argsort(-scores)
        return self.ids[order][:k].tolist()


def build_candidate_pool(df: pd.DataFrame) -> pd.DataFrame:
    # Unique TO paragraphs
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


def average_precision_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    if len(relevant) == 0:
        return 0.0
    hits = 0
    precisions = []
    for i, rid in enumerate(ranked_ids[:k], start=1):
        if rid in relevant:
            hits += 1
            precisions.append(hits / i)
    return sum(precisions) / len(relevant) if precisions else 0.0


def eval_retriever(
    retriever: TimeFilteredTFIDFRetriever,
    queries: pd.DataFrame,
    rel_map: dict[str, set[str]],
    k_list: list[int],
    cutoff_date: pd.Timestamp,
) -> pd.DataFrame:
    eval_mask = queries["DATE"] >= cutoff_date
    q_eval = queries.loc[eval_mask].copy()

    rows = []
    num_used = 0

    for _, q in tqdm(q_eval.iterrows(), total=len(q_eval)):
        qid = q["QID"]
        qdate = q["DATE"]
        qcelex = q["CELEX"]
        qtext = q["TEXT"] if isinstance(q["TEXT"], str) else ""

        # Filter relevant to those available (<= date and not same celex)
        mask = retriever._mask_for_query(qcelex, qdate)
        allowed_ids = set(retriever.ids[mask].tolist())
        relevant_all = rel_map.get(qid, set())
        relevant_allowed = relevant_all & allowed_ids

        if len(relevant_allowed) == 0:
            print(f"No relevant to_ids for query {qid}")
            continue

        num_used += 1

        # Rank once (with large K to cover max asked k)
        max_k = max(k_list)
        ranked = retriever.get_topk(qtext, qcelex, qdate, k=max_k)

        for k in k_list:
            topk = ranked[:k]
            num_hits = sum(1 for rid in topk if rid in relevant_allowed)
            prec = num_hits / k
            rec = num_hits / len(relevant_allowed)
            ap = average_precision_at_k(ranked, relevant_allowed, k)
            rows.append(
                {
                    "QID": qid,
                    "k": k,
                    "Precision@k": prec,
                    "Recall@k": rec,
                    "AP@k": ap,
                    "num_relevant": len(relevant_allowed),
                    "query_date": qdate,
                    "query_celex": qcelex,
                }
            )

    if not rows:
        return pd.DataFrame([], columns=["k", "Precision@k", "Recall@k", "AP@k"])

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
    cutoff_date = "2018-01-01"
    k_list = [10, 50, 100]
    sample = None

    df = pd.read_csv("data/clean_data.csv")
    # df = pd.read_excel("data/par-to-par-2.xlsx")
    # df = df.dropna()
    # Build IDs
    df["DATE_FROM"] = pd.to_datetime(df["DATE_FROM"])
    df["DATE_TO"] = pd.to_datetime(df["DATE_TO"])
    df["from_id"] = df["CELEX_FROM"].astype(str) + "::" + df["NUMBER_FROM"].astype(str)
    df["to_id"] = df["CELEX_TO"].astype(str) + "::" + df["NUMBER_TO"].astype(str)

    # Candidate paragraphs (TO)
    cands = build_candidate_pool(df)
    # Query paragraphs (FROM)
    queries = build_queries(df)

    if sample and sample > 0:
        # Apply sampling AFTER cutoff filtering to ensure the sample is from the evaluation subset
        mask_cut = queries["DATE"] >= cutoff_date
        qcut = queries.loc[mask_cut]
        queries = pd.concat(
            [
                queries.loc[~mask_cut],
                qcut.sample(n=min(sample, len(qcut)), random_state=42),
            ]
        ).reset_index(drop=True)

    # Ground truth map: query -> set of relevant to_ids
    rel_map = (
        df.groupby("from_id")["to_id"].apply(lambda s: set(s.astype(str))).to_dict()
    )

    # Build retriever
    retriever = TimeFilteredTFIDFRetriever(
        texts=cands["TEXT"].fillna("").tolist(),
        ids=cands["to_id"].tolist(),
        celex=cands["CELEX"].astype(str).tolist(),
        dates=pd.to_datetime(cands["DATE"]).tolist(),
    )

    summary = eval_retriever(retriever, queries, rel_map, k_list, cutoff_date)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
