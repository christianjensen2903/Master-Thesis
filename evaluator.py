from typing import Any

import pandas as pd  # type: ignore
from tqdm import tqdm  # type: ignore
from retrievers import BaseRetriever


class Evaluator:

    def __init__(
        self, k_values: list[int] | None = None, show_progress: bool = True
    ) -> None:

        self.k_values = k_values if k_values is not None else [5, 10, 50, 100]
        self.show_progress = show_progress

    @staticmethod
    def _average_precision_at_k(
        ranked_ids: list[str], relevant: set[str], k: int
    ) -> float:
        if not relevant:
            return 0.0
        hits = 0
        precisions: list[float] = []
        for i, rid in enumerate(ranked_ids[:k], start=1):
            if rid in relevant:
                hits += 1
                precisions.append(hits / i)
        return (sum(precisions) / len(relevant)) if precisions else 0.0

    def evaluate(
        self,
        retriever: BaseRetriever,
        queries: pd.DataFrame,
        relevance_map: dict[str, set[str]],
    ) -> pd.DataFrame:

        # Build the list of evaluable queries (must have at least one relevant)
        eval_records: list[dict[str, Any]] = []
        for _, q in queries.iterrows():
            qid: str = q["QID"]
            relevant: set[str] = relevance_map.get(qid, set())
            if not relevant:
                continue

            # Build record with required fields and optional metadata
            record: dict[str, Any] = {
                "qid": qid,
                "qtext": str(q["TEXT"]),
                "relevant": relevant,
            }

            # Optionally include other metadata columns if present
            if "CELEX" in q.index:
                record["qcelex"] = str(q["CELEX"])
            if "DATE" in q.index:
                record["qdate"] = q["DATE"]

            eval_records.append(record)

        if not eval_records:
            # Return empty summary if no evaluable queries
            return pd.DataFrame(
                {
                    "k": self.k_values,
                    "Precision@k": 0.0,
                    "Recall@k": 0.0,
                    "F1@k": 0.0,
                    "nDCG@k": 0.0,
                    "MAP@k": 0.0,
                    "Avg #Relevant": 0.0,
                    "Queries evaluated": 0,
                }
            )

        # Batch retrieve rankings for all queries
        q_texts = [r["qtext"] for r in eval_records]
        max_k = max(self.k_values) if len(self.k_values) > 0 else 0
        batch_ranked_docs = retriever.get_relevant_documents_batch(q_texts, k=max_k)

        # Compute metrics for each query and k value
        rows: list[dict[str, Any]] = []
        num_used = len(eval_records)

        iterator = zip(eval_records, batch_ranked_docs)
        if self.show_progress:
            iterator = tqdm(
                iterator,
                desc="Evaluating",
                total=len(eval_records),
                dynamic_ncols=True,
                leave=False,
            )

        for rec, ranked_docs in iterator:
            ranked_ids = [doc.metadata["id"] for doc in ranked_docs]
            relevant = rec["relevant"]

            for k in self.k_values:
                top_k_ids = ranked_ids[:k]
                hits = sum(1 for rid in top_k_ids if rid in relevant)
                prec = hits / k
                rec_k = hits / len(relevant)
                f1_k = (
                    (2 * prec * rec_k / (prec + rec_k)) if (prec + rec_k) > 0 else 0.0
                )
                ap_k = self._average_precision_at_k(ranked_ids, relevant, k)

                row: dict[str, Any] = {
                    "QID": rec["qid"],
                    "k": k,
                    "Precision@k": prec,
                    "Recall@k": rec_k,
                    "F1@k": f1_k,
                    "AP@k": ap_k,
                    "num_relevant": len(relevant),
                }

                # Include optional metadata
                if "qdate" in rec:
                    row["query_date"] = rec["qdate"]
                if "qcelex" in rec:
                    row["query_celex"] = rec["qcelex"]

                rows.append(row)

        # Aggregate results across queries
        df_rows = pd.DataFrame(rows)
        summary = (
            df_rows.groupby("k")
            .agg(
                {
                    "Precision@k": "mean",
                    "Recall@k": "mean",
                    "F1@k": "mean",
                    "AP@k": "mean",
                    "num_relevant": "mean",
                }
            )
            .reset_index()
            .rename(
                columns={
                    "num_relevant": "Avg #Relevant",
                    "AP@k": "MAP@k",
                }
            )
        )
        summary["Queries evaluated"] = num_used

        return summary
