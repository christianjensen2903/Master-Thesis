import pandas as pd  # type: ignore
import pyterrier as pt  # type: ignore
from utils import load_and_prepare_data


def evaluate(
    models: list[pt.terrier.Retriever],
    model_names: list[str],
    queries_df: pd.DataFrame,
    qrels_df: pd.DataFrame,
    k_values: list[int] = [],
) -> pd.DataFrame:

    metrics = ["map"]
    for k in k_values:
        metrics.extend([f"P_{k}", f"recall_{k}"])

    results = pt.Experiment(
        models,
        queries_df,
        qrels_df,
        metrics,
        names=model_names,
    )

    return results


def main() -> None:

    csv_path = "data/clean_data.csv"
    index_path = "./artifacts/pyterrier_index/data.properties"
    cutoff_date = "2018-01-01"
    k_values = [5, 10, 50, 100]

    print("=" * 80)
    print("PyTerrier BM25 Evaluation")
    print("=" * 80)

    documents_df, queries_df, qrels_df = load_and_prepare_data(csv_path, cutoff_date)

    index = pt.IndexFactory.of(index_path)
    bm25 = pt.rewrite.tokenise("utf") >> pt.terrier.Retriever(
        index, wmodel="BM25", verbose=True
    )

    models = [bm25]
    model_names = ["BM25"]

    results = evaluate(models, model_names, queries_df, qrels_df, k_values)

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(results.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
