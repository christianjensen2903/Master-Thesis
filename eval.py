import pandas as pd  # type: ignore
import pyterrier as pt  # type: ignore
from utils import load_and_prepare_data
from retrievers import HuggingFaceRetriever


def evaluate(
    models: list[pt.Transformer],
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
    print("PyTerrier Retrieval Evaluation")
    print("=" * 80)

    documents_df, queries_df, qrels_df = load_and_prepare_data(csv_path, cutoff_date)

    # BM25 retriever
    # index = pt.IndexFactory.of(index_path)
    # bm25 = pt.rewrite.tokenise("utf") >> pt.terrier.Retriever(
    #     index, wmodel="BM25", verbose=True
    # )

    # # Dense retrievers
    # sbert = DenseRetriever(
    #     documents_df=documents_df,
    #     model_name="sentence-transformers/all-MiniLM-L6-v2",
    #     use_gpu=False,
    # )

    legalbert = HuggingFaceRetriever(
        documents_df=documents_df,
        model_name="nlpaueb/legal-bert-base-uncased",
        use_gpu=False,
    )

    # models = [bm25, sbert, legalbert]
    # model_names = ["BM25", "SBERT", "LegalBERT"]
    models = [legalbert]
    model_names = ["LegalBERT"]

    results = evaluate(models, model_names, queries_df, qrels_df, k_values)  # type: ignore

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(results.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
