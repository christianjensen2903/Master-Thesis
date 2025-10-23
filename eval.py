import pandas as pd  # type: ignore
import pyterrier as pt  # type: ignore
from utils import load_and_prepare_data
from retrievers import DenseRetriever


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

    csv_path = "data/par-to-par.csv"
    index_path = "./artifacts/index/data.properties"
    cutoff_date = "2018-01-01"
    k_values = [5, 10, 50, 100]
    use_all_paragraphs = (
        True  # Set to True to use all paragraphs before cutoff as candidates
    )

    print("=" * 80)
    print("PyTerrier Retrieval Evaluation")
    print("=" * 80)
    if use_all_paragraphs:
        print("Mode: Using ALL paragraphs before cutoff date as candidates")
    else:
        print(
            "Mode: Using only target paragraphs from par-to-par dataset as candidates"
        )
    print("=" * 80)

    _, queries_df, qrels_df = load_and_prepare_data(
        csv_path, cutoff_date, use_all_paragraphs
    )

    # BM25 retriever
    index = pt.IndexFactory.of(index_path)
    bm25 = pt.rewrite.tokenise("utf") >> pt.terrier.Retriever(
        index, wmodel="BM25", verbose=True
    )

    # # Dense retrievers
    # sbert = DenseRetriever(
    #     documents_df=documents_df,
    #     model_name="sentence-transformers/all-MiniLM-L6-v2",
    #     use_gpu=False,
    # )

    # legalbert = DenseRetriever(
    #     documents_df=documents_df,
    #     model_name="nlpaueb/legal-bert-base-uncased",
    #     use_gpu=False,
    # )

    # models = [bm25, sbert, legalbert]
    # model_names = ["BM25", "SBERT", "LegalBERT"]
    models = [bm25]
    model_names = ["BM25"]

    results = evaluate(models, model_names, queries_df, qrels_df, k_values)  # type: ignore

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(results.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
