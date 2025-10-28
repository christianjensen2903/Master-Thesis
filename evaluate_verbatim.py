import numpy as np
from data_loader import load_citation_data, split_train_test, build_paragraph_index
from retrievers import VerbatimRetriever
from evaluator import Evaluator


def main():
    print("Loading data...")
    df, metadata = load_citation_data(csv_path="data/par-to-par-cleaned.csv")
    train_meta, test_meta = split_train_test(metadata, cutoff_year=2018)

    print("Building paragraph index...")
    pid_to_text, text_to_pid, paragraph_dates, paragraph_celex, paragraph_set = (
        build_paragraph_index(df, train_meta, test_meta)
    )

    print(f"Total paragraphs: {len(pid_to_text)}")
    print(f"Train paragraphs: {np.sum(paragraph_set == 'train')}")
    print(f"Test paragraphs: {np.sum(paragraph_set == 'test')}")

    # Initialize verbatim retriever with different n-gram ranges
    ngram_configs = [
        ((3, 3), "3-grams"),
        ((4, 4), "4-grams"),
        ((3, 5), "3-5 grams"),
        ((5, 7), "5-7 grams"),
    ]

    for ngram_range, desc in ngram_configs:
        print(f"\n{'='*60}")
        print(f"Testing VerbatimRetriever with {desc}")
        print("=" * 60)

        retriever = VerbatimRetriever(ngram_range=ngram_range, min_df=1)

        # Fit on all texts (verbatim matching doesn't need training)
        train_mask = paragraph_set == "train"
        embeddings = retriever.fit_transform(pid_to_text, train_mask)
        print(f"Embeddings shape: {embeddings.shape}")

        # Run evaluation
        evaluator = Evaluator(
            retriever=retriever,
            embeddings=embeddings,
            top_k=10000,
            csv_path="data/par-to-par-cleaned.csv",
        )
        score = evaluator.run()

        print(f"\nFinal MAP@10000 ({desc}): {score:.4f}")


if __name__ == "__main__":
    main()
