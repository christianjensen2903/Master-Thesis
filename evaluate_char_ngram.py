import numpy as np
from data_loader import load_citation_data, split_train_test, build_paragraph_index
from retrievers import CharNgramRetriever
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

    # Test different character n-gram ranges
    configs = [
        ((5, 5), "5-char", 2),
        ((10, 10), "10-char", 2),
        ((5, 10), "5-10 char", 2),
        ((10, 15), "10-15 char", 2),
    ]

    for ngram_range, desc, min_df in configs:
        print(f"\n{'='*60}")
        print(f"Testing CharNgramRetriever with {desc} n-grams")
        print("=" * 60)

        retriever = CharNgramRetriever(ngram_range=ngram_range, min_df=min_df)

        # Fit on training texts
        train_mask = paragraph_set == "train"
        print("Fitting vectorizer...")
        embeddings = retriever.fit_transform(pid_to_text, train_mask)
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Vocabulary size: {len(retriever.vectorizer.vocabulary_)}")

        # Run evaluation
        print("Running evaluation...")
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
