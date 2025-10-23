import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.datasets import NoDuplicatesDataLoader


def get_data(paragraph_file: str, cutoff_date: pd.Timestamp) -> list[InputExample]:
    df = pd.read_csv(paragraph_file)
    df["DATE_FROM"] = pd.to_datetime(df["DATE_FROM"])
    df["DATE_TO"] = pd.to_datetime(df["DATE_TO"])
    df["FROM_ID"] = df["CELEX_FROM"].astype(str) + "::" + df["NUMBER_FROM"].astype(str)
    df["TO_ID"] = df["CELEX_TO"].astype(str) + "::" + df["NUMBER_TO"].astype(str)

    train_df = df[df["DATE_FROM"] < cutoff_date]

    # Get training paragraphs for TF-IDF
    train_paragraphs_from = train_df["TEXT_FROM"].dropna().unique().tolist()
    train_paragraphs_to = train_df["TEXT_TO"].dropna().unique().tolist()
    train_paragraphs = list(set(train_paragraphs_from + train_paragraphs_to))

    train_examples = []
    for index, row in tqdm(
        train_df.iterrows(), total=len(train_df), desc="Creating InputExamples"
    ):
        text_from = row["TEXT_FROM"]
        text_to = row["TEXT_TO"]
        train_examples.append(InputExample(texts=[text_from, text_to]))

    return train_examples


def train_simcse(
    paragraph_file: str,
    cutoff_date: pd.Timestamp,
    model_name: str = "all-MiniLM-L6-v2",
    output_path: str = "output/simcse_citation_model",
    batch_size: int = 16,
    epochs: int = 5,
    warmup_steps: int = 100,
    checkpoint_save_steps: int = 1000,
    show_progress_bar: bool = True,
) -> SentenceTransformer:

    train_dataset = get_data(paragraph_file, cutoff_date)
    train_dataloader = NoDuplicatesDataLoader(train_dataset, batch_size=batch_size)
    model = SentenceTransformer(model_name)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    print(f"\nTraining {model_name} with Supervised SimCSE (MNRL)...")
    print(f"Total batches: {len(train_dataloader)}")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        scheduler="WarmupLinear",
        checkpoint_save_steps=checkpoint_save_steps,
        show_progress_bar=show_progress_bar,
    )

    print(f"Training finished. Model saved to {output_path}")
    return model


if __name__ == "__main__":
    train_simcse(
        paragraph_file="data/par-to-par.csv",
        cutoff_date=pd.Timestamp("2018-01-01"),
        model_name="all-mpnet-base-v2",
        output_path="artifacts/simcse_citation_model",
    )
