import pandas as pd  # type: ignore
import wandb
from tqdm import tqdm  # type: ignore
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.datasets import NoDuplicatesDataLoader
from validation_utils import (
    split_data_by_date,
    create_ir_evaluator,
    create_validation_examples,
)


def get_data(
    paragraph_file: str, cutoff_date: pd.Timestamp, validation_split: float = 0.1
) -> tuple[list[InputExample], list[InputExample], pd.DataFrame]:
    df = pd.read_csv(paragraph_file)
    df["DATE_FROM"] = pd.to_datetime(df["DATE_FROM"])
    df["DATE_TO"] = pd.to_datetime(df["DATE_TO"])
    df["FROM_ID"] = df["CELEX_FROM"].astype(str) + "::" + df["NUMBER_FROM"].astype(str)
    df["TO_ID"] = df["CELEX_TO"].astype(str) + "::" + df["NUMBER_TO"].astype(str)

    # Split into train and validation
    train_df, val_df = split_data_by_date(df, cutoff_date, validation_split)

    train_examples = []
    for _, row in tqdm(
        train_df.iterrows(), total=len(train_df), desc="Creating training InputExamples"
    ):
        text_from = row["TEXT_FROM"]
        text_to = row["TEXT_TO"]
        train_examples.append(InputExample(texts=[text_from, text_to]))

    val_examples = create_validation_examples(val_df)

    return train_examples, val_examples, val_df


def train_simcse(
    paragraph_file: str,
    cutoff_date: pd.Timestamp,
    model_name: str = "all-MiniLM-L6-v2",
    output_path: str = "output/simcse_citation_model",
    batch_size: int = 16,
    epochs: int = 5,
    warmup_steps: int = 100,
    checkpoint_save_steps: int = 1000,
    evaluation_steps: int = 1000,
    show_progress_bar: bool = True,
    validation_split: float = 0.1,
    use_wandb: bool = True,
    project_name: str = "simcse-citation-model",
) -> SentenceTransformer:

    # Initialize wandb
    if use_wandb:
        wandb.init(
            project=project_name,
            config={
                "model_name": model_name,
                "batch_size": batch_size,
                "epochs": epochs,
                "warmup_steps": warmup_steps,
                "validation_split": validation_split,
                "cutoff_date": str(cutoff_date),
            },
        )

    train_dataset, val_dataset, val_df = get_data(
        paragraph_file, cutoff_date, validation_split
    )
    train_dataloader = NoDuplicatesDataLoader(train_dataset, batch_size=batch_size)
    model = SentenceTransformer(model_name)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # Create validation evaluator if validation data exists
    evaluator = None
    if len(val_df) > 0:
        evaluator = create_ir_evaluator(val_df)

    print(f"\nTraining {model_name} with Supervised SimCSE (MNRL)...")
    print(f"Total training examples: {len(train_dataset)}")
    print(f"Total validation examples: {len(val_dataset)}")
    print(f"Total batches: {len(train_dataloader)}")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],  # type: ignore
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        scheduler="WarmupLinear",
        checkpoint_save_steps=checkpoint_save_steps,
        show_progress_bar=show_progress_bar,
        evaluator=evaluator,
        evaluation_steps=evaluation_steps,
        save_best_model=True,
        checkpoint_save_path=evaluation_steps,
    )

    # Save model to wandb
    if use_wandb:
        wandb.save(f"{output_path}/*")
        wandb.finish()

    print(f"Training finished. Model saved to {output_path}")
    return model


if __name__ == "__main__":
    train_simcse(
        paragraph_file="data/par-to-par.csv",
        cutoff_date=pd.Timestamp("2018-01-01"),
        model_name="all-mpnet-base-v2",
        output_path="artifacts/simcse_citation_model",
        validation_split=0.1,
        use_wandb=True,
        project_name="simcse-citation-model",
    )
