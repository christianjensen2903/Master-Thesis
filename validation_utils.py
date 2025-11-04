import pandas as pd  # type: ignore


def split_data_by_date(
    df: pd.DataFrame,
    cutoff_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df["DATE_FROM"] = pd.to_datetime(df["DATE_FROM"])
    cutoff_date = pd.Timestamp(year=cutoff_year, month=1, day=1)

    train_df = df[df["DATE_FROM"] < cutoff_date].copy()
    val_df = df[df["DATE_FROM"] >= cutoff_date].copy()

    print(f"\n📅 Temporal Split:")
    print(f"  Train: before {cutoff_date.date()} ({len(train_df)} citations)")
    print(f"  Val: after {cutoff_date.date()} ({len(val_df)} citations)")

    return train_df, val_df
