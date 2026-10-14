from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = [
    "timestamp",
    "entity_id",
    "scenario",
    "event_type",
    "location",
    "value_1",
    "value_2",
    "value_3",
]


def clean_events(df: pd.DataFrame) -> pd.DataFrame:

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    data = df.copy()

    # Convert timestamp
    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["timestamp", "entity_id"]
    )

    # Convert numerical columns
    for column in [
        "value_1",
        "value_2",
        "value_3"
    ]:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

        data[column] = data[column].fillna(
            data[column].median()
        )

    # Clean categorical columns
    for column in [
        "scenario",
        "event_type",
        "location"
    ]:

        data[column] = (
            data[column]
            .fillna("Unknown")
            .astype(str)
        )

    data["entity_id"] = (
        data["entity_id"]
        .astype(str)
    )

    # Remove duplicate rows
    data = data.drop_duplicates()

    # Sort by time
    data = (
        data
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Create additional features
    data["hour"] = data["timestamp"].dt.hour

    data["day_of_week"] = (
        data["timestamp"].dt.dayofweek
    )

    return data


def save_processed_data(
    df: pd.DataFrame,
    path: str = "data/processed_events.csv"
) -> str:

    output = Path(path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output,
        index=False
    )

    return str(output)