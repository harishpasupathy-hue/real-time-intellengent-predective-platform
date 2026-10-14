import sqlite3
from pathlib import Path

import pandas as pd


def save_to_sqlite(
    df: pd.DataFrame,
    db_path: str = "data/sentinelflow.db"
) -> str:

    path = Path(db_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with sqlite3.connect(path) as connection:

        df.to_sql(
            "events",
            connection,
            if_exists="replace",
            index=False
        )

    return str(path)


def read_from_sqlite(
    db_path: str = "data/sentinelflow.db"
) -> pd.DataFrame:

    with sqlite3.connect(db_path) as connection:

        return pd.read_sql_query(
            "SELECT * FROM events",
            connection
        )