import sqlite3
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SQLiteDB:
    def __init__(self, db_path="src/db/ticker.db"):
        self.conn = sqlite3.connect(db_path)

    def save_dataframe(self, df: pd.DataFrame, table_name: str):
        df.to_sql(table_name, self.conn, if_exists="replace", index=False)
        logger.info(f"Data saved to table: {table_name}")

    def load_dataframe(self, table_name: str):
        return pd.read_sql(f"SELECT * FROM {table_name}", self.conn)
