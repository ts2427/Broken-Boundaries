"""
etl.py - Extract, Transform, Load Pipeline
Loads cleaned/enriched breach data into a SQLite database.
Uses DATA_DICTIONARY from clean.py for schema definitions.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional

from clean import (
    DATA_DICTIONARY,
    get_schema,
    get_sql_types,
    generate_create_table_sql,
)


# === Configuration ===
DATA_DIR = Path(__file__).parent / "cleaned"
DB_DIR = Path(__file__).parent / "database"
DB_NAME = "breach_data.db"


# =============================================================================
# EXTRACT
# =============================================================================

def extract_from_csv(filename: str = "breach_data_enriched.csv") -> pd.DataFrame:
    """
    Extract data from the enriched CSV file.
    """
    filepath = DATA_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    print(f"=== EXTRACT ===")
    print(f"Source: {filepath}")

    df = pd.read_csv(filepath)
    print(f"Extracted {len(df)} records with {len(df.columns)} columns")

    return df


# =============================================================================
# TRANSFORM
# =============================================================================

def transform_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert date columns to proper datetime format for SQLite."""
    date_columns = ['reported_date', 'breach_date', 'end_breach_date']

    for col in date_columns:
        if col in df.columns:
            # Convert to datetime then to string format SQLite understands
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[col] = df[col].dt.strftime('%Y-%m-%d')
            df[col] = df[col].replace('NaT', None)

    return df


def transform_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure numeric columns are properly typed."""
    schema = get_schema()
    columns = schema.get("columns", {})

    for col_name, col_info in columns.items():
        if col_name not in df.columns:
            continue

        sql_type = col_info.get("sql_type", "")

        if sql_type in ["INTEGER", "BIGINT"]:
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
            # Replace NaN with None for SQLite
            df[col_name] = df[col_name].where(pd.notna(df[col_name]), None)
        elif "DECIMAL" in sql_type or sql_type == "FLOAT":
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
            df[col_name] = df[col_name].where(pd.notna(df[col_name]), None)

    return df


def transform_text(df: pd.DataFrame) -> pd.DataFrame:
    """Clean text columns - handle NaN values."""
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].replace({np.nan: None, 'nan': None, 'NaN': None})

    return df


def transform_add_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Add ETL metadata columns."""
    df['_etl_loaded_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df['_etl_source'] = 'breach_data_enriched.csv'
    return df


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all transformations on the data.
    """
    print(f"\n=== TRANSFORM ===")

    initial_cols = len(df.columns)

    df = transform_dates(df)
    print(f"  - Transformed date columns")

    df = transform_numeric(df)
    print(f"  - Transformed numeric columns")

    df = transform_text(df)
    print(f"  - Cleaned text columns")

    df = transform_add_metadata(df)
    print(f"  - Added ETL metadata")

    print(f"Transformation complete: {initial_cols} -> {len(df.columns)} columns")

    return df


# =============================================================================
# LOAD
# =============================================================================

def create_database() -> Path:
    """Create database directory and return path to database file."""
    DB_DIR.mkdir(exist_ok=True)
    return DB_DIR / DB_NAME


def get_connection(db_path: Path) -> sqlite3.Connection:
    """Get SQLite connection with proper settings."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def create_tables(conn: sqlite3.Connection):
    """Create database tables from DATA_DICTIONARY."""
    print(f"\n=== CREATE TABLES ===")

    # Main breach incidents table
    create_sql = generate_create_table_sql("breach_data")

    # Add ETL metadata columns to the SQL
    create_sql = create_sql.replace(
        "\n);",
        ",\n    _etl_loaded_at DATETIME,\n    _etl_source VARCHAR(200)\n);"
    )

    # Drop existing table and recreate
    conn.execute("DROP TABLE IF EXISTS breach_incidents")
    conn.execute(create_sql)

    print(f"  - Created table: breach_incidents")

    # Create indexes for common queries
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_org_name ON breach_incidents(org_name)",
        "CREATE INDEX IF NOT EXISTS idx_breach_date ON breach_incidents(breach_date)",
        "CREATE INDEX IF NOT EXISTS idx_stock_ticker ON breach_incidents(stock_ticker)",
        "CREATE INDEX IF NOT EXISTS idx_breach_type ON breach_incidents(breach_type)",
        "CREATE INDEX IF NOT EXISTS idx_ticker_status ON breach_incidents(ticker_status)",
        "CREATE INDEX IF NOT EXISTS idx_total_affected ON breach_incidents(total_affected)",
    ]

    for idx_sql in indexes:
        conn.execute(idx_sql)

    print(f"  - Created {len(indexes)} indexes")

    conn.commit()


def load_data(conn: sqlite3.Connection, df: pd.DataFrame):
    """Load transformed data into the database."""
    print(f"\n=== LOAD ===")

    # Get columns that exist in the dataframe
    df_columns = list(df.columns)

    # Insert data
    placeholders = ', '.join(['?' for _ in df_columns])
    columns_str = ', '.join(df_columns)

    insert_sql = f"INSERT INTO breach_incidents ({columns_str}) VALUES ({placeholders})"

    # Convert DataFrame to list of tuples, handling None values
    records = []
    for _, row in df.iterrows():
        record = []
        for col in df_columns:
            val = row[col]
            if pd.isna(val):
                record.append(None)
            else:
                record.append(val)
        records.append(tuple(record))

    conn.executemany(insert_sql, records)
    conn.commit()

    # Verify load
    cursor = conn.execute("SELECT COUNT(*) FROM breach_incidents")
    count = cursor.fetchone()[0]

    print(f"  - Loaded {count} records into breach_incidents")


def create_summary_views(conn: sqlite3.Connection):
    """Create useful summary views for analysis."""
    print(f"\n=== CREATE VIEWS ===")

    views = {
        "v_breach_summary": """
            CREATE VIEW IF NOT EXISTS v_breach_summary AS
            SELECT
                org_name,
                stock_ticker,
                breach_date,
                total_affected,
                breach_type,
                ticker_status,
                yf_sector,
                yf_market_cap,
                total_news_count
            FROM breach_incidents
            ORDER BY total_affected DESC
        """,

        "v_breaches_by_year": """
            CREATE VIEW IF NOT EXISTS v_breaches_by_year AS
            SELECT
                strftime('%Y', breach_date) as year,
                COUNT(*) as breach_count,
                SUM(total_affected) as total_affected,
                AVG(total_affected) as avg_affected
            FROM breach_incidents
            WHERE breach_date IS NOT NULL
            GROUP BY strftime('%Y', breach_date)
            ORDER BY year
        """,

        "v_breaches_by_sector": """
            CREATE VIEW IF NOT EXISTS v_breaches_by_sector AS
            SELECT
                yf_sector as sector,
                COUNT(*) as breach_count,
                SUM(total_affected) as total_affected,
                AVG(total_affected) as avg_affected,
                SUM(total_news_count) as total_news_coverage
            FROM breach_incidents
            WHERE yf_sector IS NOT NULL
            GROUP BY yf_sector
            ORDER BY breach_count DESC
        """,

        "v_breaches_by_type": """
            CREATE VIEW IF NOT EXISTS v_breaches_by_type AS
            SELECT
                breach_type,
                COUNT(*) as breach_count,
                SUM(total_affected) as total_affected,
                AVG(total_affected) as avg_affected
            FROM breach_incidents
            GROUP BY breach_type
            ORDER BY breach_count DESC
        """,

        "v_top_breaches": """
            CREATE VIEW IF NOT EXISTS v_top_breaches AS
            SELECT
                org_name,
                breach_date,
                total_affected,
                breach_type,
                yf_company_name,
                yf_sector,
                total_news_count
            FROM breach_incidents
            WHERE total_affected IS NOT NULL
            ORDER BY total_affected DESC
            LIMIT 100
        """,

        "v_news_coverage": """
            CREATE VIEW IF NOT EXISTS v_news_coverage AS
            SELECT
                org_name,
                stock_ticker,
                reddit_count,
                guardian_count,
                nyt_count,
                newsapi_count,
                total_news_count
            FROM breach_incidents
            WHERE total_news_count > 0
            ORDER BY total_news_count DESC
        """,
    }

    for view_name, view_sql in views.items():
        conn.execute(f"DROP VIEW IF EXISTS {view_name}")
        conn.execute(view_sql)
        print(f"  - Created view: {view_name}")

    conn.commit()


def load(df: pd.DataFrame) -> Path:
    """
    Load data into SQLite database.
    """
    db_path = create_database()
    print(f"Database: {db_path}")

    conn = get_connection(db_path)

    try:
        create_tables(conn)
        load_data(conn, df)
        create_summary_views(conn)
    finally:
        conn.close()

    return db_path


# =============================================================================
# ETL PIPELINE
# =============================================================================

def run_etl(source_file: str = "breach_data_enriched.csv") -> Path:
    """
    Run the complete ETL pipeline.

    Args:
        source_file: Name of the source CSV file in the cleaned directory

    Returns:
        Path to the created database
    """
    print("=" * 60)
    print("BREACH DATA ETL PIPELINE")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Extract
    df = extract_from_csv(source_file)

    # Transform
    df = transform(df)

    # Load
    db_path = load(df)

    print("\n" + "=" * 60)
    print("ETL COMPLETE")
    print("=" * 60)
    print(f"Database: {db_path}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return db_path


def query_database(query: str, db_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Execute a query against the database and return results as DataFrame.

    Args:
        query: SQL query to execute
        db_path: Path to database (uses default if not specified)

    Returns:
        Query results as pandas DataFrame
    """
    if db_path is None:
        db_path = DB_DIR / DB_NAME

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    db_path = run_etl()

    # Show sample queries
    print("\n=== SAMPLE QUERIES ===\n")

    print("Top 10 breaches by affected individuals:")
    df = query_database("SELECT * FROM v_top_breaches LIMIT 10")
    print(df.to_string(index=False))

    print("\n\nBreaches by year:")
    df = query_database("SELECT * FROM v_breaches_by_year")
    print(df.to_string(index=False))

    print("\n\nBreaches by sector:")
    df = query_database("SELECT * FROM v_breaches_by_sector LIMIT 10")
    print(df.to_string(index=False))
