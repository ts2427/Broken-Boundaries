"""
clean.py - Data Cleaning Pipeline
First stage of the Broken Boundaries data pipeline.
Pulls in raw data sources and performs cleaning operations.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# === Configuration ===
RAW_DATA_DIR = Path(__file__).parent
OUTPUT_DIR = Path(__file__).parent / "cleaned"


def load_breach_data(filepath: str = "Data_Breach_Enriched_Final.csv") -> pd.DataFrame:
    """Load the breach data CSV file."""
    data_path = RAW_DATA_DIR / filepath
    # Try UTF-8 first, fall back to latin-1 for Windows-encoded files
    try:
        df = pd.read_csv(data_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(data_path, encoding='latin-1')
    print(f"Loaded {len(df)} records from {filepath}")
    return df


def clean_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize date columns. Handles mixed formats."""
    date_columns = ['reported_date', 'breach_date', 'end_breach_date']

    for col in date_columns:
        if col in df.columns:
            # Try multiple date formats to handle mixed data
            df[col] = pd.to_datetime(df[col], format='mixed', errors='coerce')

    return df


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean numeric columns and handle missing values."""
    numeric_columns = ['total_affected', 'cik', 'sic', 'naics']

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean text columns - strip whitespace and normalize."""
    text_columns = ['org_name', 'organization_type', 'breach_type', 'stock_ticker']

    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace('nan', np.nan)

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate records."""
    initial_count = len(df)
    df = df.drop_duplicates()
    removed = initial_count - len(df)
    if removed > 0:
        print(f"Removed {removed} duplicate records")
    return df


def summarize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Print summary statistics for the cleaned data."""
    print(f"Total records: {len(df)}")
    print(f"  - With org_name: {df['org_name'].notna().sum()}")
    print(f"  - With breach_date: {df['breach_date'].notna().sum()}")
    print(f"  - With total_affected: {df['total_affected'].notna().sum()}")
    return df


def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline."""
    print("\n=== Starting Data Cleaning Pipeline ===\n")

    df = clean_dates(df)
    df = clean_numeric_columns(df)
    df = clean_text_columns(df)
    df = remove_duplicates(df)
    df = summarize_data(df)

    print("\n=== Cleaning Complete ===\n")
    return df


def save_cleaned_data(df: pd.DataFrame, filename: str = "breach_data_cleaned.csv"):
    """Save cleaned data to output directory."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / filename
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")


def main():
    """Main entry point for the cleaning pipeline."""
    # Load raw data source
    df = load_breach_data("Data_Breach_Enriched_Final.csv")

    # Display initial data info
    print(f"\nColumns: {list(df.columns)}")
    print(f"Shape: {df.shape}")

    # Run cleaning pipeline
    df_cleaned = clean_pipeline(df)

    # Save cleaned data
    save_cleaned_data(df_cleaned)

    return df_cleaned


if __name__ == "__main__":
    main()


# =============================================================================
# DATA DICTIONARY
# =============================================================================
# Source: Data_Breach_Enriched_Final.csv
# Records: 1,772 data breach incidents
#
# This dictionary is used for ETL operations, database schema creation,
# and data retrieval throughout the pipeline.
# =============================================================================

DATA_DICTIONARY = {
    "breach_data": {
        "source": "Data_Breach_Enriched_Final.csv",
        "table_name": "breach_incidents",
        "description": "Data breach incidents affecting organizations",
        "columns": {
            "org_name": {
                "python_type": "str",
                "sql_type": "VARCHAR(500)",
                "pandas_dtype": "object",
                "nullable": False,
                "description": "Name of the organization that experienced the breach",
            },
            "reported_date": {
                "python_type": "datetime",
                "sql_type": "DATE",
                "pandas_dtype": "datetime64[ns]",
                "nullable": True,
                "description": "Date the breach was reported to authorities",
            },
            "breach_date": {
                "python_type": "datetime",
                "sql_type": "DATE",
                "pandas_dtype": "datetime64[ns]",
                "nullable": True,
                "description": "Date the breach occurred or was first detected",
            },
            "end_breach_date": {
                "python_type": "datetime",
                "sql_type": "DATE",
                "pandas_dtype": "datetime64[ns]",
                "nullable": True,
                "description": "Date the breach ended (if applicable)",
            },
            "incident_details": {
                "python_type": "str",
                "sql_type": "TEXT",
                "pandas_dtype": "object",
                "nullable": True,
                "description": "Narrative description of the breach incident",
            },
            "information_affected": {
                "python_type": "str",
                "sql_type": "JSON",
                "pandas_dtype": "object",
                "nullable": True,
                "description": "Structured data on types of information compromised (encryption status, categories affected, examples)",
            },
            "organization_type": {
                "python_type": "str",
                "sql_type": "VARCHAR(50)",
                "pandas_dtype": "object",
                "nullable": True,
                "description": "Type/category of organization (e.g., BSF, BSO)",
            },
            "total_affected": {
                "python_type": "int",
                "sql_type": "INTEGER",
                "pandas_dtype": "Int64",
                "nullable": True,
                "description": "Number of individuals affected by the breach",
            },
            "breach_type": {
                "python_type": "str",
                "sql_type": "VARCHAR(50)",
                "pandas_dtype": "object",
                "nullable": True,
                "description": "Type of breach (e.g., HACK, PHYS, INSD)",
            },
            "stock_ticker": {
                "python_type": "str",
                "sql_type": "VARCHAR(20)",
                "pandas_dtype": "object",
                "nullable": True,
                "description": "Stock ticker symbol (if publicly traded)",
            },
            "cik": {
                "python_type": "int",
                "sql_type": "INTEGER",
                "pandas_dtype": "Int64",
                "nullable": True,
                "description": "SEC Central Index Key identifier",
            },
            "sic": {
                "python_type": "int",
                "sql_type": "INTEGER",
                "pandas_dtype": "Int64",
                "nullable": True,
                "description": "Standard Industrial Classification code",
            },
            "naics": {
                "python_type": "int",
                "sql_type": "INTEGER",
                "pandas_dtype": "Int64",
                "nullable": True,
                "description": "North American Industry Classification System code",
            },
        },
    }
}


def get_schema(dataset_key: str = "breach_data") -> dict:
    """Get the schema definition for a dataset."""
    return DATA_DICTIONARY.get(dataset_key, {})


def get_column_names(dataset_key: str = "breach_data") -> list:
    """Get list of column names for a dataset."""
    schema = get_schema(dataset_key)
    return list(schema.get("columns", {}).keys())


def get_sql_types(dataset_key: str = "breach_data") -> dict:
    """Get SQL type mapping for database schema creation."""
    schema = get_schema(dataset_key)
    return {
        col: info["sql_type"]
        for col, info in schema.get("columns", {}).items()
    }


def get_pandas_dtypes(dataset_key: str = "breach_data") -> dict:
    """Get pandas dtype mapping for data loading."""
    schema = get_schema(dataset_key)
    return {
        col: info["pandas_dtype"]
        for col, info in schema.get("columns", {}).items()
    }


def generate_create_table_sql(dataset_key: str = "breach_data", dialect: str = "sqlite") -> str:
    """Generate CREATE TABLE SQL statement from the data dictionary."""
    schema = get_schema(dataset_key)
    table_name = schema.get("table_name", "data")
    columns = schema.get("columns", {})

    col_definitions = []
    for col_name, col_info in columns.items():
        sql_type = col_info["sql_type"]
        nullable = "NULL" if col_info["nullable"] else "NOT NULL"
        col_definitions.append(f"    {col_name} {sql_type} {nullable}")

    # Add primary key
    col_definitions.insert(0, "    id INTEGER PRIMARY KEY AUTOINCREMENT")

    sql = f"CREATE TABLE IF NOT EXISTS {table_name} (\n"
    sql += ",\n".join(col_definitions)
    sql += "\n);"

    return sql


def apply_schema_dtypes(df: pd.DataFrame, dataset_key: str = "breach_data") -> pd.DataFrame:
    """Apply data dictionary dtypes to a DataFrame."""
    schema = get_schema(dataset_key)
    columns = schema.get("columns", {})

    for col_name, col_info in columns.items():
        if col_name not in df.columns:
            continue

        pandas_dtype = col_info["pandas_dtype"]

        if pandas_dtype == "datetime64[ns]":
            df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
        elif pandas_dtype == "Int64":
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce').astype("Int64")
        elif pandas_dtype == "object":
            df[col_name] = df[col_name].astype(str).replace('nan', np.nan)

    return df
