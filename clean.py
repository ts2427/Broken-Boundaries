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
# Column                  | Type     | Description
# ------------------------|----------|---------------------------------------------
# org_name                | string   | Name of the organization that experienced the breach
# reported_date           | datetime | Date the breach was reported to authorities
# breach_date             | datetime | Date the breach occurred or was first detected
# end_breach_date         | datetime | Date the breach ended (if applicable)
# incident_details        | string   | Narrative description of the breach incident
# information_affected    | JSON     | Structured data on types of information compromised
#                         |          | (encryption status, categories affected, examples)
# organization_type       | string   | Type/category of organization (e.g., BSF, BSO)
# total_affected          | integer  | Number of individuals affected by the breach
# breach_type             | string   | Type of breach (e.g., HACK, PHYS, INSD)
# stock_ticker            | string   | Stock ticker symbol (if publicly traded)
# cik                     | integer  | SEC Central Index Key identifier
# sic                     | integer  | Standard Industrial Classification code
# naics                   | integer  | North American Industry Classification System code
# =============================================================================
