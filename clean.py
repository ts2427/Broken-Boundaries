"""
clean.py - Data Cleaning Pipeline
First stage of the Broken Boundaries data pipeline.
Pulls in raw data sources and performs cleaning operations.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from typing import Optional


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


# =============================================================================
# STOCK DATA ENRICHMENT (Yahoo Finance)
# =============================================================================

# Ticker mapping for companies that changed their ticker symbols
# Format: old_ticker -> new_ticker
TICKER_MAPPING = {
    "SQ": "XYZ",       # Block Inc changed ticker in 2023
    "FI": "FISV",      # Fiserv correct ticker
    "DISH": "SATS",    # Dish merged with EchoStar
    "FB": "META",      # Facebook -> Meta
}

# Tickers that are delisted/acquired (no longer tradeable)
# These will be skipped with a note
DELISTED_TICKERS = {
    "ATVI": "Acquired by Microsoft (2023)",
    "TWTR": "Acquired by X Corp (2022)",
    "VMW": "Acquired by Broadcom (2023)",
    "YHOO": "Acquired by Verizon (2017)",
    "CTXS": "Taken private by Vista/Evergreen (2022)",
    "CONE": "Acquired by KKR (2022)",
    "PARA": "Merged with Skydance (2024)",
    "WBA": "Taken private (2024)",
    "ATUS": "Delisted (2024)",
    "AUDAQ": "Bankruptcy, delisted (2024)",
    "MCCC": "Taken private",
}


def get_current_ticker(ticker: str) -> tuple[str, str]:
    """
    Get the current trading ticker for a given ticker symbol.
    Returns (current_ticker, note) where note explains any mapping.
    """
    ticker = ticker.upper().strip()

    if ticker in DELISTED_TICKERS:
        return None, DELISTED_TICKERS[ticker]

    if ticker in TICKER_MAPPING:
        new_ticker = TICKER_MAPPING[ticker]
        return new_ticker, f"Mapped from {ticker}"

    return ticker, None


def fetch_stock_info(ticker: str) -> Optional[dict]:
    """Fetch stock information from Yahoo Finance for a single ticker."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Check if we got valid data (Yahoo Finance returns empty dict for invalid tickers)
        if not info or info.get('regularMarketPrice') is None:
            return None

        return {
            'yf_ticker': ticker,
            'yf_company_name': info.get('longName') or info.get('shortName'),
            'yf_sector': info.get('sector'),
            'yf_industry': info.get('industry'),
            'yf_market_cap': info.get('marketCap'),
            'yf_enterprise_value': info.get('enterpriseValue'),
            'yf_employees': info.get('fullTimeEmployees'),
            'yf_country': info.get('country'),
            'yf_website': info.get('website'),
            'yf_exchange': info.get('exchange'),
            'yf_currency': info.get('currency'),
            'yf_current_price': info.get('regularMarketPrice'),
            'yf_52week_high': info.get('fiftyTwoWeekHigh'),
            'yf_52week_low': info.get('fiftyTwoWeekLow'),
            'yf_avg_volume': info.get('averageVolume'),
            'yf_dividend_yield': info.get('dividendYield'),
            'yf_beta': info.get('beta'),
            'yf_pe_ratio': info.get('trailingPE'),
            'yf_forward_pe': info.get('forwardPE'),
            'yf_profit_margin': info.get('profitMargins'),
            'yf_revenue': info.get('totalRevenue'),
            'yf_gross_profit': info.get('grossProfits'),
            'yf_ebitda': info.get('ebitda'),
            'yf_total_debt': info.get('totalDebt'),
            'yf_total_cash': info.get('totalCash'),
        }
    except Exception as e:
        print(f"  Warning: Could not fetch data for {ticker}: {e}")
        return None


def fetch_all_stock_data(tickers: list) -> pd.DataFrame:
    """Fetch stock data for all unique tickers."""
    print(f"\n=== Fetching Stock Data from Yahoo Finance ===\n")
    print(f"Unique tickers to fetch: {len(tickers)}")

    stock_data = []
    matched = 0
    not_found = 0
    delisted = 0
    mapped = 0

    for i, ticker in enumerate(tickers):
        if pd.isna(ticker) or ticker == 'nan' or ticker == '':
            continue

        original_ticker = str(ticker).strip().upper()
        current_ticker, note = get_current_ticker(original_ticker)

        # Handle delisted tickers
        if current_ticker is None:
            print(f"  [{i+1}/{len(tickers)}] {original_ticker}... DELISTED ({note})")
            delisted += 1
            continue

        # Show mapping if applicable
        if note and "Mapped" in note:
            print(f"  [{i+1}/{len(tickers)}] {original_ticker} -> {current_ticker}...", end=" ")
            mapped += 1
        else:
            print(f"  [{i+1}/{len(tickers)}] Fetching {current_ticker}...", end=" ")

        info = fetch_stock_info(current_ticker)
        if info:
            # Store original ticker for merging back to breach data
            info['_original_ticker'] = original_ticker
            stock_data.append(info)
            matched += 1
            print("OK")
        else:
            not_found += 1
            print("Not found")

    print(f"\nStock data summary:")
    print(f"  - Matched: {matched}")
    print(f"  - Mapped to new ticker: {mapped}")
    print(f"  - Delisted/Acquired: {delisted}")
    print(f"  - Not found: {not_found}")

    if stock_data:
        return pd.DataFrame(stock_data)
    return pd.DataFrame()


def enrich_with_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich breach data with stock information from Yahoo Finance."""
    # Get unique tickers
    tickers = df['stock_ticker'].dropna().unique().tolist()

    if not tickers:
        print("No tickers found in data, skipping stock enrichment")
        return df

    # Fetch stock data
    stock_df = fetch_all_stock_data(tickers)

    if stock_df.empty:
        print("No stock data retrieved, skipping enrichment")
        return df

    # Merge stock data with breach data on ticker
    # Use _original_ticker for merging (handles mapped tickers)
    df['stock_ticker_upper'] = df['stock_ticker'].str.upper().str.strip()
    df = df.merge(
        stock_df,
        left_on='stock_ticker_upper',
        right_on='_original_ticker',
        how='left'
    )
    df = df.drop(columns=['stock_ticker_upper', '_original_ticker'])

    # Add ticker status columns
    df = add_ticker_status(df)

    # Count enriched records
    enriched_count = df['yf_ticker'].notna().sum()
    print(f"\nEnriched {enriched_count} records with stock data")

    return df


def add_ticker_status(df: pd.DataFrame) -> pd.DataFrame:
    """Add ticker_status and ticker_status_note columns to track ticker state."""

    def get_status(row):
        ticker = row.get('stock_ticker')
        if pd.isna(ticker) or ticker == 'nan':
            return 'no_ticker', None

        ticker_upper = str(ticker).upper().strip()

        # Check if delisted
        if ticker_upper in DELISTED_TICKERS:
            return 'delisted', DELISTED_TICKERS[ticker_upper]

        # Check if mapped
        if ticker_upper in TICKER_MAPPING:
            new_ticker = TICKER_MAPPING[ticker_upper]
            return 'mapped', f"{ticker_upper} -> {new_ticker}"

        # Check if we have stock data
        if pd.notna(row.get('yf_ticker')):
            return 'active', None

        return 'not_found', 'Ticker not found on Yahoo Finance'

    # Apply status to each row
    statuses = df.apply(get_status, axis=1)
    df['ticker_status'] = [s[0] for s in statuses]
    df['ticker_status_note'] = [s[1] for s in statuses]

    # Print summary
    print("\nTicker status summary:")
    for status in ['active', 'mapped', 'delisted', 'not_found', 'no_ticker']:
        count = (df['ticker_status'] == status).sum()
        if count > 0:
            print(f"  - {status}: {count} records")

    return df


def main():
    """Main entry point for the cleaning pipeline."""
    # Load raw data source
    df = load_breach_data("Data_Breach_Enriched_Final.csv")

    # Display initial data info
    print(f"\nColumns: {list(df.columns)}")
    print(f"Shape: {df.shape}")

    # Run cleaning pipeline
    df_cleaned = clean_pipeline(df)

    # Enrich with stock data from Yahoo Finance
    df_enriched = enrich_with_stock_data(df_cleaned)

    # Save enriched data
    save_cleaned_data(df_enriched, "breach_data_enriched.csv")

    return df_enriched


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
            # --- Yahoo Finance Stock Data (yf_ prefix) ---
            "yf_ticker": {
                "python_type": "str",
                "sql_type": "VARCHAR(20)",
                "pandas_dtype": "object",
                "nullable": True,
                "description": "Verified ticker symbol from Yahoo Finance",
            },
            "yf_company_name": {
                "python_type": "str",
                "sql_type": "VARCHAR(500)",
                "pandas_dtype": "object",
                "nullable": True,
                "description": "Official company name from Yahoo Finance",
            },
            "yf_sector": {
                "python_type": "str",
                "sql_type": "VARCHAR(100)",
                "pandas_dtype": "object",
                "nullable": True,
                "description": "Business sector (e.g., Technology, Healthcare)",
            },
            "yf_industry": {
                "python_type": "str",
                "sql_type": "VARCHAR(200)",
                "pandas_dtype": "object",
                "nullable": True,
                "description": "Specific industry classification",
            },
            "yf_market_cap": {
                "python_type": "float",
                "sql_type": "BIGINT",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "Market capitalization in USD",
            },
            "yf_enterprise_value": {
                "python_type": "float",
                "sql_type": "BIGINT",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "Enterprise value in USD",
            },
            "yf_employees": {
                "python_type": "int",
                "sql_type": "INTEGER",
                "pandas_dtype": "Int64",
                "nullable": True,
                "description": "Number of full-time employees",
            },
            "yf_country": {
                "python_type": "str",
                "sql_type": "VARCHAR(100)",
                "pandas_dtype": "object",
                "nullable": True,
                "description": "Country of headquarters",
            },
            "yf_website": {
                "python_type": "str",
                "sql_type": "VARCHAR(500)",
                "pandas_dtype": "object",
                "nullable": True,
                "description": "Company website URL",
            },
            "yf_exchange": {
                "python_type": "str",
                "sql_type": "VARCHAR(50)",
                "pandas_dtype": "object",
                "nullable": True,
                "description": "Stock exchange (e.g., NMS, NYQ)",
            },
            "yf_currency": {
                "python_type": "str",
                "sql_type": "VARCHAR(10)",
                "pandas_dtype": "object",
                "nullable": True,
                "description": "Trading currency",
            },
            "yf_current_price": {
                "python_type": "float",
                "sql_type": "DECIMAL(12,4)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "Current stock price",
            },
            "yf_52week_high": {
                "python_type": "float",
                "sql_type": "DECIMAL(12,4)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "52-week high price",
            },
            "yf_52week_low": {
                "python_type": "float",
                "sql_type": "DECIMAL(12,4)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "52-week low price",
            },
            "yf_avg_volume": {
                "python_type": "int",
                "sql_type": "BIGINT",
                "pandas_dtype": "Int64",
                "nullable": True,
                "description": "Average daily trading volume",
            },
            "yf_dividend_yield": {
                "python_type": "float",
                "sql_type": "DECIMAL(8,6)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "Dividend yield as decimal",
            },
            "yf_beta": {
                "python_type": "float",
                "sql_type": "DECIMAL(8,4)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "Beta coefficient (volatility measure)",
            },
            "yf_pe_ratio": {
                "python_type": "float",
                "sql_type": "DECIMAL(12,4)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "Trailing price-to-earnings ratio",
            },
            "yf_forward_pe": {
                "python_type": "float",
                "sql_type": "DECIMAL(12,4)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "Forward price-to-earnings ratio",
            },
            "yf_profit_margin": {
                "python_type": "float",
                "sql_type": "DECIMAL(8,6)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "Profit margin as decimal",
            },
            "yf_revenue": {
                "python_type": "float",
                "sql_type": "BIGINT",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "Total revenue in USD",
            },
            "yf_gross_profit": {
                "python_type": "float",
                "sql_type": "BIGINT",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "Gross profit in USD",
            },
            "yf_ebitda": {
                "python_type": "float",
                "sql_type": "BIGINT",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "EBITDA in USD",
            },
            "yf_total_debt": {
                "python_type": "float",
                "sql_type": "BIGINT",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "Total debt in USD",
            },
            "yf_total_cash": {
                "python_type": "float",
                "sql_type": "BIGINT",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "Total cash and equivalents in USD",
            },
            # --- Ticker Status Tracking ---
            "ticker_status": {
                "python_type": "str",
                "sql_type": "VARCHAR(20)",
                "pandas_dtype": "object",
                "nullable": True,
                "description": "Status of ticker: active, mapped, delisted, not_found, no_ticker",
            },
            "ticker_status_note": {
                "python_type": "str",
                "sql_type": "VARCHAR(200)",
                "pandas_dtype": "object",
                "nullable": True,
                "description": "Details about ticker status (e.g., 'Acquired by Microsoft (2023)', 'SQ -> XYZ')",
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
