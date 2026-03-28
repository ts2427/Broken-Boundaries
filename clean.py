"""
clean.py - Data Cleaning Pipeline
First stage of the Broken Boundaries data pipeline.
Pulls in raw data sources and performs cleaning operations.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import praw
import time
import os
import io
import json
import logging
import re
import zipfile
import urllib.request
from pathlib import Path
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables for API keys
load_dotenv()

# Initialize Reddit client (if credentials available)
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_CLIENT = None

if REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET:
    try:
        REDDIT_CLIENT = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent="BrokenBoundaries/1.0 (academic research)"
        )
    except Exception as e:
        logger.warning("Could not initialize Reddit client: %s", e)


def _request_with_retry(url: str, params: dict = None, max_retries: int = 3, timeout: int = 10) -> Optional[requests.Response]:
    """Make an HTTP GET request with exponential backoff on failure."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            if response.status_code == 429:  # Rate limited
                wait = 2 ** (attempt + 1)
                logger.warning("Rate limited on %s, retrying in %ds", url, wait)
                time.sleep(wait)
                continue
            return response
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning("Request to %s failed (%s), retry %d/%d in %ds",
                              url, e, attempt + 1, max_retries, wait)
                time.sleep(wait)
            else:
                logger.warning("Request to %s failed after %d retries: %s", url, max_retries, e)
    return None


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
    logger.info("Loaded %d records from %s", len(df), filepath)
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
            df[col] = df[col].str.strip()

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate records."""
    initial_count = len(df)
    df = df.drop_duplicates()
    removed = initial_count - len(df)
    if removed > 0:
        logger.info("Removed %d duplicate records", removed)
    return df


def summarize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Print summary statistics for the cleaned data."""
    logger.info("Total records: %d", len(df))
    logger.info("  - With org_name: %d", df['org_name'].notna().sum())
    logger.info("  - With breach_date: %d", df['breach_date'].notna().sum())
    logger.info("  - With total_affected: %d", df['total_affected'].notna().sum())
    return df


def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline."""
    logger.info("Starting Data Cleaning Pipeline")

    df = clean_dates(df)
    df = clean_numeric_columns(df)
    df = clean_text_columns(df)
    df = remove_duplicates(df)
    df = summarize_data(df)

    logger.info("Cleaning Complete")
    return df


def save_cleaned_data(df: pd.DataFrame, filename: str = "breach_data_cleaned.csv"):
    """Save cleaned data to output directory."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / filename
    df.to_csv(output_path, index=False)
    logger.info("Saved cleaned data to %s", output_path)


# =============================================================================
# FAMA-FRENCH FACTOR DATA (Kenneth French Data Library)
# =============================================================================
# Fetches the Fama-French 5-Factor model data from Kenneth French's website.
# Factors: Mkt-RF (market excess return), SMB (size), HML (value),
#          RMW (profitability), CMA (investment), RF (risk-free rate)
# Source: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
# Values are in percentage points (1.5 = 1.5% return)
# Daily data available from July 1963 onward for the 5-factor model.
# =============================================================================

FF_DATASET_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"


def fetch_fama_french_data() -> pd.DataFrame:
    """
    Fetch daily Fama-French 5-Factor data from Kenneth French's data library.
    Downloads the ZIP/CSV directly from Dartmouth. Falls back to
    pandas_datareader if the direct download fails.

    Returns a DataFrame with columns: Mkt-RF, SMB, HML, RMW, CMA, RF
    indexed by date. Values are in percentage points.
    """
    logger.info("Fetching Fama-French 5-Factor daily data")

    df = _fetch_ff_direct()
    if df is not None and not df.empty:
        return df

    df = _fetch_ff_datareader()
    if df is not None and not df.empty:
        return df

    logger.error("All Fama-French data sources failed")
    return pd.DataFrame()


def _fetch_ff_direct() -> Optional[pd.DataFrame]:
    """Download Fama-French factors directly from Kenneth French's website."""
    for attempt in range(3):
        try:
            logger.info("Downloading Fama-French data from Dartmouth (attempt %d/3)", attempt + 1)
            response = urllib.request.urlopen(FF_DATASET_URL, timeout=30)
            zip_data = io.BytesIO(response.read())

            with zipfile.ZipFile(zip_data) as zf:
                csv_filename = zf.namelist()[0]
                with zf.open(csv_filename) as csv_file:
                    raw_text = csv_file.read().decode('utf-8')

            lines = raw_text.strip().split('\n')

            # Find the header row containing 'Mkt-RF'
            header_idx = None
            for i, line in enumerate(lines):
                if 'Mkt-RF' in line:
                    header_idx = i
                    break

            if header_idx is None:
                logger.warning("Could not find header row in Fama-French CSV")
                return None

            # Collect data lines until we hit a blank line or non-numeric row
            data_lines = [lines[header_idx]]
            for line in lines[header_idx + 1:]:
                stripped = line.strip()
                if not stripped or not stripped[0].isdigit():
                    break
                data_lines.append(stripped)

            df = pd.read_csv(io.StringIO('\n'.join(data_lines)), index_col=0)
            df.index.name = 'date'
            df.index = pd.to_datetime(df.index.astype(str).str.strip(), format='%Y%m%d')
            df.columns = [c.strip() for c in df.columns]

            # Standardize column names
            rename_map = {
                'Mkt-RF': 'Mkt-RF', 'SMB': 'SMB', 'HML': 'HML',
                'RMW': 'RMW', 'CMA': 'CMA', 'RF': 'RF',
            }
            df = df.rename(columns=rename_map)

            logger.info("Retrieved %d daily Fama-French observations (%s to %s)",
                        len(df), df.index.min().strftime('%Y-%m-%d'), df.index.max().strftime('%Y-%m-%d'))
            return df

        except Exception as e:
            wait = 2 ** attempt
            logger.warning("Fama-French direct download failed (%s), retrying in %ds", e, wait)
            time.sleep(wait)

    return None


def _fetch_ff_datareader() -> Optional[pd.DataFrame]:
    """Fallback: fetch Fama-French data via pandas_datareader."""
    try:
        import pandas_datareader.data as web
    except ImportError:
        logger.warning("pandas_datareader not installed, skipping fallback")
        return None

    try:
        logger.info("Trying pandas_datareader fallback for Fama-French data")
        result = web.DataReader(
            'F-F_Research_Data_5_Factors_2x3_daily',
            'famafrench',
            start='1963-07-01',
        )
        df = result[0]
        df.index.name = 'date'
        df.columns = [c.strip() for c in df.columns]
        logger.info("Retrieved %d observations via pandas_datareader", len(df))
        return df
    except Exception as e:
        logger.warning("pandas_datareader Fama-French fetch failed: %s", e)
        return None


def enrich_with_fama_french_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich breach data with Fama-French 5-Factor metrics.
    Uses vectorized merge_asof for point-in-time lookups and
    searchsorted-based window averages.
    """
    ff_df = fetch_fama_french_data()

    factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
    ff_prefix_map = {
        'Mkt-RF': 'ff_mkt_rf', 'SMB': 'ff_smb', 'HML': 'ff_hml',
        'RMW': 'ff_rmw', 'CMA': 'ff_cma', 'RF': 'ff_rf',
    }
    avg_name_map = {'Mkt-RF': 'mkt_rf', 'SMB': 'smb', 'HML': 'hml', 'RMW': 'rmw', 'CMA': 'cma'}
    all_cols = (
        list(ff_prefix_map.values())
        + [f'ff_{n}_30d_avg' for n in avg_name_map.values()]
        + [f'ff_{n}_90d_avg' for n in avg_name_map.values()]
    )

    if ff_df.empty:
        logger.warning("No Fama-French data available, skipping enrichment")
        for col in all_cols:
            df[col] = None
        return df

    logger.info("Calculating Fama-French metrics for %d breach records...", len(df))

    # Prepare FF data with date column for merge_asof
    ff_work = ff_df.reset_index()
    ff_work['date'] = pd.to_datetime(ff_work['date'])
    ff_work = ff_work.sort_values('date').reset_index(drop=True)

    has_date = df['breach_date'].notna()
    for col in all_cols:
        df[col] = np.nan

    if has_date.any():
        work = df.loc[has_date, ['breach_date']].copy().sort_values('breach_date')

        # Point-in-time factor values — nearest trading day within 5 days
        m = pd.merge_asof(
            work, ff_work[['date'] + factor_cols],
            left_on='breach_date', right_on='date',
            direction='nearest', tolerance=pd.Timedelta(days=5)
        )
        m.index = work.index
        for src_col, dst_col in ff_prefix_map.items():
            df.loc[m.index, dst_col] = m[src_col].round(4).values

        # Window averages (30-day and 90-day centered)
        for src_col, name in avg_name_map.items():
            df[f'ff_{name}_30d_avg'] = _vectorized_window_avg(
                ff_work['date'], ff_work[src_col], df['breach_date'], days_before=15
            ).round(4)
            df[f'ff_{name}_90d_avg'] = _vectorized_window_avg(
                ff_work['date'], ff_work[src_col], df['breach_date'], days_before=45
            ).round(4)

    enriched_count = df['ff_mkt_rf'].notna().sum()
    logger.info("Enriched %d records with Fama-French data", enriched_count)

    return df


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
        logger.warning("Could not fetch data for %s: %s", ticker, e)
        return None


def fetch_all_stock_data(tickers: list) -> pd.DataFrame:
    """Fetch stock data for all unique tickers."""
    logger.info("Fetching Stock Data from Yahoo Finance")
    logger.info("Unique tickers to fetch: %d", len(tickers))

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
            logger.info("  [%d/%d] %s... DELISTED (%s)", i+1, len(tickers), original_ticker, note)
            delisted += 1
            continue

        # Show mapping if applicable
        if note and "Mapped" in note:
            mapped += 1

        # Rate limit Yahoo Finance API calls
        if i > 0:
            time.sleep(0.5)

        info = fetch_stock_info(current_ticker)
        if info:
            # Store original ticker for merging back to breach data
            info['_original_ticker'] = original_ticker
            stock_data.append(info)
            matched += 1
            logger.info("  [%d/%d] %s... OK", i+1, len(tickers), current_ticker)
        else:
            not_found += 1
            logger.info("  [%d/%d] %s... Not found", i+1, len(tickers), current_ticker)

    logger.info("Stock data summary: matched=%d, mapped=%d, delisted=%d, not_found=%d",
                matched, mapped, delisted, not_found)

    if stock_data:
        return pd.DataFrame(stock_data)
    return pd.DataFrame()


def enrich_with_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich breach data with stock information from Yahoo Finance."""
    # Get unique tickers
    tickers = df['stock_ticker'].dropna().unique().tolist()

    if not tickers:
        logger.info("No tickers found in data, skipping stock enrichment")
        return df

    # Fetch stock data
    stock_df = fetch_all_stock_data(tickers)

    if stock_df.empty:
        logger.warning("No stock data retrieved, skipping enrichment")
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
    logger.info("Enriched %d records with stock data", enriched_count)

    return df


def add_ticker_status(df: pd.DataFrame) -> pd.DataFrame:
    """Add ticker_status and ticker_status_note columns to track ticker state."""

    def get_status(row):
        ticker = row.get('stock_ticker')
        if pd.isna(ticker):
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

    for status in ['active', 'mapped', 'delisted', 'not_found', 'no_ticker']:
        count = (df['ticker_status'] == status).sum()
        if count > 0:
            logger.info("Ticker status %s: %d records", status, count)

    return df


# =============================================================================
# NEWS DATA ENRICHMENT (Reddit, The Guardian, New York Times)
# =============================================================================
# Fetches news articles about breached companies from 2005-2025
# Requires API keys set in .env file:
#   - GUARDIAN_API_KEY: The Guardian Open Platform API key
#   - NYT_API_KEY: New York Times Article Search API key
# Reddit uses public JSON endpoints (no key required)
# =============================================================================

# API Configuration
GUARDIAN_API_KEY = os.getenv("GUARDIAN_API_KEY", "")
NYT_API_KEY = os.getenv("NYT_API_KEY", "")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")  # Alternative: newsapi.org
NEWS_START_DATE = "2005-01-01"
NEWS_END_DATE = "2025-12-31"


def fetch_reddit_news(company_name: str, limit: int = 10) -> list:
    """
    Fetch news posts from Reddit about a company using PRAW (OAuth).
    Requires REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env
    """
    articles = []

    if not REDDIT_CLIENT:
        return articles

    try:
        # Search for posts about the company breach
        search_query = f"{company_name} breach OR hack OR data"

        for submission in REDDIT_CLIENT.subreddit("all").search(
            search_query,
            sort="relevance",
            time_filter="all",
            limit=limit
        ):
            articles.append({
                "source": "reddit",
                "title": submission.title,
                "url": f"https://reddit.com{submission.permalink}",
                "published_date": datetime.fromtimestamp(
                    submission.created_utc
                ).strftime("%Y-%m-%d"),
                "subreddit": submission.subreddit.display_name,
                "score": submission.score,
                "num_comments": submission.num_comments,
            })

        time.sleep(0.5)  # Rate limiting
    except Exception as e:
        logger.warning("Reddit fetch failed for %s: %s", company_name, e)

    return articles


def fetch_newsapi_news(company_name: str, limit: int = 10) -> list:
    """
    Fetch news articles from NewsAPI.org about a company.
    Requires NEWSAPI_KEY environment variable.
    Free tier: https://newsapi.org/ (limited to 100 requests/day)
    """
    if not NEWSAPI_KEY:
        return []

    articles = []
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": f'"{company_name}" AND (breach OR hack OR "data leak")',
            "language": "en",
            "sortBy": "relevance",
            "pageSize": limit,
            "apiKey": NEWSAPI_KEY,
        }

        response = _request_with_retry(url, params=params)
        if response and response.status_code == 200:
            data = response.json()
            for article in data.get("articles", []):
                articles.append({
                    "source": "newsapi",
                    "title": article.get("title", ""),
                    "url": article.get("url", ""),
                    "published_date": article.get("publishedAt", "")[:10] if article.get("publishedAt") else None,
                    "source_name": article.get("source", {}).get("name", ""),
                    "author": article.get("author", ""),
                    "description": article.get("description", ""),
                })
        time.sleep(0.5)  # Rate limiting
    except Exception as e:
        logger.warning("NewsAPI error for %s: %s", company_name, e)

    return articles


def fetch_guardian_news(company_name: str, limit: int = 10) -> list:
    """
    Fetch news articles from The Guardian about a company.
    Requires GUARDIAN_API_KEY environment variable.
    Free API: https://open-platform.theguardian.com/
    """
    if not GUARDIAN_API_KEY:
        return []

    articles = []
    try:
        url = "https://content.guardianapis.com/search"
        params = {
            "q": f'"{company_name}" AND (breach OR hack OR "data leak" OR cybersecurity)',
            "from-date": NEWS_START_DATE,
            "to-date": NEWS_END_DATE,
            "page-size": limit,
            "order-by": "relevance",
            "api-key": GUARDIAN_API_KEY,
            "show-fields": "headline,trailText,byline,publication"
        }

        response = _request_with_retry(url, params=params)
        if response and response.status_code == 200:
            data = response.json()
            for item in data.get("response", {}).get("results", []):
                fields = item.get("fields", {})
                articles.append({
                    "source": "guardian",
                    "title": item.get("webTitle", ""),
                    "url": item.get("webUrl", ""),
                    "published_date": item.get("webPublicationDate", "")[:10] if item.get("webPublicationDate") else None,
                    "section": item.get("sectionName", ""),
                    "byline": fields.get("byline", ""),
                    "trail_text": fields.get("trailText", ""),
                })
        time.sleep(0.5)  # Rate limiting
    except Exception as e:
        logger.warning("Guardian error for %s: %s", company_name, e)

    return articles


def fetch_nyt_news(company_name: str, limit: int = 10) -> list:
    """
    Fetch news articles from New York Times about a company.
    Requires NYT_API_KEY environment variable.
    API: https://developer.nytimes.com/
    """
    if not NYT_API_KEY:
        return []

    articles = []
    try:
        url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
        params = {
            "q": f'{company_name} breach hack cybersecurity',
            "begin_date": NEWS_START_DATE.replace("-", ""),
            "end_date": NEWS_END_DATE.replace("-", ""),
            "sort": "relevance",
            "api-key": NYT_API_KEY,
        }

        response = _request_with_retry(url, params=params)
        if response and response.status_code == 200:
            data = response.json()
            for doc in data.get("response", {}).get("docs", [])[:limit]:
                articles.append({
                    "source": "nyt",
                    "title": doc.get("headline", {}).get("main", ""),
                    "url": doc.get("web_url", ""),
                    "published_date": doc.get("pub_date", "")[:10] if doc.get("pub_date") else None,
                    "section": doc.get("section_name", ""),
                    "byline": doc.get("byline", {}).get("original", ""),
                    "lead_paragraph": doc.get("lead_paragraph", ""),
                    "word_count": doc.get("word_count", 0),
                })
        time.sleep(1)  # Rate limiting (NYT has stricter limits)
    except Exception as e:
        logger.warning("NYT error for %s: %s", company_name, e)

    return articles


def fetch_all_news_for_company(company_name: str, ticker: str = None) -> dict:
    """Fetch news from all sources for a single company.

    Uses company_name as the primary search term. If a ticker is provided
    and non-empty, it is appended to broaden the search (helps when the
    company is better known by its ticker symbol).
    """
    search_term = company_name
    if ticker and not pd.isna(ticker):
        search_term = f"{company_name} OR {ticker}"

    reddit_articles = fetch_reddit_news(search_term)
    guardian_articles = fetch_guardian_news(search_term)
    nyt_articles = fetch_nyt_news(search_term)
    newsapi_articles = fetch_newsapi_news(search_term)

    return {
        "reddit_articles": reddit_articles,
        "guardian_articles": guardian_articles,
        "nyt_articles": nyt_articles,
        "newsapi_articles": newsapi_articles,
        "reddit_count": len(reddit_articles),
        "guardian_count": len(guardian_articles),
        "nyt_count": len(nyt_articles),
        "newsapi_count": len(newsapi_articles),
        "total_news_count": len(reddit_articles) + len(guardian_articles) + len(nyt_articles) + len(newsapi_articles),
    }


def fetch_news_for_companies(companies: list) -> pd.DataFrame:
    """Fetch news data for all unique companies."""
    logger.info("Fetching News Data (Reddit, Guardian, NYT, NewsAPI)")
    logger.info("Unique companies to search: %d", len(companies))
    logger.info("Date range: %s to %s", NEWS_START_DATE, NEWS_END_DATE)
    logger.info("API Status: Reddit=%s, Guardian=%s, NYT=%s, NewsAPI=%s",
                'OK' if REDDIT_CLIENT else 'MISSING',
                'OK' if GUARDIAN_API_KEY else 'MISSING',
                'OK' if NYT_API_KEY else 'MISSING',
                'OK' if NEWSAPI_KEY else 'MISSING')

    news_data = []

    for i, company in enumerate(companies):
        company_name = company.get("name", "")
        ticker = company.get("ticker", "")

        if not company_name or company_name == 'nan':
            continue

        logger.info("  [%d/%d] %s", i+1, len(companies), company_name[:40])

        news = fetch_all_news_for_company(company_name, ticker)
        news["company_name"] = company_name
        news["stock_ticker"] = ticker

        # Convert article lists to JSON strings for storage
        news["reddit_articles_json"] = json.dumps(news["reddit_articles"]) if news["reddit_articles"] else None
        news["guardian_articles_json"] = json.dumps(news["guardian_articles"]) if news["guardian_articles"] else None
        news["nyt_articles_json"] = json.dumps(news["nyt_articles"]) if news["nyt_articles"] else None
        news["newsapi_articles_json"] = json.dumps(news["newsapi_articles"]) if news["newsapi_articles"] else None

        # Remove the list versions (keep JSON)
        del news["reddit_articles"]
        del news["guardian_articles"]
        del news["nyt_articles"]
        del news["newsapi_articles"]

        news_data.append(news)
        logger.debug("  R:%d G:%d N:%d A:%d", news['reddit_count'], news['guardian_count'], news['nyt_count'], news['newsapi_count'])

    logger.info("News fetch complete")

    if news_data:
        return pd.DataFrame(news_data)
    return pd.DataFrame()


def _normalize_company_name(name: str) -> str:
    """Normalize company name for matching: lowercase, strip suffixes and punctuation."""
    if pd.isna(name):
        return ""
    name = str(name).lower().strip()
    # Strip common corporate suffixes
    for suffix in [", inc.", ", inc", " inc.", " inc", ", llc", " llc",
                   ", ltd.", ", ltd", " ltd.", " ltd", ", corp.", ", corp",
                   " corp.", " corp", ", co.", " co.", " company", " corporation"]:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name.strip()


def enrich_with_news_data(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich breach data with news articles from multiple sources."""
    # Get unique company/ticker combinations
    companies = df[['org_name', 'stock_ticker']].drop_duplicates()
    company_list = [
        {"name": row['org_name'], "ticker": row['stock_ticker']}
        for _, row in companies.iterrows()
    ]

    if not company_list:
        logger.info("No companies found, skipping news enrichment")
        return df

    # Fetch news data
    news_df = fetch_news_for_companies(company_list)

    if news_df.empty:
        logger.warning("No news data retrieved, skipping enrichment")
        return df

    # Normalize names for merge to handle minor differences
    df['_merge_key'] = df['org_name'].apply(_normalize_company_name)
    news_df['_merge_key'] = news_df['company_name'].apply(_normalize_company_name)

    row_count_before = len(df)
    df = df.merge(
        news_df,
        on='_merge_key',
        how='left'
    )

    # Guard against fan-out from duplicate merge keys
    if len(df) != row_count_before:
        logger.warning("News merge changed row count from %d to %d (duplicate merge keys)",
                       row_count_before, len(df))
        df = df.drop_duplicates(subset=['org_name', 'breach_date'], keep='first')
        df = df.reset_index(drop=True)
        logger.info("Deduplicated back to %d rows", len(df))

    # Log companies that failed to match
    unmatched = df[df['total_news_count'].isna() & df['org_name'].notna()]['org_name'].unique()
    if len(unmatched) > 0:
        logger.warning("%d companies had no news match (possible name mismatch): %s",
                       len(unmatched), list(unmatched[:10]))

    # Clean up merge columns
    df = df.drop(columns=['_merge_key'])
    if 'company_name' in df.columns:
        df = df.drop(columns=['company_name'])
    if 'stock_ticker_y' in df.columns:
        df = df.drop(columns=['stock_ticker_y'])
        df = df.rename(columns={'stock_ticker_x': 'stock_ticker'})

    # Count enriched records
    enriched_count = df['total_news_count'].notna().sum()
    total_articles = df['total_news_count'].sum()
    logger.info("Enriched %d records with news data (%d total articles)", enriched_count, int(total_articles))

    return df


# =============================================================================
# FRED (FEDERAL RESERVE ECONOMIC DATA) — SHARED UTILITIES
# =============================================================================
# Common helpers for all FRED-sourced macroeconomic indicators:
#   VIX, CPI/Inflation, GDP, Unemployment, Interest Rates
# =============================================================================

FRED_START_DATE = "2005-01-01"
FRED_END_DATE = "2025-12-31"


def _fetch_fred_series(series_id: str, value_col: str) -> pd.DataFrame:
    """
    Fetch a single FRED series as a two-column DataFrame [date, value_col].
    Uses the public CSV endpoint (no API key required).
    Retries up to 3 times with exponential backoff on failure.
    """
    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}"
        f"&cosd={FRED_START_DATE}"
        f"&coed={FRED_END_DATE}"
    )
    for attempt in range(3):
        try:
            df = pd.read_csv(url)
            df.columns = ['date', value_col]
            df['date'] = pd.to_datetime(df['date'])
            df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
            df = df.dropna(subset=[value_col])
            return df
        except Exception as e:
            if attempt < 2:
                wait = 2 ** attempt
                logger.warning("FRED fetch for %s failed (%s), retry %d/3 in %ds",
                              series_id, e, attempt + 1, wait)
                time.sleep(wait)
            else:
                raise


def _vectorized_window_avg(source_dates: pd.Series, source_values: pd.Series,
                           target_dates: pd.Series, days_before: int,
                           days_after: int = None) -> pd.Series:
    """
    Compute windowed average of source_values around each target_date.
    Window: [target - days_before, target + days_after].
    If days_after is None, uses a symmetric window (days_after = days_before).
    Uses searchsorted + cumulative sums for O(n log m) performance.
    """
    if days_after is None:
        days_after = days_before

    src_dates = source_dates.values.astype('datetime64[ns]')
    src_vals = source_values.values.astype('float64')
    tgt_dates = target_dates.values.astype('datetime64[ns]')

    left = tgt_dates - np.timedelta64(days_before, 'D')
    right = tgt_dates + np.timedelta64(days_after, 'D')

    left_idx = np.searchsorted(src_dates, left)
    right_idx = np.searchsorted(src_dates, right, side='right')

    # Handle NaN source values in cumulative sums
    valid = ~np.isnan(src_vals)
    clean_vals = np.where(valid, src_vals, 0.0)
    cumsum = np.concatenate([[0], np.cumsum(clean_vals)])
    count_cumsum = np.concatenate([[0], np.cumsum(valid.astype(int))])

    counts = count_cumsum[right_idx] - count_cumsum[left_idx]
    sums = cumsum[right_idx] - cumsum[left_idx]

    result = np.where(counts > 0, sums / counts, np.nan)

    # NaN out rows where target_date is NaT
    nat_mask = np.isnat(tgt_dates)
    result[nat_mask] = np.nan

    return pd.Series(result, index=target_dates.index)


# =============================================================================
# VOLATILITY INDEX (VIX) FROM FEDERAL RESERVE
# =============================================================================
# Fetches the CBOE Volatility Index (VIX) from FRED (Federal Reserve Economic Data)
# VIX measures expected market volatility over the next 30 days
# Series: VIXCLS (CBOE Volatility Index: VIX)
# =============================================================================


def fetch_vix_data() -> pd.DataFrame:
    """
    Fetch VIX (Volatility Index) data from FRED.
    Returns daily VIX values from 2005-2025.
    """
    logger.info("Fetching VIX Data from Federal Reserve (FRED)")
    logger.info("Date range: %s to %s", FRED_START_DATE, FRED_END_DATE)

    try:
        vix_df = _fetch_fred_series("VIXCLS", "vix_close")

        logger.info("Retrieved %d daily VIX observations (%s to %s)",
                    len(vix_df), vix_df['date'].min().strftime('%Y-%m-%d'), vix_df['date'].max().strftime('%Y-%m-%d'))
        logger.info("VIX range: %.2f to %.2f, mean: %.2f",
                    vix_df['vix_close'].min(), vix_df['vix_close'].max(), vix_df['vix_close'].mean())

        return vix_df

    except Exception as e:
        logger.error("Error fetching VIX data: %s", e)
        return pd.DataFrame()


def enrich_with_vix_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich breach data with VIX (Volatility Index) metrics.
    Uses vectorized merge_asof for point-in-time lookups and
    searchsorted-based window averages.
    """
    vix_df = fetch_vix_data()
    vix_cols = ['vix_at_breach', 'vix_7d_before', 'vix_30d_before',
                'vix_7d_after', 'vix_30d_after', 'vix_30d_avg', 'vix_90d_avg']

    if vix_df.empty:
        logger.warning("No VIX data available, skipping enrichment")
        for col in vix_cols:
            df[col] = None
        return df

    logger.info("Calculating VIX metrics for %d breach records...", len(df))
    vix_df = vix_df.sort_values('date').reset_index(drop=True)

    has_date = df['breach_date'].notna()
    for col in vix_cols:
        df[col] = np.nan

    if has_date.any():
        work = df.loc[has_date, ['breach_date']].copy().sort_values('breach_date')

        # vix_at_breach — nearest trading day
        m = pd.merge_asof(work, vix_df, left_on='breach_date', right_on='date', direction='nearest')
        m.index = work.index
        df.loc[m.index, 'vix_at_breach'] = m['vix_close'].round(2).values

        # vix_7d_before — last observation on or before breach_date - 7d
        work['_d'] = work['breach_date'] - pd.Timedelta(days=7)
        _sorted = work[['_d']].sort_values('_d')
        m = pd.merge_asof(_sorted, vix_df, left_on='_d', right_on='date', direction='backward')
        m.index = _sorted.index
        df.loc[m.index, 'vix_7d_before'] = m['vix_close'].round(2).values

        # vix_30d_before — last observation on or before breach_date - 30d
        work['_d'] = work['breach_date'] - pd.Timedelta(days=30)
        _sorted = work[['_d']].sort_values('_d')
        m = pd.merge_asof(_sorted, vix_df, left_on='_d', right_on='date', direction='backward')
        m.index = _sorted.index
        df.loc[m.index, 'vix_30d_before'] = m['vix_close'].round(2).values

        # vix_7d_after — first observation on or after breach_date + 7d
        work['_d'] = work['breach_date'] + pd.Timedelta(days=7)
        _sorted = work[['_d']].sort_values('_d')
        m = pd.merge_asof(_sorted, vix_df, left_on='_d', right_on='date', direction='forward')
        m.index = _sorted.index
        df.loc[m.index, 'vix_7d_after'] = m['vix_close'].round(2).values

        # vix_30d_after — first observation on or after breach_date + 30d
        work['_d'] = work['breach_date'] + pd.Timedelta(days=30)
        _sorted = work[['_d']].sort_values('_d')
        m = pd.merge_asof(_sorted, vix_df, left_on='_d', right_on='date', direction='forward')
        m.index = _sorted.index
        df.loc[m.index, 'vix_30d_after'] = m['vix_close'].round(2).values

    # Window averages — vectorized via searchsorted
    df['vix_30d_avg'] = _vectorized_window_avg(
        vix_df['date'], vix_df['vix_close'], df['breach_date'], days_before=15
    ).round(2)
    df['vix_90d_avg'] = _vectorized_window_avg(
        vix_df['date'], vix_df['vix_close'], df['breach_date'], days_before=45
    ).round(2)

    enriched_count = df['vix_at_breach'].notna().sum()
    logger.info("Enriched %d records with VIX data", enriched_count)
    if enriched_count > 0:
        logger.info("VIX at breach dates: mean=%.2f, min=%.2f, max=%.2f",
                    df['vix_at_breach'].mean(), df['vix_at_breach'].min(), df['vix_at_breach'].max())

    return df


# =============================================================================
# INFLATION (CPI) FROM FEDERAL RESERVE
# =============================================================================
# Fetches Consumer Price Index for All Urban Consumers (CPIAUCSL) from FRED
# Monthly frequency — YoY inflation computed as 12-month pct change
# Series: CPIAUCSL
# =============================================================================

def fetch_inflation_data() -> pd.DataFrame:
    """
    Fetch CPI data from FRED and compute year-over-year inflation rate.
    Returns monthly CPI with a derived inflation_yoy column.
    """
    logger.info("Fetching CPI / Inflation data from FRED (CPIAUCSL)")
    try:
        cpi_df = _fetch_fred_series("CPIAUCSL", "cpi")
        cpi_df = cpi_df.sort_values('date').reset_index(drop=True)
        cpi_df['inflation_yoy'] = cpi_df['cpi'].pct_change(12) * 100
        logger.info("Retrieved %d monthly CPI observations", len(cpi_df))
        return cpi_df
    except Exception as e:
        logger.error("Error fetching CPI data: %s", e)
        return pd.DataFrame()


def enrich_with_inflation_data(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich breach data with CPI / inflation metrics using vectorized merge_asof."""
    cpi_df = fetch_inflation_data()
    cols = ['cpi_at_breach', 'inflation_yoy_at_breach', 'inflation_yoy_3m_avg', 'inflation_yoy_12m_avg']

    if cpi_df.empty:
        logger.warning("No CPI data available, skipping inflation enrichment")
        for col in cols:
            df[col] = None
        return df

    logger.info("Calculating inflation metrics for %d breach records...", len(df))

    # Precompute rolling averages on source data
    cpi_df['inflation_yoy_3m_avg'] = cpi_df['inflation_yoy'].rolling(3, min_periods=1).mean()
    cpi_df['inflation_yoy_12m_avg'] = cpi_df['inflation_yoy'].rolling(12, min_periods=1).mean()

    has_date = df['breach_date'].notna()
    for col in cols:
        df[col] = np.nan

    if has_date.any():
        work = df.loc[has_date, ['breach_date']].copy().sort_values('breach_date')
        m = pd.merge_asof(
            work, cpi_df,
            left_on='breach_date', right_on='date',
            direction='backward', tolerance=pd.Timedelta(days=45)
        )
        m.index = work.index
        df.loc[m.index, 'cpi_at_breach'] = m['cpi'].round(2).values
        df.loc[m.index, 'inflation_yoy_at_breach'] = m['inflation_yoy'].round(2).values
        df.loc[m.index, 'inflation_yoy_3m_avg'] = m['inflation_yoy_3m_avg'].round(2).values
        df.loc[m.index, 'inflation_yoy_12m_avg'] = m['inflation_yoy_12m_avg'].round(2).values

    enriched = df['cpi_at_breach'].notna().sum()
    logger.info("Enriched %d records with inflation data", enriched)
    return df


# =============================================================================
# GDP GROWTH FROM FEDERAL RESERVE
# =============================================================================
# Fetches Real GDP growth rate (% change, seasonally adjusted annual rate)
# Quarterly frequency — FRED series A191RL1Q225SBEA
# =============================================================================

def fetch_gdp_data() -> pd.DataFrame:
    """Fetch quarterly real GDP growth rate from FRED."""
    logger.info("Fetching GDP growth data from FRED (A191RL1Q225SBEA)")
    try:
        gdp_df = _fetch_fred_series("A191RL1Q225SBEA", "gdp_growth")
        gdp_df = gdp_df.sort_values('date').reset_index(drop=True)
        logger.info("Retrieved %d quarterly GDP observations", len(gdp_df))
        return gdp_df
    except Exception as e:
        logger.error("Error fetching GDP data: %s", e)
        return pd.DataFrame()


def enrich_with_gdp_data(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich breach data with GDP growth metrics using vectorized merge_asof."""
    gdp_df = fetch_gdp_data()
    cols = ['gdp_growth_at_breach', 'gdp_growth_4q_avg']

    if gdp_df.empty:
        logger.warning("No GDP data available, skipping GDP enrichment")
        for col in cols:
            df[col] = None
        return df

    logger.info("Calculating GDP metrics for %d breach records...", len(df))

    # Precompute 4-quarter rolling average
    gdp_df['gdp_growth_4q_avg'] = gdp_df['gdp_growth'].rolling(4, min_periods=1).mean()

    has_date = df['breach_date'].notna()
    for col in cols:
        df[col] = np.nan

    if has_date.any():
        work = df.loc[has_date, ['breach_date']].copy().sort_values('breach_date')
        m = pd.merge_asof(
            work, gdp_df,
            left_on='breach_date', right_on='date',
            direction='backward', tolerance=pd.Timedelta(days=120)
        )
        m.index = work.index
        df.loc[m.index, 'gdp_growth_at_breach'] = m['gdp_growth'].round(2).values
        df.loc[m.index, 'gdp_growth_4q_avg'] = m['gdp_growth_4q_avg'].round(2).values

    enriched = df['gdp_growth_at_breach'].notna().sum()
    logger.info("Enriched %d records with GDP data", enriched)
    return df


# =============================================================================
# UNEMPLOYMENT RATE FROM FEDERAL RESERVE
# =============================================================================
# Fetches civilian unemployment rate (monthly, seasonally adjusted)
# FRED series: UNRATE
# =============================================================================

def fetch_unemployment_data() -> pd.DataFrame:
    """Fetch monthly unemployment rate from FRED."""
    logger.info("Fetching unemployment data from FRED (UNRATE)")
    try:
        ur_df = _fetch_fred_series("UNRATE", "unemployment_rate")
        ur_df = ur_df.sort_values('date').reset_index(drop=True)
        logger.info("Retrieved %d monthly unemployment observations", len(ur_df))
        return ur_df
    except Exception as e:
        logger.error("Error fetching unemployment data: %s", e)
        return pd.DataFrame()


def enrich_with_unemployment_data(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich breach data with unemployment rate metrics using vectorized merge_asof."""
    ur_df = fetch_unemployment_data()
    cols = ['unemployment_at_breach', 'unemployment_3m_avg', 'unemployment_12m_avg']

    if ur_df.empty:
        logger.warning("No unemployment data available, skipping enrichment")
        for col in cols:
            df[col] = None
        return df

    logger.info("Calculating unemployment metrics for %d breach records...", len(df))

    # Precompute rolling averages on source data
    ur_df['unemployment_3m_avg'] = ur_df['unemployment_rate'].rolling(3, min_periods=1).mean()
    ur_df['unemployment_12m_avg'] = ur_df['unemployment_rate'].rolling(12, min_periods=1).mean()

    has_date = df['breach_date'].notna()
    for col in cols:
        df[col] = np.nan

    if has_date.any():
        work = df.loc[has_date, ['breach_date']].copy().sort_values('breach_date')
        m = pd.merge_asof(
            work, ur_df,
            left_on='breach_date', right_on='date',
            direction='backward', tolerance=pd.Timedelta(days=45)
        )
        m.index = work.index
        df.loc[m.index, 'unemployment_at_breach'] = m['unemployment_rate'].round(2).values
        df.loc[m.index, 'unemployment_3m_avg'] = m['unemployment_3m_avg'].round(2).values
        df.loc[m.index, 'unemployment_12m_avg'] = m['unemployment_12m_avg'].round(2).values

    enriched = df['unemployment_at_breach'].notna().sum()
    logger.info("Enriched %d records with unemployment data", enriched)
    return df


# =============================================================================
# INTEREST RATES FROM FEDERAL RESERVE
# =============================================================================
# Fetches three daily series from FRED:
#   DFF   — Federal Funds Effective Rate
#   DGS10 — 10-Year Treasury Constant Maturity Rate
#   DGS2  — 2-Year Treasury Constant Maturity Rate
# Yield spread = DGS10 - DGS2 (classic recession indicator)
# =============================================================================

def fetch_interest_rate_data() -> dict:
    """
    Fetch daily interest rate series from FRED.
    Returns a dict of DataFrames keyed by series name.
    """
    logger.info("Fetching interest rate data from FRED (DFF, DGS10, DGS2)")
    series = {
        'fed_funds': ('DFF', 'fed_funds_rate'),
        'treasury_10y': ('DGS10', 'treasury_10y'),
        'treasury_2y': ('DGS2', 'treasury_2y'),
    }
    result = {}
    for key, (series_id, col_name) in series.items():
        try:
            result[key] = _fetch_fred_series(series_id, col_name)
            logger.info("  %s: %d observations", series_id, len(result[key]))
        except Exception as e:
            logger.error("Error fetching %s: %s", series_id, e)
            result[key] = pd.DataFrame()
    return result


def enrich_with_interest_rate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich breach data with interest rate and yield curve metrics using vectorized merge_asof."""
    rate_data = fetch_interest_rate_data()
    cols = [
        'fed_funds_rate_at_breach', 'treasury_10y_at_breach', 'treasury_2y_at_breach',
        'yield_spread_10y2y_at_breach', 'fed_funds_rate_30d_avg', 'treasury_10y_30d_avg',
    ]

    all_empty = all(v.empty for v in rate_data.values())
    if all_empty:
        logger.warning("No interest rate data available, skipping enrichment")
        for col in cols:
            df[col] = None
        return df

    logger.info("Calculating interest rate metrics for %d breach records...", len(df))

    has_date = df['breach_date'].notna()
    for col in cols:
        df[col] = np.nan

    if has_date.any():
        work = df.loc[has_date, ['breach_date']].copy().sort_values('breach_date')

        # Point-in-time values — nearest within 7 days
        series_map = {
            'fed_funds': ('fed_funds_rate', 'fed_funds_rate_at_breach'),
            'treasury_10y': ('treasury_10y', 'treasury_10y_at_breach'),
            'treasury_2y': ('treasury_2y', 'treasury_2y_at_breach'),
        }
        for key, (val_col, at_col) in series_map.items():
            sdf = rate_data.get(key, pd.DataFrame())
            if sdf.empty:
                continue
            sdf = sdf.sort_values('date').reset_index(drop=True)
            m = pd.merge_asof(
                work, sdf, left_on='breach_date', right_on='date',
                direction='nearest', tolerance=pd.Timedelta(days=7)
            )
            m.index = work.index
            df.loc[m.index, at_col] = m[val_col].round(2).values

        # Yield spread = 10Y - 2Y
        t10 = df['treasury_10y_at_breach']
        t2 = df['treasury_2y_at_breach']
        both_valid = t10.notna() & t2.notna()
        df.loc[both_valid, 'yield_spread_10y2y_at_breach'] = (t10[both_valid] - t2[both_valid]).round(2)

    # 30-day trailing averages — vectorized via searchsorted
    avg_series = {
        'fed_funds': ('fed_funds_rate', 'fed_funds_rate_30d_avg'),
        'treasury_10y': ('treasury_10y', 'treasury_10y_30d_avg'),
    }
    for key, (val_col, avg_col) in avg_series.items():
        sdf = rate_data.get(key, pd.DataFrame())
        if sdf.empty:
            continue
        sdf = sdf.sort_values('date').reset_index(drop=True)
        df[avg_col] = _vectorized_window_avg(
            sdf['date'], sdf[val_col], df['breach_date'],
            days_before=30, days_after=0
        ).round(2)

    enriched = df['fed_funds_rate_at_breach'].notna().sum()
    logger.info("Enriched %d records with interest rate data", enriched)
    return df


# =============================================================================
# FINBERT SENTIMENT ANALYSIS
# =============================================================================
# Scores text using ProsusAI/finbert (BERT fine-tuned on financial text).
# Used by model.py Step 6 to analyze breach disclosure tone.
# Results cached to CSV to avoid recomputation (~2 min on first run).
# =============================================================================


def compute_finbert_sentiment(
    texts: pd.Series,
    cache_path: Path = None,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Score texts using ProsusAI/finbert.

    Returns DataFrame with columns:
      sentiment_label, sentiment_score, prob_positive, prob_negative, prob_neutral

    sentiment_score = P(positive) - P(negative), range [-1, +1].
    Caches results to CSV to avoid recomputation.
    """
    if cache_path is None:
        cache_path = OUTPUT_DIR / "sentiment_cache.csv"

    # Check cache
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if len(cached) == len(texts):
            logger.info("Loaded cached sentiment scores (%d rows) from %s", len(cached), cache_path)
            return cached.reset_index(drop=True)
        logger.info("Cache length mismatch (%d vs %d), recomputing", len(cached), len(texts))

    logger.info("Computing FinBERT sentiment for %d texts (batch_size=%d)", len(texts), batch_size)

    from transformers import pipeline as hf_pipeline

    classifier = hf_pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        truncation=True,
        max_length=512,
    )

    # Replace NaN/empty texts with a neutral placeholder
    clean_texts = texts.fillna("").astype(str).tolist()
    clean_texts = [t if t.strip() else "no information available" for t in clean_texts]

    all_results = []
    n_batches = (len(clean_texts) + batch_size - 1) // batch_size
    for i in range(0, len(clean_texts), batch_size):
        batch = clean_texts[i:i + batch_size]
        batch_num = i // batch_size + 1
        if batch_num % 10 == 1 or batch_num == n_batches:
            logger.info("  Batch %d/%d", batch_num, n_batches)
        preds = classifier(batch)
        all_results.extend(preds)

    # Parse results into structured DataFrame
    records = []
    for pred in all_results:
        label = pred['label'].lower()  # positive, negative, neutral
        score = pred['score']

        # FinBERT returns the top label + its probability.
        # We need all three probabilities to compute sentiment_score.
        prob_positive = score if label == 'positive' else 0.0
        prob_negative = score if label == 'negative' else 0.0
        prob_neutral = score if label == 'neutral' else 0.0

        records.append({
            'sentiment_label': label,
            'prob_positive': prob_positive,
            'prob_negative': prob_negative,
            'prob_neutral': prob_neutral,
        })

    result_df = pd.DataFrame(records)

    # For a more precise sentiment_score, re-run with return_all_scores
    # But the single-label approach is much faster and sufficient for regime splits.
    # sentiment_score: +1 = fully positive, -1 = fully negative, ~0 = neutral
    result_df['sentiment_score'] = result_df['prob_positive'] - result_df['prob_negative']

    # Cache results
    OUTPUT_DIR.mkdir(exist_ok=True)
    result_df.to_csv(cache_path, index=False)
    logger.info("Cached sentiment scores to %s", cache_path)

    # Summary
    label_counts = result_df['sentiment_label'].value_counts()
    logger.info("Sentiment distribution: %s", label_counts.to_dict())

    return result_df


def _build_article_text(article: dict, source: str) -> str:
    """Build scoreable text from a news article dict."""
    if source == 'reddit':
        return (article.get('title') or '').strip()
    elif source == 'guardian':
        title = (article.get('title') or '').strip()
        trail = re.sub(r'<[^>]+>', '', (article.get('trail_text') or ''))
        return f"{title}. {trail}".strip('. ') if trail.strip() else title
    elif source == 'nyt':
        title = (article.get('title') or '').strip()
        lead = (article.get('lead_paragraph') or '').strip()
        return f"{title}. {lead}".strip('. ') if lead else title
    return ''


def compute_lagged_news_sentiment(
    df: pd.DataFrame,
    windows: list = None,
    cache_path: Path = None,
) -> pd.DataFrame:
    """
    Compute pre-breach lagged news sentiment for each event.

    Two-level cache strategy:
      1. news_lagged_sentiment.csv (aggregated per event) — return immediately
      2. news_articles_scored.csv (article-level) — skip FinBERT, just re-aggregate
      3. Otherwise: parse JSONs → score with FinBERT → aggregate → cache both

    Returns DataFrame aligned to df.index with columns:
      news_sent_{w}d, news_count_{w}d for each window w
    """
    if windows is None:
        windows = [7, 30, 60]
    if cache_path is None:
        cache_path = OUTPUT_DIR / "news_lagged_sentiment.csv"
    scored_cache = OUTPUT_DIR / "news_articles_scored.csv"

    expected_cols = []
    for w in windows:
        expected_cols += [f'news_sent_{w}d', f'news_count_{w}d']

    # --- Level 1 cache: aggregated results ---
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if len(cached) == len(df) and all(c in cached.columns for c in expected_cols):
            logger.info("Loaded cached lagged news sentiment (%d rows) from %s", len(cached), cache_path)
            return cached[expected_cols].reset_index(drop=True)
        logger.info("Aggregated cache invalid (rows: %d vs %d), recomputing", len(cached), len(df))

    # --- Parse articles from JSON columns ---
    articles_scored = None
    if scored_cache.exists():
        try:
            articles_scored = pd.read_csv(scored_cache)
            if 'sentiment_score' not in articles_scored.columns:
                articles_scored = None
                logger.info("Scored cache missing sentiment_score, will rescore")
            else:
                logger.info("Loaded scored articles cache (%d rows)", len(articles_scored))
        except Exception:
            articles_scored = None

    if articles_scored is None:
        # Parse all articles from JSON columns
        logger.info("Parsing news articles from JSON columns...")
        article_records = []
        json_cols = {
            'reddit_articles_json': 'reddit',
            'guardian_articles_json': 'guardian',
            'nyt_articles_json': 'nyt',
        }

        for row_idx, row in df.iterrows():
            for col, source in json_cols.items():
                raw = row.get(col)
                if pd.isna(raw) or not raw:
                    continue
                try:
                    articles = json.loads(raw) if isinstance(raw, str) else raw
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(articles, list):
                    continue
                for art in articles:
                    if not isinstance(art, dict):
                        continue
                    text = _build_article_text(art, source)
                    if not text:
                        continue
                    pub_date = art.get('published_date') or art.get('pub_date')
                    article_records.append({
                        'row_idx': row_idx,
                        'source': source,
                        'text': text,
                        'published_date': pub_date,
                    })

        logger.info("Parsed %d articles from %d events", len(article_records), len(df))

        if not article_records:
            # No articles found — return empty results
            result = pd.DataFrame(index=df.index)
            for w in windows:
                result[f'news_sent_{w}d'] = np.nan
                result[f'news_count_{w}d'] = 0
            return result[expected_cols].reset_index(drop=True)

        articles_df = pd.DataFrame(article_records)

        # Score all article texts with FinBERT
        logger.info("Scoring %d article texts with FinBERT...", len(articles_df))
        sentiment_results = compute_finbert_sentiment(
            articles_df['text'],
            cache_path=OUTPUT_DIR / "news_article_sentiment_cache.csv",
            batch_size=32,
        )
        articles_df['sentiment_score'] = sentiment_results['sentiment_score'].values
        articles_df['sentiment_label'] = sentiment_results['sentiment_label'].values

        # Cache scored articles
        OUTPUT_DIR.mkdir(exist_ok=True)
        articles_df.to_csv(scored_cache, index=False)
        logger.info("Cached scored articles to %s", scored_cache)
        articles_scored = articles_df

    # --- Aggregate by window ---
    logger.info("Aggregating sentiment by pre-breach windows: %s", windows)

    # Parse dates
    articles_scored['pub_dt'] = pd.to_datetime(articles_scored['published_date'], format='mixed', errors='coerce')

    # Map reported_date from df onto articles
    reported_dates = df['reported_date']
    articles_scored['reported_date'] = articles_scored['row_idx'].map(reported_dates)
    articles_scored['reported_date'] = pd.to_datetime(articles_scored['reported_date'], format='mixed', errors='coerce')

    # Compute days before breach
    articles_scored['days_before'] = (articles_scored['reported_date'] - articles_scored['pub_dt']).dt.days

    result = pd.DataFrame(index=df.index)
    for w in windows:
        in_window = articles_scored[
            (articles_scored['days_before'] >= 1) &
            (articles_scored['days_before'] <= w)
        ]
        agg = in_window.groupby('row_idx')['sentiment_score'].agg(['mean', 'count'])
        agg.columns = [f'news_sent_{w}d', f'news_count_{w}d']
        # Reindex to df.index, fill missing
        agg = agg.reindex(df.index)
        agg[f'news_count_{w}d'] = agg[f'news_count_{w}d'].fillna(0).astype(int)
        result[f'news_sent_{w}d'] = agg[f'news_sent_{w}d']
        result[f'news_count_{w}d'] = agg[f'news_count_{w}d']

    # Cache aggregated results
    OUTPUT_DIR.mkdir(exist_ok=True)
    result.to_csv(cache_path, index=False)
    logger.info("Cached lagged news sentiment to %s", cache_path)

    # Summary
    for w in windows:
        has_data = (result[f'news_count_{w}d'] > 0).sum()
        logger.info("Window %dd: %d/%d events with articles (%.1f%%)",
                     w, has_data, len(df), 100 * has_data / len(df))

    return result[expected_cols].reset_index(drop=True)


def main():
    """Main entry point for the cleaning pipeline."""
    df = load_breach_data("Data_Breach_Enriched_Final.csv")
    logger.info("Columns: %s", list(df.columns))
    logger.info("Shape: %s", df.shape)

    # Clean
    df = clean_pipeline(df)

    # Enrich — each stage adds columns and returns the DataFrame
    df = enrich_with_stock_data(df)
    df = enrich_with_news_data(df)
    df = enrich_with_vix_data(df)
    df = enrich_with_fama_french_data(df)
    df = enrich_with_inflation_data(df)
    df = enrich_with_gdp_data(df)
    df = enrich_with_unemployment_data(df)
    df = enrich_with_interest_rate_data(df)

    save_cleaned_data(df, "breach_data_enriched.csv")
    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
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
            # --- News Data (Reddit, Guardian, NYT, NewsAPI) ---
            "reddit_count": {
                "python_type": "int",
                "sql_type": "INTEGER",
                "pandas_dtype": "Int64",
                "nullable": True,
                "description": "Number of Reddit posts found about the company breach",
            },
            "guardian_count": {
                "python_type": "int",
                "sql_type": "INTEGER",
                "pandas_dtype": "Int64",
                "nullable": True,
                "description": "Number of Guardian articles found about the company breach",
            },
            "nyt_count": {
                "python_type": "int",
                "sql_type": "INTEGER",
                "pandas_dtype": "Int64",
                "nullable": True,
                "description": "Number of New York Times articles found about the company breach",
            },
            "newsapi_count": {
                "python_type": "int",
                "sql_type": "INTEGER",
                "pandas_dtype": "Int64",
                "nullable": True,
                "description": "Number of NewsAPI articles found about the company breach",
            },
            "total_news_count": {
                "python_type": "int",
                "sql_type": "INTEGER",
                "pandas_dtype": "Int64",
                "nullable": True,
                "description": "Total news articles/posts found across all sources",
            },
            "reddit_articles_json": {
                "python_type": "str",
                "sql_type": "JSON",
                "pandas_dtype": "object",
                "nullable": True,
                "description": "JSON array of Reddit posts (title, url, date, subreddit, score, comments)",
            },
            "guardian_articles_json": {
                "python_type": "str",
                "sql_type": "JSON",
                "pandas_dtype": "object",
                "nullable": True,
                "description": "JSON array of Guardian articles (title, url, date, section, byline)",
            },
            "nyt_articles_json": {
                "python_type": "str",
                "sql_type": "JSON",
                "pandas_dtype": "object",
                "nullable": True,
                "description": "JSON array of NYT articles (title, url, date, section, byline, word_count)",
            },
            "newsapi_articles_json": {
                "python_type": "str",
                "sql_type": "JSON",
                "pandas_dtype": "object",
                "nullable": True,
                "description": "JSON array of NewsAPI articles (title, url, date, source, author, description)",
            },
            # --- VIX (Volatility Index) Data from Federal Reserve ---
            "vix_at_breach": {
                "python_type": "float",
                "sql_type": "DECIMAL(8,2)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "CBOE Volatility Index (VIX) at breach date",
            },
            "vix_7d_before": {
                "python_type": "float",
                "sql_type": "DECIMAL(8,2)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "VIX value 7 days before breach",
            },
            "vix_30d_before": {
                "python_type": "float",
                "sql_type": "DECIMAL(8,2)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "VIX value 30 days before breach",
            },
            "vix_7d_after": {
                "python_type": "float",
                "sql_type": "DECIMAL(8,2)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "VIX value 7 days after breach",
            },
            "vix_30d_after": {
                "python_type": "float",
                "sql_type": "DECIMAL(8,2)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "VIX value 30 days after breach",
            },
            "vix_30d_avg": {
                "python_type": "float",
                "sql_type": "DECIMAL(8,2)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "Average VIX in 30-day window around breach (+/- 15 days)",
            },
            "vix_90d_avg": {
                "python_type": "float",
                "sql_type": "DECIMAL(8,2)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "Average VIX in 90-day window around breach (+/- 45 days)",
            },
            # --- Fama-French 5-Factor Data (Kenneth French Data Library) ---
            # Values are in percentage points (1.5 = 1.5% return)
            "ff_mkt_rf": {
                "python_type": "float",
                "sql_type": "DECIMAL(10,4)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "Fama-French market excess return (Mkt-RF) at breach date, in pct points",
            },
            "ff_smb": {
                "python_type": "float",
                "sql_type": "DECIMAL(10,4)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "Fama-French Small Minus Big (SMB) size factor at breach date, in pct points",
            },
            "ff_hml": {
                "python_type": "float",
                "sql_type": "DECIMAL(10,4)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "Fama-French High Minus Low (HML) value factor at breach date, in pct points",
            },
            "ff_rmw": {
                "python_type": "float",
                "sql_type": "DECIMAL(10,4)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "Fama-French Robust Minus Weak (RMW) profitability factor at breach date, in pct points",
            },
            "ff_cma": {
                "python_type": "float",
                "sql_type": "DECIMAL(10,4)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "Fama-French Conservative Minus Aggressive (CMA) investment factor at breach date, in pct points",
            },
            "ff_rf": {
                "python_type": "float",
                "sql_type": "DECIMAL(10,4)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "Risk-free rate (1-month T-bill) at breach date, in pct points",
            },
            "ff_mkt_rf_30d_avg": {
                "python_type": "float",
                "sql_type": "DECIMAL(10,4)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "Average Mkt-RF in 30-day window around breach (+/- 15 days)",
            },
            "ff_smb_30d_avg": {
                "python_type": "float",
                "sql_type": "DECIMAL(10,4)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "Average SMB in 30-day window around breach (+/- 15 days)",
            },
            "ff_hml_30d_avg": {
                "python_type": "float",
                "sql_type": "DECIMAL(10,4)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "Average HML in 30-day window around breach (+/- 15 days)",
            },
            "ff_rmw_30d_avg": {
                "python_type": "float",
                "sql_type": "DECIMAL(10,4)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "Average RMW in 30-day window around breach (+/- 15 days)",
            },
            "ff_cma_30d_avg": {
                "python_type": "float",
                "sql_type": "DECIMAL(10,4)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "Average CMA in 30-day window around breach (+/- 15 days)",
            },
            "ff_mkt_rf_90d_avg": {
                "python_type": "float",
                "sql_type": "DECIMAL(10,4)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "Average Mkt-RF in 90-day window around breach (+/- 45 days)",
            },
            "ff_smb_90d_avg": {
                "python_type": "float",
                "sql_type": "DECIMAL(10,4)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "Average SMB in 90-day window around breach (+/- 45 days)",
            },
            "ff_hml_90d_avg": {
                "python_type": "float",
                "sql_type": "DECIMAL(10,4)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "Average HML in 90-day window around breach (+/- 45 days)",
            },
            "ff_rmw_90d_avg": {
                "python_type": "float",
                "sql_type": "DECIMAL(10,4)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "Average RMW in 90-day window around breach (+/- 45 days)",
            },
            "ff_cma_90d_avg": {
                "python_type": "float",
                "sql_type": "DECIMAL(10,4)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "Average CMA in 90-day window around breach (+/- 45 days)",
            },
            # --- Inflation / CPI Data from Federal Reserve (CPIAUCSL, monthly) ---
            "cpi_at_breach": {
                "python_type": "float",
                "sql_type": "DECIMAL(10,2)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "Consumer Price Index (CPI-U) at most recent month before breach",
            },
            "inflation_yoy_at_breach": {
                "python_type": "float",
                "sql_type": "DECIMAL(8,2)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "Year-over-year CPI inflation rate (%) at breach date",
            },
            "inflation_yoy_3m_avg": {
                "python_type": "float",
                "sql_type": "DECIMAL(8,2)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "3-month average of YoY inflation rate ending at breach month",
            },
            "inflation_yoy_12m_avg": {
                "python_type": "float",
                "sql_type": "DECIMAL(8,2)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "12-month average of YoY inflation rate ending at breach month",
            },
            # --- GDP Growth Data from Federal Reserve (A191RL1Q225SBEA, quarterly) ---
            "gdp_growth_at_breach": {
                "python_type": "float",
                "sql_type": "DECIMAL(8,2)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "Real GDP growth rate (% change SAAR) at most recent quarter before breach",
            },
            "gdp_growth_4q_avg": {
                "python_type": "float",
                "sql_type": "DECIMAL(8,2)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "4-quarter average of real GDP growth rate ending at breach quarter",
            },
            # --- Unemployment Rate from Federal Reserve (UNRATE, monthly) ---
            "unemployment_at_breach": {
                "python_type": "float",
                "sql_type": "DECIMAL(6,2)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "Civilian unemployment rate (%) at most recent month before breach",
            },
            "unemployment_3m_avg": {
                "python_type": "float",
                "sql_type": "DECIMAL(6,2)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "3-month average unemployment rate ending at breach month",
            },
            "unemployment_12m_avg": {
                "python_type": "float",
                "sql_type": "DECIMAL(6,2)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "12-month average unemployment rate ending at breach month",
            },
            # --- Interest Rates from Federal Reserve (DFF, DGS10, DGS2, daily) ---
            "fed_funds_rate_at_breach": {
                "python_type": "float",
                "sql_type": "DECIMAL(6,2)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "Federal Funds Effective Rate (%) at breach date",
            },
            "treasury_10y_at_breach": {
                "python_type": "float",
                "sql_type": "DECIMAL(6,2)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "10-Year Treasury Constant Maturity Rate (%) at breach date",
            },
            "treasury_2y_at_breach": {
                "python_type": "float",
                "sql_type": "DECIMAL(6,2)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "2-Year Treasury Constant Maturity Rate (%) at breach date",
            },
            "yield_spread_10y2y_at_breach": {
                "python_type": "float",
                "sql_type": "DECIMAL(6,2)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "Yield spread (10Y - 2Y Treasury), negative values signal potential recession",
            },
            "fed_funds_rate_30d_avg": {
                "python_type": "float",
                "sql_type": "DECIMAL(6,2)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "30-day average Federal Funds Rate ending at breach date",
            },
            "treasury_10y_30d_avg": {
                "python_type": "float",
                "sql_type": "DECIMAL(6,2)",
                "pandas_dtype": "float64",
                "nullable": True,
                "description": "30-day average 10-Year Treasury Rate ending at breach date",
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
