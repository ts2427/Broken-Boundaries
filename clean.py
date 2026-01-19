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
import json
from pathlib import Path
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv

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
        print(f"Warning: Could not initialize Reddit client: {e}")


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
        pass  # Fail silently for Reddit

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

        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
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
        print(f"    NewsAPI error for {company_name}: {e}")

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

        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
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
        print(f"    Guardian error for {company_name}: {e}")

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

        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
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
        print(f"    NYT error for {company_name}: {e}")

    return articles


def fetch_all_news_for_company(company_name: str, ticker: str = None) -> dict:
    """Fetch news from all sources for a single company."""
    # Use company name for search
    search_term = company_name

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
    print(f"\n=== Fetching News Data (Reddit, Guardian, NYT) ===\n")
    print(f"Unique companies to search: {len(companies)}")
    print(f"Date range: {NEWS_START_DATE} to {NEWS_END_DATE}")

    # Check API key status
    print(f"\nAPI Status:")
    print(f"  - Reddit: {'Available (PRAW)' if REDDIT_CLIENT else 'MISSING (set REDDIT_CLIENT_ID & REDDIT_CLIENT_SECRET)'}")
    print(f"  - Guardian: {'Available' if GUARDIAN_API_KEY else 'MISSING KEY (set GUARDIAN_API_KEY)'}")
    print(f"  - NYT: {'Available' if NYT_API_KEY else 'MISSING KEY (set NYT_API_KEY)'}")
    print(f"  - NewsAPI: {'Available' if NEWSAPI_KEY else 'MISSING KEY (set NEWSAPI_KEY)'}")
    print()

    news_data = []

    for i, company in enumerate(companies):
        company_name = company.get("name", "")
        ticker = company.get("ticker", "")

        if not company_name or company_name == 'nan':
            continue

        print(f"  [{i+1}/{len(companies)}] {company_name[:40]}...", end=" ")

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
        print(f"R:{news['reddit_count']} G:{news['guardian_count']} N:{news['nyt_count']} A:{news['newsapi_count']}")

    print(f"\nNews fetch complete.")

    if news_data:
        return pd.DataFrame(news_data)
    return pd.DataFrame()


def enrich_with_news_data(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich breach data with news articles from multiple sources."""
    # Get unique company/ticker combinations
    companies = df[['org_name', 'stock_ticker']].drop_duplicates()
    company_list = [
        {"name": row['org_name'], "ticker": row['stock_ticker']}
        for _, row in companies.iterrows()
    ]

    if not company_list:
        print("No companies found, skipping news enrichment")
        return df

    # Fetch news data
    news_df = fetch_news_for_companies(company_list)

    if news_df.empty:
        print("No news data retrieved, skipping enrichment")
        return df

    # Merge news data with breach data on company name
    df = df.merge(
        news_df,
        left_on='org_name',
        right_on='company_name',
        how='left'
    )

    # Clean up duplicate columns
    if 'company_name' in df.columns:
        df = df.drop(columns=['company_name'])
    if 'stock_ticker_y' in df.columns:
        df = df.drop(columns=['stock_ticker_y'])
        df = df.rename(columns={'stock_ticker_x': 'stock_ticker'})

    # Count enriched records
    enriched_count = df['total_news_count'].notna().sum()
    total_articles = df['total_news_count'].sum()
    print(f"\nEnriched {enriched_count} records with news data ({int(total_articles)} total articles)")

    return df


# =============================================================================
# VOLATILITY INDEX (VIX) FROM FEDERAL RESERVE
# =============================================================================
# Fetches the CBOE Volatility Index (VIX) from FRED (Federal Reserve Economic Data)
# VIX measures expected market volatility over the next 30 days
# Series: VIXCLS (CBOE Volatility Index: VIX)
# Date range: January 1, 2005 - December 31, 2025
# =============================================================================

VIX_START_DATE = "2005-01-01"
VIX_END_DATE = "2025-12-31"


def fetch_vix_data() -> pd.DataFrame:
    """
    Fetch VIX (Volatility Index) data from FRED.
    Returns daily VIX values from 2005-2025.
    Uses FRED's public API endpoint (no API key required for basic access).
    """
    print(f"\n=== Fetching VIX Data from Federal Reserve (FRED) ===\n")
    print(f"Date range: {VIX_START_DATE} to {VIX_END_DATE}")

    try:
        # Use FRED's public CSV download endpoint
        print("Fetching VIX data from FRED...")
        url = (
            f"https://fred.stlouisfed.org/graph/fredgraph.csv"
            f"?id=VIXCLS"
            f"&cosd={VIX_START_DATE}"
            f"&coed={VIX_END_DATE}"
        )

        vix_df = pd.read_csv(url)
        vix_df.columns = ['date', 'vix_close']

        # Clean the data
        vix_df['date'] = pd.to_datetime(vix_df['date'])
        vix_df['vix_close'] = pd.to_numeric(vix_df['vix_close'], errors='coerce')

        # Remove any rows with missing VIX values (FRED uses '.' for missing)
        vix_df = vix_df.dropna(subset=['vix_close'])

        print(f"Retrieved {len(vix_df)} daily VIX observations")
        print(f"Date range: {vix_df['date'].min().strftime('%Y-%m-%d')} to {vix_df['date'].max().strftime('%Y-%m-%d')}")
        print(f"VIX range: {vix_df['vix_close'].min():.2f} to {vix_df['vix_close'].max():.2f}")
        print(f"VIX mean: {vix_df['vix_close'].mean():.2f}")

        return vix_df

    except Exception as e:
        print(f"Error fetching VIX data: {e}")
        return pd.DataFrame()


def calculate_vix_metrics(vix_df: pd.DataFrame, breach_date: pd.Timestamp) -> dict:
    """
    Calculate VIX metrics around a breach date.
    Returns VIX values at breach date and surrounding periods.
    """
    if vix_df.empty or pd.isna(breach_date):
        return {
            'vix_at_breach': None,
            'vix_7d_before': None,
            'vix_30d_before': None,
            'vix_7d_after': None,
            'vix_30d_after': None,
            'vix_30d_avg': None,
            'vix_90d_avg': None,
        }

    # Convert breach_date to datetime if needed
    if isinstance(breach_date, str):
        breach_date = pd.to_datetime(breach_date)

    # Find closest VIX value to breach date
    vix_df_sorted = vix_df.copy()
    vix_df_sorted['date'] = pd.to_datetime(vix_df_sorted['date'])

    # VIX at breach date (or closest available)
    closest_idx = (vix_df_sorted['date'] - breach_date).abs().idxmin()
    vix_at_breach = vix_df_sorted.loc[closest_idx, 'vix_close']

    # VIX 7 days before
    date_7d_before = breach_date - pd.Timedelta(days=7)
    mask_7d_before = vix_df_sorted['date'] <= date_7d_before
    vix_7d_before = vix_df_sorted[mask_7d_before]['vix_close'].iloc[-1] if mask_7d_before.any() else None

    # VIX 30 days before
    date_30d_before = breach_date - pd.Timedelta(days=30)
    mask_30d_before = vix_df_sorted['date'] <= date_30d_before
    vix_30d_before = vix_df_sorted[mask_30d_before]['vix_close'].iloc[-1] if mask_30d_before.any() else None

    # VIX 7 days after
    date_7d_after = breach_date + pd.Timedelta(days=7)
    mask_7d_after = vix_df_sorted['date'] >= date_7d_after
    vix_7d_after = vix_df_sorted[mask_7d_after]['vix_close'].iloc[0] if mask_7d_after.any() else None

    # VIX 30 days after
    date_30d_after = breach_date + pd.Timedelta(days=30)
    mask_30d_after = vix_df_sorted['date'] >= date_30d_after
    vix_30d_after = vix_df_sorted[mask_30d_after]['vix_close'].iloc[0] if mask_30d_after.any() else None

    # 30-day average around breach
    mask_30d_window = (
        (vix_df_sorted['date'] >= breach_date - pd.Timedelta(days=15)) &
        (vix_df_sorted['date'] <= breach_date + pd.Timedelta(days=15))
    )
    vix_30d_avg = vix_df_sorted[mask_30d_window]['vix_close'].mean() if mask_30d_window.any() else None

    # 90-day average around breach
    mask_90d_window = (
        (vix_df_sorted['date'] >= breach_date - pd.Timedelta(days=45)) &
        (vix_df_sorted['date'] <= breach_date + pd.Timedelta(days=45))
    )
    vix_90d_avg = vix_df_sorted[mask_90d_window]['vix_close'].mean() if mask_90d_window.any() else None

    return {
        'vix_at_breach': round(vix_at_breach, 2) if vix_at_breach else None,
        'vix_7d_before': round(vix_7d_before, 2) if vix_7d_before else None,
        'vix_30d_before': round(vix_30d_before, 2) if vix_30d_before else None,
        'vix_7d_after': round(vix_7d_after, 2) if vix_7d_after else None,
        'vix_30d_after': round(vix_30d_after, 2) if vix_30d_after else None,
        'vix_30d_avg': round(vix_30d_avg, 2) if vix_30d_avg else None,
        'vix_90d_avg': round(vix_90d_avg, 2) if vix_90d_avg else None,
    }


def enrich_with_vix_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich breach data with VIX (Volatility Index) metrics.
    Adds VIX values at breach date and surrounding periods.
    """
    # Fetch VIX data
    vix_df = fetch_vix_data()

    if vix_df.empty:
        print("No VIX data available, skipping enrichment")
        # Add empty columns
        for col in ['vix_at_breach', 'vix_7d_before', 'vix_30d_before',
                    'vix_7d_after', 'vix_30d_after', 'vix_30d_avg', 'vix_90d_avg']:
            df[col] = None
        return df

    print(f"\nCalculating VIX metrics for {len(df)} breach records...")

    # Calculate VIX metrics for each breach
    vix_metrics_list = []
    for idx, row in df.iterrows():
        breach_date = row.get('breach_date')
        metrics = calculate_vix_metrics(vix_df, breach_date)
        vix_metrics_list.append(metrics)

        if (idx + 1) % 200 == 0:
            print(f"  Processed {idx + 1}/{len(df)} records")

    # Convert to DataFrame and merge
    vix_metrics_df = pd.DataFrame(vix_metrics_list)

    # Add VIX columns to main dataframe
    for col in vix_metrics_df.columns:
        df[col] = vix_metrics_df[col].values

    # Count enriched records
    enriched_count = df['vix_at_breach'].notna().sum()
    print(f"\nEnriched {enriched_count} records with VIX data")

    # Print summary statistics
    print(f"\nVIX Summary at Breach Dates:")
    print(f"  Mean VIX: {df['vix_at_breach'].mean():.2f}")
    print(f"  Min VIX: {df['vix_at_breach'].min():.2f}")
    print(f"  Max VIX: {df['vix_at_breach'].max():.2f}")

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

    # Enrich with news data from Reddit, Guardian, NYT
    df_with_news = enrich_with_news_data(df_enriched)

    # Enrich with VIX (Volatility Index) data from Federal Reserve
    df_with_vix = enrich_with_vix_data(df_with_news)

    # Save enriched data
    save_cleaned_data(df_with_vix, "breach_data_enriched.csv")

    return df_with_vix


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
