"""
model.py - Data Models and Analysis for Breach Data
Provides statistical analysis, predictive modeling, and data insights.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import logging

from database import (
    query_df,
    get_summary_stats,
    get_sector_breakdown,
    get_yearly_breakdown,
    DB_PATH,
)

logger = logging.getLogger(__name__)

# =============================================================================
# DATA LOADING
# =============================================================================

CLEANED_DATA_DIR = Path(__file__).parent / "cleaned"


def load_enriched_data(filename: str = "breach_data_enriched.csv") -> pd.DataFrame:
    """
    Load the enriched breach data produced by clean.py.
    This is the primary entry point for model.py — all analysis starts here.
    """
    filepath = CLEANED_DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Cleaned data not found at {filepath}. Run clean.py first."
        )
    try:
        df = pd.read_csv(filepath, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding="latin-1")

    # Parse date columns
    for col in ["reported_date", "breach_date", "end_breach_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="mixed", errors="coerce")

    # Parse numeric columns that may have been read as strings
    numeric_cols = [
        "total_affected", "cik", "sic", "naics",
        "yf_market_cap", "yf_enterprise_value", "yf_employees",
        "yf_current_price", "yf_52week_high", "yf_52week_low",
        "yf_avg_volume", "yf_dividend_yield", "yf_beta",
        "yf_pe_ratio", "yf_forward_pe", "yf_profit_margin",
        "yf_revenue", "yf_gross_profit", "yf_ebitda",
        "yf_total_debt", "yf_total_cash",
        "reddit_count", "guardian_count", "nyt_count",
        "newsapi_count", "total_news_count",
        "vix_at_breach", "vix_7d_before", "vix_30d_before",
        "vix_7d_after", "vix_30d_after", "vix_30d_avg", "vix_90d_avg",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("Loaded %d records with %d columns from %s", len(df), len(df.columns), filename)
    return df


# =============================================================================
# DATA MODELS
# =============================================================================

class BreachSeverity(Enum):
    """Classification of breach severity based on affected individuals."""
    LOW = "low"           # < 10,000
    MEDIUM = "medium"     # 10,000 - 100,000
    HIGH = "high"         # 100,000 - 1,000,000
    CRITICAL = "critical" # 1,000,000 - 10,000,000
    MASSIVE = "massive"   # > 10,000,000


@dataclass
class BreachRiskScore:
    """Risk assessment for a breach incident."""
    overall_score: float        # 0-100
    severity_score: float       # Based on affected count
    exposure_score: float       # Based on breach duration
    media_score: float          # Based on news coverage
    sector_risk: float          # Based on sector sensitivity
    risk_level: str             # Low, Medium, High, Critical


@dataclass
class SectorRiskProfile:
    """Risk profile for a sector."""
    sector: str
    total_breaches: int
    total_affected: int
    avg_affected: float
    breach_frequency: float     # Breaches per year
    media_attention: float      # Avg news coverage
    risk_score: float


# =============================================================================
# SEVERITY CLASSIFICATION
# =============================================================================

def classify_severity(affected: int) -> BreachSeverity:
    """Classify breach severity based on number of affected individuals."""
    if affected is None or affected < 10000:
        return BreachSeverity.LOW
    elif affected < 100000:
        return BreachSeverity.MEDIUM
    elif affected < 1000000:
        return BreachSeverity.HIGH
    elif affected < 10000000:
        return BreachSeverity.CRITICAL
    else:
        return BreachSeverity.MASSIVE


def get_severity_distribution(db_path: Optional[Path] = None) -> Dict[str, int]:
    """Get distribution of breaches by severity level."""
    df = query_df("SELECT total_affected FROM breach_incidents", db_path=db_path)

    distribution = {s.value: 0 for s in BreachSeverity}

    for affected in df['total_affected']:
        severity = classify_severity(affected)
        distribution[severity.value] += 1

    return distribution


# =============================================================================
# RISK SCORING
# =============================================================================

def calculate_severity_score(affected: int, max_affected: int = 1_000_000_000) -> float:
    """Calculate severity score (0-100) based on affected individuals."""
    if affected is None or affected <= 0:
        return 0.0
    # Log scale to handle wide range of values
    log_affected = np.log10(affected + 1)
    log_max = np.log10(max_affected + 1)
    return min(100.0, (log_affected / log_max) * 100)


def calculate_exposure_score(breach_date: str, end_date: str = None) -> float:
    """Calculate exposure score based on breach duration."""
    if not breach_date:
        return 50.0  # Default middle score if unknown

    try:
        start = datetime.strptime(breach_date, '%Y-%m-%d')
        if end_date:
            end = datetime.strptime(end_date, '%Y-%m-%d')
            duration_days = (end - start).days
        else:
            duration_days = 30  # Assume 30 days if unknown

        # Score based on duration (longer = higher risk)
        if duration_days <= 7:
            return 20.0
        elif duration_days <= 30:
            return 40.0
        elif duration_days <= 90:
            return 60.0
        elif duration_days <= 365:
            return 80.0
        else:
            return 100.0
    except:
        return 50.0


def calculate_media_score(news_count: int, max_news: int = 500) -> float:
    """Calculate media attention score (0-100)."""
    if news_count is None or news_count <= 0:
        return 0.0
    return min(100.0, (news_count / max_news) * 100)


SECTOR_SENSITIVITY = {
    'Financial Services': 0.95,
    'Healthcare': 0.90,
    'Technology': 0.80,
    'Communication Services': 0.75,
    'Consumer Cyclical': 0.60,
    'Consumer Defensive': 0.55,
    'Industrials': 0.50,
    'Energy': 0.45,
    'Basic Materials': 0.40,
    'Utilities': 0.35,
    'Real Estate': 0.30,
}


def calculate_sector_risk(sector: str) -> float:
    """Get sector risk multiplier (0-1)."""
    return SECTOR_SENSITIVITY.get(sector, 0.5) * 100


def calculate_risk_score(
    affected: int,
    breach_date: str = None,
    end_date: str = None,
    news_count: int = 0,
    sector: str = None
) -> BreachRiskScore:
    """
    Calculate comprehensive risk score for a breach.

    Returns BreachRiskScore with component scores and overall assessment.
    """
    severity = calculate_severity_score(affected)
    exposure = calculate_exposure_score(breach_date, end_date)
    media = calculate_media_score(news_count)
    sector_risk = calculate_sector_risk(sector) if sector else 50.0

    # Weighted average for overall score
    weights = {
        'severity': 0.40,
        'exposure': 0.20,
        'media': 0.15,
        'sector': 0.25,
    }

    overall = (
        severity * weights['severity'] +
        exposure * weights['exposure'] +
        media * weights['media'] +
        sector_risk * weights['sector']
    )

    # Determine risk level
    if overall >= 80:
        risk_level = "Critical"
    elif overall >= 60:
        risk_level = "High"
    elif overall >= 40:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return BreachRiskScore(
        overall_score=round(overall, 2),
        severity_score=round(severity, 2),
        exposure_score=round(exposure, 2),
        media_score=round(media, 2),
        sector_risk=round(sector_risk, 2),
        risk_level=risk_level
    )


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def get_descriptive_stats(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Get descriptive statistics for breach data."""
    df = query_df("""
        SELECT total_affected, total_news_count,
               breach_date, yf_sector, breach_type
        FROM breach_incidents
        WHERE total_affected IS NOT NULL
    """, db_path=db_path)

    stats = {
        'count': len(df),
        'affected': {
            'mean': df['total_affected'].mean(),
            'median': df['total_affected'].median(),
            'std': df['total_affected'].std(),
            'min': df['total_affected'].min(),
            'max': df['total_affected'].max(),
            'q25': df['total_affected'].quantile(0.25),
            'q75': df['total_affected'].quantile(0.75),
        },
        'news_coverage': {
            'mean': df['total_news_count'].mean(),
            'median': df['total_news_count'].median(),
            'max': df['total_news_count'].max(),
        },
        'sectors': df['yf_sector'].nunique(),
        'breach_types': df['breach_type'].nunique(),
    }

    return stats


def run_full_descriptive_statistics(df: Optional[pd.DataFrame] = None, db_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Run comprehensive descriptive statistics on ALL data columns.
    Returns detailed statistics for numeric, categorical, and date columns.

    Args:
        df: DataFrame to analyze. If None, loads from database or CSV.
        db_path: Database path (used only if df is None).
    """
    if df is None:
        try:
            df = load_enriched_data()
        except FileNotFoundError:
            df = query_df("SELECT * FROM breach_incidents", db_path=db_path)

    results = {
        'overview': {},
        'numeric': {},
        'categorical': {},
        'date': {},
        'news_sources': {},
        'stock_data': {},
        'missing_data_summary': {},
    }

    # === OVERVIEW ===
    results['overview'] = {
        'total_records': len(df),
        'total_columns': len(df.columns),
        'column_names': list(df.columns),
        'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        'duplicate_rows': int(df.duplicated().sum()),
    }

    # === NUMERIC COLUMNS (matched to actual enriched CSV columns) ===
    numeric_cols = [
        'total_affected', 'total_news_count',
        'reddit_count', 'guardian_count', 'nyt_count', 'newsapi_count',
        'yf_market_cap', 'yf_enterprise_value', 'yf_employees',
        'yf_revenue', 'yf_gross_profit', 'yf_ebitda',
        'yf_total_debt', 'yf_total_cash',
        'yf_current_price', 'yf_52week_high', 'yf_52week_low',
        'yf_avg_volume', 'yf_beta', 'yf_pe_ratio', 'yf_forward_pe',
        'yf_dividend_yield', 'yf_profit_margin',
        'vix_at_breach', 'vix_7d_before', 'vix_30d_before',
        'vix_7d_after', 'vix_30d_after', 'vix_30d_avg', 'vix_90d_avg',
    ]

    for col in numeric_cols:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors='coerce')
            valid = series.dropna()
            if len(valid) > 0:
                results['numeric'][col] = {
                    'count': len(valid),
                    'missing': len(df) - len(valid),
                    'missing_pct': round((len(df) - len(valid)) / len(df) * 100, 2),
                    'mean': round(valid.mean(), 2),
                    'std': round(valid.std(), 2),
                    'min': valid.min(),
                    'q25': round(valid.quantile(0.25), 2),
                    'median': round(valid.median(), 2),
                    'q75': round(valid.quantile(0.75), 2),
                    'max': valid.max(),
                    'skewness': round(valid.skew(), 2),
                    'kurtosis': round(valid.kurtosis(), 2),
                }

    # === CATEGORICAL COLUMNS ===
    categorical_cols = [
        'organization_type', 'breach_type',
        'yf_sector', 'yf_industry', 'yf_country', 'yf_exchange',
        'ticker_status',
    ]

    for col in categorical_cols:
        if col in df.columns:
            series = df[col].dropna()
            if len(series) > 0:
                value_counts = series.value_counts()
                results['categorical'][col] = {
                    'count': len(series),
                    'missing': len(df) - len(series),
                    'missing_pct': round((len(df) - len(series)) / len(df) * 100, 2),
                    'unique': series.nunique(),
                    'mode': value_counts.index[0] if len(value_counts) > 0 else None,
                    'mode_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    'top_5': {str(k): int(v) for k, v in value_counts.head(5).items()},
                }

    # === DATE COLUMNS ===
    date_cols = ['reported_date', 'breach_date', 'end_breach_date']

    for col in date_cols:
        if col in df.columns:
            series = pd.to_datetime(df[col], errors='coerce')
            valid = series.dropna()
            if len(valid) > 0:
                results['date'][col] = {
                    'count': len(valid),
                    'missing': len(df) - len(valid),
                    'missing_pct': round((len(df) - len(valid)) / len(df) * 100, 2),
                    'earliest': str(valid.min().date()),
                    'latest': str(valid.max().date()),
                    'range_days': (valid.max() - valid.min()).days,
                    'median': str(valid.median().date()),
                }

    # === MISSING DATA SUMMARY (all columns) ===
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    results['missing_data_summary'] = {
        col: {'count': int(missing[col]), 'pct': float(missing_pct[col])}
        for col in df.columns if missing[col] > 0
    }

    # === NEWS SOURCES BREAKDOWN ===
    news_cols = ['reddit_count', 'guardian_count', 'nyt_count', 'newsapi_count']
    total_news = 0
    for col in news_cols:
        if col in df.columns:
            col_sum = pd.to_numeric(df[col], errors='coerce').fillna(0).sum()
            results['news_sources'][col.replace('_count', '')] = int(col_sum)
            total_news += col_sum
    results['news_sources']['total'] = int(total_news)

    # === STOCK DATA COVERAGE ===
    results['stock_data']['has_ticker'] = int(df['stock_ticker'].notna().sum()) if 'stock_ticker' in df.columns else 0
    results['stock_data']['has_yf_data'] = int(df['yf_company_name'].notna().sum()) if 'yf_company_name' in df.columns else 0

    if 'ticker_status' in df.columns:
        status_counts = df['ticker_status'].value_counts().to_dict()
        results['stock_data']['ticker_status'] = {str(k): int(v) for k, v in status_counts.items()}

    return results


def print_full_descriptive_statistics(df: Optional[pd.DataFrame] = None, db_path: Optional[Path] = None):
    """Print comprehensive descriptive statistics report."""
    stats = run_full_descriptive_statistics(df, db_path)

    print("=" * 80)
    print("COMPREHENSIVE DESCRIPTIVE STATISTICS REPORT")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Overview
    print("=" * 80)
    print("DATASET OVERVIEW")
    print("=" * 80)
    print(f"Total Records: {stats['overview']['total_records']:,}")
    print(f"Total Columns: {stats['overview']['total_columns']}")
    print(f"Memory Usage: {stats['overview']['memory_usage_mb']:.2f} MB")
    print(f"Duplicate Rows: {stats['overview']['duplicate_rows']:,}")
    print()

    # Numeric Variables
    print("=" * 80)
    print("NUMERIC VARIABLES")
    print("=" * 80)

    for col, s in stats['numeric'].items():
        print(f"\n{col.upper()}")
        print("-" * 60)
        print(f"  Count: {s['count']:,}  |  Missing: {s['missing']:,} ({s['missing_pct']}%)")
        print(f"  Mean: {s['mean']:,.2f}  |  Std: {s['std']:,.2f}")
        print(f"  Min: {s['min']:,.2f}  |  Max: {s['max']:,.2f}")
        print(f"  Q25: {s['q25']:,.2f}  |  Median: {s['median']:,.2f}  |  Q75: {s['q75']:,.2f}")
        print(f"  Skewness: {s['skewness']}  |  Kurtosis: {s['kurtosis']}")

    # Categorical Variables
    print()
    print("=" * 80)
    print("CATEGORICAL VARIABLES")
    print("=" * 80)

    for col, s in stats['categorical'].items():
        print(f"\n{col.upper()}")
        print("-" * 60)
        print(f"  Count: {s['count']:,}  |  Missing: {s['missing']:,} ({s['missing_pct']}%)")
        print(f"  Unique Values: {s['unique']}")
        print(f"  Mode: {s['mode']} (n={s['mode_count']:,})")
        print(f"  Top 5 Values:")
        for val, count in s['top_5'].items():
            print(f"    - {val}: {count:,}")

    # Date Variables
    print()
    print("=" * 80)
    print("DATE VARIABLES")
    print("=" * 80)

    for col, s in stats['date'].items():
        print(f"\n{col.upper()}")
        print("-" * 60)
        print(f"  Count: {s['count']:,}  |  Missing: {s['missing']:,} ({s['missing_pct']}%)")
        print(f"  Earliest: {s['earliest']}  |  Latest: {s['latest']}")
        print(f"  Range: {s['range_days']:,} days")
        print(f"  Median: {s['median']}")

    # Missing Data Summary
    print()
    print("=" * 80)
    print("MISSING DATA SUMMARY")
    print("=" * 80)
    if stats['missing_data_summary']:
        # Sort by missing percentage descending
        sorted_missing = sorted(
            stats['missing_data_summary'].items(),
            key=lambda x: x[1]['pct'], reverse=True
        )
        print(f"  {'Column':<35s} {'Missing':>8s} {'Pct':>8s}")
        print(f"  {'-'*35} {'-'*8} {'-'*8}")
        for col, info in sorted_missing:
            print(f"  {col:<35s} {info['count']:>8,} {info['pct']:>7.1f}%")
    else:
        print("  No missing data!")

    # News Sources
    print()
    print("=" * 80)
    print("NEWS DATA SUMMARY")
    print("=" * 80)
    print(f"  Reddit Articles: {stats['news_sources'].get('reddit', 0):,}")
    print(f"  Guardian Articles: {stats['news_sources'].get('guardian', 0):,}")
    print(f"  NYT Articles: {stats['news_sources'].get('nyt', 0):,}")
    print(f"  NewsAPI Articles: {stats['news_sources'].get('newsapi', 0):,}")
    print(f"  TOTAL: {stats['news_sources'].get('total', 0):,}")

    # Stock Data Coverage
    print()
    print("=" * 80)
    print("STOCK DATA COVERAGE")
    print("=" * 80)
    print(f"  Records with Ticker: {stats['stock_data']['has_ticker']:,}")
    print(f"  Records with Yahoo Finance Data: {stats['stock_data']['has_yf_data']:,}")
    if 'ticker_status' in stats['stock_data']:
        print(f"  Ticker Status Breakdown:")
        for status, count in stats['stock_data']['ticker_status'].items():
            print(f"    - {status}: {count:,}")

    print()
    print("=" * 80)

    return stats


def analyze_trends(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Analyze yearly trends in breach data."""
    df = query_df("""
        SELECT
            strftime('%Y', breach_date) as year,
            COUNT(*) as breach_count,
            SUM(total_affected) as total_affected,
            AVG(total_affected) as avg_affected,
            SUM(total_news_count) as news_coverage
        FROM breach_incidents
        WHERE breach_date IS NOT NULL
        GROUP BY strftime('%Y', breach_date)
        ORDER BY year
    """, db_path=db_path)

    # Calculate year-over-year changes
    df['yoy_count_change'] = df['breach_count'].pct_change() * 100
    df['yoy_affected_change'] = df['total_affected'].pct_change() * 100

    return df


def analyze_sector_risk(db_path: Optional[Path] = None) -> List[SectorRiskProfile]:
    """Analyze risk profile by sector."""
    df = query_df("""
        SELECT
            yf_sector as sector,
            COUNT(*) as total_breaches,
            SUM(total_affected) as total_affected,
            AVG(total_affected) as avg_affected,
            AVG(total_news_count) as avg_news
        FROM breach_incidents
        WHERE yf_sector IS NOT NULL
        GROUP BY yf_sector
    """, db_path=db_path)

    # Get year range for frequency calculation
    years_df = query_df("""
        SELECT
            MIN(strftime('%Y', breach_date)) as min_year,
            MAX(strftime('%Y', breach_date)) as max_year
        FROM breach_incidents
    """, db_path=db_path)

    years_range = int(years_df['max_year'].iloc[0]) - int(years_df['min_year'].iloc[0]) + 1

    profiles = []
    for _, row in df.iterrows():
        sector = row['sector']
        breach_freq = row['total_breaches'] / years_range

        # Calculate composite risk score
        sensitivity = SECTOR_SENSITIVITY.get(sector, 0.5)
        volume_factor = min(1.0, row['total_breaches'] / 100)
        impact_factor = min(1.0, np.log10(row['avg_affected'] + 1) / 9)

        risk_score = (sensitivity * 0.4 + volume_factor * 0.3 + impact_factor * 0.3) * 100

        profiles.append(SectorRiskProfile(
            sector=sector,
            total_breaches=int(row['total_breaches']),
            total_affected=int(row['total_affected']),
            avg_affected=row['avg_affected'],
            breach_frequency=round(breach_freq, 2),
            media_attention=row['avg_news'],
            risk_score=round(risk_score, 2)
        ))

    # Sort by risk score
    profiles.sort(key=lambda x: x.risk_score, reverse=True)

    return profiles


def correlation_analysis(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Analyze correlations between numeric variables."""
    df = query_df("""
        SELECT
            total_affected,
            total_news_count,
            reddit_count,
            guardian_count,
            nyt_count,
            newsapi_count,
            yf_market_cap
        FROM breach_incidents
        WHERE total_affected IS NOT NULL
    """, db_path=db_path)

    # Replace None with NaN for correlation
    df = df.fillna(np.nan)

    return df.corr()


# =============================================================================
# PREDICTIVE FEATURES
# =============================================================================

def extract_features(db_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Extract features for machine learning models.
    Returns a DataFrame with engineered features.
    """
    df = query_df("""
        SELECT
            id,
            org_name,
            total_affected,
            breach_date,
            end_breach_date,
            breach_type,
            yf_sector,
            yf_market_cap,
            yf_employees,
            total_news_count,
            ticker_status
        FROM breach_incidents
    """, db_path=db_path)

    # Severity classification
    df['severity'] = df['total_affected'].apply(
        lambda x: classify_severity(x).value if pd.notna(x) else 'unknown'
    )

    # Log-transformed affected (for modeling)
    df['log_affected'] = np.log10(df['total_affected'] + 1)

    # Breach duration
    df['breach_date'] = pd.to_datetime(df['breach_date'], errors='coerce')
    df['end_breach_date'] = pd.to_datetime(df['end_breach_date'], errors='coerce')
    df['breach_duration_days'] = (df['end_breach_date'] - df['breach_date']).dt.days

    # Time features
    df['breach_year'] = df['breach_date'].dt.year
    df['breach_month'] = df['breach_date'].dt.month
    df['breach_quarter'] = df['breach_date'].dt.quarter

    # Company size features
    df['log_market_cap'] = np.log10(df['yf_market_cap'] + 1)
    df['log_employees'] = np.log10(df['yf_employees'] + 1)

    # Sector risk
    df['sector_sensitivity'] = df['yf_sector'].map(SECTOR_SENSITIVITY).fillna(0.5)

    # Is public company
    df['is_public'] = df['ticker_status'].apply(
        lambda x: 1 if x == 'active' else 0 if pd.notna(x) else None
    )

    return df


def get_high_risk_breaches(threshold: float = 70.0, db_path: Optional[Path] = None) -> pd.DataFrame:
    """Get breaches with risk score above threshold."""
    df = query_df("""
        SELECT
            org_name,
            total_affected,
            breach_date,
            end_breach_date,
            total_news_count,
            yf_sector,
            breach_type
        FROM breach_incidents
        WHERE total_affected IS NOT NULL
    """, db_path=db_path)

    risk_scores = []
    for _, row in df.iterrows():
        score = calculate_risk_score(
            affected=row['total_affected'],
            breach_date=row['breach_date'],
            end_date=row['end_breach_date'],
            news_count=row['total_news_count'],
            sector=row['yf_sector']
        )
        risk_scores.append(score.overall_score)

    df['risk_score'] = risk_scores

    return df[df['risk_score'] >= threshold].sort_values('risk_score', ascending=False)


# =============================================================================
# REPORTING
# =============================================================================

def generate_summary_report(db_path: Optional[Path] = None) -> str:
    """Generate a text summary report of breach data analysis."""
    stats = get_descriptive_stats(db_path)
    severity_dist = get_severity_distribution(db_path)
    sector_profiles = analyze_sector_risk(db_path)

    report = []
    report.append("=" * 60)
    report.append("BREACH DATA ANALYSIS REPORT")
    report.append("=" * 60)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    report.append("OVERVIEW")
    report.append("-" * 40)
    report.append(f"Total Breaches Analyzed: {stats['count']:,}")
    report.append(f"Total Individuals Affected: {stats['affected']['mean'] * stats['count']:,.0f}")
    report.append(f"Sectors Represented: {stats['sectors']}")
    report.append(f"Breach Types: {stats['breach_types']}")
    report.append("")

    report.append("IMPACT STATISTICS")
    report.append("-" * 40)
    report.append(f"Average Affected: {stats['affected']['mean']:,.0f}")
    report.append(f"Median Affected: {stats['affected']['median']:,.0f}")
    report.append(f"Largest Breach: {stats['affected']['max']:,.0f}")
    report.append(f"Smallest Breach: {stats['affected']['min']:,.0f}")
    report.append("")

    report.append("SEVERITY DISTRIBUTION")
    report.append("-" * 40)
    for severity, count in severity_dist.items():
        pct = (count / stats['count']) * 100
        report.append(f"  {severity.capitalize():10s}: {count:5d} ({pct:5.1f}%)")
    report.append("")

    report.append("TOP 5 HIGH-RISK SECTORS")
    report.append("-" * 40)
    for i, profile in enumerate(sector_profiles[:5], 1):
        report.append(f"  {i}. {profile.sector}")
        report.append(f"     Risk Score: {profile.risk_score:.1f}")
        report.append(f"     Total Breaches: {profile.total_breaches}")
        report.append(f"     Avg Affected: {profile.avg_affected:,.0f}")

    report.append("")
    report.append("=" * 60)

    return "\n".join(report)


# =============================================================================
# STEP 2: OLS REGRESSION — FAMA-FRENCH FACTORS BY VIX REGIME
# =============================================================================
# Tests whether Fama-French 5-Factor relationships differ between
# high-volatility (VIX >= 25) and low-volatility (VIX < 25) trading days.
#
# Uses daily FF factor returns and daily VIX — no breach data involved.
#
# DV:  Mkt-RF  (daily market excess return, in pct points)
# IVs: SMB, HML, RMW, CMA  (daily factor returns)
# Split: Daily VIX close >= 25 (high vol) vs < 25 (low vol)
# =============================================================================

VIX_THRESHOLD = 25

FF_REGRESSORS = ['SMB', 'HML', 'RMW', 'CMA']


def prepare_ff_vix_data() -> pd.DataFrame:
    """
    Fetch daily Fama-French 5-Factor returns and daily VIX, merge on date.

    Returns a DataFrame with columns:
      date, Mkt-RF, SMB, HML, RMW, CMA, RF, vix_close, high_vol
    """
    from clean import fetch_fama_french_data, fetch_vix_data

    ff_df = fetch_fama_french_data()
    if ff_df.empty:
        raise RuntimeError("Fama-French data unavailable.")

    vix_df = fetch_vix_data()
    if vix_df.empty:
        raise RuntimeError("VIX data unavailable.")

    # Align on date
    ff_work = ff_df.reset_index()
    ff_work['date'] = pd.to_datetime(ff_work['date'])

    vix_df['date'] = pd.to_datetime(vix_df['date'])

    merged = ff_work.merge(vix_df, on='date', how='inner')
    merged['high_vol'] = (merged['vix_close'] >= VIX_THRESHOLD).astype(int)

    logger.info(
        "FF-VIX merged data: %d trading days (%s to %s)",
        len(merged),
        merged['date'].min().strftime('%Y-%m-%d'),
        merged['date'].max().strftime('%Y-%m-%d'),
    )
    logger.info(
        "  High-vol days (VIX >= %d): %d  |  Low-vol days: %d",
        VIX_THRESHOLD, merged['high_vol'].sum(), (1 - merged['high_vol']).sum(),
    )
    return merged


def run_ols_fama_french() -> Dict[str, Any]:
    """
    Run OLS: Mkt-RF ~ SMB + HML + RMW + CMA on daily factor data,
    split by VIX regime.

    Returns dict with three model results:
      - 'pooled':   all trading days
      - 'high_vol': VIX >= 25
      - 'low_vol':  VIX < 25
    """
    import statsmodels.api as sm

    reg_df = prepare_ff_vix_data()

    results = {}
    subsets = {
        'pooled': reg_df,
        'high_vol': reg_df[reg_df['high_vol'] == 1],
        'low_vol': reg_df[reg_df['high_vol'] == 0],
    }

    for label, subset in subsets.items():
        if len(subset) < 10:
            logger.warning("Skipping '%s' — only %d observations", label, len(subset))
            continue

        y = subset['Mkt-RF']
        X = sm.add_constant(subset[FF_REGRESSORS])
        model = sm.OLS(y, X).fit()
        results[label] = model

    return results


def print_ols_results(results: Dict[str, Any]):
    """Print formatted OLS regression results for all VIX regimes."""

    print("=" * 80)
    print("STEP 2: OLS REGRESSION — FAMA-FRENCH FACTORS BY VIX REGIME")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dependent Variable: Mkt-RF (daily market excess return, pct points)")
    print(f"Independent Variables: SMB, HML, RMW, CMA (+ constant)")
    print(f"VIX Threshold: {VIX_THRESHOLD}")
    print()

    for label in ['pooled', 'high_vol', 'low_vol']:
        model = results.get(label)
        if model is None:
            print(f"--- {label.upper().replace('_', ' ')} --- SKIPPED (insufficient data)")
            print()
            continue

        regime_name = {
            'pooled': 'POOLED (All Trading Days)',
            'high_vol': f'HIGH VOLATILITY (VIX >= {VIX_THRESHOLD})',
            'low_vol': f'LOW VOLATILITY (VIX < {VIX_THRESHOLD})',
        }[label]

        print("=" * 80)
        print(f"  {regime_name}  |  N = {int(model.nobs):,}")
        print("=" * 80)
        print(f"  R-squared: {model.rsquared:.4f}  |  Adj. R-squared: {model.rsquared_adj:.4f}")
        print(f"  F-statistic: {model.fvalue:.2f}  (p = {model.f_pvalue:.4e})")
        print(f"  AIC: {model.aic:.2f}  |  BIC: {model.bic:.2f}")
        print()

        # Coefficient table
        header = f"  {'Variable':<12s} {'Coef':>10s} {'Std Err':>10s} {'t':>8s} {'P>|t|':>10s} {'[0.025':>10s} {'0.975]':>10s}  Sig"
        print(header)
        print("  " + "-" * 88)

        conf = model.conf_int()
        for var in model.params.index:
            coef = model.params[var]
            se = model.bse[var]
            t = model.tvalues[var]
            p = model.pvalues[var]
            ci_lo = conf.loc[var, 0]
            ci_hi = conf.loc[var, 1]

            if p < 0.001:
                sig = "***"
            elif p < 0.01:
                sig = "**"
            elif p < 0.05:
                sig = "*"
            elif p < 0.10:
                sig = "."
            else:
                sig = ""

            print(f"  {var:<12s} {coef:>10.4f} {se:>10.4f} {t:>8.3f} {p:>10.4f} {ci_lo:>10.4f} {ci_hi:>10.4f}  {sig}")

        print()
        print(f"  Significance: *** p<0.001, ** p<0.01, * p<0.05, . p<0.10")
        print()

    # === Regime comparison ===
    if 'high_vol' in results and 'low_vol' in results:
        hv = results['high_vol']
        lv = results['low_vol']

        print("=" * 80)
        print("  REGIME COMPARISON: HIGH VOL vs LOW VOL")
        print("=" * 80)

        print(f"  {'Variable':<12s} {'High Vol':>10s} {'Low Vol':>10s} {'Diff':>10s} {'HV p-val':>10s} {'LV p-val':>10s}")
        print("  " + "-" * 62)

        for var in FF_REGRESSORS:
            hv_coef = hv.params.get(var, np.nan)
            lv_coef = lv.params.get(var, np.nan)
            diff = hv_coef - lv_coef
            hv_p = hv.pvalues.get(var, np.nan)
            lv_p = lv.pvalues.get(var, np.nan)
            print(f"  {var:<12s} {hv_coef:>10.4f} {lv_coef:>10.4f} {diff:>10.4f} {hv_p:>10.4f} {lv_p:>10.4f}")

        print()
        print(f"  R-squared:    High Vol = {hv.rsquared:.4f}  |  Low Vol = {lv.rsquared:.4f}")
        print(f"  Observations: High Vol = {int(hv.nobs):,}    |  Low Vol = {int(lv.nobs):,}")
        print()

    print("=" * 80)


# =============================================================================
# STEP 3: FAMA-FRENCH + MACRO CONTROLS BY VIX REGIME
# =============================================================================
# Adds macroeconomic control variables to the base FF model one at a time,
# then all together:
#   (a) + Inflation (YoY CPI)
#   (b) + GDP growth
#   (c) + Unemployment rate
#   (d) + Interest rates (Fed Funds + Yield Spread 10Y-2Y)
#   (e) + ALL macro controls combined
#
# Monthly/quarterly data is forward-filled to daily frequency before merge.
# =============================================================================


def prepare_ff_vix_macro_data() -> pd.DataFrame:
    """
    Build a single daily DataFrame with FF factors, VIX, and all macro controls.

    Macro variables (forward-filled to daily):
      - inflation_yoy: Year-over-year CPI inflation rate (%)
      - gdp_growth: Real GDP growth rate (% change, SAAR)
      - unemployment_rate: Civilian unemployment rate (%)
      - fed_funds_rate: Federal Funds Effective Rate (%)
      - yield_spread: 10Y Treasury - 2Y Treasury (%)
    """
    from clean import (
        fetch_fama_french_data,
        fetch_vix_data,
        fetch_inflation_data,
        fetch_gdp_data,
        fetch_unemployment_data,
        fetch_interest_rate_data,
    )

    # --- Base data (daily) ---
    ff_df = fetch_fama_french_data()
    if ff_df.empty:
        raise RuntimeError("Fama-French data unavailable.")
    ff_work = ff_df.reset_index()
    ff_work['date'] = pd.to_datetime(ff_work['date'])

    vix_df = fetch_vix_data()
    if vix_df.empty:
        raise RuntimeError("VIX data unavailable.")
    vix_df['date'] = pd.to_datetime(vix_df['date'])

    merged = ff_work.merge(vix_df, on='date', how='inner').sort_values('date').reset_index(drop=True)
    merged['high_vol'] = (merged['vix_close'] >= VIX_THRESHOLD).astype(int)

    # --- Inflation (monthly → daily via forward-fill) ---
    try:
        cpi_df = fetch_inflation_data()
        if not cpi_df.empty:
            cpi_df['date'] = pd.to_datetime(cpi_df['date'])
            cpi_daily = cpi_df[['date', 'inflation_yoy']].sort_values('date')
            merged = pd.merge_asof(
                merged, cpi_daily, on='date', direction='backward',
            )
            logger.info("Inflation: %d non-null daily values", merged['inflation_yoy'].notna().sum())
    except Exception as e:
        logger.warning("Could not fetch inflation data: %s", e)
        merged['inflation_yoy'] = np.nan

    # --- GDP growth (quarterly → daily via forward-fill) ---
    try:
        gdp_df = fetch_gdp_data()
        if not gdp_df.empty:
            gdp_df['date'] = pd.to_datetime(gdp_df['date'])
            gdp_daily = gdp_df[['date', 'gdp_growth']].sort_values('date')
            merged = pd.merge_asof(
                merged, gdp_daily, on='date', direction='backward',
            )
            logger.info("GDP: %d non-null daily values", merged['gdp_growth'].notna().sum())
    except Exception as e:
        logger.warning("Could not fetch GDP data: %s", e)
        merged['gdp_growth'] = np.nan

    # --- Unemployment rate (monthly → daily via forward-fill) ---
    try:
        ur_df = fetch_unemployment_data()
        if not ur_df.empty:
            ur_df['date'] = pd.to_datetime(ur_df['date'])
            ur_daily = ur_df[['date', 'unemployment_rate']].sort_values('date')
            merged = pd.merge_asof(
                merged, ur_daily, on='date', direction='backward',
            )
            logger.info("Unemployment: %d non-null daily values", merged['unemployment_rate'].notna().sum())
    except Exception as e:
        logger.warning("Could not fetch unemployment data: %s", e)
        merged['unemployment_rate'] = np.nan

    # --- Interest rates (daily) ---
    try:
        rate_data = fetch_interest_rate_data()
        # Fed Funds Rate
        ff_rate = rate_data.get('fed_funds', pd.DataFrame())
        if not ff_rate.empty:
            ff_rate['date'] = pd.to_datetime(ff_rate['date'])
            ff_rate = ff_rate[['date', 'fed_funds_rate']].sort_values('date')
            merged = pd.merge_asof(
                merged, ff_rate, on='date', direction='backward',
            )

        # Yield spread = 10Y - 2Y
        t10 = rate_data.get('treasury_10y', pd.DataFrame())
        t2 = rate_data.get('treasury_2y', pd.DataFrame())
        if not t10.empty and not t2.empty:
            t10['date'] = pd.to_datetime(t10['date'])
            t2['date'] = pd.to_datetime(t2['date'])
            spread = t10.merge(t2, on='date', how='inner')
            spread['yield_spread'] = spread['treasury_10y'] - spread['treasury_2y']
            spread = spread[['date', 'yield_spread']].sort_values('date')
            merged = pd.merge_asof(
                merged, spread, on='date', direction='backward',
            )

        ir_count = merged[['fed_funds_rate', 'yield_spread']].notna().all(axis=1).sum()
        logger.info("Interest rates: %d non-null daily values", ir_count)
    except Exception as e:
        logger.warning("Could not fetch interest rate data: %s", e)
        merged['fed_funds_rate'] = np.nan
        merged['yield_spread'] = np.nan

    logger.info(
        "Full macro dataset: %d trading days, %d columns",
        len(merged), len(merged.columns),
    )
    return merged


# Model specifications: (label, extra regressors beyond SMB/HML/RMW/CMA)
MACRO_MODEL_SPECS = [
    ('base',             [],                                          'Base FF (no controls)'),
    ('inflation',        ['inflation_yoy'],                           '+ Inflation (YoY CPI)'),
    ('gdp',             ['gdp_growth'],                               '+ GDP Growth'),
    ('unemployment',     ['unemployment_rate'],                       '+ Unemployment Rate'),
    ('interest_rates',   ['fed_funds_rate', 'yield_spread'],          '+ Interest Rates (FFR + Yield Spread)'),
    ('all_controls',     ['inflation_yoy', 'gdp_growth',
                          'unemployment_rate', 'fed_funds_rate',
                          'yield_spread'],                            '+ ALL Macro Controls'),
]


def run_ols_with_macro_controls() -> Dict[str, Dict[str, Any]]:
    """
    Run the FF-VIX OLS with each macro control individually, then all combined.

    Returns nested dict: {model_label: {'pooled': model, 'high_vol': model, 'low_vol': model}}
    """
    import statsmodels.api as sm

    data = prepare_ff_vix_macro_data()
    all_results = {}

    for spec_label, extra_vars, description in MACRO_MODEL_SPECS:
        regressors = FF_REGRESSORS + extra_vars

        # Drop rows with NaN in any regressor
        subset = data.dropna(subset=['Mkt-RF'] + regressors)

        results = {}
        for regime_label, regime_df in [
            ('pooled', subset),
            ('high_vol', subset[subset['high_vol'] == 1]),
            ('low_vol', subset[subset['high_vol'] == 0]),
        ]:
            if len(regime_df) < 10:
                logger.warning("Skipping %s/%s — only %d obs", spec_label, regime_label, len(regime_df))
                continue

            y = regime_df['Mkt-RF']
            X = sm.add_constant(regime_df[regressors])
            model = sm.OLS(y, X).fit()
            results[regime_label] = model

        all_results[spec_label] = results

    return all_results


def _print_model_table(model, regime_name: str, regressors: List[str]):
    """Print a single OLS model result table."""
    print(f"  {regime_name}  |  N = {int(model.nobs):,}")
    print("  " + "-" * 88)
    print(f"  R-squared: {model.rsquared:.4f}  |  Adj. R-squared: {model.rsquared_adj:.4f}")
    print(f"  F-statistic: {model.fvalue:.2f}  (p = {model.f_pvalue:.4e})")
    print()

    header = f"  {'Variable':<20s} {'Coef':>10s} {'Std Err':>10s} {'t':>8s} {'P>|t|':>10s}  Sig"
    print(header)
    print("  " + "-" * 68)

    for var in model.params.index:
        coef = model.params[var]
        se = model.bse[var]
        t = model.tvalues[var]
        p = model.pvalues[var]

        if p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        elif p < 0.05:
            sig = "*"
        elif p < 0.10:
            sig = "."
        else:
            sig = ""

        print(f"  {var:<20s} {coef:>10.4f} {se:>10.4f} {t:>8.3f} {p:>10.4f}  {sig}")
    print()


def print_macro_control_results(all_results: Dict[str, Dict[str, Any]]):
    """Print all macro-control OLS results organized by model specification."""

    print("=" * 80)
    print("STEP 3: FAMA-FRENCH + MACROECONOMIC CONTROLS BY VIX REGIME")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"DV: Mkt-RF  |  Base IVs: SMB, HML, RMW, CMA  |  VIX threshold: {VIX_THRESHOLD}")
    print()

    for spec_label, extra_vars, description in MACRO_MODEL_SPECS:
        results = all_results.get(spec_label, {})
        if not results:
            continue

        print("=" * 80)
        print(f"  MODEL: {description}")
        if extra_vars:
            print(f"  Controls added: {', '.join(extra_vars)}")
        print("=" * 80)
        print()

        regressors = FF_REGRESSORS + extra_vars

        for regime_label in ['pooled', 'high_vol', 'low_vol']:
            model = results.get(regime_label)
            if model is None:
                continue

            regime_name = {
                'pooled': 'POOLED',
                'high_vol': f'HIGH VOL (VIX >= {VIX_THRESHOLD})',
                'low_vol': f'LOW VOL (VIX < {VIX_THRESHOLD})',
            }[regime_label]

            _print_model_table(model, regime_name, regressors)

        print()

    # === Summary comparison table ===
    _print_macro_summary_table(all_results)


def _print_macro_summary_table(all_results: Dict[str, Dict[str, Any]]):
    """Print a compact comparison table across all model specs and regimes."""

    print("=" * 80)
    print("  SUMMARY: R-SQUARED COMPARISON ACROSS MODELS AND REGIMES")
    print("=" * 80)

    header = f"  {'Model':<30s} {'Pooled':>10s} {'High Vol':>10s} {'Low Vol':>10s} {'Pooled N':>10s}"
    print(header)
    print("  " + "-" * 70)

    for spec_label, _, description in MACRO_MODEL_SPECS:
        results = all_results.get(spec_label, {})
        pooled_r2 = results['pooled'].rsquared if 'pooled' in results else np.nan
        hv_r2 = results['high_vol'].rsquared if 'high_vol' in results else np.nan
        lv_r2 = results['low_vol'].rsquared if 'low_vol' in results else np.nan
        n = int(results['pooled'].nobs) if 'pooled' in results else 0
        print(f"  {description:<30s} {pooled_r2:>10.4f} {hv_r2:>10.4f} {lv_r2:>10.4f} {n:>10,}")

    print()

    # Coefficient stability check — show FF factor coefficients across models (pooled only)
    print("=" * 80)
    print("  FF FACTOR COEFFICIENT STABILITY (POOLED, ACROSS CONTROL SETS)")
    print("=" * 80)

    header = f"  {'Model':<30s}"
    for var in FF_REGRESSORS:
        header += f" {var:>8s}"
    print(header)
    print("  " + "-" * (30 + 9 * len(FF_REGRESSORS)))

    for spec_label, _, description in MACRO_MODEL_SPECS:
        results = all_results.get(spec_label, {})
        pooled = results.get('pooled')
        if pooled is None:
            continue
        row = f"  {description:<30s}"
        for var in FF_REGRESSORS:
            coef = pooled.params.get(var, np.nan)
            row += f" {coef:>8.4f}"
        print(row)

    print()
    print("=" * 80)


# =============================================================================
# STEP 4: FAMA-FRENCH + MACRO + STOCK PRICE CONTROLS (BREACH-LEVEL)
# =============================================================================
# Shifts unit of analysis from daily trading days → individual breach events.
# Each breach is matched to FF factors, VIX, macro conditions, and the
# breached company's stock characteristics on the breach date.
#
# DV:  Mkt-RF at breach date
# IVs: SMB, HML, RMW, CMA
#       + macro controls (inflation, GDP, unemployment, FFR, yield spread)
#       + stock controls (log price, log market cap, beta)
# Split: VIX at breach date >= 25 (high vol) vs < 25 (low vol)
# =============================================================================

STOCK_CONTROLS = ['log_price', 'log_market_cap', 'yf_beta']


def prepare_breach_level_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build breach-level dataset: one row per breach event with FF factors,
    VIX, macro controls, and company stock data at the breach date.
    """
    from clean import (
        fetch_fama_french_data,
        fetch_vix_data,
        fetch_inflation_data,
        fetch_gdp_data,
        fetch_unemployment_data,
        fetch_interest_rate_data,
    )

    # --- Filter to breaches with valid dates and stock data ---
    mask = (
        df['breach_date'].notna()
        & df['yf_current_price'].notna()
        & df['yf_market_cap'].notna()
        & df['yf_beta'].notna()
        & df['vix_at_breach'].notna()
    )
    work = df.loc[mask].copy()
    work = work.sort_values('breach_date').reset_index(drop=True)
    logger.info("Breach-level base: %d events with stock data", len(work))

    # --- Stock controls ---
    work['log_price'] = np.log10(work['yf_current_price'].clip(lower=0.01))
    work['log_market_cap'] = np.log10(work['yf_market_cap'].clip(lower=1))

    # --- VIX regime (already in enriched data) ---
    work['high_vol'] = (work['vix_at_breach'] >= VIX_THRESHOLD).astype(int)

    # --- FF factors at breach date ---
    ff_df = fetch_fama_french_data()
    if ff_df.empty:
        raise RuntimeError("Fama-French data unavailable.")
    ff_work = ff_df.reset_index()
    ff_work['date'] = pd.to_datetime(ff_work['date'])
    ff_work = ff_work.sort_values('date').reset_index(drop=True)

    work = pd.merge_asof(
        work, ff_work[['date'] + FF_REGRESSORS + ['Mkt-RF']],
        left_on='breach_date', right_on='date',
        direction='nearest', tolerance=pd.Timedelta(days=5),
    )

    # --- Macro controls (same approach as Step 3) ---
    # Inflation
    try:
        cpi_df = fetch_inflation_data()
        if not cpi_df.empty:
            cpi_df['date'] = pd.to_datetime(cpi_df['date'])
            work = pd.merge_asof(
                work, cpi_df[['date', 'inflation_yoy']].sort_values('date'),
                left_on='breach_date', right_on='date', direction='backward',
                suffixes=('', '_macro'),
            )
    except Exception as e:
        logger.warning("Inflation fetch failed: %s", e)
        work['inflation_yoy'] = np.nan

    # GDP
    try:
        gdp_df = fetch_gdp_data()
        if not gdp_df.empty:
            gdp_df['date'] = pd.to_datetime(gdp_df['date'])
            work = pd.merge_asof(
                work, gdp_df[['date', 'gdp_growth']].sort_values('date'),
                left_on='breach_date', right_on='date', direction='backward',
                suffixes=('', '_macro'),
            )
    except Exception as e:
        logger.warning("GDP fetch failed: %s", e)
        work['gdp_growth'] = np.nan

    # Unemployment
    try:
        ur_df = fetch_unemployment_data()
        if not ur_df.empty:
            ur_df['date'] = pd.to_datetime(ur_df['date'])
            work = pd.merge_asof(
                work, ur_df[['date', 'unemployment_rate']].sort_values('date'),
                left_on='breach_date', right_on='date', direction='backward',
                suffixes=('', '_macro'),
            )
    except Exception as e:
        logger.warning("Unemployment fetch failed: %s", e)
        work['unemployment_rate'] = np.nan

    # Interest rates
    try:
        rate_data = fetch_interest_rate_data()
        ff_rate = rate_data.get('fed_funds', pd.DataFrame())
        if not ff_rate.empty:
            ff_rate['date'] = pd.to_datetime(ff_rate['date'])
            work = pd.merge_asof(
                work, ff_rate[['date', 'fed_funds_rate']].sort_values('date'),
                left_on='breach_date', right_on='date', direction='backward',
                suffixes=('', '_macro'),
            )
        t10 = rate_data.get('treasury_10y', pd.DataFrame())
        t2 = rate_data.get('treasury_2y', pd.DataFrame())
        if not t10.empty and not t2.empty:
            t10['date'] = pd.to_datetime(t10['date'])
            t2['date'] = pd.to_datetime(t2['date'])
            spread = t10.merge(t2, on='date', how='inner')
            spread['yield_spread'] = spread['treasury_10y'] - spread['treasury_2y']
            work = pd.merge_asof(
                work, spread[['date', 'yield_spread']].sort_values('date'),
                left_on='breach_date', right_on='date', direction='backward',
                suffixes=('', '_macro'),
            )
    except Exception as e:
        logger.warning("Interest rate fetch failed: %s", e)
        work['fed_funds_rate'] = np.nan
        work['yield_spread'] = np.nan

    # Clean up duplicate date columns from merges
    date_dups = [c for c in work.columns if c.startswith('date_')]
    if date_dups:
        work = work.drop(columns=date_dups)

    hv = work['high_vol'].sum()
    logger.info(
        "Breach-level dataset ready: %d events (high-vol=%d, low-vol=%d)",
        len(work), hv, len(work) - hv,
    )
    return work


MACRO_CONTROLS_ALL = [
    'inflation_yoy', 'gdp_growth', 'unemployment_rate',
    'fed_funds_rate', 'yield_spread',
]

STOCK_MODEL_SPECS = [
    ('base_ff',     [],                              [],              'Base FF only'),
    ('stock',       STOCK_CONTROLS,                  [],              '+ Stock Controls (price, mktcap, beta)'),
    ('macro',       [],                              MACRO_CONTROLS_ALL, '+ Macro Controls'),
    ('stock_macro', STOCK_CONTROLS,                  MACRO_CONTROLS_ALL, '+ Stock + Macro Controls'),
]


def run_ols_breach_level(df: pd.DataFrame) -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame]:
    """
    Run breach-level OLS: Mkt-RF ~ FF + stock controls + macro controls,
    split by VIX regime.

    Returns (model_results_dict, breach_level_dataframe).
    """
    import statsmodels.api as sm

    data = prepare_breach_level_data(df)
    all_results = {}

    for spec_label, stock_vars, macro_vars, description in STOCK_MODEL_SPECS:
        regressors = FF_REGRESSORS + stock_vars + macro_vars

        subset = data.dropna(subset=['Mkt-RF'] + regressors)

        results = {}
        for regime_label, regime_df in [
            ('pooled', subset),
            ('high_vol', subset[subset['high_vol'] == 1]),
            ('low_vol', subset[subset['high_vol'] == 0]),
        ]:
            if len(regime_df) < len(regressors) + 5:
                logger.warning(
                    "Skipping %s/%s — only %d obs for %d regressors",
                    spec_label, regime_label, len(regime_df), len(regressors),
                )
                continue

            y = regime_df['Mkt-RF']
            X = sm.add_constant(regime_df[regressors])
            model = sm.OLS(y, X).fit()
            results[regime_label] = model

        all_results[spec_label] = results

    return all_results, data


def print_breach_level_results(all_results: Dict[str, Dict[str, Any]], data: pd.DataFrame):
    """Print Step 4 breach-level OLS results."""

    print("=" * 80)
    print("STEP 4: FAMA-FRENCH + STOCK PRICE CONTROLS (BREACH-LEVEL)")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Unit of analysis: individual breach events")
    print(f"DV: Mkt-RF at breach date  |  VIX threshold: {VIX_THRESHOLD}")
    print(f"Stock controls: log(price), log(market_cap), beta")
    n_total = len(data)
    n_hv = data['high_vol'].sum()
    print(f"Total events: {n_total:,} (high-vol: {n_hv:,}, low-vol: {n_total - n_hv:,})")
    print()

    for spec_label, stock_vars, macro_vars, description in STOCK_MODEL_SPECS:
        results = all_results.get(spec_label, {})
        if not results:
            continue

        print("=" * 80)
        print(f"  MODEL: {description}")
        controls = stock_vars + macro_vars
        if controls:
            print(f"  Controls: {', '.join(controls)}")
        print("=" * 80)
        print()

        for regime_label in ['pooled', 'high_vol', 'low_vol']:
            model = results.get(regime_label)
            if model is None:
                continue

            regime_name = {
                'pooled': 'POOLED',
                'high_vol': f'HIGH VOL (VIX >= {VIX_THRESHOLD})',
                'low_vol': f'LOW VOL (VIX < {VIX_THRESHOLD})',
            }[regime_label]

            _print_model_table(model, regime_name, [])

        print()

    # === Summary table ===
    print("=" * 80)
    print("  SUMMARY: R-SQUARED ACROSS MODELS (BREACH-LEVEL)")
    print("=" * 80)

    header = f"  {'Model':<35s} {'Pooled':>10s} {'High Vol':>10s} {'Low Vol':>10s} {'Pooled N':>10s}"
    print(header)
    print("  " + "-" * 75)

    for spec_label, _, _, description in STOCK_MODEL_SPECS:
        results = all_results.get(spec_label, {})
        pooled_r2 = results['pooled'].rsquared if 'pooled' in results else np.nan
        hv_r2 = results['high_vol'].rsquared if 'high_vol' in results else np.nan
        lv_r2 = results['low_vol'].rsquared if 'low_vol' in results else np.nan
        n = int(results['pooled'].nobs) if 'pooled' in results else 0
        print(f"  {description:<35s} {pooled_r2:>10.4f} {hv_r2:>10.4f} {lv_r2:>10.4f} {n:>10,}")

    # === Stock control coefficients across regimes ===
    print()
    print("=" * 80)
    print("  STOCK CONTROL COEFFICIENTS BY REGIME (full model: stock + macro)")
    print("=" * 80)

    full = all_results.get('stock_macro', {})
    if full:
        header = f"  {'Variable':<20s} {'Pooled':>10s} {'High Vol':>10s} {'Low Vol':>10s}"
        print(header)
        print("  " + "-" * 50)

        for var in STOCK_CONTROLS:
            row = f"  {var:<20s}"
            for regime in ['pooled', 'high_vol', 'low_vol']:
                model = full.get(regime)
                if model and var in model.params.index:
                    coef = model.params[var]
                    p = model.pvalues[var]
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.1 else ""
                    row += f" {coef:>8.4f}{sig:<2s}"
                else:
                    row += f" {'N/A':>10s}"
            print(row)

    print()
    print("=" * 80)


# =============================================================================
# STEP 5: FAMA-FRENCH EVENT STUDY — BREACH ANNOUNCEMENTS
# =============================================================================
# Standard event study: measures abnormal stock returns around breach
# disclosure dates using the Fama-French 5-Factor model.
#
# Event date: reported_date (when market learns about the breach)
# Event window: [-1, +5] trading days (7 days total)
# Estimation window: [-252, -31] trading days (~1 year, ending 1 month before)
# Model: Ri,t - Rf = α + β1(Mkt-RF) + β2(SMB) + β3(HML) + β4(RMW) + β5(CMA) + ε
# Abnormal return: AR = (Ri,t - Rf) - [α̂ + Σ β̂·factors]
# CAR[-1, +5] = sum of AR over event window
# VIX regime: VIX at reported_date >= 25 (high) vs < 25 (low)
# Minimum estimation window: 100 trading days
# =============================================================================

EVENT_WINDOW = (-1, 10)      # relative trading days around event
EST_WINDOW = (-252, -31)     # estimation window (ends 1 month before event)
MIN_EST_DAYS = 100           # minimum usable trading days in estimation window


def run_event_study(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run Fama-French 5-Factor event study around breach announcement dates.

    For each breach with a valid yf_ticker and reported_date:
      1. Download stock price history
      2. Estimate FF 5-factor model in [-252, -31] window
      3. Compute abnormal returns in [-1, +5] event window
      4. Aggregate by event day and VIX regime

    Returns dict with:
      - 'ar_by_day': dict of DataFrames (pooled/high_vol/low_vol) with mean AR by event day
      - 'car_summary': dict of CAR[-1,+5] summary stats by regime
      - 'event_count': dict of event counts and skip reasons
      - 'event_ar': DataFrame of all individual event-day AR observations
    """
    import statsmodels.api as sm
    from scipy import stats as scipy_stats
    import yfinance as yf_lib
    from clean import fetch_fama_french_data, fetch_vix_data

    # --- 1. Factor and VIX data ---
    ff_df = fetch_fama_french_data()
    if ff_df.empty:
        raise RuntimeError("Fama-French data unavailable for event study.")
    ff_work = ff_df.reset_index()
    ff_work['date'] = pd.to_datetime(ff_work['date']).dt.normalize()
    ff_work = ff_work.sort_values('date').reset_index(drop=True)

    vix_df = fetch_vix_data()
    if vix_df.empty:
        raise RuntimeError("VIX data unavailable for event study.")
    vix_df['date'] = pd.to_datetime(vix_df['date']).dt.normalize()
    vix_df = vix_df.sort_values('date').reset_index(drop=True)

    # Trading day calendar from FF index (sorted numpy datetime64 array)
    trading_days = ff_work['date'].values
    n_td = len(trading_days)

    # --- 2. Filter breach events with ticker + reported_date ---
    mask = df['yf_ticker'].notna() & df['reported_date'].notna()
    events = df.loc[mask, ['org_name', 'yf_ticker', 'reported_date']].copy()
    events = events.reset_index(drop=True)
    logger.info("Event study: %d events with ticker + reported_date (%d unique tickers)",
                len(events), events['yf_ticker'].nunique())

    # --- 3. VIX at reported_date for regime split ---
    events = events.sort_values('reported_date').reset_index(drop=True)
    events = pd.merge_asof(
        events, vix_df[['date', 'vix_close']],
        left_on='reported_date', right_on='date',
        direction='nearest', tolerance=pd.Timedelta(days=5),
    )
    events['high_vol'] = (events['vix_close'] >= VIX_THRESHOLD).astype(int)
    events = events.drop(columns=['date'], errors='ignore')

    # --- 4. Download stock prices in batches ---
    unique_tickers = sorted(events['yf_ticker'].unique())
    earliest = events['reported_date'].min() - pd.Timedelta(days=400)
    latest = events['reported_date'].max() + pd.Timedelta(days=15)

    logger.info("Downloading prices for %d tickers (%s to %s)",
                len(unique_tickers),
                earliest.strftime('%Y-%m-%d'),
                latest.strftime('%Y-%m-%d'))

    all_prices = {}
    batch_size = 20
    for i in range(0, len(unique_tickers), batch_size):
        batch = unique_tickers[i:i + batch_size]
        try:
            data = yf_lib.download(
                batch if len(batch) > 1 else batch[0],
                start=earliest, end=latest,
                auto_adjust=True, progress=False,
            )
            if data.empty:
                continue
            if isinstance(data.columns, pd.MultiIndex):
                try:
                    close_df = data['Close']
                except KeyError:
                    continue
                if isinstance(close_df, pd.Series):
                    s = close_df.dropna()
                    if len(s) > 0:
                        all_prices[batch[0]] = s
                else:
                    for t in batch:
                        if t in close_df.columns:
                            s = close_df[t].dropna()
                            if len(s) > 0:
                                all_prices[t] = s
            else:
                if 'Close' in data.columns:
                    s = data['Close'].dropna()
                    if len(s) > 0:
                        all_prices[batch[0]] = s
        except Exception as e:
            logger.warning("Download failed for batch %d-%d: %s",
                           i, i + len(batch), e)

    logger.info("Got prices for %d / %d tickers",
                len(all_prices), len(unique_tickers))

    # --- 5. Compute daily returns (pct points) and pre-merge with FF ---
    ff_indexed = ff_work.set_index('date')

    ticker_merged = {}
    for ticker, prices in all_prices.items():
        ret = prices.pct_change() * 100  # pct points to match FF data
        ret = ret.dropna()
        ret_df = ret.to_frame('stock_ret')
        ret_df.index = pd.to_datetime(ret_df.index).normalize()
        merged = ret_df.join(ff_indexed, how='inner')
        if len(merged) > 0:
            merged['excess_ret'] = merged['stock_ret'] - merged['RF']
            ticker_merged[ticker] = merged.sort_index()

    # --- 6. Per-event: estimate model → compute AR ---
    ff_factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    event_ar_records = []
    skipped = {'no_prices': 0, 'no_td_match': 0, 'short_est': 0, 'ols_fail': 0}

    for idx, event in events.iterrows():
        ticker = event['yf_ticker']
        reported = pd.Timestamp(event['reported_date']).normalize()

        if ticker not in ticker_merged:
            skipped['no_prices'] += 1
            continue

        tm = ticker_merged[ticker]

        # Map reported_date to nearest trading day index
        td_idx = np.searchsorted(trading_days, np.datetime64(reported))
        candidates = []
        if td_idx > 0:
            candidates.append(td_idx - 1)
        if td_idx < n_td:
            candidates.append(td_idx)

        best_td = None
        best_dist = np.timedelta64(999, 'D')
        for ci in candidates:
            dist = abs(trading_days[ci] - np.datetime64(reported))
            if dist < best_dist:
                best_dist = dist
                best_td = ci

        if best_td is None or best_dist > np.timedelta64(5, 'D'):
            skipped['no_td_match'] += 1
            continue

        event_td = best_td  # index into trading_days for day 0

        # Window bounds (indices into trading_days)
        est_start_idx = max(0, event_td + EST_WINDOW[0])
        est_end_idx = event_td + EST_WINDOW[1]
        evt_start_idx = event_td + EVENT_WINDOW[0]
        evt_end_idx = event_td + EVENT_WINDOW[1]

        if evt_start_idx < 0 or evt_end_idx >= n_td or est_end_idx < 0:
            skipped['no_td_match'] += 1
            continue

        # Estimation window dates
        est_date_set = set(
            pd.Timestamp(d)
            for d in trading_days[est_start_idx:est_end_idx + 1]
        )
        est_data = tm[tm.index.isin(est_date_set)]

        if len(est_data) < MIN_EST_DAYS:
            skipped['short_est'] += 1
            continue

        # OLS: excess_ret ~ const + Mkt-RF + SMB + HML + RMW + CMA
        try:
            y_est = est_data['excess_ret']
            X_est = sm.add_constant(est_data[ff_factors])
            model = sm.OLS(y_est, X_est).fit()
        except Exception:
            skipped['ols_fail'] += 1
            continue

        # Event window: compute AR for each day
        for j in range(evt_start_idx, evt_end_idx + 1):
            d = pd.Timestamp(trading_days[j])
            if d not in tm.index:
                continue

            row = tm.loc[d]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            actual_excess = row['excess_ret']
            x_vals = np.array([1.0] + [row[f] for f in ff_factors])
            expected_excess = np.dot(model.params.values, x_vals)
            ar = actual_excess - expected_excess

            event_ar_records.append({
                'event_idx': idx,
                'ticker': ticker,
                'reported_date': reported,
                'event_day': j - event_td,
                'date': d,
                'ar': ar,
                'actual_excess': actual_excess,
                'expected_excess': expected_excess,
                'vix': event['vix_close'],
                'high_vol': event['high_vol'],
                'est_r2': model.rsquared,
                'est_nobs': int(model.nobs),
            })

    n_events_used = (
        len(set(r['event_idx'] for r in event_ar_records))
        if event_ar_records else 0
    )
    logger.info("Event study: %d AR observations from %d events",
                len(event_ar_records), n_events_used)
    logger.info("  Skipped: %s", skipped)

    if not event_ar_records:
        empty = {'pooled': pd.DataFrame(), 'high_vol': pd.DataFrame(),
                 'low_vol': pd.DataFrame()}
        return {
            'ar_by_day': empty,
            'car_summary': {k: {} for k in empty},
            'event_count': {
                'total_events': len(events), 'usable': 0,
                'skipped': skipped,
            },
            'event_ar': pd.DataFrame(),
        }

    # --- 7. Aggregation & testing ---
    ar_df = pd.DataFrame(event_ar_records)

    def _ar_day_stats(sub):
        """Mean AR, t-stat (H0: AR=0), p-value by event day."""
        rows = []
        for day in sorted(sub['event_day'].unique()):
            day_ar = sub.loc[sub['event_day'] == day, 'ar']
            if len(day_ar) < 2:
                continue
            t, p = scipy_stats.ttest_1samp(day_ar, 0)
            rows.append({
                'event_day': int(day),
                'mean_ar': day_ar.mean(),
                't_stat': t,
                'p_value': p,
                'n': len(day_ar),
            })
        return pd.DataFrame(rows)

    ar_by_day = {
        'pooled': _ar_day_stats(ar_df),
        'high_vol': _ar_day_stats(ar_df[ar_df['high_vol'] == 1]),
        'low_vol': _ar_day_stats(ar_df[ar_df['high_vol'] == 0]),
    }

    # CAR[-1, +5] per event
    car_events = ar_df.groupby('event_idx').agg(
        car=('ar', 'sum'),
        n_days=('ar', 'count'),
        high_vol=('high_vol', 'first'),
        ticker=('ticker', 'first'),
    ).reset_index()
    car_valid = car_events[car_events['n_days'] >= 5]  # require >= 5 of 7 days

    def _car_stats(sub):
        if len(sub) < 2:
            return {'mean_car': np.nan, 't_stat': np.nan, 'p_value': np.nan,
                    'n': len(sub)}
        t, p = scipy_stats.ttest_1samp(sub['car'], 0)
        return {
            'mean_car': sub['car'].mean(),
            'median_car': sub['car'].median(),
            'std_car': sub['car'].std(),
            't_stat': t,
            'p_value': p,
            'n': len(sub),
        }

    car_summary = {
        'pooled': _car_stats(car_valid),
        'high_vol': _car_stats(car_valid[car_valid['high_vol'] == 1]),
        'low_vol': _car_stats(car_valid[car_valid['high_vol'] == 0]),
    }

    event_count = {
        'total_events': len(events),
        'usable': n_events_used,
        'high_vol': int(ar_df[ar_df['high_vol'] == 1]['event_idx'].nunique()),
        'low_vol': int(ar_df[ar_df['high_vol'] == 0]['event_idx'].nunique()),
        'tickers_used': int(ar_df['ticker'].nunique()),
        'car_events': len(car_valid),
        'skipped': skipped,
    }

    return {
        'ar_by_day': ar_by_day,
        'car_summary': car_summary,
        'event_count': event_count,
        'event_ar': ar_df,
    }


def print_event_study_results(results: Dict[str, Any]):
    """Print Step 5 event study results: AR by day, CAR summary, event counts."""

    ec = results['event_count']
    ar_by_day = results['ar_by_day']
    car_summary = results['car_summary']

    print("=" * 80)
    print("STEP 5: FAMA-FRENCH EVENT STUDY — BREACH ANNOUNCEMENTS")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Event date: reported_date (breach disclosure)")
    print(f"Event window: [{EVENT_WINDOW[0]}, +{EVENT_WINDOW[1]}] trading days")
    print(f"Estimation window: [{EST_WINDOW[0]}, {EST_WINDOW[1]}] trading days")
    print(f"Model: FF 5-Factor (Mkt-RF, SMB, HML, RMW, CMA)")
    print(f"Min estimation days: {MIN_EST_DAYS}  |  VIX threshold: {VIX_THRESHOLD}")
    print()

    # --- Table 3: Event counts ---
    print("=" * 80)
    print("  EVENT COUNTS")
    print("=" * 80)
    print(f"  Total events (ticker + reported_date):  {ec['total_events']:,}")
    print(f"  Usable events (sufficient data):        {ec.get('usable', 0):,}")
    print(f"    High volatility (VIX >= {VIX_THRESHOLD}):         {ec.get('high_vol', 0):,}")
    print(f"    Low volatility (VIX < {VIX_THRESHOLD}):           {ec.get('low_vol', 0):,}")
    print(f"  Unique tickers used:                    {ec.get('tickers_used', 0):,}")
    print(f"  Events with valid CAR:                  {ec.get('car_events', 0):,}")
    sk = ec.get('skipped', {})
    if sk:
        print(f"  Skipped — no price data:                {sk.get('no_prices', 0):,}")
        print(f"  Skipped — no trading day match:         {sk.get('no_td_match', 0):,}")
        print(f"  Skipped — est. window < {MIN_EST_DAYS} days:     {sk.get('short_est', 0):,}")
        print(f"  Skipped — OLS failed:                   {sk.get('ols_fail', 0):,}")
    print()

    # --- Table 1: AR by event day ---
    print("=" * 80)
    print("  ABNORMAL RETURNS (AR) BY EVENT DAY")
    print("=" * 80)

    for regime_label, regime_name in [
        ('pooled', 'POOLED (All Events)'),
        ('high_vol', f'HIGH VOLATILITY (VIX >= {VIX_THRESHOLD})'),
        ('low_vol', f'LOW VOLATILITY (VIX < {VIX_THRESHOLD})'),
    ]:
        regime_df = ar_by_day.get(regime_label, pd.DataFrame())
        if regime_df.empty:
            print(f"\n  {regime_name}: No data")
            continue

        print(f"\n  {regime_name}")
        print(f"  {'Day':>5s} {'Mean AR%':>10s} {'t-stat':>10s} {'p-value':>10s} {'N':>8s}  Sig")
        print("  " + "-" * 53)

        for _, row in regime_df.iterrows():
            p = row['p_value']
            if p < 0.001:
                sig = "***"
            elif p < 0.01:
                sig = "**"
            elif p < 0.05:
                sig = "*"
            elif p < 0.10:
                sig = "."
            else:
                sig = ""
            print(f"  {int(row['event_day']):>5d} {row['mean_ar']:>10.4f} "
                  f"{row['t_stat']:>10.3f} {p:>10.4f} {int(row['n']):>8d}  {sig}")

    print()

    # --- Table 2: CAR summary ---
    print("=" * 80)
    print(f"  CUMULATIVE ABNORMAL RETURN  CAR[{EVENT_WINDOW[0]}, +{EVENT_WINDOW[1]}]")
    print("=" * 80)

    header = (f"  {'Regime':<25s} {'Mean CAR%':>10s} {'Median':>10s} "
              f"{'Std Dev':>10s} {'t-stat':>10s} {'p-value':>10s} {'N':>6s}  Sig")
    print(header)
    print("  " + "-" * 91)

    for regime_label, regime_name in [
        ('pooled', 'Pooled'),
        ('high_vol', f'High Vol (VIX >= {VIX_THRESHOLD})'),
        ('low_vol', f'Low Vol (VIX < {VIX_THRESHOLD})'),
    ]:
        cs = car_summary.get(regime_label, {})
        if not cs or cs.get('n', 0) < 2:
            print(f"  {regime_name:<25s}     insufficient data")
            continue

        p = cs['p_value']
        if p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        elif p < 0.05:
            sig = "*"
        elif p < 0.10:
            sig = "."
        else:
            sig = ""

        print(f"  {regime_name:<25s} {cs['mean_car']:>10.4f} "
              f"{cs.get('median_car', np.nan):>10.4f} "
              f"{cs.get('std_car', np.nan):>10.4f} "
              f"{cs['t_stat']:>10.3f} {p:>10.4f} {cs['n']:>6d}  {sig}")

    print()
    print(f"  Significance: *** p<0.001, ** p<0.01, * p<0.05, . p<0.10")
    print()
    print("=" * 80)


# =============================================================================
# STEP 6: FINBERT SENTIMENT + EVENT STUDY INTEGRATION
# =============================================================================
# Asks: does the tone of the breach disclosure predict the severity of the
# stock price reaction?
#
# Uses FinBERT (ProsusAI/finbert) to score `incident_details` text, then
# incorporates sentiment into the event study framework from Step 5.
#
# A) AR by event day x sentiment regime (negative vs non-negative)
# B) CAR[-1,+5] by sentiment regime
# C) 2x2 table: sentiment x VIX regime
# D) Cross-sectional OLS: CAR ~ sentiment_score + high_vol + controls
# =============================================================================

SENTIMENT_THRESHOLD = 0  # sentiment_score < 0 is "negative sentiment"

# Step 7: Lagged firm-level news sentiment
NEWS_SENTIMENT_THRESHOLD = 0   # news_sent < 0 is "negative news sentiment"
LAGGED_WINDOWS = [7, 30, 60]
PRIMARY_WINDOW = 30


def run_sentiment_analysis(df: pd.DataFrame, event_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run FinBERT sentiment analysis on breach disclosures and integrate
    with Step 5 event study results.

    Args:
        df: Enriched breach DataFrame (must have 'incident_details' column)
        event_results: Output from run_event_study() (must have 'event_ar' DataFrame)

    Returns dict with:
        - 'sentiment_df': DataFrame with per-record sentiment scores
        - 'ar_by_day_sentiment': AR by event day for each sentiment regime
        - 'car_by_sentiment': CAR summary by sentiment regime
        - 'car_2x2': 2x2 CAR table (sentiment x VIX)
        - 'ols_results': Cross-sectional OLS model
        - 'sentiment_stats': Summary statistics of sentiment distribution
    """
    import statsmodels.api as sm
    from scipy import stats as scipy_stats
    from clean import compute_finbert_sentiment

    # --- 1. Compute sentiment scores ---
    logger.info("Step 6: Computing FinBERT sentiment on incident_details")
    sentiment_df = compute_finbert_sentiment(df['incident_details'])
    sentiment_df.index = df.index

    # Add sentiment columns to df copy
    df_sent = df.copy()
    for col in sentiment_df.columns:
        df_sent[col] = sentiment_df[col].values
    df_sent['negative_sentiment'] = (df_sent['sentiment_score'] < SENTIMENT_THRESHOLD).astype(int)

    # Sentiment distribution summary
    n_total = len(df_sent)
    n_with_text = df_sent['incident_details'].notna().sum()
    sentiment_stats = {
        'n_total': n_total,
        'n_with_text': int(n_with_text),
        'n_negative': int(df_sent['negative_sentiment'].sum()),
        'n_non_negative': int((df_sent['negative_sentiment'] == 0).sum()),
        'mean_score': float(df_sent['sentiment_score'].mean()),
        'median_score': float(df_sent['sentiment_score'].median()),
        'std_score': float(df_sent['sentiment_score'].std()),
        'label_counts': df_sent['sentiment_label'].value_counts().to_dict(),
    }

    # --- 2. Merge sentiment with event AR data ---
    event_ar = event_results.get('event_ar', pd.DataFrame())
    if event_ar.empty:
        logger.warning("No event AR data from Step 5, cannot integrate sentiment")
        return {
            'sentiment_df': sentiment_df,
            'ar_by_day_sentiment': {},
            'car_by_sentiment': {},
            'car_2x2': {},
            'ols_results': None,
            'sentiment_stats': sentiment_stats,
        }

    # Build lookup: event_idx -> sentiment for the corresponding breach
    # event_ar has 'event_idx' which indexes into the events df used in run_event_study
    # We need to match via ticker + reported_date
    event_ar_merged = event_ar.copy()

    # Build sentiment lookup keyed by (ticker, reported_date)
    ticker_date_sent = {}
    for i, row in df_sent.iterrows():
        ticker = row.get('yf_ticker')
        rd = row.get('reported_date')
        if pd.notna(ticker) and pd.notna(rd):
            rd_ts = pd.Timestamp(rd).normalize()
            key = (ticker, rd_ts)
            if key not in ticker_date_sent:
                ticker_date_sent[key] = {
                    'sentiment_score': row['sentiment_score'],
                    'sentiment_label': row['sentiment_label'],
                    'negative_sentiment': row['negative_sentiment'],
                }

    # Map sentiment onto event_ar rows
    sent_scores = []
    neg_flags = []
    for _, ar_row in event_ar_merged.iterrows():
        key = (ar_row['ticker'], pd.Timestamp(ar_row['reported_date']).normalize())
        info = ticker_date_sent.get(key)
        if info:
            sent_scores.append(info['sentiment_score'])
            neg_flags.append(info['negative_sentiment'])
        else:
            sent_scores.append(np.nan)
            neg_flags.append(np.nan)

    event_ar_merged['sentiment_score'] = sent_scores
    event_ar_merged['negative_sentiment'] = neg_flags

    # Drop events without sentiment match
    has_sent = event_ar_merged['sentiment_score'].notna()
    event_ar_sent = event_ar_merged[has_sent].copy()
    logger.info("Step 6: %d/%d AR observations matched to sentiment",
                len(event_ar_sent), len(event_ar_merged))

    # --- 3. AR by event day x sentiment regime ---
    def _ar_day_stats(sub):
        rows = []
        for day in sorted(sub['event_day'].unique()):
            day_ar = sub.loc[sub['event_day'] == day, 'ar']
            if len(day_ar) < 2:
                continue
            t, p = scipy_stats.ttest_1samp(day_ar, 0)
            rows.append({
                'event_day': int(day),
                'mean_ar': day_ar.mean(),
                't_stat': t,
                'p_value': p,
                'n': len(day_ar),
            })
        return pd.DataFrame(rows)

    ar_by_day_sentiment = {
        'negative': _ar_day_stats(event_ar_sent[event_ar_sent['negative_sentiment'] == 1]),
        'non_negative': _ar_day_stats(event_ar_sent[event_ar_sent['negative_sentiment'] == 0]),
    }

    # --- 4. CAR[-1,+5] by sentiment regime ---
    car_events = event_ar_sent.groupby('event_idx').agg(
        car=('ar', 'sum'),
        n_days=('ar', 'count'),
        high_vol=('high_vol', 'first'),
        negative_sentiment=('negative_sentiment', 'first'),
        sentiment_score=('sentiment_score', 'first'),
        ticker=('ticker', 'first'),
    ).reset_index()
    car_valid = car_events[car_events['n_days'] >= 5]

    def _car_stats(sub):
        if len(sub) < 2:
            return {'mean_car': np.nan, 't_stat': np.nan, 'p_value': np.nan, 'n': len(sub)}
        t, p = scipy_stats.ttest_1samp(sub['car'], 0)
        return {
            'mean_car': sub['car'].mean(),
            'median_car': sub['car'].median(),
            'std_car': sub['car'].std(),
            't_stat': t,
            'p_value': p,
            'n': len(sub),
        }

    car_by_sentiment = {
        'pooled': _car_stats(car_valid),
        'negative': _car_stats(car_valid[car_valid['negative_sentiment'] == 1]),
        'non_negative': _car_stats(car_valid[car_valid['negative_sentiment'] == 0]),
    }

    # --- 5. 2x2 table: sentiment x VIX regime ---
    car_2x2 = {}
    for sent_label, sent_val in [('negative', 1), ('non_negative', 0)]:
        for vol_label, vol_val in [('high_vol', 1), ('low_vol', 0)]:
            sub = car_valid[(car_valid['negative_sentiment'] == sent_val)
                            & (car_valid['high_vol'] == vol_val)]
            key = f"{sent_label}_{vol_label}"
            car_2x2[key] = _car_stats(sub)

    # --- 6. Cross-sectional OLS: CAR ~ sentiment_score + controls ---
    ols_results = None
    # Merge with df_sent to get log_affected and log_market_cap
    car_ols = car_valid.copy()

    # Build a lookup for controls from df_sent
    control_lookup = {}
    for i, row in df_sent.iterrows():
        ticker = row.get('yf_ticker')
        rd = row.get('reported_date')
        if pd.notna(ticker) and pd.notna(rd):
            rd_ts = pd.Timestamp(rd).normalize()
            key = (ticker, rd_ts)
            ta = row.get('total_affected')
            mc = row.get('yf_market_cap')
            control_lookup[key] = {
                'log_affected': np.log10(ta + 1) if pd.notna(ta) and ta > 0 else np.nan,
                'log_market_cap': np.log10(mc) if pd.notna(mc) and mc > 0 else np.nan,
            }

    # We need to re-get reported_date for each event via event_ar_sent
    event_reported = event_ar_sent.groupby('event_idx').agg(
        ticker=('ticker', 'first'),
        reported_date=('reported_date', 'first'),
    )

    log_aff = []
    log_mc = []
    for _, row in car_ols.iterrows():
        evt = event_reported.loc[row['event_idx']] if row['event_idx'] in event_reported.index else None
        if evt is not None:
            key = (evt['ticker'], pd.Timestamp(evt['reported_date']).normalize())
            ctrls = control_lookup.get(key, {})
            log_aff.append(ctrls.get('log_affected', np.nan))
            log_mc.append(ctrls.get('log_market_cap', np.nan))
        else:
            log_aff.append(np.nan)
            log_mc.append(np.nan)

    car_ols['log_affected'] = log_aff
    car_ols['log_market_cap'] = log_mc

    # Build OLS: CAR ~ sentiment_score + high_vol + log_affected + log_market_cap + sentiment_score*high_vol
    car_ols['sent_x_highvol'] = car_ols['sentiment_score'] * car_ols['high_vol']

    ols_vars = ['sentiment_score', 'high_vol', 'log_affected', 'log_market_cap', 'sent_x_highvol']
    ols_subset = car_ols.dropna(subset=['car'] + ols_vars)

    if len(ols_subset) >= len(ols_vars) + 5:
        y = ols_subset['car']
        X = sm.add_constant(ols_subset[ols_vars])
        ols_results = sm.OLS(y, X).fit()
        logger.info("Step 6 OLS: N=%d, R²=%.4f", int(ols_results.nobs), ols_results.rsquared)
    else:
        logger.warning("Step 6: Not enough observations for OLS (%d)", len(ols_subset))

    return {
        'sentiment_df': sentiment_df,
        'ar_by_day_sentiment': ar_by_day_sentiment,
        'car_by_sentiment': car_by_sentiment,
        'car_2x2': car_2x2,
        'ols_results': ols_results,
        'sentiment_stats': sentiment_stats,
    }


def print_sentiment_analysis_results(results: Dict[str, Any]):
    """Print Step 6 sentiment-augmented event study results."""

    sentiment_stats = results['sentiment_stats']
    ar_by_day = results['ar_by_day_sentiment']
    car_by_sent = results['car_by_sentiment']
    car_2x2 = results['car_2x2']
    ols_model = results['ols_results']

    print("=" * 80)
    print("STEP 6: FINBERT SENTIMENT + EVENT STUDY INTEGRATION")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: ProsusAI/finbert (BERT fine-tuned on financial text)")
    print(f"Input: incident_details column")
    print(f"Sentiment threshold: score < {SENTIMENT_THRESHOLD} = negative")
    print()

    # --- Table 5: Sentiment distribution ---
    print("=" * 80)
    print("  SENTIMENT DISTRIBUTION")
    print("=" * 80)
    print(f"  Total records:              {sentiment_stats['n_total']:,}")
    print(f"  Records with text:          {sentiment_stats['n_with_text']:,}")
    print(f"  Negative sentiment:         {sentiment_stats['n_negative']:,}")
    print(f"  Non-negative sentiment:     {sentiment_stats['n_non_negative']:,}")
    print(f"  Mean sentiment score:       {sentiment_stats['mean_score']:.4f}")
    print(f"  Median sentiment score:     {sentiment_stats['median_score']:.4f}")
    print(f"  Std sentiment score:        {sentiment_stats['std_score']:.4f}")
    print(f"  Label counts:")
    for label, count in sorted(sentiment_stats['label_counts'].items()):
        pct = count / sentiment_stats['n_total'] * 100
        print(f"    {label:<12s}: {count:>6,} ({pct:>5.1f}%)")
    print()

    # --- Table 1: AR by event day x sentiment regime ---
    print("=" * 80)
    print("  ABNORMAL RETURNS (AR) BY EVENT DAY x SENTIMENT REGIME")
    print("=" * 80)

    for regime_label, regime_name in [
        ('negative', 'NEGATIVE SENTIMENT (score < 0)'),
        ('non_negative', 'NON-NEGATIVE SENTIMENT (score >= 0)'),
    ]:
        regime_df = ar_by_day.get(regime_label, pd.DataFrame())
        if regime_df.empty:
            print(f"\n  {regime_name}: No data")
            continue

        print(f"\n  {regime_name}")
        print(f"  {'Day':>5s} {'Mean AR%':>10s} {'t-stat':>10s} {'p-value':>10s} {'N':>8s}  Sig")
        print("  " + "-" * 53)

        for _, row in regime_df.iterrows():
            p = row['p_value']
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.10 else ""
            print(f"  {int(row['event_day']):>5d} {row['mean_ar']:>10.4f} "
                  f"{row['t_stat']:>10.3f} {p:>10.4f} {int(row['n']):>8d}  {sig}")

    print()

    # --- Table 2: CAR summary by sentiment regime ---
    print("=" * 80)
    print(f"  CUMULATIVE ABNORMAL RETURN  CAR[{EVENT_WINDOW[0]}, +{EVENT_WINDOW[1]}] BY SENTIMENT")
    print("=" * 80)

    header = (f"  {'Regime':<25s} {'Mean CAR%':>10s} {'Median':>10s} "
              f"{'Std Dev':>10s} {'t-stat':>10s} {'p-value':>10s} {'N':>6s}  Sig")
    print(header)
    print("  " + "-" * 91)

    for regime_label, regime_name in [
        ('pooled', 'Pooled'),
        ('negative', 'Negative sentiment'),
        ('non_negative', 'Non-negative sentiment'),
    ]:
        cs = car_by_sent.get(regime_label, {})
        if not cs or cs.get('n', 0) < 2:
            print(f"  {regime_name:<25s}     insufficient data")
            continue

        p = cs['p_value']
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.10 else ""
        print(f"  {regime_name:<25s} {cs['mean_car']:>10.4f} "
              f"{cs.get('median_car', np.nan):>10.4f} "
              f"{cs.get('std_car', np.nan):>10.4f} "
              f"{cs['t_stat']:>10.3f} {p:>10.4f} {cs['n']:>6d}  {sig}")

    print()

    # --- Table 3: 2x2 CAR: sentiment x VIX ---
    print("=" * 80)
    print(f"  2x2 CAR[{EVENT_WINDOW[0]}, +{EVENT_WINDOW[1]}]: SENTIMENT x VIX REGIME")
    print("=" * 80)

    print(f"  {'':>25s} {'Low Vol':>20s} {'High Vol':>20s}")
    print(f"  {'':>25s} {'(VIX < ' + str(VIX_THRESHOLD) + ')':>20s} {'(VIX >= ' + str(VIX_THRESHOLD) + ')':>20s}")
    print("  " + "-" * 65)

    for sent_label, sent_name in [('negative', 'Negative sentiment'), ('non_negative', 'Non-negative sentiment')]:
        row_str = f"  {sent_name:<25s}"
        for vol_label in ['low_vol', 'high_vol']:
            key = f"{sent_label}_{vol_label}"
            cs = car_2x2.get(key, {})
            n = cs.get('n', 0)
            if n >= 2:
                mean_car = cs['mean_car']
                p = cs['p_value']
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.10 else ""
                row_str += f" {mean_car:>8.4f}{sig:<2s} (N={n:>3d})"
            else:
                row_str += f" {'N/A':>10s} (N={n:>3d})"
        print(row_str)

    print()

    # --- Table 4: Cross-sectional OLS ---
    print("=" * 80)
    print(f"  CROSS-SECTIONAL OLS: CAR[{EVENT_WINDOW[0]}, +{EVENT_WINDOW[1]}] ~ SENTIMENT + CONTROLS")
    print("=" * 80)

    if ols_model is not None:
        print(f"  N = {int(ols_model.nobs):,}  |  R² = {ols_model.rsquared:.4f}  |  "
              f"Adj. R² = {ols_model.rsquared_adj:.4f}")
        print(f"  F-statistic: {ols_model.fvalue:.3f}  (p = {ols_model.f_pvalue:.4e})")
        print()

        header = f"  {'Variable':<22s} {'Coef':>10s} {'Std Err':>10s} {'t':>8s} {'P>|t|':>10s}  Sig"
        print(header)
        print("  " + "-" * 68)

        var_descriptions = {
            'const': 'Intercept',
            'sentiment_score': 'Sentiment score',
            'high_vol': 'High volatility (VIX>=25)',
            'log_affected': 'log10(affected)',
            'log_market_cap': 'log10(market cap)',
            'sent_x_highvol': 'Sentiment x High Vol',
        }

        for var in ols_model.params.index:
            coef = ols_model.params[var]
            se = ols_model.bse[var]
            t = ols_model.tvalues[var]
            p = ols_model.pvalues[var]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.10 else ""
            display_name = var_descriptions.get(var, var)
            print(f"  {display_name:<22s} {coef:>10.4f} {se:>10.4f} {t:>8.3f} {p:>10.4f}  {sig}")

        print()
        print(f"  Significance: *** p<0.001, ** p<0.01, * p<0.05, . p<0.10")
    else:
        print("  OLS not computed (insufficient observations)")

    print()
    print("=" * 80)


# =============================================================================
# STEP 7: LAGGED FIRM-LEVEL NEWS SENTIMENT
# =============================================================================
# Asks: does pre-breach news sentiment (a proxy for firm reputation standing)
# predict the severity of the stock price reaction?
#
# Key motivation:
# - Step 6 disclosure sentiment has 98.3% negative → narrow distribution
# - Pre-breach news sentiment has wider distribution → more statistical power
# - Temporally prior → cleaner causal identification
# - May absorb the market cap effect (reputation, not just size, protects firms)
#
# A) AR by event day × lagged news sentiment regime (30d window)
# B) CAR[-1,+5] by lagged sentiment regime
# C) 2×2 table: lagged sentiment × VIX regime
# D) Cross-sectional OLS:
#    Model A: CAR ~ news_sent + high_vol + controls + news_sent×high_vol
#    Model B: Model A + disclosure_sentiment + disclosure_sent×high_vol
#    Model C: Model A without log_market_cap (absorption test)
# E) Robustness: 7d and 60d windows
# =============================================================================


def run_lagged_sentiment_analysis(
    df: pd.DataFrame,
    event_results: Dict[str, Any],
    sentiment_results: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run lagged (pre-breach) news sentiment analysis and integrate
    with event study and disclosure sentiment results.

    Args:
        df: Enriched breach DataFrame
        event_results: Output from run_event_study()
        sentiment_results: Output from run_sentiment_analysis() (Step 6)

    Returns dict with all Step 7 results.
    """
    import statsmodels.api as sm
    from scipy import stats as scipy_stats
    from clean import compute_lagged_news_sentiment

    # --- 1. Compute lagged news sentiment ---
    logger.info("Step 7: Computing lagged news sentiment")
    lagged_df = compute_lagged_news_sentiment(df, windows=LAGGED_WINDOWS)
    lagged_df.index = df.index

    pw = PRIMARY_WINDOW  # 30d
    sent_col = f'news_sent_{pw}d'
    count_col = f'news_count_{pw}d'

    # Coverage stats per window
    coverage_stats = {}
    for w in LAGGED_WINDOWS:
        sc = f'news_count_{w}d'
        has_data = (lagged_df[sc] > 0).sum()
        coverage_stats[w] = {
            'events_with_articles': int(has_data),
            'total_events': len(df),
            'coverage_pct': 100 * has_data / len(df),
            'mean_articles': float(lagged_df.loc[lagged_df[sc] > 0, sc].mean()) if has_data > 0 else 0,
        }

    # Sentiment distribution (primary window, events with data)
    has_primary = lagged_df[count_col] > 0
    primary_scores = lagged_df.loc[has_primary, sent_col]
    sentiment_dist = {
        'n_with_data': int(has_primary.sum()),
        'n_total': len(df),
        'mean': float(primary_scores.mean()) if len(primary_scores) > 0 else np.nan,
        'median': float(primary_scores.median()) if len(primary_scores) > 0 else np.nan,
        'std': float(primary_scores.std()) if len(primary_scores) > 0 else np.nan,
        'min': float(primary_scores.min()) if len(primary_scores) > 0 else np.nan,
        'max': float(primary_scores.max()) if len(primary_scores) > 0 else np.nan,
        'pct_negative': float((primary_scores < NEWS_SENTIMENT_THRESHOLD).mean() * 100) if len(primary_scores) > 0 else np.nan,
        'pct_non_negative': float((primary_scores >= NEWS_SENTIMENT_THRESHOLD).mean() * 100) if len(primary_scores) > 0 else np.nan,
    }

    # --- 2. Merge with event AR data ---
    event_ar = event_results.get('event_ar', pd.DataFrame())
    if event_ar.empty:
        logger.warning("No event AR data from Step 5, cannot run Step 7")
        return {
            'lagged_df': lagged_df,
            'coverage_stats': coverage_stats,
            'sentiment_dist': sentiment_dist,
            'ar_by_day_news': {},
            'car_by_news_sent': {},
            'car_2x2_news': {},
            'ols_models': {},
            'robustness_ols': {},
        }

    # Build lookup: (ticker, reported_date) -> lagged sentiment + controls
    df_lagged = df.copy()
    for col in lagged_df.columns:
        df_lagged[col] = lagged_df[col].values

    # Also attach disclosure sentiment from Step 6
    sent_df_step6 = sentiment_results.get('sentiment_df', pd.DataFrame())
    if not sent_df_step6.empty and len(sent_df_step6) == len(df):
        df_lagged['disclosure_sentiment'] = sent_df_step6['sentiment_score'].values
    else:
        df_lagged['disclosure_sentiment'] = np.nan

    df_lagged['negative_news_sent'] = (df_lagged[sent_col] < NEWS_SENTIMENT_THRESHOLD).astype(int)
    # Mark events without news data as NaN
    df_lagged.loc[df_lagged[count_col] == 0, 'negative_news_sent'] = np.nan

    # Build lookup keyed by (ticker, reported_date)
    ticker_date_news = {}
    for i, row in df_lagged.iterrows():
        ticker = row.get('yf_ticker')
        rd = row.get('reported_date')
        if pd.notna(ticker) and pd.notna(rd):
            rd_ts = pd.Timestamp(rd).normalize()
            key = (ticker, rd_ts)
            if key not in ticker_date_news:
                ta = row.get('total_affected')
                mc = row.get('yf_market_cap')
                info = {
                    'negative_news_sent': row.get('negative_news_sent'),
                    'disclosure_sentiment': row.get('disclosure_sentiment'),
                    'log_affected': np.log10(ta + 1) if pd.notna(ta) and ta > 0 else np.nan,
                    'log_market_cap': np.log10(mc) if pd.notna(mc) and mc > 0 else np.nan,
                }
                for w in LAGGED_WINDOWS:
                    info[f'news_sent_{w}d'] = row.get(f'news_sent_{w}d')
                    info[f'news_count_{w}d'] = row.get(f'news_count_{w}d')
                ticker_date_news[key] = info

    # Map onto event_ar
    event_ar_merged = event_ar.copy()
    news_cols_to_map = ['negative_news_sent', 'disclosure_sentiment', 'log_affected', 'log_market_cap']
    for w in LAGGED_WINDOWS:
        news_cols_to_map += [f'news_sent_{w}d', f'news_count_{w}d']

    col_data = {c: [] for c in news_cols_to_map}
    for _, ar_row in event_ar_merged.iterrows():
        key = (ar_row['ticker'], pd.Timestamp(ar_row['reported_date']).normalize())
        info = ticker_date_news.get(key, {})
        for c in news_cols_to_map:
            col_data[c].append(info.get(c, np.nan))

    for c in news_cols_to_map:
        event_ar_merged[c] = col_data[c]

    # Filter to events with news data (primary window)
    has_news = event_ar_merged[sent_col].notna()
    event_ar_news = event_ar_merged[has_news].copy()
    logger.info("Step 7: %d/%d AR observations matched to lagged news sentiment",
                len(event_ar_news), len(event_ar_merged))

    # --- 3. AR by event day × lagged news sentiment regime ---
    def _ar_day_stats(sub):
        rows = []
        for day in sorted(sub['event_day'].unique()):
            day_ar = sub.loc[sub['event_day'] == day, 'ar']
            if len(day_ar) < 2:
                continue
            t, p = scipy_stats.ttest_1samp(day_ar, 0)
            rows.append({
                'event_day': int(day),
                'mean_ar': day_ar.mean(),
                't_stat': t,
                'p_value': p,
                'n': len(day_ar),
            })
        return pd.DataFrame(rows)

    ar_by_day_news = {
        'negative_news': _ar_day_stats(event_ar_news[event_ar_news['negative_news_sent'] == 1]),
        'non_negative_news': _ar_day_stats(event_ar_news[event_ar_news['negative_news_sent'] == 0]),
    }

    # --- 4. CAR by lagged sentiment regime ---
    car_events = event_ar_news.groupby('event_idx').agg(
        car=('ar', 'sum'),
        n_days=('ar', 'count'),
        high_vol=('high_vol', 'first'),
        negative_news_sent=('negative_news_sent', 'first'),
        ticker=('ticker', 'first'),
        **{f'news_sent_{w}d': (f'news_sent_{w}d', 'first') for w in LAGGED_WINDOWS},
        **{f'news_count_{w}d': (f'news_count_{w}d', 'first') for w in LAGGED_WINDOWS},
        disclosure_sentiment=('disclosure_sentiment', 'first'),
        log_affected=('log_affected', 'first'),
        log_market_cap=('log_market_cap', 'first'),
    ).reset_index()
    car_valid = car_events[car_events['n_days'] >= 5]

    def _car_stats(sub):
        if len(sub) < 2:
            return {'mean_car': np.nan, 't_stat': np.nan, 'p_value': np.nan, 'n': len(sub)}
        t, p = scipy_stats.ttest_1samp(sub['car'], 0)
        return {
            'mean_car': sub['car'].mean(),
            'median_car': sub['car'].median(),
            'std_car': sub['car'].std(),
            't_stat': t,
            'p_value': p,
            'n': len(sub),
        }

    car_by_news_sent = {
        'pooled': _car_stats(car_valid),
        'negative_news': _car_stats(car_valid[car_valid['negative_news_sent'] == 1]),
        'non_negative_news': _car_stats(car_valid[car_valid['negative_news_sent'] == 0]),
    }

    # --- 5. 2×2 CAR: lagged sentiment × VIX ---
    car_2x2_news = {}
    for sent_label, sent_val in [('negative_news', 1), ('non_negative_news', 0)]:
        for vol_label, vol_val in [('high_vol', 1), ('low_vol', 0)]:
            sub = car_valid[(car_valid['negative_news_sent'] == sent_val)
                            & (car_valid['high_vol'] == vol_val)]
            key = f"{sent_label}_{vol_label}"
            car_2x2_news[key] = _car_stats(sub)

    # --- 6. Cross-sectional OLS ---
    ols_models = {}
    car_ols = car_valid.copy()
    car_ols['news_sent_x_highvol'] = car_ols[sent_col] * car_ols['high_vol']

    # Model A: CAR ~ news_sent_30d + high_vol + log_affected + log_market_cap + news_sent×high_vol
    model_a_vars = [sent_col, 'high_vol', 'log_affected', 'log_market_cap', 'news_sent_x_highvol']
    ols_a = car_ols.dropna(subset=['car'] + model_a_vars)
    if len(ols_a) >= len(model_a_vars) + 5:
        y = ols_a['car']
        X = sm.add_constant(ols_a[model_a_vars])
        ols_models['model_a'] = sm.OLS(y, X).fit()
        logger.info("Step 7 Model A: N=%d, R²=%.4f", int(ols_models['model_a'].nobs), ols_models['model_a'].rsquared)

    # Model B (horse race): Model A + disclosure_sentiment + disclosure_sentiment×high_vol
    car_ols['disc_sent_x_highvol'] = car_ols['disclosure_sentiment'] * car_ols['high_vol']
    model_b_vars = model_a_vars + ['disclosure_sentiment', 'disc_sent_x_highvol']
    ols_b = car_ols.dropna(subset=['car'] + model_b_vars)
    if len(ols_b) >= len(model_b_vars) + 5:
        y = ols_b['car']
        X = sm.add_constant(ols_b[model_b_vars])
        ols_models['model_b'] = sm.OLS(y, X).fit()
        logger.info("Step 7 Model B: N=%d, R²=%.4f", int(ols_models['model_b'].nobs), ols_models['model_b'].rsquared)

    # Model C (absorption test): Model A without log_market_cap
    model_c_vars = [sent_col, 'high_vol', 'log_affected', 'news_sent_x_highvol']
    ols_c = car_ols.dropna(subset=['car'] + model_c_vars)
    if len(ols_c) >= len(model_c_vars) + 5:
        y = ols_c['car']
        X = sm.add_constant(ols_c[model_c_vars])
        ols_models['model_c'] = sm.OLS(y, X).fit()
        logger.info("Step 7 Model C: N=%d, R²=%.4f", int(ols_models['model_c'].nobs), ols_models['model_c'].rsquared)

    # --- 7. Robustness: 7d and 60d windows ---
    robustness_ols = {}
    for w in LAGGED_WINDOWS:
        if w == PRIMARY_WINDOW:
            continue  # Already done above
        w_sent_col = f'news_sent_{w}d'
        car_ols_w = car_valid.copy()
        car_ols_w['news_sent_x_highvol_w'] = car_ols_w[w_sent_col] * car_ols_w['high_vol']
        w_vars = [w_sent_col, 'high_vol', 'log_affected', 'log_market_cap', 'news_sent_x_highvol_w']
        ols_w = car_ols_w.dropna(subset=['car'] + w_vars)
        if len(ols_w) >= len(w_vars) + 5:
            y = ols_w['car']
            X = sm.add_constant(ols_w[w_vars])
            robustness_ols[f'{w}d'] = sm.OLS(y, X).fit()
            logger.info("Step 7 robustness %dd: N=%d, R²=%.4f",
                         w, int(robustness_ols[f'{w}d'].nobs), robustness_ols[f'{w}d'].rsquared)

    return {
        'lagged_df': lagged_df,
        'coverage_stats': coverage_stats,
        'sentiment_dist': sentiment_dist,
        'ar_by_day_news': ar_by_day_news,
        'car_by_news_sent': car_by_news_sent,
        'car_2x2_news': car_2x2_news,
        'ols_models': ols_models,
        'robustness_ols': robustness_ols,
    }


def print_lagged_sentiment_results(results: Dict[str, Any]):
    """Print Step 7 lagged news sentiment results."""

    coverage = results['coverage_stats']
    sent_dist = results['sentiment_dist']
    ar_by_day = results['ar_by_day_news']
    car_by_news = results['car_by_news_sent']
    car_2x2 = results['car_2x2_news']
    ols_models = results['ols_models']
    robustness = results['robustness_ols']

    print("=" * 80)
    print("STEP 7: LAGGED FIRM-LEVEL NEWS SENTIMENT")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: ProsusAI/finbert (pre-breach news articles)")
    print(f"Primary window: {PRIMARY_WINDOW} days before breach")
    print(f"Robustness windows: {[w for w in LAGGED_WINDOWS if w != PRIMARY_WINDOW]}")
    print(f"Sentiment threshold: score < {NEWS_SENTIMENT_THRESHOLD} = negative news")
    print()

    # --- Table 1: Coverage summary ---
    print("=" * 80)
    print("  TABLE 1: NEWS ARTICLE COVERAGE BY PRE-BREACH WINDOW")
    print("=" * 80)
    print(f"  {'Window':>10s} {'Events w/ articles':>20s} {'Coverage %':>12s} {'Mean articles':>15s}")
    print("  " + "-" * 60)
    for w in LAGGED_WINDOWS:
        cs = coverage.get(w, {})
        ev = cs.get('events_with_articles', 0)
        tot = cs.get('total_events', 0)
        pct = cs.get('coverage_pct', 0)
        ma = cs.get('mean_articles', 0)
        print(f"  {w:>7d}d  {ev:>10d} / {tot:<6d} {pct:>11.1f}% {ma:>14.1f}")
    print()

    # --- Table 2: Sentiment distribution (primary window) ---
    print("=" * 80)
    print(f"  TABLE 2: LAGGED NEWS SENTIMENT DISTRIBUTION ({PRIMARY_WINDOW}d WINDOW)")
    print("=" * 80)
    print(f"  Events with data:       {sent_dist['n_with_data']:,} / {sent_dist['n_total']:,}")
    if sent_dist['n_with_data'] > 0:
        print(f"  Mean sentiment:         {sent_dist['mean']:.4f}")
        print(f"  Median sentiment:       {sent_dist['median']:.4f}")
        print(f"  Std deviation:          {sent_dist['std']:.4f}")
        print(f"  Range:                  [{sent_dist['min']:.4f}, {sent_dist['max']:.4f}]")
        print(f"  % Negative (< 0):       {sent_dist['pct_negative']:.1f}%")
        print(f"  % Non-negative (>= 0):  {sent_dist['pct_non_negative']:.1f}%")
    print()

    # --- Table 3: AR by event day × lagged sentiment regime ---
    print("=" * 80)
    print(f"  TABLE 3: AR BY EVENT DAY x LAGGED NEWS SENTIMENT ({PRIMARY_WINDOW}d)")
    print("=" * 80)

    for regime_label, regime_name in [
        ('negative_news', f'NEGATIVE NEWS SENTIMENT (score < {NEWS_SENTIMENT_THRESHOLD})'),
        ('non_negative_news', f'NON-NEGATIVE NEWS SENTIMENT (score >= {NEWS_SENTIMENT_THRESHOLD})'),
    ]:
        regime_df = ar_by_day.get(regime_label, pd.DataFrame())
        if regime_df.empty:
            print(f"\n  {regime_name}: No data")
            continue

        print(f"\n  {regime_name}")
        print(f"  {'Day':>5s} {'Mean AR%':>10s} {'t-stat':>10s} {'p-value':>10s} {'N':>8s}  Sig")
        print("  " + "-" * 53)

        for _, row in regime_df.iterrows():
            p = row['p_value']
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.10 else ""
            print(f"  {int(row['event_day']):>5d} {row['mean_ar']:>10.4f} "
                  f"{row['t_stat']:>10.3f} {p:>10.4f} {int(row['n']):>8d}  {sig}")

    print()

    # --- Table 4: CAR by lagged sentiment regime ---
    print("=" * 80)
    print(f"  TABLE 4: CAR[{EVENT_WINDOW[0]}, +{EVENT_WINDOW[1]}] BY LAGGED NEWS SENTIMENT ({PRIMARY_WINDOW}d)")
    print("=" * 80)

    header = (f"  {'Regime':<25s} {'Mean CAR%':>10s} {'Median':>10s} "
              f"{'Std Dev':>10s} {'t-stat':>10s} {'p-value':>10s} {'N':>6s}  Sig")
    print(header)
    print("  " + "-" * 91)

    for regime_label, regime_name in [
        ('pooled', 'Pooled (w/ news data)'),
        ('negative_news', 'Negative news sent.'),
        ('non_negative_news', 'Non-negative news sent.'),
    ]:
        cs = car_by_news.get(regime_label, {})
        if not cs or cs.get('n', 0) < 2:
            print(f"  {regime_name:<25s}     insufficient data")
            continue

        p = cs['p_value']
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.10 else ""
        print(f"  {regime_name:<25s} {cs['mean_car']:>10.4f} "
              f"{cs.get('median_car', np.nan):>10.4f} "
              f"{cs.get('std_car', np.nan):>10.4f} "
              f"{cs['t_stat']:>10.3f} {p:>10.4f} {cs['n']:>6d}  {sig}")

    print()

    # --- Table 5: 2×2 CAR: lagged sentiment × VIX ---
    print("=" * 80)
    print(f"  TABLE 5: 2x2 CAR[{EVENT_WINDOW[0]}, +{EVENT_WINDOW[1]}]: NEWS SENTIMENT x VIX ({PRIMARY_WINDOW}d)")
    print("=" * 80)

    print(f"  {'':>25s} {'Low Vol':>20s} {'High Vol':>20s}")
    print(f"  {'':>25s} {'(VIX < ' + str(VIX_THRESHOLD) + ')':>20s} {'(VIX >= ' + str(VIX_THRESHOLD) + ')':>20s}")
    print("  " + "-" * 65)

    for sent_label, sent_name in [
        ('negative_news', 'Negative news sent.'),
        ('non_negative_news', 'Non-neg. news sent.'),
    ]:
        row_str = f"  {sent_name:<25s}"
        for vol_label in ['low_vol', 'high_vol']:
            key = f"{sent_label}_{vol_label}"
            cs = car_2x2.get(key, {})
            n = cs.get('n', 0)
            if n >= 2:
                mean_car = cs['mean_car']
                p = cs['p_value']
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.10 else ""
                row_str += f" {mean_car:>8.4f}{sig:<2s} (N={n:>3d})"
            else:
                row_str += f" {'N/A':>10s} (N={n:>3d})"
        print(row_str)

    print()

    # --- Table 6: Cross-sectional OLS comparison ---
    print("=" * 80)
    print(f"  TABLE 6: CROSS-SECTIONAL OLS COMPARISON — CAR[{EVENT_WINDOW[0]}, +{EVENT_WINDOW[1]}]")
    print("=" * 80)

    model_labels = {
        'model_a': 'Model A (news sent.)',
        'model_b': 'Model B (horse race)',
        'model_c': 'Model C (no mkt cap)',
    }

    var_descriptions = {
        'const': 'Intercept',
        f'news_sent_{PRIMARY_WINDOW}d': f'News sentiment ({PRIMARY_WINDOW}d)',
        'high_vol': f'High volatility (VIX>={VIX_THRESHOLD})',
        'log_affected': 'log10(affected)',
        'log_market_cap': 'log10(market cap)',
        'news_sent_x_highvol': 'News sent. x High Vol',
        'disclosure_sentiment': 'Disclosure sentiment',
        'disc_sent_x_highvol': 'Disc. sent. x High Vol',
    }

    # Collect all variables across models
    all_vars = []
    for mk in ['model_a', 'model_b', 'model_c']:
        m = ols_models.get(mk)
        if m is not None:
            for v in m.params.index:
                if v not in all_vars:
                    all_vars.append(v)

    if ols_models:
        # Header
        model_keys = [k for k in ['model_a', 'model_b', 'model_c'] if k in ols_models]
        header_parts = [f"  {'Variable':<25s}"]
        for mk in model_keys:
            header_parts.append(f"{model_labels.get(mk, mk):>22s}")
        print("".join(header_parts))
        print("  " + "-" * (25 + 22 * len(model_keys)))

        for var in all_vars:
            display = var_descriptions.get(var, var)
            parts = [f"  {display:<25s}"]
            for mk in model_keys:
                m = ols_models[mk]
                if var in m.params.index:
                    coef = m.params[var]
                    p = m.pvalues[var]
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.10 else ""
                    se = m.bse[var]
                    parts.append(f"  {coef:>8.4f}{sig:<2s} ({se:.4f})")
                else:
                    parts.append(f"{'':>22s}")
            print("".join(parts))

        print("  " + "-" * (25 + 22 * len(model_keys)))

        # Summary row
        parts = [f"  {'N':<25s}"]
        for mk in model_keys:
            parts.append(f"  {int(ols_models[mk].nobs):>18d}  ")
        print("".join(parts))
        parts = [f"  {'R²':<25s}"]
        for mk in model_keys:
            parts.append(f"  {ols_models[mk].rsquared:>18.4f}  ")
        print("".join(parts))
        parts = [f"  {'Adj. R²':<25s}"]
        for mk in model_keys:
            parts.append(f"  {ols_models[mk].rsquared_adj:>18.4f}  ")
        print("".join(parts))
        parts = [f"  {'F-statistic':<25s}"]
        for mk in model_keys:
            parts.append(f"  {ols_models[mk].fvalue:>18.3f}  ")
        print("".join(parts))

        print()
        print(f"  Significance: *** p<0.001, ** p<0.01, * p<0.05, . p<0.10")
        print(f"  Standard errors in parentheses")
    else:
        print("  OLS models not computed (insufficient observations)")

    print()

    # --- Table 7: Robustness OLS (7d, 60d) ---
    print("=" * 80)
    print(f"  TABLE 7: ROBUSTNESS — ALTERNATIVE PRE-BREACH WINDOWS")
    print("=" * 80)

    if robustness:
        for w_label, m in sorted(robustness.items()):
            print(f"\n  Window: {w_label}")
            print(f"  N = {int(m.nobs):,}  |  R² = {m.rsquared:.4f}  |  Adj. R² = {m.rsquared_adj:.4f}")
            print(f"  F-statistic: {m.fvalue:.3f}  (p = {m.f_pvalue:.4e})")
            print()

            rob_var_desc = {
                'const': 'Intercept',
                'high_vol': f'High volatility (VIX>={VIX_THRESHOLD})',
                'log_affected': 'log10(affected)',
                'log_market_cap': 'log10(market cap)',
            }
            # Add window-specific variable names
            for w in LAGGED_WINDOWS:
                rob_var_desc[f'news_sent_{w}d'] = f'News sentiment ({w}d)'
                rob_var_desc[f'news_sent_x_highvol_w'] = f'News sent. x High Vol'

            header_str = f"  {'Variable':<25s} {'Coef':>10s} {'Std Err':>10s} {'t':>8s} {'P>|t|':>10s}  Sig"
            print(header_str)
            print("  " + "-" * 71)

            for var in m.params.index:
                coef = m.params[var]
                se = m.bse[var]
                t = m.tvalues[var]
                p = m.pvalues[var]
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.10 else ""
                display = rob_var_desc.get(var, var)
                print(f"  {display:<25s} {coef:>10.4f} {se:>10.4f} {t:>8.3f} {p:>10.4f}  {sig}")
    else:
        print("  Robustness models not computed (insufficient observations)")

    print()
    print("=" * 80)


# =============================================================================
# STEP 8: REPEAT OFFENDER ANALYSIS
# =============================================================================


def run_repeat_offender_analysis(
    df: pd.DataFrame,
    event_results: Dict[str, Any],
    lagged_results: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Analyse whether firms with prior breach history face different market
    penalties than first-time offenders.

    Args:
        df: Enriched breach DataFrame
        event_results: Output from run_event_study()
        lagged_results: Output from run_lagged_sentiment_analysis()

    Returns dict with:
        - 'history_stats': breach sequence distribution summary
        - 'ar_by_day_repeat': AR by event day for first vs repeat
        - 'car_by_repeat': CAR summary by first/repeat regime
        - 'car_2x2_repeat_vix': 2x2 CAR (first/repeat x VIX)
        - 'ols_models': dict of OLS Models A, B, C
    """
    import statsmodels.api as sm
    from scipy import stats as scipy_stats

    logger.info("Step 8: Computing repeat offender features")

    # --- 1. Compute breach sequence features ---
    df_hist = df[['yf_ticker', 'reported_date']].copy()
    df_hist = df_hist.dropna(subset=['yf_ticker', 'reported_date'])
    df_hist['reported_date_norm'] = pd.to_datetime(df_hist['reported_date']).dt.normalize()

    # Deduplicate on (ticker, date) to avoid same-day duplicates
    df_hist = df_hist.drop_duplicates(subset=['yf_ticker', 'reported_date_norm'])
    df_hist = df_hist.sort_values(['yf_ticker', 'reported_date_norm']).reset_index(drop=True)

    # Compute ordinal position within each firm
    df_hist['breach_sequence'] = df_hist.groupby('yf_ticker').cumcount() + 1
    df_hist['is_repeat'] = (df_hist['breach_sequence'] > 1).astype(int)
    df_hist['prior_breach_count'] = df_hist['breach_sequence'] - 1

    # Days since last breach
    df_hist['prev_date'] = df_hist.groupby('yf_ticker')['reported_date_norm'].shift(1)
    df_hist['days_since_last'] = (
        df_hist['reported_date_norm'] - df_hist['prev_date']
    ).dt.days

    # Build lookup: (ticker, reported_date_norm) -> history features
    history_lookup = {}
    for _, row in df_hist.iterrows():
        key = (row['yf_ticker'], row['reported_date_norm'])
        history_lookup[key] = {
            'breach_sequence': int(row['breach_sequence']),
            'is_repeat': int(row['is_repeat']),
            'prior_breach_count': int(row['prior_breach_count']),
            'days_since_last': row['days_since_last'],
        }

    # Summary statistics
    n_first = int((df_hist['is_repeat'] == 0).sum())
    n_repeat = int((df_hist['is_repeat'] == 1).sum())
    n_total = len(df_hist)
    max_seq = int(df_hist['breach_sequence'].max())
    seq_dist = df_hist['breach_sequence'].value_counts().sort_index().to_dict()

    # Per-firm breach count distribution
    firm_counts = df_hist.groupby('yf_ticker').size()
    firm_count_dist = firm_counts.value_counts().sort_index().to_dict()

    history_stats = {
        'n_total': n_total,
        'n_first': n_first,
        'n_repeat': n_repeat,
        'pct_repeat': 100 * n_repeat / n_total if n_total > 0 else 0,
        'max_sequence': max_seq,
        'sequence_dist': seq_dist,
        'n_unique_firms': int(df_hist['yf_ticker'].nunique()),
        'firm_count_dist': firm_count_dist,
        'mean_days_since_last': float(df_hist['days_since_last'].mean()) if df_hist['days_since_last'].notna().any() else np.nan,
        'median_days_since_last': float(df_hist['days_since_last'].median()) if df_hist['days_since_last'].notna().any() else np.nan,
    }

    # --- 2. Merge with event AR data ---
    event_ar = event_results.get('event_ar', pd.DataFrame())
    if event_ar.empty:
        logger.warning("No event AR data from Step 5, cannot run Step 8")
        return {
            'history_stats': history_stats,
            'history_df': df_hist,
            'ar_by_day_repeat': {},
            'car_by_repeat': {},
            'car_2x2_repeat_vix': {},
            'ols_models': {},
        }

    event_ar_merged = event_ar.copy()

    # Map history features onto event_ar rows
    hist_cols = ['breach_sequence', 'is_repeat', 'prior_breach_count', 'days_since_last']
    col_data = {c: [] for c in hist_cols}
    for _, ar_row in event_ar_merged.iterrows():
        key = (ar_row['ticker'], pd.Timestamp(ar_row['reported_date']).normalize())
        info = history_lookup.get(key, {})
        for c in hist_cols:
            col_data[c].append(info.get(c, np.nan))

    for c in hist_cols:
        event_ar_merged[c] = col_data[c]

    # Filter to events with history data
    has_hist = event_ar_merged['is_repeat'].notna()
    event_ar_hist = event_ar_merged[has_hist].copy()
    logger.info("Step 8: %d/%d AR observations matched to breach history",
                len(event_ar_hist), len(event_ar_merged))

    # --- 3. AR by event day × first/repeat regime ---
    def _ar_day_stats(sub):
        rows = []
        for day in sorted(sub['event_day'].unique()):
            day_ar = sub.loc[sub['event_day'] == day, 'ar']
            if len(day_ar) < 2:
                continue
            t, p = scipy_stats.ttest_1samp(day_ar, 0)
            rows.append({
                'event_day': int(day),
                'mean_ar': day_ar.mean(),
                't_stat': t,
                'p_value': p,
                'n': len(day_ar),
            })
        return pd.DataFrame(rows)

    ar_by_day_repeat = {
        'first': _ar_day_stats(event_ar_hist[event_ar_hist['is_repeat'] == 0]),
        'repeat': _ar_day_stats(event_ar_hist[event_ar_hist['is_repeat'] == 1]),
    }

    # --- 4. CAR by first/repeat regime ---
    car_events = event_ar_hist.groupby('event_idx').agg(
        car=('ar', 'sum'),
        n_days=('ar', 'count'),
        high_vol=('high_vol', 'first'),
        is_repeat=('is_repeat', 'first'),
        prior_breach_count=('prior_breach_count', 'first'),
        ticker=('ticker', 'first'),
        reported_date=('reported_date', 'first'),
    ).reset_index()
    car_valid = car_events[car_events['n_days'] >= 5]

    def _car_stats(sub):
        if len(sub) < 2:
            return {'mean_car': np.nan, 't_stat': np.nan, 'p_value': np.nan, 'n': len(sub)}
        t, p = scipy_stats.ttest_1samp(sub['car'], 0)
        return {
            'mean_car': sub['car'].mean(),
            'median_car': sub['car'].median(),
            'std_car': sub['car'].std(),
            't_stat': t,
            'p_value': p,
            'n': len(sub),
        }

    car_by_repeat = {
        'pooled': _car_stats(car_valid),
        'first': _car_stats(car_valid[car_valid['is_repeat'] == 0]),
        'repeat': _car_stats(car_valid[car_valid['is_repeat'] == 1]),
    }

    # --- 5. 2×2 CAR: first/repeat × VIX ---
    car_2x2_repeat_vix = {}
    for rep_label, rep_val in [('first', 0), ('repeat', 1)]:
        for vol_label, vol_val in [('high_vol', 1), ('low_vol', 0)]:
            sub = car_valid[(car_valid['is_repeat'] == rep_val)
                            & (car_valid['high_vol'] == vol_val)]
            key = f"{rep_label}_{vol_label}"
            car_2x2_repeat_vix[key] = _car_stats(sub)

    # --- 6. Cross-sectional OLS ---
    ols_models = {}
    car_ols = car_valid.copy()

    # Merge controls from df
    control_lookup = {}
    for i, row in df.iterrows():
        ticker = row.get('yf_ticker')
        rd = row.get('reported_date')
        if pd.notna(ticker) and pd.notna(rd):
            rd_ts = pd.Timestamp(rd).normalize()
            key = (ticker, rd_ts)
            ta = row.get('total_affected')
            mc = row.get('yf_market_cap')
            control_lookup[key] = {
                'log_affected': np.log10(ta + 1) if pd.notna(ta) and ta > 0 else np.nan,
                'log_market_cap': np.log10(mc) if pd.notna(mc) and mc > 0 else np.nan,
            }

    log_aff = []
    log_mc = []
    for _, row in car_ols.iterrows():
        key = (row['ticker'], pd.Timestamp(row['reported_date']).normalize())
        ctrls = control_lookup.get(key, {})
        log_aff.append(ctrls.get('log_affected', np.nan))
        log_mc.append(ctrls.get('log_market_cap', np.nan))

    car_ols['log_affected'] = log_aff
    car_ols['log_market_cap'] = log_mc
    car_ols['is_repeat_x_highvol'] = car_ols['is_repeat'] * car_ols['high_vol']

    # Model A (binary): CAR ~ is_repeat + high_vol + log_affected + log_market_cap + is_repeat×high_vol
    model_a_vars = ['is_repeat', 'high_vol', 'log_affected', 'log_market_cap', 'is_repeat_x_highvol']
    ols_a = car_ols.dropna(subset=['car'] + model_a_vars)
    if len(ols_a) >= len(model_a_vars) + 5:
        y = ols_a['car']
        X = sm.add_constant(ols_a[model_a_vars])
        ols_models['model_a'] = sm.OLS(y, X).fit()
        logger.info("Step 8 Model A: N=%d, R²=%.4f",
                     int(ols_models['model_a'].nobs), ols_models['model_a'].rsquared)

    # Model B (count): CAR ~ prior_breach_count + high_vol + log_affected + log_market_cap + prior_count×high_vol
    car_ols['prior_count_x_highvol'] = car_ols['prior_breach_count'] * car_ols['high_vol']
    model_b_vars = ['prior_breach_count', 'high_vol', 'log_affected', 'log_market_cap', 'prior_count_x_highvol']
    ols_b = car_ols.dropna(subset=['car'] + model_b_vars)
    if len(ols_b) >= len(model_b_vars) + 5:
        y = ols_b['car']
        X = sm.add_constant(ols_b[model_b_vars])
        ols_models['model_b'] = sm.OLS(y, X).fit()
        logger.info("Step 8 Model B: N=%d, R²=%.4f",
                     int(ols_models['model_b'].nobs), ols_models['model_b'].rsquared)

    # Model C (horse race with Step 7): Model A + news_sent_30d + news_sent×high_vol
    # Get lagged news sentiment from lagged_results
    lagged_df = lagged_results.get('lagged_df', pd.DataFrame())
    if not lagged_df.empty and len(lagged_df) == len(df):
        news_sent_lookup = {}
        for i, row in df.iterrows():
            ticker = row.get('yf_ticker')
            rd = row.get('reported_date')
            if pd.notna(ticker) and pd.notna(rd):
                rd_ts = pd.Timestamp(rd).normalize()
                key = (ticker, rd_ts)
                if key not in news_sent_lookup:
                    news_val = lagged_df.iloc[i].get(f'news_sent_{PRIMARY_WINDOW}d')
                    news_count = lagged_df.iloc[i].get(f'news_count_{PRIMARY_WINDOW}d', 0)
                    news_sent_lookup[key] = news_val if pd.notna(news_count) and news_count > 0 else np.nan

        news_sent_vals = []
        for _, row in car_ols.iterrows():
            key = (row['ticker'], pd.Timestamp(row['reported_date']).normalize())
            news_sent_vals.append(news_sent_lookup.get(key, np.nan))

        car_ols[f'news_sent_{PRIMARY_WINDOW}d'] = news_sent_vals
        car_ols['news_sent_x_highvol'] = car_ols[f'news_sent_{PRIMARY_WINDOW}d'] * car_ols['high_vol']

        model_c_vars = model_a_vars + [f'news_sent_{PRIMARY_WINDOW}d', 'news_sent_x_highvol']
        ols_c = car_ols.dropna(subset=['car'] + model_c_vars)
        if len(ols_c) >= len(model_c_vars) + 5:
            y = ols_c['car']
            X = sm.add_constant(ols_c[model_c_vars])
            ols_models['model_c'] = sm.OLS(y, X).fit()
            logger.info("Step 8 Model C: N=%d, R²=%.4f",
                         int(ols_models['model_c'].nobs), ols_models['model_c'].rsquared)

    return {
        'history_stats': history_stats,
        'history_df': df_hist,
        'ar_by_day_repeat': ar_by_day_repeat,
        'car_by_repeat': car_by_repeat,
        'car_2x2_repeat_vix': car_2x2_repeat_vix,
        'ols_models': ols_models,
    }


def print_repeat_offender_results(results: Dict[str, Any]):
    """Print Step 8 repeat offender analysis results."""

    history = results['history_stats']
    ar_by_day = results['ar_by_day_repeat']
    car_by_repeat = results['car_by_repeat']
    car_2x2 = results['car_2x2_repeat_vix']
    ols_models = results['ols_models']

    print("=" * 80)
    print("STEP 8: REPEAT OFFENDER ANALYSIS")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Question: Do firms with prior breaches face different market penalties?")
    print()

    # --- Table 1: Breach history summary ---
    print("=" * 80)
    print("  TABLE 1: BREACH HISTORY SUMMARY")
    print("=" * 80)
    print(f"  Total breach events (with ticker):  {history['n_total']:,}")
    print(f"  Unique firms:                       {history['n_unique_firms']:,}")
    print(f"  First-time breaches:                {history['n_first']:,} ({100 - history['pct_repeat']:.1f}%)")
    print(f"  Repeat breaches:                    {history['n_repeat']:,} ({history['pct_repeat']:.1f}%)")
    print(f"  Max breach sequence:                {history['max_sequence']}")
    if not np.isnan(history.get('mean_days_since_last', np.nan)):
        print(f"  Mean days since last (repeats):     {history['mean_days_since_last']:.0f}")
        print(f"  Median days since last (repeats):   {history['median_days_since_last']:.0f}")
    print()

    print(f"  Breach sequence distribution:")
    print(f"  {'Sequence':>10s} {'Count':>8s} {'%':>8s}")
    print("  " + "-" * 28)
    for seq in sorted(history['sequence_dist'].keys()):
        cnt = history['sequence_dist'][seq]
        pct = 100 * cnt / history['n_total']
        print(f"  {seq:>10d} {cnt:>8d} {pct:>7.1f}%")
    print()

    print(f"  Breaches per firm distribution:")
    print(f"  {'# Breaches':>12s} {'# Firms':>10s} {'%':>8s}")
    print("  " + "-" * 32)
    for bc in sorted(history['firm_count_dist'].keys()):
        cnt = history['firm_count_dist'][bc]
        pct = 100 * cnt / history['n_unique_firms']
        print(f"  {bc:>12d} {cnt:>10d} {pct:>7.1f}%")
    print()

    # --- Table 2: AR by event day × first/repeat ---
    print("=" * 80)
    print(f"  TABLE 2: AR BY EVENT DAY x FIRST / REPEAT BREACH")
    print("=" * 80)

    for regime_label, regime_name in [
        ('first', 'FIRST BREACH (sequence = 1)'),
        ('repeat', 'REPEAT BREACH (sequence > 1)'),
    ]:
        regime_df = ar_by_day.get(regime_label, pd.DataFrame())
        if regime_df.empty:
            print(f"\n  {regime_name}: No data")
            continue

        print(f"\n  {regime_name}")
        print(f"  {'Day':>5s} {'Mean AR%':>10s} {'t-stat':>10s} {'p-value':>10s} {'N':>8s}  Sig")
        print("  " + "-" * 53)

        for _, row in regime_df.iterrows():
            p = row['p_value']
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.10 else ""
            print(f"  {int(row['event_day']):>5d} {row['mean_ar']:>10.4f} "
                  f"{row['t_stat']:>10.3f} {p:>10.4f} {int(row['n']):>8d}  {sig}")

    print()

    # --- Table 3: CAR by first/repeat regime ---
    print("=" * 80)
    print(f"  TABLE 3: CAR[{EVENT_WINDOW[0]}, +{EVENT_WINDOW[1]}] BY FIRST / REPEAT BREACH")
    print("=" * 80)

    header = (f"  {'Regime':<25s} {'Mean CAR%':>10s} {'Median':>10s} "
              f"{'Std Dev':>10s} {'t-stat':>10s} {'p-value':>10s} {'N':>6s}  Sig")
    print(header)
    print("  " + "-" * 91)

    for regime_label, regime_name in [
        ('pooled', 'Pooled'),
        ('first', 'First breach'),
        ('repeat', 'Repeat breach'),
    ]:
        cs = car_by_repeat.get(regime_label, {})
        if not cs or cs.get('n', 0) < 2:
            print(f"  {regime_name:<25s}     insufficient data")
            continue

        p = cs['p_value']
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.10 else ""
        print(f"  {regime_name:<25s} {cs['mean_car']:>10.4f} "
              f"{cs.get('median_car', np.nan):>10.4f} "
              f"{cs.get('std_car', np.nan):>10.4f} "
              f"{cs['t_stat']:>10.3f} {p:>10.4f} {cs['n']:>6d}  {sig}")

    print()

    # --- Table 4: 2×2 CAR: first/repeat × VIX ---
    print("=" * 80)
    print(f"  TABLE 4: 2x2 CAR[{EVENT_WINDOW[0]}, +{EVENT_WINDOW[1]}]: FIRST/REPEAT x VIX")
    print("=" * 80)

    print(f"  {'':>25s} {'Low Vol':>20s} {'High Vol':>20s}")
    print(f"  {'':>25s} {'(VIX < ' + str(VIX_THRESHOLD) + ')':>20s} {'(VIX >= ' + str(VIX_THRESHOLD) + ')':>20s}")
    print("  " + "-" * 65)

    for rep_label, rep_name in [
        ('first', 'First breach'),
        ('repeat', 'Repeat breach'),
    ]:
        row_str = f"  {rep_name:<25s}"
        for vol_label in ['low_vol', 'high_vol']:
            key = f"{rep_label}_{vol_label}"
            cs = car_2x2.get(key, {})
            n = cs.get('n', 0)
            if n >= 2:
                mean_car = cs['mean_car']
                p = cs['p_value']
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.10 else ""
                row_str += f" {mean_car:>8.4f}{sig:<2s} (N={n:>3d})"
            else:
                row_str += f" {'N/A':>10s} (N={n:>3d})"
        print(row_str)

    print()

    # --- Table 5: OLS comparison (Models A, B, C) ---
    print("=" * 80)
    print(f"  TABLE 5: CROSS-SECTIONAL OLS — CAR[{EVENT_WINDOW[0]}, +{EVENT_WINDOW[1]}]")
    print("=" * 80)

    model_labels = {
        'model_a': 'Model A (binary)',
        'model_b': 'Model B (count)',
        'model_c': 'Model C (+ news)',
    }

    var_descriptions = {
        'const': 'Intercept',
        'is_repeat': 'Repeat offender (0/1)',
        'high_vol': f'High volatility (VIX>={VIX_THRESHOLD})',
        'log_affected': 'log10(affected)',
        'log_market_cap': 'log10(market cap)',
        'is_repeat_x_highvol': 'Repeat x High Vol',
        'prior_breach_count': 'Prior breach count',
        'prior_count_x_highvol': 'Prior count x High Vol',
        f'news_sent_{PRIMARY_WINDOW}d': f'News sentiment ({PRIMARY_WINDOW}d)',
        'news_sent_x_highvol': 'News sent. x High Vol',
    }

    # Collect all variables across models
    all_vars = []
    for mk in ['model_a', 'model_b', 'model_c']:
        m = ols_models.get(mk)
        if m is not None:
            for v in m.params.index:
                if v not in all_vars:
                    all_vars.append(v)

    if ols_models:
        model_keys = [k for k in ['model_a', 'model_b', 'model_c'] if k in ols_models]
        header_parts = [f"  {'Variable':<25s}"]
        for mk in model_keys:
            header_parts.append(f"{model_labels.get(mk, mk):>22s}")
        print("".join(header_parts))
        print("  " + "-" * (25 + 22 * len(model_keys)))

        for var in all_vars:
            display = var_descriptions.get(var, var)
            parts = [f"  {display:<25s}"]
            for mk in model_keys:
                m = ols_models[mk]
                if var in m.params.index:
                    coef = m.params[var]
                    p = m.pvalues[var]
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.10 else ""
                    se = m.bse[var]
                    parts.append(f"  {coef:>8.4f}{sig:<2s} ({se:.4f})")
                else:
                    parts.append(f"{'':>22s}")
            print("".join(parts))

        print("  " + "-" * (25 + 22 * len(model_keys)))

        # Summary rows
        parts = [f"  {'N':<25s}"]
        for mk in model_keys:
            parts.append(f"  {int(ols_models[mk].nobs):>18d}  ")
        print("".join(parts))
        parts = [f"  {'R²':<25s}"]
        for mk in model_keys:
            parts.append(f"  {ols_models[mk].rsquared:>18.4f}  ")
        print("".join(parts))
        parts = [f"  {'Adj. R²':<25s}"]
        for mk in model_keys:
            parts.append(f"  {ols_models[mk].rsquared_adj:>18.4f}  ")
        print("".join(parts))
        parts = [f"  {'F-statistic':<25s}"]
        for mk in model_keys:
            parts.append(f"  {ols_models[mk].fvalue:>18.3f}  ")
        print("".join(parts))

        print()
        print(f"  Significance: *** p<0.001, ** p<0.01, * p<0.05, . p<0.10")
        print(f"  Standard errors in parentheses")
    else:
        print("  OLS models not computed (insufficient observations)")

    print()
    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # === STEP 1: Descriptive Statistics of cleaned data ===
    print("\nLoading enriched data from clean.py output...")
    df = load_enriched_data()
    print(f"Loaded {len(df):,} records with {len(df.columns)} columns.\n")

    stats = print_full_descriptive_statistics(df)

    # === STEP 2: OLS Fama-French by VIX regime ===
    print()
    ols_results = run_ols_fama_french()
    print_ols_results(ols_results)

    # === STEP 3: FF + Macro controls by VIX regime ===
    print()
    macro_results = run_ols_with_macro_controls()
    print_macro_control_results(macro_results)

    # === STEP 4: FF + Stock price controls (breach-level) ===
    print()
    breach_results, breach_data = run_ols_breach_level(df)
    print_breach_level_results(breach_results, breach_data)

    # === STEP 5: Event study — breach announcements ===
    print()
    event_results = run_event_study(df)
    print_event_study_results(event_results)

    # === STEP 6: Sentiment-augmented event study ===
    print()
    sentiment_results = run_sentiment_analysis(df, event_results)
    print_sentiment_analysis_results(sentiment_results)

    # === STEP 7: Lagged firm-level news sentiment ===
    print()
    lagged_results = run_lagged_sentiment_analysis(df, event_results, sentiment_results)
    print_lagged_sentiment_results(lagged_results)

    # === STEP 8: Repeat offender analysis ===
    print()
    repeat_results = run_repeat_offender_analysis(df, event_results, lagged_results)
    print_repeat_offender_results(repeat_results)

    # === Store all results in database ===
    from database import store_all_results
    store_all_results({
        'step1': stats,
        'step2': ols_results,
        'step3': macro_results,
        'step4': breach_results,
        'step5': event_results,
        'step6': sentiment_results,
        'step7': lagged_results,
        'step8': repeat_results,
    })
