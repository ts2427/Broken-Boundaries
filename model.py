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

from database import (
    query_df,
    get_summary_stats,
    get_sector_breakdown,
    get_yearly_breakdown,
    DB_PATH,
)


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
            yf_employee_count,
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
    df['log_employees'] = np.log10(df['yf_employee_count'] + 1)

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
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(generate_summary_report())

    print("\n" + "=" * 60)
    print("TREND ANALYSIS")
    print("=" * 60)
    trends = analyze_trends()
    print(trends.to_string(index=False))

    print("\n" + "=" * 60)
    print("HIGH RISK BREACHES (Score >= 70)")
    print("=" * 60)
    high_risk = get_high_risk_breaches(70.0)
    print(f"Found {len(high_risk)} high-risk breaches")
    print(high_risk[['org_name', 'total_affected', 'risk_score']].head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("CORRELATION MATRIX")
    print("=" * 60)
    corr = correlation_analysis()
    print(corr.round(2).to_string())
