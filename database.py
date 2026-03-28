"""
database.py - SQL Database Connection Module
Provides database connection, configuration, and query utilities for the breach data pipeline.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, List, Dict, Any, Union


# === Configuration ===
DB_DIR = Path(__file__).parent / "database"
DB_NAME = "breach_data.db"
DB_PATH = DB_DIR / DB_NAME


# =============================================================================
# CONNECTION MANAGEMENT
# =============================================================================

def get_db_path() -> Path:
    """Get the path to the database file."""
    return DB_PATH


def ensure_db_directory():
    """Ensure the database directory exists."""
    DB_DIR.mkdir(exist_ok=True)


@contextmanager
def get_connection(db_path: Optional[Path] = None):
    """
    Context manager for database connections.
    Ensures connections are properly closed after use.

    Usage:
        with get_connection() as conn:
            cursor = conn.execute("SELECT * FROM breach_incidents")
            results = cursor.fetchall()
    """
    if db_path is None:
        db_path = DB_PATH

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Enable column access by name

    # Enable foreign keys and WAL mode for better performance
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")

    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


# =============================================================================
# QUERY UTILITIES
# =============================================================================

def query(sql: str, params: tuple = (), db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Execute a query and return results as a list of dictionaries.

    Args:
        sql: SQL query to execute
        params: Query parameters (for parameterized queries)
        db_path: Optional path to database

    Returns:
        List of dictionaries, one per row
    """
    with get_connection(db_path) as conn:
        cursor = conn.execute(sql, params)
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]


def query_df(sql: str, params: tuple = (), db_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Execute a query and return results as a pandas DataFrame.

    Args:
        sql: SQL query to execute
        params: Query parameters
        db_path: Optional path to database

    Returns:
        Query results as pandas DataFrame
    """
    if db_path is None:
        db_path = DB_PATH

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(sql, conn, params=params)
        return df
    finally:
        conn.close()


def execute(sql: str, params: tuple = (), db_path: Optional[Path] = None) -> int:
    """
    Execute a SQL statement (INSERT, UPDATE, DELETE).

    Args:
        sql: SQL statement to execute
        params: Query parameters
        db_path: Optional path to database

    Returns:
        Number of rows affected
    """
    with get_connection(db_path) as conn:
        cursor = conn.execute(sql, params)
        return cursor.rowcount


def execute_many(sql: str, params_list: List[tuple], db_path: Optional[Path] = None) -> int:
    """
    Execute a SQL statement with multiple parameter sets.

    Args:
        sql: SQL statement to execute
        params_list: List of parameter tuples
        db_path: Optional path to database

    Returns:
        Number of rows affected
    """
    with get_connection(db_path) as conn:
        cursor = conn.executemany(sql, params_list)
        return cursor.rowcount


# =============================================================================
# SCHEMA UTILITIES
# =============================================================================

def get_tables(db_path: Optional[Path] = None) -> List[str]:
    """Get list of all tables in the database."""
    sql = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    results = query(sql, db_path=db_path)
    return [row['name'] for row in results]


def get_views(db_path: Optional[Path] = None) -> List[str]:
    """Get list of all views in the database."""
    sql = "SELECT name FROM sqlite_master WHERE type='view' ORDER BY name"
    results = query(sql, db_path=db_path)
    return [row['name'] for row in results]


def get_table_info(table_name: str, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Get column information for a table."""
    sql = f"PRAGMA table_info({table_name})"
    return query(sql, db_path=db_path)


def get_row_count(table_name: str, db_path: Optional[Path] = None) -> int:
    """Get the number of rows in a table."""
    sql = f"SELECT COUNT(*) as count FROM {table_name}"
    result = query(sql, db_path=db_path)
    return result[0]['count'] if result else 0


def table_exists(table_name: str, db_path: Optional[Path] = None) -> bool:
    """Check if a table exists in the database."""
    sql = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
    result = query(sql, (table_name,), db_path)
    return len(result) > 0


# =============================================================================
# BREACH DATA QUERIES
# =============================================================================

def get_breach_by_id(breach_id: int, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Get a single breach record by ID."""
    sql = "SELECT * FROM breach_incidents WHERE id = ?"
    results = query(sql, (breach_id,), db_path)
    return results[0] if results else None


def get_breaches_by_org(org_name: str, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Get all breaches for an organization."""
    sql = "SELECT * FROM breach_incidents WHERE org_name LIKE ? ORDER BY breach_date DESC"
    return query(sql, (f"%{org_name}%",), db_path)


def get_breaches_by_ticker(ticker: str, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Get all breaches for a stock ticker."""
    sql = "SELECT * FROM breach_incidents WHERE stock_ticker = ? ORDER BY breach_date DESC"
    return query(sql, (ticker,), db_path)


def get_breaches_by_year(year: int, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Get all breaches for a specific year."""
    sql = """
        SELECT * FROM breach_incidents
        WHERE strftime('%Y', breach_date) = ?
        ORDER BY total_affected DESC
    """
    return query(sql, (str(year),), db_path)


def get_breaches_by_sector(sector: str, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Get all breaches for a specific sector."""
    sql = "SELECT * FROM breach_incidents WHERE yf_sector = ? ORDER BY total_affected DESC"
    return query(sql, (sector,), db_path)


def get_top_breaches(limit: int = 10, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Get the top breaches by number of affected individuals."""
    sql = """
        SELECT org_name, breach_date, total_affected, breach_type, yf_sector
        FROM breach_incidents
        WHERE total_affected IS NOT NULL
        ORDER BY total_affected DESC
        LIMIT ?
    """
    return query(sql, (limit,), db_path)


def search_breaches(
    org_name: Optional[str] = None,
    ticker: Optional[str] = None,
    sector: Optional[str] = None,
    breach_type: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    min_affected: Optional[int] = None,
    limit: int = 100,
    db_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    Search breaches with multiple filters.

    Args:
        org_name: Filter by organization name (partial match)
        ticker: Filter by stock ticker
        sector: Filter by sector
        breach_type: Filter by breach type
        year_from: Filter breaches from this year
        year_to: Filter breaches up to this year
        min_affected: Minimum number of affected individuals
        limit: Maximum results to return

    Returns:
        List of matching breach records
    """
    conditions = []
    params = []

    if org_name:
        conditions.append("org_name LIKE ?")
        params.append(f"%{org_name}%")

    if ticker:
        conditions.append("stock_ticker = ?")
        params.append(ticker)

    if sector:
        conditions.append("yf_sector = ?")
        params.append(sector)

    if breach_type:
        conditions.append("breach_type = ?")
        params.append(breach_type)

    if year_from:
        conditions.append("strftime('%Y', breach_date) >= ?")
        params.append(str(year_from))

    if year_to:
        conditions.append("strftime('%Y', breach_date) <= ?")
        params.append(str(year_to))

    if min_affected:
        conditions.append("total_affected >= ?")
        params.append(min_affected)

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    sql = f"""
        SELECT * FROM breach_incidents
        WHERE {where_clause}
        ORDER BY total_affected DESC
        LIMIT ?
    """
    params.append(limit)

    return query(sql, tuple(params), db_path)


# =============================================================================
# STATISTICS
# =============================================================================

def get_summary_stats(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Get summary statistics about the breach database."""
    stats = {}

    # Total records
    stats['total_records'] = get_row_count('breach_incidents', db_path)

    # Total affected
    result = query("SELECT SUM(total_affected) as total FROM breach_incidents", db_path=db_path)
    stats['total_affected'] = result[0]['total'] if result else 0

    # Unique organizations
    result = query("SELECT COUNT(DISTINCT org_name) as count FROM breach_incidents", db_path=db_path)
    stats['unique_organizations'] = result[0]['count'] if result else 0

    # Unique tickers
    result = query("SELECT COUNT(DISTINCT stock_ticker) as count FROM breach_incidents WHERE stock_ticker IS NOT NULL", db_path=db_path)
    stats['unique_tickers'] = result[0]['count'] if result else 0

    # Date range
    result = query("SELECT MIN(breach_date) as min_date, MAX(breach_date) as max_date FROM breach_incidents", db_path=db_path)
    if result:
        stats['earliest_breach'] = result[0]['min_date']
        stats['latest_breach'] = result[0]['max_date']

    # Breach types
    result = query("SELECT COUNT(DISTINCT breach_type) as count FROM breach_incidents", db_path=db_path)
    stats['breach_types'] = result[0]['count'] if result else 0

    # Sectors
    result = query("SELECT COUNT(DISTINCT yf_sector) as count FROM breach_incidents WHERE yf_sector IS NOT NULL", db_path=db_path)
    stats['sectors'] = result[0]['count'] if result else 0

    # News coverage
    result = query("SELECT SUM(total_news_count) as total FROM breach_incidents", db_path=db_path)
    stats['total_news_articles'] = result[0]['total'] if result else 0

    return stats


def get_sector_breakdown(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Get breach statistics by sector."""
    return query_df("SELECT * FROM v_breaches_by_sector", db_path=db_path)


def get_yearly_breakdown(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Get breach statistics by year."""
    return query_df("SELECT * FROM v_breaches_by_year", db_path=db_path)


def get_type_breakdown(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Get breach statistics by type."""
    return query_df("SELECT * FROM v_breaches_by_type", db_path=db_path)


# =============================================================================
# DATABASE INFO
# =============================================================================

def print_database_info(db_path: Optional[Path] = None):
    """Print information about the database."""
    if db_path is None:
        db_path = DB_PATH

    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return

    print(f"Database: {db_path}")
    print(f"Size: {db_path.stat().st_size / 1024 / 1024:.2f} MB")
    print()

    tables = get_tables(db_path)
    print(f"Tables ({len(tables)}):")
    for table in tables:
        count = get_row_count(table, db_path)
        print(f"  - {table}: {count:,} rows")

    print()
    views = get_views(db_path)
    print(f"Views ({len(views)}):")
    for view in views:
        print(f"  - {view}")

    print()
    stats = get_summary_stats(db_path)
    print("Summary Statistics:")
    print(f"  Total breaches: {stats['total_records']:,}")
    print(f"  Total affected: {stats['total_affected']:,}")
    print(f"  Unique organizations: {stats['unique_organizations']:,}")
    print(f"  Unique tickers: {stats['unique_tickers']:,}")
    print(f"  Date range: {stats['earliest_breach']} to {stats['latest_breach']}")
    print(f"  Total news articles: {stats['total_news_articles']:,}")


# =============================================================================
# EXTERNAL DATA ACCESSORS
# =============================================================================
# These wrap clean.py fetch/compute functions so model.py imports only from
# database.py, enforcing the clean → etl → database → model data flow.

def get_fama_french_data() -> pd.DataFrame:
    """Fetch daily Fama-French 5-Factor returns (delegates to clean.py)."""
    from clean import fetch_fama_french_data
    return fetch_fama_french_data()


def get_vix_data() -> pd.DataFrame:
    """Fetch daily CBOE VIX data (delegates to clean.py)."""
    from clean import fetch_vix_data
    return fetch_vix_data()


def get_inflation_data() -> pd.DataFrame:
    """Fetch monthly CPI / inflation data (delegates to clean.py)."""
    from clean import fetch_inflation_data
    return fetch_inflation_data()


def get_gdp_data() -> pd.DataFrame:
    """Fetch quarterly real GDP growth data (delegates to clean.py)."""
    from clean import fetch_gdp_data
    return fetch_gdp_data()


def get_unemployment_data() -> pd.DataFrame:
    """Fetch monthly unemployment rate data (delegates to clean.py)."""
    from clean import fetch_unemployment_data
    return fetch_unemployment_data()


def get_interest_rate_data() -> dict:
    """Fetch interest rate data (Fed Funds, yield spread) (delegates to clean.py)."""
    from clean import fetch_interest_rate_data
    return fetch_interest_rate_data()


def get_finbert_sentiment(texts: pd.Series) -> pd.DataFrame:
    """Compute FinBERT sentiment scores for texts (delegates to clean.py)."""
    from clean import compute_finbert_sentiment
    return compute_finbert_sentiment(texts)


def get_lagged_news_sentiment(df: pd.DataFrame, windows=None) -> pd.DataFrame:
    """Compute pre-breach lagged news sentiment (delegates to clean.py)."""
    from clean import compute_lagged_news_sentiment
    return compute_lagged_news_sentiment(df, windows=windows)


# =============================================================================
# OLS HELPER
# =============================================================================

def _extract_ols(model):
    """Extract model summary dict and coefficient rows from a statsmodels OLS result."""
    summary = (
        int(model.nobs), float(model.rsquared), float(model.rsquared_adj),
        float(model.fvalue), float(model.f_pvalue), float(model.aic), float(model.bic),
    )
    conf_int = model.conf_int()
    coefs = []
    for var in model.params.index:
        ci = conf_int.loc[var]
        coefs.append((
            var, float(model.params[var]), float(model.bse[var]),
            float(model.tvalues[var]), float(model.pvalues[var]),
            float(ci[0]), float(ci[1]),
        ))
    return summary, coefs


# =============================================================================
# STEP 1: DESCRIPTIVE STATISTICS
# =============================================================================

def store_descriptive_results(results, db_path=None):
    """Store Step 1 descriptive statistics results in the database."""
    import json
    if db_path is None:
        db_path = DB_PATH

    stored = {}

    with get_connection(db_path) as conn:
        # --- descriptive_overview ---
        conn.execute("DROP TABLE IF EXISTS descriptive_overview")
        conn.execute("""CREATE TABLE descriptive_overview (
            total_records INTEGER, total_columns INTEGER,
            memory_usage_mb REAL, duplicate_rows INTEGER,
            news_reddit INTEGER, news_guardian INTEGER, news_nyt INTEGER,
            news_newsapi INTEGER, news_total INTEGER,
            has_ticker INTEGER, has_yf_data INTEGER
        )""")
        ov = results['overview']
        ns = results.get('news_sources', {})
        sd = results.get('stock_data', {})
        conn.execute(
            "INSERT INTO descriptive_overview VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (ov['total_records'], ov['total_columns'], ov['memory_usage_mb'],
             ov['duplicate_rows'], ns.get('reddit', 0), ns.get('guardian', 0),
             ns.get('nyt', 0), ns.get('newsapi', 0), ns.get('total', 0),
             sd.get('has_ticker', 0), sd.get('has_yf_data', 0)),
        )
        stored['descriptive_overview'] = 1

        # --- descriptive_numeric ---
        conn.execute("DROP TABLE IF EXISTS descriptive_numeric")
        conn.execute("""CREATE TABLE descriptive_numeric (
            column_name TEXT, count INTEGER, missing INTEGER, missing_pct REAL,
            mean REAL, std REAL, min REAL, q25 REAL, median REAL, q75 REAL,
            max REAL, skewness REAL, kurtosis REAL
        )""")
        count = 0
        for col_name, s in results.get('numeric', {}).items():
            conn.execute(
                "INSERT INTO descriptive_numeric VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (col_name, s['count'], s['missing'], s['missing_pct'],
                 s['mean'], s['std'], float(s['min']), s['q25'], s['median'],
                 s['q75'], float(s['max']), s['skewness'], s['kurtosis']),
            )
            count += 1
        stored['descriptive_numeric'] = count

        # --- descriptive_categorical ---
        conn.execute("DROP TABLE IF EXISTS descriptive_categorical")
        conn.execute("""CREATE TABLE descriptive_categorical (
            column_name TEXT, count INTEGER, missing INTEGER, missing_pct REAL,
            unique_count INTEGER, mode TEXT, mode_count INTEGER, top_5_json TEXT
        )""")
        count = 0
        for col_name, s in results.get('categorical', {}).items():
            conn.execute(
                "INSERT INTO descriptive_categorical VALUES (?,?,?,?,?,?,?,?)",
                (col_name, s['count'], s['missing'], s['missing_pct'],
                 s['unique'], str(s['mode']), s['mode_count'],
                 json.dumps(s['top_5'])),
            )
            count += 1
        stored['descriptive_categorical'] = count

        # --- descriptive_date ---
        conn.execute("DROP TABLE IF EXISTS descriptive_date")
        conn.execute("""CREATE TABLE descriptive_date (
            column_name TEXT, count INTEGER, missing INTEGER, missing_pct REAL,
            earliest TEXT, latest TEXT, range_days INTEGER, median_date TEXT
        )""")
        count = 0
        for col_name, s in results.get('date', {}).items():
            conn.execute(
                "INSERT INTO descriptive_date VALUES (?,?,?,?,?,?,?,?)",
                (col_name, s['count'], s['missing'], s['missing_pct'],
                 s['earliest'], s['latest'], s['range_days'], s['median']),
            )
            count += 1
        stored['descriptive_date'] = count

    _print_stored("STEP 1 DESCRIPTIVE STATISTICS", stored)
    return stored


def get_descriptive_results(db_path=None):
    """Retrieve Step 1 descriptive statistics from the database."""
    tables = ['descriptive_overview', 'descriptive_numeric',
              'descriptive_categorical', 'descriptive_date']
    return {t: query_df(f"SELECT * FROM {t}", db_path=db_path) for t in tables}


# =============================================================================
# STEP 2: FAMA-FRENCH OLS
# =============================================================================

def store_fama_french_results(results, db_path=None):
    """Store Step 2 Fama-French OLS results in the database."""
    if db_path is None:
        db_path = DB_PATH
    return _store_simple_ols(results, 'fama_french', db_path,
                             key_cols=('regime',), step_label="STEP 2 FAMA-FRENCH OLS")


def get_fama_french_results(db_path=None):
    """Retrieve Step 2 Fama-French OLS results from the database."""
    return {t: query_df(f"SELECT * FROM {t}", db_path=db_path)
            for t in ['fama_french_ols_models', 'fama_french_ols_coefficients']}


# =============================================================================
# STEP 3: MACRO CONTROLS OLS
# =============================================================================

def store_macro_controls_results(results, db_path=None):
    """Store Step 3 macro controls OLS results in the database."""
    if db_path is None:
        db_path = DB_PATH
    return _store_nested_ols(results, 'macro_controls', db_path,
                             step_label="STEP 3 MACRO CONTROLS OLS")


def get_macro_controls_results(db_path=None):
    """Retrieve Step 3 macro controls OLS results from the database."""
    return {t: query_df(f"SELECT * FROM {t}", db_path=db_path)
            for t in ['macro_controls_ols_models', 'macro_controls_ols_coefficients']}


# =============================================================================
# STEP 4: BREACH-LEVEL OLS
# =============================================================================

def store_breach_level_results(results, db_path=None):
    """Store Step 4 breach-level OLS results in the database."""
    if db_path is None:
        db_path = DB_PATH
    return _store_nested_ols(results, 'breach_level', db_path,
                             step_label="STEP 4 BREACH-LEVEL OLS")


def get_breach_level_results(db_path=None):
    """Retrieve Step 4 breach-level OLS results from the database."""
    return {t: query_df(f"SELECT * FROM {t}", db_path=db_path)
            for t in ['breach_level_ols_models', 'breach_level_ols_coefficients']}


# =============================================================================
# STEP 5: EVENT STUDY
# =============================================================================

def store_event_study_results(results, db_path=None):
    """Store Step 5 event study results in the database."""
    if db_path is None:
        db_path = DB_PATH

    stored = {}

    with get_connection(db_path) as conn:
        # --- event_study_counts ---
        conn.execute("DROP TABLE IF EXISTS event_study_counts")
        conn.execute("""CREATE TABLE event_study_counts (
            total_events INTEGER, usable INTEGER, high_vol INTEGER, low_vol INTEGER,
            tickers_used INTEGER, car_events INTEGER,
            skip_no_prices INTEGER, skip_no_td_match INTEGER,
            skip_short_est INTEGER, skip_ols_fail INTEGER
        )""")
        ec = results.get('event_count', {})
        sk = ec.get('skipped', {})
        conn.execute(
            "INSERT INTO event_study_counts VALUES (?,?,?,?,?,?,?,?,?,?)",
            (ec.get('total_events'), ec.get('usable'), ec.get('high_vol'),
             ec.get('low_vol'), ec.get('tickers_used'), ec.get('car_events'),
             sk.get('no_prices', 0), sk.get('no_td_match', 0),
             sk.get('short_est', 0), sk.get('ols_fail', 0)),
        )
        stored['event_study_counts'] = 1

        # --- event_study_ar_by_day ---
        conn.execute("DROP TABLE IF EXISTS event_study_ar_by_day")
        conn.execute("""CREATE TABLE event_study_ar_by_day (
            regime TEXT, event_day INTEGER, mean_ar REAL,
            t_stat REAL, p_value REAL, n INTEGER
        )""")
        stored['event_study_ar_by_day'] = _insert_ar_by_day(
            conn, 'event_study_ar_by_day', results.get('ar_by_day', {}))

        # --- event_study_car ---
        conn.execute("DROP TABLE IF EXISTS event_study_car")
        conn.execute("""CREATE TABLE event_study_car (
            regime TEXT, mean_car REAL, median_car REAL, std_car REAL,
            t_stat REAL, p_value REAL, n INTEGER
        )""")
        stored['event_study_car'] = _insert_car_stats(
            conn, 'event_study_car', results.get('car_summary', {}))

        # --- event_study_ar (granular) ---
        conn.execute("DROP TABLE IF EXISTS event_study_ar")
        conn.execute("""CREATE TABLE event_study_ar (
            event_idx INTEGER, ticker TEXT, reported_date TEXT,
            event_day INTEGER, date TEXT, ar REAL,
            actual_excess REAL, expected_excess REAL,
            vix REAL, high_vol INTEGER, est_r2 REAL, est_nobs INTEGER
        )""")
        ar_df = results.get('event_ar', pd.DataFrame())
        count = 0
        if not ar_df.empty:
            rows = []
            for _, r in ar_df.iterrows():
                rows.append((
                    int(r['event_idx']), r['ticker'], str(r['reported_date']),
                    int(r['event_day']), str(r['date']), float(r['ar']),
                    float(r['actual_excess']), float(r['expected_excess']),
                    float(r['vix']) if pd.notna(r.get('vix')) else None,
                    int(r['high_vol']),
                    float(r['est_r2']) if pd.notna(r.get('est_r2')) else None,
                    int(r['est_nobs']) if pd.notna(r.get('est_nobs')) else None,
                ))
            conn.executemany(
                "INSERT INTO event_study_ar VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            count = len(rows)
        stored['event_study_ar'] = count

    _print_stored("STEP 5 EVENT STUDY", stored)
    return stored


def get_event_study_results(db_path=None):
    """Retrieve Step 5 event study results from the database."""
    tables = ['event_study_counts', 'event_study_ar_by_day',
              'event_study_car', 'event_study_ar']
    return {t: query_df(f"SELECT * FROM {t}", db_path=db_path) for t in tables}


# =============================================================================
# STEP 6: SENTIMENT ANALYSIS
# =============================================================================

def store_sentiment_results(results, db_path=None):
    """Store Step 6 sentiment analysis results in the database."""
    if db_path is None:
        db_path = DB_PATH

    stored = {}

    with get_connection(db_path) as conn:
        # --- sentiment_stats ---
        conn.execute("DROP TABLE IF EXISTS sentiment_stats")
        conn.execute("""CREATE TABLE sentiment_stats (
            n_total INTEGER, n_with_text INTEGER, n_negative INTEGER,
            n_non_negative INTEGER, mean_score REAL, median_score REAL,
            std_score REAL, label_counts_json TEXT
        )""")
        import json
        ss = results.get('sentiment_stats', {})
        if ss:
            conn.execute(
                "INSERT INTO sentiment_stats VALUES (?,?,?,?,?,?,?,?)",
                (ss.get('n_total'), ss.get('n_with_text'), ss.get('n_negative'),
                 ss.get('n_non_negative'), ss.get('mean_score'), ss.get('median_score'),
                 ss.get('std_score'),
                 json.dumps({str(k): v for k, v in ss.get('label_counts', {}).items()})),
            )
        stored['sentiment_stats'] = 1 if ss else 0

        # --- sentiment_ar_by_day ---
        conn.execute("DROP TABLE IF EXISTS sentiment_ar_by_day")
        conn.execute("""CREATE TABLE sentiment_ar_by_day (
            regime TEXT, event_day INTEGER, mean_ar REAL,
            t_stat REAL, p_value REAL, n INTEGER
        )""")
        stored['sentiment_ar_by_day'] = _insert_ar_by_day(
            conn, 'sentiment_ar_by_day', results.get('ar_by_day_sentiment', {}))

        # --- sentiment_car ---
        conn.execute("DROP TABLE IF EXISTS sentiment_car")
        conn.execute("""CREATE TABLE sentiment_car (
            regime TEXT, mean_car REAL, median_car REAL, std_car REAL,
            t_stat REAL, p_value REAL, n INTEGER
        )""")
        stored['sentiment_car'] = _insert_car_stats(
            conn, 'sentiment_car', results.get('car_by_sentiment', {}))

        # --- sentiment_car_2x2 ---
        conn.execute("DROP TABLE IF EXISTS sentiment_car_2x2")
        conn.execute("""CREATE TABLE sentiment_car_2x2 (
            sentiment_regime TEXT, vol_regime TEXT, mean_car REAL,
            median_car REAL, std_car REAL, t_stat REAL, p_value REAL, n INTEGER
        )""")
        stored['sentiment_car_2x2'] = _insert_2x2_car(
            conn, 'sentiment_car_2x2', results.get('car_2x2', {}),
            regime_col='sentiment_regime')

        # --- sentiment_ols_model + sentiment_ols_coefficients ---
        conn.execute("DROP TABLE IF EXISTS sentiment_ols_model")
        conn.execute("""CREATE TABLE sentiment_ols_model (
            n_obs INTEGER, r_squared REAL, adj_r_squared REAL,
            f_statistic REAL, f_p_value REAL, aic REAL, bic REAL
        )""")
        conn.execute("DROP TABLE IF EXISTS sentiment_ols_coefficients")
        conn.execute("""CREATE TABLE sentiment_ols_coefficients (
            variable TEXT, coefficient REAL, std_err REAL,
            t_stat REAL, p_value REAL, ci_lower REAL, ci_upper REAL
        )""")
        ols = results.get('ols_results')
        model_count = 0
        coef_count = 0
        if ols is not None:
            summary, coefs = _extract_ols(ols)
            conn.execute(
                "INSERT INTO sentiment_ols_model VALUES (?,?,?,?,?,?,?)", summary)
            model_count = 1
            for c in coefs:
                conn.execute(
                    "INSERT INTO sentiment_ols_coefficients VALUES (?,?,?,?,?,?,?)", c)
                coef_count += 1
        stored['sentiment_ols_model'] = model_count
        stored['sentiment_ols_coefficients'] = coef_count

    _print_stored("STEP 6 SENTIMENT ANALYSIS", stored)
    return stored


def get_sentiment_results(db_path=None):
    """Retrieve Step 6 sentiment analysis results from the database."""
    tables = ['sentiment_stats', 'sentiment_ar_by_day', 'sentiment_car',
              'sentiment_car_2x2', 'sentiment_ols_model', 'sentiment_ols_coefficients']
    return {t: query_df(f"SELECT * FROM {t}", db_path=db_path) for t in tables}


# =============================================================================
# STEP 7: LAGGED SENTIMENT
# =============================================================================

def store_lagged_sentiment_results(results, db_path=None):
    """Store Step 7 lagged sentiment analysis results in the database."""
    if db_path is None:
        db_path = DB_PATH

    stored = {}

    with get_connection(db_path) as conn:
        # --- lagged_coverage_stats ---
        conn.execute("DROP TABLE IF EXISTS lagged_coverage_stats")
        conn.execute("""CREATE TABLE lagged_coverage_stats (
            window INTEGER, events_with_articles INTEGER,
            total_events INTEGER, coverage_pct REAL, mean_articles REAL
        )""")
        count = 0
        for w, s in results.get('coverage_stats', {}).items():
            conn.execute(
                "INSERT INTO lagged_coverage_stats VALUES (?,?,?,?,?)",
                (int(w), s['events_with_articles'], s['total_events'],
                 s['coverage_pct'], s['mean_articles']),
            )
            count += 1
        stored['lagged_coverage_stats'] = count

        # --- lagged_sentiment_dist ---
        conn.execute("DROP TABLE IF EXISTS lagged_sentiment_dist")
        conn.execute("""CREATE TABLE lagged_sentiment_dist (
            n_with_data INTEGER, n_total INTEGER, mean REAL, median REAL,
            std REAL, min REAL, max REAL, pct_negative REAL, pct_non_negative REAL
        )""")
        sd = results.get('sentiment_dist', {})
        if sd:
            conn.execute(
                "INSERT INTO lagged_sentiment_dist VALUES (?,?,?,?,?,?,?,?,?)",
                (sd.get('n_with_data'), sd.get('n_total'), sd.get('mean'),
                 sd.get('median'), sd.get('std'), sd.get('min'), sd.get('max'),
                 sd.get('pct_negative'), sd.get('pct_non_negative')),
            )
        stored['lagged_sentiment_dist'] = 1 if sd else 0

        # --- lagged_ar_by_day ---
        conn.execute("DROP TABLE IF EXISTS lagged_ar_by_day")
        conn.execute("""CREATE TABLE lagged_ar_by_day (
            regime TEXT, event_day INTEGER, mean_ar REAL,
            t_stat REAL, p_value REAL, n INTEGER
        )""")
        stored['lagged_ar_by_day'] = _insert_ar_by_day(
            conn, 'lagged_ar_by_day', results.get('ar_by_day_news', {}))

        # --- lagged_car ---
        conn.execute("DROP TABLE IF EXISTS lagged_car")
        conn.execute("""CREATE TABLE lagged_car (
            regime TEXT, mean_car REAL, median_car REAL, std_car REAL,
            t_stat REAL, p_value REAL, n INTEGER
        )""")
        stored['lagged_car'] = _insert_car_stats(
            conn, 'lagged_car', results.get('car_by_news_sent', {}))

        # --- lagged_car_2x2 ---
        conn.execute("DROP TABLE IF EXISTS lagged_car_2x2")
        conn.execute("""CREATE TABLE lagged_car_2x2 (
            news_regime TEXT, vol_regime TEXT, mean_car REAL,
            median_car REAL, std_car REAL, t_stat REAL, p_value REAL, n INTEGER
        )""")
        stored['lagged_car_2x2'] = _insert_2x2_car(
            conn, 'lagged_car_2x2', results.get('car_2x2_news', {}),
            regime_col='news_regime')

        # --- lagged_ols_models + lagged_ols_coefficients ---
        conn.execute("DROP TABLE IF EXISTS lagged_ols_models")
        conn.execute("""CREATE TABLE lagged_ols_models (
            model_name TEXT, n_obs INTEGER, r_squared REAL, adj_r_squared REAL,
            f_statistic REAL, f_p_value REAL, aic REAL, bic REAL
        )""")
        conn.execute("DROP TABLE IF EXISTS lagged_ols_coefficients")
        conn.execute("""CREATE TABLE lagged_ols_coefficients (
            model_name TEXT, variable TEXT, coefficient REAL, std_err REAL,
            t_stat REAL, p_value REAL, ci_lower REAL, ci_upper REAL
        )""")
        model_count = 0
        coef_count = 0
        # Main models (A, B, C) + robustness (7d, 60d)
        all_models = dict(results.get('ols_models', {}))
        for k, v in results.get('robustness_ols', {}).items():
            all_models[f'robustness_{k}'] = v
        for model_name, model in all_models.items():
            summary, coefs = _extract_ols(model)
            conn.execute(
                "INSERT INTO lagged_ols_models VALUES (?,?,?,?,?,?,?,?)",
                (model_name,) + summary,
            )
            model_count += 1
            for c in coefs:
                conn.execute(
                    "INSERT INTO lagged_ols_coefficients VALUES (?,?,?,?,?,?,?,?)",
                    (model_name,) + c,
                )
                coef_count += 1
        stored['lagged_ols_models'] = model_count
        stored['lagged_ols_coefficients'] = coef_count

    _print_stored("STEP 7 LAGGED SENTIMENT", stored)
    return stored


def get_lagged_sentiment_results(db_path=None):
    """Retrieve Step 7 lagged sentiment results from the database."""
    tables = ['lagged_coverage_stats', 'lagged_sentiment_dist',
              'lagged_ar_by_day', 'lagged_car', 'lagged_car_2x2',
              'lagged_ols_models', 'lagged_ols_coefficients']
    return {t: query_df(f"SELECT * FROM {t}", db_path=db_path) for t in tables}


# =============================================================================
# SHARED INSERTION HELPERS
# =============================================================================

def _insert_ar_by_day(conn, table_name, ar_dict):
    """Insert AR-by-day rows from a {regime: DataFrame} dict."""
    count = 0
    for regime, df in ar_dict.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            for _, r in df.iterrows():
                conn.execute(
                    f"INSERT INTO {table_name} VALUES (?,?,?,?,?,?)",
                    (regime, int(r['event_day']), float(r['mean_ar']),
                     float(r['t_stat']), float(r['p_value']), int(r['n'])),
                )
                count += 1
    return count


def _insert_car_stats(conn, table_name, car_dict):
    """Insert CAR summary rows from a {regime: stats_dict} dict."""
    count = 0
    for regime, s in car_dict.items():
        if s and s.get('n', 0) >= 2:
            conn.execute(
                f"INSERT INTO {table_name} VALUES (?,?,?,?,?,?,?)",
                (regime, s['mean_car'], s.get('median_car'), s.get('std_car'),
                 s['t_stat'], s['p_value'], s['n']),
            )
            count += 1
    return count


def _insert_2x2_car(conn, table_name, car_dict, regime_col='regime'):
    """Insert 2x2 CAR rows, splitting composite keys like 'negative_high_vol'."""
    count = 0
    for key, s in car_dict.items():
        if s and s.get('n', 0) >= 2:
            parts = key.rsplit('_', 2)  # e.g. 'negative_news_high_vol'
            # Find the vol part (always ends with _high_vol or _low_vol)
            if key.endswith('_high_vol'):
                primary = key[:-9]  # strip '_high_vol'
                vol = 'high_vol'
            elif key.endswith('_low_vol'):
                primary = key[:-8]  # strip '_low_vol'
                vol = 'low_vol'
            else:
                primary = key
                vol = 'unknown'
            conn.execute(
                f"INSERT INTO {table_name} VALUES (?,?,?,?,?,?,?,?)",
                (primary, vol, s['mean_car'], s.get('median_car'), s.get('std_car'),
                 s['t_stat'], s['p_value'], s['n']),
            )
            count += 1
    return count


def _store_simple_ols(results, prefix, db_path, key_cols=('regime',), step_label=""):
    """Store a flat dict of {regime: OLS_model} into two tables."""
    stored = {}
    with get_connection(db_path) as conn:
        models_table = f"{prefix}_ols_models"
        coefs_table = f"{prefix}_ols_coefficients"

        conn.execute(f"DROP TABLE IF EXISTS {models_table}")
        conn.execute(f"""CREATE TABLE {models_table} (
            regime TEXT, n_obs INTEGER, r_squared REAL, adj_r_squared REAL,
            f_statistic REAL, f_p_value REAL, aic REAL, bic REAL
        )""")
        conn.execute(f"DROP TABLE IF EXISTS {coefs_table}")
        conn.execute(f"""CREATE TABLE {coefs_table} (
            regime TEXT, variable TEXT, coefficient REAL, std_err REAL,
            t_stat REAL, p_value REAL, ci_lower REAL, ci_upper REAL
        )""")

        model_count = 0
        coef_count = 0
        for regime, model in results.items():
            summary, coefs = _extract_ols(model)
            conn.execute(
                f"INSERT INTO {models_table} VALUES (?,?,?,?,?,?,?,?)",
                (regime,) + summary,
            )
            model_count += 1
            for c in coefs:
                conn.execute(
                    f"INSERT INTO {coefs_table} VALUES (?,?,?,?,?,?,?,?)",
                    (regime,) + c,
                )
                coef_count += 1

        stored[models_table] = model_count
        stored[coefs_table] = coef_count

    _print_stored(step_label, stored)
    return stored


def _store_nested_ols(results, prefix, db_path, step_label=""):
    """Store a nested dict of {model_spec: {regime: OLS_model}} into two tables."""
    stored = {}
    with get_connection(db_path) as conn:
        models_table = f"{prefix}_ols_models"
        coefs_table = f"{prefix}_ols_coefficients"

        conn.execute(f"DROP TABLE IF EXISTS {models_table}")
        conn.execute(f"""CREATE TABLE {models_table} (
            model_spec TEXT, regime TEXT, n_obs INTEGER, r_squared REAL,
            adj_r_squared REAL, f_statistic REAL, f_p_value REAL,
            aic REAL, bic REAL
        )""")
        conn.execute(f"DROP TABLE IF EXISTS {coefs_table}")
        conn.execute(f"""CREATE TABLE {coefs_table} (
            model_spec TEXT, regime TEXT, variable TEXT, coefficient REAL,
            std_err REAL, t_stat REAL, p_value REAL, ci_lower REAL, ci_upper REAL
        )""")

        model_count = 0
        coef_count = 0
        for model_spec, regimes in results.items():
            for regime, model in regimes.items():
                summary, coefs = _extract_ols(model)
                conn.execute(
                    f"INSERT INTO {models_table} VALUES (?,?,?,?,?,?,?,?,?)",
                    (model_spec, regime) + summary,
                )
                model_count += 1
                for c in coefs:
                    conn.execute(
                        f"INSERT INTO {coefs_table} VALUES (?,?,?,?,?,?,?,?,?)",
                        (model_spec, regime) + c,
                    )
                    coef_count += 1

        stored[models_table] = model_count
        stored[coefs_table] = coef_count

    _print_stored(step_label, stored)
    return stored


def _print_stored(label, stored):
    """Print summary of stored rows."""
    print(f"\n{'=' * 60}")
    print(f"{label} STORED IN DATABASE")
    print(f"{'=' * 60}")
    for table, count in stored.items():
        print(f"  {table}: {count} rows")
    print()


# =============================================================================
# MASTER STORE / GET
# =============================================================================

def store_all_results(all_results, db_path=None):
    """
    Store all step results in the database.

    Args:
        all_results: dict with keys 'step1' through 'step8', each containing
                     the return value from that step's run function.
    """
    stored = {}
    if 'step1' in all_results:
        stored.update(store_descriptive_results(all_results['step1'], db_path))
    if 'step2' in all_results:
        stored.update(store_fama_french_results(all_results['step2'], db_path))
    if 'step3' in all_results:
        stored.update(store_macro_controls_results(all_results['step3'], db_path))
    if 'step4' in all_results:
        stored.update(store_breach_level_results(all_results['step4'], db_path))
    if 'step5' in all_results:
        stored.update(store_event_study_results(all_results['step5'], db_path))
    if 'step6' in all_results:
        stored.update(store_sentiment_results(all_results['step6'], db_path))
    if 'step7' in all_results:
        stored.update(store_lagged_sentiment_results(all_results['step7'], db_path))
    if 'step8' in all_results:
        stored.update(store_repeat_offender_results(all_results['step8'], db_path))
    return stored


def get_all_results(db_path=None):
    """Retrieve all step results from the database."""
    results = {}
    results.update(get_descriptive_results(db_path))
    results.update(get_fama_french_results(db_path))
    results.update(get_macro_controls_results(db_path))
    results.update(get_breach_level_results(db_path))
    results.update(get_event_study_results(db_path))
    results.update(get_sentiment_results(db_path))
    results.update(get_lagged_sentiment_results(db_path))
    results.update(get_repeat_offender_results(db_path))
    return results


# =============================================================================
# STEP 8: REPEAT OFFENDER RESULTS
# =============================================================================

def store_repeat_offender_results(results, db_path=None):
    """
    Store Step 8 repeat offender analysis results in the database.

    Takes the dict returned by run_repeat_offender_analysis().
    Drops and recreates all 6 tables (idempotent on re-runs).
    """
    if db_path is None:
        db_path = DB_PATH

    TABLE_DDL = {
        'repeat_offender_history': """
            CREATE TABLE repeat_offender_history (
                ticker TEXT,
                reported_date TEXT,
                breach_sequence INTEGER,
                is_repeat INTEGER,
                prior_breach_count INTEGER,
                days_since_last REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )""",
        'repeat_offender_ar_by_day': """
            CREATE TABLE repeat_offender_ar_by_day (
                regime TEXT,
                event_day INTEGER,
                mean_ar REAL,
                t_stat REAL,
                p_value REAL,
                n INTEGER
            )""",
        'repeat_offender_car': """
            CREATE TABLE repeat_offender_car (
                regime TEXT,
                mean_car REAL,
                median_car REAL,
                std_car REAL,
                t_stat REAL,
                p_value REAL,
                n INTEGER
            )""",
        'repeat_offender_car_2x2': """
            CREATE TABLE repeat_offender_car_2x2 (
                repeat_regime TEXT,
                vol_regime TEXT,
                mean_car REAL,
                median_car REAL,
                std_car REAL,
                t_stat REAL,
                p_value REAL,
                n INTEGER
            )""",
        'repeat_offender_ols_models': """
            CREATE TABLE repeat_offender_ols_models (
                model_name TEXT,
                n_obs INTEGER,
                r_squared REAL,
                adj_r_squared REAL,
                f_statistic REAL,
                f_p_value REAL,
                aic REAL,
                bic REAL
            )""",
        'repeat_offender_ols_coefficients': """
            CREATE TABLE repeat_offender_ols_coefficients (
                model_name TEXT,
                variable TEXT,
                coefficient REAL,
                std_err REAL,
                t_stat REAL,
                p_value REAL,
                ci_lower REAL,
                ci_upper REAL
            )""",
    }

    stored = {}

    with get_connection(db_path) as conn:
        # Drop and recreate all tables
        for table_name, ddl in TABLE_DDL.items():
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.execute(ddl)

        # 1. repeat_offender_history — per-event breach sequence features
        history_df = results.get('history_df')
        if history_df is not None and not history_df.empty:
            rows = []
            for _, row in history_df.iterrows():
                rows.append((
                    row['yf_ticker'],
                    str(row['reported_date_norm'].date()),
                    int(row['breach_sequence']),
                    int(row['is_repeat']),
                    int(row['prior_breach_count']),
                    float(row['days_since_last']) if pd.notna(row['days_since_last']) else None,
                ))
            conn.executemany(
                "INSERT INTO repeat_offender_history "
                "(ticker, reported_date, breach_sequence, is_repeat, prior_breach_count, days_since_last) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
            stored['repeat_offender_history'] = len(rows)
        else:
            stored['repeat_offender_history'] = 0

        # 2. repeat_offender_ar_by_day — AR by event day for first vs repeat
        ar_by_day = results.get('ar_by_day_repeat', {})
        count = 0
        for regime, df in ar_by_day.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                for _, row in df.iterrows():
                    conn.execute(
                        "INSERT INTO repeat_offender_ar_by_day "
                        "(regime, event_day, mean_ar, t_stat, p_value, n) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (regime, int(row['event_day']), float(row['mean_ar']),
                         float(row['t_stat']), float(row['p_value']), int(row['n'])),
                    )
                    count += 1
        stored['repeat_offender_ar_by_day'] = count

        # 3. repeat_offender_car — CAR summary by first/repeat regime
        car_by_repeat = results.get('car_by_repeat', {})
        count = 0
        for regime, stats in car_by_repeat.items():
            if stats and stats.get('n', 0) >= 2:
                conn.execute(
                    "INSERT INTO repeat_offender_car "
                    "(regime, mean_car, median_car, std_car, t_stat, p_value, n) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (regime, stats['mean_car'], stats.get('median_car'),
                     stats.get('std_car'), stats['t_stat'], stats['p_value'], stats['n']),
                )
                count += 1
        stored['repeat_offender_car'] = count

        # 4. repeat_offender_car_2x2 — 2x2 CAR (first/repeat x VIX)
        car_2x2 = results.get('car_2x2_repeat_vix', {})
        count = 0
        for key, stats in car_2x2.items():
            if stats and stats.get('n', 0) >= 2:
                # key is e.g. 'first_low_vol' -> repeat_regime='first', vol_regime='low_vol'
                parts = key.split('_', 1)
                repeat_regime = parts[0]
                vol_regime = parts[1] if len(parts) > 1 else key
                conn.execute(
                    "INSERT INTO repeat_offender_car_2x2 "
                    "(repeat_regime, vol_regime, mean_car, median_car, std_car, t_stat, p_value, n) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (repeat_regime, vol_regime, stats['mean_car'], stats.get('median_car'),
                     stats.get('std_car'), stats['t_stat'], stats['p_value'], stats['n']),
                )
                count += 1
        stored['repeat_offender_car_2x2'] = count

        # 5 & 6. OLS models and coefficients
        ols_models = results.get('ols_models', {})
        model_count = 0
        coef_count = 0
        for model_name, model in ols_models.items():
            conn.execute(
                "INSERT INTO repeat_offender_ols_models "
                "(model_name, n_obs, r_squared, adj_r_squared, f_statistic, f_p_value, aic, bic) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (model_name, int(model.nobs), float(model.rsquared),
                 float(model.rsquared_adj), float(model.fvalue),
                 float(model.f_pvalue), float(model.aic), float(model.bic)),
            )
            model_count += 1

            conf_int = model.conf_int()
            for var in model.params.index:
                ci = conf_int.loc[var]
                conn.execute(
                    "INSERT INTO repeat_offender_ols_coefficients "
                    "(model_name, variable, coefficient, std_err, t_stat, p_value, ci_lower, ci_upper) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (model_name, var, float(model.params[var]), float(model.bse[var]),
                     float(model.tvalues[var]), float(model.pvalues[var]),
                     float(ci[0]), float(ci[1])),
                )
                coef_count += 1

        stored['repeat_offender_ols_models'] = model_count
        stored['repeat_offender_ols_coefficients'] = coef_count

    print("\n" + "=" * 60)
    print("REPEAT OFFENDER RESULTS STORED IN DATABASE")
    print("=" * 60)
    for table, count in stored.items():
        print(f"  {table}: {count} rows")
    print()

    return stored


def get_repeat_offender_results(db_path=None):
    """
    Retrieve Step 8 repeat offender results from the database.

    Returns a dict of 6 DataFrames, one per table.
    """
    table_names = [
        'repeat_offender_history',
        'repeat_offender_ar_by_day',
        'repeat_offender_car',
        'repeat_offender_car_2x2',
        'repeat_offender_ols_models',
        'repeat_offender_ols_coefficients',
    ]
    results = {}
    for name in table_names:
        results[name] = query_df(f"SELECT * FROM {name}", db_path=db_path)
    return results


# =============================================================================
# FIGURES STORAGE
# =============================================================================

def _ensure_figures_table(db_path=None):
    """Create the figures table if it doesn't exist."""
    if db_path is None:
        db_path = DB_PATH
    with get_connection(db_path) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS figures (
            filename TEXT PRIMARY KEY,
            step TEXT,
            label TEXT,
            png_data BLOB NOT NULL,
            size_bytes INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")


def store_figure(filename, step, label, png_path, db_path=None):
    """
    Store a single PNG figure in the database.

    Args:
        filename: e.g. 'fig_1_1_breach_timeline.png'
        step: e.g. 'step1'
        label: e.g. '1.1 Breach Timeline'
        png_path: Path to the PNG file on disk
        db_path: Optional database path
    """
    if db_path is None:
        db_path = DB_PATH
    _ensure_figures_table(db_path)
    png_data = Path(png_path).read_bytes()
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO figures (filename, step, label, png_data, size_bytes) "
            "VALUES (?, ?, ?, ?, ?)",
            (filename, step, label, png_data, len(png_data)),
        )


def store_figures_batch(figures_list, db_path=None):
    """
    Store multiple figures in one transaction.

    Args:
        figures_list: list of (filename, step, label, png_path) tuples
        db_path: Optional database path
    """
    if db_path is None:
        db_path = DB_PATH
    _ensure_figures_table(db_path)
    with get_connection(db_path) as conn:
        for filename, step, label, png_path in figures_list:
            png_data = Path(png_path).read_bytes()
            conn.execute(
                "INSERT OR REPLACE INTO figures (filename, step, label, png_data, size_bytes) "
                "VALUES (?, ?, ?, ?, ?)",
                (filename, step, label, png_data, len(png_data)),
            )


def get_figure(filename, db_path=None):
    """
    Retrieve a figure's PNG data from the database.

    Returns:
        dict with 'filename', 'step', 'label', 'png_data', 'size_bytes',
        'created_at', or None if not found.
    """
    results = query(
        "SELECT filename, step, label, png_data, size_bytes, created_at "
        "FROM figures WHERE filename = ?",
        (filename,), db_path,
    )
    return results[0] if results else None


def get_figures_index(db_path=None):
    """
    List all stored figures (without BLOB data).

    Returns:
        DataFrame with filename, step, label, size_bytes, created_at
    """
    _ensure_figures_table(db_path)
    return query_df(
        "SELECT filename, step, label, size_bytes, created_at FROM figures ORDER BY filename",
        db_path=db_path,
    )


def export_figure(filename, output_path=None, db_path=None):
    """
    Export a figure from the database back to a PNG file.

    Args:
        filename: The figure filename stored in the DB
        output_path: Where to write the PNG (defaults to figures/<filename>)
        db_path: Optional database path

    Returns:
        Path to the written file, or None if figure not found.
    """
    row = get_figure(filename, db_path)
    if row is None:
        return None
    if output_path is None:
        out_dir = Path(__file__).parent / "figures"
        out_dir.mkdir(exist_ok=True)
        output_path = out_dir / filename
    else:
        output_path = Path(output_path)
    output_path.write_bytes(row['png_data'])
    return output_path


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BREACH DATA DATABASE")
    print("=" * 60)
    print()

    print_database_info()

    print()
    print("=" * 60)
    print("SAMPLE QUERIES")
    print("=" * 60)

    print("\nTop 5 Breaches:")
    top = get_top_breaches(5)
    for breach in top:
        print(f"  {breach['org_name']}: {breach['total_affected']:,} affected")

    print("\nBreaches by Sector:")
    sectors = get_sector_breakdown()
    print(sectors.head(5).to_string(index=False))
