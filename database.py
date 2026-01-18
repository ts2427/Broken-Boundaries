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
