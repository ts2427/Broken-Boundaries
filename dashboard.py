"""
dashboard.py - Streamlit Dashboard for Broken Boundaries
Interactive visualization and exploration of breach data.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sqlite3

# Page config
st.set_page_config(
    page_title="Broken Boundaries",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === PATHS ===
BASE_DIR = Path(__file__).parent
RAW_DATA_DIR = BASE_DIR / "raw"
CLEANED_DATA_DIR = BASE_DIR / "cleaned"
DB_PATH = BASE_DIR / "database" / "breach_data.db"

# === HEADER ===
st.title("Broken Boundaries")
st.markdown("### Data Breach Analysis Pipeline")
st.markdown("**Authors:** Ashley D. Roseboro, Abigail Hagood-Dokter, Timothy Spivey")
st.divider()

# === TABS ===
tab_code, tab_raw, tab_cleaned, tab_model = st.tabs([
    "📝 Code",
    "📁 Raw Data",
    "🧹 Cleaned Data",
    "📊 Model Results"
])


# =============================================================================
# TAB 1: CODE
# =============================================================================
with tab_code:
    st.header("Pipeline Code")
    st.markdown("View all source code files in the data pipeline.")

    code_files = {
        "clean.py": "Data cleaning and enrichment pipeline",
        "etl.py": "Extract, Transform, Load pipeline",
        "database.py": "Database connection and queries",
        "model.py": "Data models and analysis",
        "dashboard.py": "This Streamlit dashboard",
    }

    selected_file = st.selectbox(
        "Select a code file to view:",
        options=list(code_files.keys()),
        format_func=lambda x: f"{x} - {code_files[x]}"
    )

    file_path = BASE_DIR / selected_file
    if file_path.exists():
        with open(file_path, 'r') as f:
            code_content = f.read()

        st.markdown(f"**File:** `{selected_file}`")
        st.markdown(f"**Description:** {code_files[selected_file]}")
        st.markdown(f"**Lines:** {len(code_content.splitlines())}")
        st.divider()
        st.code(code_content, language="python", line_numbers=True)
    else:
        st.warning(f"File not found: {selected_file}")


# =============================================================================
# TAB 2: RAW DATA
# =============================================================================
with tab_raw:
    st.header("Raw Data Files")
    st.markdown("View the original data files before processing.")

    # Look for CSV files in raw directory and root
    raw_files = []
    if RAW_DATA_DIR.exists():
        raw_files.extend(list(RAW_DATA_DIR.glob("*.csv")))

    # Also check for the original file in root
    root_csv = BASE_DIR / "Data_Breach_Enriched_Final.csv"
    if root_csv.exists():
        raw_files.append(root_csv)

    if raw_files:
        selected_raw = st.selectbox(
            "Select a raw data file:",
            options=raw_files,
            format_func=lambda x: x.name
        )

        if selected_raw and selected_raw.exists():
            try:
                # Try different encodings
                try:
                    df_raw = pd.read_csv(selected_raw, encoding='utf-8')
                except UnicodeDecodeError:
                    df_raw = pd.read_csv(selected_raw, encoding='latin-1')

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", f"{len(df_raw):,}")
                with col2:
                    st.metric("Columns", f"{len(df_raw.columns)}")
                with col3:
                    st.metric("File Size", f"{selected_raw.stat().st_size / 1024:.1f} KB")

                st.divider()

                # Column info
                with st.expander("📋 Column Information", expanded=False):
                    col_info = pd.DataFrame({
                        'Column': df_raw.columns,
                        'Type': df_raw.dtypes.astype(str),
                        'Non-Null': df_raw.notna().sum(),
                        'Null %': (df_raw.isna().sum() / len(df_raw) * 100).round(2)
                    })
                    st.dataframe(col_info, use_container_width=True)

                # Data preview
                st.subheader("Data Preview")
                st.dataframe(df_raw, use_container_width=True, height=400)

            except Exception as e:
                st.error(f"Error reading file: {e}")
    else:
        st.info("No raw data files found. Looking for Data_Breach_Enriched_Final.csv")
        st.markdown("Expected locations:")
        st.markdown(f"- `{RAW_DATA_DIR}`")
        st.markdown(f"- `{BASE_DIR}`")


# =============================================================================
# TAB 3: CLEANED DATA
# =============================================================================
with tab_cleaned:
    st.header("Cleaned & Enriched Data")
    st.markdown("View data after processing through clean.py pipeline.")

    cleaned_files = []
    if CLEANED_DATA_DIR.exists():
        cleaned_files = list(CLEANED_DATA_DIR.glob("*.csv"))

    if cleaned_files:
        selected_cleaned = st.selectbox(
            "Select a cleaned data file:",
            options=cleaned_files,
            format_func=lambda x: x.name
        )

        if selected_cleaned and selected_cleaned.exists():
            try:
                df_cleaned = pd.read_csv(selected_cleaned)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Rows", f"{len(df_cleaned):,}")
                with col2:
                    st.metric("Columns", f"{len(df_cleaned.columns)}")
                with col3:
                    st.metric("File Size", f"{selected_cleaned.stat().st_size / 1024 / 1024:.2f} MB")
                with col4:
                    if 'total_affected' in df_cleaned.columns:
                        total = df_cleaned['total_affected'].sum()
                        st.metric("Total Affected", f"{total:,.0f}")

                st.divider()

                # Column categories
                st.subheader("Column Categories")

                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    st.markdown("**Core Breach Data**")
                    core_cols = ['org_name', 'stock_ticker', 'breach_date', 'total_affected', 'breach_type']
                    for col in core_cols:
                        if col in df_cleaned.columns:
                            st.markdown(f"- `{col}`")

                with col_b:
                    st.markdown("**Yahoo Finance Data**")
                    yf_cols = [c for c in df_cleaned.columns if c.startswith('yf_')]
                    for col in yf_cols[:10]:
                        st.markdown(f"- `{col}`")
                    if len(yf_cols) > 10:
                        st.markdown(f"- ... and {len(yf_cols) - 10} more")

                with col_c:
                    st.markdown("**News Data**")
                    news_cols = ['reddit_count', 'guardian_count', 'nyt_count', 'newsapi_count', 'total_news_count']
                    for col in news_cols:
                        if col in df_cleaned.columns:
                            st.markdown(f"- `{col}`")

                st.divider()

                # Filters
                st.subheader("Data Explorer")

                filter_col1, filter_col2 = st.columns(2)

                with filter_col1:
                    if 'yf_sector' in df_cleaned.columns:
                        sectors = ['All'] + sorted(df_cleaned['yf_sector'].dropna().unique().tolist())
                        selected_sector = st.selectbox("Filter by Sector:", sectors)

                with filter_col2:
                    if 'breach_type' in df_cleaned.columns:
                        types = ['All'] + sorted(df_cleaned['breach_type'].dropna().unique().tolist())
                        selected_type = st.selectbox("Filter by Breach Type:", types)

                # Apply filters
                df_display = df_cleaned.copy()
                if 'yf_sector' in df_cleaned.columns and selected_sector != 'All':
                    df_display = df_display[df_display['yf_sector'] == selected_sector]
                if 'breach_type' in df_cleaned.columns and selected_type != 'All':
                    df_display = df_display[df_display['breach_type'] == selected_type]

                st.markdown(f"**Showing {len(df_display):,} records**")
                st.dataframe(df_display, use_container_width=True, height=400)

            except Exception as e:
                st.error(f"Error reading file: {e}")
    else:
        st.warning("No cleaned data files found.")
        st.markdown(f"Expected location: `{CLEANED_DATA_DIR}`")
        st.markdown("Run `python clean.py` to generate cleaned data.")


# =============================================================================
# TAB 4: MODEL RESULTS
# =============================================================================
with tab_model:
    st.header("Model Results & Analysis")
    st.markdown("View statistical analysis and model outputs.")

    # Check if database exists
    if DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)

        # Summary Statistics
        st.subheader("📈 Summary Statistics")

        col1, col2, col3, col4 = st.columns(4)

        # Get counts
        total_records = pd.read_sql("SELECT COUNT(*) as count FROM breach_incidents", conn).iloc[0]['count']
        total_affected = pd.read_sql("SELECT SUM(total_affected) as total FROM breach_incidents", conn).iloc[0]['total']
        unique_orgs = pd.read_sql("SELECT COUNT(DISTINCT org_name) as count FROM breach_incidents", conn).iloc[0]['count']
        total_news = pd.read_sql("SELECT SUM(total_news_count) as total FROM breach_incidents", conn).iloc[0]['total']

        with col1:
            st.metric("Total Breaches", f"{total_records:,}")
        with col2:
            st.metric("Total Affected", f"{total_affected:,.0f}")
        with col3:
            st.metric("Unique Organizations", f"{unique_orgs:,}")
        with col4:
            st.metric("News Articles", f"{total_news:,.0f}")

        st.divider()

        # Severity Distribution
        st.subheader("🎯 Breach Severity Distribution")

        severity_query = """
            SELECT
                CASE
                    WHEN total_affected < 10000 THEN 'Low (<10K)'
                    WHEN total_affected < 100000 THEN 'Medium (10K-100K)'
                    WHEN total_affected < 1000000 THEN 'High (100K-1M)'
                    WHEN total_affected < 10000000 THEN 'Critical (1M-10M)'
                    ELSE 'Massive (>10M)'
                END as severity,
                COUNT(*) as count
            FROM breach_incidents
            WHERE total_affected IS NOT NULL
            GROUP BY severity
            ORDER BY
                CASE severity
                    WHEN 'Low (<10K)' THEN 1
                    WHEN 'Medium (10K-100K)' THEN 2
                    WHEN 'High (100K-1M)' THEN 3
                    WHEN 'Critical (1M-10M)' THEN 4
                    ELSE 5
                END
        """
        severity_df = pd.read_sql(severity_query, conn)

        col_sev1, col_sev2 = st.columns([2, 3])
        with col_sev1:
            st.dataframe(severity_df, use_container_width=True, hide_index=True)
        with col_sev2:
            st.bar_chart(severity_df.set_index('severity')['count'])

        st.divider()

        # Breaches by Sector
        st.subheader("🏢 Breaches by Sector")

        sector_df = pd.read_sql("SELECT * FROM v_breaches_by_sector", conn)

        col_sec1, col_sec2 = st.columns([2, 3])
        with col_sec1:
            st.dataframe(sector_df, use_container_width=True, hide_index=True)
        with col_sec2:
            st.bar_chart(sector_df.set_index('sector')['breach_count'])

        st.divider()

        # Breaches by Year
        st.subheader("📅 Breaches by Year")

        year_df = pd.read_sql("SELECT * FROM v_breaches_by_year", conn)

        col_yr1, col_yr2 = st.columns([2, 3])
        with col_yr1:
            st.dataframe(year_df, use_container_width=True, hide_index=True)
        with col_yr2:
            st.line_chart(year_df.set_index('year')['breach_count'])

        st.divider()

        # Breaches by Type
        st.subheader("🔓 Breaches by Type")

        type_df = pd.read_sql("SELECT * FROM v_breaches_by_type", conn)
        st.dataframe(type_df, use_container_width=True, hide_index=True)

        st.divider()

        # Top Breaches
        st.subheader("🏆 Top 10 Largest Breaches")

        top_df = pd.read_sql("SELECT * FROM v_top_breaches LIMIT 10", conn)
        st.dataframe(top_df, use_container_width=True, hide_index=True)

        st.divider()

        # News Coverage
        st.subheader("📰 News Coverage Analysis")

        news_df = pd.read_sql("SELECT * FROM v_news_coverage LIMIT 20", conn)
        st.dataframe(news_df, use_container_width=True, hide_index=True)

        st.divider()

        # Descriptive Statistics
        st.subheader("📊 Descriptive Statistics")

        desc_query = """
            SELECT
                'total_affected' as variable,
                COUNT(total_affected) as count,
                ROUND(AVG(total_affected), 2) as mean,
                MIN(total_affected) as min,
                MAX(total_affected) as max
            FROM breach_incidents
            WHERE total_affected IS NOT NULL
            UNION ALL
            SELECT
                'total_news_count' as variable,
                COUNT(total_news_count) as count,
                ROUND(AVG(total_news_count), 2) as mean,
                MIN(total_news_count) as min,
                MAX(total_news_count) as max
            FROM breach_incidents
            UNION ALL
            SELECT
                'yf_market_cap' as variable,
                COUNT(yf_market_cap) as count,
                ROUND(AVG(yf_market_cap), 2) as mean,
                MIN(yf_market_cap) as min,
                MAX(yf_market_cap) as max
            FROM breach_incidents
            WHERE yf_market_cap IS NOT NULL
        """
        desc_df = pd.read_sql(desc_query, conn)
        st.dataframe(desc_df, use_container_width=True, hide_index=True)

        conn.close()

    else:
        st.warning("Database not found.")
        st.markdown(f"Expected location: `{DB_PATH}`")
        st.markdown("Run `python etl.py` to create the database.")


# === SIDEBAR ===
with st.sidebar:
    st.header("About")
    st.markdown("""
    **Broken Boundaries** is a data analysis pipeline for studying
    corporate data breaches and their impact.

    **Pipeline Stages:**
    1. **Clean** - Data cleaning & enrichment
    2. **ETL** - Load into SQLite database
    3. **Model** - Statistical analysis
    4. **Dashboard** - Interactive visualization
    """)

    st.divider()

    st.header("Data Sources")
    st.markdown("""
    - **Breach Data**: Privacy Rights Clearinghouse
    - **Stock Data**: Yahoo Finance
    - **News**: Reddit, Guardian, NYT, NewsAPI
    """)

    st.divider()

    st.header("Quick Stats")
    if DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)
        stats = pd.read_sql("SELECT COUNT(*) as breaches FROM breach_incidents", conn)
        st.metric("Total Breaches", f"{stats.iloc[0]['breaches']:,}")
        conn.close()

    st.divider()
    st.markdown("---")
    st.markdown("*Built with Streamlit*")


# === FOOTER ===
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        Broken Boundaries © 2025 | Ashley D. Roseboro, Abigail Hagood-Dokter, Timothy Spivey
    </div>
    """,
    unsafe_allow_html=True
)
