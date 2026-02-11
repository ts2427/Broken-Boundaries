"""
dashboard.py - Executive Briefing Dashboard for Broken Boundaries
Presidential-briefing-quality visualization of the 8-step data breach analysis pipeline.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from io import BytesIO

import database as db

# =============================================================================
# CONFIG & CONSTANTS
# =============================================================================

st.set_page_config(
    page_title="Broken Boundaries",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
CLEANED_DIR = BASE_DIR / "cleaned"
RAW_CSV = BASE_DIR / "Data_Breach_Enriched_Final.csv"

NAVY = "#1B2A4A"
GOLD = "#C5A55A"
WHITE = "#FFFFFF"
LIGHT_GRAY = "#F5F5F0"

STEP_DEFINITIONS = {
    "step1": {
        "num": 1,
        "title": "Descriptive Statistics",
        "getter": db.get_descriptive_results,
    },
    "step2": {
        "num": 2,
        "title": "Fama-French OLS Regressions",
        "getter": db.get_fama_french_results,
    },
    "step3": {
        "num": 3,
        "title": "Macro Controls OLS Regressions",
        "getter": db.get_macro_controls_results,
    },
    "step4": {
        "num": 4,
        "title": "Breach-Level OLS Regressions",
        "getter": db.get_breach_level_results,
    },
    "step5": {
        "num": 5,
        "title": "Event Study Analysis",
        "getter": db.get_event_study_results,
    },
    "step6": {
        "num": 6,
        "title": "Sentiment Analysis",
        "getter": db.get_sentiment_results,
    },
    "step7": {
        "num": 7,
        "title": "Lagged Sentiment Analysis",
        "getter": db.get_lagged_sentiment_results,
    },
    "step8": {
        "num": 8,
        "title": "Repeat Offender Analysis",
        "getter": db.get_repeat_offender_results,
    },
}

DEFAULT_EXPLORER_COLS = [
    "org_name", "stock_ticker", "breach_date", "total_affected",
    "breach_type", "yf_sector", "yf_market_cap", "total_news_count",
]


# =============================================================================
# CUSTOM CSS
# =============================================================================

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Source+Serif+4:wght@300;400;600&display=swap');

/* Global typography */
html, body, [class*="css"] {{
    font-family: 'Source Serif 4', 'Georgia', 'Times New Roman', serif;
    color: {NAVY};
}}
h1, h2, h3, h4, h5, h6 {{
    font-family: 'Playfair Display', 'Georgia', serif;
    color: {NAVY};
    font-weight: 600;
}}
h1 {{ font-size: 2rem; letter-spacing: 0.02em; }}
h2 {{ font-size: 1.5rem; }}
h3 {{ font-size: 1.25rem; }}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background-color: {NAVY};
}}
section[data-testid="stSidebar"] * {{
    color: {WHITE} !important;
}}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
    color: {GOLD} !important;
    font-family: 'Playfair Display', 'Georgia', serif;
}}
section[data-testid="stSidebar"] .stMetric label {{
    color: {GOLD} !important;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
section[data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"] {{
    color: {WHITE} !important;
    font-family: 'Source Serif 4', 'Georgia', serif;
}}

/* Metric cards */
[data-testid="stMetric"] {{
    background-color: {LIGHT_GRAY};
    border-left: 4px solid {GOLD};
    padding: 0.75rem 1rem;
    border-radius: 0 4px 4px 0;
}}
[data-testid="stMetric"] label {{
    color: {NAVY} !important;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
[data-testid="stMetric"] [data-testid="stMetricValue"] {{
    color: {NAVY} !important;
    font-family: 'Playfair Display', 'Georgia', serif;
    font-weight: 600;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0;
    border-bottom: 2px solid #E0E0E0;
}}
.stTabs [data-baseweb="tab"] {{
    color: {NAVY};
    font-family: 'Playfair Display', 'Georgia', serif;
    font-weight: 600;
    padding: 0.75rem 1.5rem;
    border-bottom: 3px solid transparent;
}}
.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    color: {NAVY};
    border-bottom: 3px solid {GOLD};
    background-color: transparent;
}}

/* Expanders */
.streamlit-expanderHeader {{
    font-family: 'Playfair Display', 'Georgia', serif;
    font-weight: 600;
    color: {NAVY};
    font-size: 1.1rem;
}}

/* DataFrames */
.stDataFrame {{
    border: 1px solid #E0E0E0;
    border-radius: 4px;
}}

/* Hide Streamlit branding */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}

/* Divider */
hr {{
    border-color: {GOLD};
    opacity: 0.3;
}}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def fmt_sig(p):
    """Return significance stars for a p-value."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def fmt_num(val, dec=0):
    """Format a number with commas; return '--' for None/NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "--"
    if dec == 0:
        return f"{int(val):,}"
    return f"{val:,.{dec}f}"


def fmt_pval(p):
    """Format a p-value for display."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "--"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.4f}"


def style_coef_df(df):
    """Add significance column, rename to publication headers, round to 4 decimals."""
    if df.empty:
        return df
    out = df.copy()
    if "p_value" in out.columns:
        out["sig"] = out["p_value"].apply(fmt_sig)
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].round(4)
    rename_map = {
        "variable": "Variable",
        "coefficient": "Coef.",
        "std_err": "Std. Err.",
        "t_stat": "t",
        "p_value": "p-value",
        "ci_lower": "CI Lower",
        "ci_upper": "CI Upper",
        "sig": "Sig.",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})
    return out


def style_model_df(df):
    """Format model summary table with publication-quality headers."""
    if df.empty:
        return df
    out = df.copy()
    rename_map = {
        "regime": "Regime",
        "model_spec": "Model Spec",
        "model_name": "Model",
        "n_obs": "N",
        "r_squared": "R-sq",
        "adj_r_squared": "Adj. R-sq",
        "f_statistic": "F-stat",
        "f_p_value": "F p-value",
        "aic": "AIC",
        "bic": "BIC",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})
    if "N" in out.columns:
        out["N"] = out["N"].apply(lambda x: fmt_num(x))
    for col in ["R-sq", "Adj. R-sq"]:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "--")
    for col in ["F-stat", "AIC", "BIC"]:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: fmt_num(x, 2) if pd.notna(x) else "--")
    if "F p-value" in out.columns:
        out["F p-value"] = out["F p-value"].apply(fmt_pval)
    return out


def render_figure(png_data, caption=""):
    """Display a PNG BLOB via st.image."""
    if png_data:
        st.image(BytesIO(png_data), caption=caption, use_container_width=True)


# =============================================================================
# DATA LOADING (cached)
# =============================================================================

@st.cache_data(ttl=300)
def load_step(key):
    """Load a pipeline step's results from the database."""
    return STEP_DEFINITIONS[key]["getter"]()


@st.cache_data(ttl=300)
def load_figures_index():
    """Load the figures index (no BLOBs)."""
    return db.get_figures_index()


@st.cache_data(ttl=300)
def load_breach_data():
    """Load the full breach_incidents table."""
    return db.query_df("SELECT * FROM breach_incidents")


@st.cache_data(ttl=300)
def load_summary_stats():
    """Load summary statistics."""
    return db.get_summary_stats()


@st.cache_data(ttl=300)
def load_view_df(view_name):
    """Load a database view as DataFrame."""
    return db.query_df(f"SELECT * FROM {view_name}")


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown(f"### Broken Boundaries")
    st.markdown("Data Breach Impact on Publicly Traded Companies")
    st.divider()

    stats = load_summary_stats()

    st.metric("Total Incidents", fmt_num(stats.get("total_records")))
    st.metric("Individuals Affected", fmt_num(stats.get("total_affected")))
    st.metric("Unique Organizations", fmt_num(stats.get("unique_organizations")))
    st.metric(
        "Analysis Period",
        f"{stats.get('earliest_breach', '?')[:4]} -- {stats.get('latest_breach', '?')[:4]}",
    )

    st.divider()
    st.markdown("#### Analysis Pipeline")
    for key, defn in STEP_DEFINITIONS.items():
        st.markdown(f"{defn['num']}. {defn['title']}")

    st.divider()
    st.markdown("#### Data Sources")
    st.markdown("Privacy Rights Clearinghouse")
    st.markdown("Yahoo Finance")
    st.markdown("Reddit, Guardian, NYT, NewsAPI")
    st.markdown("Fama-French 5-Factor")
    st.markdown("CBOE VIX, FRED Macro Series")

    st.divider()
    st.caption("Roseboro, Hagood-Dokter, Spivey")


# =============================================================================
# MAIN TABS
# =============================================================================

tab_model, tab_viz, tab_enriched, tab_raw = st.tabs([
    "Model Results",
    "Visualizations",
    "Enriched Data",
    "Raw Data",
])


# =============================================================================
# TAB 1: MODEL RESULTS
# =============================================================================

with tab_model:
    st.header("Analysis Pipeline Results")
    st.markdown("Eight-step empirical analysis of data breach impact on equity returns.")
    st.divider()

    # ---- Step 1: Descriptive ----
    with st.expander("Step 1: Descriptive Statistics", expanded=False):
        data1 = load_step("step1")
        ov = data1.get("descriptive_overview", pd.DataFrame())
        num_df = data1.get("descriptive_numeric", pd.DataFrame())
        cat_df = data1.get("descriptive_categorical", pd.DataFrame())
        date_df = data1.get("descriptive_date", pd.DataFrame())

        if not ov.empty:
            row = ov.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Records", fmt_num(row.get("total_records")))
            c2.metric("Columns", fmt_num(row.get("total_columns")))
            c3.metric("News Articles", fmt_num(row.get("news_total")))
            c4.metric("Duplicate Rows", fmt_num(row.get("duplicate_rows")))

        if not num_df.empty:
            st.markdown("**Numeric Variable Summary**")
            st.dataframe(num_df.round(3), use_container_width=True, hide_index=True)

        if not cat_df.empty:
            st.markdown("**Categorical Variable Summary**")
            display_cat = cat_df.drop(columns=["top_5_json"], errors="ignore")
            st.dataframe(display_cat, use_container_width=True, hide_index=True)

        if not date_df.empty:
            st.markdown("**Date Variable Summary**")
            st.dataframe(date_df, use_container_width=True, hide_index=True)

    # ---- Step 2: Fama-French OLS ----
    with st.expander("Step 2: Fama-French OLS Regressions", expanded=False):
        data2 = load_step("step2")
        models2 = data2.get("fama_french_ols_models", pd.DataFrame())
        coefs2 = data2.get("fama_french_ols_coefficients", pd.DataFrame())

        if not models2.empty:
            c1, c2, c3 = st.columns(3)
            c1.metric("Regime Models", fmt_num(len(models2)))
            c2.metric("Unique Regimes", fmt_num(models2["regime"].nunique()) if "regime" in models2.columns else "--")
            c3.metric("Total Coefficients", fmt_num(len(coefs2)))

            st.markdown("**Model Fit Summary**")
            st.dataframe(style_model_df(models2), use_container_width=True, hide_index=True)

        if not coefs2.empty:
            st.markdown("**Coefficient Estimates**")
            st.dataframe(style_coef_df(coefs2), use_container_width=True, hide_index=True)

    # ---- Step 3: Macro Controls OLS ----
    with st.expander("Step 3: Macro Controls OLS Regressions", expanded=False):
        data3 = load_step("step3")
        models3 = data3.get("macro_controls_ols_models", pd.DataFrame())
        coefs3 = data3.get("macro_controls_ols_coefficients", pd.DataFrame())

        if not models3.empty:
            c1, c2, c3 = st.columns(3)
            specs3 = models3["model_spec"].nunique() if "model_spec" in models3.columns else 0
            regimes3 = models3["regime"].nunique() if "regime" in models3.columns else 0
            c1.metric("Model Specifications", fmt_num(specs3))
            c2.metric("Regimes", fmt_num(regimes3))
            c3.metric("Total Models", fmt_num(len(models3)))

            spec_filter = None
            if "model_spec" in models3.columns:
                spec_options = sorted(models3["model_spec"].unique().tolist())
                spec_filter = st.selectbox(
                    "Filter by model specification:",
                    ["All"] + spec_options,
                    key="step3_spec",
                )

            display_models3 = models3
            display_coefs3 = coefs3
            if spec_filter and spec_filter != "All":
                display_models3 = models3[models3["model_spec"] == spec_filter]
                if "model_spec" in coefs3.columns:
                    display_coefs3 = coefs3[coefs3["model_spec"] == spec_filter]

            st.markdown("**Model Fit Summary**")
            st.dataframe(style_model_df(display_models3), use_container_width=True, hide_index=True)

            if not display_coefs3.empty:
                st.markdown("**Coefficient Estimates**")
                st.dataframe(style_coef_df(display_coefs3), use_container_width=True, hide_index=True)

    # ---- Step 4: Breach-Level OLS ----
    with st.expander("Step 4: Breach-Level OLS Regressions", expanded=False):
        data4 = load_step("step4")
        models4 = data4.get("breach_level_ols_models", pd.DataFrame())
        coefs4 = data4.get("breach_level_ols_coefficients", pd.DataFrame())

        if not models4.empty:
            c1, c2, c3 = st.columns(3)
            specs4 = models4["model_spec"].nunique() if "model_spec" in models4.columns else 0
            regimes4 = models4["regime"].nunique() if "regime" in models4.columns else 0
            c1.metric("Model Specifications", fmt_num(specs4))
            c2.metric("Regimes", fmt_num(regimes4))
            c3.metric("Total Models", fmt_num(len(models4)))

            spec_filter4 = None
            if "model_spec" in models4.columns:
                spec_options4 = sorted(models4["model_spec"].unique().tolist())
                spec_filter4 = st.selectbox(
                    "Filter by model specification:",
                    ["All"] + spec_options4,
                    key="step4_spec",
                )

            display_models4 = models4
            display_coefs4 = coefs4
            if spec_filter4 and spec_filter4 != "All":
                display_models4 = models4[models4["model_spec"] == spec_filter4]
                if "model_spec" in coefs4.columns:
                    display_coefs4 = coefs4[coefs4["model_spec"] == spec_filter4]

            st.markdown("**Model Fit Summary**")
            st.dataframe(style_model_df(display_models4), use_container_width=True, hide_index=True)

            if not display_coefs4.empty:
                st.markdown("**Coefficient Estimates**")
                st.dataframe(style_coef_df(display_coefs4), use_container_width=True, hide_index=True)

    # ---- Step 5: Event Study ----
    with st.expander("Step 5: Event Study Analysis", expanded=False):
        data5 = load_step("step5")
        counts5 = data5.get("event_study_counts", pd.DataFrame())
        ar_day5 = data5.get("event_study_ar_by_day", pd.DataFrame())
        car5 = data5.get("event_study_car", pd.DataFrame())

        if not counts5.empty:
            row5 = counts5.iloc[0]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Events", fmt_num(row5.get("total_events")))
            c2.metric("Usable Events", fmt_num(row5.get("usable")))
            c3.metric("High Volatility", fmt_num(row5.get("high_vol")))
            c4.metric("Low Volatility", fmt_num(row5.get("low_vol")))
            c5.metric("Unique Tickers", fmt_num(row5.get("tickers_used")))

        if not ar_day5.empty:
            st.markdown("**Abnormal Returns by Event Day**")
            st.dataframe(style_coef_df(ar_day5.rename(columns={
                "regime": "Regime", "event_day": "Day", "mean_ar": "Mean AR",
                "t_stat": "t", "p_value": "p-value", "n": "N",
            })), use_container_width=True, hide_index=True)

        if not car5.empty:
            st.markdown("**Cumulative Abnormal Returns Summary**")
            display_car5 = car5.copy()
            numeric_cols = display_car5.select_dtypes(include=[np.number]).columns
            display_car5[numeric_cols] = display_car5[numeric_cols].round(4)
            if "p_value" in display_car5.columns:
                display_car5["sig"] = display_car5["p_value"].apply(fmt_sig)
            st.dataframe(display_car5, use_container_width=True, hide_index=True)

    # ---- Step 6: Sentiment Analysis ----
    with st.expander("Step 6: Sentiment Analysis", expanded=False):
        data6 = load_step("step6")
        stats6 = data6.get("sentiment_stats", pd.DataFrame())
        ar_day6 = data6.get("sentiment_ar_by_day", pd.DataFrame())
        car6 = data6.get("sentiment_car", pd.DataFrame())
        car_2x2_6 = data6.get("sentiment_car_2x2", pd.DataFrame())
        ols_model6 = data6.get("sentiment_ols_model", pd.DataFrame())
        ols_coef6 = data6.get("sentiment_ols_coefficients", pd.DataFrame())

        if not stats6.empty:
            row6 = stats6.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Events with Text", fmt_num(row6.get("n_with_text")))
            c2.metric("Negative Sentiment", fmt_num(row6.get("n_negative")))
            c3.metric("Non-Negative", fmt_num(row6.get("n_non_negative")))
            c4.metric("Mean Score", fmt_num(row6.get("mean_score"), 4) if pd.notna(row6.get("mean_score")) else "--")

        if not ar_day6.empty:
            st.markdown("**Abnormal Returns by Event Day (Sentiment Regimes)**")
            st.dataframe(style_coef_df(ar_day6.rename(columns={
                "regime": "Regime", "event_day": "Day", "mean_ar": "Mean AR",
                "t_stat": "t", "p_value": "p-value", "n": "N",
            })), use_container_width=True, hide_index=True)

        if not car6.empty:
            st.markdown("**CAR by Sentiment Regime**")
            display_car6 = car6.copy()
            numeric_cols = display_car6.select_dtypes(include=[np.number]).columns
            display_car6[numeric_cols] = display_car6[numeric_cols].round(4)
            if "p_value" in display_car6.columns:
                display_car6["sig"] = display_car6["p_value"].apply(fmt_sig)
            st.dataframe(display_car6, use_container_width=True, hide_index=True)

        if not car_2x2_6.empty:
            st.markdown("**2x2 Interaction: Sentiment x Volatility**")
            display_2x2_6 = car_2x2_6.copy()
            numeric_cols = display_2x2_6.select_dtypes(include=[np.number]).columns
            display_2x2_6[numeric_cols] = display_2x2_6[numeric_cols].round(4)
            if "p_value" in display_2x2_6.columns:
                display_2x2_6["sig"] = display_2x2_6["p_value"].apply(fmt_sig)
            st.dataframe(display_2x2_6, use_container_width=True, hide_index=True)

        if not ols_model6.empty:
            st.markdown("**OLS Model Fit**")
            st.dataframe(style_model_df(ols_model6), use_container_width=True, hide_index=True)

        if not ols_coef6.empty:
            st.markdown("**OLS Coefficient Estimates**")
            st.dataframe(style_coef_df(ols_coef6), use_container_width=True, hide_index=True)

    # ---- Step 7: Lagged Sentiment ----
    with st.expander("Step 7: Lagged Sentiment Analysis", expanded=False):
        data7 = load_step("step7")
        coverage7 = data7.get("lagged_coverage_stats", pd.DataFrame())
        dist7 = data7.get("lagged_sentiment_dist", pd.DataFrame())
        ar_day7 = data7.get("lagged_ar_by_day", pd.DataFrame())
        car7 = data7.get("lagged_car", pd.DataFrame())
        car_2x2_7 = data7.get("lagged_car_2x2", pd.DataFrame())
        ols_models7 = data7.get("lagged_ols_models", pd.DataFrame())
        ols_coefs7 = data7.get("lagged_ols_coefficients", pd.DataFrame())

        if not dist7.empty:
            row7 = dist7.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Events with Data", fmt_num(row7.get("n_with_data")))
            c2.metric("Total Events", fmt_num(row7.get("n_total")))
            c3.metric("Pct Negative", fmt_num(row7.get("pct_negative"), 1) + "%" if pd.notna(row7.get("pct_negative")) else "--")
            c4.metric("Mean Sentiment", fmt_num(row7.get("mean"), 4) if pd.notna(row7.get("mean")) else "--")

        if not coverage7.empty:
            st.markdown("**Pre-Breach News Coverage by Window**")
            st.dataframe(coverage7, use_container_width=True, hide_index=True)

        if not ar_day7.empty:
            st.markdown("**Abnormal Returns by Event Day (Lagged Sentiment Regimes)**")
            st.dataframe(style_coef_df(ar_day7.rename(columns={
                "regime": "Regime", "event_day": "Day", "mean_ar": "Mean AR",
                "t_stat": "t", "p_value": "p-value", "n": "N",
            })), use_container_width=True, hide_index=True)

        if not car7.empty:
            st.markdown("**CAR by Lagged Sentiment Regime**")
            display_car7 = car7.copy()
            numeric_cols = display_car7.select_dtypes(include=[np.number]).columns
            display_car7[numeric_cols] = display_car7[numeric_cols].round(4)
            if "p_value" in display_car7.columns:
                display_car7["sig"] = display_car7["p_value"].apply(fmt_sig)
            st.dataframe(display_car7, use_container_width=True, hide_index=True)

        if not car_2x2_7.empty:
            st.markdown("**2x2 Interaction: Lagged News Sentiment x Volatility**")
            display_2x2_7 = car_2x2_7.copy()
            numeric_cols = display_2x2_7.select_dtypes(include=[np.number]).columns
            display_2x2_7[numeric_cols] = display_2x2_7[numeric_cols].round(4)
            if "p_value" in display_2x2_7.columns:
                display_2x2_7["sig"] = display_2x2_7["p_value"].apply(fmt_sig)
            st.dataframe(display_2x2_7, use_container_width=True, hide_index=True)

        if not ols_models7.empty:
            st.markdown("**OLS Model Fit**")
            st.dataframe(style_model_df(ols_models7), use_container_width=True, hide_index=True)

        if not ols_coefs7.empty:
            st.markdown("**OLS Coefficient Estimates**")
            model_filter7 = None
            if "model_name" in ols_coefs7.columns:
                model_names7 = sorted(ols_coefs7["model_name"].unique().tolist())
                model_filter7 = st.selectbox(
                    "Filter by model:",
                    ["All"] + model_names7,
                    key="step7_model",
                )
            display_coefs7 = ols_coefs7
            if model_filter7 and model_filter7 != "All" and "model_name" in ols_coefs7.columns:
                display_coefs7 = ols_coefs7[ols_coefs7["model_name"] == model_filter7]
            st.dataframe(style_coef_df(display_coefs7), use_container_width=True, hide_index=True)

    # ---- Step 8: Repeat Offender ----
    with st.expander("Step 8: Repeat Offender Analysis", expanded=False):
        data8 = load_step("step8")
        history8 = data8.get("repeat_offender_history", pd.DataFrame())
        ar_day8 = data8.get("repeat_offender_ar_by_day", pd.DataFrame())
        car8 = data8.get("repeat_offender_car", pd.DataFrame())
        car_2x2_8 = data8.get("repeat_offender_car_2x2", pd.DataFrame())
        ols_models8 = data8.get("repeat_offender_ols_models", pd.DataFrame())
        ols_coefs8 = data8.get("repeat_offender_ols_coefficients", pd.DataFrame())

        if not history8.empty:
            n_total = len(history8)
            n_repeat = int(history8["is_repeat"].sum()) if "is_repeat" in history8.columns else 0
            n_first = n_total - n_repeat
            n_tickers = history8["ticker"].nunique() if "ticker" in history8.columns else 0
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Events", fmt_num(n_total))
            c2.metric("First Breaches", fmt_num(n_first))
            c3.metric("Repeat Breaches", fmt_num(n_repeat))
            c4.metric("Unique Tickers", fmt_num(n_tickers))

        if not ar_day8.empty:
            st.markdown("**Abnormal Returns by Event Day (First vs. Repeat)**")
            st.dataframe(style_coef_df(ar_day8.rename(columns={
                "regime": "Regime", "event_day": "Day", "mean_ar": "Mean AR",
                "t_stat": "t", "p_value": "p-value", "n": "N",
            })), use_container_width=True, hide_index=True)

        if not car8.empty:
            st.markdown("**CAR by Repeat Status**")
            display_car8 = car8.copy()
            numeric_cols = display_car8.select_dtypes(include=[np.number]).columns
            display_car8[numeric_cols] = display_car8[numeric_cols].round(4)
            if "p_value" in display_car8.columns:
                display_car8["sig"] = display_car8["p_value"].apply(fmt_sig)
            st.dataframe(display_car8, use_container_width=True, hide_index=True)

        if not car_2x2_8.empty:
            st.markdown("**2x2 Interaction: Repeat Status x Volatility**")
            display_2x2_8 = car_2x2_8.copy()
            numeric_cols = display_2x2_8.select_dtypes(include=[np.number]).columns
            display_2x2_8[numeric_cols] = display_2x2_8[numeric_cols].round(4)
            if "p_value" in display_2x2_8.columns:
                display_2x2_8["sig"] = display_2x2_8["p_value"].apply(fmt_sig)
            st.dataframe(display_2x2_8, use_container_width=True, hide_index=True)

        if not ols_models8.empty:
            st.markdown("**OLS Model Fit**")
            st.dataframe(style_model_df(ols_models8), use_container_width=True, hide_index=True)

        if not ols_coefs8.empty:
            st.markdown("**OLS Coefficient Estimates**")
            model_filter8 = None
            if "model_name" in ols_coefs8.columns:
                model_names8 = sorted(ols_coefs8["model_name"].unique().tolist())
                model_filter8 = st.selectbox(
                    "Filter by model:",
                    ["All"] + model_names8,
                    key="step8_model",
                )
            display_coefs8 = ols_coefs8
            if model_filter8 and model_filter8 != "All" and "model_name" in ols_coefs8.columns:
                display_coefs8 = ols_coefs8[ols_coefs8["model_name"] == model_filter8]
            st.dataframe(style_coef_df(display_coefs8), use_container_width=True, hide_index=True)


# =============================================================================
# TAB 2: VISUALIZATIONS
# =============================================================================

with tab_viz:
    st.header("Visualizations")
    st.markdown("Figures generated across all eight analysis steps.")
    st.divider()

    fig_index = load_figures_index()

    if fig_index.empty:
        st.info("No figures found in the database. Run the analysis pipeline to generate figures.")
    else:
        steps_with_figs = sorted(fig_index["step"].unique().tolist())

        for step_key in steps_with_figs:
            defn = STEP_DEFINITIONS.get(step_key, {"num": "?", "title": step_key})
            step_figs = fig_index[fig_index["step"] == step_key].sort_values("filename")

            with st.expander(f"Step {defn['num']}: {defn['title']} ({len(step_figs)} figures)", expanded=False):
                for _, fig_row in step_figs.iterrows():
                    fig_data = db.get_figure(fig_row["filename"])
                    if fig_data and fig_data.get("png_data"):
                        render_figure(fig_data["png_data"], caption=fig_row["label"])
                    else:
                        st.warning(f"Could not load: {fig_row['filename']}")


# =============================================================================
# TAB 3: ENRICHED DATA
# =============================================================================

with tab_enriched:
    st.header("Enriched Dataset")
    st.markdown("Cleaned and enriched breach data with financial, news, and macroeconomic variables.")
    st.divider()

    breach_df = load_breach_data()

    if breach_df.empty:
        st.info("No breach data found in the database. Run the ETL pipeline first.")
    else:
        # Summary metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Incidents", fmt_num(len(breach_df)))
        c2.metric("Variables", fmt_num(len(breach_df.columns)))
        c3.metric(
            "Tickers",
            fmt_num(breach_df["stock_ticker"].nunique()) if "stock_ticker" in breach_df.columns else "--",
        )
        c4.metric(
            "Sectors",
            fmt_num(breach_df["yf_sector"].nunique()) if "yf_sector" in breach_df.columns else "--",
        )
        c5.metric(
            "Total Affected",
            fmt_num(breach_df["total_affected"].sum()) if "total_affected" in breach_df.columns else "--",
        )

        st.divider()

        # Aggregate sub-tabs
        agg_tab1, agg_tab2, agg_tab3, agg_tab4, agg_tab5 = st.tabs([
            "By Sector", "By Year", "By Type", "Top Breaches", "News Coverage",
        ])

        with agg_tab1:
            sector_df = load_view_df("v_breaches_by_sector")
            if not sector_df.empty:
                st.dataframe(sector_df, use_container_width=True, hide_index=True, height=400)

        with agg_tab2:
            year_df = load_view_df("v_breaches_by_year")
            if not year_df.empty:
                st.dataframe(year_df, use_container_width=True, hide_index=True, height=400)

        with agg_tab3:
            type_df = load_view_df("v_breaches_by_type")
            if not type_df.empty:
                st.dataframe(type_df, use_container_width=True, hide_index=True, height=400)

        with agg_tab4:
            top_df = load_view_df("v_top_breaches")
            if not top_df.empty:
                st.dataframe(top_df, use_container_width=True, hide_index=True, height=400)

        with agg_tab5:
            news_df = load_view_df("v_news_coverage")
            if not news_df.empty:
                st.dataframe(news_df, use_container_width=True, hide_index=True, height=400)

        st.divider()

        # Filterable data explorer
        st.subheader("Data Explorer")

        filter_c1, filter_c2, filter_c3 = st.columns(3)

        with filter_c1:
            sector_options = ["All"]
            if "yf_sector" in breach_df.columns:
                sector_options += sorted(breach_df["yf_sector"].dropna().unique().tolist())
            selected_sector = st.selectbox("Sector", sector_options, key="enriched_sector")

        with filter_c2:
            type_options = ["All"]
            if "breach_type" in breach_df.columns:
                type_options += sorted(breach_df["breach_type"].dropna().unique().tolist())
            selected_type = st.selectbox("Breach Type", type_options, key="enriched_type")

        with filter_c3:
            org_search = st.text_input("Search Organization", "", key="enriched_org")

        # Apply filters
        filtered = breach_df.copy()
        if selected_sector != "All" and "yf_sector" in filtered.columns:
            filtered = filtered[filtered["yf_sector"] == selected_sector]
        if selected_type != "All" and "breach_type" in filtered.columns:
            filtered = filtered[filtered["breach_type"] == selected_type]
        if org_search and "org_name" in filtered.columns:
            filtered = filtered[
                filtered["org_name"].str.contains(org_search, case=False, na=False)
            ]

        st.markdown(f"**{len(filtered):,} records**")

        # Default column view
        display_cols = [c for c in DEFAULT_EXPLORER_COLS if c in filtered.columns]
        st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True, height=500)

        # Full dataset in expander
        with st.expander("View All Columns", expanded=False):
            st.dataframe(filtered, use_container_width=True, hide_index=True, height=500)


# =============================================================================
# TAB 4: RAW DATA
# =============================================================================

with tab_raw:
    st.header("Raw Data Files")
    st.markdown("Original CSV files before processing.")
    st.divider()

    # Collect available files
    raw_files = {}
    if RAW_CSV.exists():
        raw_files[RAW_CSV.name] = RAW_CSV
    if CLEANED_DIR.exists():
        for f in sorted(CLEANED_DIR.glob("*.csv")):
            raw_files[f"cleaned/{f.name}"] = f

    if not raw_files:
        st.info("No data files found.")
    else:
        selected_name = st.selectbox("Select file:", list(raw_files.keys()), key="raw_file")
        selected_path = raw_files[selected_name]

        # File metadata
        file_size = selected_path.stat().st_size
        try:
            try:
                df_raw = pd.read_csv(selected_path, encoding="utf-8")
            except UnicodeDecodeError:
                df_raw = pd.read_csv(selected_path, encoding="latin-1")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("File", selected_name)
            c2.metric("Size", f"{file_size / 1024:.1f} KB" if file_size < 1_048_576 else f"{file_size / 1_048_576:.2f} MB")
            c3.metric("Rows", fmt_num(len(df_raw)))
            c4.metric("Columns", fmt_num(len(df_raw.columns)))

            with st.expander("Column Information", expanded=False):
                col_info = pd.DataFrame({
                    "Column": df_raw.columns,
                    "Type": df_raw.dtypes.astype(str),
                    "Non-Null": df_raw.notna().sum().values,
                    "Null %": (df_raw.isna().sum() / len(df_raw) * 100).round(2).values,
                })
                st.dataframe(col_info, use_container_width=True, hide_index=True)

            st.dataframe(df_raw, use_container_width=True, hide_index=True, height=500)

        except Exception as e:
            st.error(f"Error reading file: {e}")


# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.markdown(
    f"""<div style='text-align: center; font-family: Source Serif 4, Georgia, serif;
    color: {NAVY}; padding: 1rem 0; font-size: 0.85rem;'>
    Broken Boundaries | Ashley D. Roseboro, Abigail Hagood-Dokter, Timothy Spivey | 2025
    </div>""",
    unsafe_allow_html=True,
)
