"""
visual.py - Publication-Quality Figure Generation
Generates 21 PNG figures from the analysis pipeline results stored in SQLite.
All data is pulled from database.py (enforcing the pipeline flow).
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

import database as db

# === Configuration ===
FIGURES_DIR = Path(__file__).parent / "figures"
DPI = 300

REGIME_COLORS = {
    'pooled': '#2c3e50',
    'high_vol': '#e74c3c',
    'low_vol': '#2980b9',
}

SENTIMENT_COLORS = {
    'negative': '#e74c3c',
    'non_negative': '#2980b9',
}

NEWS_COLORS = {
    'negative_news': '#e74c3c',
    'non_negative_news': '#2980b9',
}

REPEAT_COLORS = {
    'first': '#2980b9',
    'repeat': '#e74c3c',
}


def _setup_style():
    """Configure matplotlib defaults for publication quality."""
    sns.set_style('whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 100,
        'savefig.dpi': DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.15,
    })


_CURRENT_STEP = None   # set by main() before each step group
_CURRENT_LABEL = None  # set by main() before each plot function
_PENDING_FIGURES = []  # (filename, step, label, path) tuples for batch DB store


def _save(fig, filename):
    """Save figure to FIGURES_DIR and queue it for database storage."""
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path)
    plt.close(fig)
    _PENDING_FIGURES.append((filename, _CURRENT_STEP, _CURRENT_LABEL, str(path)))
    return path


def _sig_stars(p):
    """Return significance stars for a p-value."""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return ''


# =============================================================================
# SHARED HELPERS
# =============================================================================

def _plot_forest(df, var_col, coef_col, ci_lower_col, ci_upper_col, p_col,
                 group_col=None, title='', xlabel='Coefficient',
                 filename='forest.png', figsize=(8, 5)):
    """
    Forest plot: coefficient + 95% CI with significance stars.
    If group_col is provided, colors rows by group (e.g., regime).
    """
    fig, ax = plt.subplots(figsize=figsize)

    if group_col and group_col in df.columns:
        groups = df[group_col].unique()
        offsets = np.linspace(-0.2, 0.2, len(groups))
        variables = df[var_col].unique()
        yticks = np.arange(len(variables))

        for i, group in enumerate(groups):
            sub = df[df[group_col] == group]
            color = REGIME_COLORS.get(group, f'C{i}')
            for _, row in sub.iterrows():
                vi = np.where(variables == row[var_col])[0]
                if len(vi) == 0:
                    continue
                y = vi[0] + offsets[i]
                ax.errorbar(row[coef_col], y,
                            xerr=[[row[coef_col] - row[ci_lower_col]],
                                  [row[ci_upper_col] - row[coef_col]]],
                            fmt='o', color=color, capsize=3, markersize=5,
                            label=group if vi[0] == 0 else None)
                stars = _sig_stars(row[p_col])
                if stars:
                    ax.annotate(stars, (row[ci_upper_col], y),
                                fontsize=8, va='center', ha='left',
                                xytext=(3, 0), textcoords='offset points')

        ax.set_yticks(yticks)
        ax.set_yticklabels(variables)
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        unique_handles, unique_labels = [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen[l] = True
                unique_handles.append(h)
                unique_labels.append(l)
        ax.legend(unique_handles, unique_labels, loc='best')
    else:
        yticks = np.arange(len(df))
        ax.errorbar(df[coef_col], yticks,
                     xerr=[df[coef_col] - df[ci_lower_col],
                           df[ci_upper_col] - df[coef_col]],
                     fmt='o', color=REGIME_COLORS['pooled'], capsize=3, markersize=5)
        for i, (_, row) in enumerate(df.iterrows()):
            stars = _sig_stars(row[p_col])
            if stars:
                ax.annotate(stars, (row[ci_upper_col], i),
                            fontsize=8, va='center', ha='left',
                            xytext=(3, 0), textcoords='offset points')
        ax.set_yticks(yticks)
        ax.set_yticklabels(df[var_col])

    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    return _save(fig, filename)


def _plot_2x2_heatmap(df, row_col, col_col, val_col, p_col, n_col,
                      title='', filename='heatmap.png', figsize=(6, 4)):
    """
    Annotated 2x2 heatmap: cells show mean_car, significance stars, and n.
    """
    pivot_val = df.pivot(index=row_col, columns=col_col, values=val_col)
    pivot_p = df.pivot(index=row_col, columns=col_col, values=p_col)
    pivot_n = df.pivot(index=row_col, columns=col_col, values=n_col)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(pivot_val, annot=False, cmap='RdBu_r', center=0, ax=ax,
                linewidths=1, linecolor='white', cbar_kws={'label': 'Mean CAR (%)'})

    for i in range(pivot_val.shape[0]):
        for j in range(pivot_val.shape[1]):
            val = pivot_val.iloc[i, j]
            p = pivot_p.iloc[i, j]
            n = pivot_n.iloc[i, j]
            stars = _sig_stars(p)
            text = f'{val:.3f}{stars}\nn={int(n)}'
            ax.text(j + 0.5, i + 0.5, text,
                    ha='center', va='center', fontsize=9, fontweight='bold')

    ax.set_title(title)
    ax.set_ylabel('')
    ax.set_xlabel('')
    fig.tight_layout()
    return _save(fig, filename)


def _plot_ar_by_day(df, regime_col, colors, title='', filename='ar_by_day.png',
                    figsize=(8, 4.5)):
    """
    Line chart of mean AR by event_day. Filled markers where p < 0.05.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for regime in df[regime_col].unique():
        sub = df[df[regime_col] == regime].sort_values('event_day')
        color = colors.get(regime, '#333333')
        ax.plot(sub['event_day'], sub['mean_ar'], '-o', color=color,
                markersize=5, label=regime, linewidth=1.5, alpha=0.85)

        sig = sub[sub['p_value'] < 0.05]
        if not sig.empty:
            ax.scatter(sig['event_day'], sig['mean_ar'],
                       color=color, s=60, zorder=5, edgecolors='black',
                       linewidths=0.8, marker='D')

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Event Day')
    ax.set_ylabel('Mean Abnormal Return (%)')
    ax.set_title(title)
    ax.legend(loc='best')
    fig.tight_layout()
    return _save(fig, filename)


# =============================================================================
# STEP 1: DESCRIPTIVE STATISTICS (3 figs)
# =============================================================================

def plot_breach_timeline():
    """1.1 Dual-axis: bar (count) + line (total_affected) by year."""
    df = db.get_yearly_breakdown()
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.bar(df['year'], df['breach_count'], color='#2c3e50', alpha=0.7,
            label='Breach Count')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Number of Breaches', color='#2c3e50')
    ax1.tick_params(axis='y', labelcolor='#2c3e50')

    ax2 = ax1.twinx()
    ax2.plot(df['year'], df['total_affected'], 'o-', color='#e74c3c',
             linewidth=2, markersize=5, label='Total Affected')
    ax2.set_ylabel('Total Individuals Affected', color='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'{x / 1e6:.0f}M' if x >= 1e6 else f'{x / 1e3:.0f}K'))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax1.set_title('Data Breach Timeline: Count and Scale')
    fig.tight_layout()
    return _save(fig, 'fig_1_1_breach_timeline.png')


def plot_sector_distribution():
    """1.2 Horizontal bar sorted by breach_count."""
    df = db.get_sector_breakdown().sort_values('breach_count', ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(df['sector'], df['breach_count'], color='#2c3e50', alpha=0.8)
    ax.set_xlabel('Number of Breaches')
    ax.set_title('Breach Distribution by Sector')

    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row['breach_count'] + 2, i, str(int(row['breach_count'])),
                va='center', fontsize=8)

    fig.tight_layout()
    return _save(fig, 'fig_1_2_sector_distribution.png')


def plot_numeric_distributions():
    """1.3 2x3 grid of five-number summaries for key numeric columns."""
    results = db.get_descriptive_results()
    df = results['descriptive_numeric']

    key_cols = ['total_affected', 'yf_market_cap', 'yf_beta',
                'yf_pe_ratio', 'total_news_count', 'vix_at_breach']
    df = df[df['column_name'].isin(key_cols)]

    if df.empty:
        return None

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()

    for i, (_, row) in enumerate(df.iterrows()):
        if i >= 6:
            break
        ax = axes[i]
        stats = [row['min'], row['q25'], row['median'], row['q75'], row['max']]
        bp = ax.boxplot([stats], vert=True, patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor('#2980b9')
        bp['boxes'][0].set_alpha(0.6)
        ax.set_title(row['column_name'].replace('_', ' ').title(), fontsize=10)
        ax.set_xticklabels([])
        ax.text(0.95, 0.95, f'n={int(row["count"])}\nskew={row["skewness"]:.1f}',
                transform=ax.transAxes, fontsize=7, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    for j in range(i + 1, 6):
        axes[j].set_visible(False)

    fig.suptitle('Numeric Variable Distributions', fontsize=13, y=1.01)
    fig.tight_layout()
    return _save(fig, 'fig_1_3_numeric_distributions.png')


# =============================================================================
# STEP 2: FAMA-FRENCH OLS (2 figs)
# =============================================================================

def plot_ff_coefficients():
    """2.1 Forest plot: FF factor coefficients by regime."""
    results = db.get_fama_french_results()
    df = results['fama_french_ols_coefficients']
    df = df[df['variable'] != 'const']

    return _plot_forest(
        df, var_col='variable', coef_col='coefficient',
        ci_lower_col='ci_lower', ci_upper_col='ci_upper', p_col='p_value',
        group_col='regime',
        title='Fama-French 5-Factor Coefficients by Volatility Regime',
        filename='fig_2_1_ff_coefficients.png',
    )


def plot_ff_model_fit():
    """2.2 Grouped bar: R-squared by regime."""
    results = db.get_fama_french_results()
    df = results['fama_french_ols_models']

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(df))
    colors = [REGIME_COLORS.get(r, '#333') for r in df['regime']]
    bars = ax.bar(x, df['r_squared'], color=colors, alpha=0.8, width=0.5)

    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{row["r_squared"]:.4f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(df['regime'])
    ax.set_ylabel('R-squared')
    ax.set_title('Fama-French OLS Model Fit by Regime')
    ax.set_ylim(0, max(df['r_squared']) * 1.2)
    fig.tight_layout()
    return _save(fig, 'fig_2_2_ff_model_fit.png')


# =============================================================================
# STEP 3: MACRO CONTROLS OLS (2 figs)
# =============================================================================

def plot_macro_rsquared_comparison():
    """3.1 Heatmap: R-squared rows=model_spec, cols=regime."""
    results = db.get_macro_controls_results()
    df = results['macro_controls_ols_models']
    pivot = df.pivot(index='model_spec', columns='regime', values='r_squared')

    spec_order = ['base', 'inflation', 'gdp', 'unemployment', 'interest_rates', 'all_controls']
    pivot = pivot.reindex([s for s in spec_order if s in pivot.index])

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax,
                linewidths=0.5, linecolor='white')
    ax.set_title('Macro Controls: R-squared by Model Specification and Regime')
    ax.set_ylabel('Model Specification')
    ax.set_xlabel('Regime')
    fig.tight_layout()
    return _save(fig, 'fig_3_1_macro_rsquared.png')


def plot_macro_coefficient_comparison():
    """3.2 Forest plot: macro variables from all_controls spec by regime."""
    results = db.get_macro_controls_results()
    df = results['macro_controls_ols_coefficients']
    df = df[(df['model_spec'] == 'all_controls') & (df['variable'] != 'const')]

    return _plot_forest(
        df, var_col='variable', coef_col='coefficient',
        ci_lower_col='ci_lower', ci_upper_col='ci_upper', p_col='p_value',
        group_col='regime',
        title='Macro Control Coefficients (All Controls Specification)',
        filename='fig_3_2_macro_coefficients.png',
        figsize=(9, 6),
    )


# =============================================================================
# STEP 4: BREACH-LEVEL OLS (2 figs)
# =============================================================================

def plot_breach_level_rsquared():
    """4.1 Grouped bar: R-squared across specs by regime."""
    results = db.get_breach_level_results()
    df = results['breach_level_ols_models']

    specs = df['model_spec'].unique()
    regimes = df['regime'].unique()
    x = np.arange(len(specs))
    width = 0.8 / len(regimes)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, regime in enumerate(regimes):
        sub = df[df['regime'] == regime]
        sub = sub.set_index('model_spec').reindex(specs)
        offset = (i - len(regimes) / 2 + 0.5) * width
        bars = ax.bar(x + offset, sub['r_squared'], width,
                      label=regime, color=REGIME_COLORS.get(regime, f'C{i}'),
                      alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(specs, rotation=15)
    ax.set_ylabel('R-squared')
    ax.set_title('Breach-Level OLS: R-squared by Model and Regime')
    ax.legend()
    fig.tight_layout()
    return _save(fig, 'fig_4_1_breach_level_rsquared.png')


def plot_breach_level_coefficients():
    """4.2 Forest plot: variables from stock_macro spec."""
    results = db.get_breach_level_results()
    df = results['breach_level_ols_coefficients']
    df = df[(df['model_spec'] == 'stock_macro') & (df['variable'] != 'const')]

    return _plot_forest(
        df, var_col='variable', coef_col='coefficient',
        ci_lower_col='ci_lower', ci_upper_col='ci_upper', p_col='p_value',
        group_col='regime',
        title='Breach-Level Coefficients (Stock + Macro Specification)',
        filename='fig_4_2_breach_level_coefficients.png',
        figsize=(9, 7),
    )


# =============================================================================
# STEP 5: EVENT STUDY (3 figs)
# =============================================================================

def plot_event_study_ar():
    """5.1 AR line chart by regime with significance markers."""
    results = db.get_event_study_results()
    df = results['event_study_ar_by_day']
    return _plot_ar_by_day(
        df, regime_col='regime', colors=REGIME_COLORS,
        title='Abnormal Returns Around Breach Disclosure',
        filename='fig_5_1_event_study_ar.png',
    )


def plot_event_study_car():
    """5.2 Bar chart: mean CAR with SE error bars by regime."""
    results = db.get_event_study_results()
    df = results['event_study_car']

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(df))
    colors = [REGIME_COLORS.get(r, '#333') for r in df['regime']]
    se = df['std_car'] / np.sqrt(df['n'])

    bars = ax.bar(x, df['mean_car'], color=colors, alpha=0.8, width=0.5,
                  yerr=se, capsize=5, error_kw={'linewidth': 1.2})

    for i, (_, row) in enumerate(df.iterrows()):
        stars = _sig_stars(row['p_value'])
        label = f'{row["mean_car"]:.3f}{stars}'
        y = row['mean_car']
        offset = -0.02 if y < 0 else 0.02
        ax.text(i, y + offset, label, ha='center',
                va='top' if y < 0 else 'bottom', fontsize=9)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df['regime'])
    ax.set_ylabel('Cumulative Abnormal Return (%)')
    ax.set_title('Mean CAR [0, +10] by Volatility Regime')
    fig.tight_layout()
    return _save(fig, 'fig_5_2_event_study_car.png')


def plot_event_study_ar_distribution():
    """5.3 Violin plot: day-0 AR distribution by vol regime."""
    results = db.get_event_study_results()
    ar_df = results['event_study_ar']
    day0 = ar_df[ar_df['event_day'] == 0].copy()

    if day0.empty:
        return None

    day0['regime'] = day0['high_vol'].map({1: 'high_vol', 0: 'low_vol'})

    fig, ax = plt.subplots(figsize=(6, 5))
    palette = {'high_vol': REGIME_COLORS['high_vol'], 'low_vol': REGIME_COLORS['low_vol']}
    sns.violinplot(data=day0, x='regime', y='ar', hue='regime',
                   palette=palette, ax=ax, inner='quartile', cut=0, legend=False)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Volatility Regime')
    ax.set_ylabel('Abnormal Return (%) on Event Day 0')
    ax.set_title('Distribution of Day-0 Abnormal Returns')
    fig.tight_layout()
    return _save(fig, 'fig_5_3_event_study_ar_dist.png')


# =============================================================================
# STEP 6: SENTIMENT ANALYSIS (3 figs)
# =============================================================================

def plot_sentiment_ar_by_day():
    """6.1 AR line: negative vs non_negative sentiment regimes."""
    results = db.get_sentiment_results()
    df = results['sentiment_ar_by_day']
    return _plot_ar_by_day(
        df, regime_col='regime', colors=SENTIMENT_COLORS,
        title='Abnormal Returns by Sentiment Regime',
        filename='fig_6_1_sentiment_ar.png',
    )


def plot_sentiment_car_2x2():
    """6.2 2x2 heatmap: sentiment_regime x vol_regime."""
    results = db.get_sentiment_results()
    df = results['sentiment_car_2x2']
    return _plot_2x2_heatmap(
        df, row_col='sentiment_regime', col_col='vol_regime',
        val_col='mean_car', p_col='p_value', n_col='n',
        title='Mean CAR: Sentiment x Volatility Regime',
        filename='fig_6_2_sentiment_car_2x2.png',
    )


def plot_sentiment_ols_coefficients():
    """6.3 Forest plot: sentiment model variables."""
    results = db.get_sentiment_results()
    df = results['sentiment_ols_coefficients']

    return _plot_forest(
        df, var_col='variable', coef_col='coefficient',
        ci_lower_col='ci_lower', ci_upper_col='ci_upper', p_col='p_value',
        title='Sentiment OLS Coefficients',
        filename='fig_6_3_sentiment_ols.png',
    )


# =============================================================================
# STEP 7: LAGGED SENTIMENT (3 figs)
# =============================================================================

def plot_lagged_coverage():
    """7.1 Bar: coverage_pct and mean_articles by window."""
    results = db.get_lagged_sentiment_results()
    df = results['lagged_coverage_stats'].sort_values('window')

    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    x = np.arange(len(df))
    ax1.bar(x, df['coverage_pct'], color='#2c3e50', alpha=0.7, width=0.4,
            label='Coverage %')
    ax1.set_ylabel('Coverage (%)', color='#2c3e50')
    ax1.tick_params(axis='y', labelcolor='#2c3e50')

    ax2 = ax1.twinx()
    ax2.bar(x + 0.4, df['mean_articles'], color='#e74c3c', alpha=0.7, width=0.4,
            label='Mean Articles')
    ax2.set_ylabel('Mean Articles per Event', color='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')

    ax1.set_xticks(x + 0.2)
    ax1.set_xticklabels([f'{int(w)}d' for w in df['window']])
    ax1.set_xlabel('Pre-Breach Window')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax1.set_title('Pre-Breach News Coverage by Window Size')
    fig.tight_layout()
    return _save(fig, 'fig_7_1_lagged_coverage.png')


def plot_lagged_car_2x2():
    """7.2 2x2 heatmap: news_regime x vol_regime."""
    results = db.get_lagged_sentiment_results()
    df = results['lagged_car_2x2']
    return _plot_2x2_heatmap(
        df, row_col='news_regime', col_col='vol_regime',
        val_col='mean_car', p_col='p_value', n_col='n',
        title='Mean CAR: Pre-Breach News Sentiment x Volatility Regime',
        filename='fig_7_2_lagged_car_2x2.png',
    )


def plot_lagged_ols_comparison():
    """7.3 Two-panel: R-squared bars + key coefficient forest."""
    results = db.get_lagged_sentiment_results()
    models_df = results['lagged_ols_models']
    coefs_df = results['lagged_ols_coefficients']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                    gridspec_kw={'width_ratios': [1, 1.5]})

    # Left panel: R-squared bars
    x = np.arange(len(models_df))
    ax1.barh(x, models_df['r_squared'], color='#2c3e50', alpha=0.8, height=0.5)
    for i, (_, row) in enumerate(models_df.iterrows()):
        ax1.text(row['r_squared'] + 0.002, i,
                 f'{row["r_squared"]:.4f}', va='center', fontsize=9)
    ax1.set_yticks(x)
    ax1.set_yticklabels(models_df['model_name'])
    ax1.set_xlabel('R-squared')
    ax1.set_title('Model Fit Comparison')
    ax1.invert_yaxis()

    # Right panel: key coefficients from model_c (most complete)
    key_df = coefs_df[(coefs_df['model_name'] == 'model_c') &
                      (coefs_df['variable'] != 'const')]
    if not key_df.empty:
        yticks = np.arange(len(key_df))
        ax2.errorbar(key_df['coefficient'], yticks,
                     xerr=[key_df['coefficient'] - key_df['ci_lower'],
                           key_df['ci_upper'] - key_df['coefficient']],
                     fmt='o', color='#2c3e50', capsize=3, markersize=5)
        for i, (_, row) in enumerate(key_df.iterrows()):
            stars = _sig_stars(row['p_value'])
            if stars:
                ax2.annotate(stars, (row['ci_upper'], i),
                             fontsize=8, va='center', ha='left',
                             xytext=(3, 0), textcoords='offset points')
        ax2.set_yticks(yticks)
        ax2.set_yticklabels(key_df['variable'])
        ax2.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
        ax2.set_xlabel('Coefficient')
        ax2.set_title('Model C Coefficients')
        ax2.invert_yaxis()

    fig.suptitle('Lagged News Sentiment OLS Analysis', fontsize=13, y=1.01)
    fig.tight_layout()
    return _save(fig, 'fig_7_3_lagged_ols.png')


# =============================================================================
# STEP 8: REPEAT OFFENDER (3 figs)
# =============================================================================

def plot_repeat_offender_ar():
    """8.1 AR line: first vs repeat by event_day."""
    results = db.get_repeat_offender_results()
    df = results['repeat_offender_ar_by_day']
    return _plot_ar_by_day(
        df, regime_col='regime', colors=REPEAT_COLORS,
        title='Abnormal Returns: First vs Repeat Breaches',
        filename='fig_8_1_repeat_ar.png',
    )


def plot_repeat_car_2x2():
    """8.2 2x2 heatmap: repeat_regime x vol_regime."""
    results = db.get_repeat_offender_results()
    df = results['repeat_offender_car_2x2']
    return _plot_2x2_heatmap(
        df, row_col='repeat_regime', col_col='vol_regime',
        val_col='mean_car', p_col='p_value', n_col='n',
        title='Mean CAR: Repeat Offender x Volatility Regime',
        filename='fig_8_2_repeat_car_2x2.png',
    )


def plot_repeat_ols_coefficients():
    """8.3 Forest plot: key variables across repeat offender OLS models."""
    results = db.get_repeat_offender_results()
    df = results['repeat_offender_ols_coefficients']
    df = df[df['variable'] != 'const']

    return _plot_forest(
        df, var_col='variable', coef_col='coefficient',
        ci_lower_col='ci_lower', ci_upper_col='ci_upper', p_col='p_value',
        group_col='model_name',
        title='Repeat Offender OLS Coefficients by Model',
        filename='fig_8_3_repeat_ols.png',
        figsize=(9, 6),
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Generate all 21 figures and store them in the database."""
    global _CURRENT_STEP, _CURRENT_LABEL
    _setup_style()
    FIGURES_DIR.mkdir(exist_ok=True)
    _PENDING_FIGURES.clear()

    steps = [
        ("step1", "Step 1: Descriptive Statistics", [
            ("1.1 Breach Timeline", plot_breach_timeline),
            ("1.2 Sector Distribution", plot_sector_distribution),
            ("1.3 Numeric Distributions", plot_numeric_distributions),
        ]),
        ("step2", "Step 2: Fama-French OLS", [
            ("2.1 FF Coefficients", plot_ff_coefficients),
            ("2.2 FF Model Fit", plot_ff_model_fit),
        ]),
        ("step3", "Step 3: Macro Controls OLS", [
            ("3.1 Macro R-squared", plot_macro_rsquared_comparison),
            ("3.2 Macro Coefficients", plot_macro_coefficient_comparison),
        ]),
        ("step4", "Step 4: Breach-Level OLS", [
            ("4.1 Breach-Level R-squared", plot_breach_level_rsquared),
            ("4.2 Breach-Level Coefficients", plot_breach_level_coefficients),
        ]),
        ("step5", "Step 5: Event Study", [
            ("5.1 Event Study AR", plot_event_study_ar),
            ("5.2 Event Study CAR", plot_event_study_car),
            ("5.3 AR Distribution", plot_event_study_ar_distribution),
        ]),
        ("step6", "Step 6: Sentiment Analysis", [
            ("6.1 Sentiment AR", plot_sentiment_ar_by_day),
            ("6.2 Sentiment CAR 2x2", plot_sentiment_car_2x2),
            ("6.3 Sentiment OLS", plot_sentiment_ols_coefficients),
        ]),
        ("step7", "Step 7: Lagged Sentiment", [
            ("7.1 Lagged Coverage", plot_lagged_coverage),
            ("7.2 Lagged CAR 2x2", plot_lagged_car_2x2),
            ("7.3 Lagged OLS", plot_lagged_ols_comparison),
        ]),
        ("step8", "Step 8: Repeat Offender", [
            ("8.1 Repeat AR", plot_repeat_offender_ar),
            ("8.2 Repeat CAR 2x2", plot_repeat_car_2x2),
            ("8.3 Repeat OLS", plot_repeat_ols_coefficients),
        ]),
    ]

    total = sum(len(figs) for _, _, figs in steps)
    count = 0

    for step_key, step_name, figs in steps:
        _CURRENT_STEP = step_key
        print(f"\n{'=' * 50}")
        print(f"  {step_name}")
        print(f"{'=' * 50}")
        for label, func in figs:
            count += 1
            _CURRENT_LABEL = label
            try:
                path = func()
                status = f"-> {path}" if path else "SKIPPED (no data)"
                print(f"  [{count}/{total}] {label}: {status}")
            except Exception as e:
                print(f"  [{count}/{total}] {label}: ERROR - {e}")

    # Store all generated figures into the database
    if _PENDING_FIGURES:
        print(f"\n{'=' * 50}")
        print(f"  Storing {len(_PENDING_FIGURES)} figures in database...")
        print(f"{'=' * 50}")
        db.store_figures_batch(_PENDING_FIGURES)
        total_bytes = sum(Path(p).stat().st_size for _, _, _, p in _PENDING_FIGURES)
        print(f"  -> figures table: {len(_PENDING_FIGURES)} rows "
              f"({total_bytes / 1024:.0f} KB total)")

    print(f"\n{'=' * 50}")
    print(f"  COMPLETE: {count} figures processed")
    print(f"  Output: {FIGURES_DIR.resolve()}")
    print(f"  Database: {db.get_db_path()}")
    print(f"{'=' * 50}\n")


if __name__ == '__main__':
    main()
