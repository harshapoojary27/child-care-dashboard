"""
main_editorial.py
==================
HHS Unaccompanied Children Program — Data Analysis & Visualization Dashboard
MCA Major Project | Built with Streamlit + Plotly + Scikit-learn

Layout  : Pattern 3 — Editorial / Newspaper (no sidebar, masthead header,
          ticker KPI ribbon, 3-column newspaper grid on Home page)
Logic   : All 6 pages from original main.py — unchanged data handling,
          cleaning pipeline, 9 chart types, and ML prediction models.

Run with:
    streamlit run main_editorial.py

Pages
-----
1. Home             – Masthead + KPI ticker + 3-column newspaper layout
2. Data Overview    – Raw table + summary statistics + dataset info
3. Data Cleaning    – 7-step pipeline report + before/after + download
4. Visualizations   – 9 interactive chart types via dropdown
5. Insights         – Statistical findings + year-over-year analysis
6. Prediction       – Linear / Polynomial / Rolling Average ML forecast
"""

import os
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ── Custom modules ─────────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "modules"))

from data_loader import (
    load_data, clean_data, get_summary_stats,
    filter_by_date, get_monthly_summary, get_yearly_summary,
    COLUMN_LABELS, NUMERIC_COLS,
)
import visualizations as viz
import ml_models as ml

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="The Migration Record — HHS Dashboard",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",   # sidebar hidden; editorial nav used
)

# ══════════════════════════════════════════════════════════════════════════════
# EDITORIAL CSS  (Pattern 3 — Newspaper / Playfair Display)
# Replaces the original sidebar + Inter CSS entirely.
# All logic below this block is unchanged from main.py.
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800;900&family=Source+Sans+3:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
    background: #FAFAF8;
    color: #1A1A1A;
}

/* Hide Streamlit sidebar and header completely */
[data-testid="stSidebar"],
[data-testid="collapsedControl"],
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
#MainMenu,
header,
footer { display: none !important; }

/* Remove the top gap Streamlit adds when header is hidden */
.block-container {
    padding-top: 0 !important;
    padding-bottom: 1rem !important;
}
section[data-testid="stMain"] > div:first-child {
    padding-top: 0 !important;
}

[data-testid="stAppViewContainer"] { background: #FAFAF8; }

/* ── Masthead ── */
.masthead {
    border-top: 5px solid #1A1A1A;
    border-bottom: 1px solid #1A1A1A;
    padding: 14px 24px;
    margin: -1rem -1rem 0 -1rem;
    display: flex; align-items: center; justify-content: space-between;
    background: white;
}
.masthead-name {
    font-family: 'Playfair Display', serif;
    font-size: 26px; font-weight: 900;
    color: #1A1A1A; letter-spacing: -0.02em;
}
.masthead-tagline { font-size: 11px; color: #777; font-style: italic; }
.masthead-meta    { font-size: 11px; color: #777; text-align: right; }

/* ── Black navigation strip ── */
.nav-strip {
    background: #1A1A1A;
    margin: 0 -1rem 20px -1rem;
    padding: 0 24px;
    display: flex; gap: 0;
}
.nav-strip-item {
    color: #AAAAAA; font-size: 12px; font-weight: 600;
    padding: 10px 16px; cursor: pointer;
    letter-spacing: 0.04em; text-transform: uppercase;
    border-bottom: 3px solid transparent;
}
.nav-strip-item.active { color: white; border-bottom-color: white; }

/* ── KPI ticker ribbon ── */
.ticker {
    background: #F3F4F6;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    padding: 12px 0;
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    margin-bottom: 24px;
    text-align: center;
}
.ticker-val {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem; font-weight: 900; color: #1A1A1A;
}
.ticker-lbl {
    font-size: 10px; color: #9CA3AF;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-top: 2px; font-weight: 600;
}
.ticker-sep { border-right: 1px solid #E5E7EB; }

/* ── Newspaper column headline rule ── */
.headline-block {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem; font-weight: 700;
    color: #1A1A1A; line-height: 1.3;
    border-bottom: 2px solid #1A1A1A;
    padding-bottom: 8px; margin-bottom: 14px;
}
.headline-block.sm {
    font-size: 0.95rem;
    border-bottom-width: 1px;
}

/* ── Story card (right column briefs) ── */
.story-card {
    padding: 16px 0;
    border-bottom: 1px solid #E5E7EB;
    margin-bottom: 12px;
}
.story-eyebrow {
    font-size: 10px; color: #6B7280;
    text-transform: uppercase; letter-spacing: 0.12em;
    font-weight: 700; margin-bottom: 4px;
}
.story-title {
    font-family: 'Playfair Display', serif;
    font-size: 15px; font-weight: 700;
    color: #1A1A1A; line-height: 1.35; margin-bottom: 5px;
}
.story-body { font-size: 12.5px; color: #6B7280; line-height: 1.6; }

/* ── Chart card (white bordered newspaper box) ── */
.chart-card {
    background: white;
    border: 1px solid #E5E7EB;
    padding: 18px; margin-bottom: 14px;
}
.chart-title {
    font-family: 'Playfair Display', serif;
    font-size: 14px; font-weight: 700; color: #1A1A1A; margin-bottom: 3px;
}
.chart-sub { font-size: 11px; color: #9CA3AF; margin-bottom: 12px; }

/* ── Section header (replaces original .section-header) ── */
.section-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem; font-weight: 700; color: #1A1A1A;
    border-bottom: 2px solid #1A1A1A;
    padding-bottom: 6px; margin-bottom: 16px;
}

/* ── Insight box (Insights page panels) ── */
.insight-box {
    background: white;
    border: 1px solid #E5E7EB;
    border-left: 4px solid #1A1A1A;
    padding: 14px 18px; margin-bottom: 12px;
}
.insight-box h4 {
    font-family: 'Playfair Display', serif;
    margin: 0 0 6px; color: #1A1A1A; font-size: 0.95rem;
}
.insight-box p  { margin: 0; color: #6B7280; font-size: 0.88rem; line-height: 1.6; }

/* ── Streamlit metric cards ── */
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E5E7EB;
    padding: 12px 16px;
}
[data-testid="stMetricValue"] {
    color: #1A1A1A !important;
    font-family: 'Playfair Display', serif !important;
}
[data-testid="stMetricLabel"] { color: #9CA3AF !important; font-size: 11px !important; }
[data-testid="stMetricDelta"] svg { display: none; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"]      { border-radius: 0; }

/* ── Data table ── */
[data-testid="stDataFrame"] { border: 1px solid #E5E7EB; border-radius: 0; }

[data-testid="stMarkdownContainer"] p { color: #6B7280; }
[data-testid="stMarkdownContainer"] strong { color: #1A1A1A; }
hr { border-color: #E5E7EB !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING  (unchanged from main.py)
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_CSV = os.path.join(os.path.dirname(__file__), "data", "hhs_children_data.csv")


@st.cache_data(show_spinner="Loading and cleaning data …")
def get_clean_data(path_or_bytes, is_bytes=False):
    if is_bytes:
        try:
            raw_df = pd.read_csv(io.BytesIO(path_or_bytes))
        except Exception:
            raw_df = pd.read_excel(io.BytesIO(path_or_bytes))
    else:
        raw_df = load_data(path_or_bytes)
    df_clean, report = clean_data(raw_df)
    return raw_df, df_clean, report


# ── Masthead ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="masthead">
    <div>
        <div class="masthead-name">The Migration Record</div>
        <div class="masthead-tagline">HHS Unaccompanied Children Program — Data Intelligence Dashboard</div>
    </div>
    <div class="masthead-meta">
        <div>Data Analysis &amp; Visualization</div>
        <div style="font-family:'Playfair Display',serif;font-size:18px;font-weight:900;color:#1A1A1A;margin-top:2px;">719 Records</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── File uploader (replaces sidebar uploader) ────────────────────────────
with st.expander("📁 Upload your own dataset (CSV or Excel)", expanded=False):
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
    )

if uploaded_file:
    raw_df, df, cleaning_report = get_clean_data(uploaded_file.read(), is_bytes=True)
else:
    raw_df, df, cleaning_report = get_clean_data(DEFAULT_CSV)

# ── Navigation (replaces sidebar radio) ──────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Home"

NAV_PAGES = ["Home", "Data Overview", "Data Cleaning", "Visualizations", "Insights", "Prediction"]

nav_cols = st.columns(len(NAV_PAGES))
for i, p in enumerate(NAV_PAGES):
    with nav_cols[i]:
        if st.button(p, key=f"nav_{p}", use_container_width=True):
            st.session_state.page = p
            st.rerun()

page = st.session_state.page

# ── Date-range filter (replaces sidebar filter) ───────────────────────────
min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

fc1, fc2 = st.columns([3, 1])
with fc2:
    date_range = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        label_visibility="collapsed",
    )
with fc1:
    if len(date_range) == 2:
        df_filtered = filter_by_date(df, date_range[0], date_range[1])
    else:
        df_filtered = df.copy()
    st.caption(f"Showing **{len(df_filtered):,}** of **{len(df):,}** records · Jan 2023 – Dec 2025")

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# Editorial layout: masthead KPI ticker + 3-column newspaper grid
# ══════════════════════════════════════════════════════════════════════════════
if page == "Home":

    # KPI ticker ribbon
    st.markdown(f"""
    <div class="ticker">
        <div class="ticker-sep">
            <div class="ticker-val">{int(df['Children_Apprehended'].sum()):,}</div>
            <div class="ticker-lbl">Total Apprehended</div>
        </div>
        <div class="ticker-sep">
            <div class="ticker-val">{int(df['Children_in_HHS'].max()):,}</div>
            <div class="ticker-lbl">Peak HHS Census</div>
        </div>
        <div class="ticker-sep">
            <div class="ticker-val">{df['Children_Transferred'].mean():.0f}</div>
            <div class="ticker-lbl">Avg Daily Transfers</div>
        </div>
        <div class="ticker-sep">
            <div class="ticker-val">{int(df['Children_Discharged'].sum()):,}</div>
            <div class="ticker-lbl">Total Discharged</div>
        </div>
        <div>
            <div class="ticker-val">{int(df.sort_values('Date').iloc[-1]['Children_in_HHS']):,}</div>
            <div class="ticker-lbl">Latest HHS Count</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 3-column newspaper grid
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        st.markdown('<div class="headline-block">Children in HHS Care — Full Trend Analysis</div>',
                    unsafe_allow_html=True)

        st.markdown('<div class="chart-card">'
                    '<div class="chart-title">Daily HHS Census</div>'
                    '<div class="chart-sub">With 30-day moving average overlay</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(viz.line_with_ma(df_filtered, "Children_in_HHS", window=30),
                        use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="chart-card">'
                    '<div class="chart-title">Correlation Heatmap</div>'
                    '<div class="chart-sub">Pearson correlations between all metrics</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(viz.correlation_heatmap(df_filtered, NUMERIC_COLS),
                        use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="headline-block">Year-over-Year Comparison</div>',
                    unsafe_allow_html=True)

        st.markdown('<div class="chart-card">'
                    '<div class="chart-title">Apprehended vs Discharged</div>'
                    '<div class="chart-sub">Annual totals by year</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            viz.bar_yearly(get_yearly_summary(df_filtered),
                           ["Children_Apprehended", "Children_Discharged"]),
            use_container_width=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="chart-card">'
                    '<div class="chart-title">Monthly Heatmap Calendar</div>'
                    '<div class="chart-sub">HHS census by month and year</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(viz.monthly_heatmap(df_filtered, "Children_in_HHS"),
                        use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="headline-block sm">Key Findings</div>', unsafe_allow_html=True)

        peak_row = df.loc[df["Children_in_HHS"].idxmax()]
        avg_yr   = df.groupby("Year")["Children_in_HHS"].mean().round(0)

        for eyebrow, title, body in [
            ("ABOUT", "HHS ORR Dashboard",
             f"Tracks unaccompanied migrant children from apprehension through "
             f"HHS shelter placement to sponsor discharge. "
             f"{len(df):,} daily observations across 5 key metrics."),
            ("PEAK CUSTODY",
             f"Census reached {int(peak_row['Children_in_HHS']):,}",
             f"Recorded on {peak_row['Date'].strftime('%B %d, %Y')} during peak border activity."),
            ("DISCHARGE IMPACT",
             "68.8% predictive signal",
             "Children Discharged is the dominant driver of HHS census — ahead of apprehensions."),
            ("SEASONAL TREND",
             "Spring surge confirmed",
             "March–May records highest apprehensions across all three years in the dataset."),
            ("YEAR COMPARISON",
             "73% decline 2023→2025",
             f"Average HHS census fell from {int(avg_yr.get(2023, 0)):,} "
             f"to {int(avg_yr.get(2025, 0)):,}."),
        ]:
            st.markdown(
                f'<div class="story-card">'
                f'<div class="story-eyebrow">{eyebrow}</div>'
                f'<div class="story-title">{title}</div>'
                f'<div class="story-body">{body}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATA OVERVIEW  (logic unchanged from main.py)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Data Overview":
    st.markdown('<div class="section-header">📊 Data Overview</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📋 Raw Data Table", "📐 Summary Statistics", "🗂️ Dataset Info"])

    with tab1:
        st.markdown(
            f"**Showing {len(df_filtered):,} rows × {len(df_filtered.columns)} columns** "
            "(filtered by date range)"
        )
        show_cols = st.multiselect(
            "Select columns to display",
            options=df_filtered.columns.tolist(),
            default=["Date"] + NUMERIC_COLS,
        )
        st.dataframe(
            df_filtered[show_cols].reset_index(drop=True),
            use_container_width=True,
            height=480,
        )
        csv_bytes = df_filtered[show_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️  Download Filtered CSV",
            data=csv_bytes,
            file_name="hhs_children_filtered.csv",
            mime="text/csv",
        )

    with tab2:
        st.markdown("#### Descriptive Statistics")
        stats_df = get_summary_stats(df_filtered)
        st.dataframe(stats_df.style.format("{:.1f}"), use_container_width=True)

        st.markdown("#### Null Value Report")
        null_counts = df_filtered.isnull().sum().reset_index()
        null_counts.columns = ["Column", "Null Count"]
        null_counts["% Missing"] = (null_counts["Null Count"] / len(df_filtered) * 100).round(2)
        st.dataframe(null_counts, use_container_width=True)

    with tab3:
        i1, i2, i3 = st.columns(3)
        with i1:
            st.metric("Total Rows", f"{len(df_filtered):,}")
            st.metric("Date Range Start", str(df_filtered["Date"].min().date()))
        with i2:
            st.metric("Total Columns", len(df_filtered.columns))
            st.metric("Date Range End", str(df_filtered["Date"].max().date()))
        with i3:
            st.metric("Years Covered", df_filtered["Year"].nunique())
            st.metric("Months Covered", df_filtered["YearMonth"].nunique())

        st.markdown("#### Data Types")
        dtype_df = pd.DataFrame({
            "Column": df_filtered.dtypes.index,
            "Type":   df_filtered.dtypes.values.astype(str),
        })
        st.dataframe(dtype_df, use_container_width=True, height=260)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DATA CLEANING  (logic unchanged from main.py)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Data Cleaning":
    st.markdown('<div class="section-header">🧹 Data Cleaning Report</div>', unsafe_allow_html=True)

    st.markdown(
        "Every transformation applied to the raw dataset is documented below. "
        "The cleaning pipeline is fully automated and reproducible."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Original Rows", f"{cleaning_report.get('original_rows', len(raw_df)):,}")
    with c2:
        st.metric("Rows After Cleaning", f"{cleaning_report.get('clean_rows', len(df)):,}")
    with c3:
        st.metric("Duplicates Removed", cleaning_report.get("duplicates_removed", 0))
    with c4:
        st.metric("Nulls Filled (Median)", cleaning_report.get("nulls_filled_with_median", 0))

    st.markdown("<br>", unsafe_allow_html=True)

    steps = [
        ("✅", "Column Standardisation",
         "Column names were stripped of extra whitespace and spaces replaced with underscores "
         "for consistent programmatic access."),
        ("✅", "Date Parsing",
         f"The Date column was parsed using pandas to_datetime() with errors='coerce'. "
         f"{cleaning_report.get('bad_dates_dropped', 0)} unparseable date rows were dropped."),
        ("✅", "Chronological Sort",
         "Records were sorted in ascending date order and the index was reset."),
        ("✅", "Numeric Coercion",
         f"All metric columns were cast to numeric types. "
         f"{cleaning_report.get('new_nulls_from_coerce', 0)} non-numeric values were converted to NaN."),
        ("✅", "Null Imputation",
         f"{cleaning_report.get('nulls_filled_with_median', 0)} null values were filled with the "
         "column-wise median. Median imputation is robust to outliers, making it appropriate "
         "for this dataset."),
        ("✅", "Duplicate Removal",
         f"{cleaning_report.get('duplicates_removed', 0)} exact duplicate rows were identified "
         "and removed."),
        ("✅", "Feature Engineering",
         "Helper columns (Year, Month, Month_Name, YearMonth) were derived from the Date column "
         "for aggregation and filtering."),
    ]

    for icon, step_name, explanation in steps:
        with st.expander(f"{icon} {step_name}", expanded=False):
            st.markdown(f"<p style='color:#6B7280;'>{explanation}</p>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Before vs After Cleaning</div>', unsafe_allow_html=True)
    ba_df = pd.DataFrame({
        "Metric": ["Row Count", "Null Values", "Duplicate Rows", "Columns"],
        "Before Cleaning": [
            cleaning_report.get("original_rows", len(raw_df)),
            int(raw_df.isnull().sum().sum()),
            int(raw_df.duplicated().sum()),
            cleaning_report.get("original_cols", len(raw_df.columns)),
        ],
        "After Cleaning": [
            cleaning_report.get("clean_rows", len(df)),
            int(df[NUMERIC_COLS].isnull().sum().sum()),
            0,
            cleaning_report.get("clean_cols", len(df.columns)),
        ],
    })
    st.dataframe(ba_df.set_index("Metric"), use_container_width=True)

    st.markdown('<div class="section-header">Null Value Heatmap (Cleaned Data)</div>',
                unsafe_allow_html=True)
    null_matrix = df[NUMERIC_COLS].isnull().astype(int)
    if null_matrix.sum().sum() == 0:
        st.success("✅ No missing values remain in the cleaned dataset.")
    else:
        null_heatmap = go.Figure(go.Heatmap(
            z=null_matrix.values.T,
            x=df["Date"].dt.strftime("%Y-%m-%d"),
            y=[COLUMN_LABELS.get(c, c) for c in NUMERIC_COLS],
            colorscale=[[0, "white"], [1, "#DC2626"]],
            showscale=False,
        ))
        st.plotly_chart(null_heatmap, use_container_width=True)

    st.markdown('<div class="section-header">Download Cleaned Dataset</div>', unsafe_allow_html=True)
    display_df = df.copy()
    display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
    csv_clean = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️  Download Cleaned CSV",
        data=csv_clean,
        file_name="hhs_children_cleaned.csv",
        mime="text/csv",
        type="primary",
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — VISUALIZATIONS  (logic unchanged from main.py)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Visualizations":
    st.markdown('<div class="section-header">📈 Visualizations</div>', unsafe_allow_html=True)

    monthly = get_monthly_summary(df_filtered)
    yearly  = get_yearly_summary(df_filtered)

    chart_page = st.selectbox(
        "Choose chart type",
        [
            "📉 Trend Lines",
            "📊 Bar Charts",
            "🥧 Pie / Donut",
            "🔥 Correlation Heatmap",
            "📦 Box Plots",
            "📅 Monthly Heatmap Calendar",
            "🔵 Scatter Plot",
            "🏔️ Stacked Area",
            "📐 Distribution Histogram",
        ],
    )

    if chart_page == "📉 Trend Lines":
        st.markdown("### Trend Lines Over Time")
        selected_cols = st.multiselect(
            "Select columns to plot",
            NUMERIC_COLS,
            default=["Children_in_HHS", "Children_Apprehended"],
            format_func=lambda c: COLUMN_LABELS.get(c, c),
        )
        if selected_cols:
            st.plotly_chart(viz.line_chart_multi(df_filtered, selected_cols),
                            use_container_width=True)
        st.markdown("### With Moving Average")
        ma_col    = st.selectbox("Column", NUMERIC_COLS,
                                 format_func=lambda c: COLUMN_LABELS.get(c, c))
        ma_window = st.slider("Moving average window (days)", 3, 90, 30)
        st.plotly_chart(viz.line_with_ma(df_filtered, ma_col, ma_window),
                        use_container_width=True)

    elif chart_page == "📊 Bar Charts":
        tab_m, tab_y = st.tabs(["Monthly", "Yearly"])
        with tab_m:
            bar_col = st.selectbox("Column", NUMERIC_COLS, key="bar_monthly",
                                   format_func=lambda c: COLUMN_LABELS.get(c, c))
            st.plotly_chart(viz.bar_monthly(monthly, bar_col), use_container_width=True)
        with tab_y:
            bar_y_cols = st.multiselect(
                "Columns", NUMERIC_COLS,
                default=["Children_Apprehended", "Children_Discharged"],
                key="bar_yearly",
                format_func=lambda c: COLUMN_LABELS.get(c, c),
            )
            if bar_y_cols:
                st.plotly_chart(viz.bar_yearly(yearly, bar_y_cols), use_container_width=True)

    elif chart_page == "🥧 Pie / Donut":
        st.markdown("### Average Daily Composition")
        pie_cols = st.multiselect("Columns", NUMERIC_COLS, default=NUMERIC_COLS,
                                  format_func=lambda c: COLUMN_LABELS.get(c, c))
        if pie_cols:
            st.plotly_chart(viz.pie_chart(df_filtered, pie_cols, "Average Daily Composition"),
                            use_container_width=True)

    elif chart_page == "🔥 Correlation Heatmap":
        st.markdown("### Pearson Correlation Heatmap")
        hm_cols = st.multiselect("Columns", NUMERIC_COLS, default=NUMERIC_COLS,
                                 format_func=lambda c: COLUMN_LABELS.get(c, c))
        if hm_cols:
            st.plotly_chart(viz.correlation_heatmap(df_filtered, hm_cols),
                            use_container_width=True)

    elif chart_page == "📦 Box Plots":
        st.markdown("### Distribution by Year")
        box_col = st.selectbox("Column", NUMERIC_COLS,
                               format_func=lambda c: COLUMN_LABELS.get(c, c))
        st.plotly_chart(viz.boxplot_by_year(df_filtered, box_col), use_container_width=True)

    elif chart_page == "📅 Monthly Heatmap Calendar":
        st.markdown("### Monthly Heatmap Calendar")
        hm_cal_col = st.selectbox("Column", NUMERIC_COLS,
                                  format_func=lambda c: COLUMN_LABELS.get(c, c))
        st.plotly_chart(viz.monthly_heatmap(df_filtered, hm_cal_col), use_container_width=True)

    elif chart_page == "🔵 Scatter Plot":
        st.markdown("### Scatter — Two Variables")
        col_x = st.selectbox("X axis", NUMERIC_COLS, index=0,
                             format_func=lambda c: COLUMN_LABELS.get(c, c))
        col_y = st.selectbox("Y axis", NUMERIC_COLS, index=3,
                             format_func=lambda c: COLUMN_LABELS.get(c, c))
        st.plotly_chart(viz.scatter_two_cols(df_filtered, col_x, col_y),
                        use_container_width=True)

    elif chart_page == "🏔️ Stacked Area":
        st.markdown("### Stacked Area Chart")
        area_cols = st.multiselect(
            "Columns", NUMERIC_COLS,
            default=["Children_Apprehended", "Children_Transferred", "Children_Discharged"],
            format_func=lambda c: COLUMN_LABELS.get(c, c),
        )
        if area_cols:
            st.plotly_chart(viz.area_chart(df_filtered, area_cols), use_container_width=True)

    elif chart_page == "📐 Distribution Histogram":
        st.markdown("### Distribution Histogram")
        hist_col = st.selectbox("Column", NUMERIC_COLS,
                                format_func=lambda c: COLUMN_LABELS.get(c, c))
        bins = st.slider("Number of bins", 10, 100, 40)
        st.plotly_chart(viz.histogram(df_filtered, hist_col, bins), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — INSIGHTS  (logic unchanged from main.py)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Insights":
    st.markdown('<div class="section-header">💡 Key Insights</div>', unsafe_allow_html=True)

    yearly        = get_yearly_summary(df)
    peak_hhs_row  = df.loc[df["Children_in_HHS"].idxmax()]
    min_hhs_row   = df.loc[df["Children_in_HHS"].idxmin()]
    corr_matrix   = df[NUMERIC_COLS].corr()
    avg_by_year   = df.groupby("Year")["Children_in_HHS"].mean().round(0)

    st.markdown('<div class="section-header">Statistical Insights</div>', unsafe_allow_html=True)

    ins = [
        ("📈 Peak HHS Custody",
         f"The highest single-day HHS custody count was <strong>{int(peak_hhs_row['Children_in_HHS']):,}</strong> "
         f"children, recorded on <strong>{peak_hhs_row['Date'].strftime('%B %d, %Y')}</strong>. "
         "This coincided with a period of high border activity and capacity constraints in shelters."),

        ("📉 Lowest HHS Count",
         f"The lowest HHS census was <strong>{int(min_hhs_row['Children_in_HHS']):,}</strong> children "
         f"on <strong>{min_hhs_row['Date'].strftime('%B %d, %Y')}</strong>, reflecting improved "
         "discharge rates and a temporary slowdown in apprehensions."),

        ("🔗 Apprehensions Drive HHS Census",
         f"The correlation between daily apprehensions and HHS census is "
         f"<strong>{corr_matrix.loc['Children_Apprehended','Children_in_HHS']:.3f}</strong>. "
         "While not immediately strong on a day-to-day basis, lagged cumulative apprehensions "
         "are the primary driver of HHS population growth."),

        ("🔄 Transfer Efficiency",
         f"The correlation between daily transfers (CBP→HHS) and the HHS census change is "
         f"<strong>{corr_matrix.loc['Children_Transferred','Children_in_HHS']:.3f}</strong>. "
         "High daily transfer volumes are systematically associated with a rising HHS population."),

        ("📅 Seasonal Pattern",
         "Spring months (March–May) consistently show the highest apprehension volumes, "
         "driven by seasonal migration patterns and improved crossing conditions. "
         "December through February typically sees lower intake volumes."),

        ("📊 Year-over-Year Trend",
         "Average daily HHS census: " +
         " → ".join([f"<strong>{yr}: {int(v):,}</strong>" for yr, v in avg_by_year.items()]) +
         ". The sharp decline from 2023 to 2024/2025 reflects both policy changes and "
         "increased discharge throughput."),
    ]

    for title, text in ins:
        st.markdown(
            f'<div class="insight-box"><h4>{title}</h4><p>{text}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-header">Year-over-Year Comparison Table</div>',
                unsafe_allow_html=True)
    y_display = yearly.copy()
    y_display.columns = ["Year", "Total Apprehended", "Avg CBP Custody",
                         "Total Transferred", "Avg HHS Census", "Total Discharged"]
    y_display["Year"] = y_display["Year"].astype(str)
    st.dataframe(
        y_display.set_index("Year").style.format({
            "Total Apprehended": "{:,.0f}",
            "Avg CBP Custody":   "{:,.1f}",
            "Total Transferred": "{:,.0f}",
            "Avg HHS Census":    "{:,.1f}",
            "Total Discharged":  "{:,.0f}",
        }),
        use_container_width=True,
    )

    st.markdown('<div class="section-header">Visual Comparison</div>', unsafe_allow_html=True)
    cv1, cv2 = st.columns(2)
    with cv1:
        st.plotly_chart(
            viz.bar_yearly(yearly, ["Children_Apprehended", "Children_Discharged"]),
            use_container_width=True,
        )
    with cv2:
        st.plotly_chart(
            viz.monthly_heatmap(df, "Children_in_HHS"),
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — PREDICTION  (logic unchanged from main.py)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Prediction":
    st.markdown('<div class="section-header">🤖 Machine Learning Predictions</div>',
                unsafe_allow_html=True)

    st.markdown(
        "This page applies regression-based forecasting models to project future values "
        "of selected metrics. Three model types are available, each with different "
        "complexity and interpretability trade-offs."
    )

    st.markdown('<div class="section-header">Model Configuration</div>', unsafe_allow_html=True)
    cfg1, cfg2, cfg3 = st.columns(3)
    with cfg1:
        target_col = st.selectbox(
            "Target variable to predict",
            NUMERIC_COLS,
            index=3,
            format_func=lambda c: COLUMN_LABELS.get(c, c),
        )
    with cfg2:
        model_type = st.selectbox(
            "Model type",
            ["Linear Regression", "Polynomial Regression (deg 2)",
             "Polynomial Regression (deg 3)", "Rolling Average"],
        )
    with cfg3:
        forecast_days = st.slider("Forecast horizon (days)", 7, 180, 60)

    run_btn = st.button("▶  Run Forecast", type="primary")

    if run_btn:
        with st.spinner("Training model and generating forecast …"):
            if model_type == "Linear Regression":
                fig, metrics = ml.linear_regression_forecast(df_filtered, target_col, forecast_days)
            elif model_type == "Polynomial Regression (deg 2)":
                fig, metrics = ml.polynomial_regression_forecast(df_filtered, target_col, 2, forecast_days)
            elif model_type == "Polynomial Regression (deg 3)":
                fig, metrics = ml.polynomial_regression_forecast(df_filtered, target_col, 3, forecast_days)
            else:
                fig, metrics = ml.rolling_average_forecast(df_filtered, target_col, 30, forecast_days)

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Model Performance Metrics")
        m_cols = st.columns(len(metrics))
        for i, (metric_name, metric_val) in enumerate(metrics.items()):
            with m_cols[i]:
                st.metric(metric_name, f"{metric_val:,}")

        st.info(
            "**MAE** (Mean Absolute Error) — average absolute difference between actual and "
            "predicted values.  **RMSE** (Root Mean Squared Error) — penalises large errors "
            "more heavily than MAE.  **R²** (Coefficient of Determination) — proportion of "
            "variance explained; 1.0 is a perfect fit."
        )

    st.markdown("---")
    st.markdown('<div class="section-header">Feature Importance Analysis</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Identifies which daily metrics most strongly predict the selected target variable, "
        "using standardised regression coefficients."
    )

    fi_target   = st.selectbox(
        "Target for feature importance", NUMERIC_COLS, index=3,
        format_func=lambda c: COLUMN_LABELS.get(c, c), key="fi_target",
    )
    fi_features = [c for c in NUMERIC_COLS if c != fi_target]
    fi_fig      = ml.feature_importance(df_filtered, fi_target, fi_features)
    if fi_fig.data:
        st.plotly_chart(fi_fig, use_container_width=True)

    with st.expander("📖 How the Models Work (Viva Explanation)"):
        st.markdown("""
### Linear Regression
Fits a straight-line relationship between time (expressed as an ordinal day number)
and the target variable. The model equation is `y = mx + b` where `m` is the trend
slope and `b` is the intercept. The 95% confidence band is computed using ±1.96
standard deviations of the training residuals.

### Polynomial Regression
Extends linear regression by including higher-order terms (x², x³ …). A degree-2
model captures a single inflection (e.g. rise then fall), while degree-3 captures
two inflections. Implemented as a `sklearn` Pipeline of `PolynomialFeatures` +
`LinearRegression`.

### Rolling Average Forecast
Projects a flat line equal to the mean of the last 30 observed values. Highly
interpretable and appropriate when no clear trend is present. Useful as a baseline
against which more complex models are benchmarked.

### Feature Importance
A multiple linear regression is fitted with all other columns as predictors of
the chosen target. Predictors are standardised (mean=0, std=1) before fitting,
making the absolute coefficient magnitudes directly comparable. These are then
normalised to sum to 100% for display.
        """)
