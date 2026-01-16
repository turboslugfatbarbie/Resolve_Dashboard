import pathlib
from io import BytesIO
import base64

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


# -------------------------------------------------------------------
# Correlation helpers
# -------------------------------------------------------------------
def corr_with_p_n(df: pd.DataFrame, cols, method: str = "pearson"):
    """
    Compute pairwise correlation, p-values, and N for a set of columns.
    method: 'pearson' or 'spearman'
    Returns (r, p, n) as DataFrames.
    """
    cols = list(cols)
    r = pd.DataFrame(np.nan, index=cols, columns=cols)
    p = pd.DataFrame(np.nan, index=cols, columns=cols)
    n = pd.DataFrame(0, index=cols, columns=cols, dtype=int)

    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if j < i:
                continue  # upper triangle only

            valid = df[[a, b]].dropna()
            n_ij = len(valid)
            n.loc[a, b] = n_ij
            n.loc[b, a] = n_ij

            if n_ij < 3:
                continue

            if a == b:
                r_val, p_val = 1.0, 0.0
            else:
                if method == "spearman":
                    r_val, p_val = stats.spearmanr(valid[a], valid[b])
                else:
                    r_val, p_val = stats.pearsonr(valid[a], valid[b])

            r.loc[a, b] = r_val
            r.loc[b, a] = r_val
            p.loc[a, b] = p_val
            p.loc[b, a] = p_val

    # Diagonal: full per-variable N, r=1, p=0
    for c in cols:
        nn = df[c].notna().sum()
        n.loc[c, c] = int(nn)
        r.loc[c, c] = 1.0
        p.loc[c, c] = 0.0

    return r, p, n


def sig_stars(p):
    """Return significance stars for a p-value."""
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def build_corr_display(r: pd.DataFrame, p: pd.DataFrame, n: pd.DataFrame, show_n: bool = True):
    """
    Build a pretty display DataFrame: 'r** (n=xx)' with stars and optional N.
    """
    star_arr = np.vectorize(sig_stars)(p.values)
    stars = pd.DataFrame(star_arr, index=p.index, columns=p.columns)

    # No stars on diagonal
    for c in r.columns:
        stars.loc[c, c] = ""

    r_str = r.round(2).astype(str) + stars

    if show_n:
        n_str = n.astype(int).astype(str)
        disp = r_str + " (n=" + n_str + ")"
    else:
        disp = r_str

    return disp


# -------------------------------------------------------------------
# Streamlit page config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="RESOLVE FSHD Dashboard",
    layout="wide",
)

BASELINE_SPLIT_MONTH = 0.1

def explain_month_0_vs_0_1():
    st.sidebar.info(
        "**Why do I see month 0 vs 0.1?**\n\n"
        "- **Month 0** = baseline visit.\n"
        "- **Month 0.1** usually indicates a *baseline split/unscheduled follow-up* recorded shortly after baseline "
        "(often the next day or within a few days). It's commonly used to keep two baseline-adjacent measurements distinct "
        "while still sorting them near baseline.\n\n"
        "If your baseline testing was split over two days, you may want to **merge 0 and 0.1** into a single baseline timepoint."
    )
st.sidebar.markdown("## 0) Baseline handling")
merge_baseline_split = st.sidebar.checkbox(
    "Merge month 0 and 0.1 into a single baseline timepoint (month 0)",
    value=True,
    help="If baseline testing was split across days, this combines month 0 and 0.1 for analysis.",
)
explain_month_0_vs_0_1()

# Persist preference
st.session_state["merge_baseline_split"] = merge_baseline_split


# -------------------------------------------------------------------
# Data intake (Streamlit Cloud-safe): user uploads SAS files
# -------------------------------------------------------------------
REQUIRED_SAS_FILES = [
    "css.sas7bdat",
    "qmt.sas7bdat",
    "fshd_com.sas7bdat",
    "baseline.sas7bdat",
]

def _normalize_filename(name: str) -> str:
    # Streamlit UploadedFile.name can include paths on some browsers; keep only basename
    return pathlib.Path(name).name

def sidebar_data_intake():
    st.sidebar.markdown("## 1) Load data")
    st.sidebar.caption(
        "Upload the required SAS files. Data stays in your session memory and is not committed to the repo."
    )

    uploaded = st.sidebar.file_uploader(
        "Upload .sas7bdat files",
        type=["sas7bdat"],
        accept_multiple_files=True,
        help="Upload: css.sas7bdat, qmt.sas7bdat, fshd_com.sas7bdat, baseline.sas7bdat",
    )

    # Store bytes in session_state so reruns don't lose them
    if uploaded:
        file_bytes = {}
        for uf in uploaded:
            file_bytes[_normalize_filename(uf.name)] = uf.getvalue()
        st.session_state["sas_files_bytes"] = file_bytes

    file_bytes = st.session_state.get("sas_files_bytes", {})

    missing = [f for f in REQUIRED_SAS_FILES if f not in file_bytes]
    if missing:
        st.sidebar.warning("Missing required files:\n- " + "\n- ".join(missing))
        return None

    st.sidebar.success("All required SAS files loaded.")
    if st.sidebar.button("🧹 Clear uploaded data", use_container_width=True):
        st.session_state.pop("sas_files_bytes", None)
        st.experimental_rerun()

    return file_bytes

# -------------------------------------------------------------------
# Helpers: export plots & summarize filters
# -------------------------------------------------------------------
def merge_month_0_and_0_1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge baseline split visits by mapping month 0.1 -> 0 and aggregating duplicates per (NewID, month).

    Aggregation:
      - numeric columns: mean
      - non-numeric columns: first non-null
    """
    if df.empty or "NewID" not in df.columns or "month" not in df.columns:
        return df

    df2 = df.copy()

    # Map 0.1 -> 0.0 (be robust to float quirks)
    df2["month"] = pd.to_numeric(df2["month"], errors="coerce")
    df2.loc[np.isclose(df2["month"].values, BASELINE_SPLIT_MONTH), "month"] = 0.0

    # If this creates duplicates, aggregate them
    # Build agg dict: numeric mean, non-numeric first non-null
    numeric_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in df2.columns if c not in numeric_cols]

    agg = {}
    for c in numeric_cols:
        if c in ["month"]:  # month is part of group key; won't be aggregated
            continue
        agg[c] = "mean"

    def first_non_null(s: pd.Series):
        s2 = s.dropna()
        return s2.iloc[0] if len(s2) else np.nan

    for c in non_numeric_cols:
        if c in ["NewID", "month"]:
            continue
        agg[c] = first_non_null

    out = (
        df2.groupby(["NewID", "month"], as_index=False)
           .agg(agg)
           .sort_values(["NewID", "month"])
           .reset_index(drop=True)
    )
    return out


def save_plotly_to_report(fig, label: str, context: str):
    """
    Save a Plotly figure as an HTML snippet into the report.
    This does NOT require kaleido.
    """
    html_snippet = fig.to_html(full_html=False, include_plotlyjs="cdn")
    st.session_state["saved_plots"].append(
        {
            "label": label,
            "context": context,
            "html": html_snippet,
            "kind": "plotly",
        }
    )

def save_matplotlib_to_report(fig, label: str, context: str):
    """
    Save a matplotlib figure as a base64 PNG <img> tag into the report.
    """
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    html_snippet = f"<img src='data:image/png;base64,{encoded}' alt='{label}' />"
    st.session_state["saved_plots"].append(
        {
            "label": label,
            "context": context,
            "html": html_snippet,
            "kind": "matplotlib",
        }
    )
    
def build_html_report_for_download():
    """
    Construct the HTML report string from whatever is currently stored in
    st.session_state['saved_corrs'] and st.session_state['saved_plots'].
    Returns the HTML string, or None if nothing is saved.
    """
    saved_corrs = st.session_state.get("saved_corrs", [])
    saved_plots = st.session_state.get("saved_plots", [])

    if not saved_corrs and not saved_plots:
        return None

    html_parts = [
        "<html><head><meta charset='utf-8'>"
        "<title>RESOLVE Correlation Report</title></head><body>",
        "<h1>RESOLVE Correlation Report</h1>",
    ]

    # 1) Correlation matrices
    for idx, snap in enumerate(saved_corrs):
        html_parts.append(f"<h2>{idx + 1}. {snap['label']}</h2>")
        html_parts.append(
            "<p>"
            f"<strong>Method:</strong> {snap['method'].capitalize()}<br>"
            f"<strong>Variables:</strong> {', '.join(snap['variables'])}<br>"
            f"<strong>Filters:</strong> {snap['filters']}<br>"
            f"<strong>Context:</strong> {snap['context']}"
            "</p>"
        )
        # always include n in the HTML version
        disp_html = build_corr_display(snap["r"], snap["p"], snap["n"], show_n=True)
        html_parts.append(disp_html.to_html(escape=False, border=1))

    # 2) Saved plots
    if saved_plots:
        html_parts.append("<h1>Saved plots</h1>")
        for idx, plt_snap in enumerate(saved_plots):
            html_parts.append(f"<h2>Plot {idx + 1}. {plt_snap['label']}</h2>")
            html_parts.append(
                "<p>"
                f"<strong>Context:</strong> {plt_snap['context']}"
                "</p>"
            )
            html_parts.append(plt_snap["html"])

    html_parts.append("</body></html>")
    return "\n".join(html_parts)

def plot_download_button(fig, filename: str, key: str, label: str = "Download plot as PNG"):
    """
    Show a download button for a Plotly figure as PNG.
    Requires plotly[kaleido] installed.
    """
    try:
        img_bytes = fig.to_image(format="png")
        st.download_button(
            label=label,
            data=img_bytes,
            file_name=filename,
            mime="image/png",
            key=key,
        )
    except Exception:
        st.caption("⚠️ Unable to export PNG (make sure `plotly[kaleido]` is installed).")


def pairplot_download_button(fig, filename: str, key: str, label: str = "Download pairplot PNG"):
    """Download button for matplotlib/seaborn pairplots."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    st.download_button(
        label=label,
        data=buf.getvalue(),
        file_name=filename,
        mime="image/png",
        key=key,
    )


def seaborn_pairplot_fig(df: pd.DataFrame, cols):
    """
    Return a seaborn.pairplot figure (or None if not enough rows).
    We only require at least 2 rows; seaborn will handle NaNs pairwise.
    """
    if df.shape[0] < 2:
        return None
    sns.set(style="whitegrid")
    g = sns.pairplot(df[cols])
    return g.fig


def build_filter_summary(
    css_range,
    acss_range,
    age_range,
    gender_col,
    selected_genders,
    month_filter_mode,
    selected_months,
    selected_month_range,
    selected_ids,
):
    parts = []

    if css_range is not None:
        parts.append(f"CSS {css_range[0]:.1f}–{css_range[1]:.1f}")
    if acss_range is not None:
        parts.append(f"ACSS {acss_range[0]:.1f}–{acss_range[1]:.1f}")
    if age_range is not None:
        parts.append(f"Age_at_visit {age_range[0]:.1f}–{age_range[1]:.1f} yrs")
    if gender_col is not None and selected_genders:
        parts.append(f"{gender_col} in {list(selected_genders)}")
    if month_filter_mode == "Discrete" and selected_months:
        parts.append(f"Months ∈ {list(selected_months)}")
    elif month_filter_mode == "Range" and selected_month_range is not None:
        parts.append(f"Months {selected_month_range[0]:.0f}–{selected_month_range[1]:.0f}")
    if selected_ids:
        parts.append(f"NewID ∈ {len(selected_ids)} selected patients")

    if not parts:
        return "None (all data)"
    return " | ".join(parts)


def summarize(series: pd.Series, how: str):
    s = series.dropna()
    if s.empty:
        return np.nan
    if how == "Mean":
        return s.mean()
    elif how == "Median":
        return s.median()
    elif how == "Mode":
        mode_vals = s.mode()
        if mode_vals.empty:
            return np.nan
        return mode_vals.iloc[0]
    else:
        return np.nan


# -------------------------------------------------------------------
# Data loading / merging (from uploaded bytes)
# -------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_and_merge_data_from_bytes(file_bytes: dict):
    """
    Load the RESOLVE SAS files from in-memory bytes and merge into a single dataframe.

    file_bytes: dict mapping required SAS filenames -> raw bytes
    """
    from io import BytesIO

    def read_sas_bytes(fname, usecols=None):
        if fname not in file_bytes:
            return None
        bio = BytesIO(file_bytes[fname])
        df = pd.read_sas(bio, format="sas7bdat", encoding="latin1")
        if usecols is not None:
            existing = [c for c in usecols if c in df.columns]
            df = df[existing]
        return df

    # --- Core tables ---
    css_df = read_sas_bytes("css.sas7bdat")
    if css_df is not None:
        keep = [c for c in ["NewID", "month", "CSS"] if c in css_df.columns]
        css_df = css_df[keep]

    qmt_df = read_sas_bytes("qmt.sas7bdat")
    if qmt_df is not None:
        keep = [c for c in ["NewID", "month", "stanqmt"] if c in qmt_df.columns]
        qmt_df = qmt_df[keep]

    iwr_df = read_sas_bytes("fshd_com.sas7bdat")
    if iwr_df is not None:
        keep = ["NewID", "month", "iwr10", "wr_10", "wr_10_key", "wr10_walk", "wr10_run"]
        keep = [c for c in keep if c in iwr_df.columns]
        iwr_df = iwr_df[keep]

        if "wr_10" in iwr_df.columns and "wr_10_key" in iwr_df.columns:
            if "wr10_walk" not in iwr_df.columns:
                iwr_df["wr10_walk"] = np.nan
            if "wr10_run" not in iwr_df.columns:
                iwr_df["wr10_run"] = np.nan

            walk_keys = {1, "1", "W", "w", "walk", "Walk"}
            run_keys = {2, "2", "R", "r", "run", "Run"}

            walk_mask = iwr_df["wr_10_key"].isin(walk_keys)
            run_mask = iwr_df["wr_10_key"].isin(run_keys)

            iwr_df.loc[walk_mask, "wr10_walk"] = iwr_df.loc[walk_mask, "wr_10"]
            iwr_df.loc[run_mask, "wr10_run"] = iwr_df.loc[run_mask, "wr_10"]

    baseline_df = read_sas_bytes("baseline.sas7bdat")
    if baseline_df is not None:
        keep = ["NewID", "month", "age"]
        for gcol in ["gender", "sex", "Sex", "GENDER"]:
            if gcol in baseline_df.columns:
                keep.append(gcol)
                break
        baseline_df = baseline_df[keep]

    # Merge everything on NewID + month
    dfs = [d for d in [css_df, qmt_df, iwr_df, baseline_df] if d is not None]
    if not dfs:
        return pd.DataFrame()

    merged_df = dfs[0]
    for d in dfs[1:]:
        merged_df = pd.merge(
            merged_df,
            d,
            on=["NewID", "month"],
            how="outer",
            validate="m:m",
        )

    # Standardize column names
    merged_df.rename(columns={"stanqmt": "QMT", "wr_10": "wr10_raw"}, inplace=True)

    # Ensure numeric for relevant cols
    num_cols = ["CSS", "QMT", "iwr10", "wr10_raw", "wr10_walk", "wr10_run", "age"]
    for c in num_cols:
        if c in merged_df.columns:
            merged_df[c] = pd.to_numeric(merged_df[c], errors="coerce")

    # ---- ACSS logic ----
    if "age" in merged_df.columns and "month" in merged_df.columns:
        baseline_age = (
            merged_df.loc[merged_df["month"] == 0, ["NewID", "age"]]
            .rename(columns={"age": "baseline_age"})
        )
        merged_df = merged_df.merge(baseline_age, on="NewID", how="left")

        conditions = [
            (merged_df["month"] < 12),
            ((merged_df["month"] >= 12) & (merged_df["month"] < 24)),
            (merged_df["month"] == 24),
        ]
        choices = [
            merged_df["baseline_age"],
            merged_df["baseline_age"] + 1,
            merged_df["baseline_age"] + 2,
        ]
        merged_df["filled_age"] = np.select(conditions, choices, default=np.nan)
        merged_df["age_at_visit"] = merged_df["filled_age"]

        valid = merged_df[["CSS", "filled_age"]].notna().all(axis=1)
        merged_df["ACSS"] = np.nan
        merged_df.loc[valid, "ACSS"] = (
            2.0 * merged_df.loc[valid, "CSS"] / merged_df.loc[valid, "filled_age"]
        ) * 1000.0

    merged_df = merged_df.sort_values(["NewID", "month"]).reset_index(drop=True)
    return merged_df

# -------------------------------------------------------------------
# Slope computation (patient-level, fixed windows)
# -------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def compute_slopes_summary(df: pd.DataFrame):
    """
    Compute per-patient slopes for *all* month windows (t1 < t2),
    excluding 0.1 as an endpoint.

    Example months: {0, 0.1, 3, 12, 18, 24}
    -> windows on {0, 3, 12, 18, 24}:
       0–3, 0–12, 0–18, 0–24,
       3–12, 3–18, 3–24,
       12–18, 12–24,
       18–24, etc.

    Variables (if present):
      CSS, ACSS, QMT, iwr10, wr10_walk, wr10_run

    Returns a DataFrame indexed by NewID with slope columns.
    """
    if "NewID" not in df.columns or "month" not in df.columns:
        return pd.DataFrame()

    # Unique months present
    months = sorted(df["month"].dropna().unique())

    # Exclude 0.1 as an endpoint
    allowed_months = [m for m in months if m != 0.1]

    # All (t1, t2) with t1 < t2
    windows = []
    for i, t1 in enumerate(allowed_months):
        for t2 in allowed_months[i + 1 :]:
            windows.append((t1, t2))

    if not windows:
        return pd.DataFrame()

    candidate_vars = ["CSS", "ACSS", "QMT", "iwr10", "wr10_walk", "wr10_run"]
    variables = [v for v in candidate_vars if v in df.columns]

    if not variables:
        return pd.DataFrame()

    slopes_dict = {}

    for var in variables:
        wide = df.pivot_table(
            index="NewID",
            columns="month",
            values=var,
            aggfunc="mean",
        )

        for (t1, t2) in windows:
            if t1 not in wide.columns or t2 not in wide.columns or t2 == t1:
                continue
            col_name = f"{var}_slope_{t1:g}_{t2:g}"
            slope = (wide[t2] - wide[t1]) / (t2 - t1)
            slopes_dict[col_name] = slope

    if not slopes_dict:
        return pd.DataFrame()

    slopes_df = pd.DataFrame(slopes_dict)
    slopes_df.index.name = "NewID"
    slopes_df.reset_index(inplace=True)
    return slopes_df


# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------
file_bytes = sidebar_data_intake()
if file_bytes is None:
    st.title("RESOLVE FSHD Dashboard")
    st.info("Upload the required SAS files in the sidebar to start.")
    st.stop()

df_raw = load_and_merge_data_from_bytes(file_bytes)

merge_baseline_split = st.session_state.get("merge_baseline_split", True)
if merge_baseline_split:
    df = merge_month_0_and_0_1(df_raw)
else:
    df = df_raw

##------TOGGLE
st.title("RESOLVE FSHD Dashboard")

with st.expander("Baseline handling (0 vs 0.1 months)", expanded=False):
    st.markdown(
        "- **Month 0** = baseline visit.\n"
        "- **Month 0.1** = a baseline-adjacent measurement (often a split baseline over two days or an unscheduled near-baseline visit).\n"
        "If baseline testing was split across days, merging 0 and 0.1 into baseline avoids treating them as separate scheduled timepoints."
    )

    current = st.session_state.get("merge_baseline_split", True)
    new_val = st.checkbox(
        "Merge 0 and 0.1 into baseline (month 0)",
        value=current,
        key="merge_toggle_in_app",
    )

    if new_val != current:
        st.session_state["merge_baseline_split"] = new_val
        st.cache_data.clear()  # clears cached merge/load results (safe + simple)
        st.experimental_rerun()

# ---------- NEW: propagate gender/sex to all visits per patient ----------
gender_col_for_fill = None
for cand in ["gender", "sex", "Sex", "GENDER"]:
    if cand in df.columns:
        gender_col_for_fill = cand
        break

if gender_col_for_fill is not None:
    df[gender_col_for_fill] = (
        df.groupby("NewID")[gender_col_for_fill]
          .transform(lambda s: s.dropna().iloc[0] if s.dropna().size > 0 else np.nan)
    )
# ------------------------------------------------------------------------

st.title("RESOLVE FSHD Dashboard")

# --- Baseline (month 0) measures per patient for CSS / ACSS sliders ---
baseline_rows = df[df["month"] == 0].copy()

baseline_css = None
baseline_acss = None

if "CSS" in df.columns:
    # month-0 CSS per patient
    baseline_css = (
        baseline_rows[["NewID", "CSS"]]
        .dropna(subset=["CSS"])
        .drop_duplicates(subset=["NewID"])
        .set_index("NewID")["CSS"]
        .astype(float)
    )

if "ACSS" in df.columns:
    # month-0 ACSS per patient
    baseline_acss = (
        baseline_rows[["NewID", "ACSS"]]
        .dropna(subset=["ACSS"])
        .drop_duplicates(subset=["NewID"])
        .set_index("NewID")["ACSS"]
        .astype(float)
    )


if df.empty:
    st.error(
        "No data loaded. Make sure css.sas7bdat, qmt.sas7bdat, "
        "fshd_com.sas7bdat, baseline.sas7bdat are in DATA_DIR and "
        "adjust column names in load_and_merge_data() if needed."
    )
    st.stop()

st.caption(f"Loaded {len(df):,} rows for {df['NewID'].nunique():,} unique patients.")

with st.expander("Debug: column names"):
    st.write(list(df.columns))

# initialize storage for saved correlation matrices
if "saved_corrs" not in st.session_state:
    st.session_state["saved_corrs"] = []
# NEW: storage for saved plots
if "saved_plots" not in st.session_state:
    st.session_state["saved_plots"] = []


# ----------------- Visit month filter FIRST -----------------
month_filter_mode = None
selected_months = None
selected_month_range = None
ref_month = None  # reference month for CSS/ACSS patient-level filters

if "month" in df.columns:
    month_values = sorted(df["month"].dropna().unique())
    month_min = float(np.nanmin(month_values))
    month_max = float(np.nanmax(month_values))

    st.sidebar.markdown("### Visit month filter")
    month_filter_mode = st.sidebar.radio(
        "Mode",
        options=["Discrete", "Range"],
        index=0,
        horizontal=True,
    )

    if month_filter_mode == "Discrete":
        month_options = [float(m) for m in month_values]
        selected_months = st.sidebar.multiselect(
            "Visit month(s)",
            options=month_options,
            default=month_options,
            help="Select specific visit months to include.",
        )
        if selected_months:
            ref_month = float(min(selected_months))
    else:  # "Range"
        selected_month_range = st.sidebar.slider(
            "Visit month range",
            min_value=month_min,
            max_value=month_max,
            value=(month_min, month_max),
            step=1.0,
            help="Include visits whose month is between these values (inclusive).",
        )
        ref_month = float(selected_month_range[0])
else:
    month_filter_mode = None
    selected_months = None
    selected_month_range = None

# If no month filter at all, default ref_month to earliest month
if ref_month is None and "month" in df.columns:
    ref_month = float(df["month"].dropna().min()) if df["month"].notna().any() else None

# ----------------- CSS / ACSS sliders based on ref_month -----------------
css_ref_series = None
acss_ref_series = None

if ref_month is not None:
    ref_rows = df[df["month"] == ref_month].copy()

    if "CSS" in df.columns:
        css_ref_series = (
            ref_rows[["NewID", "CSS"]]
            .dropna(subset=["CSS"])
            .drop_duplicates(subset=["NewID"])
            .set_index("NewID")["CSS"]
            .astype(float)
        )

    if "ACSS" in df.columns:
        acss_ref_series = (
            ref_rows[["NewID", "ACSS"]]
            .dropna(subset=["ACSS"])
            .drop_duplicates(subset=["NewID"])
            .set_index("NewID")["ACSS"]
            .astype(float)
        )

# CSS range at reference month
if css_ref_series is not None and not css_ref_series.empty:
    css_min = float(css_ref_series.min())
    css_max = float(css_ref_series.max())
    css_range = st.sidebar.slider(
        f"CSS range at month {ref_month:g}",
        min_value=css_min,
        max_value=css_max,
        value=(css_min, css_max),
        step=0.5,
    )
else:
    css_range = None

# ACSS range at reference month
if acss_ref_series is not None and not acss_ref_series.empty:
    acss_min = float(acss_ref_series.min())
    acss_max = float(acss_ref_series.max())
    acss_range = st.sidebar.slider(
        f"ACSS range at month {ref_month:g}",
        min_value=acss_min,
        max_value=acss_max,
        value=(acss_min, acss_max),
        step=0.5,
    )
else:
    acss_range = None

# ----------------- Age at visit -----------------
if "age_at_visit" in df.columns:
    age_min = float(np.nanmin(df["age_at_visit"]))
    age_max = float(np.nanmax(df["age_at_visit"]))
    age_range = st.sidebar.slider(
        "Age at visit (years)",
        min_value=age_min,
        max_value=age_max,
        value=(age_min, age_max),
        step=0.5,
    )
else:
    age_range = None

# ----------------- Gender -----------------
gender_col = None
for cand in ["gender", "sex", "Sex", "GENDER"]:
    if cand in df.columns:
        gender_col = cand
        break

if gender_col is not None:
    gender_values = sorted(df[gender_col].dropna().unique())
    selected_genders = st.sidebar.multiselect(
        "Gender",
        options=gender_values,
        default=gender_values,
    )
else:
    selected_genders = None

# ----------------- Optional: specific patients -----------------
all_ids = sorted(df["NewID"].dropna().unique())
with st.sidebar.expander("Optional: subset to specific patients"):
    selected_ids = st.multiselect(
        "Patient IDs (NewID)",
        options=all_ids,
        default=[],
        help="Leave empty to include all patients.",
    )


# -------------------------------------------------------------------
# Apply filters
#   1) patient-level via CSS/ACSS at reference month
#   2) row-level non-month filters (age, gender, IDs)
#   3) row-level month filter (for main overview only)
# -------------------------------------------------------------------

# start with all patients
eligible_ids = set(df["NewID"].dropna().unique())

# CSS filter at ref_month
if css_range is not None and css_ref_series is not None and not css_ref_series.empty:
    ids_css = css_ref_series[
        css_ref_series.between(css_range[0], css_range[1])
    ].index
    eligible_ids &= set(ids_css)

# ACSS filter at ref_month
if acss_range is not None and acss_ref_series is not None and not acss_ref_series.empty:
    ids_acss = acss_ref_series[
        acss_ref_series.between(acss_range[0], acss_range[1])
    ].index
    eligible_ids &= set(ids_acss)

# ---- base dataframe: all months for eligible patients ----
filtered_base = df[df["NewID"].isin(eligible_ids)].copy()

# non-month visit filters
if age_range is not None and "age_at_visit" in filtered_base.columns:
    filtered_base = filtered_base[
        filtered_base["age_at_visit"].between(age_range[0], age_range[1])
    ]

if gender_col is not None and selected_genders:
    filtered_base = filtered_base[filtered_base[gender_col].isin(selected_genders)]

if selected_ids:
    filtered_base = filtered_base[filtered_base["NewID"].isin(selected_ids)]

# ---- main "filtered" for overview / global plots: add month filter ----
filtered = filtered_base.copy()

if month_filter_mode == "Discrete" and selected_months is not None and selected_months:
    filtered = filtered[filtered["month"].isin(selected_months)]
elif month_filter_mode == "Range" and selected_month_range is not None:
    low, high = selected_month_range
    filtered = filtered[filtered["month"].between(low, high)]

# -------------------------------------------------------------------
# Shared filter summary + CSV download
# -------------------------------------------------------------------
filter_summary = build_filter_summary(
    css_range,
    acss_range,
    age_range,
    gender_col,
    selected_genders,
    month_filter_mode,
    selected_months,
    selected_month_range,
    selected_ids,
)

st.markdown(f"**Current filters:** {filter_summary}")

csv_bytes = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    label="💾 Download filtered data (CSV)",
    data=csv_bytes,
    file_name="resolve_filtered.csv",
    mime="text/csv",
)

st.markdown("---")

# -------------------------------------------------------------------
# Top-level tabs
# -------------------------------------------------------------------
tab_overview, tab_corr = st.tabs(["Overview", "Correlations"])

# ===================================================================
# OVERVIEW TAB
# ===================================================================
with tab_overview:
    # Summary statistic choice
    stat_choice = st.selectbox(
        "Summary statistic for CSS / QMT / ACSS",
        options=["Median", "Mean", "Mode"],
        index=0,
    )
    # Choose which variable to show in the middle metric (default = QMT)
    numeric_for_metrics = list(filtered.select_dtypes(include=[np.number]).columns)
    if not numeric_for_metrics:
        metric_var = None
    else:
        default_metric_index = (
            numeric_for_metrics.index("QMT")
            if "QMT" in numeric_for_metrics
            else 0
        )
        metric_var = st.selectbox(
            "Middle metric variable (default: QMT)",
            options=numeric_for_metrics,
            index=default_metric_index,
        )

    # ----------------- Overview + table -----------------
    st.subheader("Filtered Data Overview")

    total_rows = len(df)
    total_patients = df["NewID"].nunique()

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric("Rows", f"{len(filtered):,} / {total_rows:,}")

    with c2:
        st.metric(
            "Patients (NewID)",
            f"{filtered['NewID'].nunique():,} / {total_patients:,}"
        )

    with c3:
        if "CSS" in filtered.columns:
            css_val = summarize(filtered["CSS"], stat_choice)
            label = f"{stat_choice} CSS"
            st.metric(label, f"{css_val:.2f}" if pd.notna(css_val) else "N/A")
        else:
            st.metric(f"{stat_choice} CSS", "N/A")

    with c4:
        if metric_var is not None and metric_var in filtered.columns:
            metric_val = summarize(filtered[metric_var], stat_choice)
            label = f"{stat_choice} {metric_var}"
            st.metric(label, f"{metric_val:.2f}" if pd.notna(metric_val) else "N/A")
        else:
            st.metric(f"{stat_choice} (no variable)", "N/A")


    with c5:
        if "ACSS" in filtered.columns:
            acss_val = summarize(filtered["ACSS"], stat_choice)
            label = f"{stat_choice} ACSS"
            st.metric(label, f"{acss_val:.2f}" if pd.notna(acss_val) else "N/A")
        else:
            st.metric(f"{stat_choice} ACSS", "N/A")

    st.dataframe(
        filtered.head(500),
        use_container_width=True,
        height=350,
    )

    st.markdown("---")

    # ----------------- Distributions -----------------
    st.subheader("Distributions")

    plot_cols = []
    for col in ["CSS", "ACSS", "QMT", "iwr10", "wr10_walk", "wr10_run"]:
        if col in filtered.columns:
            plot_cols.append(col)

    if plot_cols:
        col1, col2 = st.columns(2)

        with col1:
            metric_for_hist = st.selectbox(
                "Histogram metric",
                options=plot_cols,
                index=0,
            )
            fig_hist = px.histogram(
                filtered,
                x=metric_for_hist,
                nbins=30,
                title=f"Distribution of {metric_for_hist}",
            )
            st.caption(f"Screenshot note: {filter_summary}")
            st.plotly_chart(fig_hist, use_container_width=True)
            plot_download_button(fig_hist, f"hist_{metric_for_hist}.png", key="hist_png")
            
            # Save histogram to report
            hist_label = st.text_input(
                "Label for this histogram (for report)",
                value=f"Histogram of {metric_for_hist}",
                key="label_histogram",
            )
            if st.button("Save histogram to report", key="save_histogram_btn"):
                save_plotly_to_report(
                    fig_hist,
                    label=hist_label,
                    context=f"Histogram of {metric_for_hist}. Filters: {filter_summary}",
                )
                st.success(f"Saved plot: {hist_label}")


        with col2:
            metric_for_box = st.selectbox(
                "Boxplot metric",
                options=plot_cols,
                index=0,
                key="box_metric",
            )
            fig_box = px.box(
                filtered,
                y=metric_for_box,
                points="all",
                title=f"{metric_for_box} (all visits)",
            )
            st.caption(f"Screenshot note: {filter_summary}")
            st.plotly_chart(fig_box, use_container_width=True)
            plot_download_button(fig_box, f"box_{metric_for_box}.png", key="box_png")

            box_label = st.text_input(
                "Label for this boxplot (for report)",
                value=f"Boxplot of {metric_for_box}",
                key="label_boxplot",
            )
            if st.button("Save boxplot to report", key="save_boxplot_btn"):
                save_plotly_to_report(
                    fig_box,
                    label=box_label,
                    context=f"Boxplot of {metric_for_box}. Filters: {filter_summary}",
                )
                st.success(f"Saved plot: {box_label}")

            
    else:
        st.info("No numeric score columns found for plotting in this subset.")

    st.markdown("---")

    # ----------------- Relationships: generic X vs Y -----------------
    st.subheader("Relationships")

    numeric_cols = list(filtered.select_dtypes(include=[np.number]).columns)

    if len(numeric_cols) < 2:
        st.info("Not enough numeric columns to plot a relationship.")
    else:
        default_x = numeric_cols.index("CSS") if "CSS" in numeric_cols else 0
        default_y = (
            numeric_cols.index("QMT")
            if "QMT" in numeric_cols and "QMT" != numeric_cols[default_x]
            else (1 if len(numeric_cols) > 1 else 0)
        )

        col_rel1, col_rel2, col_rel3 = st.columns(3)
        with col_rel1:
            x_var = st.selectbox(
                "X variable",
                options=numeric_cols,
                index=default_x,
            )
        with col_rel2:
            y_options = [c for c in numeric_cols if c != x_var]
            default_y_opt = 0
            if "QMT" in y_options:
                default_y_opt = y_options.index("QMT")
            elif default_y < len(y_options):
                default_y_opt = default_y

            y_var = st.selectbox(
                "Y variable",
                options=y_options,
                index=default_y_opt,
            )
        with col_rel3:
            color_candidates = [None, "month"] + [
                c for c in filtered.columns
                if c not in {x_var, y_var, "month"} and filtered[c].nunique() < 20
            ]
            seen = set()
            color_options = []
            for c in color_candidates:
                if c not in seen:
                    color_options.append(c)
                    seen.add(c)

            color_choice = st.selectbox(
                "Color by",
                options=color_options,
                format_func=lambda x: "None" if x is None else str(x),
            )
            color_by = None if color_choice is None else color_choice

        hover_cols = []
        for col in ["NewID", "month"]:
            if col in filtered.columns:
                hover_cols.append(col)

        fig_rel = px.scatter(
            filtered,
            x=x_var,
            y=y_var,
            color=color_by,
            hover_data=hover_cols if hover_cols else None,
            trendline="ols",
            title=f"{y_var} vs {x_var}",
        )
        st.caption(f"Screenshot note: {filter_summary}")
        st.plotly_chart(fig_rel, use_container_width=True)
        plot_download_button(
            fig_rel,
            f"{y_var}_vs_{x_var}.png",
            key="rel_scatter_png",
        )
        scatter_label = st.text_input(
            "Label for this scatter (for report)",
            value=f"{y_var} vs {x_var}",
            key="label_scatter",
        )
        if st.button("Save scatter to report", key="save_scatter_btn"):
            save_plotly_to_report(
                fig_rel,
                label=scatter_label,
                context=f"{y_var} vs {x_var}. Filters: {filter_summary}",
            )
            st.success(f"Saved plot: {scatter_label}")

    st.markdown("---")


# ===================================================================
# CORRELATIONS TAB
# ===================================================================
with tab_corr:
    # one checkbox controlling all tables on this tab
    show_n_all = st.checkbox(
        "Show sample size (n) in correlation tables",
        value=True,
        key="show_n_all_corr",
    )
    # --- In-page jump links for this tab ---
    st.markdown(
        """
        **Jump to section:**

        - [Global correlation matrix](#global-corr)
        - [Correlations & pairplots by month](#corr-by-month)
        - [Patient-level slopes (fixed month windows)](#slopes-section)
        - [Saved correlation matrices & report](#saved-section)
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ----------------- Global correlation matrix -----------------
    st.markdown('<a id="global-corr"></a>', unsafe_allow_html=True)
    st.subheader("Global correlation matrix (current filters)")

    num_cols_all = list(filtered.select_dtypes(include=[np.number]).columns)

    if len(num_cols_all) < 2:
        st.info("Not enough numeric columns available.")
    else:
        st.markdown("**Choose variables for correlation matrix:**")

        corr_method_global = st.radio(
            "Correlation method",
            options=["pearson", "spearman"],
            index=0,
            horizontal=True,
            format_func=lambda x: x.capitalize(),
            key="global_corr_method",
        )

        selected_corr_vars = st.multiselect(
            "Variables",
            options=num_cols_all,
            default=num_cols_all,
        )

        if len(selected_corr_vars) >= 2:
            r_g, p_g, n_g = corr_with_p_n(
                filtered,
                selected_corr_vars,
                method=corr_method_global,
            )
            disp_g = build_corr_display(r_g, p_g, n_g, show_n_all)

            st.write(
                f"Correlation matrix "
                f"({corr_method_global.capitalize()}, stars = p<0.05/0.01/0.001):"
            )
            st.dataframe(
                disp_g,
                use_container_width=True,
            )
            st.download_button(
                "Download r-matrix (CSV)",
                r_g.to_csv().encode("utf-8"),
                file_name="correlation_matrix_r.csv",
                mime="text/csv",
            )

            # Save to report
            label_global = st.text_input(
                "Label for this matrix (for report)",
                value="Global correlation",
                key="label_global_corr",
            )
            if st.button("Save this matrix to report", key="save_global_corr_btn"):
                st.session_state["saved_corrs"].append(
                    {
                        "label": label_global,
                        "scope": "global",
                        "method": corr_method_global,
                        "variables": list(selected_corr_vars),
                        "filters": filter_summary,
                        "context": "Global (all visits within current filters)",
                        "r": r_g.copy(),
                        "p": p_g.copy(),
                        "n": n_g.copy(),
                    }
                )
                st.success(f"Saved matrix: {label_global}")
        else:
            st.warning("Select at least two variables to compute the correlation matrix.")

    st.markdown("---")

# ----------------- Correlations & pairplots by month -----------------
    st.markdown('<a id="corr-by-month"></a>', unsafe_allow_html=True)
    st.subheader("Correlations & pairplots by month")

    
    corr_method = st.radio(
        "Correlation method for these tables",
        options=["pearson", "spearman"],
        index=0,
        horizontal=True,
        format_func=lambda x: x.capitalize(),
        key="pairplot_corr_method",
    )
    
    numeric_cols_all = list(filtered.select_dtypes(include=[np.number]).columns)
    
    default_pair_vars = [
        v for v in ["CSS", "QMT", "iwr10", "wr10_walk", "wr10_run", "ACSS"]
        if v in numeric_cols_all
    ]
    if len(default_pair_vars) < 2:
        default_pair_vars = numeric_cols_all[: min(6, len(numeric_cols_all))]
    
    pair_vars = st.multiselect(
        "Variables to include",
        options=numeric_cols_all,
        default=default_pair_vars,
        help="These variables are used for both correlation matrices and pairplots.",
        key="pairplot_vars",
    )
    
    if len(pair_vars) < 2:
        st.info("Select at least two variables to draw correlation matrices and pairplots.")
    else:
        tab_disc, tab_ranges = st.tabs(["By discrete months", "By month ranges"])
    
        # ------------------------------------------------------------------
        # Discrete months
        # ------------------------------------------------------------------
        with tab_disc:
            st.markdown("#### Discrete months")
    
            if "month" not in df.columns:
                st.info("No 'month' column available in the data.")
            else:
                # use months present in the base-filtered dataset
                disc_months_all = sorted(filtered_base["month"].dropna().unique())
                selected_disc_months = st.multiselect(
                    "Discrete months (from full dataset; will respect other filters)",
                    options=[float(m) for m in disc_months_all],
                    default=[float(m) for m in disc_months_all],
                    key="pairplot_disc_months",
                )
    
                if not selected_disc_months:
                    st.info("Select at least one month.")
                else:
                    # ---------- First: all correlation matrices ----------
                    st.markdown("##### Correlation matrices by month")
    
                    for m in selected_disc_months:
                        df_m = filtered_base[filtered_base["month"] == m]
    
                        if df_m.shape[0] < 3:
                            st.markdown(
                                f"**Month {m:g}** – not enough rows for correlation "
                                f"(need ≥3 rows; have {df_m.shape[0]})."
                            )
                            st.markdown("---")
                            continue
    
                        st.markdown(f"**Month {m:g}**")
    
                        r_m, p_m, n_m = corr_with_p_n(df_m, pair_vars, method=corr_method)
                        disp_m = build_corr_display(r_m, p_m, n_m, show_n_all)
    
                        st.write(
                            f"Correlation matrix ({corr_method.capitalize()}, "
                            "stars = p<0.05/0.01/0.001):"
                        )
                        st.dataframe(disp_m, use_container_width=True)
                        st.download_button(
                            label=f"Download r-matrix (month {m:g}) as CSV",
                            data=r_m.to_csv().encode("utf-8"),
                            file_name=f"corr_r_month_{m:g}.csv",
                            mime="text/csv",
                            key=f"corr_csv_month_{m}",
                        )
    
                        # Save this month matrix
                        label_m = st.text_input(
                            f"Label for month {m:g} matrix",
                            value=f"Month {m:g} correlation",
                            key=f"label_month_{m}",
                        )
                        if st.button(
                            f"Save month {m:g} matrix to report",
                            key=f"save_month_{m}_btn",
                        ):
                            st.session_state["saved_corrs"].append(
                                {
                                    "label": label_m,
                                    "scope": "month",
                                    "method": corr_method,
                                    "variables": list(pair_vars),
                                    "filters": filter_summary,
                                    "context": f"Month {m:g}",
                                    "r": r_m.copy(),
                                    "p": p_m.copy(),
                                    "n": n_m.copy(),
                                }
                            )
                            st.success(f"Saved matrix: {label_m}")
    
                        st.markdown("---")
    
                    # ---------- Then: all pairplots ----------
                    st.markdown("##### Pairplots by month")
    
                    for m in selected_disc_months:
                        df_m = filtered_base[filtered_base["month"] == m]
    
                        if df_m.shape[0] < 2:
                            st.markdown(
                                f"**Month {m:g}** – not enough rows for pairplot "
                                f"(need ≥2 rows; have {df_m.shape[0]})."
                            )
                            st.markdown("---")
                            continue
    
                        st.markdown(f"**Month {m:g}**")
    
                        fig_m = seaborn_pairplot_fig(df_m, pair_vars)
                        if fig_m is None:
                            st.info("Not enough rows for pairplot.")
                        else:
                            st.pyplot(fig_m)
                            pairplot_download_button(
                                fig_m,
                                filename=f"pairplot_month_{m:g}.png",
                                key=f"pairplot_month_{m}",
                            )
                        pair_label = st.text_input(
                            f"Label for this pairplot (for report)",
                            value=f"Pairplot – month {m:g}",
                            key=f"label_pairplot_month_{m}",
                        )
                        if st.button(
                            f"Save pairplot for month {m:g} to report",
                            key=f"save_pairplot_month_{m}_btn",
                        ):
                            save_matplotlib_to_report(
                                fig_m,
                                label=pair_label,
                                context=f"Pairplot for month {m:g}. Filters: {filter_summary}",
                            )
                            st.success(f"Saved plot: {pair_label}")
                        st.markdown("---")
    
        # ------------------------------------------------------------------
        # Month ranges (non-overlapping)
        # ------------------------------------------------------------------
        with tab_ranges:
            st.markdown("#### Month ranges (non-overlapping)")
    
            if "month" not in filtered_base.columns:
                st.info("No 'month' column available in the data.")
            else:
                month_vals = filtered_base["month"].dropna().values
                if month_vals.size == 0:
                    st.info("No month values present in the data.")
                else:
                    month_min = float(np.nanmin(month_vals))
                    month_max = float(np.nanmax(month_vals))
    
                    if month_min == month_max:
                        st.info(
                            f"Only one month value is present in the data ({month_min:g}). "
                            "Month ranges require at least two distinct months. "
                            "Use the **Discrete months** tab instead."
                        )
                    else:
                        n_ranges = st.number_input(
                            "Number of month ranges",
                            min_value=1,
                            max_value=4,
                            value=1,
                            step=1,
                            help="Ranges are applied in order and made non-overlapping automatically.",
                            key="pairplot_num_ranges",
                        )
    
                        ranges = []
                        total_span = month_max - month_min
                        for i in range(int(n_ranges)):
                            default_low = month_min + (i * total_span / n_ranges)
                            default_high = month_min + ((i + 1) * total_span / n_ranges)
                            default_low = max(month_min, min(default_low, month_max))
                            default_high = max(month_min, min(default_high, month_max))
                            if default_low > default_high:
                                default_low, default_high = default_high, default_low
    
                            low_i, high_i = st.slider(
                                f"Range {i + 1}",
                                min_value=month_min,
                                max_value=month_max,
                                value=(float(default_low), float(default_high)),
                                step=1.0,
                                key=f"pairplot_range_{i}",
                            )
                            ranges.append((low_i, high_i))
    
                        if ranges:
                            # ---------- First: all correlation matrices ----------
                            st.markdown("##### Correlation matrices by month range")
    
                            remaining_mask = filtered_base["month"].notna()
                            for idx, (low, high) in enumerate(ranges):
                                current_mask = remaining_mask & filtered_base["month"].between(low, high)
                                df_r = filtered_base[current_mask]
    
                                # Prevent overlap for later ranges
                                remaining_mask = (
                                    remaining_mask
                                    & ~filtered_base["month"].between(low, high)
                                )
    
                                if df_r.shape[0] < 3:
                                    st.markdown(
                                        f"**Range {idx + 1}: months {low:g}–{high:g}** – "
                                        f"not enough rows for correlation (need ≥3; "
                                        f"have {df_r.shape[0]})."
                                    )
                                    st.markdown("---")
                                    continue
    
                                st.markdown(
                                    f"**Range {idx + 1}: months {low:g}–{high:g} "
                                    "(non-overlapping slice)**"
                                )
    
                                r_r, p_r, n_r = corr_with_p_n(df_r, pair_vars, method=corr_method)
                                disp_r = build_corr_display(r_r, p_r, n_r, show_n_all)
    
                                st.write(
                                    f"Correlation matrix ({corr_method.capitalize()}, "
                                    "stars = p<0.05/0.01/0.001):"
                                )
                                st.dataframe(disp_r, use_container_width=True)
                                st.download_button(
                                    label=f"Download r-matrix (months {low:g}–{high:g}) as CSV",
                                    data=r_r.to_csv().encode("utf-8"),
                                    file_name=f"corr_r_months_{low:g}_{high:g}.csv",
                                    mime="text/csv",
                                    key=f"corr_csv_range_{idx}",
                                )
    
                                label_r = st.text_input(
                                    f"Label for range {idx + 1} matrix",
                                    value=f"Months {low:g}–{high:g} correlation",
                                    key=f"label_range_{idx}",
                                )
                                if st.button(
                                    f"Save range {idx + 1} matrix to report",
                                    key=f"save_range_{idx}_btn",
                                ):
                                    st.session_state["saved_corrs"].append(
                                        {
                                            "label": label_r,
                                            "scope": "range",
                                            "method": corr_method,
                                            "variables": list(pair_vars),
                                            "filters": filter_summary,
                                            "context": f"Months {low:g}–{high:g}",
                                            "r": r_r.copy(),
                                            "p": p_r.copy(),
                                            "n": n_r.copy(),
                                        }
                                    )
                                    st.success(f"Saved matrix: {label_r}")
    
                                st.markdown("---")
    
                            # ---------- Then: all pairplots ----------
                            st.markdown("##### Pairplots by month range")
    
                            remaining_mask = filtered_base["month"].notna()
                            for idx, (low, high) in enumerate(ranges):
                                current_mask = remaining_mask & filtered_base["month"].between(low, high)
                                df_r = filtered_base[current_mask]
    
                                # Prevent overlap for later ranges
                                remaining_mask = (
                                    remaining_mask
                                    & ~filtered_base["month"].between(low, high)
                                )
    
                                if df_r.shape[0] < 2:
                                    st.markdown(
                                        f"**Range {idx + 1}: months {low:g}–{high:g}** – "
                                        f"not enough rows for pairplot (need ≥2; "
                                        f"have {df_r.shape[0]})."
                                    )
                                    st.markdown("---")
                                    continue
    
                                st.markdown(
                                    f"**Range {idx + 1}: months {low:g}–{high:g} "
                                    "(non-overlapping slice)**"
                                )
    
                                fig_r = seaborn_pairplot_fig(df_r, pair_vars)
                                if fig_r is None:
                                    st.info("Not enough rows for pairplot.")
                                else:
                                    st.pyplot(fig_r)
                                    pairplot_download_button(
                                        fig_r,
                                        filename=f"pairplot_months_{low:g}_{high:g}.png",
                                        key=f"pairplot_range_{idx}_png",
                                    )
    
                                st.markdown("---")
    
    
        # ----------------- Patient-level slopes (fixed windows) -----------------
        st.markdown("---")
        st.markdown('<a id="slopes-section"></a>', unsafe_allow_html=True)
        st.subheader("Patient-level slopes (fixed month windows)")

    
        slopes_full = compute_slopes_summary(df)
    
        if slopes_full.empty:
            st.info("No slope variables could be computed (missing NewID/month or key variables).")
        else:
            # restrict slopes to patients present in the current filtered view
            current_ids = filtered["NewID"].dropna().unique()
            slopes_filtered = slopes_full[slopes_full["NewID"].isin(current_ids)].copy()
    
            if slopes_filtered.shape[0] < 2:
                st.info("Not enough patients within current filters to analyze slopes.")
            else:
                slope_cols = [c for c in slopes_filtered.columns if c != "NewID" and "_slope_" in c]
                st.markdown(
                    "Slopes are computed per patient for **all available month pairs** "
                    "(t₁ < t₂), using the visit months in the data as endpoints, "
                    "excluding **0.1 months** as an endpoint."
                )
    
                # --- Parse slope column names into variables and windows ---
                # e.g. 'QMT_slope_0_12' -> var = 'QMT', window = ('0', '12')
                vars_available = sorted({c.split("_slope_")[0] for c in slope_cols})
    
                windows = set()
                for c in slope_cols:
                    _, win_part = c.split("_slope_")
                    t1_str, t2_str = win_part.split("_")
                    windows.add((t1_str, t2_str))
    
                # Sort windows numerically and build pretty labels
                windows_sorted = sorted(
                    windows,
                    key=lambda tt: (float(tt[0]), float(tt[1])),
                )
                window_labels = [f"{t1}–{t2} months" for (t1, t2) in windows_sorted]
                window_map = {
                    label: (t1, t2)
                    for label, (t1, t2) in zip(window_labels, windows_sorted)
                }
    
                corr_method_slopes = st.radio(
                    "Correlation method for slopes",
                    options=["pearson", "spearman"],
                    index=0,
                    horizontal=True,
                    format_func=lambda x: x.capitalize(),
                    key="slopes_corr_method",
                )
    
                # --- New UI: pick variables and windows separately ---
                selected_vars = st.multiselect(
                    "Variables to include",
                    options=vars_available,
                    default=vars_available,
                    key="slope_vars_selected_vars",
                )
    
                selected_window_labels = st.multiselect(
                    "Slope windows (months t₁–t₂)",
                    options=window_labels,
                    default=window_labels,
                    key="slope_windows_selected",
                )
    
                # Build the list of actual slope columns from var × window
                selected_slope_cols = []
                for var in selected_vars:
                    for label in selected_window_labels:
                        t1, t2 = window_map[label]
                        col_name = f"{var}_slope_{t1}_{t2}"
                        if col_name in slope_cols:
                            selected_slope_cols.append(col_name)
    
                # Deduplicate and keep a stable order
                selected_slope_cols = [c for c in slope_cols if c in set(selected_slope_cols)]
    
                if len(selected_slope_cols) >= 2:
                    r_s, p_s, n_s = corr_with_p_n(
                        slopes_filtered.set_index("NewID"),
                        selected_slope_cols,
                        method=corr_method_slopes,
                    )

                    disp_s = build_corr_display(r_s, p_s, n_s, show_n_all)
    
                    st.write(
                        f"Correlation matrix on slopes "
                        f"({corr_method_slopes.capitalize()}, stars = p<0.05/0.01/0.001):"
                    )
                    st.dataframe(disp_s, use_container_width=True)
                    st.download_button(
                        "Download slopes r-matrix (CSV)",
                        r_s.to_csv().encode("utf-8"),
                        file_name="correlation_slopes_r.csv",
                        mime="text/csv",
                    )
    
                    label_s = st.text_input(
                        "Label for this slopes matrix (for report)",
                        value="Slopes correlation",
                        key="label_slopes_corr",
                    )
                    if st.button("Save slopes matrix to report", key="save_slopes_corr_btn"):
                        st.session_state["saved_corrs"].append(
                            {
                                "label": label_s,
                                "scope": "slopes",
                                "method": corr_method_slopes,
                                "variables": list(selected_slope_cols),
                                "filters": filter_summary,
                                "context": "Patient-level slopes (fixed windows)",
                                "r": r_s.copy(),
                                "p": p_s.copy(),
                                "n": n_s.copy(),
                            }
                        )

                else:
                    st.info("Select at least two slope variables for the correlation matrix.")

    # ----------------- Saved matrices + HTML report -----------------
    st.markdown("---")
    st.markdown('<a id="saved-section"></a>', unsafe_allow_html=True)
    st.subheader("Saved correlation matrices & report")


    saved = st.session_state["saved_corrs"]
    saved_plots = st.session_state["saved_plots"]

    if not saved and not saved_plots:
        st.info("No correlation matrices or plots saved yet. Use the buttons above to save them.")
    else:
        # Show saved correlation matrices
        if saved:
            for idx, snap in enumerate(saved, start=1):
                with st.expander(f"{idx}. {snap['label']}"):
                    st.markdown(f"**Method:** {snap['method'].capitalize()}")
                    st.markdown(f"**Variables:** {', '.join(snap['variables'])}")
                    st.markdown(f"**Filters:** {snap['filters']}")
                    st.markdown(f"**Context:** {snap['context']}")
                    disp_snap = build_corr_display(snap["r"], snap["p"], snap["n"], show_n_all)
                    st.dataframe(disp_snap, use_container_width=True)

        # (Optional) show list of saved plots
        if saved_plots:
            st.markdown("### Saved plots")
            for idx, plt_snap in enumerate(saved_plots, start=1):
                st.write(f"{idx}. {plt_snap['label']} – {plt_snap['context']}")

        if st.button("Generate HTML report from saved items", key="gen_html_report"):
            html_report = build_html_report_for_download()
            if html_report is not None:
                st.download_button(
                    "Download correlation report (HTML)",
                    data=html_report,
                    file_name="resolve_correlation_report.html",
                    mime="text/html",
                    key="download_html_report",
                )

# -------------------------------------------------------------------
# Sidebar: Saved items & report (placed at the END so it sees final state)
# -------------------------------------------------------------------
with st.sidebar.expander("📑 Saved items for report", expanded=False):
    saved_corrs = st.session_state.get("saved_corrs", [])
    saved_plots = st.session_state.get("saved_plots", [])

    num_corr = len(saved_corrs)
    num_plots = len(saved_plots)
    total_items = num_corr + num_plots

    if total_items == 0:
        st.write("No items saved yet. Use the **Save to report** buttons in the main view.")
    else:
        st.write(f"**Total saved items:** {total_items}")
        st.write(f"- Correlation matrices: **{num_corr}**")
        st.write(f"- Plots: **{num_plots}**")

        if num_corr:
            st.markdown("**Correlation matrices:**")
            for i, snap in enumerate(saved_corrs, start=1):
                st.write(f"{i}. {snap.get('label', '(no label)')} – {snap.get('context', '')}")

        if num_plots:
            st.markdown("**Plots:**")
            for i, snap in enumerate(saved_plots, start=1):
                st.write(f"{i}. {snap.get('label', '(no label)')} – {snap.get('context', '')}")

        st.markdown("---")

        # Direct download: always build the report when there are saved items
        html_report = build_html_report_for_download()
        if html_report is not None:
            st.download_button(
                "Download report (HTML)",
                data=html_report,
                file_name="resolve_correlation_report.html",
                mime="text/html",
                key="sidebar_download_html_report",
            )

        if st.button("🧹 Clear all saved items", key="clear_saved_items_sidebar"):
            st.session_state["saved_corrs"] = []
            st.session_state["saved_plots"] = []
            st.experimental_rerun()
