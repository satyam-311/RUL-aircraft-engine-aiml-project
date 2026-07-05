# -*- coding: utf-8 -*-
"""
Aircraft Engine Health Monitor -- Phase 9 Streamlit Dashboard.

Run from project root:
    uv run streamlit run app/dashboard.py
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from rul_prediction.components.data_ingestion import parse_cmapss_bytes
from rul_prediction.components.preprocessor import Preprocessor  # noqa: F401  # pickle compat
from rul_prediction.constants.constants import ALL_COLUMNS, NON_INFORMATIVE_SENSORS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT    = Path(__file__).parent.parent
_FIGURES = _ROOT / "reports" / "figures"

# ---------------------------------------------------------------------------
# Design tokens -- all hex, ASCII-only
# ---------------------------------------------------------------------------

C_BG        = "#0f172a"   # page background   (deep navy)
C_CARD      = "#1e293b"   # card / sidebar bg (slate-800)
C_ACCENT    = "#38bdf8"   # primary accent    (sky-400)
C_TEXT      = "#f1f5f9"   # primary text      (slate-100)
C_MUTED     = "#94a3b8"   # secondary text    (slate-400)
C_BORDER    = "#334155"   # gridlines/borders (slate-700)

C_HEALTHY   = "#22c55e"   # green-500
C_WARNING   = "#f59e0b"   # amber-500
C_CRITICAL  = "#ef4444"   # red-500

# ---------------------------------------------------------------------------
# Sensor metadata (ASCII-only units)
# ---------------------------------------------------------------------------

SENSOR_INFO: dict[str, tuple[str, str, str]] = {
    "sensor_1":  ("T2",        "Total temperature at fan inlet",        "deg R"),
    "sensor_2":  ("T24",       "Total temperature at LPC outlet",       "deg R"),
    "sensor_3":  ("T30",       "Total temperature at HPC outlet",       "deg R"),
    "sensor_4":  ("T50",       "Total temperature at LPT outlet",       "deg R"),
    "sensor_5":  ("P2",        "Pressure at fan inlet",                 "psia"),
    "sensor_6":  ("P15",       "Total pressure in bypass-duct",         "psia"),
    "sensor_7":  ("P30",       "Total pressure at HPC outlet",          "psia"),
    "sensor_8":  ("Nf",        "Physical fan speed",                    "rpm"),
    "sensor_9":  ("Nc",        "Physical core speed",                   "rpm"),
    "sensor_10": ("epr",       "Engine pressure ratio (P50/P2)",        "--"),
    "sensor_11": ("Ps30",      "Static pressure at HPC outlet",         "psia"),
    "sensor_12": ("phi",       "Ratio of fuel flow to Ps30",            "pps/psi"),
    "sensor_13": ("NRf",       "Corrected fan speed",                   "rpm"),
    "sensor_14": ("NRc",       "Corrected core speed",                  "rpm"),
    "sensor_15": ("BPR",       "Bypass ratio",                         "--"),
    "sensor_16": ("farB",      "Burner fuel-air ratio",                 "--"),
    "sensor_17": ("htBleed",   "Bleed enthalpy",                        "--"),
    "sensor_18": ("Nf_dmd",    "Demanded fan speed",                    "rpm"),
    "sensor_19": ("PCNfR_dmd", "Demanded corrected fan speed",          "rpm"),
    "sensor_20": ("W31",       "HPT coolant bleed",                     "lbm/s"),
    "sensor_21": ("W32",       "LPT coolant bleed",                     "lbm/s"),
}

OP_SETTING_INFO: dict[str, tuple[str, str]] = {
    "op_setting_1": ("Altitude",                      "ft"),
    "op_setting_2": ("Mach number",                   "--"),
    "op_setting_3": ("TRA (Throttle Resolver Angle)", "deg"),
}

NON_INFORMATIVE = set(NON_INFORMATIVE_SENSORS)

REQUIRED_COLS = (
    [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# ---------------------------------------------------------------------------
# Config-driven constants (with safe fallback)
# ---------------------------------------------------------------------------

try:
    from rul_prediction.config.configuration import load_config as _load_cfg
    _cfg          = _load_cfg()
    RUL_CAP       = int(_cfg["dataset"]["rul_cap"])
    RUL_THRESHOLD = float(_cfg["model"]["rul_threshold"])
    _PROD_MODEL   = str(_cfg["model"].get("type", "lstm")).lower()
except Exception:
    RUL_CAP, RUL_THRESHOLD, _PROD_MODEL = 125, 30.0, "lstm"

WARNING_LEVEL = 60.0

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading inference pipeline...")
def _pipeline():
    from rul_prediction.config.configuration import load_config
    from rul_prediction.pipeline.inference_pipeline import InferencePipeline
    return InferencePipeline(load_config())


@st.cache_resource(show_spinner="Loading dataset...")
def _ingestion():
    from rul_prediction.components.data_ingestion import DataIngestion
    return DataIngestion()


@st.cache_data(show_spinner=False)
def _train_stats() -> pd.DataFrame:
    return _ingestion().load_train_data()


@st.cache_data(show_spinner=False)
def _model_comparison() -> pd.DataFrame:
    return pd.read_csv(_ROOT / "reports" / "model_comparison.csv")


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def _badge(label: str, bg: str, fg: str = "#ffffff") -> None:
    """Render a coloured pill badge via inline HTML."""
    st.markdown(
        f'<span style="display:inline-block;background:{bg};color:{fg};'
        f'padding:3px 10px;border-radius:12px;font-size:0.82rem;'
        f'font-weight:600;letter-spacing:0.04em;line-height:1.8">'
        f'{label}</span>',
        unsafe_allow_html=True,
    )


def _callout(body: str, accent: str = C_ACCENT) -> None:
    """Left-bordered callout box with semi-transparent card background."""
    st.markdown(
        f'<div style="border-left:3px solid {accent};padding:10px 16px;'
        f'background:rgba(30,41,59,0.55);border-radius:0 6px 6px 0;'
        f'margin:8px 0;font-size:0.95rem;line-height:1.6">{body}</div>',
        unsafe_allow_html=True,
    )


def _apply_dark_theme(fig: go.Figure) -> go.Figure:
    """Make Plotly figures transparent so they blend with the dark Streamlit background."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=C_TEXT,
    )
    fig.update_xaxes(gridcolor=C_BORDER, zerolinecolor=C_BORDER)
    fig.update_yaxes(gridcolor=C_BORDER, zerolinecolor=C_BORDER)
    return fig


def _sensor_label(col: str) -> str:
    if col in SENSOR_INFO:
        sym, desc, units = SENSOR_INFO[col]
        return f"{col}  {sym}  ({desc})"
    if col in OP_SETTING_INFO:
        desc, units = OP_SETTING_INFO[col]
        return f"{col}  {desc}"
    return col


def _risk_tier(rul: float) -> tuple[str, str, str]:
    """Return (label, colour, recommendation)."""
    if rul > WARNING_LEVEL:
        return "HEALTHY", C_HEALTHY, "No maintenance action required."
    if rul > RUL_THRESHOLD:
        return "WARNING", C_WARNING, f"Schedule inspection within {int(rul)} cycles."
    return "CRITICAL", C_CRITICAL, "Immediate maintenance required. Do not fly."


def _parse_upload(buf: bytes) -> tuple[pd.DataFrame | None, str | None]:
    """Parse uploaded CMAPSS file bytes. Returns (df, None) or (None, error)."""
    df = None

    try:
        candidate = parse_cmapss_bytes(buf, ALL_COLUMNS)
        if pd.api.types.is_numeric_dtype(candidate.iloc[:, 0]):
            df = candidate
    except Exception:
        pass

    if df is None:
        try:
            candidate = pd.read_csv(
                io.BytesIO(buf), sep=r"\s+", header=0, engine="python"
            ).dropna(axis=1, how="all")
            df = candidate
        except Exception:
            pass

    if df is None:
        try:
            df = pd.read_csv(io.BytesIO(buf), sep=",", header=0).dropna(axis=1, how="all")
        except Exception:
            return None, "File could not be parsed. Upload a valid CMAPSS CSV or text file."

    if df is None:
        return None, "File could not be parsed."

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sample = ", ".join(missing[:5]) + ("..." if len(missing) > 5 else "")
        return None, (
            f"Missing {len(missing)} required column(s): {sample}. "
            "Upload a file with the 24 CMAPSS sensor and op-setting columns."
        )
    for col in REQUIRED_COLS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return None, f"Column '{col}' contains non-numeric values."
    if len(df) == 0:
        return None, "Uploaded file has no data rows."
    return df, None


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------


def _gauge_fig(rul: float, colour: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=max(0.0, min(float(rul), RUL_CAP)),
        number={"suffix": " cycles", "font": {"size": 28, "color": C_TEXT}},
        title={"text": "Remaining Useful Life", "font": {"size": 15, "color": C_MUTED}},
        gauge={
            "axis": {"range": [0, RUL_CAP], "tickwidth": 1,
                     "tickcolor": C_MUTED, "tickfont": {"color": C_MUTED}},
            "bar": {"color": colour, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, RUL_THRESHOLD],          "color": "rgba(239,68,68,0.15)"},
                {"range": [RUL_THRESHOLD, WARNING_LEVEL], "color": "rgba(245,158,11,0.12)"},
                {"range": [WARNING_LEVEL, RUL_CAP],    "color": "rgba(34,197,94,0.10)"},
            ],
            "threshold": {
                "line": {"color": C_CRITICAL, "width": 2},
                "thickness": 0.75,
                "value": RUL_THRESHOLD,
            },
        },
    ))
    fig.update_layout(
        height=260,
        margin=dict(t=30, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color=C_TEXT,
    )
    return fig


def _trajectory_fig(cycle_ends: list[int], rul_values: list[float],
                    colour: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cycle_ends, y=rul_values,
        mode="lines", name="Predicted RUL",
        line=dict(color=C_ACCENT, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=[cycle_ends[-1]], y=[rul_values[-1]],
        mode="markers", name="Current position",
        marker=dict(color=colour, size=11, symbol="circle",
                    line=dict(color=C_TEXT, width=1)),
        showlegend=True,
    ))
    fig.add_hline(y=RUL_THRESHOLD, line_dash="dash", line_color=C_CRITICAL,
                  annotation_text="Critical threshold (30 cycles)",
                  annotation_font_color=C_CRITICAL,
                  annotation_position="top right")
    fig.add_hline(y=WARNING_LEVEL, line_dash="dot", line_color=C_WARNING,
                  annotation_text="Warning (60 cycles)",
                  annotation_font_color=C_WARNING,
                  annotation_position="top right")
    fig.update_layout(
        xaxis_title="Engine cycle",
        yaxis_title="Predicted RUL (cycles)",
        yaxis=dict(range=[0, RUL_CAP + 10]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        height=320,
        margin=dict(t=50, b=40, l=50, r=20),
    )
    return _apply_dark_theme(fig)


def _sensor_fig(df: pd.DataFrame, col: str) -> go.Figure:
    x = (df["time_in_cycles"].values if "time_in_cycles" in df.columns
         else list(range(1, len(df) + 1)))
    sym, desc, units = SENSOR_INFO.get(col, (col, col, ""))
    fig = go.Figure(go.Scatter(
        x=x, y=df[col].values,
        mode="lines",
        line=dict(color=C_ACCENT, width=1.8),
        name=sym,
    ))
    y_label = f"{sym} ({units})" if units and units != "--" else sym
    fig.update_layout(
        xaxis_title="Engine cycle",
        yaxis_title=y_label,
        title=dict(text=f"{sym}  --  {desc}", font=dict(color=C_TEXT, size=14)),
        height=300,
        margin=dict(t=50, b=40, l=55, r=20),
    )
    return _apply_dark_theme(fig)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            f'<div style="text-align:center;padding:12px 0 4px">'
            f'<span style="font-size:1.35rem;font-weight:700;color:{C_ACCENT}">'
            f'Aircraft Engine<br>Health Monitor</span><br>'
            f'<span style="font-size:0.78rem;color:{C_MUTED}">'
            f'Predictive Maintenance Research Dashboard</span></div>',
            unsafe_allow_html=True,
        )

        st.divider()

        st.markdown(
            f'<div style="font-size:0.78rem;color:{C_MUTED};line-height:2">'
            f'Dataset &nbsp; <b style="color:{C_TEXT}">NASA CMAPSS FD001</b><br>'
            f'Production model &nbsp; <b style="color:{C_TEXT}">LSTM</b><br>'
            f'Test RMSE &nbsp; <b style="color:{C_TEXT}">15.04 cycles</b><br>'
            f'Low-RUL bias &nbsp; <b style="color:{C_HEALTHY}">-0.43 (conservative)</b>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.divider()

        st.markdown(
            f'<p style="font-size:0.84rem;color:{C_MUTED};line-height:1.6">'
            "This dashboard demonstrates machine learning-based Remaining Useful Life (RUL) "
            "prediction for turbofan engines using the NASA CMAPSS benchmark dataset. "
            "An LSTM neural network trained with an asymmetric loss function prioritises "
            "conservative (safe) predictions near engine failure."
            "</p>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div style="font-size:0.84rem;line-height:2.2">'
            f'<a href="https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data/xaut-bemq" '
            f'target="_blank" style="color:{C_ACCENT};text-decoration:none">'
            f'NASA CMAPSS Dataset</a><br>'
            f'<a href="https://github.com/satyam3112003" '
            f'target="_blank" style="color:{C_ACCENT};text-decoration:none">'
            f'View on GitHub</a>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.divider()

        with st.expander("Research Disclaimer", expanded=False):
            st.markdown(
                f'<p style="font-size:0.8rem;color:{C_MUTED};line-height:1.6">'
                "This tool is a research prototype developed for educational and "
                "portfolio purposes only. Predictions generated by this system must "
                "not be used as the sole basis for airworthiness determinations, "
                "maintenance decisions, or flight operations. Always consult qualified "
                "aviation maintenance personnel and follow applicable airworthiness "
                "directives and regulations. This system is not certified for "
                "operational use under any aviation authority (FAA, EASA, or equivalent)."
                "</p>",
                unsafe_allow_html=True,
            )

        with st.expander("Terms of Use", expanded=False):
            st.markdown(
                f'<p style="font-size:0.8rem;color:{C_MUTED};line-height:1.6">'
                "By using this dashboard you agree that: (1) predictions are provided "
                "for informational and research purposes only; (2) the authors accept no "
                "liability for decisions made based on these predictions; (3) the NASA "
                "CMAPSS dataset is used under its original public-domain terms; "
                "(4) this software is provided AS IS, without warranty of any kind."
                "</p>",
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<p style="font-size:0.72rem;color:{C_BORDER};text-align:center;'
            f'margin-top:16px">Phase 9 -- RUL Prediction Project</p>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Tab 1 -- Dataset Overview
# ---------------------------------------------------------------------------


def _tab_dataset() -> None:
    st.header("FD001 Dataset Overview")

    try:
        train_df = _train_stats()
    except Exception:
        st.info(
            "Raw dataset files are not bundled with this deployment. "
            "To explore the training data locally, download the NASA CMAPSS FD001 files "
            "and place them at `data/raw/FD001/`. "
            "Use the **Upload & Predict** tab to run live RUL predictions on any engine file."
        )
        return

    n_engines = int(train_df["unit_number"].nunique())
    n_cycles  = len(train_df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training Engines",   n_engines)
    c2.metric("Total Cycles",       f"{n_cycles:,}")
    c3.metric("Model Features",     "17")
    c4.metric("RUL Cap",            f"{RUL_CAP} cycles")

    st.divider()

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("RUL Distribution (Training Set)")
        rul_capped = train_df["RUL"].clip(upper=RUL_CAP)
        fig = go.Figure(go.Histogram(
            x=rul_capped, nbinsx=40,
            marker_color=C_ACCENT, opacity=0.85,
        ))
        fig.update_layout(
            xaxis_title="RUL (cycles, capped at 125)",
            yaxis_title="Window count",
            height=280,
            margin=dict(t=10, b=40, l=50, r=10),
        )
        st.plotly_chart(_apply_dark_theme(fig), width="stretch")

    with col_right:
        st.subheader("Sensor Reference")
        rows = []
        for col, (sym, desc, units) in SENSOR_INFO.items():
            rows.append({
                "Column":      col,
                "Symbol":      sym,
                "Description": desc,
                "Units":       units,
                "In Model":    col not in NON_INFORMATIVE,
            })
        ref_df = pd.DataFrame(rows)
        st.dataframe(
            ref_df,
            hide_index=True,
            height=295,
            column_config={
                "In Model": st.column_config.CheckboxColumn(
                    "In Model", width="small"
                ),
            },
        )

    st.caption(
        "7 sensors are near-constant in FD001 and excluded from the model: "
        + ", ".join(sorted(NON_INFORMATIVE))
        + ".  Remaining: 14 sensors + 3 operating settings = 17 features."
    )


# ---------------------------------------------------------------------------
# Tab 2 -- Upload & Predict
# ---------------------------------------------------------------------------


def _tab_predict() -> None:
    st.header("Engine Health Prediction")
    st.markdown(
        f'<p style="color:{C_MUTED}">'
        "Upload a raw engine sensor file in CMAPSS format (space-delimited or CSV, "
        "with or without column headers). "
        "Required columns: op_setting_1-3 and sensor_1-21."
        "</p>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Drop sensor file here",
        type=["csv", "txt"],
        key="uploader",
    )

    if uploaded is None:
        _callout(
            "Upload a CMAPSS sensor file to generate a health prediction. "
            "Use <b>test_FD001.txt</b> to try with the benchmark test set.",
            C_ACCENT,
        )
        return

    raw_bytes = uploaded.read()
    df, err = _parse_upload(raw_bytes)
    if err:
        st.error(f"Upload error: {err}")
        return

    # ---- Engine selection ----
    if "unit_number" in df.columns and df["unit_number"].nunique() > 1:
        engine_ids = sorted(df["unit_number"].unique().tolist())
        chosen = st.selectbox(
            f"File contains {len(engine_ids)} engines -- select one to analyse:",
            options=engine_ids,
            format_func=lambda x: f"Engine {int(x)}",
        )
        engine_df = df[df["unit_number"] == chosen].reset_index(drop=True)
        engine_id = int(chosen)
    else:
        engine_df = df.reset_index(drop=True)
        engine_id = (int(df["unit_number"].iloc[0])
                     if "unit_number" in df.columns else 1)

    n_cycles_up = len(engine_df)
    st.success(f"Engine {engine_id} selected -- {n_cycles_up} cycles loaded.")

    pipeline = _pipeline()

    with st.spinner("Running LSTM inference..."):
        result = pipeline.predict(engine_df, engine_id=engine_id)

    label, colour, recommendation = _risk_tier(result.predicted_rul)
    model_display = result.model_used.split("_")[0].upper()
    safety_label  = "FLAGGED" if result.safety_flag else "NOMINAL"
    safety_colour = C_CRITICAL if result.safety_flag else C_HEALTHY

    # ---- Gauge + info panel ----
    col_gauge, col_info = st.columns([1, 1])

    with col_gauge:
        st.plotly_chart(_gauge_fig(result.predicted_rul, colour), width="stretch")

    with col_info:
        st.markdown(
            f'<p style="font-size:0.82rem;color:{C_MUTED};margin-bottom:4px">'
            f'RISK STATUS</p>',
            unsafe_allow_html=True,
        )
        _badge(label, colour)

        st.markdown("<br>", unsafe_allow_html=True)
        st.metric(
            label="Predicted RUL",
            value=f"{result.predicted_rul:.1f} cycles",
        )
        st.metric(
            label="Cycles Analyzed",
            value=f"{n_cycles_up}",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<p style="font-size:0.82rem;color:{C_MUTED};margin-bottom:6px">'
            f'MODEL &nbsp;&nbsp; SAFETY STATUS</p>',
            unsafe_allow_html=True,
        )
        col_b1, col_b2 = st.columns([1, 1])
        with col_b1:
            _badge(model_display, C_ACCENT)
        with col_b2:
            _badge(safety_label, safety_colour)

    # ---- Recommendation ----
    st.markdown("<br>", unsafe_allow_html=True)
    if label == "CRITICAL":
        st.error(f"Maintenance Action: {recommendation}")
    elif label == "WARNING":
        st.warning(f"Maintenance Action: {recommendation}")
    else:
        st.success(f"Status: {recommendation}")

    st.divider()

    # ---- Remaining life trend ----
    st.subheader("Remaining Life Trend")

    with st.spinner("Computing full trajectory..."):
        trajectory = pipeline.predict_trajectory(engine_df, engine_id=engine_id)

    window_size = pipeline.window_size
    cycle_ends = (list(range(window_size, n_cycles_up + 1))
                  if n_cycles_up >= window_size else [n_cycles_up])
    rul_values = [r.predicted_rul for r in trajectory]

    st.plotly_chart(
        _trajectory_fig(cycle_ends, rul_values, colour),
        width="stretch",
    )

    st.divider()

    # ---- Sensor visualization ----
    st.subheader("Sensor Degradation Profile")

    informative = [c for c in SENSOR_INFO
                   if c in engine_df.columns and c not in NON_INFORMATIVE]
    all_plottable = (
        [c for c in [f"op_setting_{i}" for i in range(1, 4)]
         if c in engine_df.columns]
        + informative
        + [c for c in NON_INFORMATIVE if c in engine_df.columns]
    )
    default_idx = informative.index("sensor_11") if "sensor_11" in informative else 0

    chosen_col = st.selectbox(
        "Select sensor or operating setting:",
        options=all_plottable,
        format_func=_sensor_label,
        index=default_idx,
    )

    if chosen_col in NON_INFORMATIVE:
        st.caption(
            "This sensor is near-constant in FD001 and is excluded from the model."
        )

    st.plotly_chart(_sensor_fig(engine_df, chosen_col), width="stretch")


# ---------------------------------------------------------------------------
# Tab 3 -- Model Comparison
# ---------------------------------------------------------------------------


def _tab_comparison() -> None:
    st.header("Model Comparison")

    df = _model_comparison()

    _callout(
        "<b>Why LSTM?</b>  LSTM achieves the only negative (conservative) low-RUL bias "
        "of -0.43 cycles, meaning it <i>underestimates</i> RUL near failure -- the safe "
        "direction. BiLSTM (+1.42) and XGBoost (+2.54) both overestimate, which risks "
        "missing a failing engine. LSTM is also 7x faster than BiLSTM (18 vs 134 "
        "microseconds per sample) and half the size (86 KB vs 171 KB). "
        "Overall RMSE difference vs BiLSTM is 0.02 -- statistically noise.",
        C_ACCENT,
    )

    st.divider()
    st.subheader("Full Metrics Table")

    display_cols = [
        "model", "test_rmse", "test_mae", "test_r2",
        "low_rul_rmse", "low_rul_bias", "inference_us_per_sample", "model_size_kb",
    ]
    tbl = df[[c for c in display_cols if c in df.columns]].copy()
    col_labels = {
        "model":                    "Model",
        "test_rmse":                "RMSE",
        "test_mae":                 "MAE",
        "test_r2":                  "R2",
        "low_rul_rmse":             "Low-RUL RMSE",
        "low_rul_bias":             "Low-RUL Bias",
        "inference_us_per_sample":  "Speed (us/sample)",
        "model_size_kb":            "Size (KB)",
    }
    tbl.columns = [col_labels.get(c, c) for c in tbl.columns]

    def _colour_bias(val: float) -> str:
        if isinstance(val, float) and val > 0:
            return f"color: {C_CRITICAL}"
        if isinstance(val, float) and val < 0:
            return f"color: {C_HEALTHY}"
        return ""

    styled = (tbl.style
              .map(_colour_bias, subset=["Low-RUL Bias"])
              .format(precision=3))
    st.dataframe(styled, hide_index=True, width="stretch")

    st.markdown(
        f'<p style="font-size:0.8rem;color:{C_MUTED}">'
        "Low-RUL Bias: positive = overestimates RUL near failure (unsafe, shown red). "
        "Negative = underestimates (conservative, safe, shown green)."
        "</p>",
        unsafe_allow_html=True,
    )

    st.divider()

    models  = df["model"].tolist()
    colours = [
        C_ACCENT if m == _PROD_MODEL else "#475569"
        for m in models
    ]

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Overall RMSE")
        fig = go.Figure(go.Bar(x=models, y=df["test_rmse"], marker_color=colours))
        fig.update_layout(yaxis_title="RMSE (cycles)", height=260,
                          margin=dict(t=10, b=40, l=50, r=10))
        st.plotly_chart(_apply_dark_theme(fig), width="stretch")

    with c2:
        st.subheader("Low-RUL Bias (near failure)")
        bias_colours = [C_CRITICAL if v > 0 else C_HEALTHY
                        for v in df["low_rul_bias"]]
        fig = go.Figure(go.Bar(x=models, y=df["low_rul_bias"],
                               marker_color=bias_colours))
        fig.add_hline(y=0, line_dash="dash", line_color=C_MUTED)
        fig.update_layout(
            yaxis_title="Bias (cycles)",
            height=260, margin=dict(t=10, b=40, l=50, r=10),
        )
        st.plotly_chart(_apply_dark_theme(fig), width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Inference Speed (microseconds / sample)")
        fig = go.Figure(go.Bar(
            x=models, y=df["inference_us_per_sample"], marker_color=colours,
        ))
        fig.update_layout(yaxis_title="microseconds / sample", height=260,
                          margin=dict(t=10, b=40, l=50, r=10))
        st.plotly_chart(_apply_dark_theme(fig), width="stretch")

    with c4:
        st.subheader("Model File Size (KB)")
        fig = go.Figure(go.Bar(x=models, y=df["model_size_kb"],
                               marker_color=colours))
        fig.update_layout(yaxis_title="KB", height=260,
                          margin=dict(t=10, b=40, l=50, r=10))
        st.plotly_chart(_apply_dark_theme(fig), width="stretch")

    # Production model callout
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f'<p style="color:{C_MUTED};font-size:0.84rem">'
        f'Production model: &nbsp;</p>',
        unsafe_allow_html=True,
    )
    _badge("LSTM  --  SELECTED FOR PRODUCTION", C_ACCENT)


# ---------------------------------------------------------------------------
# Tab 4 -- SHAP Explainability
# ---------------------------------------------------------------------------


def _show_fig(path: Path, caption: str = "") -> None:
    with st.container():
        if path.exists():
            st.image(str(path), width="stretch")
            if caption:
                st.caption(caption)
        else:
            st.warning(
                f"Figure not found: {path.name}. "
                "Re-run the explainer to generate it."
            )


def _tab_shap() -> None:
    st.header("SHAP Explainability")
    st.markdown(
        f'<p style="color:{C_MUTED}">'
        "SHAP (SHapley Additive exPlanations) quantifies each sensor's contribution "
        "to individual predictions. Figures are pre-generated from the LSTM (production) "
        "and XGBoost (baseline) models on all 100 FD001 test engines."
        "</p>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.subheader("Global Feature Importance")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(
            f'<p style="font-weight:600;color:{C_ACCENT}">LSTM (production)</p>',
            unsafe_allow_html=True,
        )
        _show_fig(_FIGURES / "shap_global_lstm.png",
                  "Mean absolute SHAP value per sensor -- LSTM")
        _show_fig(_FIGURES / "shap_beeswarm_lstm.png",
                  "Beeswarm: feature impact vs value -- LSTM")

    with col_r:
        st.markdown(
            f'<p style="font-weight:600;color:{C_MUTED}">XGBoost (baseline)</p>',
            unsafe_allow_html=True,
        )
        _show_fig(_FIGURES / "shap_global_xgboost.png",
                  "Mean absolute SHAP value per sensor -- XGBoost")
        _show_fig(_FIGURES / "shap_beeswarm_xgboost.png",
                  "Beeswarm: feature impact vs value -- XGBoost")

    _callout(
        "<b>Key finding:</b> Both models independently rank <b>sensor_11 (Ps30 -- "
        "HPC static pressure)</b> as their top predictor, consistent with Phase 2 EDA "
        "and known High Pressure Compressor degradation physics. sensor_4 (T50 -- LPT "
        "outlet temperature) ranks 3rd in both. All four EDA top-4 sensors appear in "
        "the LSTM top-10, validating model-EDA alignment.",
        C_HEALTHY,
    )

    st.divider()
    st.subheader("Local Explanations -- Individual Engine Analysis")
    st.markdown(
        f'<p style="color:{C_MUTED};font-size:0.9rem">'
        "Waterfall plots show how each sensor pushed the predicted RUL above or below "
        "the model's average prediction (base value) for three representative test engines."
        "</p>",
        unsafe_allow_html=True,
    )

    engine_options = {
        "Engine 34  --  Near failure (true RUL = 7 cycles)":  "engine34",
        "Engine 19  --  Median health":                        "engine19",
        "Engine 25  --  High RUL (far from failure)":          "engine25",
    }
    selected_label = st.selectbox(
        "Select engine for local explanation:",
        list(engine_options.keys()),
    )
    engine_key = engine_options[selected_label]

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(
            f'<p style="font-weight:600;color:{C_ACCENT}">LSTM</p>',
            unsafe_allow_html=True,
        )
        _show_fig(_FIGURES / f"shap_local_lstm_{engine_key}.png")
    with col_r:
        st.markdown(
            f'<p style="font-weight:600;color:{C_MUTED}">XGBoost</p>',
            unsafe_allow_html=True,
        )
        _show_fig(_FIGURES / f"shap_local_xgboost_{engine_key}.png")

    _callout(
        "For near-failure Engine 34, sensor_11 and sensor_4 are the dominant negative "
        "contributors -- they pull the predicted RUL well below the model baseline, "
        "reflecting active HPC deterioration consistent with the FD001 fault mode.",
        C_WARNING,
    )

    st.divider()


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Engine Health Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
)

_render_sidebar()

st.markdown(
    f'<h1 style="color:{C_ACCENT};margin-bottom:0;font-size:1.9rem">'
    f'Aircraft Engine Health Monitor</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<p style="color:{C_MUTED};margin-top:2px;margin-bottom:18px">'
    f'NASA CMAPSS FD001 &nbsp;&#124;&nbsp; LSTM Production Model'
    f'&nbsp;&#124;&nbsp; RUL Prediction Research Dashboard</p>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Dataset Overview",
    "Upload & Predict",
    "Model Comparison",
    "SHAP Explainability",
])

with tab1:
    _tab_dataset()

with tab2:
    _tab_predict()

with tab3:
    _tab_comparison()

with tab4:
    _tab_shap()
