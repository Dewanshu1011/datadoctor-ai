"""DataDoctor AI — a privacy-conscious data quality copilot."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from ai_analyzer import generate_ai_explanation, generate_fix
from profiler import build_report, load_dataset, report_to_markdown


ROOT = Path(__file__).parent
SAMPLE_PATH = ROOT / "sample_data" / "customer_orders_demo.csv"

st.set_page_config(page_title="DataDoctor AI", page_icon="🩺", layout="wide")

# Streamlit Community Cloud exposes configured secrets through st.secrets; make
# them available to the small service module without ever rendering the value.
for _secret_name in ("OPENAI_API_KEY", "OPENAI_MODEL"):
    try:
        if not os.getenv(_secret_name) and st.secrets.get(_secret_name):
            os.environ[_secret_name] = str(st.secrets[_secret_name])
    except FileNotFoundError:
        pass


def apply_styles() -> None:
    st.markdown(
        """<style>
        :root { --ink: #16213a; --muted: #667085; --line: #e7ebf3; --blue: #4169e1; --mint: #10b981; }
        .stApp { background: #f6f8fc; color: var(--ink); }
        [data-testid="stHeader"] { background: rgba(246,248,252,.85); }
        [data-testid="stSidebar"] { background: #101c35; }
        [data-testid="stSidebar"] * { color: #edf2ff; }
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { background: rgba(255,255,255,.08); border: 1px dashed rgba(255,255,255,.35); }
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button { color: #18233e; }
        [data-testid="stSidebar"] .stButton button { border: 1px solid rgba(255,255,255,.25); background: rgba(255,255,255,.08); }
        [data-testid="stSidebar"] .stButton button:hover { border-color: #8ba3ff; background: rgba(106,133,255,.2); }
        .hero { padding: 2rem 2.1rem; margin: .25rem 0 1.5rem; border-radius: 20px; color: white;
                background: radial-gradient(circle at 90% 0%, #6785ff 0, transparent 31%), linear-gradient(120deg, #101c35 0%, #1d3971 100%); box-shadow: 0 14px 32px rgba(29,57,113,.18); }
        .hero h1 { color: white; margin: .15rem 0 .35rem; font-size: 2.3rem; letter-spacing: -.04em; }
        .hero p { color: #dbe6ff; margin: 0; font-size: 1.05rem; }
        .eyebrow { color: #a9bcff; font-size: .75rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
        .section-kicker { color: #667085; font-size: .8rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; margin: .25rem 0 .4rem; }
        .score-card { min-height: 109px; padding: 1.05rem 1.2rem; border: 1px solid var(--line); border-radius: 14px; background: white; box-shadow: 0 4px 12px rgba(20,33,61,.045); }
        .score-label { color: var(--muted); font-size: .82rem; font-weight: 650; text-transform: uppercase; letter-spacing: .06em; }
        .score-value { font-size: 2.2rem; line-height: 1.15; font-weight: 750; letter-spacing: -.05em; }
        .score-caption { color: var(--muted); font-size: .84rem; }
        [data-testid="stMetric"] { background: white; border: 1px solid var(--line); border-radius: 14px; padding: 1rem 1.05rem; min-height: 109px; box-shadow: 0 4px 12px rgba(20,33,61,.045); }
        [data-testid="stMetricLabel"] { color: var(--muted); font-size: .84rem; }
        [data-testid="stMetricValue"] { color: var(--ink); font-size: 1.65rem; }
        .stTabs [data-baseweb="tab-list"] { gap: .35rem; border-bottom: 1px solid var(--line); }
        .stTabs [data-baseweb="tab"] { height: 42px; border-radius: 9px 9px 0 0; color: #667085; font-weight: 600; }
        .stTabs [aria-selected="true"] { color: var(--blue); background: #eef2ff; }
        .stTabs [data-baseweb="tab-highlight"] { background: var(--blue); }
        .stButton button { border-radius: 9px; font-weight: 650; border-color: #d6ddec; }
        .stButton button[kind="primary"] { background: var(--blue); border-color: var(--blue); }
        .stButton button[kind="primary"]:hover { background: #3458c7; border-color: #3458c7; }
        .feature-card { height: 100%; padding: 1.25rem; border: 1px solid var(--line); border-radius: 14px; background: white; box-shadow: 0 4px 12px rgba(20,33,61,.04); }
        .feature-card h3 { margin: .55rem 0 .35rem; color: var(--ink); font-size: 1.02rem; }
        .feature-card p { color: var(--muted); font-size: .91rem; margin: 0; }
        .feature-icon { font-size: 1.4rem; }
        [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
        [data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 10px; background: white; }
        </style>""",
        unsafe_allow_html=True,
    )


def severity_counts(report: dict) -> dict[str, int]:
    counts = {"critical": 0, "warning": 0, "passed": 0}
    for finding in report["findings"]:
        counts[finding["severity"]] = counts.get(finding["severity"], 0) + 1
    return counts


@st.cache_data(show_spinner=False)
def profile_cached(file_bytes: bytes, filename: str) -> tuple[pd.DataFrame, dict]:
    data = load_dataset(file_bytes, filename)
    return data, build_report(data, filename)


def get_data() -> tuple[pd.DataFrame | None, dict | None, str | None]:
    upload = st.session_state.get("uploaded_file")
    use_demo = st.session_state.get("use_demo", False)
    if upload is not None:
        return (*profile_cached(upload.getvalue(), upload.name), upload.name)
    if use_demo:
        return (*profile_cached(SAMPLE_PATH.read_bytes(), SAMPLE_PATH.name), SAMPLE_PATH.name)
    return None, None, None


def main() -> None:
    apply_styles()
    st.markdown("<div class='hero'><div class='eyebrow'>Data quality intelligence</div><h1>DataDoctor AI</h1><p>Find, understand, and fix data issues before they reach production.</p></div>", unsafe_allow_html=True)

    with st.sidebar:
        st.header("Dataset")
        st.file_uploader("Upload CSV or Parquet", type=["csv", "parquet"], key="uploaded_file")
        if st.button("Use demo dataset", width="stretch"):
            st.session_state.use_demo = True
        if st.button("Clear dataset", width="stretch"):
            st.session_state.use_demo = False
            st.session_state.uploaded_file = None
            st.rerun()
        st.divider()
        st.caption("Your data is profiled locally. AI requests contain only aggregated, anonymized metadata.")

    try:
        df, report, filename = get_data()
    except Exception as exc:
        st.error(f"We could not read that file: {exc}")
        return

    if df is None or report is None:
        st.markdown("<div class='section-kicker'>Start a health check</div>", unsafe_allow_html=True)
        st.markdown("### Your data deserves a second opinion.")
        st.write("Upload a CSV or Parquet file, or open the demo dataset to see a complete quality assessment in seconds.")
        one, two, three = st.columns(3)
        with one:
            st.markdown("<div class='feature-card'><div class='feature-icon'>🔎</div><h3>Profile automatically</h3><p>Inspect types, completeness, cardinality, duplicates, and numeric distributions locally.</p></div>", unsafe_allow_html=True)
        with two:
            st.markdown("<div class='feature-card'><div class='feature-icon'>🛡️</div><h3>Prioritize risk</h3><p>Get a transparent 0–100 score and clearly separated critical issues, warnings, and passes.</p></div>", unsafe_allow_html=True)
        with three:
            st.markdown("<div class='feature-card'><div class='feature-icon'>⚡</div><h3>Fix with confidence</h3><p>Turn findings into a remediation plan, SQL, or PySpark templates without sharing raw rows.</p></div>", unsafe_allow_html=True)
        st.info("Choose **Use demo dataset** in the sidebar to explore the full dashboard.")
        return

    counts = severity_counts(report)
    score = report["quality_score"]
    score_color = "#146c43" if score >= 85 else "#b54708" if score >= 60 else "#b42318"
    st.markdown("<div class='section-kicker'>Dataset health report</div>", unsafe_allow_html=True)
    st.caption(f"Analysing: **{filename}**")
    a, b, c, d = st.columns(4)
    a.markdown(f"<div class='score-card'><div class='score-label'>Data quality score</div><div class='score-value' style='color:{score_color}'>{score}<span style='font-size:1rem'>/100</span></div><div class='score-caption'>{report['score_label']}</div></div>", unsafe_allow_html=True)
    b.metric("Rows", f"{report['overview']['rows']:,}")
    c.metric("Columns", report["overview"]["columns"])
    d.metric("File size", report["overview"]["estimated_memory"])
    st.progress(score / 100, text=f"Overall health: {report['score_label']}")

    tabs = st.tabs(["Overview", "Findings", "Column profile", "AI Copilot", "Report"])
    with tabs[0]:
        left, right = st.columns([1.15, 1])
        with left:
            st.subheader("Data preview")
            st.dataframe(df.head(20), width="stretch", hide_index=True)
        with right:
            st.subheader("Dataset details")
            st.json(report["overview"])
            st.subheader("Detected types")
            st.dataframe(pd.DataFrame(report["columns"])[["column", "dtype", "non_null", "unique", "missing_pct"]], width="stretch", hide_index=True)

    with tabs[1]:
        x, y, z = st.columns(3)
        x.metric("Critical issues", counts["critical"])
        y.metric("Warnings", counts["warning"])
        z.metric("Passed checks", counts["passed"])
        for finding in report["findings"]:
            icon = {"critical": "🚨", "warning": "⚠️", "passed": "✅"}[finding["severity"]]
            with st.expander(f"{icon} {finding['title']} — {finding['severity'].title()}", expanded=finding["severity"] == "critical"):
                st.write(finding["description"])
                if finding.get("columns"):
                    st.caption("Columns: " + ", ".join(finding["columns"]))

    with tabs[2]:
        profile = pd.DataFrame(report["columns"])
        st.dataframe(profile, width="stretch", hide_index=True)

    with tabs[3]:
        st.subheader("AI interpretation & remediation")
        st.caption("Uses your OPENAI_API_KEY if configured. Only the generated profile and findings are sent, never raw rows.")
        if st.button("Explain detected problems", type="primary"):
            with st.spinner("Preparing an expert review..."):
                result = generate_ai_explanation(report)
            st.markdown(result)
        fix_col1, fix_col2 = st.columns(2)
        with fix_col1:
            if st.button("Generate SQL Fix", width="stretch"):
                with st.spinner("Drafting SQL..."):
                    st.code(generate_fix(report, "SQL"), language="sql")
        with fix_col2:
            if st.button("Generate PySpark Fix", width="stretch"):
                with st.spinner("Drafting PySpark..."):
                    st.code(generate_fix(report, "PySpark"), language="python")

    with tabs[4]:
        st.subheader("Downloadable report")
        report_json = json.dumps(report, indent=2, default=str)
        st.download_button("Download JSON report", report_json, "datadoctor_report.json", "application/json")
        st.download_button("Download Markdown report", report_to_markdown(report), "datadoctor_report.md", "text/markdown")


if __name__ == "__main__":
    main()
