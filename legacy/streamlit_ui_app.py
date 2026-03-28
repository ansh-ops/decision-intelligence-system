import sys
from pathlib import Path

# -------------------------------
# Fix Python path FIRST
# -------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -------------------------------
# Import Streamlit (no Streamlit calls before set_page_config)
# -------------------------------
import streamlit as st

st.set_page_config(
    page_title="Decision Intelligence System",
    layout="wide",
)

# -------------------------------
# Now import everything else
# -------------------------------
import pandas as pd
import inspect

from pipelines.run_analysis import run_pipeline
import agents.explainability_agent as ea
import pipelines.run_analysis as ra


def format_decision_report_local(report: dict) -> str:
    cv = report.get("cv_performance", {})

    cv_mean = (
        cv.get("cv_mean")
        or cv.get("mean")
        or cv.get("roc_auc")
        or cv.get("score")
    )
    cv_std = cv.get("cv_std") or cv.get("std")

    lines = ["## 📌 Executive Summary"]
    lines.append(f"- **Selected model:** {report.get('model_selected') or 'Unknown'}")

    if isinstance(cv_mean, (int, float)):
        if isinstance(cv_std, (int, float)):
            lines.append(f"- **CV:** {cv_mean:.3f} ± {cv_std:.3f}")
        else:
            lines.append(f"- **CV:** {cv_mean:.3f}")

    overall = report.get("overall_metrics", {})
    if overall:
        for k, v in overall.items():
            if isinstance(v, (int, float)):
                lines.append(f"- **{k.upper()} (test):** {v:.3f}")

    return "\n".join(lines)


# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("📊 Autonomous Decision Intelligence System")
st.markdown(
    """
This tool performs **end-to-end applied data science**:
- Exploratory analysis
- Model selection & validation
- Explainability (SHAP)
- Risk segmentation
- Decision intelligence output
"""
)
st.divider()

# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------
st.sidebar.header("⚙️ Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV",
    type=["csv"],
)

target_override = st.sidebar.text_input(
    "Target column (optional)",
    placeholder="e.g. Churn",
)

show_debug = st.sidebar.checkbox("Show debug info", value=False)
run_button = st.sidebar.button("🚀 Run Analysis")

# Optional debug paths (only after set_page_config)
if show_debug:
    st.sidebar.write("run_analysis.py:", inspect.getfile(ra))
    st.sidebar.write("ExplainabilityAgent:", inspect.getfile(ea))
    st.sidebar.write("ExplainabilityAgent class:", ea.ExplainabilityAgent)

# --------------------------------------------------
# Main execution
# --------------------------------------------------
if run_button:
    if uploaded_file is None:
        st.warning("Please upload a dataset.")
        st.stop()

    temp_path = Path("/tmp") / uploaded_file.name
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Running full decision intelligence pipeline..."):
        try:
            result = run_pipeline(
                str(temp_path),
                target_override=target_override or None,
            )
        except Exception as e:
            st.error("Pipeline failed.")
            st.exception(e)
            st.stop()

    st.success("Analysis complete!")

    # --------------------------------------------------
    # Overview
    # --------------------------------------------------
    st.header("🔍 Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Target", result.get("target", ""))
    col2.metric("Task Type", result.get("task_type", ""))
    col3.metric("Test Rows", result.get("splits", {}).get("test_rows", ""))

    # --------------------------------------------------
    # Performance
    # --------------------------------------------------
    st.header("📈 Model Performance (Out-of-Sample)")
    metrics = result.get("evaluation", {}).get("overall_metrics", {})
    if metrics:
        metric_cols = st.columns(len(metrics))
        for col, (k, v) in zip(metric_cols, metrics.items()):
            if isinstance(v, (int, float)):
                col.metric(k.upper(), round(v, 3))
            else:
                col.metric(k.upper(), str(v))

    thr_info = result.get("evaluation", {}).get("threshold_info")
    if thr_info:
        st.caption(
            f"Optimized decision threshold: {round(result['evaluation']['threshold'], 3)}"
        )

    # --------------------------------------------------
    # Explainability
    # --------------------------------------------------
    st.header("🧠 Key Drivers (Explainability)")
    expl = result.get("week3", {}).get("explainability")
    if isinstance(expl, list) and len(expl) > 0:
        expl_df = pd.DataFrame(expl)
        st.dataframe(expl_df.head(30), use_container_width=True, height=320)
    else:
        st.info("Explainability not available for this run.")

    # --------------------------------------------------
    # Risk segmentation
    # --------------------------------------------------
    seg = result.get("week3", {}).get("segmentation")
    if seg:
        st.header("🎯 Risk Segmentation")
        seg_df = pd.DataFrame(seg)
        st.dataframe(seg_df, use_container_width=True, height=260)

    # --------------------------------------------------
    # Error analysis
    # --------------------------------------------------
    st.header("⚠️ Error Analysis")
    error_info = result.get("week3", {}).get("error_analysis")
    if error_info:
        st.json(error_info)
    else:
        st.info("Error analysis not available for this run.")

    # --------------------------------------------------
    # Decision report
    # --------------------------------------------------
    st.header("📌 Decision Intelligence Summary")
    decision = result.get("week3", {}).get("decision_report", {})
    st.write("modeling best:", result["modeling"]["best_model"])
    st.write("decision report:", decision)
    pretty_report = format_decision_report_local(decision)
    st.markdown(pretty_report)

    if show_debug:
        st.subheader("🧪 Debug: Explainability Raw")
        st.write("type(expl):", type(expl))
        if expl is None:
            st.error("explainability is None (not computed / not returned).")
        elif isinstance(expl, list):
            st.write("len(expl):", len(expl))
            st.write("first item:", expl[0] if len(expl) else "EMPTY LIST")
        else:
            st.write(expl)

    st.divider()
    st.caption("Deterministic system • No LLM dependency • Decision-focused output")
