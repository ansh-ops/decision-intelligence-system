def format_decision_report(report: dict) -> str:
    """
    Converts raw decision intelligence JSON into a clean,
    human-readable report (Markdown).
    """

    lines = []

    # --------------------------------------------------
    # Executive summary
    # --------------------------------------------------
    lines.append("## 📌 Executive Summary")
    lines.append(
        f"- **Selected model:** {report['model_selected']}"
    )
    lines.append(
        f"- **Cross-validated ROC-AUC:** "
        f"{report['cv_performance']['cv_mean']:.3f} "
        f"(± {report['cv_performance']['cv_std']:.3f})"
    )
    lines.append(
        f"- **Test ROC-AUC:** {report['overall_metrics']['roc_auc']:.3f}"
    )
    lines.append(
        f"- **Test F1-score:** {report['overall_metrics']['f1']:.3f}"
    )

    # --------------------------------------------------
    # Model performance
    # --------------------------------------------------
    lines.append("\n## 📈 Model Performance (Out-of-Sample)")
    lines.append(
        f"- ROC-AUC measures ranking ability (good separation of churn vs non-churn)."
    )
    lines.append(
        f"- F1-score reflects precision–recall tradeoff at the chosen threshold."
    )

    # --------------------------------------------------
    # Risk segmentation
    # --------------------------------------------------
    lines.append("\n## 🎯 Risk Segmentation (Who to Act On)")
    top_segment = report["risk_segmentation_summary"][0]

    lines.append(
        f"- **Highest-risk segment (top decile):**"
    )
    lines.append(
        f"  - Avg predicted churn probability: "
        f"{top_segment['avg_pred_prob']:.2f}"
    )
    lines.append(
        f"  - Observed churn rate: "
        f"{top_segment['positive_rate']:.2%}"
    )
    lines.append(
        f"- Risk is well-concentrated in top segments → suitable for targeted interventions."
    )

    # --------------------------------------------------
    # Explainability
    # --------------------------------------------------
    lines.append("\n## 🧠 Key Drivers (Explainability)")
    if report["top_explanations"]["global_top_feature_indices"]:
        for idx in report["top_explanations"]["global_top_feature_indices"][:5]:
            lines.append(f"- Feature index {idx}")
    else:
        lines.append(
            "- Feature-level drivers available in explainability module "
            "(mapped post-preprocessing)."
        )

    # --------------------------------------------------
    # Error analysis
    # --------------------------------------------------
    lines.append("\n## ⚠️ Error Analysis (Where the Model Struggles)")
    lines.append(
        f"- **Hard examples identified:** "
        f"{report['error_analysis']['hard_example_count']}"
    )
    lines.append(
        f"- Errors cluster into "
        f"{len(report['error_analysis']['clusters'])} distinct patterns."
    )
    lines.append(
        "- These clusters can guide targeted data audits or feature enrichment."
    )

    # --------------------------------------------------
    # Limitations
    # --------------------------------------------------
    lines.append("\n## 🚧 Limitations")
    for lim in report["limitations"]:
        lines.append(f"- {lim}")

    # --------------------------------------------------
    # Recommendations
    # --------------------------------------------------
    lines.append("\n## 🔜 Recommended Next Steps")
    for step in report["recommended_next_steps"]:
        lines.append(f"- {step}")

    return "\n".join(lines)
