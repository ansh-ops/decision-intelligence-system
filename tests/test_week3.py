from pipelines.run_analysis import run_pipeline

def test_week3_outputs_exist():
    res = run_pipeline("datasets/telco_churn.csv", target_override=None)

    assert "week3" in res
    assert "explainability" in res["week3"]
    assert "error_analysis" in res["week3"]
    assert "decision_report" in res["week3"]

def test_segmentation_for_classification():
    res = run_pipeline("datasets/telco_churn.csv")
    seg = res["week3"]["segmentation"]
    assert seg is not None
    assert len(seg) >= 5  # bins may reduce if duplicates
