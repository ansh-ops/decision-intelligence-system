from core.preprocessing import build_preprocessor
from agents.feature_agent import FeatureAgent
from agents.modeling_agent import ModelingAgent


def _select_best_model(model_results):
    # Pick highest cv_mean
    best = sorted(model_results, key=lambda r: r["cv_mean"], reverse=True)[0]
    return best


def run_modeling(df, target, task_type, stats):
    X = df.drop(columns=[target])
    y = df[target]

    preprocessor, num_feats, cat_feats = build_preprocessor(df, target)

    feature_agent = FeatureAgent()
    feature_info = feature_agent.run(num_feats, cat_feats, stats)

    modeling_agent = ModelingAgent(task_type)
    model_results = modeling_agent.run(X, y, preprocessor)

    best = _select_best_model(model_results)

    # Strip pipelines from the leaderboard copy if you want JSON-safe output
    leaderboard = []
    for r in model_results:
        leaderboard.append({
            "model": r["model"],
            "cv_mean": r["cv_mean"],
            "cv_std": r["cv_std"],
            "fit_metrics": r["fit_metrics"],
            "primary_metric": r["primary_metric"],
        })

    return {
        "features": feature_info,
        "leaderboard": leaderboard,
        "best_model": {
            "model": best["model"],
            "cv_mean": best["cv_mean"],
            "cv_std": best["cv_std"],
            "fit_metrics": best["fit_metrics"],
            "primary_metric": best["primary_metric"],
            "pipeline": best["pipeline"],  
        }
    }
