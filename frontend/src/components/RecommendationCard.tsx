"use client";

import type { RecommendedModel } from "../lib/types";

type RecommendationCardProps = {
  recommendedModel?: RecommendedModel;
  taskType?: string;
};

function getModelName(model?: RecommendedModel) {
  return (
    model?.name ??
    model?.model ??
    model?.model_name ??
    model?.best_model ??
    "N/A"
  );
}

function formatValue(value: unknown) {
  if (typeof value === "number") return value.toFixed(4);
  if (value === null || value === undefined || value === "") return "N/A";
  return String(value);
}

export default function RecommendationCard({
  recommendedModel,
  taskType,
}: RecommendationCardProps) {
  if (!recommendedModel || Object.keys(recommendedModel).length === 0) {
    return (
      <div className="rounded-2xl border bg-white p-6 shadow-sm">
        <h2 className="mb-4 text-xl font-semibold">Recommended Model</h2>
        <p className="text-sm text-slate-500">No recommendation available yet.</p>
      </div>
    );
  }

  const modelName = getModelName(recommendedModel);
  const reason =
    recommendedModel?.reason || "No explanation available.";
  const cvMean =
    recommendedModel?.cv_mean ??
    recommendedModel?.primary_score ??
    recommendedModel?.score;
  const cvStd = recommendedModel?.cv_std;
  const policy =
    recommendedModel?.policy ??
    recommendedModel?.selection_policy ??
    "Performance + interpretability";

  return (
    <div className="rounded-2xl border border-violet-200 bg-violet-50 p-6 shadow-sm">
      <h2 className="mb-4 text-xl font-semibold text-violet-900">
        Recommended Model
      </h2>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-xl bg-white p-4">
          <p className="text-sm text-violet-700">Model</p>
          <p className="mt-1 text-2xl font-bold text-slate-900">{modelName}</p>
        </div>

        <div className="rounded-xl bg-white p-4">
          <p className="text-sm text-violet-700">Task</p>
          <p className="mt-1 text-lg font-semibold text-slate-900">
            {taskType || "N/A"}
          </p>
        </div>

        <div className="rounded-xl bg-white p-4">
          <p className="text-sm text-violet-700">CV Mean / Primary Score</p>
          <p className="mt-1 text-lg font-semibold text-slate-900">
            {formatValue(cvMean)}
          </p>
        </div>

        <div className="rounded-xl bg-white p-4">
          <p className="text-sm text-violet-700">CV Std</p>
          <p className="mt-1 text-lg font-semibold text-slate-900">
            {formatValue(cvStd)}
          </p>
        </div>
      </div>

      <div className="mt-4 rounded-xl bg-white p-4">
        <p className="text-sm font-medium text-violet-700">Selection Policy</p>
        <p className="mt-1 text-slate-800">{policy}</p>
      </div>

      <div className="mt-4 rounded-xl bg-white p-4">
        <p className="text-sm font-medium text-violet-700">Why this model was chosen</p>
        <p className="mt-1 text-slate-800">{reason}</p>
      </div>
    </div>
  );
}
