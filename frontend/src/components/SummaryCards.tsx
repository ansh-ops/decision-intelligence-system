"use client";

import type { RunData } from "../lib/types";

type SummaryCardsProps = {
  run: RunData;
};

function prettyMetricKey(key: string) {
  return key.replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatMetricValue(value: unknown) {
  if (typeof value === "number") {
    return value.toFixed(4);
  }
  return String(value ?? "N/A");
}

export default function SummaryCards({ run }: SummaryCardsProps) {
  const result = run?.result || {};
  const profile = result?.data_profile || {};
  const recommendedModel = result?.modeling?.recommended_model || {};
  const metrics = result?.evaluation?.overall_metrics || {};
  const splits = result?.splits || {};

  const primaryMetricEntry =
    Object.entries(metrics).find(([k]) =>
      ["f1", "roc_auc", "accuracy", "r2", "rmse", "mae"].includes(k)
    ) || [];

  const [metricName, metricValue] = primaryMetricEntry as [string, unknown];

  const cards = [
    {
      label: "Recommended Model",
      value: recommendedModel?.name || "N/A",
      tone: "bg-violet-50 text-violet-700 border-violet-200",
    },
    {
      label: metricName ? prettyMetricKey(metricName) : "Primary Metric",
      value: metricValue !== undefined ? formatMetricValue(metricValue) : "N/A",
      tone: "bg-emerald-50 text-emerald-700 border-emerald-200",
    },
    {
      label: "Task Type",
      value: result?.task_type || "N/A",
      tone: "bg-blue-50 text-blue-700 border-blue-200",
    },
    {
      label: "Target Column",
      value: result?.target || "N/A",
      tone: "bg-amber-50 text-amber-700 border-amber-200",
    },
    {
      label: "Train Rows",
      value: splits?.train_rows ?? profile?.rows ?? "N/A",
      tone: "bg-slate-50 text-slate-700 border-slate-200",
    },
    {
      label: "Test Rows",
      value: splits?.test_rows ?? "N/A",
      tone: "bg-slate-50 text-slate-700 border-slate-200",
    },
  ];

  return (
    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
      {cards.map((card) => (
        <div key={card.label} className={`rounded-2xl border p-5 shadow-sm ${card.tone}`}>
          <p className="text-sm font-medium opacity-80">{card.label}</p>
          <p className="mt-2 text-2xl font-bold break-words">{card.value}</p>
        </div>
      ))}
    </div>
  );
}
