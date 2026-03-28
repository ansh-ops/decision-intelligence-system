"use client";

import type { MetricMap, Scalar, ThresholdInfo } from "../lib/types";

type MetricsPanelProps = {
  metrics?: MetricMap | null;
  threshold?: number | null;
  thresholdInfo?: ThresholdInfo | null;
  targetDistribution?: Record<string, number> | null;
};

function getTone(key: string) {
  const normalized = key.toLowerCase();

  if (normalized.includes("auc") || normalized.includes("f1") || normalized.includes("r2")) {
    return "bg-emerald-50 text-emerald-700 border-emerald-200";
  }

  if (normalized.includes("rmse") || normalized.includes("error")) {
    return "bg-amber-50 text-amber-700 border-amber-200";
  }

  return "bg-slate-50 text-slate-700 border-slate-200";
}

export default function MetricsPanel({
  metrics,
  threshold,
  thresholdInfo,
  targetDistribution,
}: MetricsPanelProps) {
  if (!metrics || Object.keys(metrics).length === 0) {
    return (
      <div className="rounded-2xl border bg-white p-6 shadow-sm">
        <h2 className="mb-4 text-xl font-semibold">Metrics</h2>
        <p className="text-sm text-gray-500">No metrics available yet.</p>
      </div>
    );
  }

  return (
    <div className="rounded-2xl border bg-white p-6 shadow-sm">
      <h2 className="mb-4 text-xl font-semibold">Evaluation Metrics</h2>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {Object.entries(metrics).map(([key, value]) => (
          <div
            key={key}
            className={`rounded-2xl border p-4 ${getTone(key)}`}
          >
            <p className="text-sm font-medium uppercase tracking-wide">{key}</p>
            <p className="mt-2 text-2xl font-semibold">
              {typeof value === "number" ? value.toFixed(3) : String(value as Scalar)}
            </p>
          </div>
        ))}
      </div>

      {(threshold || thresholdInfo || targetDistribution) && (
        <div className="mt-6 grid gap-4 lg:grid-cols-3">
          <div className="rounded-xl border bg-slate-50 p-4">
            <h3 className="mb-2 font-semibold">Threshold</h3>
            <p className="text-sm text-slate-600">
              {typeof threshold === "number" ? threshold.toFixed(3) : "N/A"}
            </p>
          </div>

          <div className="rounded-xl border bg-slate-50 p-4">
            <h3 className="mb-2 font-semibold">Threshold Details</h3>
            <pre className="overflow-auto text-xs text-slate-600">
              {JSON.stringify(thresholdInfo || {}, null, 2)}
            </pre>
          </div>

          <div className="rounded-xl border bg-slate-50 p-4">
            <h3 className="mb-2 font-semibold">Target Distribution</h3>
            <pre className="overflow-auto text-xs text-slate-600">
              {JSON.stringify(targetDistribution || {}, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}
