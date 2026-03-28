"use client";

import type { ThresholdInfo, ThresholdScoreRow } from "../lib/types";

type ThresholdDetailsCardProps = {
  thresholdInfo?: ThresholdInfo;
  threshold?: number | null;
};

function fmt(value: unknown, digits = 3) {
  return typeof value === "number" ? value.toFixed(digits) : "N/A";
}

export default function ThresholdDetailsCard({
  thresholdInfo,
  threshold,
}: ThresholdDetailsCardProps) {
  if (!thresholdInfo || Object.keys(thresholdInfo).length === 0) {
    return (
      <div className="rounded-2xl border bg-white p-6 shadow-sm">
        <h2 className="mb-4 text-xl font-semibold">Threshold Details</h2>
        <p className="text-sm text-slate-500">No threshold optimization details available.</p>
      </div>
    );
  }

  const bestThreshold = thresholdInfo?.best_threshold ?? threshold;
  const bestScore = thresholdInfo?.best_score;
  const allScores = Array.isArray(thresholdInfo?.all_scores)
    ? thresholdInfo.all_scores
    : [];

  return (
    <div className="rounded-2xl border bg-white p-6 shadow-sm">
      <h2 className="mb-4 text-xl font-semibold">Threshold Details</h2>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-xl bg-slate-50 p-4">
          <p className="text-sm text-slate-600">Best Threshold</p>
          <p className="mt-1 text-2xl font-bold text-slate-900">
            {fmt(bestThreshold)}
          </p>
        </div>

        <div className="rounded-xl bg-slate-50 p-4">
          <p className="text-sm text-slate-600">Best Score</p>
          <p className="mt-1 text-2xl font-bold text-slate-900">
            {fmt(bestScore, 4)}
          </p>
        </div>
      </div>

      <div className="mt-4 rounded-xl border bg-slate-50 p-4">
        <p className="mb-3 text-sm font-medium text-slate-700">
          Threshold Search Table
        </p>

        <div className="max-h-72 overflow-y-auto rounded-lg border bg-white">
          <table className="min-w-full text-sm">
            <thead className="sticky top-0 bg-slate-100">
              <tr>
                <th className="px-4 py-2 text-left font-medium text-slate-700">Threshold</th>
                <th className="px-4 py-2 text-left font-medium text-slate-700">Score</th>
              </tr>
            </thead>
            <tbody>
              {allScores.map((row: ThresholdScoreRow, idx: number) => {
                const t = Array.isArray(row) ? row[0] : row?.threshold;
                const s = Array.isArray(row) ? row[1] : row?.score;
                const isBest =
                  typeof bestThreshold === "number" &&
                  typeof t === "number" &&
                  Math.abs(t - bestThreshold) < 1e-9;

                return (
                  <tr
                    key={idx}
                    className={isBest ? "bg-emerald-50" : "border-t"}
                  >
                    <td className="px-4 py-2 text-slate-800">{fmt(t)}</td>
                    <td className="px-4 py-2 text-slate-800">{fmt(s, 4)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
