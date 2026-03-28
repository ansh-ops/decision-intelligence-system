"use client";

import {
  ResponsiveContainer,
  BarChart,
  Bar,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
} from "recharts";
import type { LeaderboardRow } from "../lib/types";

type LeaderboardChartProps = {
  leaderboard?: LeaderboardRow[];
};

function num(value: unknown) {
  return typeof value === "number" ? value : null;
}

function fmt(value: unknown, digits = 3) {
  return typeof value === "number" ? value.toFixed(digits) : "N/A";
}

function modelLabel(row: LeaderboardRow) {
  return row?.model ?? row?.name ?? row?.model_name ?? "Unknown";
}

function scoreValue(row: LeaderboardRow) {
  return (
    num(row?.primary_score) ??
    num(row?.cv_mean) ??
    num(row?.metrics?.roc_auc) ??
    num(row?.metrics?.f1) ??
    num(row?.metrics?.accuracy) ??
    num(row?.metrics?.r2) ??
    0
  );
}

export default function LeaderboardChart({ leaderboard = [] }: LeaderboardChartProps) {
  if (!leaderboard.length) {
    return (
      <div className="rounded-2xl border bg-white p-6 shadow-sm">
        <h2 className="mb-4 text-xl font-semibold">Model Leaderboard</h2>
        <p className="text-sm text-slate-500">No leaderboard available yet.</p>
      </div>
    );
  }

  const data = leaderboard.map((row) => ({
    model: modelLabel(row),
    score: scoreValue(row),
  }));

  return (
    <div className="rounded-2xl border bg-white p-6 shadow-sm">
      <h2 className="mb-4 text-xl font-semibold">Model Leaderboard</h2>

      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 8, right: 24, left: 24, bottom: 8 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis type="category" dataKey="model" width={120} />
            <Tooltip />
            <Bar dataKey="score" radius={[0, 8, 8, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

        <div className="mt-5 space-y-3">
          {leaderboard.map((row) => (
            <div
              key={modelLabel(row)}
              className="rounded-xl border bg-slate-50 p-4"
            >
            <div className="flex items-center justify-between">
              <span className="font-semibold text-slate-900">{modelLabel(row)}</span>
              <span className="text-sm text-slate-600">
                Score: {fmt(scoreValue(row))}
              </span>
            </div>

            <div className="mt-2 grid gap-2 text-sm text-slate-600 md:grid-cols-3">
              <div>CV: {fmt(row?.cv_mean)} ± {fmt(row?.cv_std)}</div>
              <div>ROC-AUC: {fmt(row?.metrics?.roc_auc)}</div>
              <div>F1: {fmt(row?.metrics?.f1)}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
