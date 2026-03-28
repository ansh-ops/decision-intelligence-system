"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

type FeatureRow = {
  feature: string;
  importance: number;
};

type FeatureImportanceChartProps = {
  features: FeatureRow[];
};

export default function FeatureImportanceChart({
  features,
}: FeatureImportanceChartProps) {
  const data = (features || [])
    .slice(0, 10)
    .map((f) => ({
      feature: f.feature,
      importance: Number(f.importance?.toFixed(4)),
    }))
    .reverse();

  if (!data.length) {
    return (
      <div className="rounded-2xl border bg-white p-6 shadow-sm">
        <h2 className="mb-4 text-xl font-semibold">Feature Importance</h2>
        <p className="text-sm text-slate-500">No feature importance available yet.</p>
      </div>
    );
  }

  return (
    <div className="rounded-2xl border bg-white p-6 shadow-sm">
      <div className="mb-4">
        <h2 className="text-xl font-semibold">Top Feature Importance</h2>
        <p className="text-sm text-slate-500">
          Top 10 features ranked by SHAP importance.
        </p>
      </div>

      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ left: 12, right: 12 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis type="category" dataKey="feature" width={130} />
            <Tooltip />
            <Bar dataKey="importance" fill="#0ea5e9" radius={[0, 8, 8, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}