"use client";

import type { DataProfile } from "../lib/types";

type DataProfileCardProps = {
  profile?: DataProfile;
};

export default function DataProfileCard({ profile }: DataProfileCardProps) {
  if (!profile || Object.keys(profile).length === 0) {
    return (
      <div className="rounded-2xl border bg-white p-6 shadow-sm">
        <h2 className="mb-4 text-xl font-semibold">Data Profile</h2>
        <p className="text-sm text-gray-500">No dataset profile available.</p>
      </div>
    );
  }

  const missingSummary = profile.missing_summary || {};
  const classDistribution = profile.class_distribution || {};

  return (
    <div className="rounded-2xl border bg-white p-6 shadow-sm">
      <h2 className="mb-4 text-xl font-semibold">Data Profile</h2>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-xl bg-slate-50 p-4">
          <p><span className="font-medium">Rows:</span> {profile.rows ?? "N/A"}</p>
          <p><span className="font-medium">Columns:</span> {profile.columns ?? "N/A"}</p>
          <p><span className="font-medium">Numeric Columns:</span> {profile.numeric_columns ?? "N/A"}</p>
          <p><span className="font-medium">Categorical Columns:</span> {profile.categorical_columns ?? "N/A"}</p>
          <p><span className="font-medium">Imbalance Ratio:</span> {profile.imbalance_ratio ?? "N/A"}</p>
        </div>

        <div className="rounded-xl bg-slate-50 p-4">
          <h3 className="mb-2 font-medium">Class Distribution</h3>
          {Object.keys(classDistribution).length === 0 ? (
            <p className="text-sm text-gray-500">Not applicable.</p>
          ) : (
            <div className="space-y-2 text-sm">
              {Object.entries(classDistribution).map(([key, value]) => (
                <div key={key} className="flex items-center justify-between">
                  <span>{key}</span>
                  <span className="font-medium">{String(value)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="mt-4 rounded-xl bg-slate-50 p-4">
        <h3 className="mb-2 font-medium">Top Missing Columns</h3>
        {Object.keys(missingSummary).length === 0 ? (
          <p className="text-sm text-gray-500">No missing values detected in top columns.</p>
        ) : (
          <div className="space-y-2 text-sm">
            {Object.entries(missingSummary).map(([key, value]) => (
              <div key={key} className="flex items-center justify-between">
                <span>{key}</span>
                <span className="font-medium">{String(value)}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
