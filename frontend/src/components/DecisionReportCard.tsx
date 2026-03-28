"use client";

import type { DecisionReport } from "../lib/types";

type DecisionReportCardProps = {
  report?: DecisionReport;
};

function toList(value: string[] | string | undefined): string[] {
  if (!value) return [];
  if (Array.isArray(value)) return value.map(String);
  if (typeof value === "string") return [value];
  return [];
}

export default function DecisionReportCard({ report }: DecisionReportCardProps) {
  if (!report || Object.keys(report).length === 0) {
    return (
      <div className="rounded-2xl border bg-white p-6 shadow-sm">
        <h2 className="mb-4 text-xl font-semibold">Decision Report</h2>
        <p className="text-sm text-slate-500">No decision report available yet.</p>
      </div>
    );
  }

  const summary =
    report?.executive_summary ||
    report?.summary ||
    report?.decision_summary ||
    "No summary available.";

  const keyDriversPrimary = toList(report?.key_drivers);
  const keyDriversFallback = toList(report?.drivers);
  const keyDrivers = keyDriversPrimary.length ? keyDriversPrimary : keyDriversFallback;

  const risksPrimary = toList(report?.risks);
  const risksFallback = toList(report?.risk_flags);
  const risks = risksPrimary.length ? risksPrimary : risksFallback;

  const actionsPrimary = toList(report?.recommended_actions);
  const actionsSecondary = toList(report?.actions);
  const actionsFallback = toList(report?.recommendations);
  const actions = actionsPrimary.length
    ? actionsPrimary
    : actionsSecondary.length
      ? actionsSecondary
      : actionsFallback;

  return (
    <div className="rounded-2xl border bg-white p-6 shadow-sm">
      <h2 className="mb-4 text-xl font-semibold">Decision Report</h2>

      <div className="rounded-xl bg-blue-50 p-4">
        <p className="text-sm font-medium text-blue-700">Executive Summary</p>
        <p className="mt-2 text-slate-800">{summary}</p>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <div className="rounded-xl bg-slate-50 p-4">
          <h3 className="font-medium text-slate-900">Key Drivers</h3>
          {keyDrivers.length ? (
            <ul className="mt-2 space-y-2 text-sm text-slate-700">
              {keyDrivers.map((item, idx) => (
                <li key={idx}>• {item}</li>
              ))}
            </ul>
          ) : (
            <p className="mt-2 text-sm text-slate-500">No key drivers available.</p>
          )}
        </div>

        <div className="rounded-xl bg-amber-50 p-4">
          <h3 className="font-medium text-amber-900">Risks / Watchouts</h3>
          {risks.length ? (
            <ul className="mt-2 space-y-2 text-sm text-amber-800">
              {risks.map((item, idx) => (
                <li key={idx}>• {item}</li>
              ))}
            </ul>
          ) : (
            <p className="mt-2 text-sm text-amber-700">No major risks flagged.</p>
          )}
        </div>

        <div className="rounded-xl bg-emerald-50 p-4">
          <h3 className="font-medium text-emerald-900">Recommended Actions</h3>
          {actions.length ? (
            <ul className="mt-2 space-y-2 text-sm text-emerald-800">
              {actions.map((item, idx) => (
                <li key={idx}>• {item}</li>
              ))}
            </ul>
          ) : (
            <p className="mt-2 text-sm text-emerald-700">No actions available.</p>
          )}
        </div>
      </div>
    </div>
  );
}
