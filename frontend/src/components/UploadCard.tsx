"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { startRun, uploadFile } from "../lib/api";

export default function UploadCard() {
  const [file, setFile] = useState<File | null>(null);
  const [targetOverride, setTargetOverride] = useState("");
  const [mode, setMode] = useState<"deterministic" | "agentic" | "adaptive">("adaptive");
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  const handleAnalyze = async () => {
    if (!file) {
      alert("Please select a dataset first.");
      return;
    }

    try {
      setLoading(true);

      const uploadRes = await uploadFile(file);
      const runId = uploadRes.run_id;
      const filePath = `uploads/${uploadRes.filename}`;

      await startRun(
        runId,
        filePath,
        targetOverride.trim() ? targetOverride.trim() : undefined,
        mode,
      );

      router.push(`/runs/${runId}`);
    } catch (error) {
      console.error(error);
      alert(error instanceof Error ? error.message : "Failed to start analysis");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="rounded-3xl border bg-white p-6 shadow-sm">
      <h2 className="mb-2 text-xl font-semibold">Upload Dataset</h2>
      <p className="mb-6 text-sm text-gray-600">
        Upload a CSV or Excel file and optionally specify the target column.
      </p>

      <div className="space-y-5">
        <div>
          <label className="mb-2 block text-sm font-medium text-slate-700">Dataset file</label>
          <input
            type="file"
            accept=".csv,.xlsx,.xls"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
            className="block w-full rounded-xl border p-3 text-sm"
          />
        </div>

        <div>
          <label className="mb-2 block text-sm font-medium text-slate-700">
            Analysis mode
          </label>
          <div className="grid gap-3 md:grid-cols-3">
            <button
              type="button"
              onClick={() => setMode("agentic")}
              className={`rounded-2xl border p-4 text-left ${
                mode === "agentic"
                  ? "border-sky-400 bg-sky-50"
                  : "border-slate-200 bg-white"
              }`}
            >
              <p className="font-semibold text-slate-900">Agentic AI</p>
              <p className="mt-1 text-sm text-slate-600">
                Planner chooses tool steps and logs the reasoning trail.
              </p>
            </button>

            <button
              type="button"
              onClick={() => setMode("adaptive")}
              className={`rounded-2xl border p-4 text-left ${
                mode === "adaptive"
                  ? "border-emerald-400 bg-emerald-50"
                  : "border-slate-200 bg-white"
              }`}
            >
              <p className="font-semibold text-slate-900">Adaptive Policy</p>
              <p className="mt-1 text-sm text-slate-600">
                Builds a dataset-specific policy for model ranking and threshold strategy.
              </p>
            </button>

            <button
              type="button"
              onClick={() => setMode("deterministic")}
              className={`rounded-2xl border p-4 text-left ${
                mode === "deterministic"
                  ? "border-slate-400 bg-slate-50"
                  : "border-slate-200 bg-white"
              }`}
            >
              <p className="font-semibold text-slate-900">Deterministic</p>
              <p className="mt-1 text-sm text-slate-600">
                Fixed pipeline execution with the same final analytics outputs.
              </p>
            </button>
          </div>
        </div>

        <div>
          <label className="mb-2 block text-sm font-medium text-slate-700">
            Target column override
          </label>
          <input
            type="text"
            value={targetOverride}
            onChange={(e) => setTargetOverride(e.target.value)}
            placeholder="e.g. Churn"
            className="block w-full rounded-xl border p-3 text-sm"
          />
        </div>

        <button
          onClick={handleAnalyze}
          disabled={loading || !file}
          className="rounded-2xl bg-slate-900 px-5 py-3 text-sm font-medium text-white disabled:opacity-50"
        >
          {loading
            ? "Starting analysis..."
            : `Analyze Dataset (${
                mode === "agentic"
                  ? "Agentic AI"
                  : mode === "adaptive"
                    ? "Adaptive Policy"
                    : "Deterministic"
              })`}
        </button>
      </div>
    </div>
  );
}
