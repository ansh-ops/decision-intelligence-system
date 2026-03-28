"use client";

import type { RunData } from "../lib/types";

type AnalysisProgressScreenProps = {
  run: RunData;
};

function prettyStage(stage?: string | null) {
  if (!stage) return "Waiting";
  return stage.replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export default function AnalysisProgressScreen({ run }: AnalysisProgressScreenProps) {
  const events = Array.isArray(run.events) ? run.events : [];
  const modelEvents = events.filter((event) => event?.stage === "model_evaluated");

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top,_#e0f2fe,_#f8fafc_45%,_#ffffff_100%)] p-8">
      <div className="mx-auto max-w-6xl space-y-6">
        <section className="rounded-[2rem] border border-sky-100 bg-white/90 p-8 shadow-xl shadow-sky-100/60 backdrop-blur">
          <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
            <div className="max-w-2xl">
              <p className="text-sm font-semibold uppercase tracking-[0.3em] text-sky-600">
                Live Analysis
              </p>
              <h1 className="mt-3 text-4xl font-semibold tracking-tight text-slate-900">
                Building the decision pipeline
              </h1>
              <p className="mt-3 text-base text-slate-600">
                {run.filename || "Your dataset"} is being profiled, modeled, and translated
                into decision-ready outputs. This screen updates automatically as each step finishes.
              </p>
              <p className="mt-2 text-sm font-medium uppercase tracking-[0.2em] text-slate-500">
                Mode: {run.mode || "deterministic"}
              </p>
            </div>

            <div className="rounded-3xl border border-sky-200 bg-sky-50 px-6 py-5 text-sky-800">
              <p className="text-sm font-medium text-sky-600">Current Stage</p>
              <p className="mt-2 text-2xl font-semibold">{prettyStage(run.current_stage)}</p>
              <div className="mt-4 flex items-center gap-3">
                <span className="h-3 w-3 animate-pulse rounded-full bg-sky-500" />
                <span className="text-sm font-medium capitalize">{run.status}</span>
              </div>
            </div>
          </div>
        </section>

        <section className="grid gap-6 xl:grid-cols-[1.3fr_0.9fr]">
          <div className="rounded-3xl border bg-white p-6 shadow-sm">
            <h2 className="text-xl font-semibold text-slate-900">Actions Performed</h2>
            <p className="mt-2 text-sm text-slate-500">
              Every pipeline action is recorded here as the run advances.
            </p>

            <div className="mt-6 space-y-4">
              {events.length === 0 ? (
                <div className="rounded-2xl border border-dashed border-slate-200 p-5 text-sm text-slate-500">
                  Waiting for the first analysis event...
                </div>
              ) : (
                events.map((event, index) => (
                  <div key={`${event.stage}-${index}`} className="rounded-2xl border bg-slate-50 p-5">
                    <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
                      <div>
                        <p className="text-sm font-semibold uppercase tracking-wide text-sky-700">
                          {prettyStage(event.stage)}
                        </p>
                        <p className="mt-1 text-base font-medium text-slate-900">
                          {event.message || "Working..."}
                        </p>
                      </div>
                      <p className="text-xs text-slate-500">Step {index + 1}</p>
                    </div>

                    {event.details && Object.keys(event.details).length > 0 && (
                      <pre className="mt-3 overflow-auto rounded-xl bg-white p-3 text-xs text-slate-600">
                        {JSON.stringify(event.details, null, 2)}
                      </pre>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>

          <div className="space-y-6">
            <div className="rounded-3xl border bg-white p-6 shadow-sm">
              <h2 className="text-xl font-semibold text-slate-900">ML Pipelines</h2>
              <p className="mt-2 text-sm text-slate-500">
                Models appear here as soon as training and evaluation finish.
              </p>

              <div className="mt-5 space-y-3">
                {modelEvents.length === 0 ? (
                  <div className="rounded-2xl border border-dashed border-slate-200 p-5 text-sm text-slate-500">
                    Model training has not emitted pipeline results yet.
                  </div>
                ) : (
                  modelEvents.map((event, index) => (
                    <div key={`${event.details?.model || "model"}-${index}`} className="rounded-2xl bg-emerald-50 p-4">
                      <p className="font-semibold text-emerald-900">
                        {event.details?.model || "Model pipeline"}
                      </p>
                      <p className="mt-2 text-sm text-emerald-800">
                        CV Mean: {event.details?.cv_mean ?? "N/A"} | CV Std: {event.details?.cv_std ?? "N/A"}
                      </p>
                    </div>
                  ))
                )}
              </div>
            </div>

            {run.errors && run.errors.length > 0 && (
              <div className="rounded-3xl border border-rose-200 bg-rose-50 p-6 shadow-sm">
                <h2 className="text-xl font-semibold text-rose-900">Run Errors</h2>
                <div className="mt-4 space-y-2 text-sm text-rose-800">
                  {run.errors.map((error, index) => (
                    <p key={index}>{error}</p>
                  ))}
                </div>
              </div>
            )}
          </div>
        </section>
      </div>
    </main>
  );
}
