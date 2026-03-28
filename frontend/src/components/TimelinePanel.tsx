"use client";

import type { JsonValue, RunEvent } from "../lib/types";

type TimelinePanelProps = {
  events?: RunEvent[];
  artifacts?: Record<string, JsonValue>;
};

function prettyStage(stage: string) {
  return stage.replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export default function TimelinePanel({ events, artifacts }: TimelinePanelProps) {
  const eventSteps = Array.isArray(events) ? events : [];
  const artifactSteps = artifacts ? Object.keys(artifacts) : [];

  return (
    <div className="rounded-2xl border bg-white p-6 shadow-sm">
      <h2 className="text-xl font-semibold mb-4">Execution Timeline</h2>
      {eventSteps.length === 0 && artifactSteps.length === 0 ? (
        <p className="text-sm text-gray-500">No completed tool steps yet.</p>
      ) : (
        <div className="space-y-3">
          {eventSteps.map((event, index) => (
            <div key={`${event.stage}-${index}`} className="rounded-xl border p-4">
              <p className="font-medium">
                Step {index + 1}: {prettyStage(event.stage || "unknown")}
              </p>
              <p className="mt-1 text-sm text-slate-600">{event.message || "Completed."}</p>
            </div>
          ))}
          {eventSteps.length === 0 &&
            artifactSteps.map((step, index) => (
              <div key={step} className="rounded-xl border p-4">
                <p className="font-medium">
                  Step {index + 1}: {step}
                </p>
              </div>
            ))}
        </div>
      )}
    </div>
  );
}
