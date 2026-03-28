"use client";

type RunStatusCardProps = {
  run: {
    run_id: string;
    status: string;
    mode?: string;
    current_stage?: string | null;
    errors?: string[];
  };
};

function getStatusTone(status: string) {
  const s = status?.toLowerCase();

  if (s === "completed") return "bg-emerald-50 text-emerald-700 border-emerald-200";
  if (s === "failed") return "bg-rose-50 text-rose-700 border-rose-200";
  if (s === "running") return "bg-blue-50 text-blue-700 border-blue-200";
  return "bg-slate-50 text-slate-700 border-slate-200";
}

export default function RunStatusCard({ run }: RunStatusCardProps) {
  return (
    <div className="rounded-2xl border bg-white p-6 shadow-sm">
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h2 className="text-xl font-semibold">Run Status</h2>
          <p className="mt-2 text-sm text-slate-600 break-all">
            Run ID: {run.run_id}
          </p>
        </div>

        <div className={`rounded-full border px-4 py-2 text-sm font-medium ${getStatusTone(run.status)}`}>
          {run.status}
        </div>
      </div>

      <div className="mt-4 grid gap-4 md:grid-cols-2">
        <div className="rounded-xl border bg-slate-50 p-4">
          <p className="text-sm font-medium text-slate-500">Current Stage</p>
          <p className="mt-1 text-base font-semibold text-slate-800">
            {run.current_stage || "N/A"}
          </p>
        </div>

        <div className="rounded-xl border bg-slate-50 p-4">
          <p className="text-sm font-medium text-slate-500">Mode</p>
          <p className="mt-1 text-base font-semibold text-slate-800 capitalize">
            {run.mode || "deterministic"}
          </p>
        </div>
      </div>

      {run.errors && run.errors.length > 0 && (
        <div className="mt-4 rounded-xl border border-rose-200 bg-rose-50 p-4">
          <h3 className="mb-2 font-medium text-rose-700">Errors</h3>
          <ul className="list-disc pl-5 text-sm text-rose-700">
            {run.errors.map((err, idx) => (
              <li key={idx}>{err}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
