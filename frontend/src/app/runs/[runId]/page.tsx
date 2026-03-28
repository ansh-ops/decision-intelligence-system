"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { getRun } from "../../../lib/api";
import MetricsPanel from "../../../components/MetricsPanel";
import RunStatusCard from "../../../components/RunStatusCard";
import TimelinePanel from "../../../components/TimelinePanel";
import SummaryCards from "../../../components/SummaryCards";
import LeaderboardChart from "../../../components/LeaderboardChart";
import FeatureImportanceChart from "../../../components/FeatureImportanceChart";
import RecommendationCard from "../../../components/RecommendationCard";
import DataProfileCard from "../../../components/DataProfileCard";
import DecisionReportCard from "../../../components/DecisionReportCard";
import ThresholdDetailsCard from "../../../components/ThresholdDetailsCard";
import AnalysisProgressScreen from "../../../components/AnalysisProgressScreen";
import type { RunData } from "../../../lib/types";

export default function RunPage() {
  const params = useParams<{ runId: string }>();
  const runId = params?.runId;

  const [run, setRun] = useState<RunData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!runId) return;

    let cancelled = false;

    const fetchRun = async () => {
      try {
        const data = await getRun(runId);
        if (!cancelled) {
          setRun(data);
        }
      } catch (error) {
        console.error(error);
        if (!cancelled) {
          setRun(null);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    fetchRun();
    const interval = setInterval(fetchRun, 3000);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [runId]);

  if (loading) {
    return <div className="p-8">Loading...</div>;
  }

  if (!run) {
    return <div className="p-8">Run not found.</div>;
  }

  if (run.status === "created" || run.status === "running") {
    return <AnalysisProgressScreen run={run} />;
  }

  const result = run.result || {};
  const metrics = result?.evaluation?.overall_metrics || {};
  const leaderboard = result?.modeling?.leaderboard || [];
  const recommendedModel = result?.modeling?.recommended_model || {};
  const featureImportance = result?.explainability?.feature_importance || [];
  const dataProfile = result?.data_profile || {};
  const thresholdInfo = result?.evaluation?.threshold_info || {};
  const threshold = result?.evaluation?.threshold ?? null;
  const decisionReport =
    result?.week3?.decision_report ??
    result?.decision_report;

  return (
    <main className="min-h-screen bg-slate-50 p-8">
      <div className="mx-auto max-w-7xl space-y-6">
        <RunStatusCard run={run} />
        <SummaryCards run={run} />

        <RecommendationCard
          recommendedModel={recommendedModel}
          taskType={result?.task_type}
        />

        <div className="grid gap-6 xl:grid-cols-2">
          <LeaderboardChart leaderboard={leaderboard} />
          <FeatureImportanceChart features={featureImportance} />
        </div>

        <MetricsPanel metrics={metrics} />
        <ThresholdDetailsCard
          thresholdInfo={thresholdInfo}
          threshold={threshold}
        />
        <DataProfileCard profile={dataProfile} />
        <DecisionReportCard report={decisionReport} />
        <TimelinePanel
          events={run.events || []}
          artifacts={run.artifacts || {}}
        />

        <div className="rounded-2xl border bg-white p-6 shadow-sm">
          <h2 className="mb-4 text-xl font-semibold">Raw Run Result</h2>
          <pre className="max-h-96 overflow-auto text-sm">
            {JSON.stringify(result || {}, null, 2)}
          </pre>
        </div>
      </div>
    </main>
  );
}
