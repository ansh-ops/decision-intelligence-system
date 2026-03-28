export type Scalar = string | number | boolean | null;
export type JsonValue = Scalar | JsonValue[] | { [key: string]: JsonValue };

export type MetricMap = Record<string, Scalar | undefined>;

export type RunEvent = {
  stage?: string;
  message?: string;
  details?: Record<string, JsonValue>;
};

export type RecommendedModel = {
  name?: string;
  model?: string;
  model_name?: string;
  best_model?: string;
  reason?: string;
  cv_mean?: number | null;
  cv_std?: number | null;
  primary_score?: number | null;
  score?: number | null;
  policy?: string;
  selection_policy?: string;
  metrics?: MetricMap;
};

export type LeaderboardRow = {
  model?: string;
  name?: string;
  model_name?: string;
  primary_score?: number | null;
  cv_mean?: number | null;
  cv_std?: number | null;
  metrics?: MetricMap;
};

export type ThresholdScoreRow =
  | [number, number]
  | {
      threshold?: number | null;
      score?: number | null;
    };

export type ThresholdInfo = {
  best_threshold?: number | null;
  best_score?: number | null;
  all_scores?: ThresholdScoreRow[];
  [key: string]: JsonValue | undefined;
};

export type DataProfile = {
  rows?: number;
  columns?: number;
  numeric_columns?: number;
  categorical_columns?: number;
  imbalance_ratio?: number | null;
  missing_summary?: Record<string, Scalar>;
  class_distribution?: Record<string, Scalar>;
  [key: string]: JsonValue | undefined;
};

export type FeatureImportanceRow = {
  feature?: string;
  importance?: number | null;
};

export type DecisionReport = {
  executive_summary?: string;
  summary?: string;
  decision_summary?: string;
  key_drivers?: string[] | string;
  drivers?: string[] | string;
  risks?: string[] | string;
  risk_flags?: string[] | string;
  recommended_actions?: string[] | string;
  actions?: string[] | string;
  recommendations?: string[] | string;
};

export type RunResult = {
  target?: string;
  task_type?: string;
  data_profile?: DataProfile;
  splits?: Record<string, Scalar>;
  modeling?: {
    leaderboard?: LeaderboardRow[];
    recommended_model?: RecommendedModel;
  };
  evaluation?: {
    overall_metrics?: MetricMap;
    threshold_info?: ThresholdInfo;
    threshold?: number | null;
  };
  explainability?: {
    feature_importance?: FeatureImportanceRow[];
  };
  week3?: {
    decision_report?: DecisionReport;
  };
  decision_report?: DecisionReport;
};

export type RunData = {
  run_id: string;
  status: string;
  filename?: string | null;
  mode?: string;
  current_stage?: string | null;
  errors?: string[];
  events?: RunEvent[];
  artifacts?: Record<string, JsonValue>;
  result?: RunResult;
};
