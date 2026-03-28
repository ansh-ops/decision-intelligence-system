# Autonomous Decision Intelligence System

An end-to-end decision intelligence platform for tabular datasets. Users can upload a CSV or Excel file, run the analysis in one of three execution modes, and get a dashboard with model comparison, threshold tuning, explainability, error analysis, and a business-facing recommendation.

## Screenshots

### Upload / Mode Selection
![Upload and mode selection](pictures/front%20page.png)

### Recommendation and Results
![Recommended model and results dashboard](pictures/recommended%20models.png)

## What The System Does

- Accepts CSV and Excel uploads
- Detects likely target columns and task type
- Profiles the dataset and computes summary statistics
- Trains baseline ML pipelines with preprocessing
- Evaluates models on holdout data
- Optimizes decision thresholds for classification
- Generates feature importance and explainability outputs
- Performs segmentation and error analysis
- Produces a recommendation and decision-ready summary
- Tracks each run live in the frontend

## Execution Modes

The system supports three run modes:

### 1. Deterministic

This is the fixed pipeline mode. It follows the same execution flow every time and is the most reliable for reproducibility.

### 2. Agentic AI

This mode adds planner-driven orchestration. An LLM-backed planner selects backend tools such as schema inspection, dataset profiling, and baseline training. The selected actions are logged into the run timeline before the system hands off to the full analysis pipeline.

### 3. Adaptive Policy

This mode builds a dataset-specific modeling policy before the final run. The policy can change:

- model candidates
- model ranking metric
- threshold optimization objective

This makes the pipeline more data-aware than the fixed deterministic flow.

## Architecture

### Backend

- Python
- FastAPI
- Pandas
- NumPy
- scikit-learn
- SHAP
- MLflow
- FAISS
- sentence-transformers
- google-genai

### Frontend

- Next.js 16
- React 19
- TypeScript
- Tailwind CSS
- Recharts

### Key Backend Modules

- [api/main.py](api/main.py): FastAPI routes for upload, run start, and run status
- [services/run_service.py](services/run_service.py): async run execution for deterministic, agentic, and adaptive modes
- [pipelines/run_analysis.py](pipelines/run_analysis.py): main decision intelligence pipeline
- [pipelines/run_modeling.py](pipelines/run_modeling.py): preprocessing + model benchmarking
- [agents/modeling_agent.py](agents/modeling_agent.py): baseline model registry and training loop
- [services/ai_enrichment.py](services/ai_enrichment.py): Gemini-powered or deterministic AI summary
- [services/adaptive_policy.py](services/adaptive_policy.py): adaptive policy selection logic
- [rag/memory_store.py](rag/memory_store.py): FAISS-backed memory store
- [services/artifact_store.py](services/artifact_store.py): persistent run-state storage

## AI Integration

The project uses AI in three places:

### AI Enrichment

After a run completes, the backend adds an AI summary block to the result. If `GEMINI_API_KEY` is available, Gemini generates:

- executive summary
- recommended next step
- watchouts

If the key is not available, the system falls back to a deterministic summary.

### Agentic Planning

In `agentic` mode, the planner selects backend tools and logs the chosen actions into the run event stream. This gives the product an agentic execution path while preserving the deterministic analysis engine as a safe fallback.

### Adaptive Policy Engine

In `adaptive` mode, the system builds a run policy before the final analysis. That policy changes how the system ranks models and tunes thresholds, which can lead to different outputs than the deterministic flow.

## RAG / Memory

The project also includes a simple retrieval layer using FAISS and sentence-transformers.

What it stores:

- compact summaries of completed analyses
- metadata such as target, task type, metrics, and best model

How it helps:

- lets the system retrieve similar past analyses
- provides context for future agent planning
- creates the foundation for retrieval-augmented workflows

Current implementation:

- embeddings are generated with `all-MiniLM-L6-v2`
- vectors are stored in a FAISS index in `memory_index/`
- metadata is stored alongside the vector index

## Run Tracking

Each run is persisted in `artifacts/<run_id>/state.json` and includes:

- run id
- mode
- status
- current stage
- final result
- tool artifacts
- event timeline
- error logs

This enables live frontend polling and makes the system easier to debug.

## Comparing The Three Modes

One of the next goals for this project is to benchmark the three run modes systematically in terms of:

- speed
- accuracy
- threshold quality
- recommendation stability
- interpretability tradeoffs

### Expected Comparison Framing

- `deterministic` should be the fastest and most reproducible baseline
- `agentic` should provide richer orchestration visibility and AI-driven tool selection, but may add some planning overhead
- `adaptive` should be the most flexible in terms of policy selection and may improve model-choice quality on datasets where the default metric or threshold strategy is not ideal

### Recommended Evaluation Plan

To compare the modes fairly, run the same dataset through all three modes and log:

- total runtime
- best model selected
- primary validation metric
- holdout test metrics
- threshold metric and chosen threshold
- explainability stability
- final recommendation differences

## Local Setup

### Backend

```bash
git clone <your-repo-url>
cd decision-intelligence-system

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
uvicorn api.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on `http://localhost:3000` and the backend runs on `http://127.0.0.1:8000`.

## Environment Variables

To enable Gemini-powered AI enrichment and policy generation:

```bash
export GEMINI_API_KEY="your-api-key"
```

Restart the FastAPI server after setting the key.

## Publishing Notes

This repository intentionally excludes generated interview/export artifacts and runtime folders such as local run state and model artifacts.

## Future Improvements

- benchmark reporting across deterministic, agentic, and adaptive modes
- richer adaptive model search
- retrieval-augmented planner context from previous runs
- better side-by-side comparison UI across run modes
- stronger agentic retries and validation loops
