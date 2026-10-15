# Hotel Price Intelligence Phase 2 Implementation Plan

> For agentic workers: use superpowers:executing-plans when implementation is authorized. Implement and review one stage at a time.

**Goal:** Extend the existing notebook project into a local Streamlit dashboard, then add peer analytics and an evaluated hotel-group price prediction workflow.

**Architecture:** Keep the current generator and cleaner as the source of truth. Add reusable data-loading and analytics modules under src, with Streamlit rendering kept in a small app layer. Deliver dashboard, advanced analytics, and prediction as separate acceptance stages so modeling cannot delay the usable dashboard.

**Tech Stack:** Python 3.12 target environment, Streamlit, Plotly, Pandas, NumPy, and scikit-learn. Pin versions only after verifying them together in the supported environment.

**Spec:** HOTEL_PRICE_INTELLIGENCE_PHASE_2_CODEX.md

## Repository findings

- The project contains src/generate_data.py and src/clean_data.py. The cleaner exposes clean_dataset(raw_data), returning cleaned rows, separated invalid-price rows, and an audit.
- The raw CSV currently has 1,020 rows and 10 required fields, with 120 distinct hotel IDs, 20 exact duplicate copies, 53 missing guest ratings, 31 missing room sizes, and one blank price.
- The cleaned CSV currently has 999 rows and an additional price_outlier field, with 120 distinct hotel IDs, 50 missing guest ratings, 29 missing room sizes, and 16 flagged prices.
- The 35-cell project analysis notebook has 15 code cells and no saved outputs. The supplementary 82-cell learning notebook has saved outputs. Preserve both notebooks.
- Existing tests cover generation and cleaning behavior, but their passing status has not been established in this inspection.
- requirements.txt currently lists pandas, numpy, matplotlib, seaborn, ipykernel, and pytest without pins. The inspected machine has Python 3.14, not the target Python 3.12; Streamlit, Plotly, scikit-learn, ipykernel, pytest, Matplotlib, and Seaborn were not available in the checked environment.
- There is no project-local AGENTS.md or TODOS.md, no Git repository metadata, and no existing Streamlit app or dependency manifest beyond requirements.txt. Git status and uncommitted-file ownership therefore cannot be determined.

## Global constraints

- Start from the existing dataset; do not regenerate it as an app setup step.
- Preserve the raw CSV and both existing notebooks.
- Keep uploaded CSVs in session state; never write uploads over project data.
- Keep filtering, chart populations, summaries, tables, and exports consistent.
- Do not recalculate outlier flags when a view filter changes.
- Do not invent historical cleaning or generator audit values when source evidence is unavailable.
- Exclude identifiers, names, price-derived features, peer statistics, and outlier flags from predictive inputs.
- Split model evaluation by hotel ID; preprocessing is fitted on training rows only.
- Keep local development scope; do not add a backend, database, authentication, scraping, paid API, or deployment system.

## Review focus

- A valid upload must not replace the active dataset until schema and domain validation pass; test both valid and invalid uploads.
- An active room-size range must exclude unknown sizes unless the visible include-unknown option is selected; test both settings.
- Offer selection must still identify the same record after filtering and sorting; test stable IDs against repeated hotel IDs.
- Data-quality history must be unavailable when only a cleaned file exists; test that the UI does not fabricate prior duplicate or rejection counts.
- Model splits must have disjoint hotel IDs and identical group assignments for outlier sensitivity comparisons; test leakage and row-count reporting.

## Stage boundaries

Stage A is the first deliverable and must be usable before Stage B begins. Stage B depends on Stage A's shared filters and stable offering IDs. Stage C depends on the validated active-dataset identity and analytics layer. Do not start Stage C until the dashboard and Stage B behavior are accepted.

## Stage A — Interactive dashboard

### Task 1: Establish the supported environment

**Files:** Modify requirements.txt and README.md.

- [ ] Confirm or install the supported Python 3.12 runtime before selecting package versions.
- [ ] Add Streamlit and Plotly alongside the existing Pandas/NumPy requirements; retain notebook and test dependencies.
- [ ] Verify package versions together in the supported environment, then pin them in requirements.txt.
- [ ] Record the verified environment and exact setup commands in README.md.
- [ ] Run the existing tests and execute the existing analysis notebook from a fresh kernel before dashboard changes; record failures as baseline findings rather than rewriting the notebook.

### Task 2: Add a session-safe data boundary

**Files:** Create src/dashboard_data.py and tests/test_dashboard_data.py.

**Interfaces:**

- DatasetLoad is an immutable result with data, audit, rejected_prices, and fingerprint fields.
- validate_upload(file_bytes) returns DatasetLoad or raises an actionable DatasetValidationError.
- load_project_data(raw_path, cleaned_path) returns DatasetLoad; unavailable historical values are represented as unavailable, not zero.
- dataset_fingerprint(data) returns a content-derived identity used for cache and stale-state invalidation.
- add_offering_ids(data) adds stable IDs derived from normalized offering content and a deterministic duplicate occurrence key; IDs are assigned before UI sorting and filtering.
- activate_uploaded_data(session_state, file_bytes) validates first, then atomically swaps active state and clears stale selections/models only when the fingerprint changes.

- [ ] Reuse clean_dataset for raw-schema uploads and project raw data. Support the cleaned schema with price_outlier without claiming historical duplicate/rejection counts.
- [ ] Validate required fields, types, domains, and positive finite prices before swapping active session data.
- [ ] Keep uploaded bytes/data in Streamlit session state and leave the prior active dataset unchanged when validation fails.
- [ ] Preserve project raw and cleaned CSVs; never write uploaded data to either path.
- [ ] Test malformed headers, invalid values, successful replacement, failure rollback, stable IDs, and content identity changes.

### Task 3: Extract reusable filtering and dashboard calculations

**Files:** Create src/dashboard_analytics.py and tests/test_dashboard_analytics.py.

**Interfaces:**

- apply_filters(data, filter_state) returns the single filtered population used by every view and export.
- summarize_offerings(data) returns matching offer count, distinct hotel count, median price, median recorded guest rating, and usable counts.
- price_summaries(data, group_column) returns count, mean, median, and spread.
- correlation_summary(data) returns complete-pair counts and available coefficients, excluding identifiers.

- [x] Apply city, stars, month, room type, price, and room-size filters consistently.
- [x] Keep outlier inclusion as a view filter over the existing global price_outlier flag.
- [x] When a size range is active, exclude missing room sizes by default and expose an explicit include-unknown choice.
- [x] Handle empty populations without fabricating zero-valued summaries.
- [x] Test filter boundaries, missing-size behavior, outlier stability across city filters, summary/table population agreement, and correlation pair counts.

### Task 4: Build the integrated Streamlit views

**Files:** Create app.py and src/dashboard_views.py; create .streamlit/config.toml if the verified Streamlit theme API needs configuration.

**Interfaces:**

- render_app() owns navigation and shared session state.
- render_overview(data, filter_state), render_explore(...), render_compare(...), and render_data_quality(...) own their respective view contents.
- All views receive the same filtered DataFrame and source-audit structure; views do not implement their own filter logic.

- [x] Add common controls, active-filter summary, matching-offer count, and a working Reset filters action.
- [x] Implement Overview metrics, Plotly price histogram, and city comparison.
- [x] Implement Explore scatter, star and month comparisons, correlation display, and searchable/sortable result table.
- [x] Implement selection of up to three distinct offerings and side-by-side peer context.
- [x] Implement Data quality counts and cleaning-policy explanations, showing unavailable where source history is absent.
- [x] Use Plotly hover, zoom, and reset. Check the installed Streamlit selection API against current official documentation; if unsupported, provide a working explicit offering selector.
- [x] Add a filtered CSV download and an HTML report with filter values, summary measures, findings, and limitations.
- [ ] Test empty states, all controls, sorting/filtering selection correctness, compare selections, exports, and upload rollback using Streamlit testing support where available.

### Stage A acceptance

- Dashboard launches locally and all four pages work against the existing project data.
- Filters, metrics, charts, tables, and exports agree on the matching offerings.
- Valid uploads replace only session data; invalid uploads leave the current data usable.
- Existing notebooks and tests remain intact and baseline checks pass or have documented pre-existing failures.

## Stage B — Advanced analytics

### Task 5: Add peer comparison and deterministic insights

**Files:** Extend src/dashboard_analytics.py, src/dashboard_views.py, and tests/test_dashboard_analytics.py.

- [x] Compare a selected offering against other offerings with the same city, stars, room type, and month.
- [x] Exclude the selected offering from its peer median and peer count. Require at least five other peer observations; show insufficient evidence below that threshold.
- [x] Show peer count, peer median, and percentage difference with clear EUR and percent units.
- [x] Mark rating-at-least-8 and below-peer-median offerings as discussion candidates, not recommendations.
- [x] Generate a small number of deterministic statements from actual filtered calculations, naming population and supporting counts/statistics.
- [x] Test sparse groups, candidate qualification, percentage calculations, and deterministic statements.

### Task 6: Add contextual and outlier sensitivity views

**Files:** Extend src/dashboard_analytics.py, src/dashboard_views.py, and tests/test_dashboard_analytics.py.

- [x] Show price summaries and room-size/price correlation with and without flagged observations, including separate counts.
- [x] Add within-city and within-star contextual comparisons without causal wording.
- [x] Report complete-pair counts for every displayed correlation and omit numeric identifiers.
- [x] Test outlier row counts, correlation pair counts, constant columns, and empty subsets.

### Stage B acceptance

- Peer definitions are visible and consistent across views.
- Sparse peer groups are reported as insufficient evidence without broadening the group.
- Insights are deterministic, tied to current calculations, and state their population.

## Stage C — Evaluated prediction

### Task 7: Implement hotel-group evaluation

**Files:** Create src/prediction.py and tests/test_prediction.py; update requirements.txt.

**Interfaces:**

- make_hotel_split(data, dataset_id, random_state=42) returns train/test row indices and hotel groups.
- build_candidate_pipelines() returns median baseline, a simple regression pipeline, and one tree-based regression pipeline.
- evaluate_candidates(data, split, include_outliers) returns metrics and predictions on the same held-out rows.

- [x] Add scikit-learn after checking compatible versions in the supported environment.
- [x] Use at least 50 distinct eligible hotels as the initial minimum, with a fixed 20% held-out hotel split and at least 10 held-out hotel groups; show evaluation unavailable below the threshold.
- [x] Verify train/test hotel IDs are disjoint. Use the same hotel assignment when comparing outlier inclusion choices.
- [x] Restrict features to city, stars, guest rating, room size, room type, month, and distance where available.
- [x] Fit numeric imputation and categorical encoding inside each sklearn pipeline on training rows only; configure unknown categories safely.
- [x] Compare a median baseline, a simple regression, and one tree model with fixed documented settings; do not tune on held-out rows.
- [x] Report MAE and RMSE in EUR, row and hotel counts, and R-squared only when defined and interpretable.
- [x] Test hotel-group separation, minimum-data behavior, training-only preprocessing, identical evaluation rows, and no feature leakage.

### Task 8: Add explicit prediction actions and state invalidation

**Files:** Extend app.py, src/dashboard_views.py, src/prediction.py, tests/test_prediction.py, and README.md.

- [x] Add a Prediction page with explicit Train/Evaluate and Submit prediction actions; dashboard filters must not retrain the model.
- [x] Store model, split, evaluation, and dataset fingerprint in session state. Clear stale model and comparison state when the active dataset fingerprint changes.
- [x] Provide room-characteristic inputs, estimated nightly price, model name, training dataset identity, supplied/imputed-value display, and flags for unsupported categories or values outside training ranges.
- [x] Describe predictions as conditional training estimates, not causal effects; show no uncertainty interval without a validated method.
- [x] Label the page as a synthetic training demonstration.
- [x] Test explicit training, dataset-change invalidation, unknown categories, imputed inputs, and prediction-pipeline consistency.

### Stage C acceptance

- The held-out evaluation is hotel-disjoint, leakage-safe, and compared with the baseline on identical rows.
- Evaluation is unavailable rather than misleading when the dataset is too small.
- A model is trained only on explicit action and predictions use that same fitted pipeline.

## Task 9: Final verification and handoff

- [x] Run focused new tests and all existing tests.
- [x] Run both notebooks from fresh kernels and preserve their content.
- [x] Launch Streamlit and check navigation, filters, reset, chart selection/fallback, comparison, invalid and valid upload, exports, and prediction actions using Streamlit AppTest.
- [x] Record any interaction that could not be verified, including pre-existing notebook syntax errors.
- [x] Update README with Windows environment setup, data flow, app launch, checks, limitations, and a five-minute demo outline.
- [x] Add one learner exercise per delivered stage without leaving required behavior unfinished.

## Review status

No repository-local AGENTS.md, TODOS.md, Git metadata, or existing app workflow was found during inspection. This plan uses the phase handoff as the project specification. The user chose Python 3.14 for now; Python 3.12 compatibility remains a later check. No code, data, or notebook files were changed while preparing the plan.








