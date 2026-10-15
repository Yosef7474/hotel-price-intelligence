# Hotel Price Intelligence Phase 2

## Instructions for Codex

You are working inside my existing Hotel Price Intelligence Python project. Extend it into an interactive Streamlit application, then add advanced analytics and an evaluated price-prediction feature in stages.

I am Yosef, a Node.js developer learning Python. I have completed the first notebook-based project. Keep the implementation understandable and explain the important Python and analytics decisions briefly as you work.

This file is the handoff brief. Inspect my actual repository before choosing integration details. Do not assume that the earlier proposed file paths, datasets or dependencies exactly match what is now implemented.

### First actions

1. Read repository instructions, including applicable `AGENTS.md` files. Check Git status and preserve my uncommitted work.
2. Inspect the notebooks, Python modules, requirements, tests and hotel CSV files. Identify the real schema, row meaning, cleaning policies and reusable functions.
3. Summarize what already works, what can be reused, and any concrete incompatibilities with this brief.
4. Produce a short repository-specific design and implementation plan. Follow any required design/plan review gates in the repository's workflow. If Superpowers is installed, continue its architectural workflow for this extension without repeating the completed first project's brainstorming.
5. Once the applicable reviews are complete, work through the stages below in order. Ask only about genuinely blocking ambiguities; choose reasonable defaults for routine implementation details.

Do not replace the existing project with a template or build a disconnected demo. Do not start by regenerating my dataset. Keep the notebook working and preserve the original raw CSV.

## 1. Goal and agreed direction

The application should let a user explore hotel prices, compare room offerings, investigate pricing differences and use an evaluated model to estimate a nightly price.

- UI: **Streamlit**.
- Charts: **Plotly**.
- Analysis: reuse the existing **Pandas/NumPy** logic where practical.
- Machine learning: **scikit-learn**, introduced after the dashboard works.
- Runtime: local development, using the project's supported Python environment.
- Learning style: small modules, readable names and concise explanations.

The selected direction is both advanced analytics and machine learning, delivered in stages. Detailed defaults below are implementation guidance, not claims that the local code has already been inspected or built.

## 2. Stage A: interactive dashboard

### Common controls

Provide shared filters for city, star classification, month, room type, nightly price and room size. Include a clear Reset filters action and a toggle to include or exclude flagged outliers.

Use all valid-price offerings by default. Explain that an excluded outlier is not necessarily an incorrect observation. Keep the chosen outlier rule consistent across views; changing a city filter must not silently recalculate the flag using a different population.

For missing room sizes, provide an explicit inclusion choice. Applying an active numeric size restriction excludes unknown sizes unless a separate inclusion control says otherwise. Make this behavior visible.

Filters must update summary statistics, charts, the results table and exports consistently. Display the active filters and number of matching offerings.

### Pages

| Page | Required content |
| --- | --- |
| Overview | Matching offers, distinct hotels, median nightly price, median recorded guest rating, price histogram and city price comparison |
| Explore | Room size versus price scatter plot, star-classification price distribution, month comparison, correlation table or heatmap and a searchable/sortable results table |
| Compare | Select up to three distinct room offerings and compare attributes, prices and peer context side by side |
| Data quality | Available source audit, duplicates, invalid prices, missing values, outlier counts and the documented cleaning policy |

### Chart interaction

- Support Plotly hover details, zoom and reset controls.
- Where supported by the installed Streamlit version, let scatter-point selections show their corresponding offerings in a details table.
- Use stable offering identifiers, not the displayed row position or hotel name alone. A hotel may have multiple offerings.
- If selection events are unavailable in the installed stack, use an explicit offering selector and document the limitation. Do not leave decorative controls that do nothing.
- Label units, statistics and sample sizes. Rating and room-size calculations must show their usable observation counts.
- Explain empty filters rather than showing broken charts or misleading zero-valued statistics.

### Data sources and exports

Start with my existing dataset. Add an optional CSV upload that validates the schema before replacing the active dataset. An invalid upload must leave the previous valid dataset usable and show an actionable error.

Keep uploaded data local to the session by default. Do not overwrite project files with uploads. Offer a CSV download of the current filtered table and a compact downloadable HTML analysis report containing active filters, summary measures, findings and limitations.

If only a cleaned CSV exists, do not invent historical duplicate-removal or rejection counts. Show those audit values as unavailable unless raw data or a recorded audit supports them.

## 3. Stage B: advanced analytics

Add these features to the dashboard after Stage A is working:

### Peer price comparisons

Compare an offering against others sharing city, stars, room type and month. Show the peer definition, peer count, median nightly price and percentage difference from that median.

Use at least five total observations in the comparison group as the initial configurable threshold. Explain whether the selected offering is included in the displayed group median; use one consistent definition throughout.

Groups below the threshold must show insufficient evidence. Do not silently broaden groups just to generate results. Treat lower-price offerings with recorded guest ratings of at least 8 as candidates for discussion, not definitive best-value recommendations.

### Outlier sensitivity

Show price summaries and room-size/price correlation with and without flagged observations. Include both observation counts and explain how exclusion changes the result.

### Contextual relationships

Allow comparisons within cities or star groups. Show complete-pair counts alongside correlations. Exclude numeric identifiers from correlation analysis, and distinguish hotel classification from guest satisfaction.

### Computed insights

Generate a few concise, deterministic insight statements from actual calculations, such as a city median comparison or an outlier-sensitive mean. Each statement must name its population and include supporting numbers. No LLM or paid API is needed.

## 4. Stage C: evaluated prediction and prediction UI

Add a **Prediction** page once the analysis pages are reliable.

### Modeling design

- Target: positive, finite `price_eur`, or the inspected dataset's equivalent field.
- Candidate inputs: city, stars, guest rating, room size, room type, month and distance from the city centre, where available.
- Exclude hotel ID, hotel name, price-derived features, peer medians, price ratios and outlier flags from predictive inputs.
- Use a scikit-learn preprocessing pipeline for numeric missing values and categorical encoding. Fit preprocessing only on training data.
- Start with a median-prediction baseline, a simple regression model and one tree-based regression model. Keep the comparison small and interpretable.
- Fix random seeds and record the training configuration and dataset identity.

### Prevent leakage

Split training and held-out evaluation data by **hotel ID**, so offerings from one hotel cannot appear on both sides. Check that the groups are disjoint. If the dataset lacks a trustworthy hotel identifier, resolve that limitation before reporting generalization to unseen hotels.

Choose model settings using training data only. If cross-validation is needed, keep hotel groups separate in each fold. Evaluate the final candidate on the untouched held-out data and compare it with the baseline on the same rows.

Use the same hotel split when comparing outlier-inclusion choices, and report any resulting changes in evaluation row counts. Never select settings to improve the held-out test score.

### Evaluation interface

Show held-out MAE and RMSE in EUR, train/test row and hotel counts, and an actual-versus-predicted chart. Show R² only when it is defined and interpretable. Report when a model fails to improve on the baseline.

Require enough valid observations and distinct hotel groups for a meaningful split. Choose and document a minimum appropriate to the inspected data. If the project contains only the small learning fixture, keep the dashboard usable and explain why evaluation is unavailable; do not present a credible-model claim from that fixture.

### Prediction playground

Provide a form for the supported room characteristics. On submission, show the estimated nightly price, the model used and the dataset it was trained on. Flag unsupported categories and inputs outside observed training ranges.

Show which values were supplied and which were imputed. Predictions are conditional estimates, not proof that changing a room attribute causes a particular price change. Do not display an uncertainty interval unless a defensible method has been implemented and explained.

Training must be an explicit action, not something repeated for every slider change. A dataset replacement invalidates the previous trained model and evaluation. Dashboard display filters must not silently retrain the model or change its evaluation population.

For synthetic data, label the page as a training demonstration of generated relationships, not a real hotel-market pricing service.

## 5. UI design requirements

Use a minimal, professional analytics layout:

- Neutral page background, dark readable text, teal primary accent and restrained warning colors.
- Clear navigation, consistent spacing and a readable content width.
- A small set of useful metric cards, followed by charts and detail tables.
- Labels that make sense to a nontechnical user; explain terms such as peer median in short help text.
- Consistent currency, units and decimal formatting.
- Legible chart labels and hover text; do not rely on color alone to distinguish information.
- Useful loading, empty, invalid-file and unavailable-model states.
- A layout that remains usable on a narrower browser window.

Prefer Streamlit's supported theme and layout APIs. Avoid fragile DOM-targeting CSS, custom JavaScript workarounds and unnecessary animation. Keep implementation details out of the normal user flow.

## 6. Integration and maintainability

Adapt to the existing repository rather than imposing a new architecture. A reasonable structure is an app entry point, page/view modules, shared analytics functions, data loading/validation utilities and modeling utilities. Keep UI rendering separate from calculations so the notebook and app can share logic.

Do not turn the app into one large script or add a separate backend, database, authentication system or deployment infrastructure.

Use session state for selected data, filters, comparison selections and model state. Key cached results by the actual dataset content and relevant configuration, not only a filename. When data changes, clear stale comparisons and model results; preserve session isolation for uploaded data.

Reuse the environment and dependency manager already present. Add compatible dependencies and record versions after verification. Use current official documentation for version-sensitive Streamlit, Plotly and scikit-learn APIs.

## 7. Verification and completion criteria

Use focused tests for meaningful analytical behavior, plus UI checks:

- Existing relevant tests and notebook behavior remain intact.
- Filtered statistics, chart populations, detail rows and exports agree.
- Repeated hotel IDs remain valid when offerings differ.
- Missing values, invalid prices, empty subsets and sparse peer groups behave as documented.
- CSV upload validates fields, types and ranges and does not overwrite the raw dataset.
- Selecting an offer or chart point opens the correct record after filtering or sorting.
- Dataset changes clear stale comparison and trained-model state.
- Train/test hotel groups are disjoint; preprocessing does not learn from the test set.
- Baseline and model metrics are computed on the same evaluation rows.
- Prediction inputs follow the trained pipeline and handle unknown categories safely.

Run the Streamlit app and check navigation, filters, reset, selections, comparison, invalid upload, valid upload, exports and model actions. Use Streamlit's testing support and browser checks where available. State any interaction that could not be verified; do not call an untested flow complete.

## 8. Delivery and learning handoff

For each stage, report what changed, why, how it was verified and any remaining limitations. Finish with:

1. Working integrated Streamlit app and reusable analysis/modeling code.
2. Updated dependencies, focused tests and README.
3. Exact commands for my local setup, app launch and relevant checks, including Windows instructions where needed.
4. A short explanation of the data flow and important design decisions.
5. A five-minute demo outline for Jeet.

Include one small learner exercise per stage, such as adding a filter, explaining a peer comparison or interpreting a model error. Exercises supplement delivery; they must not leave required app behavior unfinished.

Do not add an LLM assistant, scraping, paid APIs, accounts, a map requiring invented coordinates, or cloud deployment unless I request those separately.

**Start by inspecting the repository and showing the concise integration design and staged plan required by the applicable workflow.**
