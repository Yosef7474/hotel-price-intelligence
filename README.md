# Hotel Price Intelligence

A beginner Python project for exploring how hotel location, star classification, guest ratings, room size, and month relate to nightly prices.

All records are fictional and generated from documented teaching assumptions. Results demonstrate analytical methods; they do not describe real hotel markets or establish causal effects.

## Project files

- src/generate_data.py — creates reproducible synthetic data and a generation audit.
- src/clean_data.py — validates data, removes exact duplicate rows, separates invalid prices, flags outliers, and reports a cleaning audit.
- data/hotels_raw.csv — generated input data with controlled imperfections.
- data/hotels_cleaned.csv — separate price-analysis data with an outlier flag.
- notebooks/01_hotel_analysis.ipynb — six learning stages, exercises, summaries, charts, correlations, and finding prompts.
- tests/test_generate_data.py — focused generator checks.
- tests/test_clean_data.py — focused cleaning and validation checks.
- Hotel_Price_Intelligence_Learning.ipynb — supplementary small practice notebook.
- 2026-10-12-hotel-price-intelligence-design.md — project design specification.

## Environment setup

The originally intended environment is Python 3.12, VS Code, and the Microsoft Python and Jupyter extensions. For Phase 2, the selected and verified environment is Python 3.14.2 in `.venv`; Python 3.12 compatibility has not been verified.

From PowerShell in the project root:

1. Confirm the current environment: `.\.venv\Scripts\python.exe --version`.
2. If you need a new environment, create it with your installed Python: `py -3.14 -m venv .venv`.
3. Install project packages: .\.venv\Scripts\python.exe -m pip install -r requirements.txt.
4. In VS Code, choose `.venv\Scripts\python.exe` as the Python interpreter and notebook kernel.
5. If VS Code does not list it, register the kernel with `.\.venv\Scripts\python.exe -m ipykernel install --user --name hotel-price --display-name "Python 3.14 (hotel-price)"`.

The project `.venv` has been checked with scikit-learn 1.9.1, pandas 3.0.6, NumPy 2.5.4, Streamlit 1.66.0, and Plotly 7.1.0. `requirements.txt` remains unpinned.

## Generate and analyze

Run commands from the project root.

1. Generate the raw dataset: .\.venv\Scripts\python.exe src\generate_data.py.
2. Optionally run cleaning separately: .\.venv\Scripts\python.exe src\clean_data.py.
3. Open notebooks/01_hotel_analysis.ipynb, select the project kernel, and run all cells from top to bottom.

The generator defaults to seed 42 and writes 1,000 base offerings plus 20 exact duplicate copies to data/hotels_raw.csv. It injects 50 missing guest ratings, 30 missing room sizes, and 10 unusually high but valid prices among the base offerings. It refuses to overwrite an existing raw file; pass --force only when replacement is intentional. Use --seed or --output to change the generation seed or output path. The audit prints the seed, assumptions, and injected issue counts.

The cleaner reads the raw file without modifying it and writes data/hotels_cleaned.csv. An intentional rerun refreshes that cleaned output. clean_dataset() returns cleaned rows, separated invalid-price rows, and an audit for inspection in Python or the notebook.

## Dashboard

The local Streamlit dashboard starts from the existing raw CSV (or the cleaned CSV if the raw file is unavailable). The data boundary validates and cleans the source in memory, assigns stable offering IDs, and calculates a content fingerprint. CSV uploads are validated before activation and stay in the current session; invalid uploads leave the active dataset unchanged. Uploaded files do not overwrite project data. Shared filters feed the analytics views, and prediction uses hotel-grouped training and holdout sets.

Install the listed packages in the Python environment you selected, then launch from the project root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The dashboard has Overview, Explore, Compare, Data quality, and Prediction pages. Its common filters drive the visible metrics, charts, comparison context, and CSV/HTML downloads. A room-size range excludes unknown sizes unless the include-unknown control is selected.

The Prediction page is a synthetic training demonstration. Click **Train/Evaluate** to fit and compare the median baseline, linear regression, and random forest on a hotel-disjoint holdout split. Dashboard filters do not retrain the candidates. Choose Linear regression or Random forest and submit room characteristics for a conditional nightly price estimate from that fitted pipeline. The page identifies the active dataset, marks missing inputs handled by training-only imputation, and flags unseen categories and values outside training ranges. No uncertainty interval is provided, and estimates are not causal or real-market prices. A changed active dataset clears its stored evaluation and prediction state.

## Dataset fields

Each base row represents one fictional room offering for one hotel in one month. Hotel IDs repeat when a hotel's room type or month differs. Price is one nightly amount in euros.

| Field | Meaning |
| --- | --- |
| hotel_id | Hotel identifier; repeated across offerings from the same hotel |
| hotel_name | Fictional hotel name |
| city | Synthetic location |
| star_rating | Classification from 1 to 5 |
| guest_rating | Satisfaction score from 0 to 10; may be missing |
| room_size_m2 | Positive room area in square metres; may be missing |
| room_type | Standard, Deluxe, or Suite |
| month | Month number from 1 to 12 |
| distance_center_km | Nonnegative distance from the city centre |
| price_eur | Positive finite nightly price for price analysis |

The generator varies price using synthetic city baselines, star classification, room size and type, month, distance, and random variation. Guest rating is generated separately from stars. These coefficients are not market estimates.

## Cleaning rules

- All ten required fields must exist. Invalid non-price fields or inconsistent hotel identity attributes stop cleaning with an informative error.
- Exact full-row duplicates are removed before derived flags are added. Different offerings from the same hotel are retained.
- Missing guest ratings and room sizes are kept; calculations omit them only when they need the missing field.
- Missing, nonnumeric, infinite, zero, and negative prices are separated from price analysis.
- Unusual positive finite prices are retained and flagged using global Q1 minus 1.5 times IQR and Q3 plus 1.5 times IQR fences.
- The cleaning audit reports counts and remaining missing values. Input rows minus duplicate copies must equal retained plus separated invalid-price rows.

## Focused checks

Run from the project root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The checks cover data generation and cleaning, upload validation and rollback, shared filters and summaries, offer selection and exports, hotel-group evaluation, prediction inputs, and dataset-change invalidation. The full suite passed in the selected `.venv` during Phase 2 handoff. Streamlit `AppTest` also exercised page navigation, filters/reset, offer comparison, exports, valid and invalid uploads, and train/submit prediction actions.

Both notebooks were run from fresh kernels without saving outputs. Their files were preserved. Execution currently stops on existing syntax errors: `notebooks/01_hotel_analysis.ipynb` contains malformed f-strings in its city and star comparison cells, and `Hotel_Price_Intelligence_Learning.ipynb` contains an incomplete `for n in four` statement. Earlier cells run before these failures. These learning-material issues are recorded for a separate repair.

## Five-minute demo and learner exercises

1. **Minute 1 — Start and orient:** launch the dashboard, show the synthetic-data notice, matching-offer count, and shared filters.
2. **Minute 2 — Explore:** filter to one city, inspect the price chart and table, then reset filters.
3. **Minute 3 — Compare:** select an offering and explain that peers match city, stars, room type, and month; sparse groups remain insufficient evidence.
4. **Minute 4 — Data quality:** review missing values and outlier flags; explain that a flagged price is retained and is not automatically an error.
5. **Minute 5 — Prediction:** click Train/Evaluate, compare candidates with the median baseline, then submit characteristics and review any imputation or input warnings.

Practice tasks:

- **Stage A:** apply a room-size range, compare the default exclusion of unknown sizes with the include-unknown option, and export the filtered rows.
- **Stage B:** compare offers in a matching peer group, then toggle flagged prices in Explore and describe how the summaries change.
- **Stage C:** train/evaluate once, submit a known category, then try an unseen city and an out-of-range star value; explain the resulting flags.

## Limitations

The dataset is fictional and supports practice only. It does not model taxes, availability, booking conditions, amenities, or actual customer behavior. Hotels contribute repeated offerings, so rows are not independent hotels. Missing values reduce the population for analyses that require them. The global outlier rule ignores peer differences and does not prove an observation is an error. Group comparisons and correlations do not establish causation.
