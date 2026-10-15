# Hotel Price Intelligence Design Specification

**Prepared for:** Yosef Dejene  
**Date:** 12 October 2026  
**Status:** Written specification awaiting review. All four architecture sections have conversational approval.  
**Scope of this document:** Project design only. Implementation planning and project implementation are not authorized by this document.

## 1. Purpose and learning outcomes

Hotel Price Intelligence is a beginner Python training project for exploring how hotel location, star classification, guest satisfaction and room size relate to nightly price. It supports Yosef's two-week analytics training brief and a short demonstration of the findings to Jeet.

The first deliverable is a guided Jupyter Notebook supported by small reusable Python modules. Its purpose is to make the data transformations and analytical reasoning understandable, rather than to produce a finished customer application.

By completion, the learner should be able to:

- Explain variables, lists, dictionaries, loops, functions and imports.
- Load a CSV into Pandas, inspect its schema and filter observations.
- Identify and handle duplicate rows, missing values and invalid prices.
- Calculate group counts, means, medians, distributions and correlations.
- Explain four charts and compare findings with and without unusual prices.
- Present three findings supported by computed evidence and clear limitations.

All records are fictional. Results demonstrate analytical methods and the generator's assumptions; they do not describe actual hotel markets or establish causal effects.

## 2. Approved design decisions and boundaries

| Decision | Agreed choice |
| --- | --- |
| Dataset source | Generated realistic hotel data with controlled imperfections |
| Dataset size | 1,000 base offerings, followed by approximately 20 duplicate copies |
| First deliverable | Jupyter Notebook |
| Analysis depth | Exploratory data analysis, including cleaning and correlations |
| Architecture | Notebook first, with separate generation and cleaning modules |
| Teaching method | Explanation, small example, learner exercise, review and interpretation |
| Execution | Local computer, using VS Code and an isolated Python environment |

Version one excludes live hotel APIs, web scraping, machine learning, a Streamlit dashboard, a backend, authentication and a database. It requires no external service credentials or paid services. Initial installation of Python packages may require internet access.

The existing PDF and small practice notebook are learning aids. Their 24 base offerings and deliberately negative example price do not replace the 1,000-record project or change its standard generation requirements.

## 3. Architecture and component responsibilities

The generator produces the raw CSV. The analysis notebook loads and inspects that file, applies the cleaning module, analyzes the retained observations, and presents charts and findings. Cleaning saves a separate CSV and preserves the raw file.

| Planned project path | Responsibility |
| --- | --- |
| `src/generate_data.py` | Reproducible generation of base offerings and documented imperfections |
| `src/clean_data.py` | Schema checks, full-row deduplication, price validation and cleaning audit |
| `data/hotels_raw.csv` | Original generated dataset, including injected imperfections |
| `data/hotels_cleaned.csv` | Separate dataset retained for price analysis, including an outlier flag |
| `notebooks/01_hotel_analysis.ipynb` | Six learning stages, exercises, analyses, charts and findings |
| `tests/test_clean_data.py` | Focused tests of cleaning policies and error cases |
| `tests/test_generate_data.py` | Focused tests of reproducibility, schema and injected issue counts |
| `requirements.txt` | Dependencies with versions verified during implementation |
| `README.md` | Setup, run instructions, dataset meaning and limitations |

The second test file makes the approved generator-testing requirement explicit; it does not add another subsystem. Charts and findings remain inside the notebook. Separate chart exports are optional.

The generator must not overwrite an existing raw CSV silently. Cleaning reads raw data without modifying it and may refresh the separate cleaned output on an intentional rerun. Invalid-price observations and audit counts must remain available in the notebook for inspection; an additional rejected-record CSV is optional.

This specification is saved at `docs/superpowers/specs/2026-10-12-hotel-price-intelligence-design.md`. It is a standalone document until attached to a project repository.

## 4. Dataset meaning and schema

Each base row represents a room offering for one hotel in one month, with a nightly price in euros. Repeated hotel IDs are legitimate when the offerings differ. The dataset does not represent individual bookings, customers or reviews.

| Column | Type | Meaning and valid domain |
| --- | --- | --- |
| `hotel_id` | Integer | Hotel identifier; shared by that hotel's different offerings |
| `hotel_name` | String | Fictional hotel display name |
| `city` | String | Location used for city-level comparisons |
| `star_rating` | Integer | Hotel classification from 1 to 5 |
| `guest_rating` | Float | Satisfaction score from 0 to 10; may be missing |
| `room_size_m2` | Float | Positive finite room area; may be missing |
| `room_type` | String | Room category |
| `month` | Integer | Month from 1 to 12 |
| `distance_center_km` | Float | Non-negative finite distance from the city centre |
| `price_eur` | Float | Positive finite nightly price for price analysis |

Hotel name, city, star classification and distance must remain consistent for a given hotel ID. Star classification and guest satisfaction are distinct measures.

Prices use one currency and one nightly basis. Mixed currencies and dates are excluded from this version. The generator and README must state that taxes, booking conditions and availability are not modeled.

## 5. Generation rules and controlled imperfections

The generator must create meaningful, imperfect relationships rather than assign every price independently. City baseline, star classification, room size and month influence price, with random variation preventing perfect relationships. Room category influences room size. Guest satisfaction is a separate variable and must not be interchangeable with stars.

A fixed random seed controls every random operation. The same configuration, seed and verified dependency versions must reproduce the same dataset. Generator parameters must be documented as synthetic assumptions, not estimated market facts.

Base offerings must be distinct full rows before deliberate duplicate copies are appended. The generation audit must record the base count, seed, parameters and injected issue counts. It may be returned by the generator and shown in the notebook; a separate metadata file is optional.

| Imperfection | Agreed approximate amount | Treatment |
| --- | --- | --- |
| Missing guest ratings | 5% of base offerings | Preserve missing values |
| Missing room sizes | 3% of base offerings | Preserve missing values |
| Unusually high prices | 1% of base offerings | Retain and investigate |
| Duplicate full rows | 2% of the base count, appended afterward | Remove duplicate copies during cleaning |

Percentages apply to the 1,000 base offerings. Duplicate copies can repeat missing values, so raw-file null counts need not equal the base injection counts. Missingness in different columns may overlap. Known injected high-price records are not necessarily identical to the records flagged by a statistical outlier rule.

### Concrete defaults proposed for written-spec review

The earlier approvals did not fix a seed value, exact issue counts, city list, pricing coefficients or peer-screen thresholds. The following defaults resolve those details for review without expanding scope:

| Detail | Proposed default |
| --- | --- |
| Seed | 42 |
| Cities | Paris, Lyon, Nice, Lille, Marseille and Bordeaux |
| Room categories | Standard, Deluxe and Suite |
| Seasonal coverage | All 12 months; room category must not be determined by month |
| Base and raw counts | 1,000 distinct base offerings plus exactly 20 copied rows; 1,020 raw rows |
| Missing-value injection | Exactly 50 base ratings and 30 base room sizes, selected without replacement within each column |
| High-price injection | Exactly 10 distinct base offerings, inflated while remaining positive and finite |
| Natural duplicate prevention | Unique `(hotel_id, room_type, month)` combinations among base offerings |
| Standard price validity | No deliberately invalid prices in the normal generated dataset; invalid prices appear in test fixtures |

The generator must provide enough distinct offering combinations to meet the base count. Exact synthetic pricing coefficients and hotel population size are implementation choices within the rules above; they are not market claims or requirements to reproduce a specific correlation coefficient. The notebook must be allowed to discover weaker or unexpected relationships.

## 6. Cleaning, validation and error handling

Schema validation requires all ten columns. Full-row deduplication uses the original fields, before adding derived flags. The cleaner must never deduplicate solely by hotel ID.

| Condition | Required behavior |
| --- | --- |
| Raw CSV missing | Stop with an error naming the expected path and explaining that generation is required |
| Required column missing | Stop with an error listing the missing names |
| Non-price schema violation | Stop with an informative validation error identifying the affected fields or rows |
| Duplicate full row | Retain one copy and report the removed-copy count |
| Missing guest rating | Retain the row; omit it only from rating-dependent calculations |
| Missing room size | Retain the row; omit it only from size-dependent calculations |
| Invalid price | Separate and report the row; exclude it from price analysis |
| Extreme but valid price | Flag and retain; compare analyses with and without flagged observations |
| Empty filtered dataset | Show an understandable message and skip the affected calculation or chart |
| Insufficient or constant correlation data | Report the coefficient as unavailable rather than interpreting it as zero |

For price validation, nonnumeric values are treated as invalid after attempted numeric conversion. Missing, infinite, zero and negative prices are excluded. Missing ratings and sizes are preserved without automatic imputation. Other fields follow the schema domains; valid domain checks apply to nonmissing ratings and sizes.

The cleaning audit must report input rows, duplicate copies removed, invalid-price rows separated, retained rows, remaining missing values and flagged prices. Row counts must reconcile: raw rows minus removed copies equals deduplicated rows; deduplicated rows equal retained plus rejected rows.

**Proposed outlier default for review:** flag prices outside the global `Q1 - 1.5 × IQR` and `Q3 + 1.5 × IQR` fences, with `IQR = Q3 - Q1`. Store a Boolean `price_outlier` column. Explain that the global rule ignores peer differences and does not establish an error. An empty valid-price population must stop price analysis with a clear message.

## 7. Six-stage notebook learning workflow

| Stage | Learning and analysis | Evidence produced |
| --- | --- | --- |
| 1. Python foundations | Variables, types, lists, dictionaries, loops, functions and imports using small hotel examples | Learner exercises and explanations of outputs |
| 2. Load and inspect | Load the CSV; inspect shape, types, ranges, missing values, duplicate rows and distinct hotels | Dataset description and initial quality summary |
| 3. Clean and validate | Apply the documented policies; inspect rejected rows and reconcile counts | Cleaning audit and separate cleaned CSV |
| 4. Compare and visualize | Summarize prices overall, by city, by stars and by month; include counts, means, medians and spread | Summary tables and core charts |
| 5. Explore relationships | Compare room size and ratings with price; calculate correlations, usable-pair counts and outlier sensitivity | Scatter plot, coefficients and contextual comparisons |
| 6. Explain and review | Write three measured findings, discuss limitations and reproduce the analysis | Readable conclusions and a short demo outline |

Each stage uses a short explanation, a small worked example, an exercise and a checkpoint. Markdown records the question, the population being analyzed, the interpretation and limitations. Code and results should remain understandable to a beginner.

The four required charts are a price histogram, a city price comparison, a room-size versus price scatter plot, and a star-classification price comparison. Every chart must state its statistic, units and relevant population. A correlation heatmap or month chart is optional.

## 8. Analysis and interpretation requirements

Group summaries must include observation counts. Means and medians must be distinguished, particularly when unusual prices affect results. Distinct hotel counts must not be confused with offering counts.

Correlation calculations must use complete pairs of the relevant variables and report the number of usable pairs. Numeric hotel IDs must not be treated as analytical measurements. Pearson correlation is the baseline; Spearman rank correlation is an optional comparison. Any optional method requires its dependencies to be included and verified.

Compare at least one price summary and the room-size/price correlation with and without flagged prices. Preserve both results and explain the sensitivity. Comparisons within city or star groups help explore confounding but must not be described as proving causation. Repeated offerings from the same hotel must be acknowledged.

Potential good-value offerings may be explored as a transparent heuristic rather than a definitive ranking. **Proposed screening default for review:** compare price with the median of offerings sharing city, stars, room type and month; require at least five peer observations and guest rating of at least 8. An offering below its peer median is a candidate for discussion. If no group meets the minimum count, report insufficient evidence. This screen omits amenities, cancellation terms and other booking conditions.

## 9. Development setup and reproducibility

The agreed environment is Python 3.12, VS Code with Python and Jupyter support, a local `.venv`, Pandas, NumPy, Matplotlib, Seaborn and pytest. The notebook kernel must use that environment.

During implementation, dependencies must be pinned to versions verified together. README instructions must cover environment creation, package installation, dataset generation, kernel selection and notebook execution. Paths must work from the documented project location rather than depend on an author's machine-specific absolute paths.

Restarting the kernel and running every cell in order must reproduce the analysis without relying on previously executed cells. The seed and generation configuration must be visible in the notebook or its referenced audit. This specification does not install packages or create the project environment.

## 10. Testing and validation strategy

Testing stays small and readable. Focus on behaviors that could silently change an analysis, not exhaustive infrastructure.

| Area | Required checks |
| --- | --- |
| Generator | Same seed and configuration reproduce the same rows; schema and ranges are valid; hotel attributes are stable; base and injected counts are correct |
| Duplicate handling | One full-row copy is removed; distinct offerings with the same hotel ID remain |
| Missing values | Missing ratings and sizes remain missing; unrelated valid observations are retained |
| Price validity | Positive finite values are retained; missing, nonnumeric, infinite, zero and negative values are separated |
| Validation failures | Missing required columns and invalid non-price fields produce informative errors |
| Outlier behavior | A valid high-price fixture is flagged under the chosen rule and retained |
| Audit | Removed, retained and rejected row counts reconcile; the raw input remains unchanged |
| Notebook | Fresh-kernel execution succeeds; empty filters are handled; charts, pair counts and findings agree with the analyzed rows |

Small hand-checkable test fixtures establish expected results. The generator's recorded injection audit supplies the basis for checking its controlled imperfections. Tests must not demand a predetermined real-world finding from synthetic data.

## 11. Completion criteria and final deliverables

The project is complete when:

- The documented local setup works with the verified dependencies.
- Generation reproduces 1,000 base offerings and the documented imperfections.
- Raw data is preserved and cleaned output is saved separately.
- Validation and cleaning tests pass, and cleaning counts reconcile.
- The notebook runs from a restarted kernel in documented order.
- Four labelled charts and group summaries are present and explained.
- Correlations include usable-pair counts and an outlier sensitivity comparison.
- Three findings cite calculated evidence and acknowledge synthetic data, confounding and relevant missing values.
- Yosef can independently explain and repeat the main operations and present a short walkthrough to Jeet.

Final deliverables are the generator, cleaner, tests, raw and cleaned CSVs, analysis notebook, requirements file and README. The notebook contains the charts and conclusions; the existing learning book remains supplementary.

The previously discussed five-day project schedule is a learning estimate within the broader two-week training period, not a deadline or an execution plan.

## 12. Written specification review

Review this specification for consistency with the four approved sections, especially the explicitly proposed defaults in Sections 5, 6 and 8. Those defaults turn approximate requirements into concrete review choices; they are not represented as previously approved details.

The architecture, schema, workflow and completion requirements form one notebook-based EDA project. No implementation plan, project source modules or project environment have been created as part of this specification request.

Written-spec approval is a separate review stage. Implementation planning remains paused until requested; project implementation requires the later planning and execution approvals.
