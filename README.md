# THERADIOMICS Study Designer

**A local-first radiomics study-design dashboard for small-sample binary outcome prediction.**

THERADIOMICS Study Designer is a hybrid **Bun + Python** application that helps radiomics researchers explore whether a dataset is statistically plausible for developing a compact binary prediction model. It combines patient-level radiomics preprocessing, internal validation, candidate model benchmarking, Monte Carlo sample-size simulation, pruning-threshold sweeps, and analytical guardrails inspired by shrinkage/event-rate sample-size frameworks.

> This project is intended for **research planning, feasibility analysis, and methodological transparency**. It is not a clinical decision-support device.

---

## Why this exists

Radiomics datasets are often rich in features but poor in independent patients. This creates a dangerous asymmetry:

```text
many candidate features + few patients = unstable signatures + optimistic AUC
```

THERADIOMICS Study Designer was built to make this tension visible. Instead of reporting only one attractive AUC, it asks harder questions:

- How many patient-level observations do we really have?
- How many features survive correlation pruning?
- Is the selected radiomic signature stable?
- Does performance survive nested/grouped validation?
- How often could a similar AUC arise by chance?
- How many final predictors are defensible for the available sample size?
- What happens if the pruning threshold changes?
- Are Monte Carlo estimates coherent with prevalence, event-rate, and shrinkage guardrails?

---

## Core features

### Patient-level radiomics pipeline

- Loads a radiomics Excel dataset named `Features_all.xlsx`.
- Aggregates lesion-level features to patient level.
- Builds a binary outcome column.
- Applies correlation pruning with configurable threshold.
- Runs LASSO/L1 feature selection.
- Trains compact logistic-regression models.

### Validation and robustness

- Grouped nested cross-validation.
- Leave-one-out cross-validation.
- Permutation testing.
- Bootstrap feature-stability analysis.
- Candidate model comparison for 1-, 3-, 5-, and LASSO top-k signatures.

### Monte Carlo sample-size simulation

- Simulates expected AUC behaviour across increasing sample sizes.
- Uses the observed or user-defined endpoint prevalence.
- Reports expected events and non-events.
- Estimates the probability of reaching a clinically useful AUC.

### Analytical sample-size guardrails

The dashboard adds methodological guardrails for binary prediction modelling:

- event-rate / prevalence precision;
- target global shrinkage;
- maximum safe number of predictors;
- required N for 1, 2, 3, 5, or 10 predictors;
- conservative interpretation of small-sample model development.

### Pruning-threshold sweep

From the dashboard you can run several pruning thresholds in sequence, for example:

```text
0.70,0.75,0.80,0.85,0.90
```

The sweep produces a comparative table showing:

- threshold;
- number of surviving features;
- nested CV AUC;
- LOOCV AUC;
- best candidate model;
- failed thresholds, if any.

### Local dashboard

The Bun web server provides:

- dataset upload;
- backend launch;
- real-time logs;
- results rendering;
- model listing;
- threshold sweep controls;
- integrated guide page.

---

## Architecture

```text
THERADIOMICS Study Designer
│
├── Bun / TypeScript bridge
│   ├── serves the dashboard
│   ├── uploads the Excel dataset
│   ├── launches Python analyses
│   ├── streams live logs
│   └── exposes JSON results to the frontend
│
└── Python analysis engine
    ├── dataset loading
    ├── patient-level aggregation
    ├── correlation pruning
    ├── LASSO feature selection
    ├── nested/grouped validation
    ├── permutation testing
    ├── bootstrap stability
    ├── candidate model benchmarking
    ├── Monte Carlo sample-size simulation
    └── analytical sample-size guardrails
```

---

## Expected project structure

```text
RadiomicsSimulator-main/
├── server.ts
├── package.json
├── start_theradiomics.py
├── start_theradiomics.sh
├── start_theradiomics.command
├── start_theradiomics.bat
│
├── results/
│   ├── THERADIOMICS_results_dashboard_backend.html
│   └── guide.html
│
├── data/
│   └── uploads/
│       └── Features_all.xlsx          # user-provided, not committed
│
├── models/                            # generated, not committed
│
└── python/
    ├── run_analysis.py
    ├── run_threshold_sweep.py
    ├── train_final_model.py
    ├── train_candidate_models.py
    │
    ├── core/
    │   ├── dataset_loader.py
    │   ├── feature_pruning.py
    │   └── patient_aggregation.py
    │
    ├── analysis/
    │   ├── bootstrap_stability.py
    │   ├── lasso_selection.py
    │   ├── loocv.py
    │   ├── model_benchmark.py
    │   ├── nested_logistic_cv.py
    │   ├── permutation_test.py
    │   ├── sample_size_guardrail.py
    │   └── sample_size_simulation.py
    │
    └── results/                       # generated JSON/TXT, not committed
```

---

## Requirements

### Runtime

- Python 3.10+
- Bun

### Python packages

Install the scientific stack in your environment:

```bash
pip install numpy pandas scipy scikit-learn openpyxl joblib
```

Optional but recommended:

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
pip install numpy pandas scipy scikit-learn openpyxl joblib
```

---

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/<your-user>/theradiomics-study-designer.git
cd theradiomics-study-designer
```

### 2. Add your dataset

Place your Excel file here:

```text
data/uploads/Features_all.xlsx
```

or upload it directly from the dashboard.

### 3. Start the dashboard

#### macOS / Linux

```bash
chmod +x start_theradiomics.sh
./start_theradiomics.sh
```

#### Windows

Double-click:

```text
start_theradiomics.bat
```

#### Cross-platform Python launcher

```bash
python start_theradiomics.py
```

Then open:

```text
http://127.0.0.1:8765
```

---

## Dashboard workflow

1. Upload or confirm `Features_all.xlsx`.
2. Choose the pruning threshold.
3. Choose the final-model feature count.
4. Choose the number of Monte Carlo simulations.
5. Run either:
   - a single analysis;
   - a pruning-threshold sweep.
6. Watch the logs in real time.
7. Click **Aggiorna risultati**.
8. Read the dashboard sections:
   - dataset overview;
   - validation metrics;
   - candidate models;
   - Monte Carlo simulation;
   - analytical guardrails;
   - pruning-threshold sweep;
   - saved models.

---

## Configuration from the dashboard

The frontend passes run options to Python through environment variables.

| Dashboard control | Python environment variable | Meaning |
|---|---|---|
| Single pruning threshold | `THERADIOMICS_PRUNING_THRESHOLD` | Correlation-pruning cutoff |
| Threshold sweep | `THERADIOMICS_PRUNING_THRESHOLDS` | List of thresholds to test |
| Final features k | `THERADIOMICS_TOP_N_FINAL_MODEL_FEATURES` | Number of features in final saved model |
| Primary model k | `THERADIOMICS_PRIMARY_MODEL_FEATURES` | Number of features used in validation/permutation |
| Monte Carlo simulations | `THERADIOMICS_N_SAMPLE_SIZE_SIMULATIONS` | Number of simulations per N |

Additional analytical guardrail parameters can be changed in `python/run_analysis.py` or passed as environment variables:

| Variable | Default | Meaning |
|---|---:|---|
| `THERADIOMICS_EXPECTED_EVENT_PREVALENCE` | observed prevalence | Expected event rate for design calculations |
| `THERADIOMICS_RISK_MARGIN` | 0.10 | Accepted error on event-rate estimate |
| `THERADIOMICS_TARGET_SHRINKAGE` | 0.90 | Target global shrinkage |
| `THERADIOMICS_DELTA_NAGELKERKE` | 0.05 | Maximum apparent-adjusted performance gap |
| `THERADIOMICS_EXPECTED_R2_CS_ADJ` | 0.08 | Expected adjusted Cox-Snell R² |

---

## Outputs

After each successful run, Python writes:

```text
python/results/analysis_results.json
python/results/candidate_model_comparison.json
python/results/nested_cv_fold_details.json
python/results/summary.txt
```

A threshold sweep also writes:

```text
python/results/threshold_sweep_results.json
python/results/analysis_results_threshold_070.json
python/results/analysis_results_threshold_075.json
...
```

Saved model objects are written to:

```text
models/
```

These are generated artifacts and should usually be excluded from Git.

---

## Interpreting the most important numbers

### AUC

AUC estimates discrimination. In small radiomics datasets, apparent or unstable AUC can be misleading. Prefer nested/grouped validation and compare it with permutation results.

### Nested CV AUC

This is the main internal-validation estimate. Feature selection occurs inside each fold, which reduces leakage.

### LOOCV AUC

Useful in very small datasets, but can still be unstable. Interpret together with nested CV, feature stability, and permutation testing.

### Permutation test

Randomly shuffles the labels. If random labels frequently achieve similar AUC, the observed model is probably not reliable.

### Bootstrap stability

Counts how often each feature is selected across bootstrap resamples. A good candidate feature should not appear only by chance in one split.

### Monte Carlo power

Estimates how often a synthetic study of size N reaches the selected AUC criterion under the assumed signal strength.

### Event-rate guardrail

Checks whether N is sufficient to estimate the baseline event rate with a chosen margin of error.

### k_max

Approximate maximum number of predictors that can be safely incorporated for the chosen N, prevalence, shrinkage target, and expected R².

### N shrinkage

Approximate sample size required to support a chosen number of predictors while controlling overfitting.

---

## Recommended interpretation strategy

For small radiomics studies, prefer a conservative reading:

```text
35 patients     -> hypothesis-generating only
75-100 patients -> possible one-feature feasibility signal
150-200 patients -> plausible compact 1-2 predictor signature
300-400 patients -> stronger calibration/event-rate confidence
500+ patients   -> safer multivariable development setting
```

The exact thresholds depend on endpoint prevalence, expected model strength, and the number of final predictors.

---

## Development notes

The current design intentionally keeps the heavy statistics in Python. Bun is used as a local web bridge:

- no database required;
- no cloud upload required;
- all data stay local;
- results are plain JSON files;
- the dashboard can be inspected and modified directly.

---

## What not to commit

Do not commit patient data, generated results, or trained model binaries.

Recommended `.gitignore` entries:

```gitignore
data/uploads/*.xlsx
models/*.joblib
models/*.json
python/results/*.json
python/results/*.txt
python/results/analysis_results_threshold_*.json
python/results/summary_threshold_*.txt
__pycache__/
*.pyc
.venv/
node_modules/
```

---

## Scientific caution

THERADIOMICS Study Designer does not prove that a radiomic model is clinically valid. It helps quantify whether the current dataset and modelling strategy are methodologically plausible.

External validation, transparent reporting, calibration assessment, endpoint robustness, and clinical interpretability remain essential before any translational claim.

---

## Roadmap

- Interactive configuration for prevalence and shrinkage guardrails.
- Calibration plots.
- Decision-curve analysis.
- External validation upload.
- Patient-level cluster bootstrap.
- Exportable HTML/PDF methodological report.
- GitHub Actions smoke test for Python syntax and dashboard availability.

---

## License

Add your preferred license here, for example MIT, Apache-2.0, or an institutional research-use license.

---

## Acknowledgement

Built for radiomics feasibility analysis, small-sample model development, and transparent study-design discussion.
