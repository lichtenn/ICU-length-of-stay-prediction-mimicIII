# ICU Length of Stay Prediction — MIMIC-III

This project was part of the Machine Learning Class at FCUP Porto.

Accurate prediction of ICU length of stay (LOS) is critical for 
resource allocation, staff planning, and improving patient outcomes. 
This project develops machine learning models to estimate LOS for 
heart failure patients (ICD-9: 4280) using the MIMIC-III database — 
a large, de-identified repository of real-world ICU clinical data.

A central challenge of this work is the transformation of raw, 
high-frequency time-series data into structured tabular features 
suitable for machine learning, while retaining clinically meaningful 
signal. Beyond predictive accuracy, a key objective is interpretability: 
understanding which clinical factors drive LOS predictions in a way 
that could inform ICU practice.

## Project pipeline

### 1 — Data preprocessing
- Filtering to Metavision-based ICU stays with ICD-9 diagnosis 4280
- Integration of event tables: chart events, input events, procedure 
  events
- Exclusion of tables without direct ICU stay linkage (lab events, 
  microbiology events)

### 2 — Individual patient visualisation
- Time series of vital signs with high-risk period annotation
- Medication administration timeline
- Procedure timeline
- Provides qualitative insight into a typical complex ICU stay before 
  moving to population-level analysis

### 3 — Statistical analysis
- Dataset overview: shape, data types, missing value assessment
- Descriptive statistics and distribution analysis of numeric variables
- Frequency analysis of categorical variables
- Deep dive into key variables including age distribution and LOS 
  distribution (right-skewed, mean 4.2 days)

### 4 — Feature engineering
- Demographic features used directly (age, gender, insurance type, etc.)
- Time-series event data aggregated on a daily basis (mean, sum per 
  feature per day)
- Features selected based on frequency of occurrence and correlation 
  with LOS
- Resulting feature set includes variables such as `heart_rate_1` 
  (average heart rate on day 1)

### 5 — Modelling & evaluation
Four models trained and benchmarked against a median baseline:

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| XGBoost | 1.24 | 2.59 | 0.71 |
| Random Forest | 1.27 | 2.65 | 0.69 |
| Neural Network (MLP) | 1.98 | 3.49 | 0.47 |
| Ridge Regression | 1.97 | 3.49 | 0.47 |
| Baseline (median) | 2.68 | 5.05 | -0.11 |

Validation via cross-validation with grid search. Missing value 
strategies include MICE imputation (Random Forest), KNN imputation 
(MLP, XGBoost), and Iterative imputation (Ridge).

### 6 — Interpretability
- Global feature importance via SHAP for both XGBoost and Random Forest
- Local SHAP explanations for individual predictions
- Key predictors identified: invasive ventilation (day 0), NaCl 0.9% 
  administration, arterial line placement, heart rate trajectories

## Data access

MIMIC-III requires credentialed access through PhysioNet. Raw data 
is not included in this repository.

→ [Apply for access](https://mimic.mit.edu/docs/gettingstarted/)

## Repository structure

| Folder / File | Contents |
|---------------|----------|
| `Notebook.ipynb` | Notebook including explanations and analysis (ipynb) |
| `Notebook.pdf` | Notebook including explanations and analysis (pdf) |
