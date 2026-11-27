# Prediction Engine — Clinical Trial Outcome & Adverse Event Prediction

Machine-learning models that predict **tumour response** and **adverse events (AEs / SAEs)** for patients in oncology clinical trials. The work covers four metastatic colorectal-cancer trial datasets — referred to by their dataset IDs **463, 1053, 1183 and 1186** — first one at a time and then pooled into combined cross-trial models.

The datasets are de-identified, patient-level clinical trial data (CDISC-style domains such as `adsl`, `adae`, `adlb`, `adls`, `adrsp`, plus raw `demo`, `lesions`, `exposure`, `chemotx`, `vitals`, etc.). Study 1053 corresponds to **NCT00115765** (panitumumab + bevacizumab with chemotherapy); study 1186 evaluates panitumumab + FOLFIRI.

---

## What is predicted

### 1. Objective response (RECIST-style)

| Target | Definition used |
|---|---|
| **New lesions** | Whether a patient develops any new lesion (`LSNEWANY`) |
| **Target-lesion change (LSSLD)** | Change in the sum of the longest diameters of target lesions between first and last non-zero measurement |
| **Partial Response (PR)** | ≥ 30 % decrease in target-lesion SLD (≥ 50 % for study 1053) with no progression of non-target lesions and no new lesions |
| **Complete Response (CR)** | Disappearance of all target and non-target lesions and no new lesions |
| **Non-target lesions** | Response of non-target lesions (study 1186 non-target SLD is always 0, so it is excluded there) |

### 2. Adverse events

| Target | Approach |
|---|---|
| **Segregated AEs** | Per-patient count or occurrence of AEs in each MedDRA **System Organ Class** (`AESOC_1 … AESOC_n`), as multi-output classification and regression |
| **Total AEs / SAEs per month** | Month-by-month forecasting of total AEs and serious AEs. Two variants for 1186: one feeds the **actual** previous-month count as a feature, the other feeds the **predicted** previous-month count (a realistic rolling forecast) |

## Modelling workflow

1. **EDA & dataset building** — merge 15–20 domain files per study on `SUBJID`, clean, impute, then correlation heatmaps, PCA, and Pearson / Spearman / Kendall, chi-square, t-test and ANOVA tests.
2. **Feature selection** — Random Forest feature importance, then SHAP / eli5 on the final models.
3. **Models compared** — Logistic & Linear Regression, Elastic Net, Decision Tree, Random Forest, Gradient Boosting, **XGBoost**, SVM/SVR, KNN, MLP; `MultiOutputClassifier` / `MultiOutputRegressor` for per-organ-class AE targets.
4. **Tuning & overfitting control** — GridSearchCV / RandomizedSearchCV, cross-validation, Elastic Net regularisation, SMOTE for class imbalance.
5. **Evaluation** — accuracy, classification report, confusion matrix, ROC / AUC for classifiers; MSE, R² and actual-vs-predicted plots for regressors.
6. **Combined models** — pool 1053, 1183 and 1186. Columns missing from one study are **synthetically generated** from the others using shared features (subject, age, sex, `B_METANM`, lesion site) so a single model can be trained across trials.

## Repository structure

```
.
├── 463/                                   # Study 463
│   ├── 463_RTL.ipynb                      # Target-lesion response
│   ├── 463_RNTL.ipynb                     # Non-target-lesion response
│   ├── 463_New_Lesions_Regression.ipynb
│   └── 463_Segregated_AE.ipynb            # AE by organ class (XGBoost, multi-output)
├── 1053/Objective Response/               # Study 1053 (NCT00115765)
│   ├── 1053_EDA.ipynb
│   ├── 1053_LSSLD_Target_Lesion_Regression.ipynb
│   ├── 1053_LSSLD_Non_Target_Lesion_Regression.ipynb
│   └── 1053_New_Lesions_Regression.ipynb
├── 1183/
│   ├── Objective Response/                # EDA_1183, 1183_Regression_Final
│   ├── Segregated AE prediction/          # EDA_AEs_SAEs, AE_Regression_Model
│   └── Total AE prediction/               # Total_AE_prediction_1183
├── 1186/
│   ├── Objective Response/                # EDA, LSSLD regression, new-lesion regression
│   ├── Segregated AE prediction/          # SegAE classification & regression
│   ├── total AE and SAE prediction/       # Monthly total AE/SAE forecasting (+ requirements_k.txt)
│   └── AE_SAE_SEGREGATION_1186 (5) (1).ipynb
├── Combined_model/                        # Cross-trial models (1053 + 1183 + 1186)
│   ├── TL_PR Combined 1186,1053 and 1183.ipynb
│   ├── TL_CR Combined 1186,1053 and 1183.ipynb
│   ├── non_target_PR_CR_1053_1186_1183.ipynb
│   ├── New_Lesions_Combined_1053_1186_1183.ipynb
│   ├── New Lesions Combined 1183 and 1186.ipynb
│   ├── Total_AE_SAE_Prediction_combined_for_1186_1183.ipynb
│   ├── 1053_nontarget.ipynb
│   └── *.csv                              # Processed training sets (TL PR/CR, 1186 non-target)
├── 1053_segregated_AE.ipynb               # Study 1053 AE by organ class
├── processed_dataframe_segAE_1053.csv
└── AE_SAE_SEGREGATION_1186 (5) (1).ipynb  # Copy of the 1186 notebook
```

## Data

Raw trial data is **not** included. The notebooks read the original CSVs from local paths (e.g. `C:\Users\...\Kolate.AI\<study>\Data\...`); update these paths to wherever you store the data. Links to the processed inputs used by some notebooks are in the folder-level readmes:

- `1053/Objective Response/README.md`
- `1186/Objective Response/readme.md`
- `1186/total AE and SAE prediction/readme.md`

A few small processed datasets are included (`processed_dataframe_segAE_1053.csv`, `Combined_model/*.csv`) so the combined models can be explored directly.

## Getting started

```bash
git clone <this-repo-url>
cd prediction_engine

pip install pandas numpy scikit-learn xgboost imbalanced-learn shap eli5 \
            matplotlib seaborn plotly scipy jupyter
jupyter notebook
```

`1186/total AE and SAE prediction/requirements_k.txt` contains the full pinned environment used for that part of the project (it includes many unrelated packages; the list above is enough for the notebooks).

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · imbalanced-learn (SMOTE) · SHAP · eli5 · SciPy · Matplotlib · Seaborn · Plotly · Jupyter

## Disclaimer

This is a research / educational project. The models are not validated for clinical use and must not be used to make treatment decisions.
