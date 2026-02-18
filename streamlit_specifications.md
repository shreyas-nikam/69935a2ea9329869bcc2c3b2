
# Streamlit Application Specification: SR 11-7 Model Stress Testing

## 1. Application Overview

**Purpose:**
This Streamlit application serves as a comprehensive tool for a Model Risk Manager (CFA Charterholder) to stress test a credit default classification model, ensuring compliance with Federal Reserve SR 11-7 guidance. The application aims to systematically identify and quantify model vulnerabilities under various adverse conditions, such as economic regime shifts, extreme input values, feature perturbations, adversarial attempts, and concept drift, culminating in a structured stress test report.

**High-level Story Flow:**

1.  **Introduction & Setup:** The application begins by introducing the critical role of model risk management and SR 11-7 compliance. The user (CFA Charterholder) then initiates the simulation of a credit default dataset and loads a pre-trained XGBoost credit default model, setting up the environment for stress testing.
2.  **Distribution Shift Testing:** The user evaluates how the model's performance (AUC) degrades when applied to data from different economic regimes (e.g., COVID crisis, high-interest rates) compared to its training regime (expansion). The results are visualized in a degradation heatmap, highlighting regime-specific vulnerabilities.
3.  **Extreme Value Boundary Mapping:** The user selects key features (e.g., FICO, DTI) and observes how the model's predicted default probability changes as these features are swept across and beyond their typical training ranges. This identifies "cliffs" and assesses extrapolation stability.
4.  **Feature Sensitivity Analysis:** The user quantifies the impact of small (e.g., 1%) perturbations in input features on the model's predictions. A heatmap ranks features by their influence, identifying potentially fragile inputs.
5.  **Adversarial Robustness Testing:** For "borderline" credit applications, the user determines the minimum change required for a single feature to flip the model's prediction (e.g., from 'default' to 'no default'). This reveals how "gameable" the model is to strategic applicants.
6.  **Concept Drift Detection:** The application simulates a rolling performance monitor (AUC) over time, with configurable window sizes and alert thresholds, to detect when the model's performance degrades. It also includes Population Stability Index (PSI) monitoring for input data shifts.
7.  **SR 11-7 Stress Test Report:** Finally, all findings from the preceding stress tests are aggregated into a professional, structured SR 11-7 compliant report, detailing overall assessment, severity ratings, recommended use boundaries, and sign-off sections. This report serves as the official documentation for regulatory compliance and internal risk management.

## 2. Code Requirements

### Imports

```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import joblib

# Import all functions from source.py
from source import *

# Suppress warnings from source.py if any are not handled internally
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
```

### Streamlit Application Structure and `st.session_state`

The application will use a sidebar selectbox for navigation, simulating a multi-page experience within a single `app.py` file. `st.session_state` will be extensively used to store results and user selections across these "pages".

**`st.session_state` Initialization (at the start of `app.py`):**

```python
# Initialize session state variables
if 'df_credit_data' not in st.session_state:
    st.session_state.df_credit_data = None
if 'loaded_model' not in st.session_state:
    st.session_state.loaded_model = None
if 'X_train_base' not in st.session_state:
    st.session_state.X_train_base = None
if 'y_train_base' not in st.session_state:
    st.session_state.y_train_base = None
if 'model_features' not in st.session_state:
    st.session_state.model_features = None

if 'distribution_results' not in st.session_state:
    st.session_state.distribution_results = None

if 'boundary_results' not in st.session_state:
    st.session_state.boundary_results = None
if 'selected_boundary_features' not in st.session_state:
    st.session_state.selected_boundary_features = []

if 'sensitivity_results' not in st.session_state:
    st.session_state.sensitivity_results = None
if 'perturbation_pct' not in st.session_state:
    st.session_state.perturbation_pct = 0.01

if 'adversarial_results_df' not in st.session_state:
    st.session_state.adversarial_results_df = None
if 'adv_prob_threshold' not in st.session_state:
    st.session_state.adv_prob_threshold = 0.5
if 'adv_borderline_upper' not in st.session_state:
    st.session_state.adv_borderline_upper = 0.65

if 'drift_results_df' not in st.session_state:
    st.session_state.drift_results_df = None
if 'baseline_auc' not in st.session_state:
    st.session_state.baseline_auc = None
if 'cd_window_size' not in st.session_state:
    st.session_state.cd_window_size = 90
if 'cd_step_size' not in st.session_state:
    st.session_state.cd_step_size = 30
if 'cd_yellow_thresh' not in st.session_state:
    st.session_state.cd_yellow_thresh = 0.03
if 'cd_red_thresh' not in st.session_state:
    st.session_state.cd_red_thresh = 0.07
if 'cd_psi_thresh' not in st.session_state:
    st.session_state.cd_psi_thresh = 0.25

if 'sr117_report' not in st.session_state:
    st.session_state.sr117_report = None
```

**Streamlit Sidebar Navigation:**

```python
st.sidebar.title("SR 11-7 Model Stress Testing")
page_selection = st.sidebar.selectbox(
    "Go to",
    [
        "1. Introduction & Setup",
        "2. Distribution Shift Testing",
        "3. Extreme Value Boundary Mapping",
        "4. Feature Sensitivity Analysis",
        "5. Adversarial Robustness Testing",
        "6. Concept Drift Detection",
        "7. SR 11-7 Stress Test Report",
    ],
    index=0
)
```

---

### Page 1: Introduction & Setup

**Markdown Content:**

```python
st.title("SR 11-7 Model Stress Testing for Credit Default")
st.markdown(f"**Persona:** CFA Charterholder & Model Risk Manager")

st.markdown(f"## Introduction: Probing Model Vulnerabilities for Regulatory Compliance")
st.markdown(f"As a Model Risk Manager or Quantitative Analyst at a leading financial institution, your core responsibility transcends merely building accurate models; it's about understanding their limitations and ensuring they meet stringent regulatory standards. The Federal Reserve's SR 11-7 guidance mandates a thorough understanding of model boundaries and failure conditions, especially for critical applications like credit default prediction. A model might show impressive accuracy on historical data, but the real test comes under stress—during economic shifts, extreme borrower profiles, or even attempts at strategic manipulation.")
st.markdown(f"This application takes on the persona of a CFA Charterholder, tasked with performing a comprehensive suite of stress tests on our deployed credit default classification model. Our objective is not to build a better model, but to systematically break it, identify its vulnerabilities, and quantify its risks. This process is crucial for preventing costly errors, maintaining regulatory compliance, and making sound capital allocation and lending decisions, ultimately safeguarding the institution against unforeseen model failures.")
st.markdown(f"We will execute a series of five critical stress tests:")
st.markdown(f"1.  **Distribution Shift Testing:** How does the model perform when economic conditions change dramatically?")
st.markdown(f"2.  **Extreme Value Boundary Mapping:** What happens when input features move far outside their typical ranges?")
st.markdown(f"3.  **Feature Sensitivity Analysis:** Which features drive the model's predictions the most, and are they fragile?")
st.markdown(f"4.  **Adversarial Robustness Testing:** Can borrowers strategically \"game\" the model to flip a credit decision?")
st.markdown(f"5.  **Concept Drift Detection:** How quickly does the model's performance degrade over time as underlying relationships evolve?")
st.markdown(f"Finally, we will compile our findings into a structured SR 11-7 Stress Test Report, providing a clear \"prospectus\" of the model's strengths, weaknesses, and recommended usage boundaries.")

st.markdown(f"## Setup: Simulating Data and Loading Pre-trained Model")
st.markdown(f"As a Model Risk Manager, the first step is to establish a realistic testing environment. This involves generating synthetic credit default data that mimics various economic regimes and loading the pre-trained credit default model that we need to stress test. This setup ensures our tests are conducted on a representative dataset and the actual model in question.")
st.markdown(f"For robust stress testing, we need data that reflects different economic conditions. Our existing credit default model (`xgboost_credit.pkl`) was likely trained on data from an expansionary period. To comply with SR 11-7, we must understand how it performs under adverse scenarios, such as recessions or periods of high interest rates. We will generate synthetic data with a `date` column to simulate these economic regimes and then load the pre-trained model.")
```

**UI Interactions and Function Invocation:**

```python
if st.button("Generate Data & Load Model"):
    with st.spinner("Generating synthetic data and training/loading model... This might take a moment."):
        # Generate synthetic credit data
        df_credit_data = generate_synthetic_credit_data(n_samples=15000)
        st.session_state.df_credit_data = df_credit_data

        # Train and save the model
        model, X_train_base, y_train_base = train_and_save_model(df_credit_data)
        st.session_state.X_train_base = X_train_base
        st.session_state.y_train_base = y_train_base
        
        # Load the model back to simulate using a pre-trained model
        loaded_model = joblib.load('xgboost_credit.pkl')
        st.session_state.loaded_model = loaded_model
        st.session_state.model_features = X_train_base.columns.tolist()

    st.success("Data generated and model loaded successfully!")
    st.markdown(f"**Loaded model type:** `{type(st.session_state.loaded_model).__name__}`")
    st.markdown(f"**Model baseline AUC on its training data:** `{roc_auc_score(st.session_state.y_train_base, st.session_state.loaded_model.predict_proba(st.session_state.X_train_base)[:, 1]):.4f}`")
    st.markdown(f"**Features used by model:** `{', '.join(st.session_state.model_features)}`")

st.markdown(f"### Explanation of Execution")
st.markdown(f"We have successfully simulated a comprehensive credit default dataset and established our `xgboost_credit.pkl` model. The `generate_synthetic_credit_data` function created a rich dataset with various financial features and a `default` target, crucially including a `date` column that allows us to simulate different economic regimes. The `train_and_save_model` function then trained an XGBoost classifier on an \"expansionary\" period (2015-2019) from this synthetic data and saved it. Finally, loading the model confirms it's ready for our stress tests. This setup provides a realistic foundation for applying our SR 11-7 stress testing methodologies, ensuring that the persona is working with data and a model representative of a real-world financial institution.")
```

---

### Page 2: Distribution Shift Testing

**Conditional Rendering Check:**

```python
if st.session_state.df_credit_data is None or st.session_state.loaded_model is None:
    st.warning("Please go to '1. Introduction & Setup' and generate data/load model first.")
else:
    # ... Page content ...
```

**Markdown Content:**

```python
st.title("2. Distribution Shift Testing")
st.markdown(f"## Story + Context + Real-World Relevance")
st.markdown(f"As a Model Risk Manager, one of the most critical aspects of SR 11-7 compliance is understanding how our credit model's performance degrades when the underlying economic environment changes. Our model was built in a specific regime, and it might fail unpredictably under new conditions. This test simulates exactly that: evaluating the model trained in one economic regime (e.g., expansion) on data from different regimes (e.g., COVID crisis, high interest rates). We need to quantify this degradation using AUC to identify if the model's reliability is compromised.")
st.markdown(f"The degradation is measured as the difference between the model's performance in its training regime and its performance in a new, distinct regime. A significant drop indicates the model is not robust to changing economic conditions.")

st.markdown(r"**Mathematical Formulation:**")
st.markdown(r"The Distribution Shift Degradation ($\Delta AUC$) is calculated as:")
st.markdown(r"$$\Delta AUC = AUC_{{\text{in-regime}}} - AUC_{{\text{cross-regime}}}$$")
st.markdown(r"where $AUC_{{\text{in-regime}}}$ is the Area Under the Receiver Operating Characteristic Curve when the model is evaluated on data from the same economic regime it was trained on (or a similar, stable regime).")
st.markdown(r"where $AUC_{{\text{cross-regime}}}$ is the AUC when the model is evaluated on data from a different, challenging economic regime (e.g., recession, crisis).")

st.markdown(r"**Alert Thresholds for $\Delta AUC$:**")
st.markdown(r"-   **Acceptable (Green):** $\Delta AUC < 0.05$. Model generalizes well across regimes.")
st.markdown(r"-   **Concerning (Yellow):** $0.05 \le \Delta AUC < 0.10$. Model shows moderate regime sensitivity. Document as a limitation.")
st.markdown(r"-   **Failure (Red):** $\Delta AUC \ge 0.10$. Model is regime-dependent. Not suitable for deployment without regime-conditioning or retraining triggers.")
```

**UI Interactions and Function Invocation:**

```python
if st.button("Run Distribution Shift Test"):
    with st.spinner("Performing distribution shift analysis..."):
        regime_splits = create_regime_splits(st.session_state.df_credit_data, date_col='date')
        distribution_results = distribution_shift_test(
            st.session_state.loaded_model,
            st.session_state.df_credit_data[st.session_state.model_features],
            st.session_state.df_credit_data['default'],
            regime_splits,
            st.session_state.model_features
        )
        st.session_state.distribution_results = distribution_results

    st.success("Distribution Shift Test Complete!")
    st.subheader("Results Table:")
    st.dataframe(st.session_state.distribution_results[['test_regime', 'auc', 'auc_degradation', 'train_default_rate', 'test_default_rate']])

    st.subheader("AUC Performance Across Economic Regimes")
    plt.figure(figsize=(10, 6))
    pivot_table_auc = st.session_state.distribution_results.pivot_table(index='model_train_regime', columns='test_regime', values='auc')
    sns.heatmap(pivot_table_auc, annot=True, fmt=".3f", cmap="viridis", linewidths=.5)
    plt.title('AUC Performance Across Economic Regimes (Model trained on Expansion)')
    plt.xlabel('Test Regime')
    plt.ylabel('Model Training Regime')
    st.pyplot(plt)
    plt.close() # Close figure to prevent displaying multiple times

    st.subheader("AUC Degradation vs. Expansion Regime")
    plt.figure(figsize=(10, 6))
    degradation_data = st.session_state.distribution_results[st.session_state.distribution_results['model_train_regime'] == 'expansion_2015_2019'][['test_regime', 'auc_degradation']]
    degradation_pivot = degradation_data.set_index('test_regime').T
    sns.heatmap(degradation_pivot, annot=True, fmt=".3f", cmap="Reds", linewidths=.5, cbar_kws={'label': 'AUC Degradation'})
    plt.title('AUC Degradation vs. Expansion Regime (Model trained on Expansion)')
    plt.xlabel('Test Regime')
    plt.ylabel('Degradation Source (relative to Expansion train AUC)')
    st.pyplot(plt)
    plt.close()

    st.subheader("Degradation Severity Assessment:")
    for _, row in st.session_state.distribution_results.iterrows():
        if row['test_regime'] == row['model_train_regime']:
            continue
        degradation = row['auc_degradation']
        severity = ""
        if degradation < 0.05:
            severity = "Acceptable (Green)"
        elif 0.05 <= degradation < 0.10:
            severity = "Concerning (Yellow)"
        else:
            severity = "Failure (Red)"
        st.markdown(f"- **Test Regime:** `{row['test_regime']}` | **AUC Degradation:** `{degradation:.3f}` | **Severity:** `{severity}`")

st.markdown(f"### Explanation of Execution")
st.markdown(f"The distribution shift test clearly reveals how our credit default model's performance fluctuates across different economic regimes. The `AUC Performance Across Economic Regimes` heatmap shows the model's AUC when evaluated against various periods, while the `AUC Degradation` heatmap specifically highlights the drop in performance relative to its baseline (trained on 'expansion_2015_2019').")
st.markdown(f"For the Model Risk Manager, these results are critical:")
st.markdown(f"-   A **low AUC degradation** for a test regime indicates the model generalizes well and is robust to those specific economic conditions.")
st.markdown(f"-   A **high AUC degradation**, especially above the `0.10` threshold, signals a severe failure. For example, if the AUC on 'covid_2020' data drops significantly from the 'expansion_2015_2019' baseline, it means the model's learned relationships break down during a crisis. This implies that the model might be unreliable in predicting defaults during similar future crises, leading to misinformed lending decisions and potential capital losses for the institution.")
st.markdown(f"-   The default rates for training vs. test regimes provide additional context, showing if the model is tested on a period with a significantly different underlying default propensity.")
st.markdown(f"This insight directly informs regulatory reporting by identifying specific economic conditions under which the model's predictions become unreliable, requiring either recalibration, conditional usage, or complete retraining. This addresses the SR 11-7 requirement to understand model boundaries.")
```

---

### Page 3: Extreme Value Boundary Mapping

**Conditional Rendering Check:**

```python
if st.session_state.X_train_base is None or st.session_state.loaded_model is None:
    st.warning("Please go to '1. Introduction & Setup' and generate data/load model first.")
else:
    # ... Page content ...
```

**Markdown Content:**

```python
st.title("3. Extreme Value Boundary Mapping")
st.markdown(f"## Story + Context + Real-World Relevance")
st.markdown(f"Regulators demand to know how our model behaves when confronted with input values far outside its training distribution. This is especially true for features like `DTI` (Debt-to-Income) or `FICO score`, where extreme values, though rare, can significantly impact default probability. As a Model Risk Manager, I need to identify \"cliffs\"—sudden, large changes in prediction—and assess the stability of model extrapolation. If the model behaves erratically beyond its training data, it poses a significant risk in real-world applications where such extreme cases might occur, even if infrequently.")
st.markdown(f"This test involves systematically sweeping key feature values across their theoretical ranges, extending beyond what the model was trained on, while holding other features constant at their median. We then observe the predicted probability of default.")
```

**UI Interactions and Function Invocation:**

```python
features_for_boundary = ['fico_score', 'dti', 'income', 'ltv', 'revolving_utilization', 'delinquencies_2yr']
selected_features = st.multiselect(
    "Select features to sweep for boundary mapping:",
    options=[f for f in features_for_boundary if f in st.session_state.model_features],
    default=[f for f in ['fico_score', 'dti'] if f in st.session_state.model_features],
    key='boundary_feature_select'
)
st.session_state.selected_boundary_features = selected_features

if st.button("Run Boundary Mapping", disabled=not bool(st.session_state.selected_boundary_features)):
    with st.spinner("Performing boundary mapping..."):
        boundary_results = boundary_mapping(st.session_state.loaded_model, st.session_state.X_train_base, st.session_state.selected_boundary_features)
        st.session_state.boundary_results = boundary_results
    st.success("Boundary Mapping Complete!")

if st.session_state.boundary_results:
    st.subheader("Extreme Value Boundary Maps")
    num_features = len(st.session_state.selected_boundary_features)
    if num_features > 0:
        rows = int(np.ceil(num_features / 3))
        cols = min(num_features, 3)
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten() if rows > 1 or cols > 1 else [axes] # Ensure axes is iterable

        for i, feature in enumerate(st.session_state.selected_boundary_features):
            ax = axes[i]
            res = st.session_state.boundary_results[feature]
            ax.plot(res['sweep_values'], res['predictions'], label='P(Default)')
            
            # Shade training range
            ax.axvspan(res['training_q01'], res['training_q99'], color='gray', alpha=0.2, label='Training Range (1-99% quantile)')

            if not np.isnan(res['cliff_value']):
                ax.axvline(res['cliff_value'], color='red', linestyle='--', label=f'Cliff at {res["cliff_value"]:.2f}')
            
            if not res['extrapolation_stable']:
                ax.text(0.05, 0.95, 'UNSTABLE Extrapolation', color='red', transform=ax.transAxes, fontsize=10, verticalalignment='top')

            ax.set_title(f'P(Default) vs. {feature}')
            ax.set_xlabel(feature)
            ax.set_ylabel('Predicted P(Default)')
            ax.legend(fontsize=8)
            ax.grid(True, linestyle=':', alpha=0.6)
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Boundary Mapping Summary:")
    for feature, res in st.session_state.boundary_results.items():
        st.markdown(f"- **{feature}:** Cliff at `{res['cliff_value']:.2f}`, Extrapolation: `{ 'stable' if res['extrapolation_stable'] else 'UNSTABLE'}`")
```

**Markdown Content (Explanation):**

```python
st.markdown(f"### Explanation of Execution")
st.markdown(f"The extreme value boundary maps graphically illustrate how the model's predicted probability of default responds to variations in key input features, particularly beyond their typical training ranges.")
st.markdown(f"For the Model Risk Manager, these plots are invaluable:")
st.markdown(f"-   **Cliff Detection:** A sudden, sharp change (a \"cliff\") in the predicted probability, indicated by the red dashed line, suggests model instability. If our `dti` model, for instance, shows a cliff at `DTI = 0.7`, it means a small increase beyond this point can drastically change the prediction. This implies the model is very sensitive and potentially brittle around this threshold.")
st.markdown(f"-   **Extrapolation Stability:** The shaded gray region represents the typical training data range (1st to 99th percentile). How the model behaves *outside* this range is crucial. If the curve becomes erratic, flatlines unexpectedly, or shows sudden jumps (flagged as 'UNSTABLE Extrapolation'), it means the model is extrapolating unpredictably. This is a severe risk because in real-world scenarios, a borrower with an extreme (but valid) `fico_score` or `ltv` might receive an inaccurate, confidence-inducing \"no default\" prediction, or a wrongly harsh \"default\" prediction.")
st.markdown(f"Understanding these boundaries enables us to set appropriate use limits for the model (e.g., \"model validated only for FICO scores between 500-850, and DTI between 0-60%\"). This directly fulfills SR 11-7's requirement to document model limitations and potential failure modes, ensuring users are aware of when to exercise caution or seek additional review.")
```

---

### Page 4: Feature Sensitivity Analysis

**Conditional Rendering Check:**

```python
if st.session_state.X_train_base is None or st.session_state.loaded_model is None:
    st.warning("Please go to '1. Introduction & Setup' and generate data/load model first.")
else:
    # ... Page content ...
```

**Markdown Content:**

```python
st.title("4. Feature Sensitivity Analysis")
st.markdown(f"## Story + Context + Real-World Relevance")
st.markdown(f"Understanding which features significantly influence a model's output is critical for managing model risk. As a Model Risk Manager, this is akin to calculating \"duration and convexity\" for a bond portfolio; it tells me how sensitive the model's output is to small changes in each input. A model that is highly sensitive to a single feature might be fragile and vulnerable to data quality issues or minor shifts in that feature. This analysis helps us identify the most influential, and potentially fragile, features, informing where to focus our data monitoring efforts and internal challenge.")
st.markdown(f"This test quantifies how much the model's default prediction changes in response to small perturbations (e.g., a 1% increase) in each input feature, while holding other features constant at their median.")
```

**UI Interactions and Function Invocation:**

```python
perturbation_pct = st.slider(
    "Select Perturbation Percentage (e.g., 0.01 for 1% change):",
    min_value=0.001,
    max_value=0.1,
    value=st.session_state.perturbation_pct,
    step=0.001,
    format="%.3f"
)
st.session_state.perturbation_pct = perturbation_pct

if st.button("Run Sensitivity Analysis"):
    with st.spinner("Performing feature sensitivity analysis..."):
        X_sample_sensitivity = st.session_state.X_train_base.sample(n=min(1000, len(st.session_state.X_train_base)), random_state=42)
        sensitivity_results = feature_sensitivity(st.session_state.loaded_model, X_sample_sensitivity, st.session_state.model_features, perturbation_percent=perturbation_pct)
        st.session_state.sensitivity_results = sensitivity_results
    st.success("Feature Sensitivity Analysis Complete!")

if st.session_state.sensitivity_results is not None:
    st.subheader("Feature Sensitivity Analysis Results:")
    st.dataframe(st.session_state.sensitivity_results)

    st.subheader(f"Mean Absolute Change in P(Default) for {perturbation_pct*100:.1f}% Feature Perturbation")
    plt.figure(figsize=(10, 6))
    sns.barplot(x=st.session_state.sensitivity_results.index, y=st.session_state.sensitivity_results['mean_abs_change'], palette='viridis')
    plt.title(f'Mean Absolute Change in P(Default) for {perturbation_pct*100:.1f}% Feature Perturbation')
    plt.xlabel('Feature')
    plt.ylabel('Mean Absolute Change in P(Default)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

    st.subheader(f"Percentage of Samples with >0.01 P(Default) Change for {perturbation_pct*100:.1f}% Feature Perturbation")
    plt.figure(figsize=(10, 6))
    sns.barplot(x=st.session_state.sensitivity_results.index, y=st.session_state.sensitivity_results['pct_affected_gt_0.01'], palette='cividis')
    plt.title(f'Percentage of Samples with >0.01 P(Default) Change for {perturbation_pct*100:.1f}% Feature Perturbation')
    plt.xlabel('Feature')
    plt.ylabel('Percentage of Samples Affected')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()
```

**Markdown Content (Explanation):**

```python
st.markdown(f"### Explanation of Execution")
st.markdown(f"The feature sensitivity analysis provides a quantitative \"risk profile\" for our credit default model. The bar charts, showing both the mean absolute change in predicted default probability and the percentage of samples affected, immediately highlight which features exert the most influence on the model's predictions.")
st.markdown(f"For the Model Risk Manager, these insights are crucial:")
st.markdown(f"-   **Identifying Fragile Features:** Features with a high `mean_abs_change` or `pct_affected_gt_0.01` are the most influential and potentially fragile. For example, if `fico_score` shows a large mean change, it means even a small {perturbation_pct*100:.1f}% shift in FICO can significantly alter the default probability. This makes the model highly dependent on the accuracy and stability of that particular input.")
st.markdown(f"-   **Data Quality Focus:** Knowledge of these sensitive features directs our attention to data quality monitoring. If `dti` is highly sensitive, we must ensure the `DTI` data is accurate and consistently measured, as errors could lead to substantial prediction changes and incorrect lending decisions.")
st.markdown(f"-   **Model Complexity vs. Risk:** This analysis helps in understanding the trade-off. While complex models might achieve higher accuracy, if that accuracy comes with extreme sensitivity to a few features, it introduces fragility. This understanding is essential for SR 11-7, enabling us to document the model's vulnerability to input uncertainty and recommend appropriate safeguards, such as enhanced data validation for critical inputs.")
```

---

### Page 5: Adversarial Robustness Testing

**Conditional Rendering Check:**

```python
if st.session_state.df_credit_data is None or st.session_state.loaded_model is None:
    st.warning("Please go to '1. Introduction & Setup' and generate data/load model first.")
else:
    # ... Page content ...
```

**Markdown Content:**

```python
st.title("5. Adversarial Robustness Testing")
st.markdown(f"## Story + Context + Real-World Relevance")
st.markdown(f"In credit markets, borrowers are incentivized to present the best possible financial profile to secure a loan. As a Model Risk Manager, I need to know if our credit model can be \"gamed\" by strategic applicants who make minimal changes to their data to flip a borderline decision (e.g., from 'default' to 'no default'). This adversarial robustness test helps quantify how vulnerable our model is to such manipulations, particularly for applications near the decision boundary. Understanding this vulnerability is key to preventing strategic abuse and ensuring fair, robust lending decisions.")
st.markdown(f"This test involves iterating through borderline credit applications and for each, identifying the minimum percentage change to a single input feature that would flip the model's binary prediction.")
```

**UI Interactions and Function Invocation:**

```python
adv_prob_threshold = st.slider(
    "Decision Probability Threshold (e.g., 0.5):",
    min_value=0.1, max_value=0.9, value=st.session_state.adv_prob_threshold, step=0.05
)
st.session_state.adv_prob_threshold = adv_prob_threshold

adv_borderline_upper = st.slider(
    "Borderline Probability Range (Upper Bound) - for selecting samples predicted as default but not too confident:",
    min_value=adv_prob_threshold, max_value=0.99, value=st.session_state.adv_borderline_upper, step=0.01
)
st.session_state.adv_borderline_upper = adv_borderline_upper

if st.button("Run Adversarial Test"):
    with st.spinner("Performing adversarial robustness test..."):
        X_adv_sample = st.session_state.df_credit_data[st.session_state.model_features].sample(n=min(2000, len(st.session_state.df_credit_data)), random_state=42)
        y_adv_sample = st.session_state.df_credit_data['default'].loc[X_adv_sample.index]

        adversarial_results_df = adversarial_test(
            st.session_state.loaded_model,
            X_adv_sample,
            y_adv_sample,
            st.session_state.model_features,
            prob_threshold=adv_prob_threshold,
            borderline_range=(adv_prob_threshold, adv_borderline_upper)
        )
        st.session_state.adversarial_results_df = adversarial_results_df
    st.success("Adversarial Robustness Test Complete!")

if st.session_state.adversarial_results_df is not None:
    st.subheader("Adversarial Vulnerability Ranking:")
    vuln_features_plot = st.session_state.adversarial_results_df.groupby('feature')['delta_pct'].mean().sort_values(ascending=True)
    st.markdown(f"(Lower delta_pct = easier to game)")
    st.dataframe(vuln_features_plot)

    st.subheader("Features Ranked by Minimum % Perturbation to Flip Prediction (Ease of Gaming)")
    plt.figure(figsize=(10, 6))
    sns.barplot(x=vuln_features_plot.index, y=vuln_features_plot.values, palette='coolwarm')
    plt.title('Features Ranked by Minimum % Perturbation to Flip Prediction (Ease of Gaming)')
    plt.xlabel('Feature')
    plt.ylabel('Mean Minimum % Change to Flip')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()
```

**Markdown Content (Explanation):**

```python
st.markdown(f"### Explanation of Execution")
st.markdown(f"The adversarial robustness test identifies how easily a model's prediction for a \"borderline\" credit application can be manipulated by small, strategic changes to input features. The bar chart vividly illustrates which features require the minimum percentage change to flip a 'default' prediction to a 'no default' prediction, essentially ranking them by their \"gameability.\"")
st.markdown(f"For the Model Risk Manager, this is a direct measure of model fragility against strategic behavior:")
st.markdown(f"-   **Identifying Gameable Features:** If features like `revolving_utilization` or `dti` consistently require only a small percentage change (e.g., 5-10%) to flip a decision, it means borrowers could strategically pay down debt or adjust income statements to influence the model. This is not fraud but rational optimization, yet it undermines the model's reliability for these individuals.")
st.markdown(f"-   **Risk Mitigation Strategies:** Knowing these gameable features allows the institution to implement countermeasures. For example, for applications where a sensitive feature is close to a critical threshold, additional verification (e.g., requiring bank statements for `dti`, cross-referencing `income` sources) might be warranted.")
st.markdown(f"-   **SR 11-7 Compliance:** This test directly addresses SR 11-7's call for understanding model vulnerability to strategic manipulation. It helps quantify the risk of lending decisions being swayed by minor, intentional data adjustments, ensuring that the model's 'prospectus' includes a clear warning about its susceptibility to gaming and recommends appropriate safeguards.")
```

---

### Page 6: Concept Drift Detection

**Conditional Rendering Check:**

```python
if st.session_state.df_credit_data is None or st.session_state.loaded_model is None:
    st.warning("Please go to '1. Introduction & Setup' and generate data/load model first.")
else:
    # ... Page content ...
```

**Markdown Content:**

```python
st.title("6. Concept Drift Detection")
st.markdown(f"## Story + Context + Real-World Relevance")
st.markdown(f"Models, particularly in dynamic financial environments, do not remain accurate indefinitely. The underlying relationships between credit features and default can change over time—a phenomenon known as concept drift. As a Model Risk Manager, I need a robust monitoring system to detect when our model's performance degrades in real-time. This proactive approach ensures we can trigger investigations, recalibrations, or even full retraining before significant financial risks accumulate. Implementing a rolling AUC monitor with clear alert thresholds is a key component of ongoing model risk management and SR 11-7 compliance.")
st.markdown(f"This test simulates rolling performance monitoring, calculating AUC over sequential time windows and comparing it to a baseline. We also introduce the Population Stability Index (PSI) to detect shifts in input distributions, which can often precede performance degradation.")

st.markdown(r"**Mathematical Formulation:**")
st.markdown(r"The degradation ($\Delta_t$) in rolling AUC performance is calculated as:")
st.markdown(r"$$\Delta_t = AUC_0 - AUC_t$$")
st.markdown(r"where $AUC_0$ is the baseline AUC on a validation set at deployment time (or a stable reference period).")
st.markdown(r"where $AUC_t$ is the rolling AUC calculated over a recent time window.")

st.markdown(r"**Alert Levels for $\Delta_t$:**")
st.markdown(r"-   **Green:** $\Delta_t < 0.03$. Normal variation.")
st.markdown(r"-   **Yellow:** $0.03 \le \Delta_t < 0.07$. Investigate. Check for data quality issues or genuine drift.")
st.markdown(r"-   **Red:** $\Delta_t \ge 0.07$. Freeze model for new decisions. Trigger revalidation and possible retraining.")

st.markdown(r"**Population Stability Index (PSI):**")
st.markdown(r"PSI can supplement AUC monitoring by detecting input distribution shifts even before output performance degrades.")
st.markdown(r"$$PSI = \sum_{{{\text{i=1}}}}^{{{\text{B}}}} \left(P_i^{{{\text{actual}}}} - P_i^{{{\text{expected}}}}\right) \ln\left(\frac{{P_i^{{{\text{actual}}}}}}{{P_i^{{{\text{expected}}}}}}\right)$$")
st.markdown(r"where $B$ is the number of bins partitioning the score distribution (or a specific feature's distribution).")
st.markdown(r"where $P_i^{{{\text{actual}}}}$ is the percentage of observations in bin $i$ for the actual (current) population.")
st.markdown(r"where $P_i^{{{\text{expected}}}}$ is the percentage of observations in bin $i$ for the expected (baseline) population.")
st.markdown(r"where $PSI > 0.25$ indicates significant population shift (a common alert threshold).")
```

**UI Interactions and Function Invocation:**

```python
cd_window_size = st.slider("Rolling Window Size (Days):", min_value=30, max_value=365, value=st.session_state.cd_window_size, step=30)
st.session_state.cd_window_size = cd_window_size

cd_step_size = st.slider("Window Step Size (Days):", min_value=7, max_value=90, value=st.session_state.cd_step_size, step=7)
st.session_state.cd_step_size = cd_step_size

cd_yellow_thresh = st.slider("Yellow Alert Threshold (Delta AUC):", min_value=0.01, max_value=0.1, value=st.session_state.cd_yellow_thresh, step=0.005, format="%.3f")
st.session_state.cd_yellow_thresh = cd_yellow_thresh

cd_red_thresh = st.slider("Red Alert Threshold (Delta AUC):", min_value=0.03, max_value=0.15, value=st.session_state.cd_red_thresh, step=0.005, format="%.3f")
st.session_state.cd_red_thresh = cd_red_thresh

cd_psi_thresh = st.slider("PSI Alert Threshold:", min_value=0.1, max_value=0.5, value=st.session_state.cd_psi_thresh, step=0.05, format="%.2f")
st.session_state.cd_psi_thresh = cd_psi_thresh


if st.button("Run Concept Drift Monitor"):
    with st.spinner("Performing concept drift detection..."):
        drift_results_df, baseline_auc = concept_drift_monitor(
            st.session_state.loaded_model,
            st.session_state.df_credit_data,
            st.session_state.df_credit_data['default'],
            st.session_state.df_credit_data['date'],
            st.session_state.model_features,
            window_size_days=cd_window_size,
            step_size_days=cd_step_size,
            baseline_start_date='2015-01-01',
            baseline_end_date='2020-01-01',
            alert_threshold_yellow=cd_yellow_thresh,
            alert_threshold_red=cd_red_thresh,
            psi_alert_threshold=cd_psi_thresh
        )
        st.session_state.drift_results_df = drift_results_df
        st.session_state.baseline_auc = baseline_auc
    st.success("Concept Drift Monitoring Complete!")

if st.session_state.drift_results_df is not None and not st.session_state.drift_results_df.empty:
    st.subheader(f"Concept Drift Monitor Results (Baseline AUC: {st.session_state.baseline_auc:.4f})")
    st.dataframe(st.session_state.drift_results_df[['window_start', 'window_end', 'auc', 'auc_degradation', 'alert_status', 'psi_fico_score', 'psi_dti']])

    st.subheader("Rolling AUC Performance Over Time")
    fig_auc, ax_auc = plt.subplots(figsize=(14, 7))
    ax_auc.plot(pd.to_datetime(st.session_state.drift_results_df['window_start']), st.session_state.drift_results_df['auc'], label='Rolling AUC', marker='o', markersize=4)
    ax_auc.axhline(y=st.session_state.baseline_auc, color='blue', linestyle='--', label=f'Baseline AUC ({st.session_state.baseline_auc:.3f})')
    
    # Alert zones
    ax_auc.axhspan(st.session_state.baseline_auc - cd_yellow_thresh, st.session_state.baseline_auc, color='green', alpha=0.1, label=f'Green Zone (Degradation < {cd_yellow_thresh})')
    ax_auc.axhspan(st.session_state.baseline_auc - cd_red_thresh, st.session_state.baseline_auc - cd_yellow_thresh, color='yellow', alpha=0.1, label=f'Yellow Zone ({cd_yellow_thresh} <= Degradation < {cd_red_thresh})')
    ax_auc.axhspan(0, st.session_state.baseline_auc - cd_red_thresh, color='red', alpha=0.1, label=f'Red Zone (Degradation >= {cd_red_thresh})')

    red_alerts = st.session_state.drift_results_df[st.session_state.drift_results_df['alert_status'].str.contains('Red', na=False)]
    yellow_alerts = st.session_state.drift_results_df[st.session_state.drift_results_df['alert_status'].str.contains('Yellow', na=False)]

    if not red_alerts.empty:
        ax_auc.scatter(pd.to_datetime(red_alerts['window_start']), red_alerts['auc'], color='red', s=100, zorder=5, label='Red Alert')
    if not yellow_alerts.empty:
        ax_auc.scatter(pd.to_datetime(yellow_alerts['window_start']), yellow_alerts['auc'], color='orange', s=100, zorder=5, label='Yellow Alert')

    ax_auc.set_title('Concept Drift Monitor: Rolling AUC Performance Over Time')
    ax_auc.set_xlabel('Window Start Date')
    ax_auc.set_ylabel('AUC')
    ax_auc.set_ylim(0, 1)
    ax_auc.legend()
    ax_auc.grid(True, linestyle=':', alpha=0.6)
    fig_auc.tight_layout()
    st.pyplot(fig_auc)
    plt.close(fig_auc)

    if 'psi_fico_score' in st.session_state.drift_results_df.columns:
        st.subheader("Population Stability Index (PSI) for FICO Score Over Time")
        fig_psi, ax_psi = plt.subplots(figsize=(14, 4))
        ax_psi.plot(pd.to_datetime(st.session_state.drift_results_df['window_start']), st.session_state.drift_results_df['psi_fico_score'], label='PSI (FICO Score)', marker='x', markersize=4, color='purple')
        ax_psi.axhline(y=cd_psi_thresh, color='red', linestyle='--', label=f'PSI Alert Threshold ({cd_psi_thresh:.2f})')
        ax_psi.set_title('Population Stability Index (PSI) for FICO Score Over Time')
        ax_psi.set_xlabel('Window Start Date')
        ax_psi.set_ylabel('PSI Value')
        ax_psi.legend()
        ax_psi.grid(True, linestyle=':', alpha=0.6)
        fig_psi.tight_layout()
        st.pyplot(fig_psi)
        plt.close(fig_psi)
else:
    st.info("No concept drift monitoring results to display. Run the test first.")
```

**Markdown Content (Explanation):**

```python
st.markdown(f"### Explanation of Execution")
st.markdown(f"The concept drift monitor visualizes the model's rolling AUC performance over time, alongside a baseline AUC and clearly marked alert zones (Green, Yellow, Red). Additionally, the PSI plots for key features like `fico_score` and `dti` show shifts in input data distributions.")
st.markdown(f"For the Model Risk Manager, these outputs provide an \"early warning system\":")
st.markdown(f"-   **Timely Performance Degradation Alerts:**")
st.markdown(r"    -   A `Yellow` alert indicates a moderate performance drop ($\Delta_t \ge 0.03$), prompting an investigation into potential data quality issues or initial signs of concept drift. This allows for proactive intervention.")
st.markdown(r"    -   A `Red` alert signifies a significant performance degradation ($\Delta_t \ge 0.07$), demanding immediate action. This could mean freezing the model for new decisions, triggering a full revalidation, and potentially retraining the model on more recent, relevant data.")
st.markdown(f"-   **Proactive Input Data Monitoring (PSI):** The PSI helps detect shifts in the distribution of input features (e.g., FICO scores changing significantly over time). A high PSI (e.g., > {st.session_state.cd_psi_thresh:.2f}) can alert us to underlying data changes *even before* the model's AUC performance visibly degrades. This provides a crucial early indicator, allowing us to investigate and potentially adapt the model before its predictive power is severely impacted.")
st.markdown(f"This comprehensive monitoring system is fundamental to SR 11-7 compliance, ensuring that model performance is continuously tracked, and timely actions are taken to mitigate risks associated with evolving data patterns and relationships.")
```

---

### Page 7: SR 11-7 Stress Test Report

**Conditional Rendering Check:**

```python
# Check if all preceding tests have been run.
# For simplicity, we'll check for the existence of their results DataFrames.
# In a real app, you might want more robust validation.
can_generate_report = (
    st.session_state.distribution_results is not None and
    st.session_state.boundary_results is not None and
    st.session_state.sensitivity_results is not None and
    st.session_state.adversarial_results_df is not None and
    st.session_state.drift_results_df is not None
)

if not can_generate_report:
    st.warning("Please run all preceding stress tests (2-6) to generate the comprehensive SR 11-7 report.")
else:
    # ... Page content ...
```

**Markdown Content:**

```python
st.title("7. Compile SR 11-7 Aligned Stress Test Report")
st.markdown(f"## Story + Context + Real-World Relevance")
st.markdown(f"After conducting a thorough suite of stress tests, the final and most crucial step for a Model Risk Manager is to consolidate all findings into a structured SR 11-7 report. This document is the ultimate deliverable, serving as the model's \"prospectus\" for internal stakeholders and regulators. It must clearly articulate the model's performance under stress, highlight its limitations, assign severity ratings to identified vulnerabilities, and recommend appropriate use boundaries and mitigation strategies. This report is our evidence of reasonable diligence and effective challenge, ensuring transparency and accountability in model governance.")
st.markdown(f"The report aggregates findings from distribution shift, boundary mapping, feature sensitivity, adversarial robustness, and concept drift tests.")
```

**UI Interactions and Function Invocation:**

```python
if st.button("Generate SR 11-7 Report"):
    with st.spinner("Compiling SR 11-7 Stress Test Report..."):
        sr117_report = compile_stress_report(
            st.session_state.distribution_results,
            st.session_state.boundary_results,
            st.session_state.sensitivity_results,
            st.session_state.adversarial_results_df,
            st.session_state.drift_results_df
        )
        st.session_state.sr117_report = sr117_report
    st.success("SR 11-7 Report Compiled!")

if st.session_state.sr117_report is not None:
    report = st.session_state.sr117_report
    st.subheader("SR 11-7 STRESS TEST REPORT")
    st.markdown("---")
    st.markdown(f"**Model:** `{report['model_name']}`")
    st.markdown(f"**Report Date:** `{report['report_date']}`")
    st.markdown(f"**Overall Assessment:** `{report['overall_assessment']}`")
    st.markdown("---")

    st.markdown(f"### FINDINGS:")
    for test, finding in report['findings'].items():
        st.markdown(f"#### {test.replace('_', ' ').upper()} [{finding['severity']}]:")
        st.markdown(f"- **Recommendation:** {finding['recommendation']}")
        # Add specific details if available
        if test == 'distribution_shift':
            st.markdown(f"  - Max AUC Degradation: {finding.get('max_auc_degradation', 'N/A'):.3f} (Worst: {finding.get('worst_regime_pair', 'N/A')})")
        elif test == 'extreme_values':
            st.markdown(f"  - Unstable Features: {', '.join(finding.get('unstable_features', [])) if finding.get('unstable_features') else 'None'}")
        elif test == 'sensitivity':
            st.markdown(f"  - Most Sensitive Feature: {finding.get('most_sensitive_feature', 'N/A')} (Max Sensitivity: {finding.get('max_sensitivity', np.nan):.3f})")
        elif test == 'adversarial':
            st.markdown(f"  - Most Gameable Feature: {finding.get('most_gameable_feature', 'N/A')} (Min Perturbation to Flip: {finding.get('min_perturbation_to_flip', np.nan):.2%})")
        elif test == 'concept_drift':
            st.markdown(f"  - Drift Alerts Triggered: {finding.get('drift_alerts', 0)}")
        st.markdown("") # Add a newline for spacing

    st.markdown(f"### USE BOUNDARIES:")
    if report['use_boundaries']:
        for boundary in report['use_boundaries']:
            st.markdown(f" - {boundary}")
    else:
        st.markdown(f" - No specific use boundaries recommended at this time, continue standard monitoring.")

    st.markdown(f"### SIGN-OFF REQUIRED:")
    for signoff in report['sign_off_required']:
        st.markdown(f" {signoff}")
```

**Markdown Content (Explanation):**

```python
st.markdown(f"### Explanation of Execution")
st.markdown(f"The compiled SR 11-7 Stress Test Report provides a consolidated, actionable overview of the credit default model's performance under various adverse conditions. For a Model Risk Manager, this report is the bedrock of effective model governance:")
st.markdown(f"-   **Clear Risk Quantification:** Each section (distribution shift, extreme values, sensitivity, adversarial, concept drift) explicitly states findings, severity ratings (LOW, MEDIUM, HIGH), and specific recommendations. This quantifies the risks associated with the model's deployment. For example, a `HIGH` severity for distribution shift coupled with a recommendation to \"Retrain quarterly or when regime indicator triggers\" provides a direct operational directive.")
st.markdown(f"-   **Defined Use Boundaries:** The 'USE BOUNDARIES' section is particularly critical. It translates technical findings into practical constraints on the model's application. For instance, if `fico_score` showed unstable extrapolation, the boundary \"Model shows UNSTABLE extrapolation for 'fico_score'. Limit use beyond training range (e.g., 500-850)\" gives clear guidance to end-users and model owners, preventing misuse in contexts where the model is unvalidated.")
st.markdown(f"-   **Overall Assessment & Accountability:** The 'Overall Assessment' provides a high-level summary of the model's readiness for continued use, signaling if significant risks require immediate attention. The 'SIGN-OFF REQUIRED' section ensures accountability across model development, validation, and risk management teams, fostering a robust governance framework mandated by SR 11-7.")
st.markdown(f"This structured report transforms disparate technical analyses into a cohesive narrative that informs strategic decision-making, mitigates financial exposure, and demonstrates rigorous regulatory compliance for the financial institution.")
```
