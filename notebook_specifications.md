
# SR 11-7 Aligned Stress Testing of a Credit Default Model

## Introduction: Probing Model Vulnerabilities for Regulatory Compliance

As a Model Risk Manager or Quantitative Analyst at a leading financial institution, your core responsibility transcends merely building accurate models; it's about understanding their limitations and ensuring they meet stringent regulatory standards. The Federal Reserve's SR 11-7 guidance mandates a thorough understanding of model boundaries and failure conditions, especially for critical applications like credit default prediction. A model might show impressive accuracy on historical data, but the real test comes under stress—during economic shifts, extreme borrower profiles, or even attempts at strategic manipulation.

This notebook takes on the persona of a CFA Charterholder, tasked with performing a comprehensive suite of stress tests on our deployed credit default classification model. Our objective is not to build a better model, but to systematically break it, identify its vulnerabilities, and quantify its risks. This process is crucial for preventing costly errors, maintaining regulatory compliance, and making sound capital allocation and lending decisions, ultimately safeguarding the institution against unforeseen model failures.

We will execute a series of five critical stress tests:
1.  **Distribution Shift Testing:** How does the model perform when economic conditions change dramatically?
2.  **Extreme Value Boundary Mapping:** What happens when input features move far outside their typical ranges?
3.  **Feature Sensitivity Analysis:** Which features drive the model's predictions the most, and are they fragile?
4.  **Adversarial Robustness Testing:** Can borrowers strategically "game" the model to flip a credit decision?
5.  **Concept Drift Detection:** How quickly does the model's performance degrade over time as underlying relationships evolve?

Finally, we will compile our findings into a structured SR 11-7 Stress Test Report, providing a clear "prospectus" of the model's strengths, weaknesses, and recommended usage boundaries.

## Setup: Installing Libraries and Importing Dependencies

Before we begin our stress testing, we need to ensure all necessary libraries are installed and imported. These tools will enable us to manipulate data, build predictive models, calculate performance metrics, and visualize our findings effectively.

### Required Libraries Installation

```python
!pip install scikit-learn xgboost numpy pandas matplotlib seaborn scipy joblib
```

### Importing Required Dependencies

```python
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime, timedelta
import warnings

# Suppress UserWarnings, especially for use_label_encoder in XGBoost
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

print("All required libraries imported successfully.")
```

## Step 1: Simulating Data and Loading Pre-trained Model

As a Model Risk Manager, the first step is to establish a realistic testing environment. This involves generating synthetic credit default data that mimics various economic regimes and loading the pre-trained credit default model that we need to stress test. This setup ensures our tests are conducted on a representative dataset and the actual model in question.

### Story + Context + Real-World Relevance

For robust stress testing, we need data that reflects different economic conditions. Our existing credit default model (`xgboost_credit.pkl`) was likely trained on data from an expansionary period. To comply with SR 11-7, we must understand how it performs under adverse scenarios, such as recessions or periods of high interest rates. We will generate synthetic data with a `date` column to simulate these economic regimes and then load the pre-trained model.

**No mathematical formulas for this setup section.**

### Code cell (function definition + function execution)

First, we define a function to generate synthetic credit data, including a `date` column and varying default rates across simulated economic regimes. This ensures our stress tests have realistic and diverse scenarios. We also simulate training and saving a baseline model.

```python
def generate_synthetic_credit_data(n_samples=10000, start_date='2015-01-01', end_date='2024-01-01', random_state=42):
    """
    Generates synthetic credit default data with economic regime shifts.

    Args:
        n_samples (int): Number of synthetic data samples to generate.
        start_date (str): Start date for the synthetic data.
        end_date (str): End date for the synthetic data.
        random_state (int): Seed for reproducibility.

    Returns:
        pd.DataFrame: A DataFrame containing synthetic credit data.
    """
    np.random.seed(random_state)
    dates = pd.to_datetime(pd.date_range(start=start_date, end=end_date, periods=n_samples))

    data = {
        'date': dates,
        'fico_score': np.random.normal(loc=700, scale=50, size=n_samples).clip(300, 850).astype(int),
        'dti': np.random.normal(loc=0.3, scale=0.1, size=n_samples).clip(0.01, 0.8),
        'income': np.random.normal(loc=75000, scale=25000, size=n_samples).clip(20000, 250000).astype(int),
        'ltv': np.random.normal(loc=0.7, scale=0.15, size=n_samples).clip(0.1, 0.95),
        'delinquencies_2yr': np.random.randint(0, 5, size=n_samples),
        'revolving_utilization': np.random.normal(loc=0.4, scale=0.2, size=n_samples).clip(0.01, 1.0)
    }
    df = pd.DataFrame(data)

    # Introduce some regime-specific shifts for default probability
    base_default_prob = 1 / (1 + np.exp(
        - (0.005 * (850 - df['fico_score']) +
           0.08 * df['dti'] +
           -0.00001 * df['income'] +
           0.05 * df['ltv'] +
           0.1 * df['delinquencies_2yr'] +
           0.03 * df['revolving_utilization'] - 5)
    ))

    # Economic regime shifts
    expansion_mask = (df['date'] >= '2015-01-01') & (df['date'] < '2020-01-01')
    base_default_prob[expansion_mask] *= 0.8

    covid_mask = (df['date'] >= '2020-01-01') & (df['date'] < '2021-01-01')
    base_default_prob[covid_mask] *= 1.5

    rate_hike_mask = (df['date'] >= '2022-01-01') & (df['date'] < '2024-01-01')
    base_default_prob[rate_hike_mask] *= 1.2

    df['default'] = (np.random.rand(n_samples) < base_default_prob).astype(int)

    return df

def train_and_save_model(df, model_path='xgboost_credit.pkl'):
    """
    Trains an XGBoost classifier on the 'expansion_2015_2019' regime
    and saves it to a .pkl file.

    Args:
        df (pd.DataFrame): The full synthetic credit data.
        model_path (str): Path to save the trained model.

    Returns:
        xgb.XGBClassifier: The trained XGBoost model.
    """
    features = ['fico_score', 'dti', 'income', 'ltv', 'delinquencies_2yr', 'revolving_utilization']
    X = df[features]
    y = df['default']

    # Use 'expansion_2015_2019' for training baseline
    train_mask = (df['date'] >= '2015-01-01') & (df['date'] < '2020-01-01')
    X_train_base, y_train_base = X[train_mask], y[train_mask]

    # Ensure there's enough data for training
    if len(X_train_base) == 0:
        raise ValueError("No data available for training in the specified baseline regime.")

    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42)
    model.fit(X_train_base, y_train_base)
    joblib.dump(model, model_path)
    print(f"Pre-trained model saved as {model_path}")
    return model, X_train_base, y_train_base

# --- Execution ---
# Generate synthetic credit data
df_credit_data = generate_synthetic_credit_data(n_samples=15000)

# Train and save the model
model, X_train_base, y_train_base = train_and_save_model(df_credit_data)

# Load the model back to simulate using a pre-trained model
loaded_model = joblib.load('xgboost_credit.pkl')

print(f"Loaded model type: {type(loaded_model)}")
print(f"Model baseline AUC on its training data: {roc_auc_score(y_train_base, loaded_model.predict_proba(X_train_base)[:, 1]):.4f}")

# Store features for later use
MODEL_FEATURES = X_train_base.columns.tolist()
```

### Explanation of Execution

We have successfully simulated a comprehensive credit default dataset and established our `xgboost_credit.pkl` model. The `generate_synthetic_credit_data` function created a rich dataset with various financial features and a `default` target, crucially including a `date` column that allows us to simulate different economic regimes. The `train_and_save_model` function then trained an XGBoost classifier on an "expansionary" period (2015-2019) from this synthetic data and saved it. Finally, loading the model confirms it's ready for our stress tests. This setup provides a realistic foundation for applying our SR 11-7 stress testing methodologies, ensuring that the persona is working with data and a model representative of a real-world financial institution.

## Step 2: Distribution Shift Testing

### Story + Context + Real-World Relevance

As a Model Risk Manager, one of the most critical aspects of SR 11-7 compliance is understanding how our credit model's performance degrades when the underlying economic environment changes. Our model was built in a specific regime, and it might fail unpredictably under new conditions. This test simulates exactly that: evaluating the model trained in one economic regime (e.g., expansion) on data from different regimes (e.g., COVID crisis, high interest rates). We need to quantify this degradation using AUC to identify if the model's reliability is compromised.

The degradation is measured as the difference between the model's performance in its training regime and its performance in a new, distinct regime. A significant drop indicates the model is not robust to changing economic conditions.

**Mathematical Formulation:**

The Distribution Shift Degradation ($\Delta AUC$) is calculated as:
$$
\Delta AUC = AUC_{in-regime} - AUC_{cross-regime}
$$
Where:
- $AUC_{in-regime}$ is the Area Under the Receiver Operating Characteristic Curve when the model is evaluated on data from the same economic regime it was trained on (or a similar, stable regime).
- $AUC_{cross-regime}$ is the AUC when the model is evaluated on data from a different, challenging economic regime (e.g., recession, crisis).

**Alert Thresholds for $\Delta AUC$:**
-   **Acceptable (Green):** $\Delta AUC < 0.05$. Model generalizes well across regimes.
-   **Concerning (Yellow):** $0.05 \le \Delta AUC < 0.10$. Model shows moderate regime sensitivity. Document as a limitation.
-   **Failure (Red):** $\Delta AUC \ge 0.10$. Model is regime-dependent. Not suitable for deployment without regime-conditioning or retraining triggers.

### Code cell (function definition + function execution)

```python
def create_regime_splits(df, date_col='date'):
    """
    Splits data into predefined economic regimes for stress testing.

    Args:
        df (pd.DataFrame): The input DataFrame with a 'date' column.
        date_col (str): The name of the date column.

    Returns:
        dict: A dictionary where keys are regime names and values are boolean masks.
    """
    regimes = {
        'expansion_2015_2019': (df[date_col] >= '2015-01-01') & (df[date_col] < '2020-01-01'),
        'covid_2020': (df[date_col] >= '2020-01-01') & (df[date_col] < '2021-01-01'),
        'rate_hike_2022_2023': (df[date_col] >= '2022-01-01') & (df[date_col] < '2024-01-01'),
        'full_period': df[date_col].notna(), # For overall context
    }
    return regimes

def distribution_shift_test(model, X_full, y_full, regime_data, features):
    """
    Trains the model on each regime and tests on all other regimes to measure
    AUC degradation.

    Args:
        model (object): The pre-trained classification model.
        X_full (pd.DataFrame): Full feature set.
        y_full (pd.Series): Full target variable.
        regime_data (dict): Dictionary of regime masks.
        features (list): List of feature names used by the model.

    Returns:
        pd.DataFrame: Results of the distribution shift tests.
    """
    results = []
    min_samples = 100 # Minimum samples required for a regime split to be valid

    # Ensure X_full only contains features used by the model
    X_full = X_full[features]

    # Calculate in-regime AUC for all regimes first using the pre-trained model
    # (Assuming the loaded_model is our "production" model, trained on 'expansion_2015_2019')
    in_regime_aucs = {}
    for regime_name, mask in regime_data.items():
        if mask.sum() < min_samples:
            continue
        X_regime = X_full[mask]
        y_regime = y_full[mask]
        if len(y_regime) > 0:
            probs = model.predict_proba(X_regime)[:, 1]
            in_regime_aucs[regime_name] = roc_auc_score(y_regime, probs)
        else:
            in_regime_aucs[regime_name] = np.nan # No data for this regime

    # Now, evaluate the loaded_model across all regimes
    # The 'train_regime' here refers to the regime the *loaded model* was trained on (expansion_2015_2019)
    # The 'test_regime' refers to which regime we are evaluating *against*.
    for test_regime, test_mask in regime_data.items():
        if test_mask.sum() < min_samples:
            continue

        X_te = X_full[test_mask]
        y_te = y_full[test_mask]

        if len(X_te) < min_samples:
            continue

        y_prob = loaded_model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, y_prob)

        train_default_rate = y_full[regime_data['expansion_2015_2019']].mean() # Default rate of the *training* regime for the loaded model
        test_default_rate = y_te.mean()

        results.append({
            'model_train_regime': 'expansion_2015_2019', # Explicitly state the loaded model's training regime
            'test_regime': test_regime,
            'auc': auc,
            'train_default_rate': train_default_rate,
            'test_default_rate': test_default_rate,
            'auc_in_regime_base': in_regime_aucs.get('expansion_2015_2019', np.nan) # AUC of the loaded model on its own training regime
        })

    results_df = pd.DataFrame(results)

    # Calculate degradation relative to the model's *original* training regime AUC
    # Ensure 'auc_in_regime_base' is present and not NaN
    if 'auc_in_regime_base' in results_df.columns and not results_df['auc_in_regime_base'].isna().all():
        # Get the AUC for the 'expansion_2015_2019' regime when tested on itself (should be similar to base AUC)
        base_auc_for_degradation = results_df[results_df['test_regime'] == 'expansion_2015_2019']['auc'].iloc[0]
        results_df['auc_degradation'] = base_auc_for_degradation - results_df['auc']
    else:
        results_df['auc_degradation'] = np.nan # Cannot compute degradation without a baseline AUC

    return results_df


# --- Execution ---
regime_splits = create_regime_splits(df_credit_data, date_col='date')
distribution_results = distribution_shift_test(loaded_model, df_credit_data[MODEL_FEATURES], df_credit_data['default'], regime_splits, MODEL_FEATURES)

print("DISTRIBUTION SHIFT TEST RESULTS (Evaluated from model trained on Expansion):")
print("=" * 70)
print(distribution_results[['test_regime', 'auc', 'auc_degradation', 'train_default_rate', 'test_default_rate']].to_string(index=False))

# Visualization: Heatmap of AUC values and Degradation
plt.figure(figsize=(10, 6))
pivot_table_auc = distribution_results.pivot_table(index='model_train_regime', columns='test_regime', values='auc')
sns.heatmap(pivot_table_auc, annot=True, fmt=".3f", cmap="viridis", linewidths=.5)
plt.title('AUC Performance Across Economic Regimes (Model trained on Expansion)')
plt.xlabel('Test Regime')
plt.ylabel('Model Training Regime')
plt.show()

plt.figure(figsize=(10, 6))
# For degradation heatmap, we want degradation values for testing against different regimes
# Let's create a single row for the 'model_train_regime' (expansion_2015_2019)
degradation_data = distribution_results[distribution_results['model_train_regime'] == 'expansion_2015_2019'][['test_regime', 'auc_degradation']]
degradation_pivot = degradation_data.set_index('test_regime').T

sns.heatmap(degradation_pivot, annot=True, fmt=".3f", cmap="Reds", linewidths=.5, cbar_kws={'label': 'AUC Degradation'})
plt.title('AUC Degradation vs. Expansion Regime (Model trained on Expansion)')
plt.xlabel('Test Regime')
plt.ylabel('Degradation Source (relative to Expansion train AUC)')
plt.show()

# Summarize degradation based on thresholds
print("\nDegradation Severity Assessment:")
for _, row in distribution_results.iterrows():
    if row['test_regime'] == row['model_train_regime']: # Skip self-comparison for degradation assessment
        continue
    degradation = row['auc_degradation']
    severity = ""
    if degradation < 0.05:
        severity = "Acceptable (Green)"
    elif 0.05 <= degradation < 0.10:
        severity = "Concerning (Yellow)"
    else:
        severity = "Failure (Red)"
    print(f"Test Regime: {row['test_regime']} | AUC Degradation: {degradation:.3f} | Severity: {severity}")
```

### Explanation of Execution

The distribution shift test clearly reveals how our credit default model's performance fluctuates across different economic regimes. The `AUC Performance Across Economic Regimes` heatmap shows the model's AUC when evaluated against various periods, while the `AUC Degradation` heatmap specifically highlights the drop in performance relative to its baseline (trained on 'expansion_2015_2019').

For the Model Risk Manager, these results are critical:
-   A **low AUC degradation** for a test regime indicates the model generalizes well and is robust to those specific economic conditions.
-   A **high AUC degradation**, especially above the `0.10` threshold, signals a severe failure. For example, if the AUC on 'covid_2020' data drops significantly from the 'expansion_2015_2019' baseline, it means the model's learned relationships break down during a crisis. This implies that the model might be unreliable in predicting defaults during similar future crises, leading to misinformed lending decisions and potential capital losses for the institution.
-   The default rates for training vs. test regimes provide additional context, showing if the model is tested on a period with a significantly different underlying default propensity.

This insight directly informs regulatory reporting by identifying specific economic conditions under which the model's predictions become unreliable, requiring either recalibration, conditional usage, or complete retraining. This addresses the SR 11-7 requirement to understand model boundaries.

## Step 3: Extreme Value Boundary Mapping

### Story + Context + Real-World Relevance

Regulators demand to know how our model behaves when confronted with input values far outside its training distribution. This is especially true for features like `DTI` (Debt-to-Income) or `FICO score`, where extreme values, though rare, can significantly impact default probability. As a Model Risk Manager, I need to identify "cliffs"—sudden, large changes in prediction—and assess the stability of model extrapolation. If the model behaves erratically beyond its training data, it poses a significant risk in real-world applications where such extreme cases might occur, even if infrequently.

This test involves systematically sweeping key feature values across their theoretical ranges, extending beyond what the model was trained on, while holding other features constant at their median. We then observe the predicted probability of default.

**No mathematical formulas for this setup section, cliff detection is numerical.**

### Code cell (function definition + function execution)

```python
def boundary_mapping(model, X_baseline, features_to_sweep, n_points=50):
    """
    For each feature, sweeps values from min to max (and beyond) while holding
    other features at their median. Observes prediction behavior to detect cliffs
    and assess extrapolation stability.

    Args:
        model (object): The pre-trained classification model.
        X_baseline (pd.DataFrame): A baseline DataFrame, typically from the training data.
        features_to_sweep (list): List of feature names to sweep.
        n_points (int): Number of points to sweep for each feature.

    Returns:
        dict: A dictionary containing sweep results, including sweep_values, predictions,
              cliff_value, cliff_gradient, and extrapolation_stable for each feature.
    """
    results = {}
    X_baseline = X_baseline.copy() # Avoid modifying original
    X_baseline_median = X_baseline.median().to_frame().T # Use median for other features

    for feature in features_to_sweep:
        # Define sweep range: from 0.01 quantile to 0.99 quantile * 2 (go beyond training range)
        min_val = X_baseline[feature].quantile(0.01)
        max_val = X_baseline[feature].quantile(0.99)
        sweep_values = np.linspace(min_val, max_val * 2, n_points) # Go beyond training max

        predictions = []
        for val in sweep_values:
            temp_df = X_baseline_median.copy()
            temp_df[feature] = val
            prob = model.predict_proba(temp_df)[:, 1][0]
            predictions.append(prob)

        predictions = np.array(predictions)

        # Find the "cliff" where prediction changes most rapidly
        gradients = np.abs(np.diff(predictions))
        if len(gradients) > 0:
            cliff_idx = np.argmax(gradients)
            cliff_value = sweep_values[cliff_idx]
            cliff_gradient = gradients[cliff_idx]
        else:
            cliff_idx = -1
            cliff_value = np.nan
            cliff_gradient = np.nan

        # Check for extrapolation instability
        # Identify values beyond the typical training max (0.99 quantile)
        training_max_boundary = X_baseline[feature].quantile(0.99)
        beyond_training_mask = sweep_values > training_max_boundary

        extrapolation_stable = True
        if beyond_training_mask.any():
            predictions_beyond_training = predictions[beyond_training_mask]
            if len(predictions_beyond_training) > 1:
                # Assess stability using standard deviation of predictions in this region
                # A small standard deviation implies stable extrapolation
                extrapolation_stable = np.std(predictions_beyond_training) < 0.1
            elif len(predictions_beyond_training) == 1:
                extrapolation_stable = True # Single point, cannot assess stability
            else:
                extrapolation_stable = True # No points beyond training, trivially stable

        results[feature] = {
            'sweep_values': sweep_values,
            'predictions': predictions,
            'cliff_value': cliff_value,
            'cliff_gradient': cliff_gradient,
            'extrapolation_stable': extrapolation_stable,
            'training_min': X_baseline[feature].min(),
            'training_max': X_baseline[feature].max(),
            'training_q01': X_baseline[feature].quantile(0.01),
            'training_q99': X_baseline[feature].quantile(0.99)
        }
        print(f" {feature}: cliff at {cliff_value:.2f}, extrapolation {'stable' if extrapolation_stable else 'UNSTABLE'}")
    return results

# --- Execution ---
features_to_test_boundary = ['fico_score', 'dti', 'income', 'ltv', 'revolving_utilization']
boundary_results = boundary_mapping(loaded_model, X_train_base, features_to_test_boundary)

# --- Visualization ---
plt.figure(figsize=(15, 10))
for i, feature in enumerate(features_to_test_boundary):
    plt.subplot(2, 3, i + 1)
    res = boundary_results[feature]
    plt.plot(res['sweep_values'], res['predictions'], label='P(Default)')
    
    # Shade training range
    plt.axvspan(res['training_q01'], res['training_q99'], color='gray', alpha=0.2, label='Training Range (1-99% quantile)')

    if not np.isnan(res['cliff_value']):
        plt.axvline(res['cliff_value'], color='red', linestyle='--', label=f'Cliff at {res["cliff_value"]:.2f}')
    
    if not res['extrapolation_stable']:
        plt.text(0.05, 0.95, 'UNSTABLE Extrapolation', color='red', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

    plt.title(f'P(Default) vs. {feature}')
    plt.xlabel(feature)
    plt.ylabel('Predicted P(Default)')
    plt.legend(fontsize=8)
    plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.suptitle('Extreme Value Boundary Maps for Key Features', y=1.02, fontsize=16)
plt.show()
```

### Explanation of Execution

The extreme value boundary maps graphically illustrate how the model's predicted probability of default responds to variations in key input features, particularly beyond their typical training ranges.

For the Model Risk Manager, these plots are invaluable:
-   **Cliff Detection:** A sudden, sharp change (a "cliff") in the predicted probability, indicated by the red dashed line, suggests model instability. If our `dti` model, for instance, shows a cliff at `DTI = 0.7`, it means a small increase beyond this point can drastically change the prediction. This implies the model is very sensitive and potentially brittle around this threshold.
-   **Extrapolation Stability:** The shaded gray region represents the typical training data range (1st to 99th percentile). How the model behaves *outside* this range is crucial. If the curve becomes erratic, flatlines unexpectedly, or shows sudden jumps (flagged as 'UNSTABLE Extrapolation'), it means the model is extrapolating unpredictably. This is a severe risk because in real-world scenarios, a borrower with an extreme (but valid) `fico_score` or `ltv` might receive an inaccurate, confidence-inducing "no default" prediction, or a wrongly harsh "default" prediction.

Understanding these boundaries enables us to set appropriate use limits for the model (e.g., "model validated only for FICO scores between 500-850, and DTI between 0-60%"). This directly fulfills SR 11-7's requirement to document model limitations and potential failure modes, ensuring users are aware of when to exercise caution or seek additional review.

## Step 4: Feature Sensitivity Analysis

### Story + Context + Real-World Relevance

Understanding which features significantly influence a model's output is critical for managing model risk. As a Model Risk Manager, this is akin to calculating "duration and convexity" for a bond portfolio; it tells me how sensitive the model's output is to small changes in each input. A model that is highly sensitive to a single feature might be fragile and vulnerable to data quality issues or minor shifts in that feature. This analysis helps us identify the most influential, and potentially fragile, features, informing where to focus our data monitoring efforts and internal challenge.

This test quantifies how much the model's default prediction changes in response to small perturbations (e.g., a 1% increase) in each input feature, while holding other features constant.

**No explicit mathematical formulas are displayed in this section, as the approach relies on numerical gradients (change in P(default) per unit change in feature).**

### Code cell (function definition + function execution)

```python
def feature_sensitivity(model, X_sample, features, perturbation_percent=0.01):
    """
    Computes numerical gradients: how much does prediction change when each
    feature changes by a small percentage (e.g., 1%).

    Args:
        model (object): The pre-trained classification model.
        X_sample (pd.DataFrame): A sample DataFrame to compute sensitivity on (e.g., test set).
        features (list): List of feature names to analyze.
        perturbation_percent (float): The percentage (e.g., 0.01 for 1%) to perturb features.

    Returns:
        pd.DataFrame: A DataFrame ranking features by their sensitivity.
    """
    X_sample_copy = X_sample.copy()
    baseline_probs = model.predict_proba(X_sample_copy)[:, 1]

    sensitivities = []
    for feature in features:
        X_perturbed = X_sample_copy.copy()
        
        # Calculate perturbation amount as percentage of the feature's mean in the sample
        # Using mean ensures a consistent relative perturbation across features
        perturbation_amount = X_sample_copy[feature].mean() * perturbation_percent
        
        X_perturbed[feature] += perturbation_amount # Perturb by increasing value

        perturbed_probs = model.predict_proba(X_perturbed)[:, 1]

        # Average absolute change in prediction
        delta = np.abs(perturbed_probs - baseline_probs)
        
        sensitivities.append({
            'feature': feature,
            'mean_abs_change': delta.mean(),
            'max_abs_change': delta.max(),
            'pct_affected_gt_0.01': (delta > 0.01).mean() # Percentage of samples where prediction changed by > 0.01
        })

    sens_df = pd.DataFrame(sensitivities).sort_values('mean_abs_change', ascending=False).set_index('feature')
    return sens_df

# --- Execution ---
# Use a subset of the data for sensitivity analysis for faster computation
X_sample_sensitivity = X_train_base.sample(n=1000, random_state=42)
sensitivity_results = feature_sensitivity(loaded_model, X_sample_sensitivity, MODEL_FEATURES, perturbation_percent=0.01)

print("FEATURE SENSITIVITY ANALYSIS (1% Perturbation):")
print("=" * 60)
print(sensitivity_results.to_string())

# --- Visualization ---
plt.figure(figsize=(10, 6))
sns.barplot(x=sensitivity_results.index, y=sensitivity_results['mean_abs_change'], palette='viridis')
plt.title('Mean Absolute Change in P(Default) for 1% Feature Perturbation')
plt.xlabel('Feature')
plt.ylabel('Mean Absolute Change in P(Default)')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x=sensitivity_results.index, y=sensitivity_results['pct_affected_gt_0.01'], palette='cividis')
plt.title('Percentage of Samples with >0.01 P(Default) Change for 1% Feature Perturbation')
plt.xlabel('Feature')
plt.ylabel('Percentage of Samples Affected')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()
```

### Explanation of Execution

The feature sensitivity analysis provides a quantitative "risk profile" for our credit default model. The bar charts, showing both the mean absolute change in predicted default probability and the percentage of samples affected, immediately highlight which features exert the most influence on the model's predictions.

For the Model Risk Manager, these insights are crucial:
-   **Identifying Fragile Features:** Features with a high `mean_abs_change` or `pct_affected_gt_0.01` are the most influential and potentially fragile. For example, if `fico_score` shows a large mean change, it means even a small 1% shift in FICO can significantly alter the default probability. This makes the model highly dependent on the accuracy and stability of that particular input.
-   **Data Quality Focus:** Knowledge of these sensitive features directs our attention to data quality monitoring. If `dti` is highly sensitive, we must ensure the `DTI` data is accurate and consistently measured, as errors could lead to substantial prediction changes and incorrect lending decisions.
-   **Model Complexity vs. Risk:** This analysis helps in understanding the trade-off. While complex models might achieve higher accuracy, if that accuracy comes with extreme sensitivity to a few features, it introduces fragility. This understanding is essential for SR 11-7, enabling us to document the model's vulnerability to input uncertainty and recommend appropriate safeguards, such as enhanced data validation for critical inputs.

## Step 5: Adversarial Robustness Testing

### Story + Context + Real-World Relevance

In credit markets, borrowers are incentivized to present the best possible financial profile to secure a loan. As a Model Risk Manager, I need to know if our credit model can be "gamed" by strategic applicants who make minimal changes to their data to flip a borderline decision (ee.g., from 'default' to 'no default'). This adversarial robustness test helps quantify how vulnerable our model is to such manipulations, particularly for applications near the decision boundary. Understanding this vulnerability is key to preventing strategic abuse and ensuring fair, robust lending decisions.

This test involves iterating through borderline credit applications and for each, identifying the minimum percentage change to a single input feature that would flip the model's binary prediction.

**No explicit mathematical formulas are displayed in this section, as the approach relies on iterative search for minimum perturbation.**

### Code cell (function definition + function execution)

```python
def adversarial_test(model, X_sample, y_sample, features, target_class=0, prob_threshold=0.5, borderline_range=(0.35, 0.65)):
    """
    For borderline predictions (P(default) near decision threshold), finds the minimum
    feature change that flips the prediction. Simulates a borrower gaming the model.

    Args:
        model (object): The pre-trained classification model.
        X_sample (pd.DataFrame): A sample DataFrame to test for adversarial attacks.
        y_sample (pd.Series): Corresponding true labels.
        features (list): List of feature names to perturb.
        target_class (int): The target class we want to flip *from* (e.g., 0 for 'no default' if originally 1).
                            Here, we want to flip *from* default (1) to no default (0).
        prob_threshold (float): The decision probability threshold (e.g., 0.5 for binary classification).
        borderline_range (tuple): (min_prob, max_prob) for selecting borderline cases.
                                 We are interested in samples predicted as 'default' but close to 'no default'
                                 or vice versa. For flipping from 'default' (prob > 0.5) to 'no default' (prob < 0.5),
                                 we look for samples with P(default) > 0.5 that are in the borderline range.

    Returns:
        pd.DataFrame: Results of adversarial tests, ranking features by "ease of gaming".
    """
    X_sample = X_sample.copy() # Avoid modifying original
    probs = model.predict_proba(X_sample)[:, 1]

    # Select borderline cases: predicted as default, but close to the threshold
    # For this test, let's consider samples where the model predicts default (prob > 0.5)
    # but the probability is within the upper part of the borderline range.
    # We want to flip a prediction from 'default' (1) to 'no default' (0).
    borderline_mask = (probs >= prob_threshold) & (probs <= borderline_range[1]) # Predicted default, but not too confident
    X_borderline = X_sample[borderline_mask]
    y_borderline = y_sample[borderline_mask]
    original_probs_borderline = probs[borderline_mask]

    if len(X_borderline) == 0:
        print("No borderline cases found in the specified range for flipping from default to no default.")
        return None

    adversarial_results = []
    
    # Test a subset of borderline cases to manage computation time
    num_test_cases = min(20, len(X_borderline)) 
    for i in range(num_test_cases):
        original_idx = X_borderline.index[i]
        original_sample = X_borderline.loc[original_idx:original_idx].copy() # Ensure it's a DataFrame row
        original_prob = original_probs_borderline[i]

        for feature in features:
            best_delta = np.inf # Store minimum percentage change needed
            
            # Iterate through a range of percentage changes (e.g., 1% to 50%)
            for delta_pct in np.arange(0.01, 0.51, 0.01): # Decrease by 1% to 50%
                perturbed_sample = original_sample.copy()
                
                # Perturb the feature value. For credit, often decreasing DTI, increasing FICO etc.
                # Assuming 'fico_score' should increase, others decrease for a 'better' profile
                if feature == 'fico_score':
                    perturbed_sample[feature] *= (1 + delta_pct) # Increase FICO
                else:
                    perturbed_sample[feature] *= (1 - delta_pct) # Decrease others

                new_prob = model.predict_proba(perturbed_sample)[:, 1][0]
                
                # Check if the prediction has flipped (from default >=0.5 to no default < 0.5)
                if new_prob < prob_threshold:
                    best_delta = delta_pct
                    break # Found the minimum change for this feature

            if best_delta < np.inf:
                adversarial_results.append({
                    'original_idx': original_idx,
                    'feature': feature,
                    'original_prob': original_prob,
                    'delta_pct': best_delta,
                    'flipped_to': 'no_default'
                })

    adv_df = pd.DataFrame(adversarial_results)
    if len(adv_df) == 0:
        print("No successful flips found for the tested borderline cases.")
        return None

    # Most vulnerable features (easiest to flip = smallest delta_pct)
    vuln_features = adv_df.groupby('feature')['delta_pct'].mean().sort_values()

    print("\nADVERSARIAL VULNERABILITY RANKING (features by ease of gaming):")
    print("=" * 70)
    print("(Lower delta_pct = easier to game)")
    print(vuln_features.to_string())

    return adv_df

# --- Execution ---
# Select a sample for adversarial testing. Using some test data or unseen data is ideal.
# Here we'll take a random sample from the full data, ensuring it covers a range of default probabilities.
X_adv_sample = df_credit_data[MODEL_FEATURES].sample(n=2000, random_state=42)
y_adv_sample = df_credit_data['default'].loc[X_adv_sample.index]

adversarial_results_df = adversarial_test(loaded_model, X_adv_sample, y_adv_sample, MODEL_FEATURES, prob_threshold=0.5)

# --- Visualization ---
if adversarial_results_df is not None:
    plt.figure(figsize=(10, 6))
    vuln_features_plot = adversarial_results_df.groupby('feature')['delta_pct'].mean().sort_values(ascending=True)
    sns.barplot(x=vuln_features_plot.index, y=vuln_features_plot.values, palette='coolwarm')
    plt.title('Features Ranked by Minimum % Perturbation to Flip Prediction (Ease of Gaming)')
    plt.xlabel('Feature')
    plt.ylabel('Mean Minimum % Change to Flip')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()
```

### Explanation of Execution

The adversarial robustness test identifies how easily a model's prediction for a "borderline" credit application can be manipulated by small, strategic changes to input features. The bar chart vividly illustrates which features require the minimum percentage change to flip a 'default' prediction to a 'no default' prediction, essentially ranking them by their "gameability."

For the Model Risk Manager, this is a direct measure of model fragility against strategic behavior:
-   **Identifying Gameable Features:** If features like `revolving_utilization` or `dti` consistently require only a small percentage change (e.g., 5-10%) to flip a decision, it means borrowers could strategically pay down debt or adjust income statements to influence the model. This is not fraud but rational optimization, yet it undermines the model's reliability for these individuals.
-   **Risk Mitigation Strategies:** Knowing these gameable features allows the institution to implement countermeasures. For example, for applications where a sensitive feature is close to a critical threshold, additional verification (e.g., requiring bank statements for `dti`, cross-referencing `income` sources) might be warranted.
-   **SR 11-7 Compliance:** This test directly addresses SR 11-7's call for understanding model vulnerability to strategic manipulation. It helps quantify the risk of lending decisions being swayed by minor, intentional data adjustments, ensuring that the model's 'prospectus' includes a clear warning about its susceptibility to gaming and recommends appropriate safeguards.

## Step 6: Concept Drift Detection

### Story + Context + Real-World Relevance

Models, particularly in dynamic financial environments, do not remain accurate indefinitely. The underlying relationships between credit features and default can change over time—a phenomenon known as concept drift. As a Model Risk Manager, I need a robust monitoring system to detect when our model's performance degrades in real-time. This proactive approach ensures we can trigger investigations, recalibrations, or even full retraining before significant financial risks accumulate. Implementing a rolling AUC monitor with clear alert thresholds is a key component of ongoing model risk management and SR 11-7 compliance.

This test simulates rolling performance monitoring, calculating AUC over sequential time windows and comparing it to a baseline. We also introduce the Population Stability Index (PSI) to detect shifts in input distributions, which can often precede performance degradation.

**Mathematical Formulation:**

The degradation ($\Delta_t$) in rolling AUC performance is calculated as:
$$
\Delta_t = AUC_0 - AUC_t
$$
Where:
- $AUC_0$ is the baseline AUC on a validation set at deployment time (or a stable reference period).
- $AUC_t$ is the rolling AUC calculated over a recent time window.

**Alert Levels for $\Delta_t$:**
-   **Green:** $\Delta_t < 0.03$. Normal variation.
-   **Yellow:** $0.03 \le \Delta_t < 0.07$. Investigate. Check for data quality issues or genuine drift.
-   **Red:** $\Delta_t \ge 0.07$. Freeze model for new decisions. Trigger revalidation and possible retraining.

**Population Stability Index (PSI):**
PSI can supplement AUC monitoring by detecting input distribution shifts even before output performance degrades.
$$
PSI = \sum_{i=1}^{B} \left(P_i^{actual} - P_i^{expected}\right) \ln\left(\frac{P_i^{actual}}{P_i^{expected}}\right)
$$
Where:
- $B$ is the number of bins partitioning the score distribution (or a specific feature's distribution).
- $P_i^{actual}$ is the percentage of observations in bin $i$ for the actual (current) population.
- $P_i^{expected}$ is the percentage of observations in bin $i$ for the expected (baseline) population.
- $PSI > 0.25$ indicates significant population shift (a common alert threshold).

### Code cell (function definition + function execution)

```python
def calculate_psi(expected_series, actual_series, num_bins=10):
    """
    Calculates the Population Stability Index (PSI) between two series.

    Args:
        expected_series (pd.Series): The baseline (expected) distribution.
        actual_series (pd.Series): The current (actual) distribution.
        num_bins (int): Number of bins for discretization.

    Returns:
        float: The calculated PSI value.
    """
    if expected_series.empty or actual_series.empty:
        return np.nan

    # Combine for consistent binning
    all_data = pd.concat([expected_series, actual_series])
    bins = pd.cut(all_data, bins=num_bins, duplicates='drop').categories

    if len(bins) < 2: # Not enough unique values or range for proper binning
        return np.nan

    expected_binned = pd.cut(expected_series, bins=bins, include_lowest=True, right=True)
    actual_binned = pd.cut(actual_series, bins=bins, include_lowest=True, right=True)

    expected_counts = expected_binned.value_counts(normalize=True).sort_index()
    actual_counts = actual_binned.value_counts(normalize=True).sort_index()

    # Align indices and fill missing bins with 0 to ensure proper comparison
    combined_counts = pd.DataFrame({'expected': expected_counts, 'actual': actual_counts}).fillna(0)
    
    # Replace 0s with a small value to avoid log(0)
    combined_counts.replace(0, 0.0001, inplace=True)

    psi = ((combined_counts['actual'] - combined_counts['expected']) * 
           np.log(combined_counts['actual'] / combined_counts['expected'])).sum()
    return psi

def concept_drift_monitor(model, X_data, y_data, dates, features,
                          window_size_days=90, step_size_days=30,
                          baseline_start_date='2015-01-01', baseline_end_date='2020-01-01',
                          alert_threshold_yellow=0.03, alert_threshold_red=0.07,
                          psi_alert_threshold=0.25):
    """
    Implements a rolling AUC monitor to detect concept drift.
    Also calculates PSI for key features.

    Args:
        model (object): The pre-trained classification model.
        X_data (pd.DataFrame): Full feature set.
        y_data (pd.Series): Full target variable.
        dates (pd.Series): Series of dates corresponding to X_data and y_data.
        features (list): List of feature names used by the model.
        window_size_days (int): Size of the rolling window in days.
        step_size_days (int): How many days to advance the window for the next calculation.
        baseline_start_date (str): Start date for the baseline AUC calculation.
        baseline_end_date (str): End date for the baseline AUC calculation.
        alert_threshold_yellow (float): AUC degradation threshold for yellow alert.
        alert_threshold_red (float): AUC degradation threshold for red alert.
        psi_alert_threshold (float): PSI threshold for feature distribution shifts.

    Returns:
        pd.DataFrame: Results of the concept drift monitoring over time.
    """
    results = []

    # Calculate baseline AUC from a stable period (e.g., the model's training regime)
    baseline_mask = (dates >= baseline_start_date) & (dates < baseline_end_date)
    X_baseline_auc = X_data[baseline_mask][features]
    y_baseline_auc = y_data[baseline_mask]

    if len(y_baseline_auc) > 0:
        baseline_auc_val = roc_auc_score(y_baseline_auc, model.predict_proba(X_baseline_auc)[:, 1])
    else:
        baseline_auc_val = np.nan
        print("Warning: No data for baseline AUC calculation. Baseline AUC set to NaN.")

    unique_dates = sorted(dates.unique())
    if not unique_dates:
        print("No unique dates in the data. Cannot perform rolling window analysis.")
        return pd.DataFrame()

    start_date_window = pd.to_datetime(unique_dates[0])
    end_data_date = pd.to_datetime(unique_dates[-1])

    while start_date_window + timedelta(days=window_size_days) <= end_data_date:
        window_end = start_date_window + timedelta(days=window_size_days)
        window_mask = (dates >= start_date_window) & (dates < window_end)

        X_window = X_data[window_mask][features]
        y_window = y_data[window_mask]

        if len(y_window) < 50: # Require minimum samples in window
            start_date_window += timedelta(days=step_size_days)
            continue

        probs = model.predict_proba(X_window)[:, 1]
        auc_window = roc_auc_score(y_window, probs)
        default_rate_window = y_window.mean()
        
        degradation = baseline_auc_val - auc_window if not np.isnan(baseline_auc_val) else np.nan
        
        # Determine alert status
        alert_status = "Green"
        if not np.isnan(degradation):
            if degradation >= alert_threshold_red:
                alert_status = "Red"
            elif degradation >= alert_threshold_yellow:
                alert_status = "Yellow"

        # Calculate PSI for a couple of key features (e.g., 'fico_score', 'dti')
        psi_scores = {}
        for feature in ['fico_score', 'dti']: # Example features for PSI
            if feature in features:
                psi = calculate_psi(X_baseline_auc[feature], X_window[feature])
                psi_scores[f'psi_{feature}'] = psi
                if psi > psi_alert_threshold:
                    alert_status = f"Red (PSI for {feature})" if alert_status != "Red" else alert_status # Upgrade to Red if PSI is bad

        results.append({
            'window_start': start_date_window.strftime('%Y-%m-%d'),
            'window_end': (window_end - timedelta(days=1)).strftime('%Y-%m-%d'), # Subtract 1 day to show actual end date
            'auc': auc_window,
            'default_rate': default_rate_window,
            'n_samples': len(y_window),
            'auc_degradation': degradation,
            'alert_status': alert_status,
            **psi_scores
        })
        start_date_window += timedelta(days=step_size_days)

    drift_df = pd.DataFrame(results)
    return drift_df, baseline_auc_val

# --- Execution ---
drift_results_df, baseline_auc = concept_drift_monitor(
    loaded_model, df_credit_data, df_credit_data['default'], df_credit_data['date'], MODEL_FEATURES,
    window_size_days=90, step_size_days=30,
    baseline_start_date='2015-01-01', baseline_end_date='2020-01-01'
)

print(f"CONCEPT DRIFT MONITOR (Baseline AUC: {baseline_auc:.4f})")
print("=" * 70)
if not drift_results_df.empty:
    print(drift_results_df[['window_start', 'window_end', 'auc', 'auc_degradation', 'alert_status', 'psi_fico_score', 'psi_dti']].to_string(index=False))

    # --- Visualization ---
    plt.figure(figsize=(14, 7))
    plt.plot(pd.to_datetime(drift_results_df['window_start']), drift_results_df['auc'], label='Rolling AUC', marker='o', markersize=4)
    plt.axhline(y=baseline_auc, color='blue', linestyle='--', label=f'Baseline AUC ({baseline_auc:.3f})')
    
    # Alert zones
    plt.axhspan(baseline_auc - 0.03, baseline_auc, color='green', alpha=0.1, label='Green Zone (Degradation < 0.03)')
    plt.axhspan(baseline_auc - 0.07, baseline_auc - 0.03, color='yellow', alpha=0.1, label='Yellow Zone (0.03 <= Degradation < 0.07)')
    plt.axhspan(0, baseline_auc - 0.07, color='red', alpha=0.1, label='Red Zone (Degradation >= 0.07)') # Assuming AUC won't go below 0

    # Mark alerts
    red_alerts = drift_results_df[drift_results_df['alert_status'].str.contains('Red', na=False)]
    yellow_alerts = drift_results_df[drift_results_df['alert_status'].str.contains('Yellow', na=False)]

    if not red_alerts.empty:
        plt.scatter(pd.to_datetime(red_alerts['window_start']), red_alerts['auc'], color='red', s=100, zorder=5, label='Red Alert')
    if not yellow_alerts.empty:
        plt.scatter(pd.to_datetime(yellow_alerts['window_start']), yellow_alerts['auc'], color='orange', s=100, zorder=5, label='Yellow Alert')

    plt.title('Concept Drift Monitor: Rolling AUC Performance Over Time')
    plt.xlabel('Window Start Date')
    plt.ylabel('AUC')
    plt.ylim(0, 1) # AUC range
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # PSI Plot for FICO Score
    if 'psi_fico_score' in drift_results_df.columns:
        plt.figure(figsize=(14, 4))
        plt.plot(pd.to_datetime(drift_results_df['window_start']), drift_results_df['psi_fico_score'], label='PSI (FICO Score)', marker='x', markersize=4, color='purple')
        plt.axhline(y=0.25, color='red', linestyle='--', label='PSI Alert Threshold (0.25)')
        plt.title('Population Stability Index (PSI) for FICO Score Over Time')
        plt.xlabel('Window Start Date')
        plt.ylabel('PSI Value')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

else:
    print("No drift monitoring results to display.")
```

### Explanation of Execution

The concept drift monitor visualizes the model's rolling AUC performance over time, alongside a baseline AUC and clearly marked alert zones (Green, Yellow, Red). Additionally, the PSI plots for key features like `fico_score` and `dti` show shifts in input data distributions.

For the Model Risk Manager, these outputs provide an "early warning system":
-   **Timely Performance Degradation Alerts:**
    -   A `Yellow` alert indicates a moderate performance drop ($\Delta_t \ge 0.03$), prompting an investigation into potential data quality issues or initial signs of concept drift. This allows for proactive intervention.
    -   A `Red` alert signifies a significant performance degradation ($\Delta_t \ge 0.07$), demanding immediate action. This could mean freezing the model for new decisions, triggering a full revalidation, and potentially retraining the model on more recent, relevant data.
-   **Proactive Input Data Monitoring (PSI):** The PSI helps detect shifts in the distribution of input features (e.g., FICO scores changing significantly over time). A high PSI (e.g., > 0.25) can alert us to underlying data changes *even before* the model's AUC performance visibly degrades. This provides a crucial early indicator, allowing us to investigate and potentially adapt the model before its predictive power is severely impacted.

This comprehensive monitoring system is fundamental to SR 11-7 compliance, ensuring that model performance is continuously tracked, and timely actions are taken to mitigate risks associated with evolving data patterns and relationships.

## Step 7: Compile SR 11-7 Aligned Stress Test Report

### Story + Context + Real-World Relevance

After conducting a thorough suite of stress tests, the final and most crucial step for a Model Risk Manager is to consolidate all findings into a structured SR 11-7 report. This document is the ultimate deliverable, serving as the model's "prospectus" for internal stakeholders and regulators. It must clearly articulate the model's performance under stress, highlight its limitations, assign severity ratings to identified vulnerabilities, and recommend appropriate use boundaries and mitigation strategies. This report is our evidence of reasonable diligence and effective challenge, ensuring transparency and accountability in model governance.

The report aggregates findings from distribution shift, boundary mapping, feature sensitivity, adversarial robustness, and concept drift tests.

**No explicit mathematical formulas are displayed in this section, as this is a compilation of previous findings.**

### Code cell (function definition + function execution)

```python
def compile_stress_report(dist_shift_results, boundary_results,
                          sensitivity_results, adversarial_results_df, drift_results_df,
                          model_name='XGBoost Credit Default'):
    """
    Produces an SR 11-7 aligned stress test report by aggregating all findings.

    Args:
        dist_shift_results (pd.DataFrame): Results from distribution shift tests.
        boundary_results (dict): Results from extreme value boundary mapping.
        sensitivity_results (pd.DataFrame): Results from feature sensitivity analysis.
        adversarial_results_df (pd.DataFrame): Results from adversarial robustness testing.
        drift_results_df (pd.DataFrame): Results from concept drift monitoring.
        model_name (str): Name of the model being tested.

    Returns:
        dict: A dictionary representing the structured stress test report.
    """
    report = {
        'model_name': model_name,
        'report_date': datetime.now().isoformat(),
        'stress_tests_conducted': 5,
        'overall_assessment': 'CONDITIONAL PASS', # Default, adjusted based on findings
        'findings': {
            'distribution_shift': {},
            'extreme_values': {},
            'sensitivity': {},
            'adversarial': {},
            'concept_drift': {}
        },
        'use_boundaries': [],
        'sign_off_required': [
            'Model Developer: ____________________',
            'Model Validator: ____________________',
            'Risk Committee: ____________________'
        ]
    }

    # --- Distribution Shift Findings ---
    max_auc_degradation = dist_shift_results['auc_degradation'].max()
    worst_regime_pair_row = dist_shift_results.loc[dist_shift_results['auc_degradation'].idxmax()]
    
    dist_shift_severity = 'LOW'
    dist_shift_recommendation = 'Monitor economic indicators closely.'
    if max_auc_degradation >= 0.10:
        dist_shift_severity = 'HIGH'
        dist_shift_recommendation = 'Retrain quarterly or when regime indicator triggers. Consider regime-specific models.'
        report['use_boundaries'].append(f"Model performance degrades by ~{max_auc_degradation:.2%} in '{worst_regime_pair_row['test_regime']}' regime; apply conservative overlay.")
    elif max_auc_degradation >= 0.05:
        dist_shift_severity = 'MEDIUM'
        dist_shift_recommendation = 'Investigate moderate performance degradation in specific regimes.'
        report['use_boundaries'].append(f"Model shows moderate performance degradation in '{worst_regime_pair_row['test_regime']}'.")
    
    report['findings']['distribution_shift'] = {
        'max_auc_degradation': max_auc_degradation,
        'worst_regime_pair': f"Expansion -> {worst_regime_pair_row['test_regime']}",
        'recommendation': dist_shift_recommendation,
        'severity': dist_shift_severity
    }

    # --- Extreme Values Findings ---
    unstable_features = [f for f, r in boundary_results.items() if not r['extrapolation_stable']]
    if unstable_features:
        extreme_values_severity = 'HIGH'
        extreme_values_recommendation = 'Cap input ranges to training distribution bounds or retrain with more diverse data.'
        for f in unstable_features:
            report['use_boundaries'].append(f"Model shows UNSTABLE extrapolation for '{f}'. Limit use beyond training range (e.g., {boundary_results[f]['training_q01']:.2f}-{boundary_results[f]['training_q99']:.2f}).")
    else:
        extreme_values_severity = 'LOW'
        extreme_values_recommendation = 'Model extrapolation appears stable, continue monitoring.'
    
    report['findings']['extreme_values'] = {
        'unstable_features': unstable_features,
        'recommendation': extreme_values_recommendation,
        'severity': extreme_values_severity
    }

    # --- Sensitivity Findings ---
    most_sensitive_feature = sensitivity_results.index[0]
    max_sensitivity_change = sensitivity_results['mean_abs_change'].iloc[0]
    
    sensitivity_severity = 'LOW'
    sensitivity_recommendation = 'Monitor top-3 sensitive features for data quality.'
    if max_sensitivity_change > 0.15: # Arbitrary threshold for high sensitivity, adjust as needed
        sensitivity_severity = 'MEDIUM'
        sensitivity_recommendation = 'Implement enhanced data validation for highly sensitive features.'
    
    report['findings']['sensitivity'] = {
        'most_sensitive_feature': most_sensitive_feature,
        'max_sensitivity': max_sensitivity_change,
        'recommendation': sensitivity_recommendation,
        'severity': sensitivity_severity
    }

    # --- Adversarial Findings ---
    if adversarial_results_df is not None and not adversarial_results_df.empty:
        most_gameable_feature = adversarial_results_df.groupby('feature')['delta_pct'].mean().idxmin()
        min_perturb_to_flip = adversarial_results_df.groupby('feature')['delta_pct'].mean().min()
        
        adversarial_severity = 'LOW'
        adversarial_recommendation = 'Add verification for borderline applications related to sensitive features.'
        if min_perturb_to_flip < 0.10: # If less than 10% change can flip
            adversarial_severity = 'MEDIUM'
            adversarial_recommendation = f"Feature '{most_gameable_feature}' is highly gameable ({min_perturb_to_flip:.2%} change to flip). Implement stringent verification for applications near decision boundary."
            report['use_boundaries'].append(f"Model susceptible to gaming via '{most_gameable_feature}' (min change to flip: {min_perturb_to_flip:.2%}).")
        
        report['findings']['adversarial'] = {
            'most_gameable_feature': most_gameable_feature,
            'min_perturbation_to_flip': min_perturb_to_flip,
            'recommendation': adversarial_recommendation,
            'severity': adversarial_severity
        }
    else:
        report['findings']['adversarial'] = {
            'most_gameable_feature': 'N/A',
            'min_perturbation_to_flip': np.nan,
            'recommendation': 'No significant adversarial vulnerabilities detected or insufficient data.',
            'severity': 'LOW'
        }

    # --- Concept Drift Findings ---
    drift_alerts_count = drift_results_df[drift_results_df['alert_status'].isin(['Yellow', 'Red'])].shape[0] if not drift_results_df.empty else 0
    
    concept_drift_severity = 'LOW'
    concept_drift_recommendation = 'Implement monthly rolling AUC monitoring with automated alerts.'
    if drift_alerts_count > 0:
        if drift_results_df['alert_status'].str.contains('Red', na=False).any():
            concept_drift_severity = 'HIGH'
            concept_drift_recommendation = 'Immediate investigation and model review required. Consider retraining.'
            report['use_boundaries'].append(f"Model shows active concept drift (Red alerts triggered). Consider conditional use or retraining.")
        elif drift_results_df['alert_status'].str.contains('Yellow', na=False).any():
            concept_drift_severity = 'MEDIUM'
            concept_drift_recommendation = 'Investigate yellow alerts for data quality or early drift.'
            report['use_boundaries'].append(f"Model shows signs of concept drift (Yellow alerts triggered).")

    report['findings']['concept_drift'] = {
        'drift_alerts': drift_alerts_count,
        'recommendation': concept_drift_recommendation,
        'severity': concept_drift_severity
    }

    # Overall Assessment based on highest severity
    severities = [
        report['findings']['distribution_shift']['severity'],
        report['findings']['extreme_values']['severity'],
        report['findings']['sensitivity']['severity'],
        report['findings']['adversarial']['severity'],
        report['findings']['concept_drift']['severity']
    ]
    
    if 'HIGH' in severities:
        report['overall_assessment'] = 'CONDITIONAL PASS - HIGH RISK IDENTIFIED'
    elif 'MEDIUM' in severities:
        report['overall_assessment'] = 'CONDITIONAL PASS - MEDIUM RISK IDENTIFIED'
    else:
        report['overall_assessment'] = 'PASS'

    return report

# --- Execution ---
sr117_report = compile_stress_report(
    distribution_results,
    boundary_results,
    sensitivity_results,
    adversarial_results_df,
    drift_results_df
)

print("SR 11-7 STRESS TEST REPORT")
print("=" * 60)
print(f"Model: {sr117_report['model_name']}")
print(f"Date: {sr117_report['report_date']}")
print(f"Overall Assessment: {sr117_report['overall_assessment']}")

print("\nFINDINGS:")
for test, finding in sr117_report['findings'].items():
    print(f"\n {test.upper()} [{finding['severity']}]:")
    print(f"  Recommendation: {finding['recommendation']}")

print("\nUSE BOUNDARIES:")
if sr117_report['use_boundaries']:
    for boundary in sr117_report['use_boundaries']:
        print(f" - {boundary}")
else:
    print(" - No specific use boundaries recommended at this time, continue standard monitoring.")

print("\nSIGN-OFF REQUIRED:")
for signoff in sr117_report['sign_off_required']:
    print(f" {signoff}")
```

### Explanation of Execution

The compiled SR 11-7 Stress Test Report provides a consolidated, actionable overview of the credit default model's performance under various adverse conditions. For a Model Risk Manager, this report is the bedrock of effective model governance:

-   **Clear Risk Quantification:** Each section (distribution shift, extreme values, sensitivity, adversarial, concept drift) explicitly states findings, severity ratings (LOW, MEDIUM, HIGH), and specific recommendations. This quantifies the risks associated with the model's deployment. For example, a `HIGH` severity for distribution shift coupled with a recommendation to "Retrain quarterly or when regime indicator triggers" provides a direct operational directive.
-   **Defined Use Boundaries:** The 'USE BOUNDARIES' section is particularly critical. It translates technical findings into practical constraints on the model's application. For instance, if `fico_score` showed unstable extrapolation, the boundary "Model shows UNSTABLE extrapolation for 'fico_score'. Limit use beyond training range (e.g., 500-850)" gives clear guidance to end-users and model owners, preventing misuse in contexts where the model is unvalidated.
-   **Overall Assessment & Accountability:** The 'Overall Assessment' provides a high-level summary of the model's readiness for continued use, signaling if significant risks require immediate attention. The 'SIGN-OFF REQUIRED' section ensures accountability across model development, validation, and risk management teams, fostering a robust governance framework mandated by SR 11-7.

This structured report transforms disparate technical analyses into a cohesive narrative that informs strategic decision-making, mitigates financial exposure, and demonstrates rigorous regulatory compliance for the financial institution.
