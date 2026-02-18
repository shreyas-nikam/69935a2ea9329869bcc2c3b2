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

def generate_synthetic_credit_data(n_samples: int = 10000, start_date: str = '2015-01-01', end_date: str = '2024-01-01', random_state: int = 42) -> pd.DataFrame:
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

def train_and_save_model(df: pd.DataFrame, model_path: str = 'xgboost_credit.pkl') -> tuple[xgb.XGBClassifier, pd.DataFrame, pd.Series, list]:
    """
    Trains an XGBoost classifier on the 'expansion_2015_2019' regime
    and saves it to a .pkl file.

    Args:
        df (pd.DataFrame): The full synthetic credit data.
        model_path (str): Path to save the trained model.

    Returns:
        tuple[xgb.XGBClassifier, pd.DataFrame, pd.Series, list]: The trained XGBoost model,
        training features, training target, and a list of model features.
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
    return model, X_train_base, y_train_base, features

def load_model(model_path: str = 'xgboost_credit.pkl') -> xgb.XGBClassifier:
    """
    Loads a pre-trained XGBoost model from a .pkl file.

    Args:
        model_path (str): Path to the saved model file.

    Returns:
        xgb.XGBClassifier: The loaded XGBoost model.
    """
    loaded_model = joblib.load(model_path)
    print(f"Loaded model from {model_path}. Type: {type(loaded_model)}")
    return loaded_model

def create_regime_splits(df: pd.DataFrame, date_col: str = 'date') -> dict:
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

def distribution_shift_test(model: xgb.XGBClassifier, X_full: pd.DataFrame, y_full: pd.Series, regime_data: dict, features: list) -> pd.DataFrame:
    """
    Evaluates a pre-trained model's performance (AUC) across different economic regimes
    and calculates degradation relative to its training regime.

    Args:
        model (xgb.XGBClassifier): The pre-trained classification model.
        X_full (pd.DataFrame): Full feature set.
        y_full (pd.Series): Full target variable.
        regime_data (dict): Dictionary of regime masks.
        features (list): List of feature names used by the model.

    Returns:
        pd.DataFrame: Results of the distribution shift tests.
    """
    results = []
    min_samples = 100 # Minimum samples required for a regime split to be valid

    X_full = X_full[features] # Ensure X_full only contains features used by the model

    # Calculate in-regime AUC for all regimes first using the pre-trained model
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

    # Now, evaluate the model across all regimes
    # The 'model_train_regime' here refers to the regime the *model* was trained on (expansion_2015_2019)
    for test_regime, test_mask in regime_data.items():
        if test_mask.sum() < min_samples:
            continue

        X_te = X_full[test_mask]
        y_te = y_full[test_mask]

        if len(X_te) < min_samples:
            continue

        y_prob = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, y_prob)

        # Get default rates for comparison
        train_default_rate_mask = regime_data.get('expansion_2015_2019') # Assuming model was trained on this regime
        train_default_rate = y_full[train_default_rate_mask].mean() if train_default_rate_mask is not None else np.nan
        test_default_rate = y_te.mean()

        results.append({
            'model_train_regime': 'expansion_2015_2019', # Explicitly state the model's training regime
            'test_regime': test_regime,
            'auc': auc,
            'train_default_rate': train_default_rate,
            'test_default_rate': test_default_rate,
            'auc_in_regime_base': in_regime_aucs.get('expansion_2015_2019', np.nan) # AUC of the model on its own training regime
        })

    results_df = pd.DataFrame(results)

    # Calculate degradation relative to the model's *original* training regime AUC
    if 'auc_in_regime_base' in results_df.columns and not results_df['auc_in_regime_base'].isna().all():
        # Get the AUC for the 'expansion_2015_2019' regime when tested on itself
        base_auc_for_degradation = results_df[results_df['test_regime'] == 'expansion_2015_2019']['auc'].iloc[0] if not results_df[results_df['test_regime'] == 'expansion_2015_2019'].empty else np.nan
        results_df['auc_degradation'] = base_auc_for_degradation - results_df['auc']
    else:
        results_df['auc_degradation'] = np.nan # Cannot compute degradation without a baseline AUC

    return results_df

def plot_distribution_shift_results(distribution_results: pd.DataFrame):
    """
    Visualizes the results of the distribution shift test.
    """
    if distribution_results.empty:
        print("No distribution shift results to plot.")
        return

    print("DISTRIBUTION SHIFT TEST RESULTS (Evaluated from model trained on Expansion):")
    print("=" * 70)
    print(distribution_results[['test_regime', 'auc', 'auc_degradation', 'train_default_rate', 'test_default_rate']].to_string(index=False))

    plt.figure(figsize=(10, 6))
    pivot_table_auc = distribution_results.pivot_table(index='model_train_regime', columns='test_regime', values='auc')
    sns.heatmap(pivot_table_auc, annot=True, fmt=".3f", cmap="viridis", linewidths=.5)
    plt.title('AUC Performance Across Economic Regimes (Model trained on Expansion)')
    plt.xlabel('Test Regime')
    plt.ylabel('Model Training Regime')
    plt.show()

    plt.figure(figsize=(10, 6))
    degradation_data = distribution_results[distribution_results['model_train_regime'] == 'expansion_2015_2019'][['test_regime', 'auc_degradation']]
    # Filter out the self-comparison for degradation if it's 0 or very small
    degradation_data = degradation_data[degradation_data['test_regime'] != 'expansion_2015_2019']

    if not degradation_data.empty:
        degradation_pivot = degradation_data.set_index('test_regime').T
        sns.heatmap(degradation_pivot, annot=True, fmt=".3f", cmap="Reds", linewidths=.5, cbar_kws={'label': 'AUC Degradation'})
        plt.title('AUC Degradation vs. Expansion Regime (Model trained on Expansion)')
        plt.xlabel('Test Regime')
        plt.ylabel('Degradation Source (relative to Expansion train AUC)')
        plt.show()
    else:
        print("No degradation data to plot (only self-comparison found or empty degradation data).")


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


def boundary_mapping(model: xgb.XGBClassifier, X_baseline: pd.DataFrame, features_to_sweep: list, n_points: int = 50) -> dict:
    """
    For each feature, sweeps values from min to max (and beyond) while holding
    other features at their median. Observes prediction behavior to detect cliffs
    and assess extrapolation stability.

    Args:
        model (xgb.XGBClassifier): The pre-trained classification model.
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

    print("\nRunning Extreme Value Boundary Mapping...")

    for feature in features_to_sweep:
        if feature not in X_baseline.columns:
            print(f"Warning: Feature '{feature}' not found in X_baseline. Skipping.")
            continue
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

def plot_boundary_mapping_results(boundary_results: dict, features_to_plot: list):
    """
    Visualizes the results of the extreme value boundary mapping.
    """
    if not boundary_results:
        print("No boundary mapping results to plot.")
        return

    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(features_to_plot):
        if feature not in boundary_results:
            continue
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

def feature_sensitivity(model: xgb.XGBClassifier, X_sample: pd.DataFrame, features: list, perturbation_percent: float = 0.01) -> pd.DataFrame:
    """
    Computes numerical gradients: how much does prediction change when each
    feature changes by a small percentage (e.g., 1%).

    Args:
        model (xgb.XGBClassifier): The pre-trained classification model.
        X_sample (pd.DataFrame): A sample DataFrame to compute sensitivity on (e.g., test set).
        features (list): List of feature names to analyze.
        perturbation_percent (float): The percentage (e.0.01 for 1%) to perturb features.

    Returns:
        pd.DataFrame: A DataFrame ranking features by their sensitivity.
    """
    if X_sample.empty:
        print("X_sample is empty, skipping feature sensitivity analysis.")
        return pd.DataFrame()

    X_sample_copy = X_sample.copy()
    baseline_probs = model.predict_proba(X_sample_copy)[:, 1]

    sensitivities = []
    for feature in features:
        if feature not in X_sample_copy.columns:
            print(f"Warning: Feature '{feature}' not found in X_sample. Skipping sensitivity for this feature.")
            continue

        X_perturbed = X_sample_copy.copy()

        # Calculate perturbation amount as percentage of the feature's mean in the sample
        perturbation_amount = X_sample_copy[feature].mean() * perturbation_percent

        # Ensure perturbation doesn't lead to negative values for naturally non-negative features
        if feature in ['dti', 'income', 'ltv', 'delinquencies_2yr', 'revolving_utilization']:
            X_perturbed[feature] = (X_perturbed[feature] + perturbation_amount).clip(lower=0.01)
        else: # FICO score might be an exception to lower bound or other features
            X_perturbed[feature] += perturbation_amount

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

def plot_feature_sensitivity_results(sensitivity_results: pd.DataFrame):
    """
    Visualizes the results of the feature sensitivity analysis.
    """
    if sensitivity_results.empty:
        print("No sensitivity results to plot.")
        return

    print("\nFEATURE SENSITIVITY ANALYSIS (1% Perturbation):")
    print("=" * 60)
    print(sensitivity_results.to_string())

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

def adversarial_test(model: xgb.XGBClassifier, X_sample: pd.DataFrame, y_sample: pd.Series, features: list, prob_threshold: float = 0.5, borderline_range: tuple = (0.35, 0.65)) -> pd.DataFrame:
    """
    For borderline predictions (P(default) near decision threshold), finds the minimum
    feature change that flips the prediction. Simulates a borrower gaming the model.

    Args:
        model (xgb.XGBClassifier): The pre-trained classification model.
        X_sample (pd.DataFrame): A sample DataFrame to test for adversarial attacks.
        y_sample (pd.Series): Corresponding true labels.
        features (list): List of feature names to perturb.
        prob_threshold (float): The decision probability threshold (e.g., 0.5 for binary classification).
        borderline_range (tuple): (min_prob, max_prob) for selecting borderline cases.
                                 We are interested in samples predicted as 'default' but close to 'no default'
                                 or vice versa. For flipping from 'default' (prob > 0.5) to 'no default' (prob < 0.5),
                                 we look for samples with P(default) > 0.5 that are in the borderline range.

    Returns:
        pd.DataFrame: Results of adversarial tests, ranking features by "ease of gaming".
    """
    if X_sample.empty:
        print("X_sample is empty, skipping adversarial test.")
        return pd.DataFrame()

    X_sample = X_sample.copy() # Avoid modifying original
    probs = model.predict_proba(X_sample)[:, 1]

    # Select borderline cases: predicted as default, but close to the threshold
    borderline_mask = (probs >= prob_threshold) & (probs <= borderline_range[1])
    X_borderline = X_sample[borderline_mask]
    y_borderline = y_sample[borderline_mask]
    original_probs_borderline = probs[borderline_mask]

    if len(X_borderline) == 0:
        print("No borderline cases found in the specified range for flipping from default to no default.")
        return pd.DataFrame() # Return empty DataFrame

    adversarial_results = []

    # Test a subset of borderline cases to manage computation time
    num_test_cases = min(20, len(X_borderline))
    for i in range(num_test_cases):
        original_idx = X_borderline.index[i]
        original_sample = X_borderline.loc[original_idx:original_idx].copy() # Ensure it's a DataFrame row
        original_prob = original_probs_borderline[i]

        for feature in features:
            if feature not in original_sample.columns:
                continue

            best_delta = np.inf # Store minimum percentage change needed

            # Iterate through a range of percentage changes (e.g., 1% to 50%)
            for delta_pct in np.arange(0.01, 0.51, 0.01): # Decrease by 1% to 50%
                perturbed_sample = original_sample.copy()

                # Perturb the feature value. For credit, often decreasing DTI, increasing FICO etc.
                if feature == 'fico_score':
                    # Increase FICO, cap at 850
                    perturbed_sample[feature] = min(original_sample[feature].iloc[0] * (1 + delta_pct), 850.0)
                elif feature in ['dti', 'income', 'ltv', 'revolving_utilization']:
                    # Decrease others, cap at a small positive value
                    perturbed_sample[feature] = max(original_sample[feature].iloc[0] * (1 - delta_pct), 0.01)
                elif feature == 'delinquencies_2yr':
                    # Decrease delinquencies, cap at 0 and ensure integer
                    perturbed_sample[feature] = int(max(original_sample[feature].iloc[0] * (1 - delta_pct), 0))
                else:
                    # Default behavior for other numerical features (decrease)
                    perturbed_sample[feature] *= (1 - delta_pct)


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
    if adv_df.empty:
        print("No successful flips found for the tested borderline cases.")
        return pd.DataFrame()

    print("\nADVERSARIAL VULNERABILITY RANKING (features by ease of gaming):")
    print("=" * 70)
    print("(Lower delta_pct = easier to game)")
    print(adv_df.groupby('feature')['delta_pct'].mean().sort_values().to_string())

    return adv_df

def plot_adversarial_results(adversarial_results_df: pd.DataFrame):
    """
    Visualizes the results of the adversarial test.
    """
    if adversarial_results_df.empty:
        print("No adversarial results to plot.")
        return

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

def calculate_psi(expected_series: pd.Series, actual_series: pd.Series, num_bins: int = 10) -> float:
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

    # Combine for consistent binning and range definition
    all_data = pd.concat([expected_series, actual_series]).dropna()

    if all_data.nunique() < 2:
        return np.nan # Not enough unique values to form bins

    # Create explicit bin edges
    min_val = all_data.min()
    max_val = all_data.max()

    if np.isclose(min_val, max_val):
        return np.nan # All values are effectively the same

    bins = np.linspace(min_val, max_val, num_bins + 1)
    if len(np.unique(bins)) < 2:
        # If linspace creates non-distinct bins, e.g., if range is too small,
        # adjust to ensure at least 2 distinct bins
        bins = np.histogram_bin_edges(all_data, bins=num_bins)
        if len(np.unique(bins)) < 2:
            return np.nan # Still not enough distinct bins

    # Add a small epsilon to the last bin to ensure max value is included if right=True
    bins[-1] += 1e-6

    # Use pd.cut with predefined bins to ensure consistent binning across series
    expected_binned = pd.cut(expected_series, bins=bins, include_lowest=True, right=True)
    actual_binned = pd.cut(actual_series, bins=bins, include_lowest=True, right=True)

    # Get counts for each bin, normalize, and reindex to ensure all bins are present
    expected_counts = expected_binned.value_counts(normalize=True).reindex(expected_binned.cat.categories, fill_value=0).sort_index()
    actual_counts = actual_binned.value_counts(normalize=True).reindex(actual_binned.cat.categories, fill_value=0).sort_index()

    combined_counts = pd.DataFrame({'expected': expected_counts, 'actual': actual_counts})

    # Replace 0s with a small value to avoid log(0)
    combined_counts.replace(0, 0.0001, inplace=True)

    psi = ((combined_counts['actual'] - combined_counts['expected']) *
           np.log(combined_counts['actual'] / combined_counts['expected'])).sum()
    return psi

def concept_drift_monitor(model: xgb.XGBClassifier, X_data: pd.DataFrame, y_data: pd.Series, dates: pd.Series, features: list,
                          window_size_days: int = 90, step_size_days: int = 30,
                          baseline_start_date: str = '2015-01-01', baseline_end_date: str = '2020-01-01',
                          alert_threshold_yellow: float = 0.03, alert_threshold_red: float = 0.07,
                          psi_alert_threshold: float = 0.25) -> tuple[pd.DataFrame, float]:
    """
    Implements a rolling AUC monitor to detect concept drift.
    Also calculates PSI for key features.

    Args:
        model (xgb.XGBClassifier): The pre-trained classification model.
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
        float: The baseline AUC value.
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
        return pd.DataFrame(), baseline_auc_val # Return empty df if no baseline

    unique_dates = sorted(dates.unique())
    if not unique_dates:
        print("No unique dates in the data. Cannot perform rolling window analysis.")
        return pd.DataFrame(), baseline_auc_val

    start_date_window = pd.to_datetime(unique_dates[0])
    end_data_date = pd.to_datetime(unique_dates[-1])

    print(f"\nCONCEPT DRIFT MONITOR (Baseline AUC: {baseline_auc_val:.4f})")
    print("=" * 70)

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
                # Ensure baseline and window series are not empty before calculating PSI
                if not X_baseline_auc[feature].empty and not X_window[feature].empty:
                    psi = calculate_psi(X_baseline_auc[feature], X_window[feature])
                else:
                    psi = np.nan # Cannot compute PSI if data is missing

                psi_scores[f'psi_{feature}'] = psi
                if psi is not np.nan and psi > psi_alert_threshold:
                    # Upgrade alert status if PSI is bad, but don't downgrade Red
                    alert_status = f"Red (PSI for {feature})" if alert_status != "Red" else alert_status

        results.append({
            'window_start': start_date_window.strftime('%Y-%m-%d'),
            'window_end': (window_end - timedelta(days=1)).strftime('%Y-%m-%d'),
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

def plot_concept_drift_results(drift_results_df: pd.DataFrame, baseline_auc: float,
                                alert_threshold_yellow: float = 0.03, alert_threshold_red: float = 0.07,
                                psi_alert_threshold: float = 0.25):
    """
    Visualizes the results of the concept drift monitor.
    """
    if drift_results_df.empty:
        print("No drift monitoring results to display.")
        return

    print(drift_results_df[['window_start', 'window_end', 'auc', 'auc_degradation', 'alert_status', 'psi_fico_score', 'psi_dti']].to_string(index=False))

    plt.figure(figsize=(14, 7))
    plt.plot(pd.to_datetime(drift_results_df['window_start']), drift_results_df['auc'], label='Rolling AUC', marker='o', markersize=4)
    plt.axhline(y=baseline_auc, color='blue', linestyle='--', label=f'Baseline AUC ({baseline_auc:.3f})')

    # Alert zones
    plt.axhspan(baseline_auc - alert_threshold_yellow, baseline_auc, color='green', alpha=0.1, label=f'Green Zone (Degradation < {alert_threshold_yellow})')
    plt.axhspan(baseline_auc - alert_threshold_red, baseline_auc - alert_threshold_yellow, color='yellow', alpha=0.1, label=f'Yellow Zone ({alert_threshold_yellow} <= Deg. < {alert_threshold_red})')
    plt.axhspan(0, baseline_auc - alert_threshold_red, color='red', alpha=0.1, label=f'Red Zone (Degradation >= {alert_threshold_red})')

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
        plt.axhline(y=psi_alert_threshold, color='red', linestyle='--', label=f'PSI Alert Threshold ({psi_alert_threshold})')
        plt.title('Population Stability Index (PSI) for FICO Score Over Time')
        plt.xlabel('Window Start Date')
        plt.ylabel('PSI Value')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

    # PSI Plot for DTI
    if 'psi_dti' in drift_results_df.columns:
        plt.figure(figsize=(14, 4))
        plt.plot(pd.to_datetime(drift_results_df['window_start']), drift_results_df['psi_dti'], label='PSI (DTI)', marker='x', markersize=4, color='green')
        plt.axhline(y=psi_alert_threshold, color='red', linestyle='--', label=f'PSI Alert Threshold ({psi_alert_threshold})')
        plt.title('Population Stability Index (PSI) for DTI Over Time')
        plt.xlabel('Window Start Date')
        plt.ylabel('PSI Value')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()


def compile_stress_report(dist_shift_results: pd.DataFrame, boundary_results: dict,
                          sensitivity_results: pd.DataFrame, adversarial_results_df: pd.DataFrame, drift_results_df: pd.DataFrame,
                          model_name: str = 'XGBoost Credit Default') -> dict:
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
    max_auc_degradation = dist_shift_results['auc_degradation'].max() if not dist_shift_results.empty and not dist_shift_results['auc_degradation'].isnull().all() else 0.0
    worst_regime_pair_row = dist_shift_results.loc[dist_shift_results['auc_degradation'].idxmax()] if not dist_shift_results.empty and not dist_shift_results['auc_degradation'].isnull().all() else None

    dist_shift_severity = 'LOW'
    dist_shift_recommendation = 'Monitor economic indicators closely.'
    if worst_regime_pair_row is not None and max_auc_degradation >= 0.10:
        dist_shift_severity = 'HIGH'
        dist_shift_recommendation = 'Retrain quarterly or when regime indicator triggers. Consider regime-specific models.'
        report['use_boundaries'].append(f"Model performance degrades by ~{max_auc_degradation:.2%} in '{worst_regime_pair_row['test_regime']}' regime; apply conservative overlay.")
    elif worst_regime_pair_row is not None and max_auc_degradation >= 0.05:
        dist_shift_severity = 'MEDIUM'
        dist_shift_recommendation = 'Investigate moderate performance degradation in specific regimes.'
        report['use_boundaries'].append(f"Model shows moderate performance degradation in '{worst_regime_pair_row['test_regime']}'.")

    report['findings']['distribution_shift'] = {
        'max_auc_degradation': max_auc_degradation,
        'worst_regime_pair': f"Expansion -> {worst_regime_pair_row['test_regime']}" if worst_regime_pair_row is not None else 'N/A',
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
    if not sensitivity_results.empty:
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
    else:
        report['findings']['sensitivity'] = {
            'most_sensitive_feature': 'N/A',
            'max_sensitivity': np.nan,
            'recommendation': 'No sensitivity results available.',
            'severity': 'LOW'
        }


    # --- Adversarial Findings ---
    if adversarial_results_df is not None and not adversarial_results_df.empty:
        mean_delta_by_feature = adversarial_results_df.groupby('feature')['delta_pct'].mean()
        most_gameable_feature = mean_delta_by_feature.idxmin()
        min_perturb_to_flip = mean_delta_by_feature.min()

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
        if not drift_results_df.empty and drift_results_df['alert_status'].str.contains('Red', na=False).any():
            concept_drift_severity = 'HIGH'
            concept_drift_recommendation = 'Immediate investigation and model review required. Consider retraining.'
            report['use_boundaries'].append(f"Model shows active concept drift (Red alerts triggered). Consider conditional use or retraining.")
        elif not drift_results_df.empty and drift_results_df['alert_status'].str.contains('Yellow', na=False).any():
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

def print_stress_report(sr117_report: dict):
    """
    Prints the structured stress test report.
    """
    print("\nSR 11-7 STRESS TEST REPORT")
    print("=" * 60)
    print(f"Model: {sr117_report['model_name']}")
    print(f"Date: {sr117_report['report_date']}")
    print(f"Overall Assessment: {sr117_report['overall_assessment']}")

    print("\nFINDINGS:")
    for test, finding in sr117_report['findings'].items():
        print(f"\n {test.upper()} [{finding['severity']}]:")
        print(f"  Recommendation: {finding['recommendation']}")
        if test == 'distribution_shift':
            print(f"  Max AUC Degradation: {finding['max_auc_degradation']:.3f}")
            print(f"  Worst Regime Pair: {finding['worst_regime_pair']}")
        elif test == 'extreme_values':
            print(f"  Unstable Features: {', '.join(finding['unstable_features']) if finding['unstable_features'] else 'None'}")
        elif test == 'sensitivity':
            print(f"  Most Sensitive Feature: {finding['most_sensitive_feature']}")
            if not np.isnan(finding['max_sensitivity']):
                print(f"  Max Sensitivity Change: {finding['max_sensitivity']:.3f}")
        elif test == 'adversarial':
            print(f"  Most Gameable Feature: {finding['most_gameable_feature']}")
            if not np.isnan(finding['min_perturbation_to_flip']):
                print(f"  Min % Perturbation to Flip: {finding['min_perturbation_to_flip']:.3f}")
        elif test == 'concept_drift':
            print(f"  Drift Alerts Triggered: {finding['drift_alerts']}")


    print("\nUSE BOUNDARIES:")
    if sr117_report['use_boundaries']:
        for boundary in sr117_report['use_boundaries']:
            print(f" - {boundary}")
    else:
        print(" - No specific use boundaries recommended at this time, continue standard monitoring.")

    print("\nSIGN-OFF REQUIRED:")
    for signoff in sr117_report['sign_off_required']:
        print(f" {signoff}")

def run_all_stress_tests(n_samples: int = 15000, model_path: str = 'xgboost_credit.pkl',
                         enable_plots: bool = True) -> dict:
    """
    Orchestrates the entire stress testing process: data generation, model training,
    all stress tests, visualization, and report compilation.

    Args:
        n_samples (int): Number of synthetic data samples to generate.
        model_path (str): Path to save/load the trained model.
        enable_plots (bool): Whether to show plots generated by the tests.

    Returns:
        dict: A dictionary containing the structured stress test report.
    """
    print("Starting Model Stress Testing process...")

    # 1. Generate synthetic credit data
    df_credit_data = generate_synthetic_credit_data(n_samples=n_samples)
    print(f"Generated {len(df_credit_data)} synthetic credit data samples.")

    # 2. Train and save the model
    model, X_train_base, y_train_base, MODEL_FEATURES = train_and_save_model(df_credit_data, model_path=model_path)
    print(f"Model baseline AUC on its training data: {roc_auc_score(y_train_base, model.predict_proba(X_train_base)[:, 1]):.4f}")

    # 3. Create regime splits
    regime_splits = create_regime_splits(df_credit_data, date_col='date')

    # 4. Distribution Shift Test
    distribution_results = distribution_shift_test(model, df_credit_data, df_credit_data['default'], regime_splits, MODEL_FEATURES)
    if enable_plots:
        plot_distribution_shift_results(distribution_results)

    # 5. Extreme Value Boundary Mapping
    features_to_test_boundary = MODEL_FEATURES
    boundary_results = boundary_mapping(model, X_train_base, features_to_test_boundary)
    if enable_plots:
        plot_boundary_mapping_results(boundary_results, features_to_test_boundary)

    # 6. Feature Sensitivity Analysis
    X_sample_sensitivity = X_train_base.sample(n=min(1000, len(X_train_base)), random_state=42)
    sensitivity_results = feature_sensitivity(model, X_sample_sensitivity, MODEL_FEATURES, perturbation_percent=0.01)
    if enable_plots:
        plot_feature_sensitivity_results(sensitivity_results)

    # 7. Adversarial Test
    X_adv_sample = df_credit_data[MODEL_FEATURES].sample(n=min(2000, len(df_credit_data)), random_state=42)
    y_adv_sample = df_credit_data['default'].loc[X_adv_sample.index]
    adversarial_results_df = adversarial_test(model, X_adv_sample, y_adv_sample, MODEL_FEATURES, prob_threshold=0.5)
    if enable_plots:
        plot_adversarial_results(adversarial_results_df)

    # 8. Concept Drift Monitor
    drift_results_df, baseline_auc = concept_drift_monitor(
        model, df_credit_data, df_credit_data['default'], df_credit_data['date'], MODEL_FEATURES,
        window_size_days=90, step_size_days=30,
        baseline_start_date='2015-01-01', baseline_end_date='2020-01-01'
    )
    if enable_plots:
        plot_concept_drift_results(drift_results_df, baseline_auc)

    # 9. Compile and Print SR 11-7 Report
    sr117_report = compile_stress_report(
        distribution_results,
        boundary_results,
        sensitivity_results,
        adversarial_results_df,
        drift_results_df
    )
    print_stress_report(sr117_report)

    print("\nModel Stress Testing process completed.")
    return sr117_report

if __name__ == "__main__":
    # Example usage when run as a script:
    final_report = run_all_stress_tests(n_samples=15000, model_path='xgboost_credit.pkl', enable_plots=True)
    # The final_report dictionary can be further processed, e.g., saved to JSON, displayed in a web app, etc.
