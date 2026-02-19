import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import joblib
from sklearn.metrics import roc_auc_score

# Import all functions from source.py
from source import *

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# -----------------------------
# Page Config + Global Styling
# -----------------------------
st.set_page_config(
    page_title="QuLab: Lab 37: Stress Testing a Model", layout="wide")

st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()

st.title("QuLab: Lab 37: Stress Testing a Model")
st.divider()


# -----------------------------
# Session State Initialization
# -----------------------------
DEFAULT_STATE = {
    "df_credit_data": None,
    "loaded_model": None,
    "X_train_base": None,
    "y_train_base": None,
    "model_features": None,

    "distribution_results": None,

    "boundary_results": None,
    "selected_boundary_features": [],

    "sensitivity_results": None,
    "perturbation_pct": 0.01,

    "adversarial_results_df": None,
    "adv_prob_threshold": 0.5,
    "adv_borderline_upper": 0.65,

    "drift_results_df": None,
    "baseline_auc": None,
    "cd_window_size": 90,
    "cd_step_size": 30,
    "cd_yellow_thresh": 0.03,
    "cd_red_thresh": 0.07,
    "cd_psi_thresh": 0.25,

    "sr117_report": None,

    # Pedagogy toggles
    "learning_mode": True,
    "show_interpretation_prompts": True,
}

for k, v in DEFAULT_STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v


# -----------------------------
# Helpers (UI)
# -----------------------------
def callout(title: str, body: str, kind: str = "info"):
    if kind == "info":
        st.info(f"**{title}**\n\n{body}")
    elif kind == "success":
        st.success(f"**{title}**\n\n{body}")
    elif kind == "warning":
        st.warning(f"**{title}**\n\n{body}")
    else:
        st.write(f"**{title}**\n\n{body}")


def kpi_row(items):
    """items: list of tuples (label, value)"""
    cols = st.columns(len(items))
    for i, (label, value) in enumerate(items):
        cols[i].metric(label, value)


def require_setup_guard():
    if st.session_state.df_credit_data is None or st.session_state.loaded_model is None:
        st.warning(
            "Please go to **1. Introduction & Setup** and click **Generate Data & Load Model** first.")
        return True
    return False


def progress_tracker():
    # checks = [
    #     ("Setup Complete", st.session_state.df_credit_data is not None and st.session_state.loaded_model is not None),
    #     ("Distribution Shift Done", st.session_state.distribution_results is not None),
    #     ("Boundary Mapping Done", st.session_state.boundary_results is not None),
    #     ("Sensitivity Done", st.session_state.sensitivity_results is not None),
    #     ("Adversarial Done", st.session_state.adversarial_results_df is not None),
    #     ("Concept Drift Done", st.session_state.drift_results_df is not None),
    #     ("Report Generated", st.session_state.sr117_report is not None),
    # ]
    # done = sum(int(x[1]) for x in checks)
    # st.sidebar.markdown("### Progress")
    # st.sidebar.progress(done / len(checks))
    # for label, ok in checks:
    #     st.sidebar.write(("✅ " if ok else "⬜ ") + label)
    pass


def severity_from_degradation(deg: float):
    if deg < 0.05:
        return "Acceptable (Green)"
    if 0.05 <= deg < 0.10:
        return "Concerning (Yellow)"
    return "Failure (Red)"


def render_interpretation_prompts(prompts):
    if st.session_state.show_interpretation_prompts:
        with st.expander("Interpretation prompts (for decision-usefulness)", expanded=st.session_state.learning_mode):
            for p in prompts:
                st.markdown(f"- {p}")


progress_tracker()


# -----------------------------
# Sidebar Controls + Navigation
# -----------------------------
st.sidebar.title("SR 11-7 Model Stress Testing")

st.sidebar.toggle(
    "Learning mode (more guidance)",
    key="learning_mode",
    help="When on, the app shows structured context, prompts, and checklists."
)
st.sidebar.toggle(
    "Show interpretation prompts",
    key="show_interpretation_prompts",
    help="When on, the app provides short prompts to connect outputs to real decisions."
)

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


# -----------------------------
# Page 1: Introduction & Setup
# -----------------------------
def page_intro_setup():
    st.title("SR 11-7 Model Stress Testing for Credit Default")
    st.markdown("**Persona:** CFA Charterholder & Model Risk Manager")

    if st.session_state.learning_mode:
        callout(
            "What you’re doing here",
            "You’re not trying to improve the model. You’re trying to **stress it** and document **where it fails**, "
            "so governance teams can set **use boundaries**, **monitoring triggers**, and **controls**.",
            kind="info",
        )

    st.markdown(
        "## Introduction: Probing Model Vulnerabilities for Regulatory Compliance")
    st.markdown(
        "As a Model Risk Manager or Quantitative Analyst at a leading financial institution, your core responsibility "
        "transcends merely building accurate models; it's about understanding their limitations and ensuring they meet "
        "stringent regulatory standards. The Federal Reserve's SR 11-7 guidance mandates a thorough understanding of model "
        "boundaries and failure conditions, especially for critical applications like credit default prediction. A model "
        "might show impressive accuracy on historical data, but the real test comes under stress—during economic shifts, "
        "extreme borrower profiles, or even attempts at strategic manipulation."
    )
    st.markdown(
        "This application takes on the persona of a CFA Charterholder, tasked with performing a comprehensive suite of "
        "stress tests on our deployed credit default classification model. Our objective is not to build a better model, "
        "but to systematically break it, identify its vulnerabilities, and quantify its risks. This process is crucial for "
        "preventing costly errors, maintaining regulatory compliance, and making sound capital allocation and lending "
        "decisions, ultimately safeguarding the institution against unforeseen model failures."
    )
    st.markdown("We will execute a series of five critical stress tests:")
    st.markdown(
        "1.  **Distribution Shift Testing:** How does the model perform when economic conditions change dramatically?")
    st.markdown(
        "2.  **Extreme Value Boundary Mapping:** What happens when input features move far outside their typical ranges?")
    st.markdown(
        "3.  **Feature Sensitivity Analysis:** Which features drive the model's predictions the most, and are they fragile?")
    st.markdown(
        "4.  **Adversarial Robustness Testing:** Can borrowers strategically \"game\" the model to flip a credit decision?")
    st.markdown("5.  **Concept Drift Detection:** How quickly does the model's performance degrade over time as underlying relationships evolve?")
    st.markdown(
        "Finally, we will compile our findings into a structured SR 11-7 Stress Test Report, providing a clear "
        "\"prospectus\" of the model's strengths, weaknesses, and recommended usage boundaries."
    )

    st.markdown("## Setup: Simulating Data and Loading Pre-trained Model")
    st.markdown(
        "As a Model Risk Manager, the first step is to establish a realistic testing environment. This involves "
        "generating synthetic credit default data that mimics various economic regimes and loading the pre-trained credit "
        "default model that we need to stress test. This setup ensures our tests are conducted on a representative dataset "
        "and the actual model in question."
    )
    st.markdown(
        "For robust stress testing, we need data that reflects different economic conditions. Our existing credit default "
        "model (`xgboost_credit.pkl`) was likely trained on data from an expansionary period. To comply with SR 11-7, we must "
        "understand how it performs under adverse scenarios, such as recessions or periods of high interest rates. We will "
        "generate synthetic data with a `date` column to simulate these economic regimes and then load the pre-trained model."
    )

    cols = st.columns([1, 1, 2])
    with cols[0]:
        run = st.button("Generate Data & Load Model")
    with cols[1]:
        show_sample = st.button("Preview Data (if available)")

    if run:
        with st.spinner("Generating synthetic data and training/loading model... This might take a moment."):
            df_credit_data = generate_synthetic_credit_data(n_samples=15000)
            st.session_state.df_credit_data = df_credit_data

            model, X_train_base, y_train_base, _ = train_and_save_model(
                df_credit_data)
            st.session_state.X_train_base = X_train_base
            st.session_state.y_train_base = y_train_base

            loaded_model = joblib.load("xgboost_credit.pkl")
            st.session_state.loaded_model = loaded_model
            st.session_state.model_features = X_train_base.columns.tolist()

        st.success("Data generated and model loaded successfully!")

    if st.session_state.loaded_model is not None and st.session_state.X_train_base is not None:
        baseline_auc = roc_auc_score(
            st.session_state.y_train_base,
            st.session_state.loaded_model.predict_proba(
                st.session_state.X_train_base)[:, 1]
        )

        kpi_row([
            ("Model type", type(st.session_state.loaded_model).__name__),
            ("Baseline AUC (train)", f"{baseline_auc:.4f}"),
            ("# Features", str(len(st.session_state.model_features))),
        ])

        with st.expander("Features used by model", expanded=False):
            st.code(", ".join(st.session_state.model_features))

    if show_sample and st.session_state.df_credit_data is not None:
        st.subheader("Data preview")
        st.dataframe(st.session_state.df_credit_data.head(25))

    st.markdown("---")
    st.markdown(
        "We have successfully simulated a comprehensive credit default dataset and established our `xgboost_credit.pkl` model. "
        "The `generate_synthetic_credit_data` function created a rich dataset with various financial features and a `default` "
        "target, crucially including a `date` column that allows us to simulate different economic regimes. The "
        "`train_and_save_model` function then trained an XGBoost classifier on an \"expansionary\" period (2015-2019) from "
        "this synthetic data and saved it. Finally, loading the model confirms it's ready for our stress tests. This setup "
        "provides a realistic foundation for applying our SR 11-7 stress testing methodologies, ensuring that the persona is "
        "working with data and a model representative of a real-world financial institution."
    )

    if st.session_state.learning_mode:
        render_interpretation_prompts([
            "If the baseline AUC looks strong, what would make you *still* cautious before deployment?",
            "Which governance stakeholders (Model Owner, Validator, Risk Committee) will care about which outputs later?",
        ])


# -----------------------------
# Page 2: Distribution Shift Testing
# -----------------------------
def page_distribution_shift():
    if require_setup_guard():
        return

    st.title("2. Distribution Shift Testing")

    tabs = st.tabs(["Context", "Run the test", "Interpretation"])
    with tabs[0]:

        st.markdown(
            "As a Model Risk Manager, one of the most critical aspects of SR 11-7 compliance is understanding how our credit "
            "model's performance degrades when the underlying economic environment changes. Our model was built in a specific "
            "regime, and it might fail unpredictably under new conditions. This test simulates exactly that: evaluating the "
            "model trained in one economic regime (e.g., expansion) on data from different regimes (e.g., COVID crisis, high "
            "interest rates). We need to quantify this degradation using AUC to identify if the model's reliability is compromised."
        )
        st.markdown(
            "The degradation is measured as the difference between the model's performance in its training regime and its "
            "performance in a new, distinct regime. A significant drop indicates the model is not robust to changing economic conditions."
        )

        # --- PRESERVE FORMULAE VERBATIM ---
        st.markdown(r"**Mathematical Formulation:**")
        st.markdown(
            r"The Distribution Shift Degradation ($\Delta AUC$) is calculated as:")

        # Fixed the missing closing braces in the subscripts
        st.markdown(r"""
        $$
        \Delta AUC = AUC_{\text{in-regime}} - AUC_{\text{cross-regime}}
        $$
        """)

        # Added missing closing $ for the variables in descriptions
        st.markdown(
            r"where $AUC_{\text{in-regime}}$ is the Area Under the Receiver Operating Characteristic Curve when the model is evaluated on data from the same economic regime it was trained on.")
        st.markdown(
            r"where $AUC_{\text{cross-regime}}$ is the AUC when the model is evaluated on data from a different, challenging economic regime (e.g., recession, crisis).")

        st.markdown(r"**Alert Thresholds for $\Delta AUC$:**")
        st.markdown(
            r"- **Acceptable (Green):** $\Delta AUC < 0.05$. Model generalizes well across regimes.")
        st.markdown(
            r"- **Concerning (Yellow):** $0.05 \le \Delta AUC < 0.10$. Model shows moderate regime sensitivity.")
        st.markdown(
            r"- **Failure (Red):** $\Delta AUC \ge 0.10$. Model is regime-dependent.")
    with tabs[1]:
        if st.button("Run Distribution Shift Test"):
            with st.spinner("Performing distribution shift analysis..."):
                regime_splits = create_regime_splits(
                    st.session_state.df_credit_data, date_col="date")
                distribution_results = distribution_shift_test(
                    st.session_state.loaded_model,
                    st.session_state.df_credit_data[st.session_state.model_features],
                    st.session_state.df_credit_data["default"],
                    regime_splits,
                    st.session_state.model_features,
                )
                st.session_state.distribution_results = distribution_results

            st.success("Distribution Shift Test Complete!")

        if st.session_state.distribution_results is not None:
            st.subheader("Results Table")
            show_cols = ["test_regime", "auc", "auc_degradation",
                         "train_default_rate", "test_default_rate"]
            st.dataframe(st.session_state.distribution_results[show_cols])

            # AUC heatmap (matplotlib-based)
            st.subheader("AUC Performance Across Economic Regimes")
            pivot_auc = st.session_state.distribution_results.pivot_table(
                index="model_train_regime", columns="test_regime", values="auc"
            )

            fig1, ax1 = plt.subplots(figsize=(10, 5))
            im = ax1.imshow(pivot_auc.values, aspect="auto")
            ax1.set_xticks(np.arange(pivot_auc.shape[1]))
            ax1.set_yticks(np.arange(pivot_auc.shape[0]))
            ax1.set_xticklabels(pivot_auc.columns, rotation=30, ha="right")
            ax1.set_yticklabels(pivot_auc.index)
            ax1.set_title(
                "AUC Performance Across Economic Regimes (Model trained on Expansion)")
            fig1.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
            # annotate
            for i in range(pivot_auc.shape[0]):
                for j in range(pivot_auc.shape[1]):
                    ax1.text(
                        j, i, f"{pivot_auc.values[i, j]:.3f}", ha="center", va="center")
            fig1.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)

            st.subheader("AUC Degradation vs. Expansion Regime")
            deg_df = st.session_state.distribution_results[
                st.session_state.distribution_results["model_train_regime"] == "expansion_2015_2019"
            ][["test_regime", "auc_degradation"]]
            deg_vals = deg_df.set_index("test_regime").T

            fig2, ax2 = plt.subplots(figsize=(10, 2.8))
            im2 = ax2.imshow(deg_vals.values, aspect="auto")
            ax2.set_xticks(np.arange(deg_vals.shape[1]))
            ax2.set_yticks([0])
            ax2.set_xticklabels(deg_vals.columns, rotation=30, ha="right")
            ax2.set_yticklabels(["ΔAUC vs Expansion"])
            ax2.set_title(
                "AUC Degradation vs. Expansion Regime (Model trained on Expansion)")
            fig2.colorbar(im2, ax=ax2, fraction=0.046,
                          pad=0.04, label="AUC Degradation")
            for j in range(deg_vals.shape[1]):
                ax2.text(
                    j, 0, f"{deg_vals.values[0, j]:.3f}", ha="center", va="center")
            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

    with tabs[2]:
        if st.session_state.distribution_results is None:
            st.info(
                "Run the test first to see interpretation guidance tied to your outputs.")
            return

        st.subheader("Degradation severity assessment")
        for _, row in st.session_state.distribution_results.iterrows():
            if row["test_regime"] == row["model_train_regime"]:
                continue
            degradation = float(row["auc_degradation"])
            st.markdown(
                f"- **Test Regime:** `{row['test_regime']}` | "
                f"**AUC Degradation:** `{degradation:.3f}` | "
                f"**Severity:** `{severity_from_degradation(degradation)}`"
            )

        if st.session_state.learning_mode:
            render_interpretation_prompts([
                "Which regimes (crisis/high-rate) create the largest degradation—and would you limit model use there?",
                "If degradation is Yellow/Red, what governance action follows: monitoring trigger vs retraining vs freeze?",
                "Do default-rate shifts explain degradation, or does it look like a relationship change (true regime break)?",
            ])

    st.markdown("---")
    st.markdown(
        "The distribution shift test clearly reveals how our credit default model's performance fluctuates across different "
        "economic regimes. The `AUC Performance Across Economic Regimes` heatmap shows the model's AUC when evaluated against "
        "various periods, while the `AUC Degradation` heatmap specifically highlights the drop in performance relative to its "
        "baseline (trained on 'expansion_2015_2019')."
    )
    st.markdown("For the Model Risk Manager, these results are critical:")
    st.markdown("-   A **low AUC degradation** for a test regime indicates the model generalizes well and is robust to those specific economic conditions.")
    st.markdown("-   A **high AUC degradation**, especially above the `0.10` threshold, signals a severe failure. For example, if the AUC on 'covid_2020' data drops significantly from the 'expansion_2015_2019' baseline, it means the model's learned relationships break down during a crisis. This implies that the model might be unreliable in predicting defaults during similar future crises, leading to misinformed lending decisions and potential capital losses for the institution.")
    st.markdown("-   The default rates for training vs. test regimes provide additional context, showing if the model is tested on a period with a significantly different underlying default propensity.")
    st.markdown(
        "This insight directly informs regulatory reporting by identifying specific economic conditions under which the model's "
        "predictions become unreliable, requiring either recalibration, conditional usage, or complete retraining. This addresses "
        "the SR 11-7 requirement to understand model boundaries."
    )


# -----------------------------
# Page 3: Extreme Value Boundary Mapping
# -----------------------------
def page_boundary_mapping():
    if st.session_state.X_train_base is None or st.session_state.loaded_model is None:
        st.warning(
            "Please go to '1. Introduction & Setup' and generate data/load model first.")
        return

    st.title("3. Extreme Value Boundary Mapping")

    if st.session_state.learning_mode:
        callout(
            "Decision use",
            "You’re identifying **cliffs** and **unreliable extrapolation** so you can define **validated input ranges** "
            "and force **manual review** outside those ranges.",
            "info",
        )

    st.markdown(
        "Regulators demand to know how our model behaves when confronted with input values far outside its training distribution. "
        "This is especially true for features like `DTI` (Debt-to-Income) or `FICO score`, where extreme values, though rare, can "
        "significantly impact default probability. As a Model Risk Manager, I need to identify \"cliffs\"—sudden, large changes in "
        "prediction—and assess the stability of model extrapolation. If the model behaves erratically beyond its training data, it "
        "poses a significant risk in real-world applications where such extreme cases might occur, even if infrequently."
    )
    st.markdown(
        "This test involves systematically sweeping key feature values across their theoretical ranges, extending beyond what the "
        "model was trained on, while holding other features constant at their median. We then observe the predicted probability of default."
    )

    features_for_boundary = ["fico_score", "dti", "income",
                             "ltv", "revolving_utilization", "delinquencies_2yr"]
    selected_features = st.multiselect(
        "Select features to sweep for boundary mapping:",
        options=[
            f for f in features_for_boundary if f in st.session_state.model_features],
        default=[f for f in ["fico_score", "dti"]
                 if f in st.session_state.model_features],
        key="boundary_feature_select",
        help="Pick the levers you want to stress beyond typical ranges."
    )
    st.session_state.selected_boundary_features = selected_features

    if st.button("Run Boundary Mapping", disabled=not bool(st.session_state.selected_boundary_features)):
        with st.spinner("Performing boundary mapping..."):
            boundary_results = boundary_mapping(
                st.session_state.loaded_model,
                st.session_state.X_train_base,
                st.session_state.selected_boundary_features
            )
            st.session_state.boundary_results = boundary_results
        st.success("Boundary Mapping Complete!")

    if st.session_state.boundary_results:
        st.subheader("Extreme Value Boundary Maps")
        num_features = len(st.session_state.selected_boundary_features)
        if num_features > 0:
            rows = int(np.ceil(num_features / 3))
            cols = min(num_features, 3)
            fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
            axes = axes.flatten() if rows > 1 or cols > 1 else [axes]

            for i, feature in enumerate(st.session_state.selected_boundary_features):
                ax = axes[i]
                res = st.session_state.boundary_results[feature]
                ax.plot(res["sweep_values"],
                        res["predictions"], label="P(Default)")

                ax.axvspan(res["training_q01"], res["training_q99"],
                           alpha=0.2, label="Training Range (1-99% quantile)")

                if not np.isnan(res["cliff_value"]):
                    ax.axvline(res["cliff_value"], linestyle="--",
                               label=f'Cliff at {res["cliff_value"]:.2f}')

                if not res["extrapolation_stable"]:
                    ax.text(0.05, 0.95, "UNSTABLE Extrapolation",
                            transform=ax.transAxes, fontsize=10, va="top")

                ax.set_title(f"P(Default) vs. {feature}")
                ax.set_xlabel(feature)
                ax.set_ylabel("Predicted P(Default)")
                ax.legend(fontsize=8)
                ax.grid(True, linestyle=":", alpha=0.6)

            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Boundary Mapping Summary")
        summary_rows = []
        for feature, res in st.session_state.boundary_results.items():
            summary_rows.append({
                "feature": feature,
                "cliff_value": res["cliff_value"],
                "extrapolation": "stable" if res["extrapolation_stable"] else "UNSTABLE",
                "training_q01": res["training_q01"],
                "training_q99": res["training_q99"],
            })
        st.dataframe(pd.DataFrame(summary_rows))

        if st.session_state.learning_mode:
            render_interpretation_prompts([
                "Which features show cliffs near realistic borrower values (not just absurd extremes)?",
                "What would you put in a policy: validated range + action outside range (manual review / deny / re-score)?",
                "If extrapolation is unstable, do you treat that as a model limitation or a monitoring trigger?",
            ])

    st.markdown("---")
    st.markdown(
        "The extreme value boundary maps graphically illustrate how the model's predicted probability of default responds to "
        "variations in key input features, particularly beyond their typical training ranges."
    )
    st.markdown("For the Model Risk Manager, these plots are invaluable:")
    st.markdown(
        "-   **Cliff Detection:** A sudden, sharp change (a \"cliff\") in the predicted probability, indicated by the dashed line, "
        "suggests model instability. If our `dti` model, for instance, shows a cliff at `DTI = 0.7`, it means a small increase "
        "beyond this point can drastically change the prediction. This implies the model is very sensitive and potentially brittle "
        "around this threshold."
    )
    st.markdown(
        "-   **Extrapolation Stability:** The shaded region represents the typical training data range (1st to 99th percentile). "
        "How the model behaves *outside* this range is crucial. If the curve becomes erratic, flatlines unexpectedly, or shows "
        "sudden jumps (flagged as 'UNSTABLE Extrapolation'), it means the model is extrapolating unpredictably."
    )
    st.markdown(
        "Understanding these boundaries enables us to set appropriate use limits for the model (e.g., \"model validated only for "
        "FICO scores between 500-850, and DTI between 0-60%\"). This directly fulfills SR 11-7's requirement to document model "
        "limitations and potential failure modes, ensuring users are aware of when to exercise caution or seek additional review."
    )


# -----------------------------
# Page 4: Feature Sensitivity Analysis
# -----------------------------
def page_sensitivity():
    if st.session_state.X_train_base is None or st.session_state.loaded_model is None:
        st.warning(
            "Please go to '1. Introduction & Setup' and generate data/load model first.")
        return

    st.title("4. Feature Sensitivity Analysis")

    st.markdown(
        "Understanding which features significantly influence a model's output is critical for managing model risk. "
        "As a Model Risk Manager, this is akin to calculating \"duration and convexity\" for a bond portfolio; "
        "it tells me how sensitive the model's output is to small changes in each input."
    )
    st.markdown(
        "This test quantifies how much the model's default prediction changes in response to small perturbations "
        "(e.g., a 1% increase) in each input feature, while holding other features constant at their median."
    )

    perturbation_pct = st.slider(
        "Select Perturbation Percentage (e.g., 0.01 for 1% change):",
        min_value=0.001,
        max_value=0.1,
        value=st.session_state.perturbation_pct,
        step=0.001,
        format="%.3f",
        help="This is a small 'shock' to each feature to see how much predicted risk moves."
    )
    st.session_state.perturbation_pct = perturbation_pct

    if st.button("Run Sensitivity Analysis"):
        with st.spinner("Performing feature sensitivity analysis..."):
            X_sample = st.session_state.X_train_base.sample(
                n=min(1000, len(st.session_state.X_train_base)),
                random_state=42
            )
            sensitivity_results = feature_sensitivity(
                st.session_state.loaded_model,
                X_sample,
                st.session_state.model_features,
                perturbation_percent=perturbation_pct
            )
            st.session_state.sensitivity_results = sensitivity_results
        st.success("Feature Sensitivity Analysis Complete!")

    if st.session_state.sensitivity_results is not None:
        st.subheader("Feature Sensitivity Analysis Results")
        st.dataframe(st.session_state.sensitivity_results)

        # Plot: mean_abs_change
        st.subheader(
            f"Mean Absolute Change in P(Default) for {perturbation_pct*100:.1f}% Feature Perturbation")
        fig, ax = plt.subplots(figsize=(10, 5))
        vals = st.session_state.sensitivity_results["mean_abs_change"].values
        labels = st.session_state.sensitivity_results.index.tolist()
        ax.bar(labels, vals)
        ax.set_xlabel("Feature")
        ax.set_ylabel("Mean Absolute Change in P(Default)")
        ax.set_title("Mean Absolute Change in P(Default)")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", linestyle=":", alpha=0.7)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Plot: pct_affected_gt_0.01
        st.subheader(
            f"Percentage of Samples with >0.01 P(Default) Change for {perturbation_pct*100:.1f}% Feature Perturbation")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        vals2 = st.session_state.sensitivity_results["pct_affected_gt_0.01"].values
        ax2.bar(labels, vals2)
        ax2.set_xlabel("Feature")
        ax2.set_ylabel("Percentage of Samples Affected")
        ax2.set_title("Share of samples with >0.01 change in P(Default)")
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(axis="y", linestyle=":", alpha=0.7)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        if st.session_state.learning_mode:
            render_interpretation_prompts([
                "If one feature dominates sensitivity, what does that imply about data QA controls for that field?",
                "Would you require additional verification when that feature is near a decision boundary?",
                "If sensitivity is broad-based, does that suggest robustness—or a need for stronger monitoring across inputs?",
            ])

    st.markdown("---")
    st.markdown(
        "The feature sensitivity analysis provides a quantitative \"risk profile\" for our credit default model. "
        "The bar charts, showing both the mean absolute change in predicted default probability and the percentage "
        "of samples affected, immediately highlight which features exert the most influence on the model's predictions."
    )
    st.markdown("For the Model Risk Manager, these insights are crucial:")
    st.markdown(
        f"-   **Identifying Fragile Features:** Features with a high `mean_abs_change` or `pct_affected_gt_0.01` are the most "
        f"influential and potentially fragile."
    )
    st.markdown(
        "-   **Data Quality Focus:** Knowledge of these sensitive features directs our attention to data quality monitoring."
    )
    st.markdown(
        "-   **Model Complexity vs. Risk:** This analysis helps in understanding the trade-off."
    )


# -----------------------------
# Page 5: Adversarial Robustness Testing
# -----------------------------
def page_adversarial():
    if require_setup_guard():
        return

    st.title("5. Adversarial Robustness Testing")

    st.markdown(
        "In credit markets, borrowers are incentivized to present the best possible financial profile to secure a loan. "
        "As a Model Risk Manager, I need to know if our credit model can be \"gamed\" by strategic applicants who make minimal "
        "changes to their data to flip a borderline decision (e.g., from 'default' to 'no default')."
    )
    st.markdown(
        "This test involves iterating through borderline credit applications and for each, identifying the minimum percentage "
        "change to a single input feature that would flip the model's binary prediction."
    )

    adv_prob_threshold = st.slider(
        "Decision Probability Threshold (e.g., 0.5):",
        min_value=0.1, max_value=0.9,
        value=st.session_state.adv_prob_threshold,
        step=0.05
    )
    st.session_state.adv_prob_threshold = adv_prob_threshold

    adv_borderline_upper = st.slider(
        "Borderline Probability Range (Upper Bound) - for selecting samples predicted as default but not too confident:",
        min_value=adv_prob_threshold, max_value=0.99,
        value=st.session_state.adv_borderline_upper,
        step=0.01
    )
    st.session_state.adv_borderline_upper = adv_borderline_upper

    if st.button("Run Adversarial Test"):
        with st.spinner("Performing adversarial robustness test..."):
            X_adv_sample = st.session_state.df_credit_data[st.session_state.model_features].sample(
                n=min(2000, len(st.session_state.df_credit_data)),
                random_state=42
            )
            y_adv_sample = st.session_state.df_credit_data["default"].loc[X_adv_sample.index]

            adversarial_results_df = adversarial_test(
                st.session_state.loaded_model,
                X_adv_sample,
                y_adv_sample,
                st.session_state.model_features,
                prob_threshold=adv_prob_threshold,
                borderline_range=(adv_prob_threshold, adv_borderline_upper),
            )
            st.session_state.adversarial_results_df = adversarial_results_df
        st.success("Adversarial Robustness Test Complete!")

    if st.session_state.adversarial_results_df is not None:
        if st.session_state.adversarial_results_df.empty or "feature" not in st.session_state.adversarial_results_df.columns:
            st.warning(
                "No adversarial vulnerabilities found. This could mean:\n"
                "- No borderline cases exist in the specified probability range\n"
                "- No successful prediction flips were achieved with the tested perturbations\n"
                "- Try adjusting the probability threshold or borderline range parameters"
            )
        else:
            st.subheader("Adversarial Vulnerability Ranking")
            vuln_features = (
                st.session_state.adversarial_results_df.groupby("feature")[
                    "delta_pct"]
                .mean()
                .sort_values(ascending=True)
            )
            st.markdown("(Lower delta_pct = easier to game)")
            st.dataframe(vuln_features)

            st.subheader(
                "Features Ranked by Minimum % Perturbation to Flip Prediction (Ease of Gaming)")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(vuln_features.index.tolist(), vuln_features.values)
            ax.set_xlabel("Feature")
            ax.set_ylabel("Mean Minimum % Change to Flip")
            ax.set_title("Ease of gaming by feature (lower is worse)")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(axis="y", linestyle=":", alpha=0.7)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            if st.session_state.learning_mode:
                render_interpretation_prompts([
                    "If a feature is easy to game, do you: (a) verify it more, (b) reduce reliance on it, or (c) change policy near the boundary?",
                    "Which gaming behaviors are acceptable (financial improvement) vs unacceptable (misreporting), and how does governance handle each?",
                    "Does your institution need a 'second look' rule for borderline approvals to reduce strategic flips?",
                ])

    st.markdown("---")
    st.markdown(
        "The adversarial robustness test identifies how easily a model's prediction for a \"borderline\" credit application can be "
        "manipulated by small, strategic changes to input features."
    )


# -----------------------------
# Page 6: Concept Drift Detection
# -----------------------------
def page_concept_drift():
    if require_setup_guard():
        return

    st.title("6. Concept Drift Detection")

    st.markdown(
        "Models, particularly in dynamic financial environments, do not remain accurate indefinitely. The underlying relationships "
        "between credit features and default can change over time—a phenomenon known as concept drift."
    )
    st.markdown(
        "This test simulates rolling performance monitoring, calculating AUC over sequential time windows and comparing it to a baseline. "
        "We also introduce the Population Stability Index (PSI) to detect shifts in input distributions, which can often precede performance degradation."
    )

    # --- PRESERVE FORMULAE VERBATIM ---
    st.markdown(r"**Mathematical Formulation:**")
    st.markdown(
        r"The degradation ($\Delta_t$) in rolling AUC performance is calculated as:")
    st.markdown(r"""
$$
\Delta_t = AUC_0 - AUC_t
$$
""")
    st.markdown(
        r"where $AUC_0$ is the baseline AUC on a validation set at deployment time (or a stable reference period).")
    st.markdown(
        r"where $AUC_t$ is the rolling AUC calculated over a recent time window.")

    st.markdown(r"**Alert Levels for $\Delta_t$:**")
    st.markdown(r"-   **Green:** $\Delta_t < 0.03$. Normal variation.")
    st.markdown(
        r"-   **Yellow:** $0.03 \le \Delta_t < 0.07$. Investigate. Check for data quality issues or genuine drift.")
    st.markdown(
        r"-   **Red:** $\Delta_t \ge 0.07$. Freeze model for new decisions. Trigger revalidation and possible retraining.")

    st.markdown(r"**Population Stability Index (PSI):**")
    st.markdown(
        r"PSI can supplement AUC monitoring by detecting input distribution shifts even before output performance degrades.")
    st.markdown(r"""
$$
PSI = \sum_{i=1}^{B} \left( P_i^{\text{actual}} - P_i^{\text{expected}} \right) \ln \left( \frac{P_i^{\text{actual}}}{P_i^{\text{expected}}} \right)
$$
""")

    st.markdown(
        r"Where $B$ is the number of bins partitioning the score distribution.")
    st.markdown(
        r"Where $P_i^{\text{actual}}$ is the percentage of observations in bin $i$ for the actual population.")
    st.markdown(
        r"Where $P_i^{\text{expected}}$ is the percentage of observations in bin $i$ for the expected population.")
    st.markdown(r"Where $PSI > 0.25$ indicates a significant population shift.")

    # Add a render-friendly PSI (without removing original)
    with st.expander("PSI formula (standard version)", expanded=False):
        st.markdown(r"""
    $$
    PSI = \sum_{i=1}^{B} (P_i^{actual} - P_i^{expected}) \ln \left( \frac{P_i^{actual}}{P_i^{expected}} \right)
    $$
    """)
        st.caption(
            "Note: This is added for readability; the original formula block above is preserved exactly as in your file.")

    cd_window_size = st.slider(
        "Rolling Window Size (Days):",
        min_value=30, max_value=365,
        value=st.session_state.cd_window_size,
        step=30
    )
    st.session_state.cd_window_size = cd_window_size

    cd_step_size = st.slider(
        "Window Step Size (Days):",
        min_value=7, max_value=90,
        value=st.session_state.cd_step_size,
        step=7
    )
    st.session_state.cd_step_size = cd_step_size

    cd_yellow_thresh = st.slider(
        "Yellow Alert Threshold (Delta AUC):",
        min_value=0.01, max_value=0.1,
        value=st.session_state.cd_yellow_thresh,
        step=0.005, format="%.3f"
    )
    st.session_state.cd_yellow_thresh = cd_yellow_thresh

    cd_red_thresh = st.slider(
        "Red Alert Threshold (Delta AUC):",
        min_value=0.03, max_value=0.15,
        value=st.session_state.cd_red_thresh,
        step=0.005, format="%.3f"
    )
    st.session_state.cd_red_thresh = cd_red_thresh

    cd_psi_thresh = st.slider(
        "PSI Alert Threshold:",
        min_value=0.1, max_value=0.5,
        value=st.session_state.cd_psi_thresh,
        step=0.05, format="%.2f"
    )
    st.session_state.cd_psi_thresh = cd_psi_thresh

    if st.button("Run Concept Drift Monitor"):
        with st.spinner("Performing concept drift detection..."):
            drift_results_df, baseline_auc = concept_drift_monitor(
                st.session_state.loaded_model,
                st.session_state.df_credit_data,
                st.session_state.df_credit_data["default"],
                st.session_state.df_credit_data["date"],
                st.session_state.model_features,
                window_size_days=cd_window_size,
                step_size_days=cd_step_size,
                baseline_start_date="2015-01-01",
                baseline_end_date="2020-01-01",
                alert_threshold_yellow=cd_yellow_thresh,
                alert_threshold_red=cd_red_thresh,
                psi_alert_threshold=cd_psi_thresh
            )
            st.session_state.drift_results_df = drift_results_df
            st.session_state.baseline_auc = baseline_auc
        st.success("Concept Drift Monitoring Complete!")

    if st.session_state.drift_results_df is not None and not st.session_state.drift_results_df.empty:
        st.subheader(
            f"Concept Drift Monitor Results (Baseline AUC: {st.session_state.baseline_auc:.4f})")
        st.dataframe(
            st.session_state.drift_results_df[
                ["window_start", "window_end", "auc", "auc_degradation",
                    "alert_status", "psi_fico_score", "psi_dti"]
            ]
        )

        st.subheader("Rolling AUC Performance Over Time")
        fig_auc, ax_auc = plt.subplots(figsize=(14, 6))
        x = pd.to_datetime(st.session_state.drift_results_df["window_start"])
        y = st.session_state.drift_results_df["auc"].values
        ax_auc.plot(x, y, marker="o", markersize=4)

        ax_auc.axhline(
            y=st.session_state.baseline_auc,
            linestyle="--",
            label=f"Baseline AUC ({st.session_state.baseline_auc:.3f})"
        )

        ax_auc.axhspan(st.session_state.baseline_auc - cd_yellow_thresh, st.session_state.baseline_auc,
                       alpha=0.1, label=f"Green Zone (Degradation < {cd_yellow_thresh})")
        ax_auc.axhspan(st.session_state.baseline_auc - cd_red_thresh, st.session_state.baseline_auc -
                       cd_yellow_thresh, alpha=0.1, label=f"Yellow Zone ({cd_yellow_thresh} <= Degradation < {cd_red_thresh})")
        ax_auc.axhspan(0, st.session_state.baseline_auc - cd_red_thresh,
                       alpha=0.1, label=f"Red Zone (Degradation >= {cd_red_thresh})")

        ax_auc.set_title(
            "Concept Drift Monitor: Rolling AUC Performance Over Time")
        ax_auc.set_xlabel("Window Start Date")
        ax_auc.set_ylabel("AUC")
        ax_auc.set_ylim(0, 1)
        ax_auc.grid(True, linestyle=":", alpha=0.6)
        ax_auc.legend()
        fig_auc.tight_layout()
        st.pyplot(fig_auc)
        plt.close(fig_auc)

        if "psi_fico_score" in st.session_state.drift_results_df.columns:
            st.subheader(
                "Population Stability Index (PSI) for FICO Score Over Time")
            fig_psi, ax_psi = plt.subplots(figsize=(14, 4))
            ax_psi.plot(
                x,
                st.session_state.drift_results_df["psi_fico_score"].values,
                marker="x",
                markersize=4
            )
            ax_psi.axhline(y=cd_psi_thresh, linestyle="--",
                           label=f"PSI Alert Threshold ({cd_psi_thresh:.2f})")
            ax_psi.set_title("PSI (FICO Score) Over Time")
            ax_psi.set_xlabel("Window Start Date")
            ax_psi.set_ylabel("PSI Value")
            ax_psi.grid(True, linestyle=":", alpha=0.6)
            ax_psi.legend()
            fig_psi.tight_layout()
            st.pyplot(fig_psi)
            plt.close(fig_psi)

        if st.session_state.learning_mode:
            render_interpretation_prompts([
                "When you see Yellow alerts, what investigation steps come first (data quality vs macro regime vs policy change)?",
                "If PSI spikes but AUC doesn’t drop yet, do you treat PSI as an early-warning trigger?",
                "What is your 'stop-the-line' rule (Red) and who must sign off before resuming use?",
            ])
    else:
        st.info("No concept drift monitoring results to display. Run the test first.")

    st.markdown("---")
    st.markdown(
        "The concept drift monitor visualizes the model's rolling AUC performance over time, alongside a baseline AUC and clearly marked "
        "alert zones (Green, Yellow, Red). Additionally, the PSI plots for key features like `fico_score` and `dti` show shifts in input "
        "data distributions."
    )


# -----------------------------
# Page 7: SR 11-7 Stress Test Report
# -----------------------------
def page_report():
    can_generate_report = (
        st.session_state.distribution_results is not None and
        st.session_state.boundary_results is not None and
        st.session_state.sensitivity_results is not None and
        st.session_state.adversarial_results_df is not None and
        st.session_state.drift_results_df is not None
    )

    if not can_generate_report:
        st.warning(
            "Please run all preceding stress tests (2-6) to generate the comprehensive SR 11-7 report.")
        return

    st.title("7. Compile SR 11-7 Aligned Stress Test Report")

    st.markdown(
        "After conducting a thorough suite of stress tests, the final and most crucial step for a Model Risk Manager is to "
        "consolidate all findings into a structured SR 11-7 report. This document is the ultimate deliverable, serving as "
        "the model's \"prospectus\" for internal stakeholders and regulators."
    )
    st.markdown(
        "The report aggregates findings from distribution shift, boundary mapping, feature sensitivity, adversarial robustness, "
        "and concept drift tests."
    )

    if st.button("Generate SR 11-7 Report"):
        with st.spinner("Compiling SR 11-7 Stress Test Report..."):
            sr117_report = compile_stress_report(
                st.session_state.distribution_results,
                st.session_state.boundary_results,
                st.session_state.sensitivity_results,
                st.session_state.adversarial_results_df,
                st.session_state.drift_results_df,
            )
            st.session_state.sr117_report = sr117_report
        st.success("SR 11-7 Report Compiled!")

    if st.session_state.sr117_report is None:
        st.info("Generate the report to view it here and download it.")
        return

    report = st.session_state.sr117_report

    with st.container(border=True):
        st.subheader("SR 11-7 STRESS TEST REPORT")
        st.markdown("---")
        st.markdown(f"**Model:** `{report['model_name']}`")
        st.markdown(f"**Report Date:** `{report['report_date']}`")
        st.markdown(
            f"**Overall Assessment:** `{report['overall_assessment']}`")
        st.markdown("---")

        st.markdown("### FINDINGS:")
        for test, finding in report["findings"].items():
            st.markdown(
                f"#### {test.replace('_', ' ').upper()} [{finding['severity']}]:")
            st.markdown(f"- **Recommendation:** {finding['recommendation']}")

            if test == "distribution_shift":
                st.markdown(
                    f"  - Max AUC Degradation: {finding.get('max_auc_degradation', 'N/A'):.3f} (Worst: {finding.get('worst_regime_pair', 'N/A')})")
            elif test == "extreme_values":
                st.markdown(
                    f"  - Unstable Features: {', '.join(finding.get('unstable_features', [])) if finding.get('unstable_features') else 'None'}")
            elif test == "sensitivity":
                st.markdown(
                    f"  - Most Sensitive Feature: {finding.get('most_sensitive_feature', 'N/A')} (Max Sensitivity: {finding.get('max_sensitivity', np.nan):.3f})")
            elif test == "adversarial":
                st.markdown(
                    f"  - Most Gameable Feature: {finding.get('most_gameable_feature', 'N/A')} (Min Perturbation to Flip: {finding.get('min_perturbation_to_flip', np.nan):.2%})")
            elif test == "concept_drift":
                st.markdown(
                    f"  - Drift Alerts Triggered: {finding.get('drift_alerts', 0)}")

            st.markdown("")

        st.markdown("### USE BOUNDARIES:")
        if report["use_boundaries"]:
            for boundary in report["use_boundaries"]:
                st.markdown(f" - {boundary}")
        else:
            st.markdown(
                " - No specific use boundaries recommended at this time, continue standard monitoring.")

        st.markdown("### SIGN-OFF REQUIRED:")
        for signoff in report["sign_off_required"]:
            st.markdown(f" {signoff}")

    st.markdown("---")

    # Create formatted text report (kept very close to original behavior)
    report_text = f"""SR 11-7 STRESS TEST REPORT
{'='*80}

Model: {report['model_name']}
Report Date: {report['report_date']}
Overall Assessment: {report['overall_assessment']}

{'='*80}

FINDINGS:
{'-'*80}

"""
    for test, finding in report["findings"].items():
        report_text += f"\n{test.replace('_', ' ').upper()} [{finding['severity']}]:\n"
        report_text += f"  Recommendation: {finding['recommendation']}\n"

        if test == "distribution_shift":
            max_deg = finding.get("max_auc_degradation", "N/A")
            formatted_max_deg = f"{max_deg:.3f}" if isinstance(
                max_deg, (int, float)) else str(max_deg)
            report_text += f"  Max AUC Degradation: {formatted_max_deg}\n"
            report_text += f"  Worst Regime Pair: {finding.get('worst_regime_pair', 'N/A')}\n"
        elif test == "extreme_values":
            unstable = ", ".join(finding.get("unstable_features", [])) if finding.get(
                "unstable_features") else "None"
            report_text += f"  Unstable Features: {unstable}\n"
        elif test == "sensitivity":
            report_text += f"  Most Sensitive Feature: {finding.get('most_sensitive_feature', 'N/A')}\n"
            max_sens = finding.get("max_sensitivity", np.nan)
            if not np.isnan(max_sens):
                report_text += f"  Max Sensitivity: {max_sens:.3f}\n"
        elif test == "adversarial":
            report_text += f"  Most Gameable Feature: {finding.get('most_gameable_feature', 'N/A')}\n"
            min_pert = finding.get("min_perturbation_to_flip", np.nan)
            if not np.isnan(min_pert):
                report_text += f"  Min Perturbation to Flip: {min_pert:.2%}\n"
        elif test == "concept_drift":
            report_text += f"  Drift Alerts Triggered: {finding.get('drift_alerts', 0)}\n"

        report_text += "\n"

    report_text += f"\n{'-'*80}\n\nUSE BOUNDARIES:\n{'-'*80}\n\n"
    if report["use_boundaries"]:
        for boundary in report["use_boundaries"]:
            report_text += f" - {boundary}\n"
    else:
        report_text += " - No specific use boundaries recommended at this time, continue standard monitoring.\n"

    report_text += f"\n{'-'*80}\n\nSIGN-OFF REQUIRED:\n{'-'*80}\n\n"
    for signoff in report["sign_off_required"]:
        report_text += f" {signoff}\n"

    report_text += f"\n{'='*80}\nEnd of Report\n{'='*80}\n"

    st.download_button(
        label="📥 Download SR 11-7 Report (TXT)",
        data=report_text,
        file_name=f"SR_11-7_Stress_Test_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        help="Download the complete SR 11-7 stress test report as a text file",
    )

    if st.session_state.learning_mode:
        render_interpretation_prompts([
            "If this report went to a Model Risk Committee, what 3 decisions should it enable immediately?",
            "What would you require before re-approval: retraining, tighter boundaries, more monitoring, or process controls?",
        ])


# -----------------------------
# Router
# -----------------------------
if page_selection == "1. Introduction & Setup":
    page_intro_setup()
elif page_selection == "2. Distribution Shift Testing":
    page_distribution_shift()
elif page_selection == "3. Extreme Value Boundary Mapping":
    page_boundary_mapping()
elif page_selection == "4. Feature Sensitivity Analysis":
    page_sensitivity()
elif page_selection == "5. Adversarial Robustness Testing":
    page_adversarial()
elif page_selection == "6. Concept Drift Detection":
    page_concept_drift()
elif page_selection == "7. SR 11-7 Stress Test Report":
    page_report()


# -----------------------------
# License (DO NOT REMOVE)
# -----------------------------
st.caption('''
---
## QuantUniversity License

© QuantUniversity 2026  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
''')
