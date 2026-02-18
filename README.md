# 🏛️ QuLab: Lab 37: SR 11-7 Model Stress Testing for Credit Default

![QuantUniversity Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

## Project Title: SR 11-7 Model Stress Testing for Credit Default

## Project Description

This Streamlit application, "QuLab: Lab 37," provides a comprehensive suite of tools for stress testing a credit default classification model, in line with the Federal Reserve's SR 11-7 guidance. Designed for a Model Risk Manager or Quantitative Analyst persona, the application focuses not on building a model, but on systematically understanding and quantifying its vulnerabilities under various adverse conditions.

The objective is to identify how a pre-trained credit default model (`xgboost_credit.pkl`) behaves when exposed to:
1.  **Distribution Shifts**: Changes in economic regimes (e.g., recessions, crises).
2.  **Extreme Value Inputs**: Feature values far outside the model's training distribution.
3.  **Feature Sensitivity**: Small perturbations in individual features.
4.  **Adversarial Manipulations**: Strategic attempts by applicants to "game" the model.
5.  **Concept Drift**: Degradation of performance over time due to evolving underlying relationships.

The application culminates in generating a structured SR 11-7 Stress Test Report, providing a clear "prospectus" of the model's strengths, weaknesses, recommended usage boundaries, and necessary mitigation strategies. This process is crucial for preventing costly errors, maintaining regulatory compliance, and making sound capital allocation and lending decisions.

## Features

The application is structured into several interactive pages, each dedicated to a specific stress testing methodology:

1.  **Introduction & Setup**:
    *   Explains the project's context, persona, and regulatory relevance (SR 11-7).
    *   Allows users to generate synthetic credit default data simulating various economic regimes.
    *   Loads a pre-trained `xgboost_credit.pkl` model for stress testing.

2.  **Distribution Shift Testing**:
    *   **Purpose**: Evaluates model performance (AUC) under different simulated economic regimes (e.g., expansion, recession, crisis).
    *   **Functionality**: Calculates and visualizes AUC degradation ($\Delta AUC$) relative to the training regime, with alert thresholds (Green, Yellow, Red) for severity.
    *   **Relevance**: Addresses how model reliability is compromised by changing economic conditions.

3.  **Extreme Value Boundary Mapping**:
    *   **Purpose**: Assesses model behavior when input features move far outside their typical training ranges.
    *   **Functionality**: Sweeps selected feature values to observe changes in predicted probability of default, identifying "cliffs" (sudden changes) and unstable extrapolation.
    *   **Relevance**: Identifies unpredictable model behavior in rare, extreme cases, informing model use limits.

4.  **Feature Sensitivity Analysis**:
    *   **Purpose**: Quantifies how much the model's prediction changes in response to small perturbations in each input feature.
    *   **Functionality**: Calculates the mean absolute change in P(Default) and the percentage of samples affected for a user-defined perturbation percentage.
    *   **Relevance**: Pinpoints influential and potentially fragile features, guiding data monitoring and internal challenge efforts.

5.  **Adversarial Robustness Testing**:
    *   **Purpose**: Determines if the model can be "gamed" by strategic applicants making minimal changes to flip a borderline decision.
    *   **Functionality**: Identifies the minimum percentage change to a single feature required to flip a prediction for borderline cases, ranking features by "ease of gaming."
    *   **Relevance**: Quantifies vulnerability to strategic manipulation, informing risk mitigation strategies and additional verification steps.

6.  **Concept Drift Detection**:
    *   **Purpose**: Monitors model performance degradation over time as underlying relationships evolve.
    *   **Functionality**: Simulates rolling AUC monitoring with alert thresholds (Green, Yellow, Red) and incorporates Population Stability Index (PSI) to detect input data distribution shifts.
    *   **Relevance**: Provides an early warning system for model decay, triggering investigations, recalibrations, or retraining before significant financial risks accumulate.

7.  **SR 11-7 Stress Test Report**:
    *   **Purpose**: Compiles all findings into a structured, regulatory-aligned report.
    *   **Functionality**: Aggregates severity ratings, recommendations, and use boundaries from all stress tests into a final summary.
    *   **Relevance**: The ultimate deliverable for internal stakeholders and regulators, demonstrating rigorous model governance and compliance.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/quolab-lab37-stress-testing.git
    cd quolab-lab37-stress-testing
    ```
    *(Note: Replace `your-username/quolab-lab37-stress-testing.git` with the actual repository URL if available.)*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    *   **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install dependencies:**
    Create a `requirements.txt` file in the project root with the following content:

    ```
    streamlit>=1.0.0
    pandas>=1.3.0
    numpy>=1.21.0
    matplotlib>=3.4.0
    seaborn>=0.11.0
    scikit-learn>=0.24.0
    joblib>=1.0.0
    xgboost>=1.4.0
    ```

    Then install them:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Streamlit application:**
    Ensure your virtual environment is activated and you are in the project's root directory.
    ```bash
    streamlit run app.py
    ```

2.  **Access the Application:**
    The application will open in your web browser, typically at `http://localhost:8501`.

3.  **Navigate and Interact:**
    *   Use the sidebar on the left to navigate through the different stress testing pages.
    *   Start with the "1. Introduction & Setup" page to generate synthetic data and load the pre-trained model. This is a prerequisite for running other tests.
    *   Proceed sequentially through the stress testing pages (2-6), running the tests as prompted.
    *   Finally, generate the comprehensive SR 11-7 report on page 7.

## Project Structure

```
.
├── app.py                     # Main Streamlit application script
├── source.py                  # Contains all helper functions for data generation, model training/loading, and stress tests
├── requirements.txt           # List of Python dependencies
├── README.md                  # This README file
└── xgboost_credit.pkl         # (Generated upon first run) Pre-trained XGBoost model for credit default
```

*   `app.py`: This is the core Streamlit file that defines the UI, manages session state, and orchestrates calls to the functions defined in `source.py`.
*   `source.py`: This file encapsulates all the backend logic, including `generate_synthetic_credit_data`, `train_and_save_model`, `distribution_shift_test`, `boundary_mapping`, `feature_sensitivity`, `adversarial_test`, `concept_drift_monitor`, and `compile_stress_report`. This modular design keeps the Streamlit app clean and focused on presentation.

## Technology Stack

*   **Frontend/UI**: [Streamlit](https://streamlit.io/)
*   **Backend Logic**: [Python 3](https://www.python.org/)
*   **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
*   **Machine Learning**: [Scikit-learn](https://scikit-learn.org/) (for metrics), [XGBoost](https://xgboost.readthedocs.io/) (for the credit model)
*   **Model Persistence**: [Joblib](https://joblib.readthedocs.io/en/latest/)
*   **Data Visualization**: [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)

## Contributing

Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
5.  Push to the branch (`git push origin feature/AmazingFeature`).
6.  Open a Pull Request.

Please ensure your code adheres to good practices and includes relevant documentation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
*(Note: You will need to create a `LICENSE` file in your repository with the MIT License text.)*

## Contact

For any questions or feedback, please reach out:

*   **QuantUniversity** - [Website](https://www.quantuniversity.com/)
*   **Project Issues**: [GitHub Issues](https://github.com/your-username/quolab-lab37-stress-testing/issues)
*   **Your Name/Email**: [Your GitHub Profile](https://github.com/your-username) or `your.email@example.com`
