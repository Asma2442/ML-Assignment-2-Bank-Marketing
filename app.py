
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bank Marketing Classifier",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# APPLICATION TITLE
# ============================================================

st.title("Bank Marketing Classification")

st.write(
    "Predict whether a bank customer will subscribe to a term deposit."
)


# ============================================================
# DATASET COLUMNS
# ============================================================

FEATURE_COLUMNS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome"
]

TARGET_COLUMN = "y"


# ============================================================
# MODEL FILES
# ============================================================

MODEL_FILES = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "K-Nearest Neighbors": "models/k-nearest_neighbors.pkl",
    "Gaussian Naive Bayes": "models/gaussian_naive_bayes.pkl",
    "Random Forest": "models/random_forest.pkl"
}


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

@st.cache_resource
def load_models():

    models = {}

    for model_name, model_path in MODEL_FILES.items():

        models[model_name] = joblib.load(model_path)

    return models


models = load_models()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Model Configuration")


uploaded_file = st.sidebar.file_uploader(
    "Upload test CSV",
    type=["csv"]
)


selected_model = st.sidebar.selectbox(
    "Select Model",
    list(MODEL_FILES.keys())
)


# ============================================================
# SELECTED MODEL
# ============================================================

st.subheader("Selected Model")

st.info(
    f"Current model: **{selected_model}**"
)


# ============================================================
# PROCESS UPLOADED CSV
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # READ CSV
    # --------------------------------------------------------

    test_data = pd.read_csv(uploaded_file)


    # --------------------------------------------------------
    # DISPLAY DATASET INFORMATION
    # --------------------------------------------------------

    st.subheader("Uploaded Test Data")

    st.write(
        f"Rows: {test_data.shape[0]} | "
        f"Columns: {test_data.shape[1]}"
    )

    st.dataframe(
        test_data.head(10),
        use_container_width=True
    )


    # --------------------------------------------------------
    # VALIDATE REQUIRED FEATURES
    # --------------------------------------------------------

    missing_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in test_data.columns
    ]

    if missing_features:

        st.error(
            "The uploaded CSV is missing the following "
            f"required features: {missing_features}"
        )

        st.stop()


    # --------------------------------------------------------
    # SEPARATE FEATURES AND TARGET
    # --------------------------------------------------------

    X_uploaded = test_data[FEATURE_COLUMNS]

    has_target = TARGET_COLUMN in test_data.columns


    if has_target:

        y_uploaded = test_data[TARGET_COLUMN]


    # ========================================================
    # PREDICTION
    # ========================================================

    selected_pipeline = models[selected_model]


    predictions = selected_pipeline.predict(
        X_uploaded
    )


    probabilities = selected_pipeline.predict_proba(
        X_uploaded
    )[:, 1]


    # ========================================================
    # DISPLAY PREDICTIONS
    # ========================================================

    prediction_output = X_uploaded.copy()

    prediction_output["Predicted"] = predictions

    prediction_output["Predicted Label"] = np.where(
        predictions == 1,
        "Yes",
        "No"
    )


    st.subheader("Predictions")

    st.dataframe(
        prediction_output,
        use_container_width=True
    )


    # ========================================================
    # MODEL EVALUATION
    # ========================================================

    if has_target:

        # ====================================================
        # MODEL COMPARISON
        # ====================================================

        st.subheader("Model Comparison")

        comparison_results = []


        for model_name, model_pipeline in models.items():

            comparison_predictions = model_pipeline.predict(
                X_uploaded
            )

            comparison_probabilities = (
                model_pipeline.predict_proba(
                    X_uploaded
                )[:, 1]
            )


            comparison_accuracy = accuracy_score(
                y_uploaded,
                comparison_predictions
            )


            comparison_auc = roc_auc_score(
                y_uploaded,
                comparison_probabilities
            )


            comparison_precision = precision_score(
                y_uploaded,
                comparison_predictions,
                zero_division=0
            )


            comparison_recall = recall_score(
                y_uploaded,
                comparison_predictions,
                zero_division=0
            )


            comparison_f1 = f1_score(
                y_uploaded,
                comparison_predictions,
                zero_division=0
            )


            comparison_mcc = matthews_corrcoef(
                y_uploaded,
                comparison_predictions
            )


            comparison_results.append(
                {
                    "Model": model_name,
                    "Accuracy": comparison_accuracy,
                    "AUC": comparison_auc,
                    "Precision": comparison_precision,
                    "Recall": comparison_recall,
                    "F1 Score": comparison_f1,
                    "MCC": comparison_mcc
                }
            )


        # ----------------------------------------------------
        # CREATE COMPARISON DATAFRAME
        # ----------------------------------------------------

        comparison_df = pd.DataFrame(
            comparison_results
        )


        metric_columns = [
            "Accuracy",
            "AUC",
            "Precision",
            "Recall",
            "F1 Score",
            "MCC"
        ]


        comparison_df[metric_columns] = (
            comparison_df[metric_columns].round(4)
        )


        # ----------------------------------------------------
        # DISPLAY COMPARISON TABLE
        # ----------------------------------------------------

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )


        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        best_model_index = comparison_df[
            "F1 Score"
        ].idxmax()


        best_model_name = comparison_df.loc[
            best_model_index,
            "Model"
        ]


        best_f1 = comparison_df.loc[
            best_model_index,
            "F1 Score"
        ]


        st.success(
            f"Best model based on F1 Score: "
            f"**{best_model_name}** "
            f"({best_f1:.4f})"
        )


        # ====================================================
        # SELECTED MODEL EVALUATION
        # ====================================================

        st.subheader("Model Evaluation")


        # ----------------------------------------------------
        # CALCULATE METRICS
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_uploaded,
            predictions
        )


        auc = roc_auc_score(
            y_uploaded,
            probabilities
        )


        precision = precision_score(
            y_uploaded,
            predictions,
            zero_division=0
        )


        recall = recall_score(
            y_uploaded,
            predictions,
            zero_division=0
        )


        f1 = f1_score(
            y_uploaded,
            predictions,
            zero_division=0
        )


        mcc = matthews_corrcoef(
            y_uploaded,
            predictions
        )


        # ----------------------------------------------------
        # DISPLAY METRICS
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(3)


        col1.metric(
            "Accuracy",
            f"{accuracy:.4f}"
        )


        col2.metric(
            "AUC",
            f"{auc:.4f}"
        )


        col3.metric(
            "Precision",
            f"{precision:.4f}"
        )


        col4, col5, col6 = st.columns(3)


        col4.metric(
            "Recall",
            f"{recall:.4f}"
        )


        col5.metric(
            "F1 Score",
            f"{f1:.4f}"
        )


        col6.metric(
            "MCC",
            f"{mcc:.4f}"
        )


        # ====================================================
        # CONFUSION MATRIX
        # ====================================================

        st.subheader("Confusion Matrix")


        cm = confusion_matrix(
            y_uploaded,
            predictions
        )


        fig, ax = plt.subplots(
            figsize=(6, 5)
        )


        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=["No", "Yes"],
            yticklabels=["No", "Yes"],
            ax=ax
        )


        ax.set_xlabel("Predicted")

        ax.set_ylabel("Actual")

        ax.set_title(
            f"{selected_model} - Confusion Matrix"
        )


        st.pyplot(fig)


        # ====================================================
        # CLASSIFICATION REPORT
        # ====================================================

        st.subheader("Classification Report")


        report = classification_report(
            y_uploaded,
            predictions,
            target_names=["No", "Yes"],
            output_dict=True,
            zero_division=0
        )


        report_df = pd.DataFrame(
            report
        ).transpose()


        st.dataframe(
            report_df.round(4),
            use_container_width=True
        )


    # ========================================================
    # PREDICTION-ONLY MODE
    # ========================================================

    else:

        st.info(
            "The uploaded CSV does not contain the target "
            "column 'y'. Predictions are displayed, but "
            "evaluation metrics cannot be calculated."
        )
