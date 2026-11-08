
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🫘",
    layout="wide"
)

st.title("Dry Bean Classification Dashboard")

st.write(
    "Choose a saved classifier, explore the test data "
    "and review its classification performance."
)

# --------------------------------------------------
# FILE PATHS AND MODEL CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

TARGET = "Class"
MODEL_DIR = BASE_DIR / "model"
DEFAULT_CSV = BASE_DIR / "test_data.csv"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_nb.joblib",
    "SVM": "svm.joblib",
}

# --------------------------------------------------
# LOAD SAVED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model(path_string):
    return joblib.load(path_string)

# --------------------------------------------------
# CSV SELECTION
# --------------------------------------------------

st.subheader("Test Dataset")

uploaded_file = st.file_uploader(
    "Upload another CSV file (optional)",
    type=["csv"]
)

try:
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.success("Your uploaded CSV file is being used.")

    else:
        if not DEFAULT_CSV.exists():
            st.error(
                "Default test_data.csv was not found. "
                "Please check that it exists in your GitHub repository."
            )
            st.stop()

        data = pd.read_csv(DEFAULT_CSV)
        st.info("Automatically loaded the default test_data.csv.")

    # --------------------------------------------------
    # DATA PREVIEW
    # --------------------------------------------------

    st.subheader("Data Preview")
    st.dataframe(data.head(), use_container_width=True)

    st.write(f"Total rows: {data.shape[0]}")
    st.write(f"Total columns: {data.shape[1]}")

    # --------------------------------------------------
    # MODEL SELECTION
    # --------------------------------------------------

    st.subheader("Select a Saved Model")

    selected_name = st.selectbox(
        "Choose a classifier",
        list(MODEL_FILES.keys())
    )

    model_path = MODEL_DIR / MODEL_FILES[selected_name]

    if not model_path.exists():
        st.error(f"Saved model not found: {model_path}")
        st.stop()

    model = load_model(str(model_path))

    # --------------------------------------------------
    # FEATURE VALIDATION
    # --------------------------------------------------

    expected_features = list(model.feature_names_in_)

    missing_features = [
        column for column in expected_features
        if column not in data.columns
    ]

    if missing_features:
        st.error(
            "CSV is missing required feature columns: "
            + ", ".join(missing_features)
        )
        st.stop()

    X_upload = data[expected_features]

    # --------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------

    predictions = model.predict(X_upload)

    st.subheader("Predictions")

    prediction_table = pd.DataFrame({
        "Predicted_Class": predictions
    })

    if TARGET in data.columns:
        prediction_table.insert(
            0,
            "Actual_Class",
            data[TARGET].values
        )

    st.dataframe(prediction_table, use_container_width=True)

    # --------------------------------------------------
    # EVALUATION METRICS
    # --------------------------------------------------

    if TARGET not in data.columns:

        st.warning(
            "The CSV does not contain the 'Class' target column. "
            "Predictions are available, but evaluation metrics "
            "cannot be calculated."
        )

    else:

        y_true = data[TARGET]
        y_probability = model.predict_proba(X_upload)

        metrics = {
            "Accuracy": accuracy_score(y_true, predictions),

            "AUC (weighted OVR)": roc_auc_score(
                y_true,
                y_probability,
                labels=model.classes_,
                multi_class="ovr",
                average="weighted"
            ),

            "Precision (weighted)": precision_score(
                y_true,
                predictions,
                average="weighted",
                zero_division=0
            ),

            "Recall (weighted)": recall_score(
                y_true,
                predictions,
                average="weighted",
                zero_division=0
            ),

            "F1 (weighted)": f1_score(
                y_true,
                predictions,
                average="weighted",
                zero_division=0
            ),

            "MCC": matthews_corrcoef(
                y_true,
                predictions
            ),
        }

        st.subheader("Evaluation Metrics")

        cols = st.columns(3)

        for i, (metric_name, metric_value) in enumerate(metrics.items()):
            cols[i % 3].metric(
                metric_name,
                f"{metric_value:.4f}"
            )

        # --------------------------------------------------
        # CONFUSION MATRIX
        # --------------------------------------------------

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y_true,
            predictions,
            labels=model.classes_
        )

        fig, ax = plt.subplots(figsize=(8, 6))

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=model.classes_
        )

        display.plot(
            ax=ax,
            xticks_rotation=45,
            colorbar=False
        )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # --------------------------------------------------
        # CLASSIFICATION REPORT
        # --------------------------------------------------

        st.subheader("Classification Report")

        report = classification_report(
            y_true,
            predictions,
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(
            report_df,
            use_container_width=True
        )

except Exception as exc:

    st.error(f"Could not process the CSV or model: {exc}")
