from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)

st.set_page_config(page_title="Dry Bean Classification", page_icon="🫘", layout="wide")
st.title("Dry Bean Classification Dashboard")
st.write("Upload a labelled CSV, choose a saved classifier and review its test-set performance.")

TARGET = "Class"
MODEL_DIR = Path("model")
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_nb.joblib",
    "SVM": "svm.joblib",
}

@st.cache_resource
def load_model(path_string):
    return joblib.load(path_string)

uploaded_file = st.file_uploader("Upload test CSV", type=["csv"])
selected_name = st.selectbox("Select a saved model", list(MODEL_FILES.keys()))

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.subheader("Uploaded data preview")
        st.dataframe(data.head())

        model_path = MODEL_DIR / MODEL_FILES[selected_name]
        if not model_path.exists():
            st.error(f"Saved model not found: {model_path}")
            st.stop()

        model = load_model(str(model_path))
        expected_features = list(model.feature_names_in_)
        missing_features = [c for c in expected_features if c not in data.columns]

        if missing_features:
            st.error("CSV is missing required feature columns: " + ", ".join(missing_features))
            st.stop()

        X_upload = data[expected_features]
        predictions = model.predict(X_upload)

        st.subheader("Predictions")
        prediction_table = pd.DataFrame({"Predicted_Class": predictions})
        if TARGET in data.columns:
            prediction_table.insert(0, "Actual_Class", data[TARGET].values)
        st.dataframe(prediction_table)

        if TARGET not in data.columns:
            st.warning("The CSV has no 'Class' target column. Predictions are shown, but evaluation metrics cannot be calculated.")
        else:
            y_true = data[TARGET]
            y_probability = model.predict_proba(X_upload)
            metrics = {
                "Accuracy": accuracy_score(y_true, predictions),
                "AUC (weighted OVR)": roc_auc_score(
                    y_true, y_probability, labels=model.classes_,
                    multi_class="ovr", average="weighted"
                ),
                "Precision (weighted)": precision_score(y_true, predictions, average="weighted", zero_division=0),
                "Recall (weighted)": recall_score(y_true, predictions, average="weighted", zero_division=0),
                "F1 (weighted)": f1_score(y_true, predictions, average="weighted", zero_division=0),
                "MCC": matthews_corrcoef(y_true, predictions),
            }

            st.subheader("Evaluation metrics")
            cols = st.columns(3)
            for i, (metric_name, metric_value) in enumerate(metrics.items()):
                cols[i % 3].metric(metric_name, f"{metric_value:.4f}")

            st.subheader("Confusion matrix")
            cm = confusion_matrix(y_true, predictions, labels=model.classes_)
            fig, ax = plt.subplots(figsize=(8, 6))
            display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
            display.plot(ax=ax, xticks_rotation=45, colorbar=False)
            st.pyplot(fig)

            st.subheader("Classification report")
            report = classification_report(y_true, predictions, output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).transpose())

    except Exception as exc:
        st.error(f"Could not process this CSV: {exc}")
else:
    st.info("Upload a CSV file to begin. For metrics, the CSV must include the 'Class' target column.")
