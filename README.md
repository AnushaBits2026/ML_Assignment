# Machine Learning Classification Assignment - 2

## BITS Pilani — Work Integrated Learning Programmes Division

**Programme:** M.Tech (AIML)  
**Course:** Machine Learning  
**Assignment:** Assignment - 2  
**Project:** Dry Bean Classification Using Machine Learning

---

## a. Problem Statement

The objective of this project is to develop, evaluate and deploy an end-to-end supervised machine learning pipeline that classifies dry beans into different varieties based on their physical measurements.

Five different classification algorithms are implemented and evaluated using six standard statistical evaluation metrics. The models are compared using the same training and testing dataset to identify the best-performing classifier.

An interactive Streamlit application is also developed to allow users to upload test data, select a trained model and view its evaluation results without retraining during inference.

## b. Dataset Description

**Dataset Name:** Dry Bean Dataset  
**Dataset Source:** UCI Machine Learning Repository  
**Dataset Link:** https://archive.ics.uci.edu/dataset/602/dry+bean+dataset  
**Dataset File:** Dry_Bean_Dataset.xlsx  
**Number of Instances after preprocessing:** 13543  
**Number of Input Features:** 16  
**Target Variable:** `Class`  
**Number of Classes:** 7  
**Class Names:** BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA  
**Problem Type:** Multiclass Classification

### Feature Description

The dataset contains numerical measurements describing the physical characteristics and shape of dry beans.

The input features used in this project are:

- Area
- Perimeter
- MajorAxisLength
- MinorAxisLength
- AspectRation
- Eccentricity
- ConvexArea
- EquivDiameter
- Extent
- Solidity
- roundness
- Compactness
- ShapeFactor1
- ShapeFactor2
- ShapeFactor3
- ShapeFactor4

### Data Preprocessing

- Loaded the dataset using Pandas.
- Examined the dataset structure and checked for missing values.
- Removed duplicate records where applicable.
- Separated input features and target variable.
- Divided the dataset using an 80:20 stratified train-test split.
- Used the same test dataset to evaluate all five classifiers.

## c. GitHub Repository Link

**GitHub Repository:**  
https://github.com/AnushaBits2026/ML_Assignment

**Live Streamlit Application:**  
https://drybean-ml2-2026aj05149.streamlit.app/

## d. Models Used and Comparison Table

The following five machine learning classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

### Evaluation Metrics

The following six metrics were used:

- Accuracy
- AUC Score
- Precision (weighted)
- Recall (weighted)
- F1 Score (weighted)
- Matthews Correlation Coefficient (MCC)

### Evaluation Metrics Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| SVM | 0.9210 | 0.9933 | 0.9214 | 0.9210 | 0.9211 | 0.9045 |
| Logistic Regression | 0.9192 | 0.9934 | 0.9197 | 0.9192 | 0.9193 | 0.9023 |
| KNN | 0.9155 | 0.9811 | 0.9163 | 0.9155 | 0.9157 | 0.8978 |
| Decision Tree | 0.8955 | 0.9357 | 0.8954 | 0.8955 | 0.8953 | 0.8737 |
| Gaussian Naive Bayes | 0.7630 | 0.9644 | 0.7647 | 0.7630 | 0.7607 | 0.7143 |

### Observations on Model Performance

### SVM

The model achieved Accuracy of **0.9210**, AUC of **0.9933**, Precision of **0.9214**, Recall of **0.9210**, F1 Score of **0.9211**, and MCC of **0.9045**. SVM with an RBF kernel can learn non-linear decision boundaries between bean varieties. Its rankings among the five models were: Accuracy 1.0, AUC 2.0, F1 Score 1.0, and MCC 1.0.

### Logistic Regression

The model achieved Accuracy of **0.9192**, AUC of **0.9934**, Precision of **0.9197**, Recall of **0.9192**, F1 Score of **0.9193**, and MCC of **0.9023**. Logistic Regression uses a linear decision boundary to distinguish between bean varieties. Its rankings among the five models were: Accuracy 2.0, AUC 1.0, F1 Score 2.0, and MCC 2.0.

### KNN

The model achieved Accuracy of **0.9155**, AUC of **0.9811**, Precision of **0.9163**, Recall of **0.9155**, F1 Score of **0.9157**, and MCC of **0.8978**. KNN predicts a class based on the nearest training observations. Its performance depends on distances between feature values. Its rankings among the five models were: Accuracy 3.0, AUC 3.0, F1 Score 3.0, and MCC 3.0.

### Decision Tree

The model achieved Accuracy of **0.8955**, AUC of **0.9357**, Precision of **0.8954**, Recall of **0.8955**, F1 Score of **0.8953**, and MCC of **0.8737**. Decision Tree classifies observations using feature-based decision rules. Training and testing scores should be compared before concluding whether overfitting is present. Its rankings among the five models were: Accuracy 4.0, AUC 5.0, F1 Score 4.0, and MCC 4.0.

### Gaussian Naive Bayes

The model achieved Accuracy of **0.7630**, AUC of **0.9644**, Precision of **0.7647**, Recall of **0.7630**, F1 Score of **0.7607**, and MCC of **0.7143**. Gaussian Naive Bayes uses a probabilistic approach and assumes conditional independence between features given the class. Its rankings among the five models were: Accuracy 5.0, AUC 4.0, F1 Score 5.0, and MCC 5.0.


### Overall Best Model

Based on the average ranking across all six evaluation metrics, **SVM** is the best-performing model in this comparison.

Its evaluation results are:

- **Accuracy:** 0.9210
- **AUC:** 0.9933
- **Precision:** 0.9214
- **Recall:** 0.9210
- **F1 Score:** 0.9211
- **MCC:** 0.9045

The final model selection should also consider the confusion matrix and class-wise classification results to understand how well the model predicts each bean variety.

## e. Streamlit Application

The Streamlit application is designed to provide an interactive interface for evaluating the trained classification models.

### Application Features

1. Upload test data in CSV format.
2. Select one of the five trained classification models.
3. Load the saved trained model without retraining.
4. Display the evaluation metrics.
5. Display the confusion matrix and classification report.

**Live Application Link:** https://drybean-ml2-2026aj05149.streamlit.app/

## f. Repository Structure

```text
ML_Assignment/
│
├── 2026aj05149_ML2.ipynb
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
│
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── gaussian_naive_bayes.joblib
    └── svm.joblib
```

*Ensure that the filenames shown above match the actual files uploaded to GitHub.*

## g. Conclusion

This project demonstrates the implementation and comparison of five supervised machine learning classification algorithms using the Dry Bean Dataset.

The classifiers were evaluated using Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient. Their results were compared to identify a suitable model for multiclass dry bean classification.

The trained models are also integrated into a Streamlit application to support interactive evaluation using uploaded test data.
