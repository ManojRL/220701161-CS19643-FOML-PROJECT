# Import libraries
from __future__ import division

import pandas as pd
import matplotlib.pyplot as plt  # plotting
import seaborn as sns
import numpy as np  # linear algebra
import math
import os
import copy
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    auc,
    precision_score,
    recall_score,
    precision_recall_curve,
    f1_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

from collections import Counter
from sklearn.datasets import make_classification


# Get the absolute path to the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the folder path relative to the script's location
folder = os.path.join(script_dir, "../pkg/flowOutput/")
folder = os.path.abspath(folder)

fname_malicious = 'webgoat_flow_stats.csv'
fname_benign = "2017-05-02_kali-normal22_flow_stats.csv"

file_path_malicious = os.path.join(folder, fname_malicious)
file_path_benign = os.path.join(folder, fname_benign)

# Malicious
pd_malicious = pd.read_csv(file_path_malicious)
pd_malicious.drop(pd_malicious.tail(1).index, inplace=True)
pd_malicious["Type"] = "Malicious"

# Benign
pd_benign = pd.read_csv(file_path_benign)
pd_benign["Type"] = "Benign"

print(pd_benign.shape)

# Combine and shuffle
pd_comb = pd.concat([pd_malicious, pd_benign])
pd_comb = pd_comb.sample(frac=1)

# Add throughput columns
colsPerTime = [
    "flowLength",
    "fwdFlowLength",
    "bwdFlowLength",
    "packetSizeTotal",
    "fwdPacketSizeTotal",
    "bwdPacketSizeTotal",
]

for feature in colsPerTime:
    pd_comb[feature + "PerTime"] = pd_comb[feature] / pd_comb["flowDuration"]

# Features
feature_cols = [
    "flowDuration",
    "flowLength",
    "fwdFlowLength",
    "bwdFlowLength",
    "packetSizeTotal",
    "packetSizeMean",
    "packetSizeStd",
    "packetSizeMin",
    "packetSizeMax",
    "fwdPacketSizeTotal",
    "bwdPacketSizeTotal",
    "fwdPacketSizeMean",
    "bwdPacketSizeMean",
    "fwdPacketSizeStd",
    "bwdPacketSizeStd",
    "fwdPacketSizeMin",
    "bwdPacketSizeMin",
    "fwdPacketSizeMax",
    "bwdPacketSizeMax",
    "IATMean",
    "IATStd",
    "IATMin",
    "IATMax",
    "fwdIATTotal",
    "bwdIATTotal",
    "fwdIATMean",
    "bwdIATMean",
    "fwdIATStd",
    "bwdIATStd",
    "fwdIATMin",
    "bwdIATMin",
    "fwdIATMax",
    "bwdIATMax",
    "flowLengthPerTime",
    "fwdFlowLengthPerTime",
    "bwdFlowLengthPerTime",
    "packetSizeTotalPerTime",
    "fwdPacketSizeTotalPerTime",
    "bwdPacketSizeTotalPerTime",
    "Type",
]

# Select features
pd_comb_features = pd_comb[feature_cols]

# Clean dataset
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    df_X = df.iloc[:, :-1]
    df_Y = df.iloc[:, -1]
    indices_to_keep = ~df_X.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    return df_X[indices_to_keep].astype(np.float64).values, df_Y[indices_to_keep].values

# Cleaned features
pd_comb_features_cp = pd_comb_features.copy(deep=True)
X, y = clean_dataset(pd_comb_features_cp)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Scaling
scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

# Class weight grid
w = [
    {0: 0.10, 1: 99.90},
    {0: 0.25, 1: 99.75},
    {0: 0.50, 1: 99.50},
    {0: 0.75, 1: 99.25},
    {0: 1.00, 1: 99.00},
    {
        0: 100 * np.sum(y == "Malicious") / (np.sum(y == "Benign") + np.sum(y == "Malicious")),
        1: 100 * np.sum(y == "Benign") / (np.sum(y == "Benign") + np.sum(y == "Malicious")),
    },
]
crange = np.arange(0.1, 1.0, 0.2)
hyperparam_grid = {
    "class_weight": w,
    "penalty": ["l1", "l2"],
    "C": crange,
    "fit_intercept": [True, False],
}

# Convert target to int
y_train2 = np.copy(y_train)
y_train2[np.where(y_train == "Benign")[0]] = 0
y_train2[np.where(y_train == "Malicious")[0]] = 1

# Model
lg = LogisticRegression(random_state=13)
grid = GridSearchCV(lg, hyperparam_grid, scoring="roc_auc", cv=10, n_jobs=-1, refit=True)
grid.fit(X_train_scale, y_train2.astype("int32"))

print(f"Best score: {grid.best_score_} with param: {grid.best_params_}")

# Prediction
y_pred_wt = grid.predict(X_test_scale)
y_test2 = np.copy(y_test)
y_test2[np.where(y_test == "Benign")[0]] = 0
y_test2[np.where(y_test == "Malicious")[0]] = 1

# Evaluation
conf_mat = confusion_matrix(y_test2.astype("int32"), y_pred_wt)

print(f"Accuracy Score: {accuracy_score(y_test2.astype('int32'),y_pred_wt)}")
print(f"✅ Model Accuracy: {100 * accuracy_score(y_test2.astype('int32'), y_pred_wt):.2f}%")
print(f"Confusion Matrix: \n{conf_mat}")
print(f"Area Under Curve: {roc_auc_score(y_test2.astype('int32'), y_pred_wt)}")
print(f"Recall score: {100*recall_score(y_test2.astype('int32'), y_pred_wt)}")
print(f"Data reduction: { np.round(100.0 * conf_mat.T[1].sum() / conf_mat.sum(), 2)}%")
print(f"Malicious in console output: { np.round(100.0 * conf_mat.T[1][1] / conf_mat.T[1].sum(), 2)}%")
print("F1 score: ", f1_score(y_test2.astype("int32"), y_pred_wt, average="weighted"))

# Best fit model for saving
best_fit_model = LogisticRegression(
    class_weight=grid.best_params_["class_weight"],
    penalty=grid.best_params_["penalty"],
    C=grid.best_params_["C"],
    fit_intercept=grid.best_params_["fit_intercept"],
    random_state=13,
    max_iter=5,
)
best_fit_model.fit(X_train_scale, y_train2.astype("int32"))

# Save parameters
directory = "../pkg/ml/parameters/"
os.makedirs(directory, exist_ok=True)

np.savetxt(os.path.join(directory, "mean.txt"), scaler.mean_, delimiter=",")
np.savetxt(os.path.join(directory, "std.txt"), scaler.scale_, delimiter=",")
np.savetxt(os.path.join(directory, "weights.txt"), best_fit_model.coef_[0], delimiter=",")
np.savetxt(os.path.join(directory, "intercept.txt"), best_fit_model.intercept_, delimiter=",")

# Feature importance
important_features = pd_comb_features_cp.iloc[:, :-1].columns.values[
    np.argsort(-1 * np.abs(best_fit_model.coef_[0]))
]
print(important_features)

# -------- VISUALIZATION -------- #

# Confusion Matrix Plot
ConfusionMatrixDisplay.from_predictions(y_test2.astype("int32"), y_pred_wt)
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
RocCurveDisplay.from_predictions(y_test2.astype("int32"), y_pred_wt)
plt.title("ROC Curve")
plt.show()

# Precision-Recall Curve
PrecisionRecallDisplay.from_predictions(y_test2.astype("int32"), y_pred_wt)
plt.title("Precision-Recall Curve")
plt.show()

# Feature Importance Plot
importances = np.abs(best_fit_model.coef_[0])
feature_names = pd_comb_features_cp.iloc[:, :-1].columns.values
feature_importance = pd.Series(importances, index=feature_names)
top_features = feature_importance.sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_features.values, y=top_features.index)
plt.title("Top 10 Important Features")
plt.xlabel("Coefficient Magnitude")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# Collect scores for visualization
accuracy = accuracy_score(y_test2.astype("int32"), y_pred_wt)
recall = recall_score(y_test2.astype("int32"), y_pred_wt)
f1 = f1_score(y_test2.astype("int32"), y_pred_wt, average="weighted")
auc_score = roc_auc_score(y_test2.astype("int32"), y_pred_wt)
data_reduction = np.round(100.0 * conf_mat.T[1].sum() / conf_mat.sum(), 2)
malicious_detection = np.round(100.0 * conf_mat.T[1][1] / conf_mat.T[1].sum(), 2)

# Bar Plot for performance scores
plt.figure(figsize=(8, 5))
metrics = ['Accuracy', 'Recall', 'F1 Score', 'AUC']
values = [accuracy, recall, f1, auc_score]
sns.barplot(x=metrics, y=values, palette='viridis')
plt.title('Model Performance Metrics')
plt.ylim(0, 1)
for index, value in enumerate(values):
    plt.text(index, value + 0.02, f"{value:.2f}", ha='center')
plt.ylabel('Score')
plt.tight_layout()
plt.show()

# Pie chart for Data Reduction and Malicious Detection rates
plt.figure(figsize=(6, 6))
labels = ['Malicious Detected', 'Others']
sizes = [malicious_detection, 100 - malicious_detection]
colors = ['#ff6f61', '#6baed6']
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
plt.title('Malicious Detection in Console Output')
plt.axis('equal')
plt.show()

# Another pie for Data Reduction effectiveness
plt.figure(figsize=(6, 6))
labels = ['Data Reduction (Processed)', 'Remaining']
sizes = [data_reduction, 100 - data_reduction]
colors = ['#2ca25f', '#deebf7']
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
plt.title('Data Reduction Rate')
plt.axis('equal')
plt.show()