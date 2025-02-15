#!/usr/bin/env python
# main.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Silence warnings from joblib and xgboost
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

from scipy.stats import shapiro
from sklearn.preprocessing import LabelEncoder, RobustScaler, label_binarize
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb
from sklearn.multiclass import OneVsRestClassifier
from imblearn.over_sampling import SMOTE

# -------------------------------
# 1. Load Local CSV Files
# -------------------------------
train_file = "UNSW_NB15_training-set.csv"
test_file  = "UNSW_NB15_testing-set.csv"

df_train = pd.read_csv(train_file)
df_test  = pd.read_csv(test_file)

print("Training set shape:", df_train.shape)
print("Testing set shape:", df_test.shape)

# -------------------------------
# 2. Drop Extraneous Columns
# -------------------------------
def drop_unwanted_columns(df):
    cols_to_drop = [col for col in df.columns if col.lower() in ['id', 'label']]
    return df.drop(columns=cols_to_drop)

df_train = drop_unwanted_columns(df_train)
df_test  = drop_unwanted_columns(df_test)

print("Training set shape after dropping unwanted columns:", df_train.shape)
print("Testing set shape after dropping unwanted columns:", df_test.shape)

# -------------------------------
# 3. Categorical Encoding
# -------------------------------
categorical_cols = ['proto', 'service', 'state']
for col in categorical_cols:
    if col in df_train.columns:
        le = LabelEncoder()
        df_train[col] = le.fit_transform(df_train[col].astype(str))
    if col in df_test.columns:
        le = LabelEncoder()
        df_test[col] = le.fit_transform(df_test[col].astype(str))

# Process target "attack_cat"
if 'attack_cat' in df_train.columns:
    df_train['attack_cat'] = df_train['attack_cat'].astype(str)
    le_target = LabelEncoder()
    df_train['attack_cat'] = le_target.fit_transform(df_train['attack_cat'])
else:
    raise ValueError("Training set must have an 'attack_cat' column for multi-class classification.")

if 'attack_cat' in df_test.columns:
    df_test['attack_cat'] = df_test['attack_cat'].astype(str)
    df_test['attack_cat'] = le_target.transform(df_test['attack_cat'])

# -------------------------------
# 4. Scaling and Normality Checks (Training Set)
# -------------------------------
numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
if 'attack_cat' in numeric_cols:
    numeric_cols.remove('attack_cat')

print("\nNormality check (Shapiro-Wilk) for numeric features (sample up to 5000 values):")
for col in numeric_cols:
    data = df_train[col].dropna()
    sample = data.sample(5000, random_state=42) if len(data) > 5000 else data
    stat, p = shapiro(sample)
    print(f" - {col}: p-value = {p:.4f}")

scaler = RobustScaler()
df_train[numeric_cols] = scaler.fit_transform(df_train[numeric_cols])
numeric_cols_test = df_test.select_dtypes(include=[np.number]).columns.tolist()
if 'attack_cat' in numeric_cols_test:
    numeric_cols_test.remove('attack_cat')
df_test[numeric_cols_test] = scaler.transform(df_test[numeric_cols_test])

print("\nSample of training data after scaling:")
print(df_train.head())

# -------------------------------
# 5. Prepare Features and Target
# -------------------------------
X_train = df_train.drop(columns=['attack_cat'])
y_train = df_train['attack_cat']

if 'attack_cat' in df_test.columns:
    X_test = df_test.drop(columns=['attack_cat'])
    y_test = df_test['attack_cat']
else:
    X_test = df_test.copy()
    y_test = None

print("\nTraining features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)

# -------------------------------
# 6. Balance the Training Set using SMOTE
# -------------------------------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("\nResampled training features shape:", X_train_res.shape)
print("Resampled training target distribution:")
print(pd.Series(y_train_res).value_counts())

# -------------------------------
# 7. Hyperparameter Tuning (Example with RandomForest using GridSearchCV)
# -------------------------------
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced']
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
grid_rf.fit(X_train_res, y_train_res)
print("\nBest parameters for RandomForest:", grid_rf.best_params_)
best_rf = grid_rf.best_estimator_

# -------------------------------
# 8. Define and Train Classifiers
# -------------------------------
dt_clf  = DecisionTreeClassifier(random_state=42, class_weight='balanced')
rf_clf  = best_rf
et_clf  = ExtraTreesClassifier(random_state=42, n_estimators=100, class_weight='balanced')
# For XGBoost, we try early stopping if supported.
xgb_clf = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss', n_estimators=200)

models = {
    'Decision Tree': dt_clf,
    'Random Forest': rf_clf,
    'Extra Trees': et_clf,
    'XGBoost': xgb_clf
}

if y_test is not None:
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        if name == 'XGBoost':
            # Split off 10% as validation for early stopping
            X_train_xgb, X_val, y_train_xgb, y_val = train_test_split(X_train_res, y_train_res, test_size=0.1, stratify=y_train_res, random_state=42)
            try:
                model.fit(X_train_xgb, y_train_xgb, early_stopping_rounds=10, eval_set=[(X_val, y_val)], verbose=False)
            except TypeError:
                print("early_stopping_rounds not supported; training without it.")
                model.fit(X_train_xgb, y_train_xgb)
        else:
            model.fit(X_train_res, y_train_res)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=le_target.classes_, digits=4, zero_division=0))
    
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
        plt.title(f"{name} Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()
else:
    print("No target labels found in test set. Consider using cross-validation for evaluation.")

# -------------------------------
# 9. Multi-Class ROC Curve (Optional)
# -------------------------------
if y_test is not None:
    from sklearn.preprocessing import label_binarize
    y_test_bin = label_binarize(y_test, classes=np.arange(len(le_target.classes_)))
    n_classes = y_test_bin.shape[1]

    ovr_rf = OneVsRestClassifier(rf_clf)
    y_score = ovr_rf.fit(X_train_res, y_train_res).predict_proba(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(8,6))
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average ROC (area = {roc_auc["micro"]:.2f})',
             color='deeppink', linestyle=':', linewidth=4)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Random Forest - Multi-Class ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
