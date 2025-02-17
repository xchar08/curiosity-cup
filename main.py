#!/usr/bin/env python
# main.py

import os
import sys
import argparse
import time
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pyshark  # For live packet capture; ensure it's installed (pip install pyshark)

# Silence warnings from joblib and xgboost (as much as possible)
os.environ["LOKY_MAX_CPU_COUNT"] = "8"
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

from scipy.stats import shapiro
from sklearn.preprocessing import LabelEncoder, RobustScaler, label_binarize
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, roc_curve, auc, 
                             precision_recall_fscore_support, f1_score)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
import xgboost as xgb
from sklearn.multiclass import OneVsRestClassifier
from imblearn.over_sampling import SMOTE
from joblib import dump, load

# -------------------------------
# FUNCTIONS FOR TRAINING PIPELINE
# -------------------------------
def drop_unwanted_columns(df):
    # Drop columns named 'id' or 'label' (ignoring case)
    cols_to_drop = [col for col in df.columns if col.lower() in ['id', 'label']]
    return df.drop(columns=cols_to_drop)

def train_pipeline(train_file, test_file):
    # 1. Load CSV Files
    df_train = pd.read_csv(train_file)
    df_test  = pd.read_csv(test_file)

    print("Training set shape:", df_train.shape)
    print("Testing set shape:", df_test.shape)

    # 2. Drop Extraneous Columns
    df_train = drop_unwanted_columns(df_train)
    df_test  = drop_unwanted_columns(df_test)

    print("Training set shape after dropping unwanted columns:", df_train.shape)
    print("Testing set shape after dropping unwanted columns:", df_test.shape)

    # 3. Categorical Encoding
    categorical_cols = ['proto', 'service', 'state']
    for col in categorical_cols:
        if col in df_train.columns:
            le = LabelEncoder()
            df_train[col] = le.fit_transform(df_train[col].astype(str))
        if col in df_test.columns:
            le = LabelEncoder()
            df_test[col] = le.fit_transform(df_test[col].astype(str))

    # Process the target column "attack_cat" (for multi-class classification)
    if 'attack_cat' in df_train.columns:
        df_train['attack_cat'] = df_train['attack_cat'].astype(str)
        le_target = LabelEncoder()
        df_train['attack_cat'] = le_target.fit_transform(df_train['attack_cat'])
    else:
        raise ValueError("Training set must have an 'attack_cat' column for multi-class classification.")

    if 'attack_cat' in df_test.columns:
        df_test['attack_cat'] = df_test['attack_cat'].astype(str)
        df_test['attack_cat'] = le_target.transform(df_test['attack_cat'])

    # 4. Scaling and Normality Checks (Training Set)
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

    # 5. Prepare Features and Target
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

    # 6. Balance the Training Set using SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print("\nResampled training features shape:", X_train_res.shape)
    print("Resampled training target distribution:")
    print(pd.Series(y_train_res).value_counts())

    # 7. Hyperparameter Tuning for RandomForest (Example)
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

    # 8. Define and Train Base Classifiers
    dt_clf  = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    rf_clf  = best_rf  # tuned RandomForest
    et_clf  = ExtraTreesClassifier(random_state=42, n_estimators=100, class_weight='balanced')
    # XGBoost with CPU settings to avoid deprecated GPU warnings
    xgb_clf = xgb.XGBClassifier(
        random_state=42,
        eval_metric='mlogloss',
        n_estimators=200,
        tree_method='hist'  # Use CPU hist method
    )

    models = {
        'Decision Tree': dt_clf,
        'Random Forest': rf_clf,
        'Extra Trees': et_clf,
        'XGBoost': xgb_clf
    }

    def evaluate_model(name, model, X_test, y_test, le_target, X_train_res, y_train_res):
        """
        Evaluates the model on X_test/y_test, prints classification metrics, plots
        confusion matrix and ROC curves, and returns the weighted F1 score.
        """
        print(f"\n--- Evaluating {name} ---")
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {acc:.4f}\n")
        
        # Detailed per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, zero_division=0)
        results_df = pd.DataFrame({
            "Class": le_target.inverse_transform(np.arange(len(precision))),
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Support": support
        })
        print("Per-class Metrics:")
        print(results_df.to_string(index=False, float_format="%.4f"))
        
        print("\nOverall Classification Report:")
        print(classification_report(y_test, y_pred, target_names=le_target.classes_, digits=4, zero_division=0))
        
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
        plt.title(f"{name} Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()
        
        # --- ROC AUC computation with NaN handling ---
        try:
            X_test_filled = X_test.fillna(0)
            X_train_res_filled = X_train_res.fillna(0)
            y_test_bin = label_binarize(y_test, classes=np.arange(len(le_target.classes_)))
            # Check if each class has at least one instance
            if y_test_bin.shape[1] < 2 or np.any(np.sum(y_test_bin, axis=0) == 0):
                print("Not all classes are represented in y_test. Skipping ROC AUC computation.")
            else:
                n_classes = y_test_bin.shape[1]
                ovr = OneVsRestClassifier(model)
                y_score = ovr.fit(X_train_res_filled, y_train_res).predict_proba(X_test_filled)
                if np.isnan(y_score).any():
                    print("Predicted probabilities contain NaN values. Skipping ROC AUC computation.")
                else:
                    fpr = dict()
                    tpr = dict()
                    roc_auc = dict()
                    for i in range(n_classes):
                        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                        roc_auc[i] = auc(fpr[i], tpr[i])
                    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
                    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                    print("\nROC AUC (micro-average): {:.4f}".format(roc_auc["micro"]))
                    for i in range(n_classes):
                        print(f"Class {le_target.inverse_transform([i])[0]} ROC AUC: {roc_auc[i]:.4f}")
        except Exception as e:
            print("ROC AUC computation failed:", e)
        
        weighted_f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"{name} Weighted F1 Score: {weighted_f1:.4f}\n")
        return weighted_f1

    # 8a. Train and Evaluate Base Models, and Record Their Scores
    model_scores = {}  # to store weighted F1 scores

    if y_test is not None:
        for name, model in models.items():
            print(f"\n--- Training {name} ---")
            if name == 'XGBoost':
                X_train_xgb, X_val, y_train_xgb, y_val = train_test_split(
                    X_train_res, y_train_res, test_size=0.1, stratify=y_train_res, random_state=42
                )
                try:
                    model.fit(X_train_xgb, y_train_xgb, early_stopping_rounds=10, eval_set=[(X_val, y_val)], verbose=False)
                except TypeError:
                    print("early_stopping_rounds not supported; training without it.")
                    model.fit(X_train_xgb, y_train_xgb)
            else:
                model.fit(X_train_res, y_train_res)
            
            score = evaluate_model(name, model, X_test, y_test, le_target, X_train_res, y_train_res)
            model_scores[name] = score
    else:
        print("No target labels found in test set. Consider using cross-validation for evaluation.")

    if model_scores:
        best_model_name = max(model_scores, key=model_scores.get)
        best_model_score = model_scores[best_model_name]
        print("\n==============================")
        print(f"Best Model: {best_model_name} with a Weighted F1 Score of {best_model_score:.4f}")
        print("==============================\n")
    else:
        best_model_name = None

    # 9. Train a Stacking Ensemble for Further Accuracy
    estimators = [
        ('dt', dt_clf),
        ('rf', rf_clf),
        ('et', et_clf),
        ('xgb', xgb_clf)
    ]
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'),
        cv=cv,
        n_jobs=-1,
        passthrough=False
    )
    stacking_clf.fit(X_train_res, y_train_res)
    print("\n--- Evaluating Stacking Classifier ---")
    y_pred_stack = stacking_clf.predict(X_test)
    acc_stack = accuracy_score(y_test, y_pred_stack)
    print(f"Stacking Classifier Accuracy: {acc_stack:.4f}\n")
    print("Stacking Classifier Detailed Report:")
    print(classification_report(y_test, y_pred_stack, target_names=le_target.classes_, digits=4, zero_division=0))
    cm_stack = confusion_matrix(y_test, y_pred_stack)
    print("Stacking Classifier Confusion Matrix:")
    print(cm_stack)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm_stack, annot=True, fmt="d", cmap='Blues')
    plt.title("Stacking Classifier Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    # Save the stacking classifier and scaler for live detection/prevention use
    dump(stacking_clf, "trained_stacking_model.pkl")
    dump(scaler, "trained_scaler.pkl")
    print("[INFO] Trained stacking model saved as 'trained_stacking_model.pkl'.")
    print("[INFO] Trained scaler saved as 'trained_scaler.pkl'.")
    return stacking_clf, le_target, scaler

# -------------------------------
# FUNCTIONS FOR LIVE TRAFFIC MONITORING
# -------------------------------
def extract_features(packet):
    """
    Extract features from a live packet captured via pyshark.
    In this example we extract the packet length and source IP.
    Modify this function to extract the features used during training.
    """
    features = {}
    try:
        features['length'] = int(packet.length)
    except AttributeError:
        features['length'] = 0
    try:
        features['src_ip'] = packet.ip.src
    except AttributeError:
        features['src_ip'] = None
    return features

def live_ddos_detection(model, interface, capture_duration, scaler, port=None, ip_threshold=100):
    """
    Capture live traffic (optionally filtered by port), extract features,
    scale them, predict with the ML model, and count per-source packet rates.
    Returns True if DDoS is detected.
    """
    print("[*] Starting live DDoS detection mode on interface:", interface)
    display_filter = f"tcp.port == {port}" if port is not None else None
    capture = pyshark.LiveCapture(interface=interface, display_filter=display_filter)
    start_time = time.time()
    feature_list = []
    ip_count = {}
    for packet in capture.sniff_continuously():
        if time.time() - start_time > capture_duration:
            break
        features = extract_features(packet)
        feature_list.append(features)
        src_ip = features.get('src_ip')
        if src_ip:
            ip_count[src_ip] = ip_count.get(src_ip, 0) + 1

    if not feature_list:
        print("No packets captured.")
        return False, []

    df_features = pd.DataFrame(feature_list)
    # Scale numeric features using the pre-fitted scaler (if available)
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols and scaler:
        df_features[numeric_cols] = scaler.transform(df_features[numeric_cols])
    
    # Use the ML model to predict for each packet
    predictions = model.predict(df_features)
    ddos_ml_count = (predictions == 1).sum()  # assuming class "1" indicates DDoS-like traffic
    total_packets = len(predictions)
    print(f"[*] Processed {total_packets} packets; {ddos_ml_count} flagged as potential DDoS by ML model.")

    # Check for any IP exceeding the threshold
    suspicious_ips = [ip for ip, count in ip_count.items() if count > ip_threshold]
    if suspicious_ips:
        print(f"[*] Heuristic detection: The following IP(s) sent more than {ip_threshold} packets: {suspicious_ips}")

    # Combine ML and heuristic logic: flag if either more than 50% of packets are flagged OR any IP exceeds the threshold.
    if (total_packets > 0 and (ddos_ml_count / total_packets) > 0.5) or suspicious_ips:
        print("[ALERT] Potential DDoS attack detected!")
        return True, suspicious_ips
    else:
        print("[INFO] Traffic appears normal.")
        return False, []

def live_ddos_prevention(model, interface, capture_duration, scaler, port=None, ip_threshold=100):
    """
    Runs detection and, if DDoS is detected, executes prevention by blocking the suspicious IP(s).
    """
    detected, suspicious_ips = live_ddos_detection(model, interface, capture_duration, scaler, port, ip_threshold)
    if detected and suspicious_ips:
        print("[*] Initiating prevention measures...")
        for ip in suspicious_ips:
            try:
                subprocess.run(
                    ["sudo", "iptables", "-A", "INPUT", "-s", ip, "-j", "DROP"],
                    check=True
                )
                print(f"[ALERT] Added iptables rule to drop packets from {ip}.")
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Failed to add iptables rule for {ip}: {e}")
    elif detected:
        print("[ALERT] DDoS attack detected by ML model but no specific IP identified for blocking.")
    else:
        print("[INFO] No prevention action taken.")

# -------------------------------
# MAIN FUNCTION: Select Action Based on CLI Args
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Network Security ML Pipeline: Train model or monitor live traffic for DDoS")
    parser.add_argument('--action', choices=['train', 'monitor'], default='train',
                        help="Action to perform: 'train' to run the training pipeline, 'monitor' to run live traffic monitoring.")
    # Training files (used when action == 'train')
    parser.add_argument('--train_file', type=str, default="UNSW_NB15_training-set.csv", help="Training CSV file path")
    parser.add_argument('--test_file', type=str, default="UNSW_NB15_testing-set.csv", help="Testing CSV file path")
    # Live monitoring arguments (used when action == 'monitor')
    parser.add_argument('--ddos_mode', choices=['detection', 'prevention'], help="In monitor mode, choose 'detection' or 'prevention'")
    parser.add_argument('--interface', type=str, help="Network interface to capture traffic (e.g., eth0)")
    parser.add_argument('--duration', type=int, default=30, help="Capture duration in seconds for live monitoring")
    parser.add_argument('--port', type=int, help="Optional port number to filter captured traffic")
    # Optionally set a threshold for suspicious packet counts per IP
    parser.add_argument('--ip_threshold', type=int, default=100, help="Packet count threshold per IP to flag as suspicious")
    # Allow specifying model and scaler files
    parser.add_argument('--model_file', type=str, default="trained_stacking_model.pkl", help="Path to the trained model file")
    parser.add_argument('--scaler_file', type=str, default="trained_scaler.pkl", help="Path to the trained scaler file")
    args = parser.parse_args()

    if args.action == 'train':
        # Run the training pipeline
        model, le_target, scaler = train_pipeline(args.train_file, args.test_file)
    elif args.action == 'monitor':
        # In monitor mode, load the saved model and scaler (trained previously)
        try:
            model = load(args.model_file)
            scaler = load(args.scaler_file)
        except Exception as e:
            print(f"[ERROR] Could not load trained model or scaler: {e}")
            sys.exit(1)
        # Ensure required live mode arguments are provided
        if not args.interface or not args.ddos_mode:
            print("[ERROR] In monitor mode, both --interface and --ddos_mode must be specified.")
            sys.exit(1)
        if args.ddos_mode == 'detection':
            live_ddos_detection(model, args.interface, args.duration, scaler, args.port, args.ip_threshold)
        elif args.ddos_mode == 'prevention':
            live_ddos_prevention(model, args.interface, args.duration, scaler, args.port, args.ip_threshold)
    else:
        print("[ERROR] Invalid action specified.")

if __name__ == "__main__":
    main()
