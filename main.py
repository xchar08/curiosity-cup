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
import platform

# Silence warnings from joblib and xgboost
os.environ["LOKY_MAX_CPU_COUNT"] = "8"
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

from scipy.stats import shapiro
from sklearn.preprocessing import LabelEncoder, RobustScaler, label_binarize
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_curve, auc, precision_recall_fscore_support, f1_score)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
import xgboost as xgb
from sklearn.multiclass import OneVsRestClassifier
from imblearn.over_sampling import SMOTE
from joblib import dump, load, Memory

# Set up caching for expensive computations
memory = Memory(location='./cachedir', verbose=0)

# -------------------------------
# Helper function for cross‑platform IP blocking
# -------------------------------
def block_ip(ip):
    os_type = platform.system().lower()
    if os_type == 'windows':
        cmd = [
            "netsh", "advfirewall", "firewall", "add", "rule",
            f"name=Block_{ip}", "protocol=TCP", "dir=in", f"remoteip={ip}",
            "action=block"
        ]
    else:
        cmd = ["sudo", "iptables", "-A", "INPUT", "-s", ip, "-j", "DROP"]
    try:
        subprocess.run(cmd, check=True)
        print(f"[ALERT] Successfully blocked IP {ip}.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to block IP {ip}: {e}")

# -------------------------------
# Data Preprocessing Function (cached)
# -------------------------------
@memory.cache
def preprocess_data(df, target_col='attack_cat'):
    # Drop extraneous columns (e.g., 'id' and 'label').
    for col in ['id', 'label']:
        if col in df.columns:
            df = df.drop(columns=[col])
    # Separate target (if it exists).
    target = None
    if target_col in df.columns:
        target = df[target_col]
        df = df.drop(columns=[target_col])
    # Get numeric and categorical column lists.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    # Clamp extreme numeric values.
    for col in numeric_cols:
        med = df[col].median()
        if med > 0 and df[col].max() > 10 * med and df[col].max() > 10:
            perc95 = df[col].quantile(0.95)
            df[col] = np.where(df[col] > perc95, perc95, df[col])
    # Log-transform numeric features that are continuous (with >50 unique values).
    for col in numeric_cols:
        if df[col].nunique() > 50:
            if df[col].min() == 0:
                df[col] = np.log(df[col] + 1)
            else:
                df[col] = np.log(df[col])
    # For high-cardinality categorical features, keep only the top 5 categories; others become '-'
    for col in categorical_cols:
        if df[col].nunique() > 6:
            top_categories = df[col].value_counts().head(5).index
            df[col] = df[col].apply(lambda x: x if x in top_categories else '-')
    # One‑hot encode the categorical features.
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    # If we had a target column, add it back.
    if target is not None:
        df[target_col] = target
    return df

# -------------------------------
# Training Pipeline
# -------------------------------
def train_pipeline(train_file, test_file):
    # 1. Load CSV files.
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    print("Original training set shape:", df_train.shape)
    print("Original testing set shape:", df_test.shape)
    
    # 2. Preprocess data.
    df_train = preprocess_data(df_train, target_col='attack_cat')
    df_test = preprocess_data(df_test, target_col='attack_cat')
    print("Processed training set shape:", df_train.shape)
    print("Processed testing set shape:", df_test.shape)
    
    # 3. Encode target column.
    if 'attack_cat' in df_train.columns:
        df_train['attack_cat'] = df_train['attack_cat'].astype(str)
        le_target = LabelEncoder()
        df_train['attack_cat'] = le_target.fit_transform(df_train['attack_cat'])
    else:
        raise ValueError("Training set must have an 'attack_cat' column.")
    if 'attack_cat' in df_test.columns:
        df_test['attack_cat'] = df_test['attack_cat'].astype(str)
        df_test['attack_cat'] = le_target.transform(df_test['attack_cat'])
    
    # 4. Prepare features and target.
    X_train = df_train.drop(columns=['attack_cat'])
    y_train = df_train['attack_cat']
    X_test = df_test.drop(columns=['attack_cat'])
    y_test = df_test['attack_cat']
    print("\nTraining features shape:", X_train.shape)
    print("Testing features shape:", X_test.shape)
    
    # 5. Make sure X_test has exactly the same columns as X_train.
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    
    # 6. Save training feature names (after one‑hot encoding) for live monitoring.
    dump(X_train.columns, "trained_feature_names.pkl")
    
    # 7. Scale features.
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)
    
    # 8. Balance training set using SMOTE.
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    print("\nResampled training features shape:", X_train_res.shape)
    print("Resampled training target distribution:")
    print(pd.Series(y_train_res).value_counts())
    
    # 9. Hyperparameter tuning for RandomForest.
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced']
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_rf = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), param_grid, cv=cv,
                           scoring='f1_weighted', n_jobs=-1)
    grid_rf.fit(X_train_res, y_train_res)
    print("\nBest parameters for RandomForest:", grid_rf.best_params_)
    best_rf = grid_rf.best_estimator_
    
    # 10. Define base classifiers (with parallelization when possible).
    dt_clf = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    rf_clf = best_rf
    et_clf = ExtraTreesClassifier(random_state=42, n_estimators=100, class_weight='balanced', n_jobs=-1)
    xgb_clf = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss', n_estimators=200, tree_method='hist', n_jobs=-1)
    models = {
        'Decision Tree': dt_clf,
        'Random Forest': rf_clf,
        'Extra Trees': et_clf,
        'XGBoost': xgb_clf
    }
    
    from sklearn.metrics import classification_report
    def evaluate_model(name, model, X_test, y_test, le_target, X_train_res, y_train_res):
        print(f"\n--- Evaluating {name} ---")
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {acc:.4f}\n")
        pr, rec, f1, supp = precision_recall_fscore_support(y_test, y_pred, zero_division=0)
        results_df = pd.DataFrame({
            "Class": le_target.inverse_transform(np.arange(len(pr))),
            "Precision": pr,
            "Recall": rec,
            "F1-Score": f1,
            "Support": supp
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
        # ROC AUC computation
        try:
            X_test_filled = pd.DataFrame(X_test).fillna(0)
            X_train_res_filled = pd.DataFrame(X_train_res).fillna(0)
            y_test_bin = label_binarize(y_test, classes=np.arange(len(le_target.classes_)))
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
    
    # Train and evaluate each base model.
    model_scores = {}
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        if name == 'XGBoost':
            X_train_xgb, X_val, y_train_xgb, y_val = train_test_split(X_train_res, y_train_res,
                                                                      test_size=0.1,
                                                                      stratify=y_train_res,
                                                                      random_state=42)
            try:
                model.fit(X_train_xgb, y_train_xgb, early_stopping_rounds=10,
                          eval_set=[(X_val, y_val)], verbose=False)
            except TypeError:
                print("early_stopping_rounds not supported; training without it.")
                model.fit(X_train_xgb, y_train_xgb)
        else:
            model.fit(X_train_res, y_train_res)
        score = evaluate_model(name, model, X_test_scaled, y_test, le_target, X_train_res, y_train_res)
        model_scores[name] = score
    if model_scores:
        best_model_name = max(model_scores, key=model_scores.get)
        best_model_score = model_scores[best_model_name]
        print("\n==============================")
        print(f"Best Model: {best_model_name} with a Weighted F1 Score of {best_model_score:.4f}")
        print("==============================\n")
    else:
        best_model_name = None
    
    # Build and train a stacking classifier.
    estimators = [
        ('dt', dt_clf),
        ('rf', rf_clf),
        ('et', et_clf),
        ('xgb', xgb_clf)
    ]
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced', n_jobs=-1),
        cv=cv,
        n_jobs=-1,
        passthrough=False
    )
    stacking_clf.fit(X_train_res, y_train_res)
    print("\n--- Evaluating Stacking Classifier ---")
    y_pred_stack = stacking_clf.predict(X_test_scaled)
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
    
    # Save the trained models and scaler.
    dump(stacking_clf, "trained_stacking_model.pkl")
    dump(scaler, "trained_scaler.pkl")
    dump(le_target, "trained_label_encoder.pkl")
    print("[INFO] Trained stacking model saved as 'trained_stacking_model.pkl'.")
    print("[INFO] Trained scaler saved as 'trained_scaler.pkl'.")
    print("[INFO] Trained label encoder saved as 'trained_label_encoder.pkl'.")
    return stacking_clf, le_target, scaler

# -------------------------------
# Live Traffic Monitoring Functions
# -------------------------------
def extract_features(packet):
    """
    Live feature extraction that builds a dictionary matching the training features.
    It loads the expected feature names (saved during training) and initializes them to 0.
    Then it extracts some numeric fields from the packet.
    NOTE: You may need to extend this to more closely match your training features.
    """
    try:
        expected_features = load("trained_feature_names.pkl")
    except Exception as e:
        print("Error loading expected feature names:", e)
        expected_features = []
    # Initialize dictionary with all expected features set to 0.
    features = {col: 0 for col in expected_features}
    # Example extraction:
    try:
        pkt_length = int(packet.length)
    except Exception:
        pkt_length = 0
    if 'sbytes' in features:
        features['sbytes'] = pkt_length
    try:
        if hasattr(packet, 'tcp'):
            features['sport'] = int(packet.tcp.srcport)
        elif hasattr(packet, 'udp'):
            features['sport'] = int(packet.udp.srcport)
    except Exception:
        features['sport'] = 0
    try:
        if hasattr(packet, 'tcp'):
            features['dsport'] = int(packet.tcp.dstport)
        elif hasattr(packet, 'udp'):
            features['dsport'] = int(packet.udp.dstport)
    except Exception:
        features['dsport'] = 0
    # Process protocol: set one-hot values for features starting with "proto_"
    try:
        proto = packet.transport_layer.lower() if hasattr(packet, 'transport_layer') else ""
    except Exception:
        proto = ""
    for col in features:
        if col.startswith("proto_"):
            # For example, if col is "proto_tcp", set to 1 if proto matches.
            if proto == col.split("_")[1]:
                features[col] = 1
    # Save source IP (for heuristic detection)
    try:
        features['src_ip'] = packet.ip.src
    except Exception:
        features['src_ip'] = "0.0.0.0"
    return features

def live_ddos_detection(model, interface, capture_duration, scaler, port=None, ip_threshold=100):
    print("[*] Starting live DDoS detection mode on interface:", interface)
    display_filter = f"tcp.port == {port}" if port is not None else None
    capture = pyshark.LiveCapture(interface=interface, display_filter=display_filter)
    start_time = time.time()
    feature_list = []
    ip_count = {}
    for packet in capture.sniff_continuously():
        if time.time() - start_time > capture_duration:
            break
        feat = extract_features(packet)
        feature_list.append(feat)
        src_ip = feat.get('src_ip')
        if src_ip:
            ip_count[src_ip] = ip_count.get(src_ip, 0) + 1
    if not feature_list:
        print("No packets captured.")
        return False, []
    print("[DEBUG] Captured IP counts:", ip_count)
    total_packets = len(feature_list)
    print(f"[DEBUG] Total packets captured: {total_packets}")
    df_features = pd.DataFrame(feature_list)
    # Load expected feature names.
    try:
        expected_cols = load("trained_feature_names.pkl")
    except Exception as e:
        print("Error loading trained feature names:", e)
        expected_cols = df_features.columns
    # Build a DataFrame with all expected columns, filling missing ones with 0.
    df_full = pd.DataFrame(0, index=df_features.index, columns=expected_cols)
    for col in df_features.columns:
        if col in expected_cols:
            df_full[col] = df_features[col]
    # Debug: print shape of df_full.
    print("[DEBUG] Live feature DataFrame shape (before scaling):", df_full.shape)
    try:
        df_features_scaled = scaler.transform(df_full)
    except Exception as e:
        print("Error during scaling:", e)
        df_features_scaled = df_full.values
    # Predict using the model.
    predictions = model.predict(df_features_scaled)
    ddos_ml_count = (predictions == 1).sum()  # adjust if class "1" means DDoS in your encoding.
    print(f"[DEBUG] Total packets processed by ML model: {len(predictions)}")
    print(f"[DEBUG] Number of packets flagged as potential DDoS by ML model: {ddos_ml_count}")
    suspicious_ips = [ip for ip, count in ip_count.items() if count > ip_threshold]
    if suspicious_ips:
        print(f"[*] Heuristic detection: The following IP(s) sent more than {ip_threshold} packets: {suspicious_ips}")
    if (total_packets > 0 and (ddos_ml_count / total_packets) > 0.5) or suspicious_ips:
        print("[ALERT] Potential DDoS attack detected!")
        return True, suspicious_ips
    else:
        print("[INFO] Traffic appears normal.")
        return False, []

def live_ddos_prevention(model, interface, capture_duration, scaler, port=None, ip_threshold=100):
    detected, suspicious_ips = live_ddos_detection(model, interface, capture_duration, scaler, port, ip_threshold)
    if detected and suspicious_ips:
        print("[*] Initiating prevention measures...")
        for ip in suspicious_ips:
            block_ip(ip)
    elif detected:
        print("[ALERT] DDoS attack detected by ML model but no specific IP identified for blocking.")
    else:
        print("[INFO] No prevention action taken.")

# -------------------------------
# Main Function
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Network Security ML Pipeline: Train model or monitor live traffic for DDoS")
    parser.add_argument('--action', choices=['train', 'monitor'], default='train',
                        help="Action to perform: 'train' or 'monitor'.")
    parser.add_argument('--train_file', type=str, default="UNSW_NB15_training-set.csv", help="Training CSV file path")
    parser.add_argument('--test_file', type=str, default="UNSW_NB15_testing-set.csv", help="Testing CSV file path")
    parser.add_argument('--ddos_mode', choices=['detection', 'prevention'], help="For monitoring: 'detection' or 'prevention'")
    parser.add_argument('--interface', type=str, help="Network interface for capturing traffic (e.g., eth0 or Wi-Fi)")
    parser.add_argument('--duration', type=int, default=30, help="Capture duration (seconds) for live monitoring")
    parser.add_argument('--port', type=int, help="Optional port number for traffic filtering")
    parser.add_argument('--ip_threshold', type=int, default=100, help="Packet count threshold per IP to flag as suspicious")
    parser.add_argument('--model_file', type=str, default="trained_stacking_model.pkl", help="Path to the trained model file")
    parser.add_argument('--scaler_file', type=str, default="trained_scaler.pkl", help="Path to the trained scaler file")
    args = parser.parse_args()
    
    if args.action == 'train':
        model, le_target, scaler = train_pipeline(args.train_file, args.test_file)
    elif args.action == 'monitor':
        try:
            model = load(args.model_file)
            scaler = load(args.scaler_file)
        except Exception as e:
            print(f"[ERROR] Could not load trained model or scaler: {e}")
            sys.exit(1)
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
