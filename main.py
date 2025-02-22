#!/usr/bin/env python
# main.py

import os
import sys
import argparse
import time
import subprocess
import pandas as pd
import numpy as np
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

@memory.cache
def preprocess_data(df, target_col='attack_cat'):
    for col in ['id', 'label']:
        if col in df.columns:
            df = df.drop(columns=[col])
    target = None
    if target_col in df.columns:
        target = df[target_col]
        df = df.drop(columns=[target_col])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    for col in numeric_cols:
        med = df[col].median()
        if med > 0 and df[col].max() > 10 * med and df[col].max() > 10:
            perc95 = df[col].quantile(0.95)
            df[col] = np.where(df[col] > perc95, perc95, df[col])
    for col in numeric_cols:
        if df[col].nunique() > 50:
            if df[col].min() == 0:
                df[col] = np.log(df[col] + 1)
            else:
                df[col] = np.log(df[col])
    for col in categorical_cols:
        if df[col].nunique() > 6:
            top_categories = df[col].value_counts().head(5).index
            df[col] = df[col].apply(lambda x: x if x in top_categories else '-')
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    if target is not None:
        df[target_col] = target
    return df

def train_pipeline(train_file, test_file):
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    print("Original training set shape:", df_train.shape)
    print("Original testing set shape:", df_test.shape)
    
    df_train = preprocess_data(df_train, target_col='attack_cat')
    df_test = preprocess_data(df_test, target_col='attack_cat')
    print("Processed training set shape:", df_train.shape)
    print("Processed testing set shape:", df_test.shape)
    
    if 'attack_cat' in df_train.columns:
        df_train['attack_cat'] = df_train['attack_cat'].astype(str)
        le_target = LabelEncoder()
        df_train['attack_cat'] = le_target.fit_transform(df_train['attack_cat'])
    else:
        raise ValueError("Training set must have an 'attack_cat' column.")
    if 'attack_cat' in df_test.columns:
        df_test['attack_cat'] = df_test['attack_cat'].astype(str)
        df_test['attack_cat'] = le_target.transform(df_test['attack_cat'])
    
    X_train = df_train.drop(columns=['attack_cat'])
    y_train = df_train['attack_cat']
    X_test = df_test.drop(columns=['attack_cat'])
    y_test = df_test['attack_cat']
    print("\nTraining features shape:", X_train.shape)
    print("Testing features shape:", X_test.shape)
    
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    dump(X_train.columns, "trained_feature_names.pkl")
    
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)
    
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    print("\nResampled training features shape:", X_train_res.shape)
    print("Resampled training target distribution:")
    print(pd.Series(y_train_res).value_counts())
    
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
        # Plotting code removed for faster, unattended training.
        weighted_f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"{name} Weighted F1 Score: {weighted_f1:.4f}\n")
        return acc, weighted_f1, cm
    
    model_scores = {}
    perf_dict = {}  # To store performance for each model
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
        acc, f1_score_model, _ = evaluate_model(name, model, X_test_scaled, y_test, le_target, X_train_res, y_train_res)
        model_scores[name] = f1_score_model
        perf_dict[name] = {"accuracy": acc, "weighted_f1": f1_score_model}
        
    if model_scores:
        best_model_name = max(model_scores, key=model_scores.get)
        best_model_score = model_scores[best_model_name]
        print("\n==============================")
        print(f"Best Model: {best_model_name} with a Weighted F1 Score of {best_model_score:.4f}")
        print("==============================\n")
    else:
        best_model_name = None
    
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
    
    dump(stacking_clf, "trained_stacking_model.pkl")
    dump(scaler, "trained_scaler.pkl")
    dump(le_target, "trained_label_encoder.pkl")
    print("[INFO] Trained stacking model saved as 'trained_stacking_model.pkl'.")
    print("[INFO] Trained scaler saved as 'trained_scaler.pkl'.")
    print("[INFO] Trained label encoder saved as 'trained_label_encoder.pkl'.")
    
    # --- Export CSV Files for SAS Visualization ---
    # 1. Export feature distribution for key features from X_train_scaled
    key_features = ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes']
    def export_feature_distribution(df, features, output_file="feature_distribution.csv"):
        rows = []
        for feature in features:
            if feature in df.columns:
                for val in df[feature]:
                    rows.append({"feature": feature, "value": val, "plot_type": "value"})
        pd.DataFrame(rows).to_csv(output_file, index=False)
        print(f"Exported feature distribution to {output_file}")
    
    export_feature_distribution(X_train_scaled, key_features)
    
    # 2. Export class distribution: original from y_train and balanced from y_train_res
    def export_class_distribution(original_series, balanced_series, target_col="attack_cat", output_file="class_distribution.csv"):
        original = original_series.value_counts().reset_index()
        original.columns = [target_col, "Count"]
        original["dataset"] = "original"
        balanced = pd.Series(balanced_series).value_counts().reset_index()
        balanced.columns = [target_col, "Count"]
        balanced["dataset"] = "balanced"
        pd.concat([original, balanced]).to_csv(output_file, index=False)
        print(f"Exported class distribution to {output_file}")
    
    export_class_distribution(y_train, y_train_res)
    
    # 3. Export hyperparameter tuning results from grid_rf.cv_results_
    def export_hyperparameter_tuning(cv_results, output_file="hyperparameter_tuning.csv"):
        params = cv_results["params"]
        mean_test_score = cv_results["mean_test_score"]
        rows = []
        for param, score in zip(params, mean_test_score):
            rows.append({
                "n_estimators": param.get("n_estimators"),
                "min_samples_split": param.get("min_samples_split"),
                "mean_test_score": score
            })
        pd.DataFrame(rows).to_csv(output_file, index=False)
        print(f"Exported hyperparameter tuning results to {output_file}")
    
    export_hyperparameter_tuning(grid_rf.cv_results_)
    
    # 4. Export classifier performance results
    def export_classifier_performance(perf_dict, output_file="classifier_performance.csv"):
        rows = []
        for model, metrics in perf_dict.items():
            for metric, score in metrics.items():
                rows.append({
                    "Model": model,
                    "Metric": metric,
                    "Score": score
                })
        pd.DataFrame(rows).to_csv(output_file, index=False)
        print(f"Exported classifier performance to {output_file}")
    
    export_classifier_performance(perf_dict)
    
    # 5. Export confusion matrix from stacking classifier
    def export_confusion_matrix(cm, output_file="confusion_matrix.csv"):
        rows = []
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                rows.append({
                    "TrueLabel": i,
                    "PredLabel": j,
                    "Count": int(cm[i, j])
                })
        pd.DataFrame(rows).to_csv(output_file, index=False)
        print(f"Exported confusion matrix to {output_file}")
    
    export_confusion_matrix(cm_stack)
    
    return stacking_clf, le_target, scaler

def extract_features(packet):
    try:
        expected_features = load("trained_feature_names.pkl")
    except Exception as e:
        print("Error loading expected feature names:", e)
        expected_features = []
    features = {col: 0 for col in expected_features}
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
    try:
        proto = packet.transport_layer.lower() if hasattr(packet, 'transport_layer') else ""
    except Exception:
        proto = ""
    for col in features:
        if col.startswith("proto_"):
            if proto == col.split("_")[1]:
                features[col] = 1
    try:
        features['src_ip'] = packet.ip.src
    except Exception:
        features['src_ip'] = "0.0.0.0"
    return features

def live_ddos_detection(model, interface, capture_duration, scaler, port=None, ddos_threshold=0.5):
    print("[*] Starting live DDoS detection mode on interface:", interface)
    display_filter = f"tcp.port == {port}" if port is not None else None
    capture = pyshark.LiveCapture(interface=interface, display_filter=display_filter)
    start_time = time.time()
    
    feature_list = []
    ip_list = []
    
    for packet in capture.sniff_continuously():
        if time.time() - start_time > capture_duration:
            break
        feat = extract_features(packet)
        feature_list.append(feat)
        ip_list.append(feat.get('src_ip', "0.0.0.0"))
    
    if not feature_list:
        print("No packets captured.")
        return False, []
    
    try:
        expected_cols = load("trained_feature_names.pkl")
    except Exception as e:
        print("Error loading trained feature names:", e)
        expected_cols = list(feature_list[0].keys())
    
    df_features = pd.DataFrame(feature_list)
    df_features = df_features[[col for col in expected_cols if col in df_features.columns]]
    
    for col in expected_cols:
        if col not in df_features.columns:
            df_features[col] = 0
    
    try:
        df_features_scaled = scaler.transform(df_features)
    except Exception as e:
        print("Error during scaling:", e)
        df_features_scaled = df_features.values
    
    predictions = model.predict(df_features_scaled)
    df_preds = pd.DataFrame({'src_ip': ip_list, 'prediction': predictions})
    ip_group = df_preds.groupby('src_ip')['prediction'].agg(['mean', 'count']).reset_index()
    print("[DEBUG] IP grouping based on model predictions:")
    print(ip_group)
    
    suspicious_ips = ip_group[ip_group['mean'] >= ddos_threshold]['src_ip'].tolist()
    
    if suspicious_ips:
        print("[ALERT] Potential DDoS attack detected from IP(s):", suspicious_ips)
        return True, suspicious_ips
    else:
        print("[INFO] Traffic appears normal.")
        return False, []

def live_ddos_prevention(model, interface, capture_duration, scaler, port=None, ddos_threshold=0.5):
    detected, suspicious_ips = live_ddos_detection(model, interface, capture_duration, scaler, port, ddos_threshold)
    if detected and suspicious_ips:
        print("[*] Initiating prevention measures...")
        for ip in suspicious_ips:
            block_ip(ip)
    elif detected:
        print("[ALERT] DDoS attack detected by ML model but no specific IP identified for blocking.")
    else:
        print("[INFO] No prevention action taken.")

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
    parser.add_argument('--ddos_threshold', type=float, default=0.5, help="Threshold for average prediction to flag an IP as malicious")
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
            live_ddos_detection(model, args.interface, args.duration, scaler, args.port, args.ddos_threshold)
        elif args.ddos_mode == 'prevention':
            live_ddos_prevention(model, args.interface, args.duration, scaler, args.port, args.ddos_threshold)
    else:
        print("[ERROR] Invalid action specified.")

if __name__ == "__main__":
    main()
