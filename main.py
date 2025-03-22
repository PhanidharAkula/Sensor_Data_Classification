"""
main.py
-------
Author: Phanidhar Akula
-----------------------
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# ----------------------------- Configuration -----------------------------
DATASET_PATH = "./DataSet2"                # Directory for the sensor data
OUTPUT_FEATURE_CSV = "DataSet2_AllSensors_Featured.csv"
WINDOW_SIZE = 100                          # Rows per time-series window
USE_ENSEMBLE = True                        # Whether to create a VotingClassifier
PLOTTING = True                            # If False, does not create sensor plots
TEST_SIZE = 0.20                           # Fraction of data used for testing

# The user & activity for single-activity prediction
PREDICTION_USER = 16
PREDICTION_ACTIVITY = 8

# Which classifiers to use.
# "all" => use all. Or specify e.g. ["NaiveBayes","RandomForest"].
SELECT_MODELS = ["NaiveBayes", "NaiveBayes", "DecisionTree", "RandomForest"]

# Official 15 Activity Labels from "Data collection sheet 2"
ACTIVITY_LABELS = {
    1:  "Sitting - Reading a book",
    2:  "Sitting - Writing in a notebook",
    3:  "Using computer - Typing",
    4:  "Using computer - Browsing",
    5:  "While sitting - Moving head, body",
    6:  "While sitting - Moving chair",
    7:  "Sitting - Stand up from sitting",
    8:  "Standing",
    9:  "Walking",
    10: "Running",
    11: "Taking stairs",
    12: "Sitting - Stationary, Wear it, Put it back, Stationary",
    13: "Standing Stationary, Wear it, Put it back, Stationary",
    14: "Sitting - Pick up items from floor",
    15: "Standing - Pick up items from floor"
}


# ------------------------- Data Processing -------------------------
def get_activity_count(user_id):
    """
    Returns how many activities 'user_id' has in this dataset:
      - 13..23 => 15
      - 24, 26, 27 => 12
      - 25 => 13
      - Others => 0
    """
    if 13 <= user_id <= 23:
        return 15
    elif user_id in [24, 26, 27]:
        return 12
    elif user_id == 25:
        return 13
    else:
        return 0

def load_sensor_data(file_path):
    """
    Reads a CSV with header=0, drops empty rows, sorts by 'time (-13:00)' if present.
    Returns a DataFrame; if missing or error, returns an empty DataFrame.
    """
    if not os.path.exists(file_path):
        print(f"[WARNING] Missing file: {file_path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path, header=0)
        df.dropna(how="all", inplace=True)
        if "time (-13:00)" in df.columns:
            df.sort_values("time (-13:00)", inplace=True, ignore_index=True)
        return df
    except Exception as e:
        print(f"[ERROR] Could not process {file_path}: {e}")
        return pd.DataFrame()


# ------------------------- Plotting Functions -------------------------
def plot_four_sensors(acc_df, gyro_df, mag_df, press_df, user_id, activity_id):
    """
    If PLOTTING=True, saves a 4-subplot figure of (Acc, Gyro, Mag, Pressure).
    No on-screen display. If the file already exists, skip plotting.
    """
    if not PLOTTING:
        return
    os.makedirs("Plottings", exist_ok=True)
    plot_filename = f"Plottings/User{user_id}_Activity{activity_id}.png"

    if os.path.exists(plot_filename):
        print(f"[INFO] Plot {plot_filename} already exists. Skipping regeneration.")
        return

    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=False)
    fig.suptitle(f"User {user_id}, Activity {activity_id}", fontsize=16)

    # Accelerometer
    if not acc_df.empty and "time (-13:00)" in acc_df.columns:
        t = acc_df["time (-13:00)"]
        if "x-axis (g)" in acc_df.columns:
            axs[0].plot(t, acc_df["x-axis (g)"], label="Accel X")
        if "y-axis (g)" in acc_df.columns:
            axs[0].plot(t, acc_df["y-axis (g)"], label="Accel Y")
        if "z-axis (g)" in acc_df.columns:
            axs[0].plot(t, acc_df["z-axis (g)"], label="Accel Z")
        axs[0].set_title("Accelerometer")
        axs[0].legend()
    else:
        axs[0].set_title("Accelerometer (No Data)")

    # Gyroscope
    if not gyro_df.empty and "time (-13:00)" in gyro_df.columns:
        t = gyro_df["time (-13:00)"]
        if "x-axis (deg/s)" in gyro_df.columns:
            axs[1].plot(t, gyro_df["x-axis (deg/s)"], label="Gyro X")
        if "y-axis (deg/s)" in gyro_df.columns:
            axs[1].plot(t, gyro_df["y-axis (deg/s)"], label="Gyro Y")
        if "z-axis (deg/s)" in gyro_df.columns:
            axs[1].plot(t, gyro_df["z-axis (deg/s)"], label="Gyro Z")
        axs[1].set_title("Gyroscope")
        axs[1].legend()
    else:
        axs[1].set_title("Gyroscope (No Data)")

    # Magnetometer
    if not mag_df.empty and "time (-13:00)" in mag_df.columns:
        t = mag_df["time (-13:00)"]
        if "x-axis (T)" in mag_df.columns:
            axs[2].plot(t, mag_df["x-axis (T)"], label="Mag X")
        if "y-axis (T)" in mag_df.columns:
            axs[2].plot(t, mag_df["y-axis (T)"], label="Mag Y")
        if "z-axis (T)" in mag_df.columns:
            axs[2].plot(t, mag_df["z-axis (T)"], label="Mag Z")
        axs[2].set_title("Magnetometer")
        axs[2].legend()
    else:
        axs[2].set_title("Magnetometer (No Data)")

    # Pressure
    if not press_df.empty and "time (-13:00)" in press_df.columns:
        t = press_df["time (-13:00)"]
        pcol = None
        for c in press_df.columns:
            if "pressure" in c.lower():
                pcol = c
                break
        if pcol:
            axs[3].plot(t, press_df[pcol], label="Pressure")
            axs[3].set_title("Pressure")
            axs[3].legend()
        else:
            axs[3].set_title("Pressure (Column not found)")
    else:
        axs[3].set_title("Pressure (No Data)")

    plt.tight_layout()
    plt.savefig(plot_filename)
    print(f"[INFO] Saved plot to {plot_filename}")
    plt.close(fig)

def plot_all_activities():
    """
    If PLOTTING=True, loops over user=13..27 for all activities,
    loads raw CSVs, calls plot_four_sensors(...) for each.
    """
    if not PLOTTING:
        print("\n[INFO] Plotting is disabled; skipping plot pipeline.")
        return

    for user_id in range(13, 28):
        act_count = get_activity_count(user_id)
        if act_count == 0:
            continue
        for activity_id in range(1, act_count + 1):
            acc_file   = os.path.join(DATASET_PATH, f"User{user_id}", f"Activity{activity_id}", "Accelerometer.csv")
            gyro_file  = os.path.join(DATASET_PATH, f"User{user_id}", f"Activity{activity_id}", "Gyroscope.csv")
            mag_file   = os.path.join(DATASET_PATH, f"User{user_id}", f"Activity{activity_id}", "Magnetometer.csv")
            press_file = os.path.join(DATASET_PATH, f"User{user_id}", f"Activity{activity_id}", "Pressure.csv")

            acc_df   = load_sensor_data(acc_file)
            gyro_df  = load_sensor_data(gyro_file)
            mag_df   = load_sensor_data(mag_file)
            press_df = load_sensor_data(press_file)

            plot_four_sensors(acc_df, gyro_df, mag_df, press_df, user_id, activity_id)


# ------------------------- Feature Extraction -------------------------
def window_data(df, window_size=WINDOW_SIZE):
    """
    Splits df into consecutive windows of 'window_size' rows.
    Only returns full windows.
    """
    windows = []
    n = len(df)
    for start in range(0, n, window_size):
        end = start + window_size
        if end <= n:
            windows.append(df.iloc[start:end])
    return windows

def extract_features(sensor_name, window_df):
    """
    Extracts (mean, std, min, max, median) from each axis/column
    for the given sensor window, storing them in 'feats'.
    """
    feats = {}

    # ACCELEROMETER
    if sensor_name == "accelerometer":
        xcol, ycol, zcol = "x-axis (g)", "y-axis (g)", "z-axis (g)"
        if xcol in window_df.columns:
            feats["acc_x_mean"]   = window_df[xcol].mean()
            feats["acc_x_std"]    = window_df[xcol].std()
            feats["acc_x_min"]    = window_df[xcol].min()
            feats["acc_x_max"]    = window_df[xcol].max()
            feats["acc_x_median"] = window_df[xcol].median()
        if ycol in window_df.columns:
            feats["acc_y_mean"]   = window_df[ycol].mean()
            feats["acc_y_std"]    = window_df[ycol].std()
            feats["acc_y_min"]    = window_df[ycol].min()
            feats["acc_y_max"]    = window_df[ycol].max()
            feats["acc_y_median"] = window_df[ycol].median()
        if zcol in window_df.columns:
            feats["acc_z_mean"]   = window_df[zcol].mean()
            feats["acc_z_std"]    = window_df[zcol].std()
            feats["acc_z_min"]    = window_df[zcol].min()
            feats["acc_z_max"]    = window_df[zcol].max()
            feats["acc_z_median"] = window_df[zcol].median()

    # GYROSCOPE
    elif sensor_name == "gyroscope":
        xcol, ycol, zcol = "x-axis (deg/s)", "y-axis (deg/s)", "z-axis (deg/s)"
        if xcol in window_df.columns:
            feats["gyro_x_mean"]   = window_df[xcol].mean()
            feats["gyro_x_std"]    = window_df[xcol].std()
            feats["gyro_x_min"]    = window_df[xcol].min()
            feats["gyro_x_max"]    = window_df[xcol].max()
            feats["gyro_x_median"] = window_df[xcol].median()
        if ycol in window_df.columns:
            feats["gyro_y_mean"]   = window_df[ycol].mean()
            feats["gyro_y_std"]    = window_df[ycol].std()
            feats["gyro_y_min"]    = window_df[ycol].min()
            feats["gyro_y_max"]    = window_df[ycol].max()
            feats["gyro_y_median"] = window_df[ycol].median()
        if zcol in window_df.columns:
            feats["gyro_z_mean"]   = window_df[zcol].mean()
            feats["gyro_z_std"]    = window_df[zcol].std()
            feats["gyro_z_min"]    = window_df[zcol].min()
            feats["gyro_z_max"]    = window_df[zcol].max()
            feats["gyro_z_median"] = window_df[zcol].median()

    # MAGNETOMETER
    elif sensor_name == "magnetometer":
        xcol, ycol, zcol = "x-axis (T)", "y-axis (T)", "z-axis (T)"
        if xcol in window_df.columns:
            feats["mag_x_mean"]   = window_df[xcol].mean()
            feats["mag_x_std"]    = window_df[xcol].std()
            feats["mag_x_min"]    = window_df[xcol].min()
            feats["mag_x_max"]    = window_df[xcol].max()
            feats["mag_x_median"] = window_df[xcol].median()
        if ycol in window_df.columns:
            feats["mag_y_mean"]   = window_df[ycol].mean()
            feats["mag_y_std"]    = window_df[ycol].std()
            feats["mag_y_min"]    = window_df[ycol].min()
            feats["mag_y_max"]    = window_df[ycol].max()
            feats["mag_y_median"] = window_df[ycol].median()
        if zcol in window_df.columns:
            feats["mag_z_mean"]   = window_df[zcol].mean()
            feats["mag_z_std"]    = window_df[zcol].std()
            feats["mag_z_min"]    = window_df[zcol].min()
            feats["mag_z_max"]    = window_df[zcol].max()  # fix the bug here
            feats["mag_z_median"] = window_df[zcol].median()

    # PRESSURE
    elif sensor_name == "pressure":
        pcol = None
        for c in window_df.columns:
            if "pressure" in c.lower():
                pcol = c
                break
        if pcol:
            feats["press_mean"]   = window_df[pcol].mean()
            feats["press_std"]    = window_df[pcol].std()
            feats["press_min"]    = window_df[pcol].min()
            feats["press_max"]    = window_df[pcol].max()
            feats["press_median"] = window_df[pcol].median()

    return feats


# ------------------------- Classification -------------------------
def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on X_test,y_test (accuracy, precision, recall, F1, possible AUC).
    """
    y_pred = model.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    unique_labels = np.unique(y_test)
    if len(unique_labels) == 2 and hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
        auc_val = roc_auc_score(y_test, y_score)
        print(f"AUC:       {auc_val:.4f}")
    elif len(unique_labels) == 2:
        print("Model has no predict_proba; cannot compute AUC.")
    else:
        print("Multi-class => skipping single AUC metric.")

def run_classification(features_df):
    """
    1) Splits features_df into train/test
    2) Tunes each chosen classifier with 5-fold CV
    3) Picks best param set, retrains on entire train set
    4) Evaluates final test set metrics
    5) If USE_ENSEMBLE=True & multiple models, build VotingClassifier
    6) Otherwise pick single best model
    7) Return final model
    """
    all_classifiers = {
        "LogisticRegression": (
            LogisticRegression(solver='liblinear', max_iter=1000),
            {
                "C": [0.01, 0.1, 1, 10],
                "penalty": ["l1", "l2"]
            }
        ),
        "NaiveBayes": (
            GaussianNB(),
            {
                "var_smoothing": [1e-9, 1e-8, 1e-7]
            }
        ),
        "DecisionTree": (
            DecisionTreeClassifier(random_state=42),
            {
                "max_depth": [None, 5, 10, 20],
                "min_samples_leaf": [1, 2, 5]
            }
        ),
        "SVM": (
            SVC(probability=True),
            {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf"]
            }
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=42),
            {
                "n_estimators": [50, 100],
                "max_depth": [None, 10, 20],
                "min_samples_leaf": [1, 2]
            }
        )
    }

    # Decide which classifiers to use
    if SELECT_MODELS == "all":
        classifiers_to_use = all_classifiers
        print(f"\n[INFO] Using ALL classifiers: {list(classifiers_to_use.keys())}")
    else:
        chosen = {}
        for m in SELECT_MODELS:
            if m in all_classifiers:
                chosen[m] = all_classifiers[m]
            else:
                print(f"[WARNING] '{m}' is not a known model. Skipping.")
        classifiers_to_use = chosen
        print(f"\n[INFO] Using SELECTED classifiers: {list(classifiers_to_use.keys())}")

    print("\n[INFO] Starting classification process...")
    df = features_df.copy().fillna(0)
    y = df["activity_id"].astype(str)
    X = df.drop(columns=["activity_id", "user_id"], errors="ignore").select_dtypes(include=[np.number])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    print("[INFO] Train shape:", X_train.shape, "& Test shape:", X_test.shape)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_models = {}
    best_model_scores = {}

    for clf_name, (clf_template, param_dict) in classifiers_to_use.items():
        print(f"\n=== Tuning {clf_name} ===")
        param_list = list(ParameterGrid(param_dict))
        total_candidates = len(param_list)

        best_score = -1.0
        best_params = None
        best_model = None

        for idx, params in enumerate(param_list):
            print(f"\n--> (Candidate {idx+1}/{total_candidates}) => {params}")
            model = clf_template.set_params(**params)
            fold_scores = []

            for fold_i, (train_idxF, val_idxF) in enumerate(cv.split(X_train, y_train)):
                X_trF = X_train.iloc[train_idxF]
                X_valF = X_train.iloc[val_idxF]
                y_trF = y_train.iloc[train_idxF]
                y_valF = y_train.iloc[val_idxF]

                model.fit(X_trF, y_trF)
                acc_fold = model.score(X_valF, y_valF)
                print(f"    [fold {fold_i+1}/{cv.n_splits}] => accuracy={acc_fold:.4f}")
                fold_scores.append(acc_fold)

            mean_cv = np.mean(fold_scores)
            print(f"    => Candidate mean CV score={mean_cv:.4f}")
            if mean_cv > best_score:
                best_score = mean_cv
                best_params = params
                best_model = clf_template.set_params(**params)

        best_model.fit(X_train, y_train)
        print(f"\n[INFO] Best params for {clf_name}: {best_params}")
        print(f"[INFO] Best CV score for {clf_name}: {best_score:.4f}")

        # Evaluate final test
        print(f"\n--- Final Test Evaluation: {clf_name} ---")
        evaluate_model(best_model, X_test, y_test)

        best_models[clf_name] = best_model
        best_model_scores[clf_name] = best_score

    if USE_ENSEMBLE and len(best_models) > 1:
        print("\n[INFO] Combining best models into a soft VotingClassifier.")
        estimators = [(n, m) for n, m in best_models.items()]
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        ensemble.fit(X_train, y_train)
        print("\n--- Final Test Evaluation: Ensemble (soft) ---")
        evaluate_model(ensemble, X_test, y_test)
        return ensemble

    # Pick the single best model out of best_models
    best_model_name = max(best_model_scores, key=best_model_scores.get)
    final_model = best_models[best_model_name]
    print(f"\n[INFO] Among chosen models, '{best_model_name}' had the best CV score = {best_model_scores[best_model_name]:.4f}.")
    print("[INFO] Returning that single model for predictions.")

    return final_model


# ------------------------- Prediction -------------------------
def print_activity_mapping():
    """
    Prints the official 15 activities, showing their numeric IDs and descriptive names.
    """
    print("\nActivity Labels:")
    for act_id, act_name in sorted(ACTIVITY_LABELS.items()):
        print(f"  Activity {act_id} => {act_name}")

def predict_activity_for_single_activity(user_id, activity_id, trained_model):
    """
    Loads the 4 sensor CSVs for (user_id, activity_id), windows them,
    extracts features, then does majority-vote prediction among the windows.
    """
    print(f"\n[INFO] Predicting activity for User {user_id}, Activity {activity_id}...\n")

    activity_dir = os.path.join(DATASET_PATH, f"User{user_id}", f"Activity{activity_id}")
    acc_file   = os.path.join(activity_dir, "Accelerometer.csv")
    gyro_file  = os.path.join(activity_dir, "Gyroscope.csv")
    mag_file   = os.path.join(activity_dir, "Magnetometer.csv")
    press_file = os.path.join(activity_dir, "Pressure.csv")

    acc_df   = load_sensor_data(acc_file)
    gyro_df  = load_sensor_data(gyro_file)
    mag_df   = load_sensor_data(mag_file)
    press_df = load_sensor_data(press_file)

    acc_w   = window_data(acc_df, WINDOW_SIZE)
    gyro_w  = window_data(gyro_df, WINDOW_SIZE)
    mag_w   = window_data(mag_df, WINDOW_SIZE)
    press_w = window_data(press_df, WINDOW_SIZE)

    n_w = min(len(acc_w), len(gyro_w), len(mag_w), len(press_w))
    if n_w == 0:
        print("[WARNING] No complete windows => can't predict.")
        return None

    all_feats = []
    for i in range(n_w):
        feats = {}
        feats.update(extract_features("accelerometer", acc_w[i]))
        feats.update(extract_features("gyroscope",     gyro_w[i]))
        feats.update(extract_features("magnetometer",  mag_w[i]))
        feats.update(extract_features("pressure",      press_w[i]))
        all_feats.append(feats)

    df_features = pd.DataFrame(all_feats).fillna(0)

    y_pred = trained_model.predict(df_features)
    majority_label = pd.Series(y_pred).value_counts().idxmax()
    print(f"[PREDICTION] Majority predicted class = {majority_label}")

    try:
        label_int = int(majority_label)
        if label_int in ACTIVITY_LABELS:
            print(f"[PREDICTION] This corresponds to: {ACTIVITY_LABELS[label_int]}")
    except ValueError:
        pass

    return majority_label


# ------------------------- Main Pipeline -------------------------
def main():
    print("\n[INFO] Starting the pipeline...")
    print_activity_mapping()

    # Phase 1: Check for feature CSV
    if os.path.exists(OUTPUT_FEATURE_CSV):
        print(f"\n[INFO] Found existing {OUTPUT_FEATURE_CSV}; skipping feature generation.")
        features_df = pd.read_csv(OUTPUT_FEATURE_CSV)
    else:
        print("\n[INFO] No feature CSV found; generating from raw sensor data...\n")
        all_rows = []
        for user_id in range(13, 28):
            num_activities = get_activity_count(user_id)
            if num_activities == 0:
                continue
            for activity_id in range(1, num_activities + 1):
                activity_dir = os.path.join(DATASET_PATH, f"User{user_id}", f"Activity{activity_id}")
                acc_file   = os.path.join(activity_dir, "Accelerometer.csv")
                gyro_file  = os.path.join(activity_dir, "Gyroscope.csv")
                mag_file   = os.path.join(activity_dir, "Magnetometer.csv")
                press_file = os.path.join(activity_dir, "Pressure.csv")

                acc_df   = load_sensor_data(acc_file)
                gyro_df  = load_sensor_data(gyro_file)
                mag_df   = load_sensor_data(mag_file)
                press_df = load_sensor_data(press_file)

                # We skip plotting here, focusing only on feature generation.
                # If you prefer to plot here as well, call `plot_four_sensors(...)`.

                # Window each sensor
                acc_w = window_data(acc_df, WINDOW_SIZE)
                gyr_w = window_data(gyro_df, WINDOW_SIZE)
                mag_w = window_data(mag_df, WINDOW_SIZE)
                prs_w = window_data(press_df, WINDOW_SIZE)

                n_w = min(len(acc_w), len(gyr_w), len(mag_w), len(prs_w))
                for i in range(n_w):
                    feats = {}
                    feats.update(extract_features("accelerometer", acc_w[i]))
                    feats.update(extract_features("gyroscope",     gyr_w[i]))
                    feats.update(extract_features("magnetometer",  mag_w[i]))
                    feats.update(extract_features("pressure",      prs_w[i]))
                    feats["user_id"]     = user_id
                    feats["activity_id"] = activity_id
                    all_rows.append(feats)

        features_df = pd.DataFrame(all_rows)
        print(f"\n[INFO] Feature extraction complete. Shape = {features_df.shape}")
        features_df.to_csv(OUTPUT_FEATURE_CSV, index=False)
        print(f"[INFO] Saved features to {OUTPUT_FEATURE_CSV}")

    # Phase 2: If PLOTTING=True, run a separate pipeline for plotting all user-activity sensor data
    if PLOTTING:
        print("\n[INFO] Plotting pipeline started...")
        plot_all_activities()

    # Phase 3: Classification
    print("\n[INFO] Classification pipeline started...")
    best_model = run_classification(features_df)
    print("\n[INFO] Classification step complete. Proceeding with single-activity prediction...")

    # Phase 4: Single-activity prediction
    predict_activity_for_single_activity(PREDICTION_USER, PREDICTION_ACTIVITY, best_model)
    print("\n[INFO] Pipeline finished successfully.\n")


if __name__ == "__main__":
    main()
