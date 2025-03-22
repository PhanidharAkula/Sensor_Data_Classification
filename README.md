# Sensor Data Classification

Author: **Phanidhar Akula**

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Data Description](#2-data-description)
3. [Project Structure](#3-project-structure)
4. [Installation and Requirements](#4-installation-and-requirements)
5. [Feature Extraction Pipeline](#5-feature-extraction-pipeline)
6. [Plotting Pipeline (Optional)](#6-plotting-pipeline-optional)
7. [Classification Pipeline](#7-classification-pipeline)
   - [Models and Hyperparameters](#models-and-hyperparameters)
   - [Training and Cross-Validation](#training-and-cross-validation)
   - [Ensembling](#ensembling)
   - [Evaluation Metrics](#evaluation-metrics)
8. [Single-Activity Prediction Demo](#8-single-activity-prediction-demo)
9. [Running the Entire Pipeline](#9-running-the-entire-pipeline)
10. [Results and Logs](#10-results-and-logs)
11. [Conclusion and Future Work](#11-conclusion-and-future-work)
12. [Code Reference](#12-code-reference)

---

## 1. Introduction

This project **classifies human activities** based on **multi-sensor data**:

- Accelerometer
- Gyroscope
- Magnetometer
- Pressure

The data is collected from multiple users (IDs 13..27 or 28) performing different activities (15 total, although some users may have fewer).

The project provides a **complete pipeline**:

1. **Data ingestion** from raw CSVs.
2. **Windowing** and **feature extraction** (statistical features) for each sensor axis.
3. **Optional plotting** to visually verify sensor readings.
4. **Training** various classification models (Logistic Regression, Naive Bayes, Decision Tree, SVM, Random Forest) with **5-fold cross-validation** and hyperparameter tuning.
5. **Ensembling** the best models (optional).
6. **Predicting** a single user’s single activity to validate end-to-end correctness.

---

## 2. Data Description

- The data is located in `./DataSet/`:

  ```
  ./DataSet/
      User13/
          Activity1/
              Accelerometer.csv
              Gyroscope.csv
              Magnetometer.csv
              Pressure.csv
          Activity2/
              ...
      User14/
          Activity1/
              ...
      ...
  ```

- Each user folder (e.g., `User13`) contains subfolders for activities (e.g., `Activity1` through `Activity15`, though some have fewer).
- Each activity folder has up to 4 CSV files, each storing raw sensor readings:

  - **Accelerometer.csv**
  - **Gyroscope.csv**
  - **Magnetometer.csv**
  - **Pressure.csv**

- In each CSV:
  - Timestamps (e.g., `time (-13:00)`)
  - Sensor-specific axes (e.g., `x-axis (g)` for accelerometer)
  - Some CSV files may be missing or have different naming patterns (the pipeline handles missing files gracefully).

---

## 3. Project Structure

A minimal layout for this repository might look like:

```
.
├─ DataSet/
│   ├─ User13/
│   │   ├─ Activity1/
│   │   │   ├─ Accelerometer.csv
│   │   │   ├─ Gyroscope.csv
│   │   │   ├─ Magnetometer.csv
│   │   │   └─ Pressure.csv
│   │   ├─ Activity2/
│   │   └─ ...
│   ├─ User14/
│   │   └─ ...
│   └─ ...
│
├─ Plottings/
│   └─ ... (Generated plots saved here)
│
├─ Dataset_AllSensors_Featured.csv
│   └─ ... (Generated featured csv saved here)
│
├─ main.py
└─ README.md  (This file)
```

- **DataSet/** – Raw data directories.
- **Plottings/** – Generated plots (if `PLOTTING=True`).
- **main.py** – Main pipeline script.
- **DataSet_AllSensors_Featured.csv** (generated if it doesn’t already exist) – Feature-extracted dataset.

---

## 4. Installation and Requirements

### Using `requirements.txt`

1. **Create and activate a virtual environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux or macOS
   venv\Scripts\activate     # On Windows
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

> **Note**: The `requirements.txt` file should list packages like:
>
> ```
> pandas
> numpy
> matplotlib
> scikit-learn
> ```
>
> (Adjust as necessary for your environment.)

### Ensure Data Availability

- Place your sensor CSV files in `./DataSet/` with the described folder structure.
- By default, the script expects user IDs in the range [13..27]. Adjust if needed in `main.py`.

---

## 5. Feature Extraction Pipeline

1. **Windowing**

   - The script reads each sensor file (Accelerometer, Gyroscope, Magnetometer, Pressure).
   - It splits the time-series into **fixed-size windows** (`WINDOW_SIZE=100` by default).
   - Only complete windows of size 100 are retained.

2. **Extracting Features**

   - For each window and each sensor axis, the pipeline computes:
     - **Mean**
     - **Median**
     - **Standard Deviation**
     - **Minimum**
     - **Maximum**
   - These statistics are stored in columns such as `acc_x_mean`, `acc_x_std`, `gyro_x_mean`, etc.

3. **Combining All Sensors**

   - Features for Accelerometer, Gyroscope, Magnetometer, and Pressure (if present) are combined into **one row** per window.
   - Each row also records `user_id` and `activity_id` as labels.

4. **Caching the Result**
   - If `DataSet_AllSensors_Featured.csv` is not found, the script **generates** it automatically by iterating through all users (13..27) and their respective activities.
   - Once generated, subsequent runs will detect this file and skip re-generation, saving time.

---

## 6. Plotting Pipeline (Optional)

If `PLOTTING = True` in `main.py`, the script performs **sensor data plotting**:

- It reads each user’s raw sensor files (Accelerometer, Gyroscope, Magnetometer, Pressure).
- Generates a **4-panel figure** with each sensor’s time-series.
- Saves the figure as a `.png` in `./Plottings/` (one file per user-activity).

This is useful to **visually inspect** the sensor signals or identify anomalies.  
If you do **not** want to generate plots, set `PLOTTING = False`.

---

## 7. Classification Pipeline

After feature extraction, the script will:

1. **Load** the features from `DataSet_AllSensors_Featured.csv`.
2. **Split** the data into **train** and **test** sets (`test_size=0.20`) with stratification on `activity_id`.
3. **Train** one or more classifiers, each with its own hyperparameter grid.

### Models and Hyperparameters

By default, we support these models:

- **LogisticRegression**
  - `C ∈ {0.01, 0.1, 1, 10}`, `penalty ∈ {l1, l2}`
- **NaiveBayes (GaussianNB)**
  - `var_smoothing ∈ {1e-9, 1e-8, 1e-7}`
- **DecisionTree**
  - `max_depth ∈ {None, 5, 10, 20}`, `min_samples_leaf ∈ {1, 2, 5}`
- **SVM**
  - `C ∈ {0.1, 1, 10}`, `kernel ∈ {linear, rbf}`
- **RandomForest**
  - `n_estimators ∈ {50, 100}`, `max_depth ∈ {None, 10, 20}`, `min_samples_leaf ∈ {1, 2}`

You can pick which classifiers to run by editing `SELECT_MODELS` in `main.py`.

### Training and Cross-Validation

- **5-Fold Stratified Cross-Validation** is used for each model’s hyperparameter grid.
- For each (model, hyperparam) combination:
  1. Fit on 4 folds
  2. Validate on the 5th
  3. Compute mean accuracy
- The combination with the **best mean CV score** is selected.

### Ensembling

If `USE_ENSEMBLE = True` and more than one model is selected, the script:

- Builds a **soft-voting** ensemble (`VotingClassifier`) of the best individual models.
- Retrains them on the full training set, then evaluates on the test set.

If `USE_ENSEMBLE = False` or only one model is selected, the script picks **the single best** classifier.

### Evaluation Metrics

Metrics are computed on the **test set** after final model retraining:

- **Accuracy** (fraction of correct predictions)
- **Precision** (weighted average across classes)
- **Recall** (weighted average across classes)
- **F1-Score** (weighted average across classes)
- **AUC** (if the problem is binary and `predict_proba` is available)

---

## 8. Single-Activity Prediction Demo

After training, we demonstrate classification on **one specific user’s** single activity (specified by `PREDICTION_USER` and `PREDICTION_ACTIVITY` in `main.py`).

1. The script loads the raw sensor CSVs for that user-activity.
2. Windows them (same `WINDOW_SIZE`), extracts the same features.
3. Makes predictions per window, then determines a **majority** predicted activity.
4. Prints the final predicted activity, mapping the numeric code (1..15) to a more descriptive label (e.g., “Standing”).

Example snippet:

```
[INFO] Predicting activity for User 16, Activity 8...
[PREDICTION] Majority predicted class = 8
[PREDICTION] This corresponds to: Standing
```

---

## 9. Running the Entire Pipeline

1. **Prepare your data**: Place your raw sensor data in `./DataSet/` with the required folder structure.
2. **Adjust configurations** in `main.py`:
   - `DATASET_PATH` (default is `./DataSet`)
   - `WINDOW_SIZE`
   - `PLOTTING`
   - `USE_ENSEMBLE`
   - `SELECT_MODELS`
   - `PREDICTION_USER`, `PREDICTION_ACTIVITY`
3. **Run the script**:
   ```bash
   python main.py
   ```
4. **Observe the output**. It will:
   - Generate or detect `DataSet_AllSensors_Featured.csv`
   - Optionally produce sensor plots in `./Plottings/`
   - Print cross-validation progress and final test metrics
   - Print single-activity prediction results

---

## 10. Results and Logs

During execution, you’ll see logging like:

```
[INFO] Starting classification process...
=== Tuning RandomForest ===
--> (Candidate 1/8) => {'n_estimators': 50, 'max_depth': 10, 'min_samples_leaf': 1}
[fold 1/5] => accuracy=0.6893
...
Best CV score for RandomForest: 0.7250

--- Final Test Evaluation: RandomForest ---
Accuracy:  0.7050
Precision: 0.7012
Recall:    0.7050
F1 Score:  0.6901
```

If you enable the ensemble, it will display something like:

```
[INFO] Combining best models into a soft VotingClassifier.

--- Final Test Evaluation: Ensemble (soft) ---
Accuracy:  0.7200
Precision: 0.7150
Recall:    0.7200
F1 Score:  0.7100
```

---

## 11. Conclusion and Future Work

This project demonstrates a **complete pipeline** for sensor-based human activity recognition, from raw data reading and plotting to final classification. Key features:

- **Multi-sensor feature extraction**
- **Flexible** training with multiple ML algorithms
- **Hyperparameter tuning** with cross-validation
- **Optional ensembling** for improved accuracy
- **Real-time style** single-activity prediction

**Possible future enhancements**:

- **Deep Learning** (LSTM/GRU/CNN) for time-series classification.
- **Advanced hyperparameter optimization** (e.g., Bayesian search).
- **Additional domain-specific features** or feature selection.
- **Real-time streaming** classification.

---

## 12. Code Reference

Below is the reference to the **main.py** script (truncated for brevity). Please see the file in the repository for the full code.

<details>
<summary>Click to expand main.py snippet</summary>

```python
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# ----------------------------- Configuration -----------------------------
DATASET_PATH = "./DataSet"
OUTPUT_FEATURE_CSV = "DataSet_AllSensors_Featured.csv"
WINDOW_SIZE = 100
USE_ENSEMBLE = True
PLOTTING = True
TEST_SIZE = 0.20
PREDICTION_USER = 16
PREDICTION_ACTIVITY = 8
SELECT_MODELS = ["NaiveBayes", "DecisionTree", "RandomForest"]

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

# ... (remaining code for data loading, feature extraction, classification, main pipeline) ...
```

</details>

---

**Thank you for exploring this project.** If you have any questions, feedback, or improvements, feel free to open an issue or submit a pull request!

---

_© 2025 Phanidhar Akula_
