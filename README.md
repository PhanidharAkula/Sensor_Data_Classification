Sensor Data Classification

Author : Phanidhar Akula

## 1. Introduction

This project aims to **classify human activities** based on **sensor data** (accelerometer, gyroscope, magnetometer, and pressure) collected from multiple users. Specifically, we:

1. **Load** raw sensor CSVs from `DataSet2`, each containing multiple users (ID range 13..27 or 28) who performed several activities (1..15).
2. **Window** each sensor’s time-series data and **extract features** (mean, standard deviation, min, max, median) for each axis.
3. Optionally **plot** the sensor data if `PLOTTING = True`.
4. **Train** multiple ML classifiers (LogisticRegression, NaiveBayes, DecisionTree, SVM, RandomForest) with **5-fold cross-validation** for hyperparameter tuning.
5. Optionally build an **ensemble** (`VotingClassifier`) of the best models.
6. Demonstrate **single-activity prediction** on a chosen user/activity pair.

The entire pipeline is self-contained, from raw CSV reading to final classification metrics.

## 2. Data Description

- **DataSet2** is assumed to have the following structure:
  ```
  ./DataSet2/
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
- Each **User** has a set of **Activities** (1..15, or fewer if incomplete). Each activity has 4 sensor CSVs.
- Sensors have columns like `time (-13:00)`, `(x-axis (g))`, etc.
- The project includes code to handle missing files gracefully.

## 3. Feature Extraction

1. **Windowing**: We segment each sensor CSV into fixed-size windows (e.g., 100 rows).
2. **Feature Computation**: For each window, we compute:
   - **mean**, **std**, **min**, **max**, **median** for each axis (e.g., `acc_x_mean`).
3. We combine features from each sensor into one row, adding `user_id` and `activity_id` columns to label them.

If no existing CSV with features is found (named by default `DataSet2_AllSensors_Featured.csv`), the pipeline auto-generates it.

## 4. Plotting Pipeline (Optional)

When `PLOTTING = True`:

- The script (in a separate step) **loads raw sensor data** for each user/activity.
- Generates a **4-subplot figure**: (Accelerometer, Gyroscope, Magnetometer, Pressure).
- Saves the figure as a `.png` in a `Plottings` folder.

This helps visually inspect and verify sensor signals or identify possible anomalies.

## 5. Classification

1. **Train/Test Split**  
   We split the extracted feature data (`features_df`) into **train** and **test** sets (`test_size=0.20` by default). We stratify on `activity_id` to keep balanced classes.

2. **Multiple Classifiers**  
   By default, we run:

   - **LogisticRegression** (penalty ∈ {l1, l2}, C ∈ {0.01, 0.1, 1, 10})
   - **NaiveBayes** (var_smoothing ∈ {1e-9, 1e-8, 1e-7})
   - **DecisionTree** (max_depth, min_samples_leaf combos)
   - **SVM** (C ∈ {0.1, 1, 10}, kernel ∈ {linear, rbf})
   - **RandomForest** (n_estimators ∈ {50, 100}, max_depth, min_samples_leaf combos)

3. **5-Fold Cross-Validation & Hyperparam Tuning**

   - For each classifier, we do a grid search over the specified hyperparams.
   - For each combination, we **fit** on 4 folds and **evaluate** on the remaining fold.
   - We record the **mean CV accuracy** and choose the **best** hyperparameters.

4. **Ensemble**
   - If `USE_ENSEMBLE=True` and more than one classifier is chosen, we build a **soft-voting** ensemble from the best models.
   - The final model is tested again on the hold-out test set, printing metrics.

## 6. Evaluation Metrics

- **Accuracy**: fraction of correct predictions.
- **Precision**, **Recall**, **F1 Score**: computed in a _weighted_ fashion across classes (scikit-learn’s `average='weighted'`).
- If the problem is binary (2 classes only) and the model supports `predict_proba`, we also compute **AUC**.

Typically, we see logs like:

```
[INFO] Starting classification process...
=== Tuning RandomForest ===
--> (Candidate 1/8) => {'n_estimators': 50, 'max_depth': 10, 'min_samples_leaf': 1}
[fold 1/5] => accuracy=...
...
Best CV score for RandomForest: 0.72
--- Final Test Evaluation: RandomForest ---
Accuracy:  0.7050
Precision: 0.7012
Recall:    0.7050
F1 Score:  0.6901
...
```

## 7. Single-Activity Prediction

After classification, we demonstrate an **activity prediction** on a single user’s single activity (e.g., `User16, Activity8`). The code:

- Loads the 4 sensor CSVs for that user/activity.
- Windows them, extracts the same features.
- Produces predictions across all windows, then picks the **majority** label.
- If the label is numeric 1..15, it prints the descriptive name (e.g., “**Standing**”).

For example:

```
[INFO] Predicting activity for User 16, Activity 8...
[PREDICTION] Majority predicted class = 8
[PREDICTION] This corresponds to: Standing
```

## 8. Conclusion

With this pipeline:

- We can **automate** the feature extraction from multiple raw sensor CSV files.
- We optionally **visualize** each user-activity’s sensor data as 4 subplots.
- We **train** multiple ML classifiers, do hyperparam tuning, and possibly build an ensemble.
- We **evaluate** the final model on a test set.
- We **predict** an activity from an unseen single user’s raw data, verifying end-to-end correctness.

**Potential Next Steps**:

- Incorporate advanced deep learning models (LSTM, CNN) for time-series classification.
- Integrate more sophisticated hyperparam search (e.g., Bayesian optimization).
- Evaluate on real mobile-sensor data or additional domain-specific features.

**Overall**, this pipeline demonstrates a robust approach to sensor-based activity recognition, from **data ingestion** to **final classification metrics**.
