"""
Module 5 Week A — Lab: Regression & Evaluation

Build and evaluate logistic and linear regression models on the
Petra Telecom customer churn dataset.

Run: python lab_regression.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                             mean_absolute_error, r2_score,
                             accuracy_score, precision_score, 
                             recall_score, f1_score, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt


def load_data(filepath=None):
    """Load the telecom churn dataset.

    Returns:
        DataFrame with all columns.
    """
    if filepath is None:
        # Try root path first, then nested path for local running
        filepath = "data/telecom_churn.csv" if os.path.exists("data/telecom_churn.csv") else "starter/data/telecom_churn.csv"
    return pd.read_csv(filepath)


def split_data(df, target_col, test_size=0.2, random_state=42):
    """Split data into train and test sets with stratification.

    Args:
        df: DataFrame with features and target.
        target_col: Name of the target column.
        test_size: Fraction for test set.
        random_state: Random seed.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Stratify only for classification (Task 2 vs Task 4 logic)
    stratify = y if target_col == "churned" else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)


def build_logistic_pipeline():
    """Build a Pipeline with StandardScaler and LogisticRegression.

    Returns:
        sklearn Pipeline object.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(
            random_state=42, 
            max_iter=1000, 
            class_weight="balanced"
        ))
    ])


def build_ridge_pipeline():
    """Build a Pipeline with StandardScaler and Ridge regression.

    Returns:
        sklearn Pipeline object.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=1.0))
    ])


def build_lasso_pipeline():
    """Build a Pipeline with StandardScaler and Lasso regression.

    Returns:
        sklearn Pipeline object.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', Lasso(alpha=0.1))
    ])


def evaluate_classifier(pipeline, X_train, X_test, y_train, y_test):
    """Train the pipeline and return classification metrics.

    Args:
        pipeline: sklearn Pipeline with a classifier.
        X_train, X_test: Feature arrays.
        y_train, y_test: Label arrays.

    Returns:
        Dictionary with keys: 'accuracy', 'precision', 'recall', 'f1'.
    """
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Visual display (Task 3)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }


def evaluate_regressor(pipeline, X_train, X_test, y_train, y_test):
    """Train the pipeline and return regression metrics.

    Args:
        pipeline: sklearn Pipeline with a regressor.
        X_train, X_test: Feature arrays.
        y_train, y_test: Target arrays.

    Returns:
        Dictionary with keys: 'mae', 'r2'.
    """
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    return {
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }


def run_cross_validation(pipeline, X_train, y_train, cv=5):
    """Run stratified cross-validation on the pipeline.

    Args:
        pipeline: sklearn Pipeline.
        X_train: Training features.
        y_train: Training labels.
        cv: Number of folds.

    Returns:
        Array of cross-validation scores.
    """
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv_splitter, scoring="accuracy")
    return scores


if __name__ == "__main__":
    df = load_data()
    if df is not None:
        # --- TASK 1: Basic EDA ---
        print(f"--- Task 1: EDA ---")
        print(f"Shape: {df.shape}")
        print(f"Missing Values:\n{df.isnull().sum()}")
        print(f"Churn Distribution:\n{df['churned'].value_counts(normalize=True)}")
        print("-" * 30)

        # Define common features
        features = ["tenure", "monthly_charges", "total_charges",
                           "num_support_calls", "senior_citizen",
                           "has_partner", "has_dependents"]

        # --- TASK 2: Split with Stratification Check ---
        df_cls = df[features + ["churned"]].dropna()
        X_train, X_test, y_train, y_test = split_data(df_cls, "churned")
        
        print("--- Task 2: Split Confirmation ---")
        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        print(f"Train Churn Rate: {y_train.mean():.4f}")
        print(f"Test Churn Rate: {y_test.mean():.4f}")
        print("-" * 30)

        # --- TASK 3: Classification Evaluation ---
        pipe = build_logistic_pipeline()
        metrics = evaluate_classifier(pipe, X_train, X_test, y_train, y_test)
        print(f"Logistic Regression Metrics: {metrics}")
        plt.show() # Call show once here so it doesn't block evaluation logic

        # --- TASK 6: Cross-Validation ---
        print("\n--- Task 6: Cross-Validation ---")
        scores = run_cross_validation(pipe, X_train, y_train)
        for i, score in enumerate(scores):
            print(f"Fold {i+1} Accuracy: {score:.4f}")
        print(f"Overall CV Accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")
        print("-" * 30)

        # --- TASK 4 & 5: Regression (Ridge vs Lasso) ---
        # For regression, monthly_charges is the target, so we use other features
        df_reg = df[features].dropna() 
        X_tr, X_te, y_tr, y_te = split_data(df_reg, "monthly_charges")
        
        ridge_pipe = build_ridge_pipeline()
        reg_metrics = evaluate_regressor(ridge_pipe, X_tr, X_te, y_tr, y_te)
        print(f"Ridge Regression Metrics: {reg_metrics}")

        lasso_pipe = build_lasso_pipeline()
        lasso_pipe.fit(X_tr, y_tr)
        
        # Inspect coefficients from the fitted models within the pipelines
        r_coefs = ridge_pipe.named_steps['ridge'].coef_
        l_coefs = lasso_pipe.named_steps['lasso'].coef_
        
        print("\n--- Task 5: Regularization Comparison ---")
        print(f"{'Feature':<20} | {'Ridge Coef':>10} | {'Lasso Coef':>10}")
        print("-" * 46)
        for name, r_c, l_c in zip(X_tr.columns, r_coefs, l_coefs):
            print(f"{name:<20} | {r_c:10.4f} | {l_c:10.4f}")
        print("-" * 46)

        # Final Task 7 Summary (Professional Comment)
        """
        Task 7 Summary:
        - Important Features: Total charges and tenure are the primary predictors.
        - Performance: The model shows a balanced recall (0.51) for identifying churners,
          which is prioritized over precision due to the cost of customer loss.
        - Next Steps: Consider hyperparameter tuning or non-linear models for better fit.
        """