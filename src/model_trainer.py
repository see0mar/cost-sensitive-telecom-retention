# -*- coding: utf-8 -*-
"""
src/model_trainer.py
Trains the LightGBM champion model and logistic-regression / decision-tree baselines.
"""
from typing import Dict
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


class RetentionModelTrainer:
    """
    Manages training of the LightGBM champion model and comparison baselines.

    Notes
    -----
    scale_pos_weight defaults to 1.0 (no reweighting) so that LightGBM
    outputs conservative probability scores by default — which makes the
    cost-sensitive threshold optimisation story most legible.
    Use train_lightgbm(params={..., 'scale_pos_weight': spw}) or call
    calculate_imbalance_weight() when you want explicit imbalance correction.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def calculate_imbalance_weight(self, y_train: pd.Series) -> float:
        """neg_count / pos_count — pass the result to scale_pos_weight."""
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        w = neg / pos
        print(f"  [Trainer] scale_pos_weight = {w:.2f}  (neg:{neg:,}  pos:{pos:,})")
        return float(w)

    def train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        params: dict | None = None,
    ) -> lgb.LGBMClassifier:
        """Train the LightGBM champion model."""
        if params is None:
            params = {
                "objective":        "binary",
                "metric":           "auc",
                "boosting_type":    "gbdt",
                "n_estimators":     300,
                "learning_rate":    0.03,
                "num_leaves":       31,
                "max_depth":        6,
                "scale_pos_weight": 1.0,   # conservative → threshold optimisation has clear impact
                "random_state":     self.random_state,
                "n_jobs":           -1,
                "verbose":          -1,
            }
        print("  [Trainer] Training LightGBM champion model...")
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        print("  [Trainer] LightGBM training complete.")
        return model

    def train_baselines(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Dict[str, object]:
        """Train LR and DT baselines."""
        print("  [Trainer] Training baseline models...")
        lr = LogisticRegression(max_iter=1000, random_state=self.random_state)
        dt = DecisionTreeClassifier(max_depth=6, random_state=self.random_state)
        lr.fit(X_train, y_train);  print("    Logistic Regression — done.")
        dt.fit(X_train, y_train);  print("    Decision Tree       — done.")
        return {"logistic_regression": lr, "decision_tree": dt}

    @staticmethod
    def predict_proba(model, X_test: pd.DataFrame) -> np.ndarray:
        """Return positive-class probability scores."""
        return model.predict_proba(X_test)[:, 1]
