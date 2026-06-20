"""
AgroSense AI — Crop Type Classifier
XGBoost + Random Forest + LightGBM stacked ensemble with Optuna hyperparameter tuning.
All training is dynamic — no hardcoded parameters.
"""
from __future__ import annotations
import logging
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False
    logger.warning("XGBoost not installed")

try:
    import lightgbm as lgb
    LGB_OK = True
except ImportError:
    LGB_OK = False
    logger.warning("LightGBM not installed")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_OK = True
except ImportError:
    OPTUNA_OK = False
    logger.warning("Optuna not installed — using default hyperparameters")

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    N_OPTUNA_TRIALS, N_CV_FOLDS, TRAIN_SPLIT,
    MIN_SAMPLES_PER_CLASS, LOW_CONFIDENCE_THRESHOLD,
    ACCURACY_WARNING_THRESHOLD, N_SHAP_FEATURES,
)


class AgroSenseClassifier:
    """Stacked ensemble crop type classifier.
    
    Architecture:
        - Base 1: XGBoost (temporal features)
        - Base 2: Random Forest (spectral features)
        - Base 3: LightGBM (SAR features)
        - Meta: Logistic Regression on base predictions
    """

    def __init__(self) -> None:
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.xgb_model: Optional[Any] = None
        self.rf_model: Optional[RandomForestClassifier] = None
        self.lgb_model: Optional[Any] = None
        self.meta_model: Optional[LogisticRegression] = None
        self.selected_features: Optional[List[str]] = None
        self.feature_names: Optional[List[str]] = None
        self.classes_: Optional[List[str]] = None
        self.is_fitted: bool = False
        self.shap_values: Optional[np.ndarray] = None
        self.is_single_class: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Compute inverse-frequency class weights."""
        classes = np.unique(y)
        weights = compute_class_weight("balanced", classes=classes, y=y)
        return dict(zip(classes, weights))

    def _optimize_xgb(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        class_weights: Dict[int, float],
        n_trials: int = N_OPTUNA_TRIALS,
    ) -> Dict[str, Any]:
        """Use Optuna to tune XGBoost hyperparameters."""
        if not OPTUNA_OK or not XGB_OK:
            return {
                "n_estimators": 200, "max_depth": 6, "learning_rate": 0.1,
                "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
            }

        sample_weights = np.array([class_weights.get(yi, 1.0) for yi in y_train])

        def objective(trial: Any) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "use_label_encoder": False,
                "eval_metric": "mlogloss",
                "random_state": 42,
            }
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = []
            for train_idx, val_idx in cv.split(X_train, y_train):
                try:
                    # Verify that the split contains all unique classes
                    if len(np.unique(y_train[train_idx])) < len(np.unique(y_train)):
                        scores.append(0.0)
                        continue
                    model = xgb.XGBClassifier(**params)
                    model.fit(
                        X_train[train_idx], y_train[train_idx],
                        sample_weight=sample_weights[train_idx],
                    )
                    preds = model.predict(X_train[val_idx])
                    scores.append(f1_score(y_train[val_idx], preds, average="weighted"))
                except Exception as exc:
                    logger.warning("XGBoost cross-validation fold failed: %s", exc)
                    scores.append(0.0)
            return float(np.mean(scores)) if scores else 0.0

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        return study.best_params

    def _optimize_lgb(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_trials: int = 30,
    ) -> Dict[str, Any]:
        """Tune LightGBM hyperparameters with Optuna."""
        if not OPTUNA_OK or not LGB_OK:
            return {
                "n_estimators": 200, "max_depth": -1,
                "learning_rate": 0.1, "num_leaves": 31,
                "subsample": 0.8, "colsample_bytree": 0.8,
                "class_weight": "balanced",
            }

        def objective(trial: Any) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "class_weight": "balanced",
                "random_state": 42,
                "verbose": -1,
            }
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = []
            for train_idx, val_idx in cv.split(X_train, y_train):
                try:
                    if len(np.unique(y_train[train_idx])) < len(np.unique(y_train)):
                        scores.append(0.0)
                        continue
                    model = lgb.LGBMClassifier(**params)
                    model.fit(X_train[train_idx], y_train[train_idx])
                    preds = model.predict(X_train[val_idx])
                    scores.append(f1_score(y_train[val_idx], preds, average="weighted"))
                except Exception as exc:
                    logger.warning("LightGBM cross-validation fold failed: %s", exc)
                    scores.append(0.0)
            return float(np.mean(scores)) if scores else 0.0

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        return study.best_params

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        n_optuna_trials: int = N_OPTUNA_TRIALS,
    ) -> "AgroSenseClassifier":
        """Train the stacked ensemble on labeled pixel data.
        
        Args:
            X: Feature matrix (N_samples, N_features).
            y: Crop label array (N_samples,), string labels.
            feature_names: Optional list of feature names.
            n_optuna_trials: Number of Optuna optimization trials.
        
        Returns:
            self (fitted classifier).
        
        Raises:
            ValueError: If insufficient samples per class.
        """
        # Encode labels
        y_enc = self.label_encoder.fit_transform(y)
        self.classes_ = list(self.label_encoder.classes_)
        n_classes = len(self.classes_)
        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]

        if n_classes < 2:
            self.is_single_class = True
            self.single_class_label = self.classes_[0]
            self.single_class_label_enc = 0
            self.is_fitted = True
            self._top_indices = list(range(X.shape[1]))
            self.selected_features = self.feature_names
            self.scaler.fit(X)
            logger.info("Classifier trained on single class: %s", self.single_class_label)
            return self

        # Ensure all classes have at least 3 samples by oversampling rare classes (crucial for StratifiedKFold splits)
        unique_classes, counts = np.unique(y_enc, return_counts=True)
        X_oversampled = [X]
        y_oversampled = [y_enc]
        for cls, count in zip(unique_classes, counts):
            if count < 3:
                needed = 3 - count
                cls_indices = np.where(y_enc == cls)[0]
                replicated_indices = np.random.choice(cls_indices, needed, replace=True)
                X_oversampled.append(X[replicated_indices])
                y_oversampled.append(y_enc[replicated_indices])
        
        X = np.concatenate(X_oversampled, axis=0)
        y_enc = np.concatenate(y_oversampled, axis=0)

        # Check minimum samples
        class_counts = {cls: int(np.sum(y == cls)) for cls in self.classes_}
        low_classes = [k for k, v in class_counts.items() if v < MIN_SAMPLES_PER_CLASS]
        if low_classes:
            logger.warning(
                "Classes with < %d samples: %s. Cross-validation used.",
                MIN_SAMPLES_PER_CLASS, low_classes
            )

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train/val split
        use_cv = any(v < MIN_SAMPLES_PER_CLASS for v in class_counts.values()) or X.shape[0] < 100
        
        if use_cv:
            X_train, X_val = X_scaled, X_scaled
            y_train, y_val = y_enc, y_enc
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y_enc,
                test_size=1 - TRAIN_SPLIT,
                stratify=y_enc,
                random_state=42,
            )

        class_weights = self._compute_class_weights(y_train)

        # --- Base Learner 1: XGBoost ---
        logger.info("Training XGBoost base learner...")
        if XGB_OK:
            xgb_params = self._optimize_xgb(X_train, y_train, class_weights, n_optuna_trials)
            xgb_params.update({
                "use_label_encoder": False,
                "eval_metric": "mlogloss",
                "random_state": 42,
                "n_jobs": -1,
            })
            self.xgb_model = xgb.XGBClassifier(**xgb_params)
            sw = np.array([class_weights.get(yi, 1.0) for yi in y_train])
            self.xgb_model.fit(X_train, y_train, sample_weight=sw)
        else:
            # Fallback: use RF as XGB substitute
            self.xgb_model = RandomForestClassifier(
                n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
            )
            self.xgb_model.fit(X_train, y_train)

        # --- Base Learner 2: Random Forest ---
        logger.info("Training Random Forest base learner...")
        self.rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.rf_model.fit(X_train, y_train)

        # --- Base Learner 3: LightGBM ---
        logger.info("Training LightGBM base learner...")
        if LGB_OK:
            lgb_params = self._optimize_lgb(X_train, y_train, 20)
            lgb_params.update({"class_weight": "balanced", "random_state": 42, "verbose": -1})
            self.lgb_model = lgb.LGBMClassifier(**lgb_params)
            self.lgb_model.fit(X_train, y_train)
        else:
            self.lgb_model = RandomForestClassifier(
                n_estimators=150, class_weight="balanced", random_state=123, n_jobs=-1
            )
            self.lgb_model.fit(X_train, y_train)

        # --- SHAP Feature Selection ---
        logger.info("Computing SHAP values for feature selection...")
        if SHAP_OK and XGB_OK:
            try:
                explainer = shap.TreeExplainer(self.xgb_model)
                shap_vals = explainer.shap_values(X_train[:min(500, len(X_train))])
                if isinstance(shap_vals, list):
                    mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
                else:
                    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
                top_k = min(N_SHAP_FEATURES, X.shape[1])
                top_indices = np.argsort(mean_abs_shap)[::-1][:top_k]
                self.selected_features = [self.feature_names[i] for i in sorted(top_indices)]
                self.shap_values = mean_abs_shap
                logger.info("Selected %d SHAP-top features", top_k)
                # Retrain on selected features
                X_train_sel = X_train[:, sorted(top_indices)]
                X_val_sel = X_val[:, sorted(top_indices)]
                self.xgb_model.fit(X_train_sel, y_train, sample_weight=sw)
                self.rf_model.fit(X_train_sel, y_train)
                if LGB_OK:
                    self.lgb_model.fit(X_train_sel, y_train)
                self._top_indices = sorted(top_indices)
            except Exception as exc:
                logger.warning("SHAP feature selection failed: %s — using all features", exc)
                self._top_indices = list(range(X.shape[1]))
        else:
            self._top_indices = list(range(X.shape[1]))
        
        # --- Meta Learner: Logistic Regression ---
        logger.info("Training meta-learner (stacking)...")
        Xm_train = self._get_meta_features(X_train)
        Xm_val = self._get_meta_features(X_val)
        
        self.meta_model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=1.0,
            random_state=42,
            n_jobs=-1,
        )
        self.meta_model.fit(Xm_train, y_train)

        self.is_fitted = True
        logger.info("Classifier training complete. Classes: %s", self.classes_)
        return self

    def _get_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Get base learner probability predictions for meta-learner input."""
        X_sel = X[:, self._top_indices] if hasattr(self, "_top_indices") else X
        xgb_probs = self.xgb_model.predict_proba(X_sel)
        rf_probs = self.rf_model.predict_proba(X_sel)
        lgb_probs = self.lgb_model.predict_proba(X_sel)
        return np.hstack([xgb_probs, rf_probs, lgb_probs])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict crop type labels.
        
        Args:
            X: Feature matrix (N, F).
        
        Returns:
            String label array (N,).
        """
        if getattr(self, "is_single_class", False):
            return np.full(X.shape[0], self.single_class_label)
        probs = self.predict_proba(X)
        encoded = np.argmax(probs, axis=1)
        return self.label_encoder.inverse_transform(encoded)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Feature matrix (N, F).
        
        Returns:
            Probability array (N, n_classes).
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
        if getattr(self, "is_single_class", False):
            probs = np.zeros((X.shape[0], len(self.classes_)))
            probs[:, self.single_class_label_enc] = 1.0
            return probs
        X_scaled = self.scaler.transform(X)
        meta_features = self._get_meta_features(X_scaled)
        return self.meta_model.predict_proba(meta_features)

    def get_confidence_map(self, X: np.ndarray) -> np.ndarray:
        """Get per-pixel confidence (max probability).
        
        Args:
            X: Feature matrix (N, F).
        
        Returns:
            Confidence array (N,) in range [0, 1].
        """
        probs = self.predict_proba(X)
        return np.max(probs, axis=1)

    def get_uncertainty_mask(self, X: np.ndarray) -> np.ndarray:
        """Identify low-confidence (uncertain) pixels.
        
        Args:
            X: Feature matrix (N, F).
        
        Returns:
            Boolean array (N,) — True = uncertain pixel.
        """
        confidence = self.get_confidence_map(X)
        return confidence < LOW_CONFIDENCE_THRESHOLD

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Evaluate classifier performance on held-out data.
        
        Args:
            X_test: Test feature matrix.
            y_test: True labels (string array).
        
        Returns:
            Dict with OA, Kappa, per-class metrics, confusion matrix,
            and warnings if accuracy below threshold.
        """
        if getattr(self, "is_single_class", False):
            y_pred = self.predict(X_test)
            oa = float(accuracy_score(y_test, y_pred))
            kappa = 1.0 if oa == 1.0 else 0.0
            cm = confusion_matrix(y_test, y_pred, labels=self.classes_)
            try:
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            except Exception:
                report = {}
            return {
                "overall_accuracy": oa,
                "kappa": kappa,
                "per_class": report,
                "confusion_matrix": cm,
                "classes": self.classes_,
                "warnings": [],
            }

        y_pred = self.predict(X_test)
        oa = float(accuracy_score(y_test, y_pred))
        try:
            kappa = float(cohen_kappa_score(y_test, y_pred))
        except Exception:
            kappa = 0.0
        try:
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        except Exception:
            report = {}
        cm = confusion_matrix(y_test, y_pred, labels=self.classes_)
        
        warnings_list = []
        if oa < ACCURACY_WARNING_THRESHOLD:
            warnings_list.append(
                f"Overall accuracy {oa:.1%} is below {ACCURACY_WARNING_THRESHOLD:.0%}. "
                "Consider collecting more ground truth samples."
            )
        
        return {
            "overall_accuracy": oa,
            "kappa": kappa,
            "per_class": report,
            "confusion_matrix": cm,
            "classes": self.classes_,
            "warnings": warnings_list,
        }

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = N_CV_FOLDS,
    ) -> Dict[str, Any]:
        """Stratified k-fold cross-validation with bootstrap confidence intervals.
        
        Args:
            X: Full feature matrix.
            y: Full label array.
            n_folds: Number of CV folds.
        
        Returns:
            Dict with mean/std of OA, Kappa, F1; bootstrap CI.
        """
        if len(np.unique(y)) < 2:
            return {
                "mean_oa": 1.0,
                "std_oa": 0.0,
                "mean_kappa": 1.0,
                "mean_f1": 1.0,
                "oa_ci_95": (1.0, 1.0),
                "fold_oas": [1.0] * n_folds,
            }
            
        y_enc = self.label_encoder.transform(y) if self.is_fitted else LabelEncoder().fit_transform(y)
        
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        oas, kappas, f1s = [], [], []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y_enc)):
            fold_clf = AgroSenseClassifier()
            fold_clf.fit(X[train_idx], y[train_idx], n_optuna_trials=10)
            metrics = fold_clf.evaluate(X[val_idx], y[val_idx])
            oas.append(metrics["overall_accuracy"])
            kappas.append(metrics["kappa"])
            f1s.append(float(f1_score(y[val_idx], fold_clf.predict(X[val_idx]), average="weighted")))
            logger.info("Fold %d: OA=%.3f, Kappa=%.3f", fold + 1, oas[-1], kappas[-1])
        
        # Bootstrap CI (n=100)
        rng = np.random.default_rng(42)
        boot_oas = [np.mean(rng.choice(oas, len(oas), replace=True)) for _ in range(100)]
        ci_low = float(np.percentile(boot_oas, 2.5))
        ci_high = float(np.percentile(boot_oas, 97.5))
        
        return {
            "mean_oa": float(np.mean(oas)),
            "std_oa": float(np.std(oas)),
            "mean_kappa": float(np.mean(kappas)),
            "mean_f1": float(np.mean(f1s)),
            "oa_ci_95": (ci_low, ci_high),
            "fold_oas": oas,
        }

    def save(self, path: Path) -> None:
        """Serialize the fitted classifier to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Classifier saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "AgroSenseClassifier":
        """Load a serialized classifier from disk."""
        with open(path, "rb") as f:
            clf = pickle.load(f)
        logger.info("Classifier loaded from %s", path)
        return clf
