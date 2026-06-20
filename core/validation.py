"""
AgroSense AI — Validation Engine
Accuracy metrics with bootstrap CIs, ET₀ validation, stress cross-validation,
and SHAP feature importance analysis.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False


# ---------------------------------------------------------------------------
# Crop Classification Validation
# ---------------------------------------------------------------------------

def validate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    n_bootstrap: int = 100,
) -> Dict[str, Any]:
    """Comprehensive classification accuracy assessment.
    
    Args:
        y_true: True labels (N,).
        y_pred: Predicted labels (N,).
        class_names: List of class name strings.
        n_bootstrap: Number of bootstrap resamples for CI computation.
    
    Returns:
        Dict with:
          - overall_accuracy, kappa, weighted_f1
          - per_class: DataFrame with Precision, Recall, F1
          - confusion_matrix: 2D array
          - oa_ci_95: (lower, upper) bootstrap 95% CI
          - warnings: list of alert messages
    """
    oa = float(accuracy_score(y_true, y_pred))
    try:
        kappa = float(cohen_kappa_score(y_true, y_pred))
    except Exception:
        kappa = 1.0 if oa == 1.0 else 0.0
    try:
        report = classification_report(
            y_true, y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
    except Exception:
        try:
            report = classification_report(
                y_true, y_pred,
                output_dict=True,
                zero_division=0,
            )
        except Exception:
            report = {}
    
    labels = class_names or sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Per-class DataFrame
    per_class_rows = []
    for cls in labels:
        if cls in report and isinstance(report[cls], dict):
            per_class_rows.append({
                "Class": cls,
                "Precision": round(report[cls].get("precision", 0), 3),
                "Recall": round(report[cls].get("recall", 0), 3),
                "F1": round(report[cls].get("f1-score", 0), 3),
                "Support": int(report[cls].get("support", 0)),
            })
    per_class_df = pd.DataFrame(per_class_rows)
    
    weighted_f1 = float(report.get("weighted avg", {}).get("f1-score", 0))
    
    # Bootstrap CI for OA
    rng = np.random.default_rng(42)
    N = len(y_true)
    boot_oas = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, N, size=N)
        boot_oa = float(accuracy_score(y_true[idx], y_pred[idx]))
        boot_oas.append(boot_oa)
    ci_low = float(np.percentile(boot_oas, 2.5))
    ci_high = float(np.percentile(boot_oas, 97.5))
    
    warnings = []
    if oa < 0.70:
        warnings.append(
            f"⚠️ Overall accuracy ({oa:.1%}) is below 70%. "
            "Consider collecting more labelled samples or increasing temporal composites."
        )
    if kappa < 0.40:
        warnings.append(
            f"⚠️ Cohen's Kappa ({kappa:.3f}) indicates poor agreement. "
            "Class imbalance or feature insufficiency may be the cause."
        )
    
    return {
        "overall_accuracy": oa,
        "kappa": kappa,
        "weighted_f1": weighted_f1,
        "per_class": per_class_df,
        "confusion_matrix": cm,
        "class_names": labels,
        "oa_ci_95": (ci_low, ci_high),
        "warnings": warnings,
    }


def validate_with_independent_points(
    classifier: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, Any]:
    """Validate classifier on a completely independent validation set.
    
    Args:
        classifier: Fitted AgroSenseClassifier instance.
        X_val: Validation feature matrix.
        y_val: Validation labels.
    
    Returns:
        Validation metrics dict.
    """
    y_pred = classifier.predict(X_val)
    return validate_classification(
        y_true=y_val,
        y_pred=y_pred,
        class_names=classifier.classes_,
    )


# ---------------------------------------------------------------------------
# ET₀ Validation (vs. IMD/FAO station data)
# ---------------------------------------------------------------------------

def validate_et0(
    et0_computed: np.ndarray,
    et0_observed: np.ndarray,
    station_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Validate computed ET₀ against station observations.
    
    Args:
        et0_computed: Penman-Monteith ET₀ estimates (N,), mm/day.
        et0_observed: Ground-truth station ET₀ (N,), mm/day.
        station_ids: Optional list of station IDs for per-station report.
    
    Returns:
        Dict with RMSE, MAE, R², bias, scatter data, and per-station breakdown.
    """
    mask = ~(np.isnan(et0_computed) | np.isnan(et0_observed))
    comp = et0_computed[mask]
    obs = et0_observed[mask]
    
    if len(comp) < 2:
        return {"error": "Insufficient valid data pairs for ET₀ validation"}
    
    rmse = float(np.sqrt(mean_squared_error(obs, comp)))
    mae = float(mean_absolute_error(obs, comp))
    r2 = float(r2_score(obs, comp))
    bias = float(np.mean(comp - obs))
    
    scatter_df = pd.DataFrame({
        "Observed_mm_day": obs,
        "Computed_mm_day": comp,
        "Residual_mm": comp - obs,
    })
    if station_ids is not None and len(station_ids) == len(et0_computed):
        scatter_df["Station"] = np.array(station_ids)[mask]
    
    return {
        "rmse_mm": round(rmse, 3),
        "mae_mm": round(mae, 3),
        "r2": round(r2, 4),
        "bias_mm": round(bias, 3),
        "n_samples": int(len(comp)),
        "scatter_data": scatter_df,
        "interpretation": (
            f"RMSE={rmse:.2f} mm/d, MAE={mae:.2f} mm/d, R²={r2:.3f}, "
            f"Bias={bias:+.2f} mm/d"
        ),
    }


# ---------------------------------------------------------------------------
# Stress Detection Cross-Validation
# ---------------------------------------------------------------------------

def cross_validate_stress(
    vhi: np.ndarray,
    cwsi: np.ndarray,
) -> Dict[str, Any]:
    """Cross-validate VHI and CWSI stress agreement.
    
    Where both indices agree on stressed pixels → high confidence.
    Where they disagree → flag as uncertain.
    
    Args:
        vhi: VHI map (H, W), 0-100.
        cwsi: CWSI map (H, W), 0-1.
    
    Returns:
        Dict with agreement map, confidence percentages.
    """
    # Convert to binary stressed (True = stressed)
    vhi_stressed = vhi < 40.0
    cwsi_stressed = cwsi > 0.5

    agreement = vhi_stressed == cwsi_stressed
    both_stressed = vhi_stressed & cwsi_stressed
    both_unstressed = (~vhi_stressed) & (~cwsi_stressed)
    uncertain = ~agreement

    n_total = vhi.size
    return {
        "agreement_map": agreement.astype(np.int8),
        "uncertain_map": uncertain.astype(np.int8),
        "pct_high_confidence": round(float(agreement.sum()) / n_total * 100, 1),
        "pct_uncertain": round(float(uncertain.sum()) / n_total * 100, 1),
        "pct_both_stressed": round(float(both_stressed.sum()) / n_total * 100, 1),
        "pct_both_unstressed": round(float(both_unstressed.sum()) / n_total * 100, 1),
    }


# ---------------------------------------------------------------------------
# Data Quality Report
# ---------------------------------------------------------------------------

def generate_data_quality_report(
    scene_counts: List[int],
    dates: List[Tuple[str, str]],
    quality_flags: List[str],
    coverage_pct: float,
) -> Dict[str, Any]:
    """Generate a comprehensive data quality report.
    
    Args:
        scene_counts: Number of S2 scenes per composite period.
        dates: List of (start, end) date tuples.
        quality_flags: Quality flag per period ('good', 'low_data', 'no_data').
        coverage_pct: Percentage of periods with good data.
    
    Returns:
        Dict with quality summary DataFrame and overall rating.
    """
    rows = []
    for i, ((ps, pe), count, flag) in enumerate(zip(dates, scene_counts, quality_flags)):
        rows.append({
            "Period": f"{ps} → {pe}",
            "Scenes": count,
            "Quality": flag.replace("_", " ").title(),
            "Status": "✅" if flag == "good" else ("⚠️" if flag == "low_data" else "❌"),
        })
    
    df = pd.DataFrame(rows)
    
    if coverage_pct >= 80:
        rating = "Excellent"
    elif coverage_pct >= 60:
        rating = "Good"
    elif coverage_pct >= 40:
        rating = "Fair"
    else:
        rating = "Poor"
    
    return {
        "period_quality": df,
        "coverage_pct": round(coverage_pct, 1),
        "overall_rating": rating,
        "n_good": sum(1 for f in quality_flags if f == "good"),
        "n_low": sum(1 for f in quality_flags if f == "low_data"),
        "n_missing": sum(1 for f in quality_flags if f == "no_data"),
    }


# ---------------------------------------------------------------------------
# SHAP Feature Importance
# ---------------------------------------------------------------------------

def compute_shap_importance(
    classifier: Any,
    X_sample: np.ndarray,
    feature_names: List[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """Compute SHAP-based feature importance for the fitted classifier.
    
    Args:
        classifier: Fitted AgroSenseClassifier.
        X_sample: Sample of feature matrix for SHAP computation.
        feature_names: Names of features.
        top_n: Number of top features to return.
    
    Returns:
        DataFrame with Feature and MeanAbsSHAP columns, sorted descending.
    """
    if not SHAP_OK:
        logger.warning("SHAP not installed — returning RF feature importances")
        if hasattr(classifier, "rf_model") and classifier.rf_model is not None:
            importances = classifier.rf_model.feature_importances_
        else:
            return pd.DataFrame({"Feature": feature_names, "MeanAbsSHAP": 0.0})
        df = pd.DataFrame({"Feature": feature_names, "MeanAbsSHAP": importances})
        return df.sort_values("MeanAbsSHAP", ascending=False).head(top_n).reset_index(drop=True)
    
    try:
        X_scaled = classifier.scaler.transform(X_sample)
        if hasattr(classifier, "_top_indices"):
            X_sel = X_scaled[:, classifier._top_indices]
            sel_names = [feature_names[i] for i in classifier._top_indices]
        else:
            X_sel = X_scaled
            sel_names = feature_names
        
        explainer = shap.TreeExplainer(classifier.xgb_model)
        shap_vals = explainer.shap_values(X_sel[:min(200, len(X_sel))])
        
        if isinstance(shap_vals, list):
            mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
        else:
            mean_abs = np.abs(shap_vals).mean(axis=0)
        
        df = pd.DataFrame({"Feature": sel_names, "MeanAbsSHAP": mean_abs})
        return df.sort_values("MeanAbsSHAP", ascending=False).head(top_n).reset_index(drop=True)
    
    except Exception as exc:
        logger.error("SHAP computation failed: %s", exc)
        return pd.DataFrame({"Feature": feature_names[:top_n], "MeanAbsSHAP": 0.0})
