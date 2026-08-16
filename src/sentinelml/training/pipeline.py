"""Train all models, lock thresholds on val, write artifacts, select champion."""

from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from sentinelml.data.splits import split_manifest, temporal_split
from sentinelml.features.engineering import (
    FEATURE_NAMES,
    FeatureState,
    matrix_from_frame,
    transform_frame,
)
from sentinelml.models.anomaly import IsolationForestModel
from sentinelml.models.baseline import AmountRuleBaseline, DummyBaseline, LogisticModel
from sentinelml.models.neural import HAS_TORCH, SklearnMLPModel, TorchMLPModel
from sentinelml.models.tree import (
    HAS_LIGHTGBM,
    HAS_XGBOOST,
    HistGBDTModel,
    LightGBMModel,
    XGBoostModel,
)
from sentinelml.training.calibration import choose_calibration
from sentinelml.training.metrics import classification_metrics
from sentinelml.training.optuna_tune import tune_lightgbm
from sentinelml.training.registry import configure_mlflow, log_metrics_safe, promotion_gate
from sentinelml.training.selection import pick_champion
from sentinelml.training.thresholds import lock_thresholds
from sentinelml.utils.config import load_yaml
from sentinelml.utils.hashing import sha256_json
from sentinelml.utils.logging import get_logger

log = get_logger("training")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def _latency_ms(model: Any, x: np.ndarray, n: int = 64) -> dict[str, float]:
    sample = x[: min(n, len(x))]
    times: list[float] = []
    for i in range(min(30, len(sample))):
        row = sample[i : i + 1]
        t0 = time.perf_counter()
        model.predict_proba(row)
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times) if times else np.array([0.0])
    return {
        "latency_ms_p50": float(np.percentile(arr, 50)),
        "latency_ms_p95": float(np.percentile(arr, 95)),
        "latency_ms_p99": float(np.percentile(arr, 99)),
    }


def build_feature_tables(
    df: pd.DataFrame, train_frac: float, val_frac: float, label_delay_days: int
) -> tuple[Any, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split = temporal_split(df, train_fraction=train_frac, val_fraction=val_frac)
    state = FeatureState(label_delay_days=label_delay_days)
    train_f, state = transform_frame(split.train, state=state, label_delay_days=label_delay_days)
    val_f, state = transform_frame(split.val, state=state, label_delay_days=label_delay_days)
    test_f, _ = transform_frame(split.test, state=state, label_delay_days=label_delay_days)
    return split, train_f, val_f, test_f


def _instantiate_models(cfg: dict[str, Any], lgb_params: dict[str, Any] | None = None) -> dict[str, Any]:
    models: dict[str, Any] = {}
    mcfg = cfg.get("models", {})
    if mcfg.get("dummy", {}).get("enabled", True):
        models["dummy_majority"] = DummyBaseline()
        models["rule_amount"] = AmountRuleBaseline()
    if mcfg.get("logistic", {}).get("enabled", True):
        models["logistic_regression"] = LogisticModel(
            C=float(mcfg.get("logistic", {}).get("C", 1.0)),
            max_iter=int(mcfg.get("logistic", {}).get("max_iter", 400)),
        )
    if mcfg.get("lightgbm", {}).get("enabled", True) and HAS_LIGHTGBM:
        params = dict(mcfg.get("lightgbm", {}))
        params.pop("enabled", None)
        if lgb_params:
            params.update(lgb_params)
        models["lightgbm"] = LightGBMModel(
            n_estimators=int(params.get("n_estimators", 80)),
            max_depth=int(params.get("max_depth", 4)),
            learning_rate=float(params.get("learning_rate", 0.08)),
            num_leaves=int(params.get("num_leaves", 24)),
        )
    if mcfg.get("hist_gbdt", {}).get("enabled", True):
        h = mcfg.get("hist_gbdt", {})
        models["hist_gradient_boosting"] = HistGBDTModel(
            max_iter=int(h.get("max_iter", 80)),
            max_depth=int(h.get("max_depth", 4)),
            learning_rate=float(h.get("learning_rate", 0.08)),
        )
    if HAS_XGBOOST:
        models["xgboost"] = XGBoostModel()
    if mcfg.get("mlp_torch", {}).get("enabled", True) and HAS_TORCH:
        t = mcfg.get("mlp_torch", {})
        models["mlp_torch"] = TorchMLPModel(
            hidden_dims=list(t.get("hidden_dims", [32, 16])),
            epochs=int(t.get("epochs", 12)),
            batch_size=int(t.get("batch_size", 256)),
            lr=float(t.get("lr", 0.01)),
        )
    if mcfg.get("mlp_sklearn", {}).get("enabled", True):
        s = mcfg.get("mlp_sklearn", {})
        hidden = tuple(s.get("hidden_layer_sizes", [32, 16]))
        models["mlp_sklearn"] = SklearnMLPModel(
            hidden_layer_sizes=hidden,
            max_iter=int(s.get("max_iter", 80)),
        )
    if mcfg.get("isolation_forest", {}).get("enabled", True):
        iso = mcfg.get("isolation_forest", {})
        models["isolation_forest"] = IsolationForestModel(
            n_estimators=int(iso.get("n_estimators", 80)),
            contamination=float(iso.get("contamination", 0.03)),
        )
    return models


def train_from_config(
    config_path: str | Path = "configs/model.yaml",
    n_optuna_trials: int | None = None,
) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    cost_cfg = load_yaml("configs/business_cost.yaml")
    fp_cost = float(cost_cfg.get("false_positive", 1.0))
    fn_cost = float(cost_cfg.get("false_negative", 25.0))
    artifacts = Path(cfg.get("artifacts_dir", "artifacts"))
    artifacts.mkdir(parents=True, exist_ok=True)

    data_path = Path(cfg["data_path"])
    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    split_cfg = cfg.get("split", {})
    label_delay = int(cfg.get("label_delay_days", 7))
    split, train_f, val_f, test_f = build_feature_tables(
        df,
        train_frac=float(split_cfg.get("train_fraction", 0.6)),
        val_frac=float(split_cfg.get("val_fraction", 0.2)),
        label_delay_days=label_delay,
    )
    manifest = split_manifest(split)
    manifest["label_delay_days"] = label_delay
    _write_json(artifacts / "split_manifest.json", manifest)

    x_train = matrix_from_frame(train_f)
    y_train = train_f["is_fraud"].to_numpy(dtype=int)
    x_val = matrix_from_frame(val_f)
    y_val = val_f["is_fraud"].to_numpy(dtype=int)
    x_test = matrix_from_frame(test_f)
    y_test = test_f["is_fraud"].to_numpy(dtype=int)

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_val_s = scaler.transform(x_val)
    x_test_s = scaler.transform(x_test)
    joblib.dump({"scaler": scaler, "feature_names": FEATURE_NAMES}, artifacts / "preprocessor.joblib")

    optuna_cfg = cfg.get("optuna", {})
    trials = n_optuna_trials if n_optuna_trials is not None else int(optuna_cfg.get("n_trials", 8))
    lgb_tune = tune_lightgbm(
        x_train,
        y_train,
        x_val,
        y_val,
        n_trials=trials,
        timeout_seconds=int(optuna_cfg.get("timeout_seconds", 90)),
    )
    _write_json(artifacts / "optuna.json", {k: v for k, v in lgb_tune.items() if k != "study"})

    models = _instantiate_models(cfg, lgb_params=lgb_tune.get("best_params") or None)
    scaled_names = {"logistic_regression", "mlp_sklearn", "mlp_torch"}

    try:
        configure_mlflow("file:./mlruns", "sentinelml")
        mlflow_ok = True
    except Exception as exc:
        mlflow_ok = False
        log.warning("mlflow_unavailable", error=str(exc))

    comparison: list[dict[str, Any]] = []
    bundles: dict[str, Any] = {}
    default_strategy = cfg.get("thresholds", {}).get("default_strategy", "minimize_configured_cost")

    import mlflow

    for name, model in models.items():
        xtr = x_train_s if name in scaled_names else x_train
        xv = x_val_s if name in scaled_names else x_val
        xt = x_test_s if name in scaled_names else x_test
        tracemalloc.start()
        t0 = time.perf_counter()
        try:
            model.fit(xtr, y_train)
        except Exception as exc:
            tracemalloc.stop()
            log.warning("model_fit_failed", model=name, error=str(exc))
            comparison.append({"model": name, "error": str(exc), "pr_auc": 0.0, "recall": 0.0, "fpr": 1.0, "expected_cost": 1e9, "latency_ms_p50": 1e9, "rss_mb": 1e9})
            continue
        train_s = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss_mb = peak / (1024 * 1024)

        p_val = model.predict_proba(xv)
        locked = lock_thresholds(
            y_val,
            p_val,
            max_fpr=float(cfg.get("thresholds", {}).get("max_fpr", 0.05)),
            fp_cost=fp_cost,
            fn_cost=fn_cost,
        )
        threshold = locked[default_strategy]
        cal = choose_calibration(y_val, p_val, threshold, fp_cost, fn_cost)
        calibrator = cal["calibrator"]
        p_val_c = calibrator.transform(p_val)
        p_test = calibrator.transform(model.predict_proba(xt))

        val_metrics = classification_metrics(y_val, p_val_c, threshold, fp_cost, fn_cost)
        test_metrics = classification_metrics(y_test, p_test, threshold, fp_cost, fn_cost)
        lat = _latency_ms(model, xt)
        row = {
            "model": name,
            "framework": model.get_metadata().get("framework"),
            "train_seconds": train_s,
            "rss_mb": rss_mb,
            **lat,
            "val": val_metrics,
            "test": test_metrics,
            "thresholds_locked_on_val": locked,
            "operating_threshold_strategy": default_strategy,
            "operating_threshold": threshold,
            "calibration": {
                "method": cal["method"],
                "val_comparison": cal["val_comparison"],
                "note": cal["note"],
                "reliability_uncalibrated": cal["reliability_uncalibrated"],
                "reliability_chosen": cal["reliability_chosen"],
            },
            "pr_auc": test_metrics["pr_auc"],
            "recall": test_metrics["recall"],
            "fpr": test_metrics["fpr"],
            "expected_cost": test_metrics["expected_cost"],
            "roc_auc": test_metrics["roc_auc"],
            "metadata": model.get_metadata(),
        }
        comparison.append(row)
        bundles[name] = {
            "model": model,
            "calibrator": calibrator,
            "threshold": threshold,
            "scaled": name in scaled_names,
        }
        if mlflow_ok:
            with mlflow.start_run(run_name=name):
                mlflow.log_params({"model": name, "threshold": threshold, "calibration": cal["method"]})
                log_metrics_safe({k: v for k, v in test_metrics.items() if not isinstance(v, dict)})

        # persist small sklearn-style bundle
        joblib.dump(
            {
                "model": model,
                "calibrator": calibrator,
                "threshold": threshold,
                "scaled": name in scaled_names,
                "feature_names": FEATURE_NAMES,
                "name": name,
            },
            artifacts / f"model_{name}.joblib",
        )

    viable = [r for r in comparison if "error" not in r and "test" in r]
    if not viable:
        raise RuntimeError("all models failed to train")
    weights = cfg.get("selection", {}).get("weights", {})
    picked = pick_champion(viable, weights)
    champion_name = picked["champion"]["model"]
    champ_row = next(r for r in comparison if r["model"] == champion_name)
    gate = promotion_gate(champ_row, champion=None)
    registry = {
        "champion": champion_name,
        "challenger": picked["ranked"][1]["model"] if len(picked["ranked"]) > 1 else None,
        "aliases": {"champion": champion_name, "challenger": picked["ranked"][1]["model"] if len(picked["ranked"]) > 1 else None},
        "promotion_gate": gate,
        "why": picked["champion"]["why"],
        "selection_weights": weights,
        "ranked": [
            {"model": r["model"], "selection_score": r["selection_score"], "pr_auc": r["pr_auc"]}
            for r in picked["ranked"]
        ],
    }

    metrics_out = {
        "split": manifest,
        "models": comparison,
        "champion": champ_row,
        "registry": registry,
        "feature_names": FEATURE_NAMES,
        "label_delay_days": label_delay,
        "optuna": {k: v for k, v in lgb_tune.items() if k != "study"},
        "hash": sha256_json({"n_train": manifest["n_train"], "n_val": manifest["n_val"], "n_test": manifest["n_test"]}),
    }
    _write_json(artifacts / "metrics.json", metrics_out)
    _write_json(artifacts / "model_comparison.json", comparison)
    pd.DataFrame(
        [
            {
                "model": r["model"],
                "pr_auc": r["pr_auc"],
                "roc_auc": r["roc_auc"],
                "precision": r["test"]["precision"],
                "recall": r["recall"],
                "f1": r["test"]["f1"],
                "fpr": r["fpr"],
                "expected_cost": r["expected_cost"],
                "latency_ms_p50": r["latency_ms_p50"],
                "rss_mb": r["rss_mb"],
                "selection_score": next(
                    x["selection_score"] for x in picked["ranked"] if x["model"] == r["model"]
                ),
            }
            for r in viable
        ]
    ).to_csv(artifacts / "model_comparison.csv", index=False)
    _write_json(artifacts / "registry.json", registry)
    _write_json(
        artifacts / "champion.json",
        {
            "model": champion_name,
            "threshold": champ_row["operating_threshold"],
            "strategy": default_strategy,
            "calibration": champ_row["calibration"]["method"],
            "test_pr_auc": champ_row["pr_auc"],
            "test_recall": champ_row["recall"],
            "test_fpr": champ_row["fpr"],
            "test_expected_cost": champ_row["expected_cost"],
            "why": registry["why"],
        },
    )

    # keep a named champion pointer (small joblib)
    joblib.dump(bundles[champion_name] | {"name": champion_name, "feature_names": FEATURE_NAMES}, artifacts / "champion.joblib")

    train_f.to_parquet(Path("data/generated") / "transactions_features.parquet", index=False)
    return metrics_out
