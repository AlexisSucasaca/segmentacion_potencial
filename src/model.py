"""Modelo de potencial: LightGBM por clase, predicción cruzada y SHAP.

Reproduce, sin cambiar resultados, lo que hacían `potential_cls.py` y el bucle de cada
notebook (`validate_performance`, `run_potential_estimations`, `train_predict_split`),
pero dirigido por la config de cada vertical y sin MLflow.

Cómo funciona (igual que antes)
-------------------------------
Se entrena un modelo por cada clase salvo la más alta, de la más baja a la más alta. El modelo
de la clase C usa las filas de C y de todas las clases superiores:

    2 clases            ->  1 modelo   (Bronce  vs  Plata+)
    bodegas, 4 clases   ->  3 modelos  (Bronce vs resto, Plata vs Oro y Oro+, Oro vs Oro+)

Para cada modelo:
  * Validación: split 80/20 estratificado; métricas del modelo y de tres baselines "dummy".
  * Estimación: las filas de la clase C se parten en dos mitades; se entrena con una mitad
    (más las clases superiores) y se predice la otra, y viceversa. Así ninguna fila de la
    clase C se predice con un modelo que la haya visto.
  * SHAP: de cada fila se guarda el SHAP de la clase predicha.

`class_order` va de la clase más alta a la más baja, como en el resto del paquete.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_curve, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

N_ESTIMATORS = 500
SEED = 42


@dataclass
class ModelResult:
    """Resultado de `fit_potential`."""
    estimates: pd.DataFrame                       # una fila por cliente de las clases objetivo
    validation: Dict[str, Dict[str, Any]] = field(default_factory=dict)   # por clase
    params: Dict[str, Dict[str, Any]] = field(default_factory=dict)       # parámetros LGBM usados
    class_order: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------------------
# Parámetros y utilidades
# --------------------------------------------------------------------------------------
def build_params(n_classes: int, clase_params: Dict[str, Any]) -> Dict[str, Any]:
    """Parámetros base multiclase + los de la clase (lambda_l2, num_leaves, ...)."""
    base = {
        "objective": "multiclass",
        "num_class": n_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
    }
    return {**base, **clase_params}


def _classifier(params: Dict[str, Any], n_estimators: int, seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        **params, n_estimators=n_estimators, random_state=seed,
        class_weight="balanced", verbosity=-1,
    )


def _as_category(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = df[c].astype("category")
    return df


def _ensure_numeric_features(df: pd.DataFrame, features: Sequence[str], categorical_feats: Sequence[str]) -> pd.DataFrame:
    """Las columnas de `features` que no son categóricas pasan a numéricas si llegan como
    `object` (p. ej. un NUMERIC de BigQuery leído como Decimal). Solo toca esas columnas —
    nunca `target_col`, `id_col` ni `cluster_col`, aunque estén en el mismo DataFrame."""
    cat = set(categorical_feats)
    df = df.copy()
    for c in features:
        if c not in cat and df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _shap_of_predicted_class(model: lgb.LGBMClassifier, X: pd.DataFrame, preds: np.ndarray) -> np.ndarray:
    """SHAP de la clase predicha de cada fila, con forma (n_filas, n_features).

    Según la versión de shap, TreeExplainer devuelve una lista (una matriz por clase) o un
    arreglo (filas, features, clases); aquí se aceptan ambas.
    """
    import shap  # se importa aquí para no exigir shap si add_shap=False

    values = shap.TreeExplainer(model).shap_values(X)
    if isinstance(values, list):
        values = np.stack(values, axis=-1)
    class_idx = pd.Index(model.classes_).get_indexer(preds)
    return values[np.arange(len(X)), :, class_idx]


# --------------------------------------------------------------------------------------
# Validación (split 80/20 con baselines dummy)
# --------------------------------------------------------------------------------------
def validate_performance(
    X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series,
    categorical_feats: Sequence[str], lgbm_params: Dict[str, Any], target_class: str,
    n_estimators: int = N_ESTIMATORS, seed: int = SEED,
) -> Dict[str, Any]:
    model = _classifier(lgbm_params, n_estimators, seed)
    model.fit(X_train, y_train, categorical_feature=list(categorical_feats))
    y_pred = model.predict(X_valid)

    idx = int(np.flatnonzero(model.classes_ == target_class)[0])
    proba = model.predict_proba(X_valid)[:, idx]
    is_target = y_valid == target_class
    precision, recall, thresholds = precision_recall_curve(is_target, proba)

    def bench(estimator) -> Dict[str, float]:
        pred = estimator.predict(X_valid)
        return {
            "accuracy": accuracy_score(y_valid, pred),
            "f1_macro": f1_score(y_valid, pred, average="macro"),
            "auc": roc_auc_score(is_target, estimator.predict_proba(X_valid)[:, idx]),
        }

    benchmarks = {
        "model": {
            "accuracy": accuracy_score(y_valid, y_pred),
            "f1_macro": f1_score(y_valid, y_pred, average="macro"),
            "auc": roc_auc_score(is_target, proba),
        }
    }
    for nombre, strategy in [("majority", "most_frequent"), ("stratified", "stratified"), ("uniform", "uniform")]:
        dummy = DummyClassifier(strategy=strategy, random_state=seed).fit(X_train, y_train)
        benchmarks[nombre] = bench(dummy)

    importance = (
        pd.DataFrame({"feature": X_train.columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False).reset_index(drop=True)
    )
    return {
        "pr_curves": {"thresholds": thresholds, "precision": precision, "recall": recall},
        "lgbm_params": lgbm_params,
        "feature_importance": importance,
        "benchmarks": benchmarks,
    }


def validation_table(result: ModelResult) -> pd.DataFrame:
    """Tabla (en %) de métricas del modelo y los baselines, una fila por clase y métrica."""
    partes = {
        cls: (100 * pd.DataFrame(v["benchmarks"])).round(1)
        for cls, v in result.validation.items()
    }
    if not partes:
        return pd.DataFrame()
    out = pd.concat(partes, names=["clase", "metrica"]).reset_index()
    return out[out["metrica"] != "accuracy"].reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Estimación con predicción cruzada
# --------------------------------------------------------------------------------------
def train_predict_split(
    train_df: pd.DataFrame, test_df: pd.DataFrame, others_df: pd.DataFrame,
    params: Dict[str, Any], categorical_feats: Sequence[str], features: Sequence[str],
    target_col: str, target_class: str, id_col: str, cluster_col: str,
    add_shap: bool = True, n_estimators: int = N_ESTIMATORS, seed: int = SEED,
) -> pd.DataFrame:
    """Entrena con `train_df` + `others_df` y predice `test_df`."""
    train_all = pd.concat([train_df, others_df])
    train_all = _ensure_numeric_features(train_all, features, categorical_feats)
    train_all = _as_category(train_all, categorical_feats)
    test = _ensure_numeric_features(test_df, features, categorical_feats)
    test = _as_category(test, categorical_feats)
    for c in categorical_feats:
        # las categorías de test se alinean con las de train; una categoría nueva queda NaN
        test[c] = test[c].astype(train_all[c].dtype)

    features = list(features)
    model = _classifier(params, n_estimators, seed)
    model.fit(train_all[features], train_all[target_col], categorical_feature=list(categorical_feats))

    X_test = test[features]
    proba = model.predict_proba(X_test)
    preds = model.predict(X_test)

    out = test[[id_col, target_col, cluster_col]].copy()
    out["predicted"] = preds
    for i, label in enumerate(model.classes_):
        out[f"proba_{label}"] = proba[:, i]
    idx = int(np.flatnonzero(model.classes_ == target_class)[0])
    out["proba_potential"] = 1 - proba[:, idx]

    out = out.reset_index(drop=True)
    if add_shap:
        shap_sel = _shap_of_predicted_class(model, X_test, preds)
        shap_df = pd.DataFrame(shap_sel, columns=[f"shap_{f}" for f in features])
        out = pd.concat([out, shap_df], axis=1)
    return out


def run_potential_estimations(
    df_filtered: pd.DataFrame, target_class: str, categorical_feats: Sequence[str],
    features: Sequence[str], target_col: str, cluster_col: str, id_col: str,
    lgbm_params: Dict[str, Any], add_shap: bool = True,
    n_estimators: int = N_ESTIMATORS, seed: int = SEED,
) -> pd.DataFrame:
    """Predicción cruzada de las filas de `target_class` (una mitad predice a la otra)."""
    df_class = df_filtered[df_filtered[target_col] == target_class].copy()
    df_others = df_filtered[df_filtered[target_col] != target_class].copy()

    counts = df_class[cluster_col].value_counts()
    if counts.min() < 3:                                  # no se puede estratificar
        half1, half2 = train_test_split(df_class, test_size=0.5, random_state=seed)
    else:
        half1, half2 = train_test_split(
            df_class, test_size=0.5, stratify=df_class[cluster_col], random_state=seed
        )

    common = dict(
        params=lgbm_params, categorical_feats=categorical_feats, features=features,
        target_col=target_col, target_class=target_class, id_col=id_col,
        cluster_col=cluster_col, add_shap=add_shap, n_estimators=n_estimators, seed=seed,
    )
    pred1 = train_predict_split(half2, half1, df_others, **common)
    pred2 = train_predict_split(half1, half2, df_others, **common)
    return pd.concat([pred1, pred2], ignore_index=True)


# --------------------------------------------------------------------------------------
# Optuna (opcional)
# --------------------------------------------------------------------------------------
def tune_params(
    X_train: pd.DataFrame, y_train: pd.Series, categorical_feats: Sequence[str],
    n_trials: int = 100, enqueue: Optional[Dict[str, Any]] = None,
    n_estimators: int = N_ESTIMATORS, seed: int = SEED,
):
    """Busca hiperparámetros con Optuna (F1 macro, 3 folds). Devuelve (params, study)."""
    import optuna

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "objective": "multiclass", "num_class": int(len(np.unique(y_train))),
            "metric": "multi_logloss", "boosting_type": "gbdt",
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 50),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }
        model = _classifier(params, n_estimators, seed)
        scores = []
        for tr, va in StratifiedKFold(n_splits=3, shuffle=True, random_state=seed).split(X_train, y_train):
            model.fit(X_train.iloc[tr], y_train.iloc[tr], categorical_feature=list(categorical_feats))
            scores.append(f1_score(y_train.iloc[va], model.predict(X_train.iloc[va]), average="macro"))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    if enqueue:
        study.enqueue_trial(enqueue)
    study.optimize(objective, n_trials=n_trials)
    best = {**study.best_params, "objective": "multiclass",
            "num_class": int(len(np.unique(y_train))), "metric": "multi_logloss"}
    return best, study


# --------------------------------------------------------------------------------------
# Punto de entrada
# --------------------------------------------------------------------------------------
def map_target(series: pd.Series, mapping: Optional[Dict[str, str]]) -> pd.Series:
    """Agrupa las clases del segmento transaccional (p. ej. Oro+/Oro/Plata -> 'Plata - Oro - Oro+')."""
    return series if not mapping else series.map(mapping)


def fit_potential(
    df: pd.DataFrame,
    features: Sequence[str],
    categorical_feats: Sequence[str],
    class_order: Sequence[str],
    lgbm_params: Dict[str, Dict[str, Any]],
    *,
    id_col: str,
    cluster_col: str,
    target_col: str = "segmento_trx",
    split_valid: float = 0.2,
    add_shap: bool = True,
    optuna_trials: Optional[int] = None,
    n_estimators: int = N_ESTIMATORS,
    seed: int = SEED,
) -> ModelResult:
    """Entrena un modelo por clase (salvo la más alta) y devuelve las estimaciones.

    lgbm_params  {clase: {lambda_l2, num_leaves, ...}} para cada clase que tiene modelo.
    optuna_trials si se indica, busca hiperparámetros con Optuna partiendo de `lgbm_params`.
    """
    features = list(dict.fromkeys(features))
    categorical_feats = list(categorical_feats)
    low_to_high = list(class_order)[::-1]
    modeled = low_to_high[:-1]

    faltan = [c for c in modeled if c not in lgbm_params]
    if faltan:
        raise KeyError(f"Faltan hiperparámetros LGBM para las clases: {faltan}")
    missing_cols = [c for c in [id_col, target_col, cluster_col, *features] if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Faltan columnas en df: {missing_cols}")
    if df[cluster_col].isna().any():
        raise ValueError(
            f"'{cluster_col}' tiene {int(df[cluster_col].isna().sum())} valores vacíos; "
            "se usa para estratificar la predicción cruzada."
        )
    fuera = df.loc[df[target_col].notna() & ~df[target_col].isin(low_to_high), target_col]
    if len(fuera):
        warnings.warn(
            f"{len(fuera)} filas con '{target_col}' fuera de class_order "
            f"({sorted(fuera.unique().tolist())}) se excluyen del modelo.",
            stacklevel=2,
        )

    result = ModelResult(estimates=pd.DataFrame(), class_order=list(class_order))
    partes: List[pd.DataFrame] = []
    for i, current in enumerate(modeled):
        higher = low_to_high[i:]
        df_filtered = df[df[target_col].isin(higher)].copy()
        X = _as_category(_ensure_numeric_features(df_filtered[features], features, categorical_feats), categorical_feats)
        y = df_filtered[target_col]

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, stratify=y, test_size=split_valid, random_state=seed
        )
        if optuna_trials:
            params, _ = tune_params(
                X_train, y_train, categorical_feats, optuna_trials,
                enqueue=lgbm_params[current], n_estimators=n_estimators, seed=seed,
            )
        else:
            params = build_params(len(higher), lgbm_params[current])

        result.params[current] = params
        result.validation[current] = validate_performance(
            X_train, y_train, X_valid, y_valid, categorical_feats, params, current,
            n_estimators=n_estimators, seed=seed,
        )
        partes.append(run_potential_estimations(
            df_filtered, current, categorical_feats, features, target_col, cluster_col,
            id_col, params, add_shap=add_shap, n_estimators=n_estimators, seed=seed,
        ))
    result.estimates = pd.concat(partes, ignore_index=True)
    return result
