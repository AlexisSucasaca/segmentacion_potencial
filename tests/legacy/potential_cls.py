# COPIA DE REFERENCIA (no editar). Es el potential_cls.py original, sin cambios.
# Solo la usa tests/test_model.py para comprobar que src/model.py da los mismos resultados.
# Se puede borrar cuando el pipeline nuevo reemplace a los notebooks.
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_recall_curve, roc_auc_score
)
from sklearn.dummy import DummyClassifier

import lightgbm as lgb
import optuna
import pandas as pd
import numpy as np
import shap
from typing import List, Dict, Any, Optional

N_ESTIMATORS = 500
PROBA_COLUMNS = {
    '1 Oro+': 'proba_1 Oro+',
    '2 Oro': 'proba_2 Oro',
    '3 Plata': 'proba_3 Plata',
    '4 Bronce': 'proba_4 Bronce',
}

ORDER_CLASSES = [
    "1 Oro+", "2 Oro", "3 Plata", "4 Bronce"
]

def optuna_objective(
    trial: optuna.Trial, 
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    categorical_feats: List[str]
) -> float:
    params = {
        "objective": "multiclass",
        "num_class": len(np.unique(y_train)),
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 50),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    model = lgb.LGBMClassifier(
        **params,
        n_estimators=N_ESTIMATORS,
        random_state=42,
        class_weight="balanced",
        verbosity=-1
    )

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model.fit(X_tr, y_tr, categorical_feature=categorical_feats)
        y_pred = model.predict(X_val)
        score = f1_score(y_val, y_pred, average="macro")
        scores.append(score)

    return np.mean(scores)

def optuna_objective_reg(
    trial: optuna.Trial, 
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    y_train_cls: pd.Series,
    categorical_feats: List[str]
) -> float:
    params = {
        "boosting_type": "gbdt",
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 50),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    model = lgb.LGBMRegressor(
        **params,
        n_estimators=N_ESTIMATORS,
        random_state=42,
        verbosity=-1
    )

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X_train, y_train_cls):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model.fit(X_tr, y_tr, categorical_feature=categorical_feats)
        y_pred = model.predict(X_val)
        score = np.median(abs(y_pred - y_val)/y_val)
        scores.append(score)

    return np.mean(scores)

def validate_performance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    categorical_feats: List[str],
    lgbm_params: Dict[str, Any],
    target_class: str,
) -> Dict[str, Any]:
    
    model = lgb.LGBMClassifier(
        **lgbm_params,
        n_estimators=N_ESTIMATORS,
        random_state=42,
        class_weight="balanced",
        verbosity=-1,
    )
    model.fit(X_train, y_train, categorical_feature=categorical_feats)
    y_pred = model.predict(X_valid)

    acc = accuracy_score(y_valid, y_pred)
    f1 = f1_score(y_valid, y_pred, average="macro")

    current_class_idx = np.nonzero(model.classes_ == target_class)[0]
    proba_current_class = model.predict_proba(X_valid)[:, current_class_idx]
    auc_score = roc_auc_score(y_valid == target_class, proba_current_class)
    
    print(f"✅ Validation Accuracy: {acc:.4f}, F1 Macro: {f1:.4f}, AUC: {auc_score:.4f}")
    model_precision, model_recall, model_thresholds = precision_recall_curve(y_valid == target_class, proba_current_class)
    
    feature_importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)


    # === BENCHMARK MODELS ===
    dummy_majority = DummyClassifier(strategy="most_frequent", random_state=42)
    dummy_majority.fit(X_train, y_train)
    majority_acc = accuracy_score(y_valid, dummy_majority.predict(X_valid))
    majority_f1 = f1_score(y_valid, dummy_majority.predict(X_valid), average="macro")
    majority_auc = roc_auc_score(y_valid == target_class, dummy_majority.predict_proba(X_valid)[:, current_class_idx])

    dummy_stratified = DummyClassifier(strategy="stratified", random_state=42)
    dummy_stratified.fit(X_train, y_train)
    stratified_acc = accuracy_score(y_valid, dummy_stratified.predict(X_valid))
    stratified_f1 = f1_score(y_valid, dummy_stratified.predict(X_valid), average="macro")
    stratified_auc = roc_auc_score(y_valid == target_class, dummy_stratified.predict_proba(X_valid)[:, current_class_idx])

    dummy_uniform = DummyClassifier(strategy="uniform", random_state=42)
    dummy_uniform.fit(X_train, y_train)
    uniform_acc = accuracy_score(y_valid, dummy_uniform.predict(X_valid))
    uniform_f1 = f1_score(y_valid, dummy_uniform.predict(X_valid), average="macro")
    uniform_auc = roc_auc_score(y_valid == target_class, dummy_uniform.predict_proba(X_valid)[:, current_class_idx])

    return {
        "pr_curves": {
            "thresholds": model_thresholds,
            "precision": model_precision,
            "recall": model_recall,
        },
        "lgbm_params": lgbm_params,
        "feature_importance": feature_importance,
        "benchmarks": {
            "model": {"accuracy": acc, "f1_macro": f1, "auc": auc_score},
            "majority": {"accuracy": majority_acc, "f1_macro": majority_f1, "auc": majority_auc},
            "stratified": {"accuracy": stratified_acc, "f1_macro": stratified_f1, "auc": stratified_auc},
            "uniform": {"accuracy": uniform_acc, "f1_macro": uniform_f1, "auc": uniform_auc},
        }
    }

def train_predict_split(
    train_df, 
    test_df, 
    others_df, 
    params,
    categorical_feats: List[str],
    features: List[str],
    target_col: str,
    target_class: str,
    client_id_col: str,
    cluster_col: str,
    target_reg: Optional[str] = None,
    params_reg: Optional[Dict[str, Any]] = None,
    add_shap: bool = False,
) -> pd.DataFrame:

    train_df_all = pd.concat([train_df, others_df])
    for c in categorical_feats:
        train_df_all[c] = train_df_all[c].astype("category")
        test_df[c] = test_df[c].astype(train_df_all[c].dtype)

    X_train = train_df_all[features]
    y_train = train_df_all[target_col]
    X_test = test_df[features]

    model = lgb.LGBMClassifier(
        **params,
        n_estimators=N_ESTIMATORS,
        random_state=42,
        class_weight="balanced"
    )  
    model.fit(X_train, y_train, categorical_feature=categorical_feats)
    proba = model.predict_proba(X_test)
    preds = model.predict(X_test)

    df_result = test_df[[client_id_col, target_col, cluster_col]].copy()
    df_result["predicted"] = preds
    current_class_idx = np.nonzero(model.classes_ == target_class)[0]
    for i, label in enumerate(model.classes_):
        df_result[f"proba_{label}"] = proba[:, i]

    df_result["proba_potential"] = 1 - proba[:, current_class_idx]

    # SHAP para clase predicha
    if add_shap:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        
        shap_selected = []
        for i, pred_class in enumerate(preds):
            class_idx = list(model.classes_).index(pred_class)
            shap_row = shap_values[i, :, class_idx]
            shap_selected.append(shap_row)

        shap_pred_df = pd.DataFrame(shap_selected, columns=[f"shap_{f}" for f in features])
        df_result = pd.concat([df_result.reset_index(drop=True), shap_pred_df.reset_index(drop=True)], axis=1)

    # Regresión opcional
    if target_reg:
        y_train_reg = train_df_all[target_reg]
        model_reg = lgb.LGBMRegressor(
            **params_reg,
            n_estimators=N_ESTIMATORS,
            random_state=42,
        )
        model_reg.fit(X_train, y_train_reg, categorical_feature=categorical_feats)
        df_result["margin_pred"] = model_reg.predict(X_test)
        df_result["margin"] = test_df[target_reg].values

    return df_result

def run_potential_estimations(
    df_filtered: pd.DataFrame,
    target_class: str,
    categorical_feats: List[str],
    features: List[str],
    target_col: str,
    cluster_col: str,
    client_id_col: str,
    lgbm_params: Dict[str, Any],
    lgbm_params_reg: Optional[Dict[str, Any]] = None,
    target_reg: Optional[str] = None,
    add_shap: bool = False,
) -> pd.DataFrame:
    
    df_class_X = df_filtered[df_filtered[target_col] == target_class].copy()
    df_others = df_filtered[df_filtered[target_col] != target_class].copy()

    count_examples = df_class_X[cluster_col].value_counts()
    if min(count_examples) < 3:
        df_X1, df_X2 = train_test_split(
            df_class_X, test_size=0.5, random_state=42
        )
    else:
        df_X1, df_X2 = train_test_split(
            df_class_X, test_size=0.5, stratify=df_class_X[cluster_col], random_state=42
        )

    pred_X1 = train_predict_split(
        df_X2, df_X1, df_others, 
        lgbm_params,
        categorical_feats,
        features,
        target_col,
        target_class,
        client_id_col,
        cluster_col,
        target_reg=target_reg,
        params_reg=lgbm_params_reg,
        add_shap=add_shap,
    )
    pred_X2 = train_predict_split(
        df_X1, df_X2, df_others, 
        lgbm_params,
        categorical_feats,
        features,
        target_col,
        target_class,
        client_id_col,
        cluster_col,
        target_reg=target_reg,
        params_reg=lgbm_params_reg,
        add_shap=add_shap,
    )
    return pd.concat([pred_X1, pred_X2], ignore_index=True)

# Now classify potential_confidence based on the probability thresholds
def classify_confidence(proba, high_thresh: float=0.75, medium_thresh: float=0.5):
    if proba < medium_thresh:
        return 'Low'
    elif proba < high_thresh:
        return 'Medium'
    else:
        return 'High'


def predict_segment(row, proba_columns=PROBA_COLUMNS, order_classes=ORDER_CLASSES, high_thresh: float=0.75, medium_thresh: float=0.5):
    pred_high = None
    pred_medium = None



    for i in range(len(order_classes)):
        cls = order_classes[i]
        higher_or_equal_classes = order_classes[:(i + 1)] 
        cumulative_prob = sum(row[proba_columns[c]] for c in higher_or_equal_classes)

        if pred_high is None and cumulative_prob >= high_thresh:
            pred_high = cls
        if pred_medium is None and cumulative_prob >= medium_thresh:
            pred_medium = cls
    
    class_rank =  {cls: i for i, cls in enumerate(order_classes)}
    actual_class = row["segmento_trx"]
    actual_rank = class_rank[actual_class]

    if pred_high and class_rank[pred_high] < actual_rank:
        return pd.Series([pred_high, "High"])
    elif pred_medium and class_rank[pred_medium] < actual_rank:
        idx = order_classes.index(pred_medium)
        pred = order_classes[idx + 1] if idx + 1 < len(order_classes) else pred_medium
        return pd.Series([pred, "Medium"])
    else:
        return pd.Series([actual_class, None])



def format_potential_df_2(
    df_potential_estimates: pd.DataFrame,
    client_data: pd.DataFrame,
    features: List[str],
    client_id_col: str = "Cod Cliente DEX",
    high_thresh: float = 0.75,
    medium_thresh: float = 0.5,
    proba_columns=PROBA_COLUMNS,
    order_classes=ORDER_CLASSES,
) -> pd.DataFrame:
    """Format potential estimates with predictions, confidence, and client features."""
    
    # Deduplicate features to avoid conflicts
    features = list(dict.fromkeys(features))
    
    # Step 1: Add probability predictions
    df_potential_estimates["proba_pred_segment"] = df_potential_estimates["predicted"].map(proba_columns)
    df_potential_estimates["proba_pred_segment"] = df_potential_estimates.apply(
        lambda row: row[row["proba_pred_segment"]], axis=1
    )

    # Step 2: Classify confidence
    df_potential_estimates["prediction_conf"] = df_potential_estimates["proba_pred_segment"].apply(classify_confidence)
    
    # Step 3: Predict final segment with thresholds
    df_potential_estimates[['final_pred', 'final_pred_conf']] = df_potential_estimates.apply(
        predict_segment,
        proba_columns=proba_columns,
        order_classes=order_classes,
        high_thresh=high_thresh,
        medium_thresh=medium_thresh,
        axis=1
    )
    
    # Step 4: Flag potential clients
    df_potential_estimates["flag_potential"] = df_potential_estimates["final_pred"] != df_potential_estimates["segmento_trx"]

    # Step 5: Define columns to merge from client_data
    geo_cols = [client_id_col]
    if "des_ciudad" in client_data.columns:
        geo_cols.append("des_ciudad")
    if "des_distri" in client_data.columns:
        geo_cols.append("des_distri")

    # Step 6: Merge geographic data
    df_potential_estimates = pd.merge(
        df_potential_estimates,
        client_data[geo_cols],
        on=client_id_col,
        how="left"
    )

    # Step 7: Drop cluster column if exists
    if "cluster" in df_potential_estimates.columns:
        df_potential_estimates = df_potential_estimates.drop(columns=["cluster"])

    # Step 8: Merge features and spatial data
    feature_cols = [client_id_col, "X", "Y"] + features
    df_potential_estimates = df_potential_estimates.merge(
        client_data[feature_cols],
        on=client_id_col,
        how="left"
    )

    # Step 9: Add reference segment data
    reference_cols = [client_id_col, "X", "Y", "segmento_trx"] + geo_cols[1:] + features
    available_cols = list(dict.fromkeys([col for col in reference_cols if col in client_data.columns]))

    df_reference = client_data[available_cols].loc[client_data["segmento_trx"] == order_classes[0]]

    # Step 10: Concatenate with reference segment
    df_potential_estimates = pd.concat(
        [df_potential_estimates, df_reference],
        ignore_index=True
    )

    return df_potential_estimates


def format_potential_df(
    df_potential_estimates: pd.DataFrame,
    client_data: pd.DataFrame,
    features: List[str],
    client_id_col: str="Cod Cliente DEX",
    high_thresh: float=0.75, medium_thresh: float=0.5,
    proba_columns=PROBA_COLUMNS,
    order_classes=ORDER_CLASSES,
) -> pd.DataFrame:

    # Get the probability for the current segment (segmento_trx)
    df_potential_estimates["proba_pred_segment"] = df_potential_estimates["predicted"].map(proba_columns)
    df_potential_estimates["proba_pred_segment"] = df_potential_estimates.apply(
        lambda row: row[row["proba_pred_segment"]], axis=1
    )

    df_potential_estimates["prediction_conf"] = df_potential_estimates["proba_pred_segment"].apply(classify_confidence)
    df_potential_estimates[['final_pred', 'final_pred_conf']] = df_potential_estimates.apply(
        predict_segment
        , proba_columns=proba_columns
        , order_classes=order_classes
        , high_thresh=high_thresh
        , medium_thresh=medium_thresh
        , axis=1
    )
    df_potential_estimates["flag_potential"] = df_potential_estimates["final_pred"] != df_potential_estimates["segmento_trx"]

    cols_client = [client_id_col]
    if "des_ciudad" in client_data.columns:
        cols_client += ["des_ciudad"]
    if "des_distri" in client_data.columns:
        cols_client += ["des_distri"]

    df_potential_estimates = pd.merge(
        df_potential_estimates,
        client_data[cols_client],
        on=client_id_col,
        how="left"
    )

    if "cluster" in df_potential_estimates.columns:
        df_potential_estimates = df_potential_estimates.drop(columns=["cluster"])
    df_potential_estimates_feats = df_potential_estimates.merge(
        client_data[[client_id_col, "X", "Y"]+features], on=client_id_col, how="left"
    )

    cols_oro_plus = [
        client_id_col, 
        "X", "Y",
        "segmento_trx", 
    ]

    if "des_ciudad" in client_data.columns:
        cols_oro_plus += ["des_ciudad"]
    if "des_distri" in client_data.columns:
        cols_oro_plus += ["des_distri"]

    df_potential_estimates_feats = pd.concat(
        [
            df_potential_estimates_feats,
            client_data[cols_oro_plus+features].query(
                f"segmento_trx == '{order_classes[0]}'"
            )
        ]
    )



    return df_potential_estimates_feats

def compute_feature_mean(
    df_potential_estimates_feats: pd.DataFrame,
    sorted_classes: List[str],
    rmv_feats: List[str],
) -> pd.DataFrame:
    shap_cols = [c for c in df_potential_estimates_feats.columns if "shap_" in c]
    shap_potential = df_potential_estimates_feats.query("flag_potential == 1")[shap_cols].quantile(0.5)
    shap_potential.rename(index=lambda x: x.replace("shap_", ""), inplace=True)

    features = [f for f in shap_potential.index if f not in rmv_feats]
    records = []

    # Identificamos la última clase (sin potencial)
    last_class = sorted_classes[-1]
    potential_classes = sorted_classes[:-1]

    for feat in features:
        row = {"feature": feat}

        # 1. Agregar medias por clase (todas)
        for cls in sorted_classes:
            df_cls = df_potential_estimates_feats.query(f"segmento_trx == '{cls}'")
            row[f"mean_{cls}"] = round(df_cls[feat].mean(), 2)

        # 2. Agregar medias de potencial por clase (menos la última)
        for cls in potential_classes:
            df_cls_pot = df_potential_estimates_feats.query(f"segmento_trx == '{cls}' and flag_potential == 1")
            row[f"mean_pot_{cls}"] = round(df_cls_pot[feat].mean(), 2) if not df_cls_pot.empty else None

        # 3. Agregar medias de no potencial por clase (menos la última)
        for cls in potential_classes:
            df_cls_non_pot = df_potential_estimates_feats.query(f"segmento_trx == '{cls}' and flag_potential == 0")
            row[f"mean_non_pot_{cls}"] = round(df_cls_non_pot[feat].mean(), 2) if not df_cls_non_pot.empty else None

        records.append(row)

    df_result = pd.DataFrame(records)

    # Definir el orden exacto de columnas
    col_order = (
        ["feature"]
        + [f"mean_{cls}" for cls in sorted_classes]
        + [f"mean_pot_{cls}" for cls in potential_classes]
        + [f"mean_non_pot_{cls}" for cls in potential_classes]
    )

    return df_result[col_order]

def save_pot_results(
    df_potential_estimates_feats: pd.DataFrame,
    survey_dict: pd.DataFrame,
    path: str,
    sorted_classes: Optional[List[str]]=None,
    rmv_feats: Optional[List[str]]=None,
):
    with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
        df_potential_estimates_feats.to_excel(writer, sheet_name='Potencial', index=False)
        survey_dict.to_excel(writer, sheet_name='Diccionario Encuesta', index=False)

        if sorted_classes and rmv_feats:
            feat_mean = compute_feature_mean(
                df_potential_estimates_feats,
                sorted_classes,
                rmv_feats
            )
            feat_mean.to_excel(writer, sheet_name='Feature mean', index=False)

        workbook  = writer.book
        worksheet = writer.sheets['Potencial']

        worksheets = [
            writer.sheets['Potencial'],
        ]

        #if base_class and higher_classes and rmv_feats:
        #    worksheets.append(writer.sheets['Feature mean'])

        for worksheet in worksheets:

            # Formatos
            format_percent = workbook.add_format({'num_format': '0.0%'})
            format_two_decimals = workbook.add_format({'num_format': '0.00'})
            format_default = None  # para columnas no numéricas

            percent_cols = [col for col in df_potential_estimates_feats.columns if "proba" in col]

            for i, col in enumerate(df_potential_estimates_feats.columns):
                col_data = df_potential_estimates_feats[col]
                max_len = max(col_data.astype(str).map(len).max(), len(col)) + 2

                # Elegimos formato
                if col in percent_cols:
                    fmt = format_percent
                elif pd.api.types.is_numeric_dtype(col_data):
                    fmt = format_two_decimals
                else:
                    fmt = format_default

                worksheet.set_column(i, i, max_len, fmt)