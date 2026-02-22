import pandas as pd
import numpy as np
from pathlib import Path

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

BASE_DIR = Path(__file__).resolve().parent

TRAIN_PATH = BASE_DIR / "train.csv"
TEST_PATH  = BASE_DIR / "test_for_participants.csv"
SAMPLE_SUB_PATH = BASE_DIR / "sample_submission.csv"
OUT_PATH = BASE_DIR / "Blixen_Capital.csv"

RANDOM_STATE = 42

VAL_WEEKS = 6
VAL_HOURS = VAL_WEEKS * 7 * 24 

MAX_NAN_FRAC = 0.60

EXOG_BASE_COLS = ["load_forecast", "wind_forecast", "solar_forecast"]
EXOG_DELTA_LAGS = [1, 24]
NETLOAD_ROLL_WINDOWS = [24, 168]

HOURS_PER_WEEK = 24 * 7


A_CLIP_Q_LOW, A_CLIP_Q_HIGH = 0.005, 0.995
A_ALPHA_GRID = np.linspace(0.0, 2, 41)  

XGB_BASE = dict(
    n_estimators=12000,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    min_child_weight=10,
    random_state=RANDOM_STATE,
    tree_method="hist",
    eval_metric="rmse",
    n_jobs=-1,
)

MARKET_PARAMS = {
    "Market A": dict(max_depth=4, min_child_weight=60, reg_lambda=12.0, subsample=0.70, colsample_bytree=0.70),
    "Market B": dict(max_depth=6, min_child_weight=10, reg_lambda=2.0,  subsample=0.80, colsample_bytree=0.80),
    "Market C": dict(max_depth=6, min_child_weight=10, reg_lambda=2.0,  subsample=0.80, colsample_bytree=0.80),
    "Market D": dict(max_depth=6, min_child_weight=10, reg_lambda=2.0,  subsample=0.80, colsample_bytree=0.80),
    "Market E": dict(max_depth=6, min_child_weight=12, reg_lambda=3.0,  subsample=0.80, colsample_bytree=0.80),
    "Market F": dict(max_depth=5, min_child_weight=12, reg_lambda=2.0,  subsample=0.85, colsample_bytree=0.85),
}

EARLY_STOPPING_ROUNDS = 300


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def prep_and_fix_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["delivery_start", "delivery_end"]:
        if col in df.columns:
            dt = pd.to_datetime(df[col], errors="coerce", dayfirst=False)
            if dt.isna().mean() > 0.2:
                dt = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            df[col] = dt

    obj_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    for c in obj_cols:
        if c != "market":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "market" in df.columns:
        df["market"] = df["market"].astype(str).str.strip()
    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ds = df["delivery_start"]
    df["hour"] = ds.dt.hour
    df["dow"] = ds.dt.dayofweek
    df["month"] = ds.dt.month
    df["hour_of_week"] = df["dow"] * 24 + df["hour"]
    df["how_sin"] = np.sin(2 * np.pi * df["hour_of_week"] / HOURS_PER_WEEK)
    df["how_cos"] = np.cos(2 * np.pi * df["hour_of_week"] / HOURS_PER_WEEK)
    return df

def add_wind_dir_sin_cos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "wind_direction_80m" in df.columns:
        ang = np.deg2rad(df["wind_direction_80m"])
        df["wind_dir_80m_sin"] = np.sin(ang)
        df["wind_dir_80m_cos"] = np.cos(ang)
        df = df.drop(columns=["wind_direction_80m"], errors="ignore")
    return df

def add_net_load_and_peak(df: pd.DataFrame, peak_start=16, peak_end=20) -> pd.DataFrame:
    df = df.copy()
    if all(c in df.columns for c in ["load_forecast", "wind_forecast", "solar_forecast"]):
        df["net_load"] = df["load_forecast"] - df["wind_forecast"] - df["solar_forecast"]
    if "delivery_start" in df.columns:
        h = df["delivery_start"].dt.hour
        df["is_peak_16_20"] = h.between(peak_start, peak_end).astype(int)
    return df

def add_exog_deltas(df: pd.DataFrame, cols, lags) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["market", "delivery_start"]).reset_index(drop=True)
    for col in cols:
        if col not in df.columns:
            continue
        g = df.groupby("market")[col]
        for k in lags:
            lagcol = f"{col}_lag_{k}"
            df[lagcol] = g.shift(k)
            df[f"{col}_delta_{k}"] = df[col] - df[lagcol]
    return df

def add_netload_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["market", "delivery_start"]).reset_index(drop=True)

    needed = {"load_forecast", "wind_forecast", "solar_forecast"}
    if not needed.issubset(df.columns):
        return df

    if "net_load" not in df.columns:
        df["net_load"] = df["load_forecast"] - df["wind_forecast"] - df["solar_forecast"]

    denom = (df["load_forecast"].abs() + 1e-6)
    df["ren_share"]   = (df["wind_forecast"] + df["solar_forecast"]) / denom
    df["wind_share"]  = df["wind_forecast"] / denom
    df["solar_share"] = df["solar_forecast"] / denom

    gnl = df.groupby("market")["net_load"]
    df["net_load_lag_1"] = gnl.shift(1)
    df["net_load_diff_1"] = df["net_load"] - df["net_load_lag_1"]
    df["net_load_absdiff_1"] = df["net_load_diff_1"].abs()

    for W in NETLOAD_ROLL_WINDOWS:
        df[f"net_load_rollmean_{W}"] = gnl.shift(1).rolling(W, min_periods=max(6, W // 4)).mean()
        df[f"net_load_rollstd_{W}"]  = gnl.shift(1).rolling(W, min_periods=max(6, W // 4)).std()

    for col in ["load_forecast", "wind_forecast", "solar_forecast"]:
        gc = df.groupby("market")[col]
        d1 = gc.diff(1)
        df[f"{col}_diff_1"] = d1
        df[f"{col}_absdiff_1"] = d1.abs()
        df[f"{col}_diff_24"] = df[col] - gc.shift(24)
        df[f"{col}_d1_rollstd_24"] = d1.shift(1).rolling(24, min_periods=12).std()

    return df

def merged_xgb_params(market: str, n_estimators=None, early_stopping_rounds=None) -> dict:
    p = dict(XGB_BASE)
    p.update(MARKET_PARAMS.get(market, {}))
    if n_estimators is not None:
        p["n_estimators"] = int(n_estimators)
    if early_stopping_rounds is not None:
        p["early_stopping_rounds"] = int(early_stopping_rounds)
    else:
        p.pop("early_stopping_rounds", None)
    return p

def build_feature_frame(df: pd.DataFrame, cols_keep=None, max_nan_frac=0.60) -> pd.DataFrame:
    drop_cols = {"target", "id", "market", "delivery_start", "delivery_end"}
    X = df.drop(columns=[c for c in df.columns if c in drop_cols], errors="ignore")
    X = X.select_dtypes(include=[np.number]).copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    if cols_keep is None:
        nan_frac = X.isna().mean()
        keep = nan_frac[nan_frac <= max_nan_frac].index
        X = X[keep]
        nunique = X.nunique(dropna=True)
        X = X.loc[:, nunique > 1]
    else:
        for c in cols_keep:
            if c not in X.columns:
                X[c] = np.nan
        X = X[cols_keep]
    return X

def fit_baseline_how_only(train_df: pd.DataFrame) -> pd.DataFrame:
    return (
        train_df.groupby(["market", "hour_of_week"])["target"]
        .median().rename("baseline_how").reset_index()
    )

def apply_baseline_how_only(df: pd.DataFrame, baseline_tbl: pd.DataFrame, global_mean: float) -> pd.DataFrame:
    out = df.merge(baseline_tbl, on=["market", "hour_of_week"], how="left")
    out["baseline_how"] = out["baseline_how"].fillna(global_mean)
    return out

def clip_preds_to_train_quantiles(train_df: pd.DataFrame, market: str, preds: np.ndarray,
                                 q_low=0.005, q_high=0.995) -> np.ndarray:
    y = train_df.loc[train_df["market"] == market, "target"].values
    lo = float(np.quantile(y, q_low))
    hi = float(np.quantile(y, q_high))
    return np.clip(preds, lo, hi)

def choose_alpha_blend(y_true: np.ndarray, pred_model: np.ndarray, pred_base: np.ndarray) -> float:
    
    best_a, best_r = 0.0, float("inf")
    for a in A_ALPHA_GRID:
        p = a * pred_model + (1.0 - a) * pred_base
        r = rmse(y_true, p)
        if r < best_r:
            best_r = r
            best_a = float(a)
    return best_a



train_raw = pd.read_csv(TRAIN_PATH)
test_raw  = pd.read_csv(TEST_PATH)
print(f"Test data shape: {test_raw.shape}")
print(test_raw.columns)

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
print(f"Submission shape: {sample_sub.shape}")
print(sample_sub.head())

train_raw = add_wind_dir_sin_cos(add_time_features(prep_and_fix_types(train_raw)))
test_raw  = add_wind_dir_sin_cos(add_time_features(prep_and_fix_types(test_raw)))

assert train_raw["delivery_start"].notna().all(), "NaT in train delivery_start"
assert test_raw["delivery_start"].notna().all(), "NaT in test delivery_start"

VAL_WEEKS = 12
VAL_HOURS = VAL_WEEKS * 7 * 24 

A_TRAIN_MONTHS = 18 

train_mkts = sorted(train_raw["market"].unique())
test_mkts  = sorted(test_raw["market"].unique())


train_max = train_raw.groupby("market")["delivery_start"].max()
test_min  = test_raw.groupby("market")["delivery_start"].min()

for m in train_mkts:
    overlap = bool(test_min.get(m) < train_max.get(m))
    
    

if A_TRAIN_MONTHS is not None and A_TRAIN_MONTHS > 0:
    a_mask = train_raw["market"].eq("Market A")
    if a_mask.any():
        a_max_date = train_raw.loc[a_mask, "delivery_start"].max()
        a_cutoff = a_max_date - pd.DateOffset(months=A_TRAIN_MONTHS)

        before = a_mask.sum()
        train_raw = pd.concat([
            train_raw.loc[~a_mask],
            train_raw.loc[a_mask & (train_raw["delivery_start"] >= a_cutoff)]
        ], axis=0, ignore_index=True)

        after = train_raw["market"].eq("Market A").sum()
        


train_raw = add_net_load_and_peak(train_raw)
test_raw  = add_net_load_and_peak(test_raw)

global_mean = float(train_raw["target"].mean())
baseline_tbl = fit_baseline_how_only(train_raw)

train_raw = apply_baseline_how_only(train_raw, baseline_tbl, global_mean)
test_raw  = apply_baseline_how_only(test_raw,  baseline_tbl, global_mean)

full = pd.concat([train_raw.assign(_is_train=1), test_raw.assign(_is_train=0)],
                 axis=0, ignore_index=True)

full = add_netload_extra_features(full)
DELTA_COLS = EXOG_BASE_COLS + (["net_load"] if "net_load" in full.columns else [])
full = add_exog_deltas(full, DELTA_COLS, EXOG_DELTA_LAGS)

train_raw = full.loc[full["_is_train"] == 1].drop(columns=["_is_train"]).reset_index(drop=True)
test_raw  = full.loc[full["_is_train"] == 0].drop(columns=["_is_train"]).reset_index(drop=True)


models = {}
feature_cols_by_market = {}
cv_rmse_by_market = {}
marketA_alpha = 1.0  


for m, g in train_raw.groupby("market", sort=False):
    g = g.sort_values("delivery_start").reset_index(drop=True)

    if m == "Market A":
        
        n = len(g)
        if n >= 3 * VAL_HOURS:
            val_end = n - VAL_HOURS
            val_start = val_end - VAL_HOURS
            tr = g.iloc[:val_start].copy()
            va = g.iloc[val_start:val_end].copy()
        else:
            val_start = n - VAL_HOURS
            tr = g.iloc[:val_start].copy()
            va = g.iloc[val_start:].copy()

        X_tr = build_feature_frame(tr, cols_keep=None, max_nan_frac=MAX_NAN_FRAC)
        X_va = build_feature_frame(va, cols_keep=X_tr.columns, max_nan_frac=MAX_NAN_FRAC)

        
        y_tr = (tr["target"] - tr["baseline_how"]).values
        y_va_fit = (va["target"] - va["baseline_how"]).values
        y_va_true = va["target"].values
        pred_base_va = va["baseline_how"].values

        p_es = merged_xgb_params("Market A", early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        model_es = XGBRegressor(**p_es)
        model_es.fit(X_tr, y_tr, eval_set=[(X_va, y_va_fit)], verbose=False)

        best_iter = getattr(model_es, "best_iteration", None)
        best_n = int(best_iter) + 1 if best_iter is not None else 300
        best_n = int(np.clip(best_n, 50, 1500))

        p = merged_xgb_params("Market A", n_estimators=best_n, early_stopping_rounds=None)
        model_eval = XGBRegressor(**p).fit(X_tr, y_tr)

        pred_resid_va = model_eval.predict(X_va)
        pred_model_va = pred_base_va + pred_resid_va

        
        marketA_alpha = choose_alpha_blend(y_va_true, pred_model_va, pred_base_va)

        
        pred_blend_va = marketA_alpha * pred_model_va + (1.0 - marketA_alpha) * pred_base_va
        pred_blend_va = clip_preds_to_train_quantiles(train_raw, "Market A", pred_blend_va,
                                                      q_low=A_CLIP_Q_LOW, q_high=A_CLIP_Q_HIGH)

        r = rmse(y_va_true, pred_blend_va)

        
        X_full = build_feature_frame(g, cols_keep=X_tr.columns, max_nan_frac=MAX_NAN_FRAC)
        y_full = (g["target"] - g["baseline_how"]).values
        model_final = XGBRegressor(**p).fit(X_full, y_full)

        models[m] = model_final
        feature_cols_by_market[m] = X_tr.columns
        cv_rmse_by_market[m] = r

        
        continue

    
    val_size = min(VAL_HOURS, max(200, int(len(g) * 0.1)))
    tr = g.iloc[:-val_size].copy()
    va = g.iloc[-val_size:].copy()

    X_tr = build_feature_frame(tr, cols_keep=None, max_nan_frac=MAX_NAN_FRAC)
    X_va = build_feature_frame(va, cols_keep=X_tr.columns, max_nan_frac=MAX_NAN_FRAC)
    y_tr = tr["target"].values
    y_va = va["target"].values

    p_es = merged_xgb_params(m, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    model_es = XGBRegressor(**p_es).fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    best_iter = getattr(model_es, "best_iteration", None)
    best_n = int(best_iter) + 1 if best_iter is not None else int(p_es.get("n_estimators", 1000))

    p = merged_xgb_params(m, n_estimators=best_n, early_stopping_rounds=None)
    model_eval = XGBRegressor(**p).fit(X_tr, y_tr)

    pred_va = model_eval.predict(X_va)
    r = rmse(y_va, pred_va)

    
    X_full = build_feature_frame(g, cols_keep=X_tr.columns, max_nan_frac=MAX_NAN_FRAC)
    y_full = g["target"].values
    model_final = XGBRegressor(**p).fit(X_full, y_full)

    models[m] = model_final
    feature_cols_by_market[m] = X_tr.columns
    cv_rmse_by_market[m] = r

    


total = 0.0
count = 0
for m, g in train_raw.groupby("market", sort=False):
    val_size = min(VAL_HOURS, max(200, int(len(g) * 0.1)))
    total += (cv_rmse_by_market[m] ** 2) * val_size
    count += val_size
weighted_rmse = float(np.sqrt(total / count))



test_preds = np.zeros(len(test_raw), dtype=float)

for m, gtest in test_raw.groupby("market", sort=False):
    idxs = gtest.index
    cols = feature_cols_by_market[m]
    X_te = build_feature_frame(gtest, cols_keep=cols, max_nan_frac=MAX_NAN_FRAC)

    pred = models[m].predict(X_te)

    if m == "Market A":
        
        pred_model = gtest["baseline_how"].values + pred
        pred_blend = marketA_alpha * pred_model + (1.0 - marketA_alpha) * gtest["baseline_how"].values
        pred_blend = clip_preds_to_train_quantiles(train_raw, "Market A", pred_blend,
                                                   q_low=A_CLIP_Q_LOW, q_high=A_CLIP_Q_HIGH)
        test_preds[idxs] = pred_blend
    else:
        test_preds[idxs] = pred

pred_df = pd.DataFrame({"id": test_raw["id"].values, "target": test_preds})
pred_df = pred_df.set_index("id").loc[sample_sub["id"]].reset_index()

assert list(pred_df.columns) == ['id', 'target']
assert len(pred_df) == 13098
assert pred_df['id'].min() == 133627
assert pred_df['id'].max() == 146778
assert pred_df['target'].notna().all()
assert np.isfinite(pred_df['target']).all()

print("✅ Validation passed!")

pred_df.to_csv(OUT_PATH, index=False)
print("\n Submission saved:", OUT_PATH, pred_df.shape)
