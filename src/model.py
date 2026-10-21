import json
import sys
import time

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.baseline import split_by_time, category, MIN_HOURS
from src.config import DB_PATH, ROOT

SEED = 42
FEATURES = ["pm25_today", "lag1", "lag2", "lag3", "lag7", "roll7", "temp", "hum", "wind",
            "pres", "rain", "dow", "month", "doy_sin", "doy_cos", "city_code"]
CITY_CODES = {"Islamabad": 0, "Karachi": 1, "Lahore": 2, "Peshawar": 3}
OUT = ROOT / "data" / "model_results.json"
MODEL_DIR = ROOT / "data" / "models"


def load_features() -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("""SELECT city, ts_utc, pm25, temperature_2m, relative_humidity_2m,
                        wind_speed_10m, surface_pressure, precipitation
                        FROM gold_city_hour WHERE pm25 IS NOT NULL""").df()
    con.close()
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df["date"] = df["ts_utc"].dt.tz_convert("Asia/Karachi").dt.tz_localize(None).dt.normalize()
    daily = (df.groupby(["city", "date"])
               .agg(pm25_today=("pm25", "mean"), n_hours=("pm25", "count"),
                    temp=("temperature_2m", "mean"), hum=("relative_humidity_2m", "mean"),
                    wind=("wind_speed_10m", "mean"), pres=("surface_pressure", "mean"),
                    rain=("precipitation", lambda s: s.sum(min_count=1)))
               .reset_index())
    daily.loc[daily["n_hours"] < MIN_HOURS, "pm25_today"] = np.nan   # same rule as the baseline
    parts = []
    for city, g in daily.groupby("city"):
        g = g.set_index("date").reindex(pd.date_range(g["date"].min(), g["date"].max()))
        g["city"] = city
        for k in (1, 2, 3, 7):
            g[f"lag{k}"] = g["pm25_today"].shift(k)           # real calendar days, not row order
        g["roll7"] = g["pm25_today"].rolling(7, min_periods=4).mean()   # includes today, known at forecast time
        g["target"] = g["pm25_today"].shift(-1)
        g.index.name = "date"
        parts.append(g.reset_index())
    d = pd.concat(parts, ignore_index=True)
    d = d[d["pm25_today"].notna() & d["target"].notna()].copy()    # same rows as the baseline
    d["dow"] = d["date"].dt.dayofweek
    d["month"] = d["date"].dt.month
    d["doy_sin"] = np.sin(2 * np.pi * d["date"].dt.dayofyear / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * d["date"].dt.dayofyear / 365.25)
    d["city_code"] = d["city"].map(CITY_CODES)
    return d.sort_values(["city", "date"]).reset_index(drop=True)


def _params(objective, alpha=None):
    p = dict(objective=objective, n_estimators=800, learning_rate=0.03, num_leaves=15,
             min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
             reg_lambda=1.0, random_state=SEED, verbose=-1, n_jobs=2)
    if alpha is not None:
        p["alpha"] = alpha
    return p


def fit(train, val):
    models = {}
    for name, obj, alpha in [("point", "regression_l1", None),
                             ("lo", "quantile", 0.10), ("hi", "quantile", 0.90)]:
        m = lgb.LGBMRegressor(**_params(obj, alpha))
        m.fit(train[FEATURES], train["target"],
              eval_set=[(val[FEATURES], val["target"])],
              callbacks=[lgb.early_stopping(50, verbose=False)])   # tuned on validation only
        models[name] = m
    return models

def predict(models, d, q=0.0):
    out = d[["city", "date", "target", "pm25_today"]].copy()
    out["pred"] = models["point"].predict(d[FEATURES]).clip(min=0)
    lo = models["lo"].predict(d[FEATURES]).clip(min=0)
    hi = models["hi"].predict(d[FEATURES]).clip(min=0)
    out["lo"] = np.maximum(np.minimum(lo, out["pred"]) - q, 0)
    out["hi"] = np.maximum(hi, out["pred"]) + q
    return out

def conformal_offset(models, cal, coverage=0.80) -> float:
    p = predict(models, cal)
    score = np.maximum(p["lo"] - p["target"], p["target"] - p["hi"])
    k = int(np.ceil((len(score) + 1) * coverage))
    return float(np.sort(score.values)[min(k, len(score)) - 1])

def metrics(y, p, lo=None, hi=None) -> dict:
    err = p - y
    r = {"n": int(len(y)), "MAE": round(float(err.abs().mean()), 2),
         "RMSE": round(float(np.sqrt((err ** 2).mean())), 2),
         "cat_acc_%": round(float((category(p) == category(y)).mean() * 100), 1)}
    if lo is not None:
        r["cover_80_%"] = round(float(((y >= lo) & (y <= hi)).mean() * 100), 1)
        r["width"] = round(float((hi - lo).mean()), 1)
    return r


def report(pr: pd.DataFrame, name: str) -> dict:
    res = {"model": metrics(pr["target"], pr["pred"], pr["lo"], pr["hi"]),
           "baseline": metrics(pr["target"], pr["pm25_today"])}
    print(f"\n=== {name} ===")
    print("MODEL   :", res["model"])
    print("BASELINE:", res["baseline"])
    groups = {"city": pr["city"], "month": pr["date"].dt.strftime("%Y-%m"),
              "pollution_level": category(pr["pm25_today"]).astype(str)}
    for gname, key in groups.items():
        rows = []
        for k, g in pr.groupby(key):
            m, b = metrics(g["target"], g["pred"], g["lo"], g["hi"]), metrics(g["target"], g["pm25_today"])
            rows.append({gname: k, "n": m["n"], "model_MAE": m["MAE"], "base_MAE": b["MAE"],
                         "gain_%": round(100 * (b["MAE"] - m["MAE"]) / b["MAE"], 1),
                         "cover_80_%": m["cover_80_%"]})
        t = pd.DataFrame(rows)
        res[f"by_{gname}"] = t.to_dict("records")
        print(f"\nBy {gname}:\n{t.to_string(index=False)}")
        worst = t.sort_values("gain_%").iloc[0]
        print(f"-> weakest {gname}: {worst[gname]} (gain {worst['gain_%']}%)")
    return res

if __name__ == "__main__":
    t0 = time.time()
    d = load_features()
    train, val, test = split_by_time(d)
    print(f"train {len(train)}  val {len(val)}  test {len(test)}  features {len(FEATURES)}")
    models = fit(train, val)
    print("trees used:", {k: m.best_iteration_ for k, m in models.items()})
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for k, m in models.items():
        m.booster_.save_model(str(MODEL_DIR / f"{k}.txt"))
    q = conformal_offset(models, val)
    print(f"conformal offset: {q:.2f} ug/m3")
    results = {"conformal_offset": round(q, 2),
               "validation": report(predict(models, val, q), "VALIDATION")}
    if "--final" in sys.argv:        # test set: run ONCE, at the very end
        results["test"] = report(predict(models, test, q), "TEST (final)")
    results["train_seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\ntotal time {results['train_seconds']} s; saved {OUT}")
