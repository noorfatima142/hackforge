import json
import sys

import duckdb
import numpy as np
import pandas as pd

from src.config import DB_PATH, ROOT

MIN_HOURS = 18
TEST_DAYS = 90
VAL_DAYS = 90
BINS = [0, 12.0, 35.4, 55.4, 150.4, 250.4, np.inf]
LABELS = ["Good", "Moderate", "Unhealthy for sensitive", "Unhealthy",
          "Very unhealthy", "Hazardous"]
OUT = ROOT / "data" / "baseline_results.json"


def load_daily() -> pd.DataFrame:
    """Daily mean PM2.5 per city, with the next day's mean as the target."""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute(
        "SELECT city, ts_utc, pm25 FROM gold_city_hour WHERE pm25 IS NOT NULL").df()
    con.close()
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df["date"] = (df["ts_utc"].dt.tz_convert("Asia/Karachi")
                  .dt.tz_localize(None).dt.normalize())
    daily = (df.groupby(["city", "date"])
               .agg(pm25_today=("pm25", "mean"), n_hours=("pm25", "count"))
               .reset_index())
    daily = daily[daily["n_hours"] >= MIN_HOURS]

    nxt = daily[["city", "date", "pm25_today"]].copy()
    nxt["date"] = nxt["date"] - pd.Timedelta(days=1)       # match on real dates, not row order
    nxt = nxt.rename(columns={"pm25_today": "target"})
    out = daily.merge(nxt, on=["city", "date"], how="inner")
    return out.sort_values(["city", "date"]).reset_index(drop=True)


def split_by_time(d: pd.DataFrame):
    last = d["date"].max()
    test_start = last - pd.Timedelta(days=TEST_DAYS - 1)
    val_start = test_start - pd.Timedelta(days=VAL_DAYS)
    train = d[d["date"] < val_start]
    val = d[(d["date"] >= val_start) & (d["date"] < test_start)]
    test = d[d["date"] >= test_start]
    assert train["date"].max() < val["date"].min() <= val["date"].max() < test["date"].min()
    return train, val, test


def category(x: pd.Series) -> pd.Series:
    return pd.cut(x, bins=BINS, labels=LABELS, include_lowest=True)


def score(d: pd.DataFrame) -> dict:
    err = d["pm25_today"] - d["target"]
    return {
        "n_days": int(len(d)),
        "MAE": round(float(err.abs().mean()), 2),
        "RMSE": round(float(np.sqrt((err ** 2).mean())), 2),
        "category_accuracy_%": round(float(
            (category(d["pm25_today"]) == category(d["target"])).mean() * 100), 1),
    }


def report(d: pd.DataFrame, name: str) -> dict:
    res = {"overall": score(d), "by_city": {}, "by_month": {}}
    for c, g in d.groupby("city"):
        res["by_city"][c] = score(g)
    for m, g in d.groupby(d["date"].dt.strftime("%Y-%m")):
        res["by_month"][m] = score(g)
    print(f"\n=== Persistence baseline on {name} ===")
    print("overall:", res["overall"])
    print(pd.DataFrame(res["by_city"]).T)
    print(pd.DataFrame(res["by_month"]).T)
    return res


if __name__ == "__main__":
    d = load_daily()
    train, val, test = split_by_time(d)
    print(f"train {len(train)} rows ({train['date'].min().date()} to {train['date'].max().date()})")
    print(f"val   {len(val)} rows ({val['date'].min().date()} to {val['date'].max().date()})")
    print(f"test  {len(test)} rows ({test['date'].min().date()} to {test['date'].max().date()})")
    results = {"validation": report(val, "VALIDATION")}
    if "--final" in sys.argv:                      # test set: run this once, at the very end
        results["test"] = report(test, "TEST")
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nsaved {OUT}")