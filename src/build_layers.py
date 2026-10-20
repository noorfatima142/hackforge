import hashlib
import json
import duckdb
import pandas as pd

from src.config import DB_PATH, RAW_DIR


def build_cleaned(con: duckdb.DuckDBPyConnection) -> None:
    # ---- air quality ----
    rows = []
    for f in sorted((RAW_DIR / "openaq").glob("*.json")):
        p = json.loads(f.read_text(encoding="utf-8"))
        for r in p["results"]:
            rows.append({
                "city": p["city"],
                "sensor_id": p["sensor_id"],
                "ts_utc": r["period"]["datetimeFrom"]["utc"],
                "pm25": r["value"],
                "unit": r["parameter"]["units"],
            })
    aq = pd.DataFrame(rows)
    aq["ts_utc"] = pd.to_datetime(aq["ts_utc"], utc=True)
    aq = aq.drop_duplicates(subset=["sensor_id", "ts_utc"], keep="last")   # duplicates handled
    aq["pm25_negative_flag"] = aq["pm25"] < 0       # flagged, not silently deleted
    aq.loc[aq["pm25"] < 0, "pm25"] = None            # missing stays missing (NULL)
    con.register("aq_df", aq)
    con.execute("CREATE OR REPLACE TABLE cleaned_aq AS SELECT * FROM aq_df")

    # ---- weather ----
    wrows = []
    for f in sorted((RAW_DIR / "openmeteo").glob("*.json")):
        p = json.loads(f.read_text(encoding="utf-8"))
        h = p["data"]["hourly"]
        df = pd.DataFrame(h).rename(columns={"time": "ts_utc"})
        df["city"] = p["city"]
        wrows.append(df)
    w = pd.concat(wrows, ignore_index=True)
    w["ts_utc"] = pd.to_datetime(w["ts_utc"], utc=True)
    w = w.drop_duplicates(subset=["city", "ts_utc"], keep="last")
    con.register("w_df", w)
    con.execute("CREATE OR REPLACE TABLE cleaned_weather AS SELECT * FROM w_df")


def build_processed(con: duckdb.DuckDBPyConnection) -> None:
    gold = con.execute("""
        SELECT a.city, a.ts_utc, a.pm25, a.n_sensors,
               w.temperature_2m, w.relative_humidity_2m, w.wind_speed_10m,
               w.surface_pressure, w.precipitation
        FROM (
            SELECT city, ts_utc, AVG(pm25) AS pm25, COUNT(pm25) AS n_sensors
            FROM cleaned_aq GROUP BY city, ts_utc
        ) a
        LEFT JOIN cleaned_weather w ON a.city = w.city AND a.ts_utc = w.ts_utc
        ORDER BY a.city, a.ts_utc
    """).df()
    # stable row id: same city + hour always gives the same id
    gold["row_id"] = [hashlib.md5(f"{c}|{t.isoformat()}".encode()).hexdigest()[:12]
                      for c, t in zip(gold["city"], gold["ts_utc"])]
    assert gold["row_id"].is_unique, "row_id collision"
    con.register("gold_df", gold)
    con.execute("CREATE OR REPLACE TABLE gold_city_hour AS SELECT * FROM gold_df")


def build_all(db_path=None) -> dict:
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    build_cleaned(con)
    build_processed(con)
    counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ["cleaned_aq", "cleaned_weather", "gold_city_hour"]}
    con.close()
    return counts