"""Unit test of the Data Quality Agent on a tiny hand-made table.
LABELLED SYNTHETIC: used only to test the rules, never for training or evaluation."""
import duckdb
import pandas as pd

from src.dq_agent import DataQualityAgent


def run_agent():
    hours = pd.date_range("2026-10-01", periods=60, freq="h", tz="UTC")
    pm = [50.0 + 0.5 * i for i in range(60)]
    pm[40] = -3.0          # negative -> FIX
    pm[50] = 5000.0        # impossible -> FLAG
    df = pd.DataFrame({"city": "Lahore", "sensor_id": 1, "ts_utc": hours,
                       "pm25": pm, "unit": "µg/m³"})
    df = df.drop(index=[10, 11, 30, 31, 32, 33, 34]).reset_index(drop=True)  # 2 h gap, 5 h gap
    w = pd.DataFrame({"city": ["Lahore"], "ts_utc": [hours[0]], "temperature_2m": [30.0],
                      "relative_humidity_2m": [40.0], "wind_speed_10m": [5.0]})
    con = duckdb.connect()
    con.register("aq", df)
    con.execute("CREATE TABLE cleaned_aq AS SELECT * FROM aq")
    con.register("wx", w)
    con.execute("CREATE TABLE cleaned_weather AS SELECT * FROM wx")
    agent = DataQualityAgent(con)
    agent.run(log=False)
    return con, {d.rule_id: d for d in agent.decisions}


if __name__ == "__main__":
    con, d = run_agent()
    assert d["R04_negative_pm25"].action == "FIX" and d["R04_negative_pm25"].n_rows == 1
    assert d["R05_implausible_high"].action == "FLAG" and d["R05_implausible_high"].n_rows == 1
    assert d["R07_short_gaps"].action == "FILL" and d["R07_short_gaps"].n_rows == 3   # hours 10, 11, 40
    assert d["R08_long_gaps"].action == "FLAG" and d["R08_long_gaps"].n_rows == 5
    assert con.execute("SELECT COUNT(*) FROM cleaned_aq WHERE is_filled").fetchone()[0] == 3
    assert con.execute("SELECT COUNT(*) FROM cleaned_aq WHERE pm25 > 1000 AND NOT dq_exclude").fetchone()[0] == 0
    print("all tests passed")