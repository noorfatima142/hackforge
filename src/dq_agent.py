"""Data Quality Agent.

Tools   : pandera schemas + pandas checks
Decide  : FIX (certain correction) / FILL (marked estimate) / FLAG (leave, warn, exclude)
Explain : each decision stores a reason, a rule id, a row count and example row ids
Log     : appended to quality_log (data/status.sqlite) with a timestamp
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd

try:
    import pandera.pandas as pa
except ImportError:                      # older pandera versions
    import pandera as pa
from pandera.errors import SchemaErrors

from src.config import CITIES
from src.status import _conn, update_quality

# ---- settings: every one is a decision you must be able to justify ----
MAX_FILL_HOURS = 3          # gaps up to 3 h are filled; longer gaps are only flagged
HIGH_PM25 = 1000.0          # above this is treated as physically implausible (sensor limit)
STUCK_HOURS = 24            # identical value for 24+ hours = stuck sensor
STALE_HOURS = {"OpenAQ": 6, "Open-Meteo": 3}
UNIT_CONVERT = {"mg/m3": 1000.0}     # known conversion to ug/m3
MAX_EXCLUDED_SHARE = 0.30            # stop the run if more than 30% of rows are excluded


class DataQualityError(RuntimeError):
    """Raised when the data is too broken to publish."""


@dataclass
class Decision:
    rule_id: str
    table: str
    action: str            # PASS, FIX, FILL or FLAG
    n_rows: int
    reason: str
    example_row_ids: list = field(default_factory=list)


def _norm_unit(u) -> str:
    return str(u).lower().replace("µ", "u").replace("μ", "u").replace("³", "3").strip()


def _ids(sub: pd.DataFrame, k: int = 10) -> list:
    sub = sub.head(k)
    return [f"{s}|{t.strftime('%Y-%m-%dT%H:%MZ')}" for s, t in zip(sub["sensor_id"], sub["ts_utc"])]


class DataQualityAgent:
    def __init__(self, con: duckdb.DuckDBPyConnection):
        self.con = con
        self.run_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.decisions: list[Decision] = []

    # ------------------------------------------------------------ helpers
    def _record(self, rule_id, action, n, reason, ids=None, table="cleaned_aq"):
        self.decisions.append(Decision(rule_id, table, action, int(n), reason, ids or []))

    # ------------------------------------------------------------ main
    def run(self, log: bool = True) -> list:
        try:
            df = self.con.execute("SELECT * FROM cleaned_aq").df()
            self._check_schema(df)
            df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
            df["dq_exclude"] = False
            df["is_filled"] = False
            df["dq_note"] = ""
            df = self._duplicates(df)
            df = self._units(df)
            df = self._negative(df)
            df = self._implausible_high(df)
            df = self._stuck(df)
            df = self._gaps(df)
            self._weather_ranges()
            self._freshness(df)
            self._guard(df)
            self._save(df)
        finally:
            self._print()
            if log:
                self._log()
        return self.decisions

    # ------------------------------------------------------------ R01
    def _check_schema(self, df):
        schema = pa.DataFrameSchema({
            "city": pa.Column(checks=pa.Check.isin(list(CITIES))),
            "sensor_id": pa.Column(checks=pa.Check(lambda s: pd.api.types.is_integer_dtype(s))),
            "ts_utc": pa.Column(checks=pa.Check(lambda s: pd.api.types.is_datetime64_any_dtype(s))),
            "pm25": pa.Column(nullable=True, checks=pa.Check(lambda s: pd.api.types.is_numeric_dtype(s))),
            "unit": pa.Column(nullable=True),
        })
        try:
            schema.validate(df, lazy=True)
        except SchemaErrors as e:
            n = len(e.failure_cases)
            self._record("R01_schema", "FLAG", n,
                         f"pandera schema failed with {n} failures; run stopped, nothing published")
            raise DataQualityError(f"Schema check failed ({n} failures)") from e
        self._record("R01_schema", "PASS", 0, "columns, types and city names match the schema")

    # ------------------------------------------------------------ R02
    def _duplicates(self, df):
        dup = df.duplicated(["sensor_id", "ts_utc"], keep="last")
        n = int(dup.sum())
        if n:
            ids = _ids(df[dup])
            df = df[~dup].reset_index(drop=True)
            self._record("R02_duplicates", "FIX", n,
                         "same sensor and hour appeared more than once (hourly refresh re-fetches "
                         "72 h); kept the latest copy so late corrections win", ids)
        else:
            self._record("R02_duplicates", "PASS", 0, "no duplicate (sensor, hour) keys")
        return df

    # ------------------------------------------------------------ R03
    def _units(self, df):
        norm = df["unit"].map(_norm_unit)
        ok = norm == "ug/m3"
        factor = norm.map(UNIT_CONVERT)
        fixable = ~ok & factor.notna()
        unknown = ~ok & factor.isna()
        if fixable.any():
            df.loc[fixable, "pm25"] = df.loc[fixable, "pm25"] * factor[fixable]
            df.loc[fixable, "dq_note"] = "unit_converted"
            self._record("R03_units", "FIX", fixable.sum(),
                         "unit had a known conversion (mg/m3 -> ug/m3); multiplied by 1000",
                         _ids(df[fixable]))
        if unknown.any():
            df.loc[unknown, "dq_exclude"] = True
            df.loc[unknown, "dq_note"] = "unit_unknown"
            self._record("R03_units_unknown", "FLAG", unknown.sum(),
                         "unit is not ug/m3 and has no known conversion; excluded, not guessed",
                         _ids(df[unknown]))
        if not fixable.any() and not unknown.any():
            self._record("R03_units", "PASS", 0, "all rows are in ug/m3")
        df["unit"] = "ug/m3"        # one spelling for the whole table
        return df

    # ------------------------------------------------------------ R04
    def _negative(self, df):
        neg = df["pm25"] < 0
        n = int(neg.sum())
        if n:
            ids = _ids(df[neg])
            df.loc[neg, "pm25"] = np.nan
            df.loc[neg, "dq_note"] = "negative_set_missing"
            self._record("R04_negative_pm25", "FIX", n,
                         "a concentration cannot be negative; set to missing (not zero). "
                         "Raw file keeps the original value", ids)
        else:
            self._record("R04_negative_pm25", "PASS", 0, "no negative PM2.5 values")
        return df

    # ------------------------------------------------------------ R05
    def _implausible_high(self, df):
        high = (df["pm25"] > HIGH_PM25) & ~df["dq_exclude"]
        n = int(high.sum())
        if n:
            df.loc[high, "dq_exclude"] = True
            df.loc[high, "dq_note"] = "implausible_high"
            self._record("R05_implausible_high", "FLAG", n,
                         f"PM2.5 above {HIGH_PM25:.0f} ug/m3 cannot be confirmed; value kept in "
                         "cleaned but excluded from the gold table", _ids(df[high]))
        else:
            self._record("R05_implausible_high", "PASS", 0, f"no values above {HIGH_PM25:.0f} ug/m3")
        return df

    # ------------------------------------------------------------ R06
    def _stuck(self, df):
        d = df.sort_values(["sensor_id", "ts_utc"])
        v = d[d["pm25"].notna() & ~d["dq_exclude"]]
        run = v["pm25"].ne(v.groupby("sensor_id")["pm25"].shift()).cumsum()
        run_len = v.groupby(run)["pm25"].transform("size")
        stuck_idx = v.index[(run_len >= STUCK_HOURS).values]
        if len(stuck_idx):
            df.loc[stuck_idx, "dq_exclude"] = True
            df.loc[stuck_idx, "dq_note"] = "stuck_sensor"
            self._record("R06_stuck_sensor", "FLAG", len(stuck_idx),
                         f"same exact value for {STUCK_HOURS}+ consecutive readings looks like a "
                         "frozen sensor; excluded from gold", _ids(df.loc[stuck_idx]))
        else:
            self._record("R06_stuck_sensor", "PASS", 0, "no frozen sensors found")
        return df

    # ------------------------------------------------------------ R07 / R08
    def _gaps(self, df):
        # rows without any value carry no information: they stay missing (no row)
        df = df[~(df["pm25"].isna() & ~df["dq_exclude"])].copy()
        new_rows, long_hours, long_runs = [], 0, 0
        for sid, g in df.groupby("sensor_id"):
            gi = g.sort_values("ts_utc").set_index("ts_utc")
            s = gi["pm25"].where(~gi["dq_exclude"]).asfreq("h")
            blocked = gi["dq_exclude"].reindex(s.index, fill_value=False).astype(bool)
            isna = s.isna()
            grp = isna.ne(isna.shift()).cumsum()
            run_len = isna.groupby(grp).transform("sum")
            run_blocked = blocked.groupby(grp).transform("max")
            inside = s.ffill().notna() & s.bfill().notna()       # gap has valid readings on both sides
            fill = isna & inside & (run_len <= MAX_FILL_HOURS) & ~run_blocked
            long_gap = isna & inside & (run_len > MAX_FILL_HOURS) & ~blocked
            if fill.any():
                interp = s.interpolate(method="linear", limit_area="inside")
                new_rows.append(pd.DataFrame({
                    "city": g["city"].iloc[0], "sensor_id": sid,
                    "ts_utc": s.index[fill.values], "pm25": interp[fill].values,
                    "unit": "ug/m3", "dq_exclude": False, "is_filled": True,
                    "dq_note": "filled_short_gap"}))
            long_hours += int(long_gap.sum())
            long_runs += int(grp[long_gap].nunique())
        if new_rows:
            filled = pd.concat(new_rows, ignore_index=True)
            df = pd.concat([df, filled], ignore_index=True)
            self._record("R07_short_gaps", "FILL", len(filled),
                         f"gaps of {MAX_FILL_HOURS} h or less between valid readings filled by "
                         "linear interpolation and marked is_filled=TRUE (visible downstream)",
                         _ids(filled))
        else:
            self._record("R07_short_gaps", "PASS", 0, "no short gaps to fill")
        if long_hours:
            self._record("R08_long_gaps", "FLAG", long_hours,
                         f"{long_runs} gaps longer than {MAX_FILL_HOURS} h left missing: filling "
                         "them would be guessing, so the uncertainty stays visible")
        else:
            self._record("R08_long_gaps", "PASS", 0, "no long gaps")
        return df

    # ------------------------------------------------------------ R09
    def _freshness(self, df):
        now = pd.Timestamp.now(tz="UTC")
        use = df[~df["dq_exclude"] & ~df["is_filled"]]
        limit = STALE_HOURS["OpenAQ"]
        for city in CITIES:
            g = use[use["city"] == city]
            if g.empty:
                self._record(f"R09_stale_{city}", "FLAG", 0, f"{city}: no usable readings at all")
                continue
            age = (now - g["ts_utc"].max()).total_seconds() / 3600
            if age > limit:
                self._record(f"R09_stale_{city}", "FLAG", 0,
                             f"newest {city} reading is {age:.1f} h old (limit {limit} h)")
            else:
                self._record(f"R09_stale_{city}", "PASS", 0,
                             f"newest {city} reading is {age:.1f} h old (limit {limit} h)")
        latest = self.con.execute(
            "SELECT MAX(ts_utc) AS m FROM cleaned_weather WHERE ts_utc <= now()").df()["m"].iloc[0]
        wl = STALE_HOURS["Open-Meteo"]
        if pd.isna(latest):
            self._record("R09_stale_weather", "FLAG", 0, "no weather data found", table="cleaned_weather")
        else:
            age = (now - pd.to_datetime(latest, utc=True)).total_seconds() / 3600
            self._record("R09_stale_weather", "FLAG" if age > wl else "PASS", 0,
                         f"newest weather value is {age:.1f} h old (limit {wl} h)",
                         table="cleaned_weather")

    # ------------------------------------------------------------ R10
    def _weather_ranges(self):
        w = self.con.execute(
            "SELECT temperature_2m, relative_humidity_2m, wind_speed_10m FROM cleaned_weather").df()
        schema = pa.DataFrameSchema({
            "temperature_2m": pa.Column(nullable=True, checks=pa.Check.in_range(-20, 55)),
            "relative_humidity_2m": pa.Column(nullable=True, checks=pa.Check.in_range(0, 100)),
            "wind_speed_10m": pa.Column(nullable=True, checks=pa.Check.ge(0)),
        })
        try:
            schema.validate(w, lazy=True)
            self._record("R10_weather_ranges", "PASS", 0,
                         "temperature (C), humidity (%) and wind are in plausible ranges",
                         table="cleaned_weather")
        except SchemaErrors as e:
            n = int(e.failure_cases["index"].nunique())
            self._record("R10_weather_ranges", "FLAG", n,
                         "weather values outside plausible ranges (possible unit error, e.g. Kelvin)",
                         table="cleaned_weather")

    # ------------------------------------------------------------ R11
    def _guard(self, df):
        n = int(df["dq_exclude"].sum())
        share = float(df["dq_exclude"].mean()) if len(df) else 1.0
        if share > MAX_EXCLUDED_SHARE:
            self._record("R11_exclusion_share", "FLAG", n,
                         f"{share:.0%} of rows excluded (limit {MAX_EXCLUDED_SHARE:.0%}); "
                         "run stopped so a broken table is never published")
            raise DataQualityError(f"Too many rows excluded: {share:.0%}")
        self._record("R11_exclusion_share", "PASS", n,
                     f"{share:.2%} of rows excluded (limit {MAX_EXCLUDED_SHARE:.0%})")

    # ------------------------------------------------------------ output
    def _save(self, df):
        out = df.sort_values(["city", "sensor_id", "ts_utc"]).reset_index(drop=True)
        self.con.register("dq_df", out)
        self.con.execute("CREATE OR REPLACE TABLE cleaned_aq AS SELECT * FROM dq_df")
        self.con.unregister("dq_df")

    def checks(self) -> list:
        return [{"check_name": f"{d.rule_id} -> {d.action}", "value": f"{d.n_rows} rows",
                 "status": "WARN" if d.action == "FLAG" else "OK",
                 "checked_utc": self.run_ts} for d in self.decisions]

    def _print(self):
        for d in self.decisions:
            print(f"[dq] {d.rule_id:<24} {d.action:<5} {d.n_rows:>8} rows")

    def _log(self):
        con = _conn()
        with con:
            for d in self.decisions:
                con.execute(
                    "INSERT INTO quality_log (run_ts_utc, rule_id, table_name, action, n_rows, "
                    "reason, example_row_ids) VALUES (?,?,?,?,?,?,?)",
                    (self.run_ts, d.rule_id, d.table, d.action, d.n_rows, d.reason,
                     ", ".join(d.example_row_ids)))
            con.execute("DELETE FROM quality_status")
        con.close()
        update_quality(self.checks())