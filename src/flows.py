import os
import sys
from datetime import datetime, timezone

import duckdb
from prefect import flow, task

from src import status
from src.build_layers import build_all
from src.config import DB_PATH
from src.fetch_recent import fetch_recent_openaq, fetch_recent_weather
from src.hf_publish import publish_status


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def basic_checks(db_path) -> list[dict]:
    """Simple health checks. The full Data Quality Agent replaces this in Module 4."""
    con = duckdb.connect(str(db_path), read_only=True)
    now, out = _now(), []

    def add(name, value, ok):
        out.append({"check_name": name, "value": str(value),
                    "status": "OK" if ok else "WARN", "checked_utc": now})

    total = con.execute("SELECT COUNT(*) FROM gold_city_hour").fetchone()[0]
    add("Gold rows (need 50,000+)", total, total >= 50000)
    neg = con.execute("SELECT COUNT(*) FROM cleaned_aq WHERE pm25_negative_flag").fetchone()[0]
    add("Negative PM2.5 readings set to missing", neg, True)
    huge = con.execute("SELECT COUNT(*) FROM cleaned_aq WHERE pm25 > 1000").fetchone()[0]
    add("Implausible readings (> 1000 ug/m3) present", huge, huge == 0)
    n = con.execute("""SELECT COUNT(DISTINCT city) FROM gold_city_hour
                       WHERE ts_utc >= now() - INTERVAL 6 HOUR""").fetchone()[0]
    add("Cities with data in last 6 hours (of 4)", n, n == 4)
    miss = con.execute("""SELECT 100.0 * AVG(CASE WHEN temperature_2m IS NULL THEN 1 ELSE 0 END)
                          FROM gold_city_hour""").fetchone()[0]
    add("Rows missing weather (%)", round(miss or 0, 1), (miss or 0) < 10)
    con.close()
    return out


@task(retries=3, retry_delay_seconds=60)
def fetch_openaq_task() -> dict:
    return fetch_recent_openaq()


@task(retries=3, retry_delay_seconds=60)
def fetch_weather_task() -> dict:
    return fetch_recent_weather()


@task
def rebuild_task():
    """Build into a temporary database, and swap it in only if everything worked."""
    tmp = DB_PATH.with_name("warehouse_tmp.duckdb")
    if tmp.exists():
        tmp.unlink()
    counts = build_all(tmp)
    checks = basic_checks(tmp)
    try:
        os.replace(tmp, DB_PATH)
    except PermissionError:
        raise RuntimeError("warehouse.duckdb is open in another program. Close it and retry.")
    return counts, checks


@flow(name="hourly-airquality", log_prints=True)
def hourly_flow() -> None:
    started = _now()
    try:
        aq = fetch_openaq_task()
        wx = fetch_weather_task()
        counts, checks = rebuild_task()
        finished = _now()
        status.update_source("OpenAQ", finished, aq["latest_data_utc"], aq["new_rows"])
        status.update_source("Open-Meteo", finished, wx["latest_data_utc"], wx["new_rows"])
        status.update_quality(checks)
        status.record_run(started, finished, "SUCCESS",
                          f"{aq['failed']} sensors failed", aq["new_rows"],
                          counts["gold_city_hour"])
    except Exception as e:
        status.record_run(started, _now(), "FAILED", str(e)[:500], 0, 0)
        raise
    finally:
        status.export_status_json()
        try:
            publish_status()
        except Exception as e:
            print(f"[publish] upload failed (flow result unaffected): {e}")


if __name__ == "__main__":
    if "--once" in sys.argv:
        hourly_flow()
    else:
        hourly_flow.serve(name="hourly-airquality-deployment", cron="5 * * * *")