import json
from datetime import datetime, timedelta, timezone

import requests

from src.config import CITIES, RAW_DIR
from src.fetch_openaq import _get, _fmt, OPENAQ_BASE, LICENCE
from src.fetch_weather import VARS
from src.manifest import save_raw_and_log

LOOKBACK_HOURS = 72
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_LICENCE = "Open-Meteo, CC BY 4.0 (attribution required)"


def known_sensors() -> list[tuple[str, int]]:
    """Sensors found by make_data.py, read from the raw file names (city_sensorid.json)."""
    out = []
    for f in sorted((RAW_DIR / "openaq").glob("*.json")):
        if "_inc_" in f.stem:
            continue
        city, sid = f.stem.rsplit("_", 1)
        out.append((city, int(sid)))
    return out


def fetch_recent_openaq() -> dict:
    sensors = known_sensors()
    if not sensors:
        raise RuntimeError("No raw OpenAQ files found. Run make_data.py first.")
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=LOOKBACK_HOURS)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    new_rows, failed, latest = 0, 0, ""
    for city, sid in sensors:
        try:
            data = _get(f"{OPENAQ_BASE}/sensors/{sid}/hours",
                        {"datetime_from": _fmt(start), "datetime_to": _fmt(now), "limit": 1000})
        except Exception as e:
            failed += 1
            print(f"[recent] sensor {sid} failed: {e}")
            continue
        rows = data.get("results", [])
        if not rows:
            continue
        latest = max([latest] + [r["period"]["datetimeFrom"]["utc"] for r in rows])
        payload = {"city": city, "sensor_id": sid, "location": "incremental", "results": rows}
        path = RAW_DIR / "openaq" / f"{city}_{sid}_inc_{stamp}.json"
        save_raw_and_log(json.dumps(payload).encode("utf-8"), path,
                         source="OpenAQ v3 /sensors/{id}/hours (hourly refresh)",
                         query=f"city={city}; sensor={sid}; last {LOOKBACK_HOURS}h",
                         licence=LICENCE)
        new_rows += len(rows)
    if failed > len(sensors) / 2:
        raise RuntimeError(f"{failed} of {len(sensors)} OpenAQ sensors failed")
    print(f"[recent] openaq: {new_rows} rows, {failed} sensors failed, latest={latest}")
    return {"new_rows": new_rows, "latest_data_utc": latest, "failed": failed}


def fetch_recent_weather() -> dict:
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    now_key = now.strftime("%Y-%m-%dT%H:%M")
    total, latest = 0, ""
    for city, (lat, lon) in CITIES.items():
        r = requests.get(WEATHER_URL, params={
            "latitude": lat, "longitude": lon, "hourly": VARS,
            "past_days": 7, "forecast_days": 1, "timezone": "UTC"}, timeout=60)
        r.raise_for_status()
        j = r.json()
        times = [t for t in j["hourly"]["time"] if t <= now_key]
        if times:
            latest = max(latest, times[-1])
        path = RAW_DIR / "openmeteo" / f"{city}_recent_{stamp}.json"
        save_raw_and_log(json.dumps({"city": city, "data": j}).encode("utf-8"), path,
                         source="Open-Meteo Forecast API (past_days=7)",
                         query=f"city={city}; past 7 days; hourly={VARS}",
                         licence=WEATHER_LICENCE)
        total += len(times)
    print(f"[recent] weather: {total} rows, latest={latest}")
    return {"new_rows": total, "latest_data_utc": latest}