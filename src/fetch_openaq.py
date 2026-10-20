import json
import time
from datetime import datetime, timedelta, timezone

import requests

from src.config import (CITIES, OPENAQ_API_KEY, OPENAQ_BASE, RAW_DIR, SEARCH_RADIUS_M,
                        START_DATE, MAX_SENSORS_PER_CITY)
from src.manifest import save_raw_and_log

HEADERS = {"X-API-Key": OPENAQ_API_KEY}
LICENCE = "OpenAQ aggregated data; licence depends on each provider (verify at openaq.org)"
WINDOW_DAYS = 30            # 30 days of hourly data = 720 rows, fits in one page
PAUSE = 1.1                 # seconds between requests (free tier allows ~60 per minute)
RETRY_CODES = {408, 500, 502, 503, 504}


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _get(url: str, params: dict) -> dict:
    """GET with retries for timeouts, server errors and rate limits."""
    for attempt in range(6):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=90)
        except (requests.Timeout, requests.ConnectionError):
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 200:
            time.sleep(PAUSE)
            return r.json()
        if r.status_code == 429:
            time.sleep(15 * (attempt + 1))
            continue
        if r.status_code in RETRY_CODES:
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
    raise RuntimeError(f"Failed after retries: {url} {params}")


def _pm25_sensors(city: str, lat: float, lon: float) -> list[dict]:
    data = _get(f"{OPENAQ_BASE}/locations",
                {"coordinates": f"{lat},{lon}", "radius": SEARCH_RADIUS_M, "limit": 100})
    start_cut = _parse(START_DATE + "T00:00:00Z")
    recent = datetime.now(timezone.utc) - timedelta(days=14)
    sensors = []
    for loc in data.get("results", []):
        if "test" in loc["name"].lower():          # never use test devices
            continue
        first = (loc.get("datetimeFirst") or {}).get("utc")
        last = (loc.get("datetimeLast") or {}).get("utc")
        if not first or not last:
            continue
        for s in loc.get("sensors", []):
            if s["parameter"]["name"] == "pm25":
                sensors.append({"id": s["id"], "location": loc["name"],
                                "first": _parse(first), "last": _parse(last)})
    # keep sensors that overlap our period and are either still reporting
    # or have history reaching back to START_DATE
    keep = [s for s in sensors
            if s["last"] >= start_cut and (s["last"] >= recent or s["first"] <= start_cut)]
    keep.sort(key=lambda s: s["first"])            # longest history first
    return keep[:MAX_SENSORS_PER_CITY]


def _fetch_window(sensor_id: int, start: datetime, end: datetime) -> list[dict]:
    """Fetch one date window. If it keeps failing, split it in half and retry."""
    rows, page = [], 1
    try:
        while True:
            data = _get(f"{OPENAQ_BASE}/sensors/{sensor_id}/hours",
                        {"datetime_from": _fmt(start), "datetime_to": _fmt(end),
                         "limit": 1000, "page": page})
            batch = data.get("results", [])
            rows.extend(batch)
            if len(batch) < 1000:
                return rows
            page += 1
    except RuntimeError:
        if end - start <= timedelta(days=2):
            raise                                   # fail loudly, no silent gaps
        mid = start + (end - start) / 2
        return _fetch_window(sensor_id, start, mid) + _fetch_window(sensor_id, mid, end)


def fetch_sensor_hours(sensor_id: int, first: datetime) -> list[dict]:
    begin = max(_parse(START_DATE + "T00:00:00Z"), first)
    now = datetime.now(timezone.utc)
    rows, cur = [], begin
    while cur < now:
        nxt = min(cur + timedelta(days=WINDOW_DAYS), now)
        rows.extend(_fetch_window(sensor_id, cur, nxt))
        cur = nxt
    return rows


def fetch_all_openaq() -> int:
    saved = 0
    for city, (lat, lon) in CITIES.items():
        sensors = _pm25_sensors(city, lat, lon)
        print(f"[openaq] {city}: {len(sensors)} pm25 sensors")
        for s in sensors:
            rows = fetch_sensor_hours(s["id"], s["first"])
            if not rows:
                continue
            payload = {"city": city, "sensor_id": s["id"], "location": s["location"], "results": rows}
            path = RAW_DIR / "openaq" / f"{city}_{s['id']}.json"
            save_raw_and_log(json.dumps(payload).encode("utf-8"), path,
                             source="OpenAQ v3 /sensors/{id}/hours",
                             query=f"city={city}; sensor={s['id']}; location={s['location']}; from={START_DATE}",
                             licence=LICENCE)
            saved += 1
            print(f"  saved {path.name}: {s['location']} ({len(rows)} rows)")
    return saved