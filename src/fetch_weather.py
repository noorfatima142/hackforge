import json
from datetime import date, timedelta
import requests

from src.config import CITIES, RAW_DIR, START_DATE
from src.manifest import save_raw_and_log

URL = "https://archive-api.open-meteo.com/v1/archive"
LICENCE = "Open-Meteo, CC BY 4.0 (non-commercial free use; attribution required)"
VARS = "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure,precipitation"


def fetch_all_weather() -> int:
    end = (date.today() - timedelta(days=6)).isoformat()   # archive lags a few days
    saved = 0
    for city, (lat, lon) in CITIES.items():
        params = {"latitude": lat, "longitude": lon, "start_date": START_DATE,
                  "end_date": end, "hourly": VARS, "timezone": "UTC"}
        r = requests.get(URL, params=params, timeout=120)
        r.raise_for_status()
        path = RAW_DIR / "openmeteo" / f"{city}.json"
        payload = {"city": city, "data": r.json()}
        save_raw_and_log(json.dumps(payload).encode("utf-8"), path,
                         source="Open-Meteo Archive API",
                         query=f"city={city}; {START_DATE}..{end}; hourly={VARS}",
                         licence=LICENCE)
        saved += 1
        print(f"[weather] saved {path.name}")
    return saved