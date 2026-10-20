import requests
from src.config import CITIES, OPENAQ_API_KEY, OPENAQ_BASE, SEARCH_RADIUS_M

headers = {"X-API-Key": OPENAQ_API_KEY}

for city, (lat, lon) in CITIES.items():
    r = requests.get(
        f"{OPENAQ_BASE}/locations",
        headers=headers,
        params={"coordinates": f"{lat},{lon}", "radius": SEARCH_RADIUS_M, "limit": 100},
        timeout=60,
    )
    if r.status_code != 200:
        print(city, "ERROR", r.status_code, r.text[:200])
        continue
    results = r.json().get("results", [])
    print(f"\n=== {city}: {len(results)} locations ===")
    for loc in results:
        pm25 = [s for s in loc.get("sensors", []) if s["parameter"]["name"] == "pm25"]
        if pm25:
            first = (loc.get("datetimeFirst") or {}).get("utc", "?")
            last = (loc.get("datetimeLast") or {}).get("utc", "?")
            print(f"  {loc['name']}: pm25 sensors={len(pm25)}  from {first[:10]} to {last[:10]}")