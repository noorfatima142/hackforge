import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True, encoding="utf-8-sig")

OPENAQ_API_KEY = os.environ.get("OPENAQ_API_KEY", "").strip().strip('"').strip("'")

RAW_DIR = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "warehouse.duckdb"
MANIFEST_PATH = ROOT / "manifest.csv"

OPENAQ_API_KEY = os.environ.get("OPENAQ_API_KEY", "")
OPENAQ_BASE = "https://api.openaq.org/v3"

# city -> (latitude, longitude)
CITIES = {
    "Lahore": (31.5204, 74.3587),
    "Karachi": (24.8607, 67.0011),
    "Islamabad": (33.6844, 73.0479),
    "Peshawar": (34.0151, 71.5249),
}

START_DATE = "2024-10-01"   # at least 2 years of hourly data
SEARCH_RADIUS_M = 25000     # OpenAQ's maximum search radius
MAX_SENSORS_PER_CITY = 8    # keep the run fast