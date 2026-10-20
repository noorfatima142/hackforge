import shutil
import sys

from src.config import RAW_DIR, DB_PATH, MANIFEST_PATH
from src.fetch_openaq import fetch_all_openaq
from src.fetch_weather import fetch_all_weather
from src.manifest import verify_manifest
from src.build_layers import build_all


def main() -> None:
    print("1/5 Starting from an empty working folder...")
    shutil.rmtree(RAW_DIR, ignore_errors=True)
    for p in (DB_PATH, MANIFEST_PATH):
        if p.exists():
            p.unlink()

    print("2/5 Fetching raw data...")
    n1, n2 = fetch_all_openaq(), fetch_all_weather()

    print("3/5 Verifying SHA-256 hashes against manifest.csv...")
    problems = verify_manifest()
    if problems:
        print("MANIFEST PROBLEMS:", problems)
        sys.exit(1)

    print("4/5 Building cleaned and processed layers...")
    counts = build_all()

    print("5/5 Summary")
    print(f"  raw files: openaq={n1}, weather={n2}")
    for table, n in counts.items():
        print(f"  {table}: {n} rows")
    print("DONE")


if __name__ == "__main__":
    main()