import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from src.config import MANIFEST_PATH, ROOT

FIELDS = ["source", "query", "fetch_time_utc", "sha256", "licence", "path"]


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_raw_and_log(content: bytes, path: Path, source: str, query: str, licence: str) -> None:
    """Write the raw file once, then add a row to manifest.csv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    row = {
        "source": source,
        "query": query,
        "fetch_time_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": sha256_of_file(path),
        "licence": licence,
        "path": str(path.relative_to(ROOT)),
    }
    new_file = not MANIFEST_PATH.exists()
    with open(MANIFEST_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(row)


def verify_manifest() -> list[str]:
    """Recompute every hash. Returns a list of problems (empty list = all good)."""
    problems = []
    if not MANIFEST_PATH.exists():
        return ["manifest.csv does not exist"]
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p = ROOT / row["path"]
            if not p.exists():
                problems.append(f"missing file: {row['path']}")
            elif sha256_of_file(p) != row["sha256"]:
                problems.append(f"hash mismatch: {row['path']}")
    return problems