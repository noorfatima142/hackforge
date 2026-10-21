import json
import sqlite3
from datetime import datetime, timezone

from src.config import ROOT

STATUS_DB = ROOT / "data" / "status.sqlite"
STATUS_JSON = ROOT / "data" / "status.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _conn() -> sqlite3.Connection:
    STATUS_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(STATUS_DB)
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS run_status (
        run_id INTEGER PRIMARY KEY AUTOINCREMENT,
        started_utc TEXT, finished_utc TEXT, status TEXT, message TEXT,
        new_openaq_rows INTEGER, gold_rows INTEGER)""")
    con.execute("""CREATE TABLE IF NOT EXISTS source_status (
        source TEXT PRIMARY KEY, last_fetch_utc TEXT,
        latest_data_utc TEXT, rows_in_last_fetch INTEGER)""")
    con.execute("""CREATE TABLE IF NOT EXISTS quality_status (
        check_name TEXT PRIMARY KEY, value TEXT, status TEXT, checked_utc TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS quality_log (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_ts_utc TEXT, rule_id TEXT, table_name TEXT, action TEXT,
        n_rows INTEGER, reason TEXT, example_row_ids TEXT)""")
    return con


def record_run(started, finished, status, message, new_rows, gold_rows) -> None:
    con = _conn()
    with con:
        con.execute("""INSERT INTO run_status
            (started_utc, finished_utc, status, message, new_openaq_rows, gold_rows)
            VALUES (?,?,?,?,?,?)""",
                    (started, finished, status, message, new_rows, gold_rows))
    con.close()


def update_source(source, last_fetch, latest_data, rows) -> None:
    con = _conn()
    with con:
        con.execute("INSERT OR REPLACE INTO source_status VALUES (?,?,?,?)",
                    (source, last_fetch, latest_data, rows))
    con.close()


def update_quality(checks: list[dict]) -> None:
    con = _conn()
    with con:
        for c in checks:
            con.execute("INSERT OR REPLACE INTO quality_status VALUES (?,?,?,?)",
                        (c["check_name"], c["value"], c["status"], c["checked_utc"]))
    con.close()


def export_status_json() -> None:
    """Write everything the dashboard needs into one small JSON file."""
    con = _conn()
    recent = [dict(r) for r in con.execute(
        "SELECT * FROM run_status ORDER BY run_id DESC LIMIT 30")]
    ok = con.execute(
        "SELECT * FROM run_status WHERE status='SUCCESS' ORDER BY run_id DESC LIMIT 1").fetchone()
    last_ts = con.execute("SELECT MAX(run_ts_utc) FROM quality_log").fetchone()[0]
    decisions = [dict(r) for r in con.execute(
        "SELECT * FROM quality_log WHERE run_ts_utc = ? ORDER BY log_id", (last_ts,))] if last_ts else []
    data = {
        "generated_utc": _now(),
        "last_run": recent[0] if recent else None,
        "last_success": dict(ok) if ok else None,
        "sources": [dict(r) for r in con.execute("SELECT * FROM source_status")],
        "quality": [dict(r) for r in con.execute("SELECT * FROM quality_status")],
        "recent_runs": recent,
    }
    con.close()
    STATUS_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")