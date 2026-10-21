"""Insight Agent (RAG).
Rule 1: every number comes from gold-table rows (row ids stored) or from a cited document.
Rule 2: every answer cites a document section.
Rule 3: if unsupported, answer "I don't know" and say what is missing.
Wording is a code template, so no language model can invent a number.
"""
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone

import chromadb
import duckdb
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.baseline import category
from src.config import CITIES, DB_PATH, ROOT

DOCS_DIR = ROOT / "docs"
CHROMA_DIR = ROOT / "data" / "chroma"
LOG_FILE = ROOT / "data" / "insights.jsonl"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
MIN_SIM = 0.35        # below this similarity a section is not "relevant"; tune with the tests
TOP_K = 3
WINDOW_H = 24
MAX_AGE_H = 6         # data older than this is too stale to advise on
ALIASES = {"لاہور": "Lahore", "کراچی": "Karachi", "اسلام آباد": "Islamabad", "پشاور": "Peshawar"}
EMERGENCY = ("This is information, not medical advice. If anyone has trouble breathing, chest pain "
             "or severe symptoms, contact local emergency services (in Pakistan, Rescue 1122) "
             "or go to the nearest hospital.")
NUM = re.compile(r"\d+(?:\.\d+)?")

_model = None


@dataclass
class Insight:
    question: str
    answered: bool
    text: str
    city: str | None = None
    gold_row_ids: list = field(default_factory=list)
    citations: list = field(default_factory=list)


def _embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def load_chunks() -> list[dict]:
    chunks = []
    for f in sorted(DOCS_DIR.glob("*.md")):
        head, *sections = re.split(r"(?m)^## ", f.read_text(encoding="utf-8"))
        meta = dict(re.findall(r"(?m)^(source|url):\s*(.+)$", head))
        for s in sections:
            title, _, body = s.partition("\n")
            body = body.strip()
            if not body or "[PARAPHRASE" in body:
                continue
            chunks.append({"id": f"{f.stem}::{title.strip()}", "text": body,
                           "source": meta.get("source", f.stem), "url": meta.get("url", ""),
                           "section": title.strip()})
    return chunks


def build_index() -> int:
    chunks = load_chunks()
    if not chunks:
        raise RuntimeError("No usable sections in docs/. Run extract_docs.py first.")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection("docs")
    except Exception:
        pass
    col = client.create_collection("docs", metadata={"hnsw:space": "cosine"})
    emb = _embedder().encode([c["text"] for c in chunks], normalize_embeddings=True).tolist()
    col.add(ids=[c["id"] for c in chunks], embeddings=emb,
            documents=[c["text"] for c in chunks],
            metadatas=[{"source": c["source"], "url": c["url"], "section": c["section"]}
                       for c in chunks])
    return len(chunks)


def retrieve(query: str) -> list[dict]:
    col = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection("docs")
    q = _embedder().encode([query], normalize_embeddings=True).tolist()
    r = col.query(query_embeddings=q, n_results=TOP_K)
    out = []
    for doc, meta, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0]):
        sim = 1 - dist
        if sim >= MIN_SIM:
            out.append({**meta, "text": doc, "similarity": round(float(sim), 3)})
    return out


def find_city(question: str) -> str | None:
    low = question.lower()
    for c in CITIES:
        if c.lower() in low:
            return c
    for urdu, c in ALIASES.items():
        if urdu in question:
            return c
    return None


def latest_rows(city: str) -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute(f"""
        SELECT row_id, ts_utc, pm25 FROM gold_city_hour
        WHERE city = ? AND pm25 IS NOT NULL
          AND ts_utc > (SELECT MAX(ts_utc) FROM gold_city_hour
                        WHERE city = ? AND pm25 IS NOT NULL) - INTERVAL {WINDOW_H} HOUR
        ORDER BY ts_utc""", [city, city]).df()
    con.close()
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    return df


def validate_numbers(text: str, allowed_sources: list[str]) -> bool:
    """Every number in the answer must appear in a gold-row fact or a cited document."""
    allowed = set(NUM.findall(" ".join(allowed_sources)))
    found = set(NUM.findall(text.replace("PM2.5", "")))
    return found <= allowed


def _idk(question, reason, city=None) -> Insight:
    return Insight(question, False, f"I don't know. {reason}", city)


def answer(question: str) -> Insight:
    city = find_city(question)
    if city is None:
        return _idk(question, "Missing: which city? I cover Lahore, Karachi, Islamabad and Peshawar.")
    rows = latest_rows(city)
    if rows.empty:
        return _idk(question, f"Missing: no recent gold-table readings for {city}.", city)
    age_h = (pd.Timestamp.now(tz="UTC") - rows["ts_utc"].max()).total_seconds() / 3600
    if age_h > MAX_AGE_H:
        return _idk(question, f"Missing: the newest {city} reading is {age_h:.0f} hours old, too stale to advise on.", city)

    mean = float(rows["pm25"].mean())
    band = str(category(pd.Series([mean])).iloc[0])
    hits = retrieve(f"{question} {band} air quality advice")
    if not hits:
        return _idk(question, "Missing: none of my trusted documents covers this question.", city)

    top = hits[0]
    last_pkt = rows["ts_utc"].max().tz_convert("Asia/Karachi")
    facts = {"mean": f"{mean:.1f}", "n": str(len(rows)), "window": str(WINDOW_H),
             "time": last_pkt.strftime("%d %b %H:%M")}
    text = (f"{city}: the average PM2.5 over the last {facts['window']} hours was {facts['mean']} µg/m³ "
            f"({facts['n']} hourly readings, latest {facts['time']} PKT), which is in the '{band}' band. "
            f"Guidance from {top['source']}, section '{top['section']}': {top['text']} {EMERGENCY}")
    allowed = list(facts.values()) + [top["text"], top["source"], top["section"], EMERGENCY]
    if not validate_numbers(text, allowed):
        return _idk(question, "Missing: I could not verify every number in the draft answer, so I withheld it.", city)
    return Insight(question, True, text, city, rows["row_id"].tolist(),
                   [{"source": h["source"], "section": h["section"], "url": h["url"],
                     "similarity": h["similarity"]} for h in hits])


def ask(question: str) -> Insight:
    ins = answer(question)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts_utc": datetime.now(timezone.utc).isoformat(), **ins.__dict__},
                           ensure_ascii=False) + "\n")
    return ins


if __name__ == "__main__":
    if "--build" in sys.argv:
        print(f"indexed {build_index()} document sections")
    else:
        r = ask(" ".join(sys.argv[1:]) or "Is the air in Lahore safe for a morning walk?")
        print(r.text)
        print("answered:", r.answered, "| rows used:", len(r.gold_row_ids))
        print("citations:", r.citations)