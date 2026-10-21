import json
import os
from datetime import datetime, timezone, timedelta

import gradio as gr
import pandas as pd
import requests

STATUS_URL = ("https://huggingface.co/datasets/noor142/"
              "hackforge-status/resolve/main/status.json")
LOCAL_FILE = "data/status.json"          # used when you run it on your own laptop
LATE_AFTER_HOURS = {"OpenAQ": 6, "Open-Meteo": 3}
PKT = timezone(timedelta(hours=5))

CSS = """
.gradio-container {max-width: 1100px !important; margin: auto;}
.hero {padding: 22px 26px; border-radius: 14px; margin-bottom: 14px;
       background: linear-gradient(120deg, #0f2a43 0%, #14532d 100%); color: #f1f5f9;}
.hero h1 {margin: 0 0 6px 0; font-size: 1.7rem; font-weight: 700;}
.hero p {margin: 0; opacity: .85; font-size: .95rem;}
.cards {display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px;}
.card {border: 1px solid rgba(148,163,184,.35); border-radius: 12px; padding: 14px 16px;
       background: rgba(148,163,184,.08);}
.card .label {font-size: .8rem; opacity: .7; margin-bottom: 4px;}
.card .value {font-size: 1.25rem; font-weight: 650;}
.card .sub {font-size: .82rem; opacity: .75; margin-top: 4px;}
.pill {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem;
       font-weight: 650; color: #fff;}
.ok {background: #15803d;} .warn {background: #b45309;} .bad {background: #b91c1c;}
.banner {padding: 12px 16px; border-radius: 10px; margin: 10px 0; font-weight: 550;}
.banner.ok {background: rgba(21,128,61,.18); border: 1px solid #15803d; color: inherit;}
.banner.bad {background: rgba(185,28,28,.18); border: 1px solid #b91c1c; color: inherit;}
.note {font-size: .85rem; opacity: .75;}
"""


def load_status() -> dict:
    if os.path.exists(LOCAL_FILE):
        with open(LOCAL_FILE, encoding="utf-8") as f:
            return json.load(f)
    r = requests.get(STATUS_URL, params={"t": datetime.now().timestamp()}, timeout=20)
    r.raise_for_status()
    return r.json()


def _p(ts):
    if not ts:
        return None
    d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _pkt(ts) -> str:
    d = _p(ts)
    return d.astimezone(PKT).strftime("%d %b %Y, %H:%M PKT") if d else "n/a"


def _pill(text, kind) -> str:
    return f'<span class="pill {kind}">{text}</span>'


def _card(label, value, sub="") -> str:
    return (f'<div class="card"><div class="label">{label}</div>'
            f'<div class="value">{value}</div><div class="sub">{sub}</div></div>')


def render():
    empty = pd.DataFrame()
    try:
        s = load_status()
    except Exception as e:
        err = f'<div class="banner bad">Could not load status: {e}</div>'
        return err, "", empty, empty, empty, empty
    now = datetime.now(timezone.utc)

    ls, lr = s.get("last_success"), s.get("last_run")
    if lr and lr["status"] == "FAILED":
        banner = (f'<div class="banner bad">Latest run FAILED at {_pkt(lr["finished_utc"])}: '
                  f'{lr["message"]}. Showing the last successful run below; data may be getting old.</div>')
    elif lr:
        banner = (f'<div class="banner ok">Pipeline healthy. Latest run succeeded at '
                  f'{_pkt(lr["finished_utc"])}.</div>')
    else:
        banner = '<div class="banner bad">No run recorded yet.</div>'

    # data freshness
    fresh, cards = [], []
    n_late = 0
    for src in s.get("sources", []):
        latest = _p(src["latest_data_utc"])
        age = round((now - latest).total_seconds() / 3600, 1) if latest else None
        limit = LATE_AFTER_HOURS.get(src["source"], 6)
        late = age is None or age > limit
        n_late += int(late)
        fresh.append({"Source": src["source"],
                      "Last fetched": _pkt(src["last_fetch_utc"]),
                      "Newest data point": _pkt(src["latest_data_utc"]),
                      "Age (h)": age, "Limit (h)": limit,
                      "Status": "LATE" if late else "FRESH"})
        cards.append(_card(f'{src["source"]} data',
                           _pill("LATE" if late else "FRESH", "bad" if late else "ok"),
                           f"newest point {age if age is not None else 'n/a'} h old (limit {limit} h)"))

    quality = s.get("quality", [])
    n_warn = sum(1 for q in quality if q["status"] != "OK")
    cards.insert(0, _card("Last successful run",
                          _pkt(ls["finished_utc"]) if ls else "none",
                          f'{ls["gold_rows"]:,} gold rows' if ls else ""))
    cards.append(_card("Data Quality Agent",
                       _pill("ALL CLEAR" if n_warn == 0 else f"{n_warn} WARN",
                             "ok" if n_warn == 0 else "warn"),
                       f"{len(quality)} checks in latest run"))
    cards_html = f'<div class="cards">{"".join(cards)}</div>'
    cards_html += '<p class="note">Times shown in Pakistan time (PKT = UTC + 5). Page refreshes when you press Refresh.</p>'

    qual = pd.DataFrame([{"Check": q["check_name"], "Value": q["value"], "Status": q["status"],
                          "Checked": _pkt(q["checked_utc"])} for q in quality])
    dq = pd.DataFrame([{"Rule": d["rule_id"], "Action": d["action"], "Rows": d["n_rows"],
                        "Why (agent's explanation)": d["reason"]}
                       for d in s.get("quality_decisions", [])])
    runs = pd.DataFrame([{"Run": r["run_id"], "Started": _pkt(r["started_utc"]),
                          "Status": r["status"], "New OpenAQ rows": r["new_openaq_rows"]}
                         for r in s.get("recent_runs", [])])
    return banner, cards_html, pd.DataFrame(fresh), qual, dq, runs


ABOUT = """
### Who this is for
An environment officer deciding where and when to issue smog advisories in
**Lahore, Karachi, Islamabad and Peshawar**.

### What the pipeline does every hour
1. Fetches the newest PM2.5 readings (OpenAQ) and weather (Open-Meteo).
2. Rebuilds raw, cleaned and processed layers, with SHA-256 hashes in `manifest.csv`.
3. The Data Quality Agent runs its rules and logs each decision as fix, fill or flag.
4. This page reads the latest status. Nothing here is typed in by hand.

### Honest limits
- Air quality monitors are few and not evenly spread; some areas have no sensor.
- Long gaps are left missing and flagged, never guessed.
- Outputs are information, not professional advice. In an emergency or if someone has
  trouble breathing, contact **Rescue 1122** or the nearest hospital.

Data: OpenAQ (licence depends on each provider) and Open-Meteo (CC BY 4.0).
"""

with gr.Blocks(title="HackInsight: Pakistan Air Quality Pipeline", css=CSS,
               theme=gr.themes.Soft(primary_hue="emerald", neutral_hue="slate")) as demo:
    gr.HTML('<div class="hero"><h1>Pakistan Air Quality: Data Pipeline Status</h1>'
            '<p>Live health of the hourly pipeline behind the PM2.5 forecast. '
            'HackInsight 2026, Air Quality track.</p></div>')
    banner = gr.HTML()
    cards = gr.HTML()
    btn = gr.Button("Refresh now", variant="primary")
    with gr.Tabs():
        with gr.Tab("Data freshness"):
            fresh_t = gr.Dataframe(interactive=False, wrap=True)
        with gr.Tab("Quality status"):
            qual_t = gr.Dataframe(interactive=False, wrap=True)
        with gr.Tab("Agent decisions"):
            gr.Markdown("Every fix, fill or flag the Data Quality Agent made in the latest run, "
                        "with the rule and the number of rows affected.")
            dq_t = gr.Dataframe(interactive=False, wrap=True)
        with gr.Tab("Run history"):
            runs_t = gr.Dataframe(interactive=False, wrap=True)
        with gr.Tab("About and limits"):
            gr.Markdown(ABOUT)
    outs = [banner, cards, fresh_t, qual_t, dq_t, runs_t]
    demo.load(render, None, outs)
    btn.click(render, None, outs)

if __name__ == "__main__":
    demo.launch()