import json
import os
from datetime import datetime, timezone

import gradio as gr
import pandas as pd
import requests

STATUS_URL = ("https://huggingface.co/datasets/noor142/"
              "hackforge-status/resolve/main/status.json")
LOCAL_FILE = "data/status.json"          # used when you run it on your own laptop
LATE_AFTER_HOURS = {"OpenAQ": 6, "Open-Meteo": 3}


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


def render():
    try:
        s = load_status()
    except Exception as e:
        return f"Could not load status: {e}", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    now = datetime.now(timezone.utc)

    ls, lr = s.get("last_success"), s.get("last_run")
    md = "### Last successful run\n"
    md += (f"**{ls['finished_utc']}** (UTC), gold rows: {ls['gold_rows']:,}\n\n"
           if ls else "No successful run yet.\n\n")
    if lr:
        md += f"Latest run status: **{lr['status']}** at {lr['finished_utc']}"
        if lr["status"] == "FAILED":
            md += f" ({lr['message']})"
    md += "\n\n_All times are UTC. Pakistan time (PKT) = UTC + 5 hours._"

    fresh = []
    for src in s.get("sources", []):
        latest = _p(src["latest_data_utc"])
        age = round((now - latest).total_seconds() / 3600, 1) if latest else None
        limit = LATE_AFTER_HOURS.get(src["source"], 6)
        fresh.append({"Source": src["source"], "Last fetched (UTC)": src["last_fetch_utc"],
                      "Newest data point (UTC)": src["latest_data_utc"],
                      "Data age (hours)": age,
                      "Status": "LATE" if age is None or age > limit else "FRESH"})

    qual = pd.DataFrame([{"Check": q["check_name"], "Value": q["value"],
                          "Status": q["status"], "Checked (UTC)": q["checked_utc"]}
                         for q in s.get("quality", [])])
    runs = pd.DataFrame([{"Run": r["run_id"], "Started (UTC)": r["started_utc"],
                          "Status": r["status"], "New rows": r["new_openaq_rows"]}
                         for r in s.get("recent_runs", [])])
    dq = pd.DataFrame([{"Rule": d["rule_id"], "Action": d["action"], "Rows": d["n_rows"],
                        "Why": d["reason"]} for d in s.get("quality_decisions", [])])
    return md, pd.DataFrame(fresh), qual, dq, runs

with gr.Blocks(title="HackInsight: Pakistan Air Quality Pipeline") as demo:
    gr.Markdown("# HackInsight: Pakistan Air Quality Pipeline")
    gr.Markdown("Data: OpenAQ and Open-Meteo (CC BY 4.0). Information only, not professional advice.")
    btn = gr.Button("Refresh")
    run_md = gr.Markdown()
    gr.Markdown("### Data freshness")
    fresh_t = gr.Dataframe()
    gr.Markdown("### Quality status (Data Quality Agent)")
    qual_t = gr.Dataframe()
    gr.Markdown("### Data Quality Agent decisions (latest run)")
    dq_t = gr.Dataframe()
    gr.Markdown("### Recent runs")
    runs_t = gr.Dataframe()
    outs = [run_md, fresh_t, qual_t, dq_t, runs_t]
    demo.load(render, None, outs)
    btn.click(render, None, outs)

if __name__ == "__main__":
    demo.launch()