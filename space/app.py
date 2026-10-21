
import os
import json
from datetime import datetime, timezone, timedelta
from html import escape

import gradio as gr
import pandas as pd
import requests
import chromadb
from groq import Groq
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STATUS_URL = (
    "https://huggingface.co/datasets/noor142/"
    "hackforge-status/resolve/main/status.json"
)

LOCAL_FILE = os.path.join(BASE_DIR, "data", "status.json")
CHROMA_PATH = os.path.join(BASE_DIR, "data", "chroma")

GROQ_MODEL = "openai/gpt-oss-20b"
CHROMA_COLLECTION = "hackforge_knowledge"

LATE_AFTER_HOURS = {
    "OpenAQ": 6,
    "Open-Meteo": 3,
}

PKT = timezone(timedelta(hours=5))


# ============================================================
# STYLING
# ============================================================

CSS = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto;
}

.hero {
    padding: 22px 26px;
    border-radius: 14px;
    margin-bottom: 14px;
    background: linear-gradient(120deg, #0f2a43 0%, #14532d 100%);
    color: #f1f5f9;
}

.hero h1 {
    margin: 0 0 6px 0;
    font-size: 1.7rem;
    font-weight: 700;
}

.hero p {
    margin: 0;
    opacity: .85;
    font-size: .95rem;
}

.cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 12px;
}

.card {
    border: 1px solid rgba(148,163,184,.35);
    border-radius: 12px;
    padding: 14px 16px;
    background: rgba(148,163,184,.08);
}

.card .label {
    font-size: .8rem;
    opacity: .7;
    margin-bottom: 4px;
}

.card .value {
    font-size: 1.25rem;
    font-weight: 650;
}

.card .sub {
    font-size: .82rem;
    opacity: .75;
    margin-top: 4px;
}

.pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: .78rem;
    font-weight: 650;
    color: #fff;
}

.ok {
    background: #15803d;
}

.warn {
    background: #b45309;
}

.bad {
    background: #b91c1c;
}

.banner {
    padding: 12px 16px;
    border-radius: 10px;
    margin: 10px 0;
    font-weight: 550;
}

.banner.ok {
    background: rgba(21,128,61,.18);
    border: 1px solid #15803d;
    color: inherit;
}

.banner.bad {
    background: rgba(185,28,28,.18);
    border: 1px solid #b91c1c;
    color: inherit;
}

.note {
    font-size: .85rem;
    opacity: .75;
}

.agent-intro {
    padding: 16px;
    border-radius: 12px;
    background: rgba(148,163,184,.08);
    border: 1px solid rgba(148,163,184,.3);
    margin-bottom: 12px;
}
"""


# ============================================================
# STATUS PIPELINE FUNCTIONS
# ============================================================

def load_status() -> dict:
    """Load local status JSON, or fetch the published status."""

    if os.path.exists(LOCAL_FILE):
        with open(LOCAL_FILE, encoding="utf-8") as file:
            return json.load(file)

    response = requests.get(
        STATUS_URL,
        params={"t": datetime.now().timestamp()},
        timeout=20,
    )
    response.raise_for_status()

    return response.json()


def parse_timestamp(ts):
    if not ts:
        return None

    value = datetime.fromisoformat(ts.replace("Z", "+00:00"))

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)

    return value


def format_pkt(ts) -> str:
    value = parse_timestamp(ts)

    if value is None:
        return "n/a"

    return value.astimezone(PKT).strftime(
        "%d %b %Y, %H:%M PKT"
    )


def make_pill(label, kind) -> str:
    return (
        f'<span class="pill {escape(str(kind))}">'
        f'{escape(str(label))}</span>'
    )


def make_card(label, value, subtitle="") -> str:
    return (
        '<div class="card">'
        f'<div class="label">{escape(str(label))}</div>'
        f'<div class="value">{value}</div>'
        f'<div class="sub">{escape(str(subtitle))}</div>'
        '</div>'
    )


def render_dashboard():
    """Render pipeline health, data freshness, quality and run history."""

    empty = pd.DataFrame()

    try:
        status = load_status()

    except Exception as exc:
        error_html = (
            '<div class="banner bad">'
            'Could not load pipeline status: '
            f'{escape(str(exc))}'
            '</div>'
        )

        return (
            error_html,
            "",
            empty,
            empty,
            empty,
            empty,
        )

    now = datetime.now(timezone.utc)

    last_success = status.get("last_success")
    last_run = status.get("last_run")

    # Latest pipeline status
    if last_run and last_run.get("status") == "FAILED":
        banner = (
            '<div class="banner bad">'
            f'Latest run FAILED at '
            f'{escape(format_pkt(last_run.get("finished_utc")))}: '
            f'{escape(str(last_run.get("message", "No error message")))}. '
            'Showing the last successful run when available; '
            'data may be getting old.'
            '</div>'
        )

    elif last_run:
        banner = (
            '<div class="banner ok">'
            'Latest pipeline run succeeded at '
            f'{escape(format_pkt(last_run.get("finished_utc")))}.'
            '</div>'
        )

    else:
        banner = (
            '<div class="banner bad">'
            'No pipeline run recorded yet.'
            '</div>'
        )

    # Data freshness
    freshness_rows = []
    cards = []
    late_count = 0

    for source in status.get("sources", []):
        latest = parse_timestamp(source.get("latest_data_utc"))

        age = (
            round((now - latest).total_seconds() / 3600, 1)
            if latest
            else None
        )

        source_name = source.get("source", "Unknown")
        limit = LATE_AFTER_HOURS.get(source_name, 6)

        is_late = age is None or age > limit
        late_count += int(is_late)

        freshness_rows.append({
            "Source": source_name,
            "Last fetched": format_pkt(source.get("last_fetch_utc")),
            "Newest data point": format_pkt(
                source.get("latest_data_utc")
            ),
            "Age (h)": age,
            "Limit (h)": limit,
            "Status": "LATE" if is_late else "FRESH",
        })

        cards.append(
            make_card(
                f"{source_name} data",
                make_pill(
                    "LATE" if is_late else "FRESH",
                    "bad" if is_late else "ok",
                ),
                f"Newest point {age if age is not None else 'n/a'} "
                f"hours old (limit {limit} hours)",
            )
        )

    # Last successful run card
    cards.insert(
        0,
        make_card(
            "Last successful run",
            (
                escape(format_pkt(last_success.get("finished_utc")))
                if last_success
                else "None"
            ),
            (
                f'{int(last_success.get("gold_rows", 0)):,} gold rows'
                if last_success
                else "No successful run recorded"
            ),
        ),
    )

    # Data quality
    quality = status.get("quality", [])

    warning_count = sum(
        1 for item in quality
        if item.get("status") != "OK"
    )

    cards.append(
        make_card(
            "Data Quality Agent",
            make_pill(
                "ALL CLEAR" if warning_count == 0
                else f"{warning_count} WARN",
                "ok" if warning_count == 0 else "warn",
            ),
            f"{len(quality)} checks in the latest run",
        )
    )

    cards_html = (
        f'<div class="cards">{"".join(cards)}</div>'
        '<p class="note">'
        'Times shown in Pakistan Standard Time (PKT = UTC + 5). '
        'Click Refresh now to reload the status.'
        '</p>'
    )

    # Quality checks table
    quality_df = pd.DataFrame([
        {
            "Check": item.get("check_name", ""),
            "Value": item.get("value", ""),
            "Status": item.get("status", ""),
            "Checked": format_pkt(item.get("checked_utc")),
        }
        for item in quality
    ])

    # Agent decisions table
    decisions_df = pd.DataFrame([
        {
            "Rule": item.get("rule_id", ""),
            "Action": item.get("action", ""),
            "Rows": item.get("n_rows", 0),
            "Why (agent's explanation)": item.get("reason", ""),
        }
        for item in status.get("quality_decisions", [])
    ])

    # Run history table
    runs_df = pd.DataFrame([
        {
            "Run": item.get("run_id", ""),
            "Started": format_pkt(item.get("started_utc")),
            "Status": item.get("status", ""),
            "New OpenAQ rows": item.get("new_openaq_rows", 0),
        }
        for item in status.get("recent_runs", [])
    ])

    freshness_df = pd.DataFrame(freshness_rows)

    return (
        banner,
        cards_html,
        freshness_df,
        quality_df,
        decisions_df,
        runs_df,
    )


# ============================================================
# RAG AI INSIGHT AGENT
# ============================================================

def ask_agent(user_query):
    """Retrieve relevant documents from ChromaDB and ask Groq."""

    if not user_query or not user_query.strip():
        return "Please enter a question about your air quality data."

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return (
            "**Configuration error:** `GROQ_API_KEY` was not found. "
            "Add it to your `.env` file or set it in your terminal."
        )

    if not os.path.isdir(CHROMA_PATH):
        return (
            "**Knowledge base not found.** Expected the ChromaDB folder at "
            f"`{CHROMA_PATH}`. Make sure your database has been created."
        )

    try:
        # Connect to the persistent vector database
        chroma_client = chromadb.PersistentClient(
            path=CHROMA_PATH
        )

        collection = chroma_client.get_collection(
            name=CHROMA_COLLECTION
        )

        if collection.count() == 0:
            return "The knowledge base exists, but it contains no documents."

        # Retrieve the two most relevant documents
        results = collection.query(
            query_texts=[user_query.strip()],
            n_results=min(2, collection.count()),
        )

        documents = results.get("documents", [[]])[0] or []
        metadatas = results.get("metadatas", [[]])[0] or []

        if not documents:
            return (
                "No relevant information was found in the knowledge base."
            )

        # Build context with source names
        context_parts = []

        for index, document in enumerate(documents):
            metadata = (
                metadatas[index]
                if index < len(metadatas) and metadatas[index]
                else {}
            )

            source = metadata.get("source", "unknown")

            context_parts.append(
                f"Document content:\n{document}\n"
                f"Source: {source}"
            )

        context = "\n\n---\n\n".join(context_parts)

        # Ask the Groq-hosted language model
        groq_client = Groq(api_key=api_key)

        prompt = f"""
You are the hackForge AI Insight Agent for a Pakistan air quality
data pipeline.

Answer the user's question using only the retrieved context below.

Rules:
1. Do not invent measurements, statistics, or sources.
2. If the context does not answer the question, say so clearly.
3. Cite the source names supplied in the retrieved context.
4. Explain technical information in simple, clear language.
5. Do not claim to have checked live data unless it appears in
   the retrieved context.
6. Clearly distinguish retrieved facts from any explanation.

Retrieved context:
{context}

User question:
{user_query.strip()}
"""

        completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=GROQ_MODEL,
            temperature=0.2,
        )

        answer = completion.choices[0].message.content

        return answer or "The AI agent returned an empty response."

    except Exception as exc:
        return (
            "The AI Insight Agent encountered an error:\n\n"
            f"`{str(exc)}`"
        )


# ============================================================
# ABOUT SECTION
# ============================================================

ABOUT = """
### Who this is for

An environment officer deciding where and when to issue smog
advisories in **Lahore, Karachi, Islamabad and Peshawar**.

### What the pipeline does every hour

1. Fetches the newest PM2.5 readings from OpenAQ and weather data
   from Open-Meteo.
2. Rebuilds raw, cleaned and processed layers, with SHA-256 hashes
   recorded in `manifest.csv`.
3. Runs Data Quality Agent rules and logs each decision as fix,
   fill or flag.
4. Publishes pipeline status for this application to display.
5. Provides a RAG-based AI Insight Agent that retrieves relevant
   knowledge from ChromaDB and generates answers through Groq.

### Honest limitations

- Air quality monitors are limited and unevenly distributed.
- Some locations may have no sensor coverage.
- Long data gaps are left missing and flagged, not guessed.
- The AI agent is limited by the documents available in its
  knowledge base and may not have the latest readings.
- Pipeline status and the AI knowledge base are separate:
  the AI agent does not automatically query the latest status JSON.
- Outputs are informational, not professional medical advice.
  In an emergency or if someone has trouble breathing, contact
  Rescue 1122 or the nearest hospital.

### Data sources

OpenAQ (licence depends on each provider) and Open-Meteo
(CC BY 4.0).
"""


# ============================================================
# COMPLETE GRADIO APPLICATION
# ============================================================

with gr.Blocks(
    title="HackInsight: Pakistan Air Quality Pipeline",
    css=CSS,
    theme=gr.themes.Soft(
        primary_hue="emerald",
        neutral_hue="slate",
    ),
) as demo:

    # Header
    gr.HTML(
        """
        <div class="hero">
            <h1>Pakistan Air Quality: Data Pipeline Status</h1>
            <p>
                Live health of the hourly pipeline behind the PM2.5
                forecast. HackInsight 2026, Air Quality track.
            </p>
        </div>
        """
    )

    # Dashboard status and summary cards
    status_banner = gr.HTML()
    summary_cards = gr.HTML()

    refresh_btn = gr.Button(
        "Refresh now",
        variant="primary",
    )

    with gr.Tabs():

        # ----------------------------------------------------
        # DATA FRESHNESS
        # ----------------------------------------------------
        with gr.Tab("Data freshness"):
            gr.Markdown(
                "Check when each source was last fetched and how "
                "old its newest available data point is."
            )

            freshness_table = gr.Dataframe(
                label="Source freshness",
                interactive=False,
                wrap=True,
            )

        # ----------------------------------------------------
        # QUALITY STATUS
        # ----------------------------------------------------
        with gr.Tab("Quality status"):
            gr.Markdown(
                "Results of the latest Data Quality Agent checks."
            )

            quality_table = gr.Dataframe(
                label="Data quality checks",
                interactive=False,
                wrap=True,
            )

        # ----------------------------------------------------
        # AGENT DECISIONS
        # ----------------------------------------------------
        with gr.Tab("Agent decisions"):
            gr.Markdown(
                "Every fix, fill or flag made by the Data Quality "
                "Agent, including the rule, affected rows and reason."
            )

            decisions_table = gr.Dataframe(
                label="Quality agent decisions",
                interactive=False,
                wrap=True,
            )

        # ----------------------------------------------------
        # RUN HISTORY
        # ----------------------------------------------------
        with gr.Tab("Run history"):
            gr.Markdown(
                "Recent recorded pipeline runs and their status."
            )

            runs_table = gr.Dataframe(
                label="Recent runs",
                interactive=False,
                wrap=True,
            )

        # ----------------------------------------------------
        # AI INSIGHT AGENT
        # ----------------------------------------------------
        with gr.Tab("AI Insight Agent"):

            gr.HTML(
                """
                <div class="agent-intro">
                    <h3>hackForge AI Insight Agent</h3>
                    <p>
                        Ask questions about air quality, PM2.5,
                        data quality and your knowledge base.
                        ChromaDB retrieves relevant documents, and
                        Groq generates an answer with source citations.
                    </p>
                </div>
                """
            )

            query_box = gr.Textbox(
                label="Ask a question about your air quality data",
                placeholder=(
                    "Example: What factors affect PM2.5 levels?"
                ),
                lines=3,
            )

            with gr.Row():
                ask_btn = gr.Button(
                    "Generate Insight",
                    variant="primary",
                )

                clear_btn = gr.Button("Clear")

            answer_box = gr.Markdown(
                label="AI Response",
            )

            ask_btn.click(
                fn=ask_agent,
                inputs=query_box,
                outputs=answer_box,
            )

            query_box.submit(
                fn=ask_agent,
                inputs=query_box,
                outputs=answer_box,
            )

            clear_btn.click(
                fn=lambda: ("", ""),
                inputs=None,
                outputs=[query_box, answer_box],
            )

        # ----------------------------------------------------
        # ABOUT
        # ----------------------------------------------------
        with gr.Tab("About and limits"):
            gr.Markdown(ABOUT)

    # Refresh all dashboard components together
    dashboard_outputs = [
        status_banner,
        summary_cards,
        freshness_table,
        quality_table,
        decisions_table,
        runs_table,
    ]

    demo.load(
        fn=render_dashboard,
        inputs=None,
        outputs=dashboard_outputs,
    )

    refresh_btn.click(
        fn=render_dashboard,
        inputs=None,
        outputs=dashboard_outputs,
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    demo.launch()
