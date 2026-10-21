"""Converts the official PDFs in pdfs/ into text sections in docs/ for the RAG agent.
The text comes from the PDFs by code, nothing is typed by hand."""
import re
from pathlib import Path

import pdfplumber

PDF_DIR = Path("pdfs")
DOCS = Path("docs")
MAX_CHARS = 900

# Known files. The heatwave plan is about heat, so only the two pages that discuss
# air pollution are kept: PDF page 8 right half (section 1.5) and
# PDF page 11 left half (section 2.3). Pages are double-page spreads.
KNOWN = {
    "national_environmental_quality_standards_for_ambient_air_dr.pdf": dict(
        out="pak_neqs",
        source="Pakistan Environmental Protection Agency (Pak-EPA), draft National "
               "Environmental Quality Standards for Ambient Air",
        pages=None),
    "Heatwave_compressed.pdf": dict(
        out="punjab_heatwave_plan",
        source="Punjab Heatwave Action Plan 2026, Environment Protection and Climate "
               "Change Department, Government of the Punjab",
        pages={8: ["right"], 11: ["left"]}),
}


def page_text(page, halves):
    if not halves:
        return [("", page.extract_text() or "")]
    w, h = page.width, page.height
    boxes = {"left": (0, 0, w / 2, h), "right": (w / 2, 0, w, h)}
    return [(f" {side} half", page.crop(boxes[side]).extract_text() or "") for side in halves]


def chunks(text):
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) > MAX_CHARS and cur:
            out.append(cur.strip())
            cur = ""
        cur += s + " "
    if cur.strip():
        out.append(cur.strip())
    return [c for c in out if len(c) >= 150]


def convert(pdf_path):
    cfg = KNOWN.get(pdf_path.name, dict(out=pdf_path.stem.lower().replace(" ", "_"),
                                        source=pdf_path.stem, pages=None))
    lines = [f"source: {cfg['source']}", f"url: file {pdf_path.name}", ""]
    n = 0
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            if cfg["pages"] and i not in cfg["pages"]:
                continue
            for label, txt in page_text(page, (cfg["pages"] or {}).get(i)):
                for k, c in enumerate(chunks(txt), start=1):
                    n += 1
                    lines += [f"## page {i}{label}, part {k}", c, ""]
    if n == 0:
        print(f"WARNING: no text found in {pdf_path.name} (scanned PDF?)")
        return
    DOCS.mkdir(exist_ok=True)
    (DOCS / f"{cfg['out']}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"{pdf_path.name}: {n} sections -> docs/{cfg['out']}.md")


if __name__ == "__main__":
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print("Put the PDF files in the pdfs folder first.")
    for p in pdfs:
        convert(p)