"""100 GSM8K example walkthroughs — 50 RIGHT + 50 WRONG.

For each example: question, gold chain, per-step force-decode emit, parsed
value, and which marker / gold-final value matches at each step.

The PDF has:
  - Cover page
  - Table of contents listing the RIGHT section and WRONG section with their
    page ranges
  - 50 RIGHT examples (one per page)
  - 50 WRONG examples (one per page)

After matplotlib renders the PDF, we post-process with pypdf to add an
OUTLINE (bookmarks): TOC → RIGHT section → WRONG section + individual
example bookmarks. Most PDF viewers render these as a clickable sidebar.

Output: examples_100_slideshow_gsm8k.pdf
"""
from __future__ import annotations

import json, re
import random
from pathlib import Path

import matplotlib
matplotlib.rcParams["text.parse_math"] = False   # don't interpret $...$ as math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from pypdf import PdfReader, PdfWriter

PD = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
FD_JSON = REPO / "experiments" / "computation_probes" / "force_decode_per_step_gsm8k.json"
OUT_PDF = PD / "examples_100_slideshow_gsm8k.pdf"

N_RIGHT = 50
N_WRONG = 50
SEED = 0


def parse_markers(s):
    s = s.replace(",", "")
    return re.findall(r"<<(-?\d+\.?\d*)\s*([+\-*/])\s*(-?\d+\.?\d*)\s*=\s*(-?\d+\.?\d*)>>", s)


def emit_final(s):
    s = s.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums: return None
    try: return float(nums[-1])
    except: return None


def trunc(s, n):
    return s if len(s) <= n else s[:n - 1] + "…"


def render_example_page(pdf, ex, kind, ordinal, page_no):
    """Render one example as a single page. Returns the page number used."""
    fig, ax = plt.subplots(figsize=(11.5, 8.5))
    ax.axis("off")
    tag = "✓ CORRECT" if kind == "right" else "✗ WRONG"
    body = f"{kind.upper()} #{ordinal} / 50    [{tag}]   GSM8K idx {ex['idx']}    page {page_no}\n\n"
    body += "Q: " + trunc(ex["q"], 380) + "\n\n"
    body += f"GOLD chain ({ex['n_markers']} markers, gold final = {ex['gold']}):\n"
    for i, m in enumerate(ex["markers"], 1):
        body += f"  m{i}: <<{m['a']:g} {m['op']} {m['b']:g} = {m['c']:g}>>\n"
    body += "\nForce-decode per step:\n"
    body += f"  {'step':<5} {'parsed':<12} {'match':<18} {'emit (truncated)':<55}\n"
    for k in range(len(ex["step_emits"])):
        em = trunc(ex["step_emits"][k], 50)
        pv = ex["emit_vals"][k]
        match = ""
        if pv is not None:
            if abs(pv - ex["gold"]) < 1e-3: match = "= GOLD"
            else:
                for i, m in enumerate(ex["markers"], 1):
                    if abs(pv - m["c"]) < 1e-3:
                        match = f"= marker {i}"; break
                if not match: match = "(neither)"
        body += (f"  {k+1:<5} {(str(pv) if pv is not None else '?'):<12} "
                 f"{match:<18} {em!r}\n")
    ax.text(0.01, 0.99, body, va="top", ha="left", family="monospace", fontsize=8.5)
    ax.set_title(f"{kind.upper()} example #{ordinal}/50  (GSM8K idx={ex['idx']})",
                 fontsize=11, fontweight="bold",
                 color="#2ca02c" if kind == "right" else "#d62728")
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def main():
    rng = random.Random(SEED)
    print("loading force_decode + GSM8K", flush=True)
    fd = json.load(open(FD_JSON))
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main")["test"]
    rows = fd["rows"]
    n_steps = len(rows[0]["step_emits"])

    # Prep all problems with their final-step correctness
    prepped = []
    for r in rows:
        idx = r["idx"]
        ex = ds[idx]
        gold_match = re.search(r"####\s*(-?\d+\.?\d*)", ex["answer"].replace(",", ""))
        if gold_match is None: continue
        gold = float(gold_match.group(1))
        markers = parse_markers(ex["answer"])
        emit_vals = [emit_final(e) for e in r["step_emits"]]
        is_correct = (emit_vals[-1] is not None and abs(emit_vals[-1] - gold) < 1e-3)
        prepped.append({
            "idx": idx,
            "q": ex["question"].strip().replace("  ", " "),
            "gold": gold,
            "markers": [{"a": float(a), "op": op, "b": float(b), "c": float(c)}
                        for a, op, b, c in markers],
            "n_markers": len(markers),
            "step_emits": r["step_emits"],
            "emit_vals": emit_vals,
            "is_correct": is_correct,
        })

    right = [p for p in prepped if p["is_correct"]]
    wrong = [p for p in prepped if not p["is_correct"]]
    print(f"  total parseable: {len(prepped)}  right: {len(right)}  wrong: {len(wrong)}")
    # Sample 50 of each (mixed chain lengths to be representative)
    rng.shuffle(right); rng.shuffle(wrong)
    sample_right = right[:N_RIGHT]; sample_wrong = wrong[:N_WRONG]

    # Track page numbers for the outline. matplotlib pages are 0-indexed in
    # pypdf after we write — page 0 is cover, page 1 is TOC, page 2 onward
    # are examples.
    cover_page = 0
    toc_page = 1
    right_start_page = 2
    right_end_page = right_start_page + len(sample_right) - 1
    wrong_start_page = right_end_page + 1
    wrong_end_page = wrong_start_page + len(sample_wrong) - 1
    total_pages = wrong_end_page + 1

    print(f"  pages: cover={cover_page+1}, TOC={toc_page+1}, "
          f"RIGHT={right_start_page+1}-{right_end_page+1}, "
          f"WRONG={wrong_start_page+1}-{wrong_end_page+1}, total={total_pages}")

    with PdfPages(OUT_PDF) as pdf:
        # ----- Cover (page 1) -----
        fig, ax = plt.subplots(figsize=(11.5, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.70, "100 GSM8K Example Walkthroughs",
                ha="center", va="center", fontsize=24, fontweight="bold",
                transform=ax.transAxes)
        ax.text(0.5, 0.55, f"{N_RIGHT} CORRECT + {N_WRONG} WRONG",
                ha="center", va="center", fontsize=16, transform=ax.transAxes,
                color="#444")
        ax.text(0.5, 0.40,
                f"Source: force_decode_per_step_gsm8k.json on GSM8K test\n"
                f"Sample seed: {SEED}",
                ha="center", va="center", fontsize=11, transform=ax.transAxes,
                family="monospace", color="#666")
        ax.text(0.5, 0.12,
                "Each example shows: question, gold marker chain, "
                "per-step force-decode emit,\n"
                "parsed final number, and which marker/gold-value matches at each step.",
                ha="center", va="center", fontsize=10, transform=ax.transAxes,
                family="monospace", color="#666")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ----- TOC (page 2) -----
        fig, ax = plt.subplots(figsize=(11.5, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.95, "Table of Contents", ha="center", fontsize=20,
                fontweight="bold", transform=ax.transAxes)
        toc_text = (f"\n"
                    f"  Cover ......................................... page {cover_page+1}\n"
                    f"  Table of Contents ............................. page {toc_page+1}\n\n"
                    f"  RIGHT (CORRECT) examples — {N_RIGHT} total ........... pages {right_start_page+1}–{right_end_page+1}\n")
        for i in range(0, N_RIGHT, 5):
            chunk = sample_right[i:i+5]
            line = "    " + "  ".join(
                f"#{i+j+1} (idx={chunk[j]['idx']}, gold={chunk[j]['gold']:g}) → p.{right_start_page + i + j + 1}"
                for j in range(len(chunk)))
            toc_text += line + "\n"
        toc_text += f"\n  WRONG examples — {N_WRONG} total ..................... pages {wrong_start_page+1}–{wrong_end_page+1}\n"
        for i in range(0, N_WRONG, 5):
            chunk = sample_wrong[i:i+5]
            line = "    " + "  ".join(
                f"#{i+j+1} (idx={chunk[j]['idx']}, gold={chunk[j]['gold']:g}) → p.{wrong_start_page + i + j + 1}"
                for j in range(len(chunk)))
            toc_text += line + "\n"
        toc_text += ("\n  Bookmarks (outline) are also embedded — use your PDF viewer's\n"
                     "  sidebar/outline panel to jump directly to the RIGHT or WRONG section.\n")
        ax.text(0.02, 0.88, toc_text, va="top", ha="left",
                family="monospace", fontsize=8, transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ----- RIGHT examples (pages right_start_page+1 ..) -----
        for i, ex in enumerate(sample_right):
            page_no = right_start_page + i + 1
            render_example_page(pdf, ex, "right", i + 1, page_no)

        # ----- WRONG examples -----
        for i, ex in enumerate(sample_wrong):
            page_no = wrong_start_page + i + 1
            render_example_page(pdf, ex, "wrong", i + 1, page_no)

    print(f"  base matplotlib PDF written to {OUT_PDF}")

    # ----- Post-process: add outline (bookmarks) via pypdf -----
    reader = PdfReader(OUT_PDF)
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)

    # Top-level bookmarks
    cover_bm   = writer.add_outline_item("Cover",  cover_page)
    toc_bm     = writer.add_outline_item("Table of Contents", toc_page)
    right_bm   = writer.add_outline_item("RIGHT (CORRECT) — 50 examples", right_start_page)
    wrong_bm   = writer.add_outline_item("WRONG — 50 examples", wrong_start_page)

    # Per-example bookmarks under each section
    for i, ex in enumerate(sample_right):
        writer.add_outline_item(
            f"RIGHT #{i+1} (idx={ex['idx']}, gold={ex['gold']:g})",
            right_start_page + i, parent=right_bm)
    for i, ex in enumerate(sample_wrong):
        writer.add_outline_item(
            f"WRONG #{i+1} (idx={ex['idx']}, gold={ex['gold']:g})",
            wrong_start_page + i, parent=wrong_bm)

    with open(OUT_PDF, "wb") as f:
        writer.write(f)

    sz_mb = OUT_PDF.stat().st_size / 1e6
    print(f"\nsaved {OUT_PDF}  ({sz_mb:.1f} MB, {len(writer.pages)} pages, outline added)")


if __name__ == "__main__":
    main()
