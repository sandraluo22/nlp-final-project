"""Per-example walkthrough of CODI's 6 latent steps for a few GSM8K problems.

For a curated set of problems (some correct, some wrong):
  - Show the question + gold marker chain + gold final answer
  - For each latent step k = 1..6:
      * force-decode emit (from force_decode_per_step_gsm8k.json)
      * codi_extract'd final answer
      * for that problem, the logit-lens top-3 tokens at (step k, last layer
        L11, resid_post) — what the model would emit at that point if we
        cut the loop here.
  - Mark which gold marker values appear in the emit, and which step first
    surfaces each one.

This grounds the averaged numbers in concrete cases.

Output: per_example_walkthrough_gsm8k.{json,pdf}
"""
from __future__ import annotations

import json, re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

PD = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
FD_JSON = REPO / "experiments" / "computation_probes" / "force_decode_per_step_gsm8k.json"
LL_JSON = REPO / "experiments" / "computation_probes" / "logit_lens_gsm8k.json"
OUT_JSON = PD / "per_example_walkthrough_gsm8k.json"
OUT_PDF = PD / "per_example_walkthrough_gsm8k.pdf"


def parse_markers(s):
    s = s.replace(",", "")
    return re.findall(r"<<(-?\d+\.?\d*)\s*([+\-*/])\s*(-?\d+\.?\d*)\s*=\s*(-?\d+\.?\d*)>>", s)


def emit_final(s):
    s = s.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums: return None
    try: return float(nums[-1])
    except: return None


def main():
    fd = json.load(open(FD_JSON))
    ll = json.load(open(LL_JSON))
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main")["test"]

    # Find sample problems by matching the logit_lens samples' question text
    # to GSM8K test indices.
    samples = ll.get("sample_top5", [])
    print(f"  logit_lens has {len(samples)} sample problems")
    test_q_to_idx = {}
    for i, ex in enumerate(ds):
        q = ex["question"].strip().replace("  ", " ")
        test_q_to_idx[q] = i

    n_layers = ll["N_LAYERS"]; n_lat = ll["N_LAT"]
    SUBL = ll["SUBLAYERS"]
    last_L_key = lambda s, sl: f"step{s+1}_L{n_layers-1}_{sl}"

    # Match force_decode rows by GSM8K idx
    fd_rows = {r["idx"]: r for r in fd["rows"]}

    selected = []
    for s in samples:
        q = s["q"]
        idx = test_q_to_idx.get(q)
        if idx is None: continue
        ex = ds[idx]
        gold = float(re.search(r"####\s*(-?\d+\.?\d*)", ex["answer"].replace(",", "")).group(1))
        markers = parse_markers(ex["answer"])
        fdr = fd_rows.get(idx)
        if fdr is None: continue
        step_emits = fdr["step_emits"]
        emit_vals = [emit_final(e) for e in step_emits]
        step_correct = [v is not None and abs(v - gold) < 1e-3 for v in emit_vals]
        finally_correct = step_correct[-1] if step_correct else False
        ll_top_per_step = []
        for k in range(n_lat):
            cell = s["cells"].get(last_L_key(k, "resid_post"), {})
            top = list(zip(cell.get("top_tokens", []), cell.get("top_probs", [])))
            ll_top_per_step.append(top[:3])
        selected.append({
            "idx": idx, "q": q, "gold": gold,
            "markers": [{"a": float(a), "op": op, "b": float(b), "c": float(c)}
                        for a, op, b, c in markers],
            "n_markers": len(markers),
            "finally_correct": bool(finally_correct),
            "step_emits": step_emits,
            "emit_vals": emit_vals,
            "step_correct": [bool(b) for b in step_correct],
            "ll_resid_post_L11_top3": ll_top_per_step,
        })

    OUT_JSON.write_text(json.dumps(selected, indent=2))
    print(f"  saved {OUT_JSON} with {len(selected)} problems")

    # ---------- Render PDF: one page per problem ----------
    with PdfPages(OUT_PDF) as pdf:
        # Title page
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.axis("off")
        body = (
            "Per-example walkthrough of CODI's 6 latent steps on GSM8K\n\n"
            "For each problem we show:\n"
            "  - the question text\n"
            "  - the gold marker chain  <<a op b = c>>...  #### answer\n"
            "  - per-step force-decode emit + parsed final number\n"
            "  - per-step modal top-3 tokens at L11 resid_post (what would emit\n"
            "    if the loop stopped here)\n"
            "  - which gold marker values appear in each step's emit\n\n"
            "Examples include both correctly-solved and failed problems.\n"
        )
        ax.text(0.04, 0.95, body, va="top", ha="left", family="monospace", fontsize=10)
        ax.set_title("Per-example walkthrough — interpretation guide",
                     fontsize=13, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        for p in selected:
            fig, ax = plt.subplots(figsize=(14, 10))
            ax.axis("off")
            ok_tag = "✓ CORRECT" if p["finally_correct"] else "✗ WRONG"
            txt = f"GSM8K idx {p['idx']}  [{ok_tag}]\n\n"
            txt += f"Q: {p['q']}\n\n"
            txt += f"GOLD chain ({p['n_markers']} markers, final = {p['gold']}):\n"
            for i, m in enumerate(p["markers"], 1):
                txt += f"  marker {i}: <<{m['a']:g} {m['op']} {m['b']:g} = {m['c']:g}>>\n"
            txt += "\n"
            txt += "Per-latent-step force-decode emit + parsed answer:\n"
            txt += f"  {'step':<5} {'emit (truncated)':<60s}  {'parsed':<10} {'L11 modal':<30}\n"
            for k in range(n_lat):
                em = p["step_emits"][k][:55]
                pv = p["emit_vals"][k]
                ll_top = p["ll_resid_post_L11_top3"][k]
                ll_str = "  ".join(f"{repr(t)[:5]}({pp:.2f})" for t, pp in ll_top)
                marker_match = ""
                if pv is not None:
                    for i, m in enumerate(p["markers"], 1):
                        if abs(pv - m["c"]) < 1e-3:
                            marker_match = f"  ← matches marker {i}"
                            break
                    if not marker_match and abs(pv - p["gold"]) < 1e-3:
                        marker_match = "  ← matches gold final"
                txt += (f"  {k+1:<5} {em!r:<60s}  "
                        f"{(str(pv) if pv is not None else '?'):<10} {ll_str:<30}{marker_match}\n")
            ax.text(0.01, 0.99, txt, va="top", ha="left", family="monospace",
                    fontsize=8.5)
            ax.set_title(f"Problem walkthrough — idx={p['idx']}",
                         fontsize=12, fontweight="bold")
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    print(f"saved {OUT_PDF}")

    # Also print a console summary
    print("\nConsole summary:")
    for p in selected:
        ok = "✓" if p["finally_correct"] else "✗"
        print(f"\n{ok} idx={p['idx']}: gold={p['gold']}, {p['n_markers']} markers")
        print(f"  Q: {p['q'][:90]}...")
        for i, m in enumerate(p["markers"], 1):
            print(f"  m{i}: <<{m['a']:g}{m['op']}{m['b']:g}={m['c']:g}>>")
        for k in range(n_lat):
            pv = p["emit_vals"][k]
            mark = ""
            if pv is not None:
                for i, m in enumerate(p["markers"], 1):
                    if abs(pv - m["c"]) < 1e-3:
                        mark = f" ← m{i}"
                if not mark and abs(pv - p["gold"]) < 1e-3:
                    mark = " ← gold"
            print(f"  step{k+1}: emit={pv}{mark}  | {p['step_emits'][k][:70]!r}")


if __name__ == "__main__":
    main()
