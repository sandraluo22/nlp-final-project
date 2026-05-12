"""Every table written in this analysis, gathered into a single PDF.

Sources: JSONs and NPZs on disk, plus hardcoded summaries for tables that
were derived narratively (e.g., steering null tally, mode trichotomy).

Output: all_tables_summary_gsm8k.pdf
"""
from __future__ import annotations

import json, re
from pathlib import Path

import matplotlib
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

PD = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
CP = REPO / "experiments" / "computation_probes"
OPER = PD / "operator-probe"

OUT_PDF = PD / "all_tables_summary_gsm8k.pdf"


def load(p):
    try:
        if str(p).endswith(".npz"):
            return np.load(p)
        return json.load(open(p))
    except FileNotFoundError:
        return None


def make_table_page(pdf, title, headers, rows,
                    subtitle="", footnote="",
                    col_widths=None, row_colors=None,
                    fontsize=10, figsize=(13, 7.5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    if subtitle:
        ax.text(0.5, 0.95, subtitle, ha="center", va="top",
                fontsize=11, transform=ax.transAxes, color="#444")
    table = ax.table(
        cellText=rows, colLabels=headers, loc="center",
        cellLoc="center", colLoc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.5)
    # Header style
    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_facecolor("#4c72b0")
        cell.set_text_props(color="white", fontweight="bold")
    if row_colors:
        for i, color in enumerate(row_colors):
            for j in range(len(headers)):
                table[i + 1, j].set_facecolor(color)
    if footnote:
        ax.text(0.5, 0.03, footnote, ha="center", va="bottom",
                fontsize=9, transform=ax.transAxes, color="#666",
                family="monospace")
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def section(pdf, title, subtitle=""):
    fig, ax = plt.subplots(figsize=(11.5, 4))
    ax.axis("off")
    ax.text(0.5, 0.65, title, ha="center", va="center",
            fontsize=20, fontweight="bold", transform=ax.transAxes)
    if subtitle:
        ax.text(0.5, 0.35, subtitle, ha="center", va="center",
                fontsize=12, transform=ax.transAxes, color="#444",
                family="monospace")
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def main():
    # === Load data ===
    fdec = load(CP / "force_decode_per_step_gsm8k.json")
    flow = load(CP / "flow_map_gsm8k.npz")
    flow_meta = load(CP / "flow_map_gsm8k_meta.json")
    transitions = load(PD / "step_transitions_gsm8k.json")
    mop = load(OPER / "multi_op_probe_gsm8k.json")
    mop_refined = load(OPER / "multi_op_probe_refined_gsm8k.json")
    mop_evenodd = load(OPER / "multi_op_probe_evenodd_gsm8k.json")
    mop_best_per_step = load(OPER / "multi_op_probe_best_per_step_gsm8k.json")
    mop_trajectory = load(OPER / "multi_op_probe_trajectory_gsm8k.json")
    mop_shuf = load(OPER / "multi_op_probe_shuffle_gsm8k.json")
    correct_trace = load(PD / "correct_trace_patterns_gsm8k.json")
    other_num = load(PD / "other_number_patterns_gsm8k.json")
    wrong_val = load(PD / "wrong_value_patterns_gsm8k.json")
    final_wrong = load(PD / "final_wrong_patterns_gsm8k.json")
    late_beh = load(PD / "late_step_behavior_gsm8k.json")
    corr_v2 = load(OPER / "correctness_probe_gsm8k_v2.json")
    chain_emerge = load(PD / "chain_emergence_gsm8k.json")
    attn_num = load(PD / "correctness-analysis" / "attn_to_numbers_gsm8k.json")
    ll = load(CP / "logit_lens_gsm8k.json")

    with PdfPages(OUT_PDF) as pdf:
        # ===== Cover =====
        section(pdf, "All Tables Summary",
                "Every numerical table written during this CODI-GSM8K analysis,\n"
                "consolidated for easy reference.")

        # ===========================================================
        # SECTION A: Step-level descriptive
        # ===========================================================
        section(pdf, "A. Step-level descriptive",
                "Per-step accuracy, attention, block norms, transitions.")

        # T1: Force-decode accuracy per step
        if fdec:
            N = fdec["N"]
            accs = [sum(c) / N * 100 for c in fdec["correct_per_step"]]
            rows = []
            for k in range(6):
                delta = (accs[k] - accs[k-1]) if k > 0 else None
                d_str = f"{delta:+.1f}pp" if delta is not None else "—"
                rows.append([f"step {k+1}", f"{accs[k]:.1f}%", d_str])
            make_table_page(pdf,
                "T1. Force-decode accuracy per latent step",
                ["step", "accuracy", "Δ vs prev"], rows,
                subtitle=f"N={N} GSM8K test problems",
                footnote="Step 2→3 has the largest jump (+13.8pp); step 4→5 second (+4.2pp).")

        # T2: w→r / r→w transitions
        if transitions:
            rows = []
            for r in transitions["transitions"]:
                rows.append([f"{r['from']}→{r['to']}",
                              str(r["wr"]), str(r["rw"]),
                              str(r["rr"]), str(r["ww"]),
                              f"{r['net']:+d}"])
            make_table_page(pdf,
                "T2. Per-example correctness transitions",
                ["step pair", "w→r", "r→w", "r→r", "w→w", "net"], rows,
                subtitle=f"N={transitions['N']}; net = w→r minus r→w",
                footnote="Step 1→2 is a NET REGRESSION (−30). Step 2→3 rescues +182.")

        # T3: Per-step attention shares (avg over heads + layers)
        if flow is not None and flow_meta:
            ATTN = flow["mean_attn"]
            CLASS_NAMES = flow_meta["class_names"]
            attn_avg = ATTN[0, :6].mean(axis=(1, 2))
            classes = ["Q", "BOT", "L1", "L2", "L3", "L4", "L5", "L6"]
            cls_idx = [CLASS_NAMES.index(c) for c in classes]
            rows = []
            for s in range(6):
                row = [f"step {s+1}"]
                for ci in cls_idx:
                    row.append(f"{attn_avg[s, ci]:.2f}")
                rows.append(row)
            make_table_page(pdf,
                "T3. Per-step attention shares (avg over heads × layers)",
                ["step"] + classes, rows,
                subtitle="Fraction of last-token's attention to each position class",
                footnote="72-91% of attention every step goes to Q. Step 3→L2 (13%) and step 5→L4 (12%) are the chaining peaks.")

        # T4: Per-step block-output norms (sum over layers)
        if flow is not None:
            attn_norms = flow["mean_attn_norm"][0, :6].sum(axis=-1)
            mlp_norms = flow["mean_mlp_norm"][0, :6].sum(axis=-1)
            rows = []
            for s in range(6):
                marker = " ← largest" if mlp_norms[s] == mlp_norms.max() else ""
                rows.append([f"step {s+1}", f"{attn_norms[s]:.1f}",
                              f"{mlp_norms[s]:.1f}{marker}"])
            make_table_page(pdf,
                "T4. Per-step block-output norms (summed over 12 layers)",
                ["step", "‖attn out‖", "‖MLP out‖"], rows,
                subtitle="Mean L2 norms of last-token output at each block, summed across layers",
                footnote="Step 2's MLP is the loop's biggest single write (563.6).")

        # T5: Mode trichotomy (hardcoded from analysis)
        rows = [
            ["1. Loop did the work", "first-correct at step ≥ 2", "280", "21.2%"],
            ["2. Easy / shortcut right", "first-correct at step 1", "271", "20.5%"],
            ["3. Shortcut wrong", "wrong at step 6", "768", "58.2%"],
        ]
        make_table_page(pdf,
            "T5. Three-mode trichotomy (correctness × where it happened)",
            ["mode", "description", "n problems", "% of 1319"], rows,
            subtitle="CODI's three operating modes on GSM8K test",
            footnote="Modes 2 and 3 use the same Benford-shaped 'guess generator'; only the lottery differs.")

        # ===========================================================
        # SECTION B: Operator/marker probes
        # ===========================================================
        section(pdf, "B. Probe results (operator, marker, refined)")

        # T6: Multi-op probe best cells per marker
        if mop and "best" in mop:
            best = mop["best"]
            rows = []
            for p in ["op", "a_ld", "c_ld"]:
                for m in range(1, 5):
                    b = best[p].get(str(m))
                    if b:
                        rows.append([p, f"m={m}",
                                      f"step{b['step']} L{b['layer']}",
                                      f"{b['acc']:.3f}",
                                      f"{b['f1']:.3f}"])
            chance = "op: 0.25, a_ld: 0.10, c_ld: 0.10"
            make_table_page(pdf,
                "T6. Multi-op probe — best cells per (probe, marker)",
                ["probe", "marker", "best cell", "acc", "F1"], rows,
                subtitle=f"Chance: {chance}",
                footnote="RidgeClassifier(α=1.0, class-balanced) + StandardScaler, 80/20 split per cell.")

        # T7: Refined probe accuracy by filter
        if mop_refined:
            rows = []
            filters = ["original", "length_matched", "correct_only", "correct_and_length_matched"]
            for m in range(1, 5):
                for filt in filters:
                    if filt not in mop_refined: continue
                    info = mop_refined[filt].get("op", {}).get(str(m))
                    if info and info.get("best"):
                        b = info["best"]
                        rows.append([filt, f"m={m}",
                                      f"step{b['step']} L{b['layer']}",
                                      f"{b['acc']:.3f}",
                                      str(info.get("n_train_total", "?"))])
            make_table_page(pdf,
                "T7. Refined op-probe: best cells by data filter",
                ["filter", "marker", "best cell", "acc", "N"], rows,
                subtitle="Clean filter (correct + length-matched) lifts op decodability to 80-90%",
                fontsize=8.5)

        # T8: Cross-marker probe-transfer matrix
        if mop and "best" in mop:
            best = mop["best"]
            M = 4
            xfer = np.full((M, M), np.nan)
            for m1 in range(1, M + 1):
                b1 = best["op"].get(str(m1))
                if b1 is None: continue
                s_1, l_1 = b1["step"] - 1, b1["layer"]
                for m2 in range(1, M + 1):
                    G2 = np.array(mop["acc"]["op"][str(m2)])
                    xfer[m1 - 1, m2 - 1] = G2[s_1, l_1]
            rows = []
            for i in range(M):
                cell = best["op"][str(i+1)]
                row = [f"cell-m{i+1} (step{cell['step']} L{cell['layer']})"]
                for j in range(M):
                    star = "*" if i == j else ""
                    row.append(f"{xfer[i,j]:.3f}{star}")
                rows.append(row)
            make_table_page(pdf,
                "T8. Cross-marker probe-transfer matrix (op)",
                ["home cell"] + [f"predict m={j+1}" for j in range(M)], rows,
                subtitle="Acc when cell-m1's best cell is tested on marker m2's label. Diagonal = self.",
                footnote="Cell-m2 (step2 L0) actually predicts m=1 better than m=2 (0.553 > 0.493)! Weak marker-specificity.",
                fontsize=10)

        # T9: Shuffle baseline (real vs shuffled best)
        if mop_shuf:
            rows = []
            for p in ["op", "a_ld", "c_ld"]:
                for m in ["1", "2", "3", "4"]:
                    rb = mop_shuf["real_best"][p][m]
                    sb = mop_shuf["shuf_best"][p][m]
                    if rb and sb:
                        delta = rb["acc"] - sb["acc_mean"]
                        rows.append([p, f"m={m}",
                                      f"{rb['acc']:.3f}",
                                      f"{sb['acc_mean']:.3f}±{sb['acc_std']:.3f}",
                                      f"{delta:+.3f}"])
            make_table_page(pdf,
                "T9. Shuffle-step null baseline (real vs step-shuffled)",
                ["probe", "marker", "real best", "shuffled best (mean±std)", "Δ"], rows,
                subtitle="3 seeds of per-example step permutation, re-fit probes",
                footnote="Real beats shuffled by +0.07-0.17pp — step ordering carries marker-specific info.")

        # T10: Even/odd best cells
        if mop_evenodd and "odd_even" in mop_evenodd:
            oe = mop_evenodd["odd_even"]
            rows = []
            for p in ["op", "a_ld", "c_ld"]:
                for m in ["1", "2", "3", "4"]:
                    s = oe[p][m]
                    o_cell = f"step{s['odd_max_cell'][0]} L{s['odd_max_cell'][1]}"
                    e_cell = f"step{s['even_max_cell'][0]} L{s['even_max_cell'][1]}"
                    delta = s["delta_max_odd_minus_even"]
                    rows.append([p, f"m={m}", o_cell, f"{s['odd_max']:.3f}",
                                  e_cell, f"{s['even_max']:.3f}",
                                  f"{delta:+.3f}"])
            make_table_page(pdf,
                "T10. Best ODD-step cell vs best EVEN-step cell per marker",
                ["probe", "marker", "ODD cell", "ODD acc", "EVEN cell", "EVEN acc", "Δ"], rows,
                subtitle="c_ld: odd-step wins all 4 markers. op: even-step wins for m≥2.",
                fontsize=9)

        # T11: Best-anywhere-per-step (max-over-layers per step) for op
        if mop_best_per_step:
            d = mop_best_per_step
            rows = []
            for p in ["op", "c_ld"]:
                for m in range(1, 5):
                    accs = d["best_at_step"][p][str(m)]
                    layers = d["best_at_step_layer"][p][str(m)]
                    row = [p, f"m={m}"]
                    for k in range(6):
                        row.append(f"{accs[k]:.2f}/L{layers[k]}")
                    rows.append(row)
            make_table_page(pdf,
                "T11. Best-anywhere-per-step accuracy (max over layers per step, with best layer)",
                ["probe", "marker", "step1", "step2", "step3", "step4", "step5", "step6"], rows,
                subtitle="At each step k, the highest-acc layer's accuracy (with which layer)",
                fontsize=8)

        # T12: Trajectory per-step acc (CV OOF)
        if mop_trajectory:
            t = mop_trajectory
            rows = []
            for p in ["op", "a_ld", "c_ld"]:
                for m in range(1, 5):
                    accs = t["correctness_per_step"][p][str(m)]
                    row = [p, f"m={m}"]
                    for v in accs:
                        row.append(f"{v:.3f}")
                    rows.append(row)
            make_table_page(pdf,
                "T12. CV-OOF per-step trajectory accuracy",
                ["probe", "marker", "step1", "step2", "step3", "step4", "step5", "step6"], rows,
                subtitle="5-fold cross-validated probe at best-layer-per-step",
                fontsize=9)

        # ===========================================================
        # SECTION C: Logit lens & attention to numbers
        # ===========================================================
        section(pdf, "C. Logit lens + attention to numbers")

        # T13: Logit lens at L11 resid_post per step
        if ll:
            modal = ll["modal_token"]
            conf = np.array(ll["mean_top1_conf"])
            SUBL = ll["SUBLAYERS"]
            attn_i = SUBL.index("attn_out")
            mlp_i = SUBL.index("mlp_out")
            rp_i = SUBL.index("resid_post")
            rows = []
            for s in range(6):
                row = [f"step {s+1}"]
                for sl_i, _ in [(attn_i, "attn"), (mlp_i, "mlp"), (rp_i, "resid")]:
                    tk = (modal[s][11][sl_i] or "")
                    c = conf[s, 11, sl_i]
                    row.append(f"{repr(tk)[:10]} ({c:.2f})")
                rows.append(row)
            make_table_page(pdf,
                "T13. Logit lens at L11 (last layer) — modal top-1 token per step",
                ["step", "attn_out @ L11", "mlp_out @ L11", "resid_post @ L11"], rows,
                subtitle=f"N={ll['N_examples']} GSM8K problems; modal = most-common top-1",
                footnote="Odd steps emit '>>' (marker close); even steps emit digits. MLP outputs are mostly noise word-fragments.",
                fontsize=9.5)

        # T14: Logit lens attn_out at L10 per step (the "operator commit" cell)
        if ll:
            modal = ll["modal_token"]
            conf = np.array(ll["mean_top1_conf"])
            attn_i = SUBL.index("attn_out")
            rows = []
            for s in range(6):
                tk = (modal[s][10][attn_i] or "")
                c = conf[s, 10, attn_i]
                rows.append([f"step {s+1}", repr(tk)[:14], f"{c:.2f}"])
            make_table_page(pdf,
                "T14. Logit lens at L10 attn_out per step (the operator-commit cell)",
                ["step", "modal token", "confidence"], rows,
                subtitle="At step 1 L10 attn_out, modal token is ' *' (0.58)",
                footnote="Step 1: writes '*' (operator). Steps 3,5: writes '<<' (marker open).")

        # T15: Attention to numbers per step
        if attn_num:
            rows = []
            for s, f in enumerate(attn_num["per_step_num_frac_avg_lh"]["latent"]):
                rows.append([f"latent step {s+1}", f"{f*100:.1f}%"])
            for d, f in enumerate(attn_num["per_step_num_frac_avg_lh"]["decode"][:6]):
                rows.append([f"decode step {d+1}", f"{f*100:.1f}%"])
            make_table_page(pdf,
                "T15. % of Q-attention going to NUMBER tokens specifically",
                ["step", "% to numbers"], rows,
                subtitle="Per-step fraction of question-attention directed at numeric tokens",
                footnote="Even steps preferentially read numbers (9-11%); odd steps 4-5%.")

        # ===========================================================
        # SECTION D: Steering null tally
        # ===========================================================
        section(pdf, "D. Steering experiments (all null)")

        # T16: 7 steering experiments
        rows = [
            ["1", "residual @ block out", "step1 L11", "LDA add↔mul (cf_op_strict)",
             "α ∈ [−100, +100]", "0"],
            ["2", "residual @ block out", "refined cells × 4", "per-cell LDA (cf_natural)",
             "α ∈ [−80, +80]", "0"],
            ["3", "residual @ block out", "refined cells × 4", "vary-op template direction",
             "α_eff = ±465", "0"],
            ["4", "residual @ block out", "1/2/3/4 cells joint", "vary-op template",
             "α_eff = ±467", "0"],
            ["5", "residual @ block out", "step5 L11", "gradient-trained v (training-mismatch eval)",
             "loss 27→13", "0"],
            ["6", "residual @ block out", "step5 L11", "gradient-trained v (matched eval)",
             "loss 21→13", "0"],
            ["7", "ATTENTION output", "step1 L10", "LM-head direction *→{+,-,/}",
             "α ∈ [−3, +12]", "0"],
        ]
        make_table_page(pdf,
            "T16. Seven independent steering experiments",
            ["#", "intervention", "cell", "direction", "α range", "flips"], rows,
            subtitle="~5000 total steered runs, 0 operator flips",
            footnote="Even the LM-head direction at the operator-committing cell fails.",
            fontsize=8.5)

        # T17: trained vector eval table
        rows = [
            ["0.0", "3", "2", "16"],
            ["0.25", "3", "2", "16"],
            ["0.5", "3", "2", "16"],
            ["1.0", "3", "2", "16"],
            ["2.0", "3", "2", "16"],
            ["4.0", "3", "2", "16"],
        ]
        make_table_page(pdf,
            "T17. Trained-vector eval (matched protocol, N=21 held-out cf_op_strict ADD)",
            ["α", "match_add", "match_mul", "match_other"], rows,
            subtitle="Identical distribution at every α — argmax doesn't move despite loss dropping during training")

        # T18: Multi-cell joint result
        if True:
            rows = [
                ["1-cell", "75/80=94%", "0", "0", "0"],
                ["2-cell", "74", "0", "0", "0"],
                ["3-cell", "74", "0", "0", "0"],
                ["4-cell", "73", "0", "0", "0"],
            ]
            make_table_page(pdf,
                "T18. Multi-cell joint vary-op intervention (α_eff=±467 results)",
                ["# cells", "baseline", "α=−3", "α=0", "α=+3"], rows,
                subtitle="Joint intervention at 1-4 refined-probe cells; flips_to_target counts",
                footnote="No improvement from distribution. Wider/different cell sets remain untested.")

        # ===========================================================
        # SECTION E: Correctness probe
        # ===========================================================
        section(pdf, "E. Correctness probe (the 'CODI knows when it's wrong' finding)")

        if corr_v2:
            g1 = np.array(corr_v2["probe1_predict_final"])
            g2 = np.array(corr_v2["probe2_predict_self_step"])
            ps_corr = corr_v2["per_step_correct_pct"]
            maj_final = corr_v2["maj_baseline_final_pct"]
            rows = []
            for s in range(6):
                p1 = g1[s].max() * 100
                p2 = g2[s].max() * 100
                p1_layer = int(np.argmax(g1[s]))
                p2_layer = int(np.argmax(g2[s]))
                self_acc = ps_corr[s]
                maj_self = max(self_acc, 100 - self_acc)
                rows.append([f"step {s+1}",
                              f"{p1:.1f}% (L{p1_layer})",
                              f"+{p1-maj_final:.1f}pp",
                              f"{p2:.1f}% (L{p2_layer})",
                              f"{maj_self:.1f}%"])
            make_table_page(pdf,
                "T19. Correctness probe — per-step trajectory",
                ["step", "Probe1: predict final", "vs base", "Probe2: predict this-step", "P2 base"], rows,
                subtitle=f"Probe 1 majority baseline: {maj_final:.1f}% (since 58% wrong)",
                footnote="At step 6, residual predicts final correctness with 78.0% acc (+20pp over baseline).")

        # ===========================================================
        # SECTION F: Correct-trace patterns
        # ===========================================================
        section(pdf, "F. Correct-trace patterns")

        # T20: first-correct-step distribution
        if correct_trace:
            fcs = correct_trace["first_correct_distribution"]
            total = correct_trace["n_correct_total"]
            rows = []
            for k in range(1, 7):
                n = fcs.get(str(k), 0)
                rows.append([f"step {k}", str(n), f"{n/total*100:.1f}%"])
            make_table_page(pdf,
                "T20. First-correct-step distribution (among CODI-correct problems)",
                ["first-correct at", "n problems", "% of 551"], rows,
                subtitle=f"Total correct-at-step-6: {total}",
                footnote="49% correct from step 1 (no rescue); 23% rescued specifically at step 3.")

        # T21: chain length × first-correct (rebuilt from correct_trace.by_nmarkers_x_firstk)
        if correct_trace:
            mat = correct_trace["by_nmarkers_x_firstk"]
            nm_keys = sorted(int(k) for k in mat.keys() if int(k) <= 5)
            rows = []
            for nm in nm_keys:
                d = mat[str(nm)]
                row = [str(nm), str(sum(d.values()))]
                for k in range(1, 7):
                    row.append(str(d.get(str(k), 0)))
                rows.append(row)
            make_table_page(pdf,
                "T21. First-correct step × chain length (n_markers)",
                ["n_markers", "total correct", "k=1", "k=2", "k=3", "k=4", "k=5", "k=6"], rows,
                subtitle="How many correct problems with N markers were first-correct at each step",
                footnote="Longer chains (3-4 markers) disproportionately use the step-5 rescue cohort.")

        # T22: early-emit categories by first-correct-step
        if correct_trace and "early_emit_categories_by_firstk" in correct_trace:
            d = correct_trace["early_emit_categories_by_firstk"]
            rows = []
            for k_str in sorted(d.keys(), key=lambda x: int(x)):
                c = d[k_str]
                tot = sum(c.values())
                if tot == 0: continue
                rows.append([f"k={k_str}",
                              f"{c.get('matches_marker', 0)} ({c.get('matches_marker', 0)/tot*100:.0f}%)",
                              f"{c.get('other_number', 0)} ({c.get('other_number', 0)/tot*100:.0f}%)",
                              f"{c.get('no_match', 0)}",
                              f"{c.get('zero', 0)}"])
            make_table_page(pdf,
                "T22. Earlier-wrong emit categories by first-correct-step",
                ["first_correct_k", "matches a marker", "other number", "no_match", "zero"], rows,
                subtitle="Of the earlier wrong-emit steps before becoming correct, what category?",
                footnote="~88% of pre-rescue wrong emits are 'other_number' — unrelated to gold markers.")

        # ===========================================================
        # SECTION G: Wrong-emit value patterns
        # ===========================================================
        section(pdf, "G. Wrong-value patterns")

        # T23: pre-rescue wrong-emit category bar (multi-label)
        if other_num:
            counts = other_num["category_counts"]
            total = other_num["n_other_number_wrong_emits"]
            rows = [[c, str(counts[c]), f"{counts[c]/total*100:.1f}%"]
                    for c in sorted(counts.keys(), key=lambda k: -counts[k])]
            make_table_page(pdf,
                "T23. Pre-rescue wrong-emit categories (multi-label)",
                ["category", "n", "% of N"], rows,
                subtitle=f"N={total} 'other_number' wrong emits from rescued problems")

        # T24: top wrong values (rescue cohort)
        if wrong_val:
            rows = [[f"{e['value']:g}", str(e["count"]), f"{e['pct']:.1f}%"]
                    for e in wrong_val["top_values"][:15]]
            make_table_page(pdf,
                "T24. Top-15 specific wrong-emit values (rescue cohort, N=534)",
                ["value", "count", "% of N"], rows,
                footnote="The number 5 alone = 4.5% of rescue-cohort wrong emits.")

        # T25: First-digit Benford (rescue)
        if wrong_val:
            fd_dist = wrong_val["first_digit_dist_pct"]
            benford = wrong_val["benford_pct"]
            rows = []
            for d in range(9):
                rows.append([str(d+1), f"{fd_dist[d]:.1f}%", f"{benford[d]:.1f}%"])
            make_table_page(pdf,
                "T25. First-digit distribution — rescue-cohort wrong vs Benford",
                ["first digit", "wrong-emit %", "Benford %"], rows,
                subtitle="N=534 pre-rescue wrong emits",
                footnote="Near-perfect Benford match — model emits natural-looking numbers.")

        # T26: Last-digit (rescue)
        if wrong_val:
            ld_dist = wrong_val["last_digit_dist_pct"]
            rows = [[str(d), f"{ld_dist[d]:.1f}%"] for d in range(10)]
            make_table_page(pdf,
                "T26. Last-digit distribution — rescue cohort",
                ["last digit", "% of N"], rows,
                subtitle="Uniform expectation: 10% per digit",
                footnote="0: 30.3%, 5: 14.6% — strong round-number bias.")

        # T27: Top wrong-FINAL values
        if final_wrong:
            rows = [[f"{e['value']:g}", str(e["count"]), f"{e['pct']:.1f}%"]
                    for e in final_wrong["top_values"][:15]]
            make_table_page(pdf,
                "T27. Top-15 wrong-FINAL values (N=768 never-correct problems)",
                ["value", "count", "% of N"], rows,
                footnote="More diverse than rescue-cohort wrongs; no single value > 2.7%.")

        # T28: First-digit (wrong final)
        if final_wrong:
            fd_dist = final_wrong["first_digit_dist_pct"]
            benford = final_wrong["benford_pct"]
            rows = []
            for d in range(9):
                rows.append([str(d+1), f"{fd_dist[d]:.1f}%", f"{benford[d]:.1f}%"])
            make_table_page(pdf,
                "T28. First-digit distribution — wrong final vs Benford",
                ["first digit", "wrong-final %", "Benford %"], rows,
                subtitle="N=768 problems CODI never gets right")

        # T29: Last-digit (wrong final)
        if final_wrong:
            ld_dist = final_wrong["last_digit_dist_pct"]
            rows = [[str(d), f"{ld_dist[d]:.1f}%"] for d in range(10)]
            make_table_page(pdf,
                "T29. Last-digit distribution — wrong final",
                ["last digit", "% of N"], rows,
                footnote="Last-digit 0: 37.0% (even stronger round-bias than rescue cohort's 30.3%).")

        # T30: Roundness comparison
        if final_wrong:
            r = final_wrong["roundness"]
            rows = [
                ["multiple of 5", f"{r['mult5_pct']:.1f}%", f"{r['gold_mult5_pct']:.1f}%"],
                ["multiple of 10", f"{r['mult10_pct']:.1f}%", f"{r['gold_mult10_pct']:.1f}%"],
                ["multiple of 100", f"{r['mult100_pct']:.1f}%", "—"],
            ]
            make_table_page(pdf,
                "T30. Roundness comparison — wrong final vs gold",
                ["pattern", "wrong-final %", "gold %"], rows,
                subtitle="N=768 wrong; gold = the correct answers these problems were supposed to have",
                footnote="Wrong answers' round-number profile NEARLY MATCHES gold's. Model has learned the answer DISTRIBUTION.")

        # T31: Distance to gold
        if final_wrong:
            d = final_wrong["distance_to_gold"]
            rows = [
                ["median absolute |emit−gold|", f"{d['median']:g}"],
                ["mean absolute (skewed)", f"{d['mean']:g}"],
                ["max", f"{d['max']:g}"],
                ["median relative |Δ|/|gold|", f"{d['median_rel']:.2f}×"],
            ]
            make_table_page(pdf,
                "T31. Distance from gold (wrong-final answers)",
                ["statistic", "value"], rows,
                footnote="Median wrong answer is within 22 of gold (and within 0.5× relative).")

        # T32: Match question content (final wrong)
        if final_wrong:
            d = final_wrong["match_question_content"]
            rows = [
                ["= a question operand", f"{d['operand_pct']:.1f}%"],
                ["= a pair-op of two operands", f"{d['op_pair_pct']:.1f}%"],
                ["= an intermediate marker", f"{d['marker_pct']:.1f}%"],
            ]
            make_table_page(pdf,
                "T32. Wrong-final emit matches what?",
                ["match type", "% of 768"], rows,
                footnote="~30% trace to a recognizable partial-computation mistake; 70% are 'GSM8K-shaped' guesses.")

        # ===========================================================
        # SECTION H: Late-step behavior
        # ===========================================================
        section(pdf, "H. Late-step behavior (the 'after step 3, just approximation' findings)")

        # T33: H1 recovery by first-wrong step
        if late_beh:
            # Compute per-step recovery
            rows = [["step 3", f"{late_beh['h1_recovery_after_step3']['pct']:.1f}%",
                      "86% who are wrong at step 3 stay wrong"]]
            make_table_page(pdf,
                "T33. H1: Recovery rate after first-wrong step (selected)",
                ["wrong at step", "recovery rate", "interpretation"], rows,
                subtitle="If wrong by step 3, recovery is rare")

        # T34: H2 emit stability across late steps
        if late_beh:
            s = late_beh["h2_stability"]
            rows = []
            for k in sorted(s.keys()):
                info = s[k]
                rows.append([k, f"{info['n_same']}/{info['n_total']}",
                              f"{info['pct']:.1f}%"])
            make_table_page(pdf,
                "T34. H2: Emit stability across late steps for wrong-final problems",
                ["transition", "identical emit", "%"], rows,
                subtitle=f"N=768 wrong-final problems",
                footnote="step 5 == step 6: 84.6% identical — strong late-step stabilization.")

        # T35: H2b unique emits across steps 4-6
        if late_beh:
            d = late_beh["h2b_unique_late_emits"]
            rows = []
            for k_str in sorted(d.keys(), key=lambda x: int(x)):
                n = d[k_str]
                rows.append([k_str, str(n)])
            make_table_page(pdf,
                "T35. H2b: # unique emit values across steps 4-6 for wrong-final",
                ["# unique values", "# problems"], rows,
                footnote="48% have just ONE emit across steps 4-6 — fully settled.")

        # T36: H3 first-digit profile across steps (wrong emits)
        if late_beh:
            bf_per_step = late_beh["h3_first_digit_pct_by_step"]
            rows = []
            for d in range(9):
                row = [str(d+1)]
                for k in range(6):
                    row.append(f"{bf_per_step[k][d]:.1f}")
                rows.append(row)
            make_table_page(pdf,
                "T36. H3: First-digit distribution of WRONG emits, per step (%)",
                ["digit", "step1", "step2", "step3", "step4", "step5", "step6"], rows,
                subtitle="Wrong-guess generator is essentially step-invariant",
                footnote="Distribution at step 1 ≈ distribution at step 6. No learning across steps.")

        # T37: H4 avg n_markers by first-correct
        if late_beh:
            d = late_beh["h4_first_correct_x_chain_length"]
            rows = []
            # we need the per-cohort enriched problems to compute avg; but we have n_marker_dist
            for k_str in sorted(d.keys(), key=lambda x: int(x)):
                info = d[k_str]
                # compute avg from dist
                nm_dist = info["n_marker_dist"]
                total = sum(nm_dist.values())
                if total == 0: continue
                avg = sum(int(nm) * cnt for nm, cnt in nm_dist.items()) / total
                rows.append([f"k={k_str}", str(info["n"]), f"{avg:.2f}",
                              ", ".join(f"{nm}×{cnt}" for nm, cnt in sorted(nm_dist.items(), key=lambda x: int(x[0])))[:50]])
            make_table_page(pdf,
                "T37. H4: First-correct cohort × avg chain length",
                ["first-correct k", "n", "avg n_markers", "dist (truncated)"], rows,
                subtitle="Late-rescue (step 5) cohorts are dominated by long chains",
                footnote="step-5 cohort avg = 3.27 markers vs step-1 cohort avg = 2.15 — second compute window for long chains.",
                fontsize=8)

        # ===========================================================
        # SECTION I: Chain emergence
        # ===========================================================
        section(pdf, "I. Chain emergence in emits")

        if chain_emerge:
            frac = np.array(chain_emerge["frac_present_per_step_marker"])
            rows = []
            for k in range(frac.shape[0]):
                row = [f"step {k+1}"]
                for m in range(4):
                    row.append(f"{frac[k, m]*100:.1f}%")
                rows.append(row)
            make_table_page(pdf,
                "T38. Marker emergence: % of problems where emit final = marker m's gold",
                ["step", "m=1", "m=2", "m=3", "m=4"], rows,
                subtitle="Strict test on N=1319 GSM8K test problems",
                footnote="m=2 and m=3 both first plateau at step 3 (~20%); m=4 at step 5 (~15%).")

        # ===========================================================
        # SECTION J: Evidence quality grading
        # ===========================================================
        section(pdf, "J. Evidence-quality grading")

        # T39: solid/inferential/guessed
        rows = [
            ["force-decode acc per step",
             "✓ Solid", "direct measurement N=1319"],
            ["w→r / r→w transitions",
             "✓ Solid", "direct measurement"],
            ["86% wrong-at-step-3 stay wrong",
             "✓ Solid", "direct measurement"],
            ["48% wrong-final have 1 unique emit across 4-6",
             "✓ Solid", "direct measurement"],
            ["Step-invariant Benford profile",
             "✓ Solid", "direct distributional check"],
            ["Late-rescue cohort has avg 3.27 markers",
             "✓ Solid", "direct measurement"],
            ["Correctness probe: 78% predicting final",
             "✓ Solid", "RidgeClassifier on residual"],
            ["72-91% Q-attention",
             "✓ Solid", "flow_map measurement"],
            ["7 steering experiments all null",
             "✓ Solid", "5000+ steered runs"],
            ["Step 2 commits, step 3 synthesizes via L2-attention",
             "⚠ Inferential", "fits data but not causally tested"],
            ["Step 4→5 is a second compute window",
             "⚠ Inferential", "chain-length correlation only"],
            ["Step 2's MLP carries computation",
             "⚠ Inferential", "MLP norm largest but content unclear"],
            ["Mode 2 'shortcut' is distinct from Mode 3 'wrong guess'",
             "✗ Guessed", "boundary is fuzzy"],
            ["'Approximation mode' is the right framing",
             "✗ Guessed", "consistent but not separately tested"],
            ["Probes correlational only (not causal)",
             "✗ Partly Guessed", "7 null steers ≠ all interventions tested"],
        ]
        make_table_page(pdf,
            "T39. Evidence-quality grading",
            ["claim", "verdict", "basis"], rows,
            subtitle="Honest assessment of which findings are direct measurement vs interpretation",
            fontsize=8.5)

    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
