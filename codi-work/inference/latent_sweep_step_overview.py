"""Behavioral (final-accuracy) overview across forced latent steps N=0..K.

Mirrors `latent_filter_step_overview.py` but uses end-to-end `correct` from each
latent-sweep `results.json` instead of logit-lens gold-token ranks. This gives
gained/lost question lists per adjacent-N pair, where "gained" means a question
that was wrong at N=i but right at N=i+1 in the actual decoded answer.

Inputs:
  --sweep_dir <DIR>         directory containing N0/, N1/, ... each with results.json
  --out_dir   <DIR>         where to write outputs
  [--ns 0 1 2 3 4 5 6]      which N indices to include (default: auto-detect)

Outputs (under --out_dir):
  step_overview.json        per-N counts, pairwise transitions, per-question correctness traces
  step_overview.png         per-N right/wrong stacked bar chart
  transitions.png           grouped gained/lost bar chart per adjacent N pair
  transitions/N{a}_N{b}.json  per-transition question lists (gained / lost / stable)

Usage:
  python latent_sweep_step_overview.py \\
      --sweep_dir ../latent-sweep/svamp_latent_sweep_llama \\
      --out_dir   analysis/svamp_sweep_overview \\
      --dataset_name "SVAMP (Llama CODI)"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_INF = Path(__file__).resolve().parent
if str(_INF) not in sys.path:
    sys.path.insert(0, str(_INF))

from question_features import agg_group, print_breakdown_table, question_features  # noqa: E402


def _load_results(p: Path) -> list[dict]:
    with open(p) as f:
        rows = json.load(f)
    return rows


def _by_idx(rows: list[dict]) -> dict[int, dict]:
    return {int(r["idx"]): r for r in rows}


def _detect_ns(sweep_dir: Path) -> list[int]:
    ns = []
    for d in sweep_dir.iterdir():
        if not d.is_dir():
            continue
        m = re.fullmatch(r"N(\d+)", d.name)
        if m and (d / "results.json").is_file():
            ns.append(int(m.group(1)))
    return sorted(ns)


def _q_entry(r: dict, n_a: int, n_b: int, corr_a: bool, corr_b: bool) -> dict:
    return {
        "idx": r["idx"],
        "question": r["question"],
        "gold": r["gold"],
        f"pred_N{n_a}": r.get(f"pred_N{n_a}"),
        f"pred_N{n_b}": r.get(f"pred_N{n_b}"),
        f"response_N{n_a}": r.get(f"response_N{n_a}"),
        f"response_N{n_b}": r.get(f"response_N{n_b}"),
        f"correct_N{n_a}": corr_a,
        f"correct_N{n_b}": corr_b,
    }


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--sweep_dir", required=True,
                   help="Directory containing N0/, N1/, ... each with results.json")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--ns", type=int, nargs="+", default=None,
                   help="Explicit list of N values to include (default: auto-detect)")
    p.add_argument("--dataset_name", default=None,
                   help="Label for plots (auto-detected from path if omitted)")
    args = p.parse_args()

    sweep_dir = Path(args.sweep_dir).expanduser().resolve()
    if not sweep_dir.is_dir():
        raise SystemExit(f"sweep_dir not found: {sweep_dir}")

    ns = args.ns if args.ns is not None else _detect_ns(sweep_dir)
    if len(ns) < 2:
        raise SystemExit(f"need >=2 N values, found {ns} under {sweep_dir}")

    dataset_name = args.dataset_name or sweep_dir.name

    # Load per-N results and build a master index of unique idx values.
    per_n_rows: dict[int, dict[int, dict]] = {}
    for n in ns:
        path = sweep_dir / f"N{n}" / "results.json"
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        per_n_rows[n] = _by_idx(_load_results(path))

    # Sanity: index sets should match across N. Use intersection to be safe.
    idx_sets = [set(per_n_rows[n].keys()) for n in ns]
    common = sorted(set.intersection(*idx_sets))
    if not common:
        raise SystemExit("no common idx across N runs; cannot align")
    N_questions = len(common)

    # Stack per-question correctness traces: row idx -> {N: bool}
    traces: dict[int, dict] = {}
    features: dict[int, dict] = {}
    for q in common:
        first = per_n_rows[ns[0]][q]
        tr = {
            "idx": q,
            "question": first["question"],
            "gold": first["gold"],
            "correct_per_N": {n: bool(per_n_rows[n][q]["correct"]) for n in ns},
            "pred_per_N": {n: per_n_rows[n][q].get("pred") for n in ns},
            "response_per_N": {n: per_n_rows[n][q].get("response") for n in ns},
        }
        traces[q] = tr
        features[q] = question_features(first["question"], first["gold"])

    # Per-N right/wrong counts on the common index.
    per_n_summary = []
    for n in ns:
        rcount = sum(1 for q in common if traces[q]["correct_per_N"][n])
        per_n_summary.append({"N": n, "right": rcount, "wrong": N_questions - rcount})

    # Pairwise adjacent transitions on `correct`.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    transitions_dir = out_dir / "transitions"
    transitions_dir.mkdir(exist_ok=True)

    transition_summaries = []
    for i in range(len(ns) - 1):
        n_a, n_b = ns[i], ns[i + 1]
        ca = {q for q in common if traces[q]["correct_per_N"][n_a]}
        cb = {q for q in common if traces[q]["correct_per_N"][n_b]}
        gained = sorted(cb - ca)            # wrong @ a, right @ b
        lost = sorted(ca - cb)              # right @ a, wrong @ b
        stable_right = sorted(ca & cb)
        stable_wrong = sorted((set(common) - ca) - cb)

        breakdown = {
            "gained": agg_group([features[q] for q in gained]),
            "lost": agg_group([features[q] for q in lost]),
            "stable_right": agg_group([features[q] for q in stable_right]),
            "stable_wrong": agg_group([features[q] for q in stable_wrong]),
        }

        t = {
            "from_N": n_a,
            "to_N": n_b,
            "gained_wrong_to_right": len(gained),
            "lost_right_to_wrong": len(lost),
            "stable_right": len(stable_right),
            "stable_wrong": len(stable_wrong),
            "feature_breakdown": breakdown,
        }
        transition_summaries.append(t)

        def _detail_entry(q: int) -> dict:
            tr = traces[q]
            ra = per_n_rows[n_a][q]
            rb = per_n_rows[n_b][q]
            return {
                "idx": q,
                "question": tr["question"],
                "gold": tr["gold"],
                f"pred_N{n_a}": ra.get("pred"),
                f"pred_N{n_b}": rb.get("pred"),
                f"response_N{n_a}": ra.get("response"),
                f"response_N{n_b}": rb.get("response"),
                f"correct_N{n_a}": bool(ra["correct"]),
                f"correct_N{n_b}": bool(rb["correct"]),
                "features": features[q],
            }

        detail = {
            **t,
            "gained_questions": [_detail_entry(q) for q in gained],
            "lost_questions":   [_detail_entry(q) for q in lost],
        }
        with open(transitions_dir / f"N{n_a}_N{n_b}.json", "w") as f:
            json.dump(detail, f, indent=2, ensure_ascii=False)

    overview = {
        "dataset": dataset_name,
        "sweep_dir": str(sweep_dir),
        "ns": ns,
        "N_common_questions": N_questions,
        "per_N": per_n_summary,
        "transitions": transition_summaries,
    }
    overview_path = out_dir / "step_overview.json"
    with open(overview_path, "w") as f:
        json.dump({"summary": overview,
                   "trajectories": list(traces.values())},
                  f, indent=2, ensure_ascii=False)
    print(f"\n[sweep-overview] wrote {overview_path}", flush=True)

    print(f"\n{'N':<6}{'Right':<8}{'Wrong':<8}{'Acc':<8}")
    for row in per_n_summary:
        acc = row["right"] / N_questions if N_questions else 0.0
        print(f"{row['N']:<6}{row['right']:<8}{row['wrong']:<8}{acc:<8.3f}")

    print(f"\n{'Transition':<14}{'Gained':<10}{'Lost':<10}{'Stable R':<10}{'Stable W':<10}")
    for t in transition_summaries:
        label = f"N{t['from_N']}->N{t['to_N']}"
        print(
            f"{label:<14}{t['gained_wrong_to_right']:<10}{t['lost_right_to_wrong']:<10}"
            f"{t['stable_right']:<10}{t['stable_wrong']:<10}"
        )

    print_breakdown_table(transition_summaries, lambda t: f"N{t['from_N']}->N{t['to_N']}")

    # ---- Plots ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[sweep-overview] matplotlib not installed; skipping plots.")
        return

    # Plot 1: per-N stacked bar of right/wrong.
    fig, ax = plt.subplots(figsize=(max(6, len(ns) * 1.2), 5))
    bar_w = 0.55
    xs = range(len(ns))
    rights = [row["right"] for row in per_n_summary]
    wrongs = [row["wrong"] for row in per_n_summary]
    ax.bar(xs, rights, bar_w, label="Right (final answer correct)", color="#4CAF50")
    ax.bar(xs, wrongs, bar_w, bottom=rights, label="Wrong", color="#F44336")
    for i, (r, w) in enumerate(zip(rights, wrongs)):
        if r > 0:
            ax.text(i, r / 2, str(r), ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white")
        if w > 0:
            ax.text(i, r + w / 2, str(w), ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"N={n}" for n in ns])
    ax.set_ylabel("Number of questions")
    ax.set_title(f"{dataset_name}: final correctness per forced N")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "step_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[sweep-overview] saved step_overview.png", flush=True)

    # Plot 2: gained/lost per adjacent pair.
    labels = [f"N{t['from_N']}\u2192N{t['to_N']}" for t in transition_summaries]
    gained = [t["gained_wrong_to_right"] for t in transition_summaries]
    lost = [t["lost_right_to_wrong"] for t in transition_summaries]

    fig, ax = plt.subplots(figsize=(max(6, len(transition_summaries) * 1.5), 5))
    xs = range(len(labels))
    w = 0.35
    bars_g = ax.bar([i - w / 2 for i in xs], gained, w,
                    label="Gained (wrong\u2192right)", color="#4CAF50")
    bars_l = ax.bar([i + w / 2 for i in xs], lost, w,
                    label="Lost (right\u2192wrong)", color="#F44336")
    for bar in list(bars_g) + list(bars_l):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, str(int(h)),
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Adjacent N transition")
    ax.set_ylabel("Number of questions")
    ax.set_title(f"{dataset_name}: questions gained/lost between adjacent N")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "transitions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[sweep-overview] saved transitions.png", flush=True)
    print(f"[sweep-overview] per-transition question lists in {transitions_dir}/",
          flush=True)


if __name__ == "__main__":
    main()
