"""Shared per-question feature extraction + group aggregation.

Used by both `latent_sweep_step_overview.py` (behavioral churn between forced-N
runs) and `latent_filter_step_overview.py` (mechanistic churn between latent
steps in the logit lens), so the two analyses produce directly comparable
feature breakdowns.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter

_NUM_RE = re.compile(r"-?\d+\.?\d*")


def magnitude_bucket(x: float) -> str:
    ax = abs(x)
    if ax < 10:
        return "0-9"
    if ax < 100:
        return "10-99"
    if ax < 1000:
        return "100-999"
    if ax < 10000:
        return "1000-9999"
    return "10000+"


def question_features(q: str, gold) -> dict:
    """Cheap, dataset-agnostic features. Handles SVAMP numerics and LOGIC-701 MC."""
    feats: dict = {
        "n_words": len(q.split()),
        "n_chars": len(q),
        "n_numbers_in_question": len(_NUM_RE.findall(q)),
        "is_multiple_choice": "Options:" in q,
    }
    try:
        gold_num = float(gold)
        feats["gold_numeric"] = gold_num
        feats["gold_magnitude_bucket"] = magnitude_bucket(gold_num)
        if feats["is_multiple_choice"] and float(gold_num).is_integer() and 1 <= gold_num <= 5:
            feats["gold_option"] = int(gold_num)
    except (TypeError, ValueError):
        pass
    return feats


def _stats(vals: list[float]) -> dict:
    if not vals:
        return {}
    return {
        "mean": round(statistics.fmean(vals), 2),
        "median": round(statistics.median(vals), 2),
        "min": min(vals),
        "max": max(vals),
    }


def agg_group(feats_list: list[dict]) -> dict:
    """Aggregate per-question features over a group."""
    if not feats_list:
        return {"count": 0}

    out: dict = {"count": len(feats_list)}
    for key in ("n_words", "n_chars", "n_numbers_in_question"):
        vals = [f[key] for f in feats_list if key in f]
        if vals:
            out[key] = _stats(vals)
    gold_vals = [f["gold_numeric"] for f in feats_list if "gold_numeric" in f]
    if gold_vals:
        out["gold_numeric"] = _stats(gold_vals)
    mag = Counter(f["gold_magnitude_bucket"] for f in feats_list
                  if "gold_magnitude_bucket" in f)
    if mag:
        out["gold_magnitude_dist"] = dict(sorted(mag.items()))
    opt = Counter(f["gold_option"] for f in feats_list if "gold_option" in f)
    if opt:
        out["gold_option_dist"] = dict(sorted(opt.items()))
    mc = Counter(f["is_multiple_choice"] for f in feats_list)
    if len(mc) > 1:
        out["is_multiple_choice_dist"] = dict(mc)
    return out


def print_breakdown_table(transition_summaries: list[dict], label_fn) -> None:
    """Compact 'top group features' table for stdout.

    transition_summaries: each element must have a 'feature_breakdown' dict with
    keys gained/lost/stable_right/stable_wrong, each an `agg_group` output.
    label_fn(t) -> short label like 'N0->N1' or '1->2'.
    """
    def _mean_or_dash(d: dict, key: str) -> str:
        if not d or key not in d or "mean" not in d[key]:
            return "-"
        return f"{d[key]['mean']:.1f}"

    print(f"\n{'Transition':<12}{'Group':<14}{'count':<8}{'words(mean)':<14}"
          f"{'gold_mag(top)':<16}{'opt_top':<10}")
    for t in transition_summaries:
        label = label_fn(t)
        for grp in ("gained", "lost", "stable_right", "stable_wrong"):
            agg = t["feature_breakdown"][grp]
            mag_top = ""
            if "gold_magnitude_dist" in agg:
                mag_top = max(agg["gold_magnitude_dist"].items(), key=lambda kv: kv[1])[0]
            opt_top = ""
            if "gold_option_dist" in agg:
                opt_top = str(max(agg["gold_option_dist"].items(), key=lambda kv: kv[1])[0])
            print(f"{label:<12}{grp:<14}{agg.get('count', 0):<8}"
                  f"{_mean_or_dash(agg, 'n_words'):<14}{mag_top:<16}{opt_top:<10}")
