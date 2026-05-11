"""For problems CODI gets wrong, classify into 'silver' traces:
  - Silver: pred matches some valid math operation on the input numbers
    (a+b, a-b, b-a, a*b, a/b, b/a, plus a few less-canonical: a*2, b*2, a, b)
  - Junk: pred matches no recognizable formula

Same analysis on the non-latent teacher (if available)."""

from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
from datasets import concatenate_datasets, load_dataset

REPO = Path(__file__).resolve().parents[2]
PD = REPO / "experiments" / "computation_probes"


def parse_two_operands(equation):
    nums = re.findall(r"-?\d+\.?\d*", equation)
    if len(nums) < 2: return None
    try: return float(nums[0]), float(nums[1])
    except: return None


def silver_label(pred, a, b, gold, tol=1e-3):
    """Return tag describing what 'silver' formula matches the pred."""
    if pred is None: return "no_pred"
    candidates = {
        "gold (a OP b — correct)": gold,
        "a+b": a + b,
        "a-b": a - b,
        "b-a": b - a,
        "a*b": a * b,
        "a/b": a / b if b != 0 else None,
        "b/a": b / a if a != 0 else None,
        "a": a,
        "b": b,
        "a+a": 2 * a,
        "b+b": 2 * b,
        "a*a": a * a,
        "b*b": b * b,
    }
    matches = []
    for k, v in candidates.items():
        if v is None: continue
        if abs(pred - round(v)) <= tol or (abs(v) > 1 and abs(pred - v) <= tol):
            matches.append(k)
    if not matches: return "junk"
    return matches[0]   # prefer gold first per dict order


def analyze(name, results, examples_full):
    """Compute silver/junk distribution."""
    print(f"\n=== {name} ===")
    n = len(results)
    n_correct = sum(1 for r in results if r.get("correct", False))
    print(f"  total: {n}, correct: {n_correct} ({100*n_correct/n:.1f}%)")

    # For each WRONG example, classify
    counts = {}
    parse_failed = 0
    for r, ex in zip(results, examples_full):
        if r.get("correct", False): continue
        ab = parse_two_operands(ex["Equation"])
        if ab is None: parse_failed += 1; continue
        a, b = ab
        gold = float(str(ex["Answer"]).replace(",", ""))
        pred = r.get("pred")
        if pred is None: parse_failed += 1; continue
        try: pred = float(pred)
        except: parse_failed += 1; continue
        tag = silver_label(pred, a, b, gold)
        counts[tag] = counts.get(tag, 0) + 1
    n_wrong = n - n_correct
    print(f"  wrong examples analyzed: {n_wrong - parse_failed} (parse-failed: {parse_failed})")
    print(f"  silver-trace breakdown of wrong predictions:")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {k:>40s}: {v:4d} ({100*v/n_wrong:.1f}% of wrong)")
    n_silver = sum(v for k, v in counts.items() if k != "junk")
    print(f"  any-formula match (silver): {n_silver}/{n_wrong} = {100*n_silver/n_wrong:.1f}%")
    return {"counts": counts, "n_wrong": n_wrong, "n_silver": int(n_silver),
            "n_correct": n_correct, "n_total": n}


def main():
    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])

    summary = {}

    # Student CODI on SVAMP
    sv_codi = json.load(open(REPO / "inference" / "runs" / "svamp_student_gpt2" / "results.json"))
    summary["codi_student_svamp"] = analyze("CODI student on SVAMP",
                                              sv_codi, [full[r["idx"]] for r in sv_codi])

    # CODI student on cf_balanced
    cf_data = json.load(open(REPO.parent / "cf-datasets" / "cf_balanced.json"))
    cf_codi = json.load(open(REPO / "inference" / "runs" / "cf_balanced_student_gpt2" / "results.json"))
    cf_examples = [{"Equation": cf_data[i]["orig_equation"],
                     "Answer": cf_data[i].get("cf_answer", cf_codi[i]["gold"])}
                    for i in range(len(cf_codi))]
    summary["codi_student_cf"] = analyze("CODI student on cf_balanced",
                                            cf_codi, cf_examples)

    # NON-LATENT TEACHER on SVAMP — explicitly use svamp_teacher
    sv_teacher_path = REPO / "inference" / "runs" / "svamp_teacher" / "results.json"
    if sv_teacher_path.exists():
        sv_teacher = json.load(open(sv_teacher_path))
        summary["non_latent_teacher_svamp"] = analyze(
            "non-latent teacher on SVAMP",
            sv_teacher, [full[r["idx"]] for r in sv_teacher])

    # Also CODI-Llama (svamp_student) — bigger student
    sv_llama_path = REPO / "inference" / "runs" / "svamp_student" / "results.json"
    if sv_llama_path.exists():
        sv_llama = json.load(open(sv_llama_path))
        summary["codi_llama_svamp"] = analyze(
            "CODI-Llama-1B student on SVAMP",
            sv_llama, [full[r["idx"]] for r in sv_llama])

    # CF teachers (use CF equations not SVAMP)
    for cf_dir, cf_json in [("cf_balanced", REPO.parent / "cf-datasets" / "cf_balanced.json"),
                             ("cf_magmatched", REPO.parent / "cf-datasets" / "cf_magmatched.json"),
                             ("cf_under99_b", REPO.parent / "cf-datasets" / "cf_under99_b.json"),
                             ("cf_gpt", REPO.parent / "cf-datasets" / "cf_gpt_transformed.json")]:
        teacher_path = REPO / "inference" / "runs" / f"{cf_dir}_teacher" / "results.json"
        if not teacher_path.exists() or not cf_json.exists(): continue
        tres = json.load(open(teacher_path))
        cf = json.load(open(cf_json))
        cf_examples_t = [{"Equation": cf[i].get("orig_equation",
                                                   cf[i].get("equation", "")),
                          "Answer": cf[i].get("cf_answer", tres[i].get("gold"))}
                          for i in range(min(len(tres), len(cf)))]
        summary[f"non_latent_teacher_{cf_dir}"] = analyze(
            f"non-latent teacher on {cf_dir}",
            tres[:len(cf_examples_t)], cf_examples_t)

    out = PD / "silver_trace_classification.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
