"""
Prompt consistency evaluation.

Runs the extraction pipeline (main.py's `extract_one`, unchanged) N times against
the same input file, then grades how consistent the results are across runs:
  - do the same entities / relationships come back each time?
  - do relationship subject/object values actually match a canonical entity name
    (the alias-duplication bug relationships subject/object == entity alias, not name)?
  - how often does the call succeed without needing a repair retry?

Usage:
  python eval.py                       # first file in input/, 10 runs
  python eval.py input/foo.pdf         # a specific file, 10 runs
  python eval.py input/foo.pdf --runs 5 --out eval/foo-quick
"""

import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from itertools import combinations

from anthropic import Anthropic

from main import INPUT_DIR, MODEL, extract_one, load_file

DEFAULT_RUNS = 10


# ---------------------------------------------------------------------------
# Extraction of comparable sets from a single run's data
# ---------------------------------------------------------------------------

def entity_name_set(data):
    return {e["name"] for e in data.get("entities", []) if isinstance(e, dict) and e.get("name")}


def entity_type_map(data):
    return {e["name"]: e.get("type") for e in data.get("entities", []) if isinstance(e, dict) and e.get("name")}


def relationship_triple_set(data):
    return {
        (r["subject"], r["relation"], r["object"])
        for r in data.get("relationships", [])
        if isinstance(r, dict) and r.get("subject") and r.get("relation") and r.get("object")
    }


def relationship_connection_set(data):
    """Subject/object pairs, ignoring the relation label — looser than the full triple."""
    return {
        tuple(sorted((r["subject"], r["object"])))
        for r in data.get("relationships", [])
        if isinstance(r, dict) and r.get("subject") and r.get("object")
    }


def canonical_name_consistency(data):
    """
    Fraction of relationship subject/object references that exactly match one of
    this run's own entity names. Catches the case where `relation` uses an alias
    ("Periša") instead of the canonical entity name ("Darko Periša"),
    which fragments the graph into duplicate nodes.
    Returns None when there's nothing to check (no entities or no relationships).
    """
    names = entity_name_set(data)
    refs = [
        v
        for r in data.get("relationships", [])
        if isinstance(r, dict)
        for v in (r.get("subject"), r.get("object"))
        if v
    ]
    if not names or not refs:
        return None
    return sum(1 for ref in refs if ref in names) / len(refs)


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

def jaccard(a, b):
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def pairwise_jaccard_stats(sets_list):
    scores = [jaccard(a, b) for a, b in combinations(sets_list, 2)]
    return stats_summary(scores)


def word_overlap(a, b):
    wa, wb = set((a or "").lower().split()), set((b or "").lower().split())
    return jaccard(wa, wb)


def entity_type_agreement(type_maps):
    """Of entity names seen in >=1 run, what fraction were assigned the same type everywhere?"""
    types_by_name = {}
    for tm in type_maps:
        for name, t in tm.items():
            types_by_name.setdefault(name, set()).add(t)
    if not types_by_name:
        return None
    agree = sum(1 for types in types_by_name.values() if len(types) == 1)
    return agree / len(types_by_name)


def stats_summary(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "min": None, "max": None, "stdev": None, "n": 0}
    return {
        "mean": round(statistics.mean(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "stdev": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
        "n": len(values),
    }


# ---------------------------------------------------------------------------
# Running the pipeline N times
# ---------------------------------------------------------------------------

def run_evaluation(path, runs, out_dir):
    content_block, fmt = load_file(path)
    if content_block is None:
        print(f"Error: could not load '{path}'.")
        sys.exit(1)

    client = Anthropic()
    os.makedirs(out_dir, exist_ok=True)

    print(f"Running {runs} extraction(s) on {path} (model={MODEL})...\n")
    results = []
    for i in range(1, runs + 1):
        print(f"-> run {i}/{runs}")
        try:
            data, warnings, error_reason = extract_one(client, content_block)
        except Exception as e:
            data, warnings, error_reason = None, [], str(e)

        record = {"data": data, "warnings": warnings, "error_reason": error_reason}
        results.append(record)

        with open(os.path.join(out_dir, f"run_{i:02d}.json"), "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        print(f"   {'OK' if error_reason is None else f'FAILED: {error_reason}'}")
        if warnings:
            print(f"   warnings: {'; '.join(warnings)}")

    return results


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def grade(results, path, runs):
    successful = [r["data"] for r in results if r["error_reason"] is None and isinstance(r["data"], dict)]
    passed = len(successful)

    entity_sets = [entity_name_set(d) for d in successful]
    type_maps = [entity_type_map(d) for d in successful]
    triple_sets = [relationship_triple_set(d) for d in successful]
    connection_sets = [relationship_connection_set(d) for d in successful]
    summaries = [d.get("summary") or "" for d in successful]
    cnc_scores = [canonical_name_consistency(d) for d in successful]

    entity_jaccard = pairwise_jaccard_stats(entity_sets)
    triple_jaccard = pairwise_jaccard_stats(triple_sets)
    connection_jaccard = pairwise_jaccard_stats(connection_sets)
    summary_overlap = stats_summary([word_overlap(a, b) for a, b in combinations(summaries, 2)])
    eta = entity_type_agreement(type_maps)
    cnc = stats_summary(cnc_scores)
    validity_rate = passed / runs if runs else 0.0

    # Overall score: weighted mean of whichever components have data, renormalized
    # over the available weight so a metric with no data (e.g. only 1 successful
    # run) doesn't silently drag the score to zero.
    components = [
        (0.30, entity_jaccard["mean"]),
        (0.30, connection_jaccard["mean"]),
        (0.15, cnc["mean"]),
        (0.125, eta),
        (0.125, validity_rate),
    ]
    available = [(w, v) for w, v in components if v is not None]
    overall = 100 * sum(w * v for w, v in available) / sum(w for w, _ in available) if available else None
    grade_letter = None
    if overall is not None:
        grade_letter = next(
            g for threshold, g in ((90, "A"), (80, "B"), (70, "C"), (60, "D"), (0, "F")) if overall >= threshold
        )

    return {
        "document": path,
        "model": MODEL,
        "runs": runs,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "validity": {"passed": passed, "failed": runs - passed, "rate": round(validity_rate, 3)},
        "entity_count": stats_summary([len(s) for s in entity_sets]),
        "relationship_count": stats_summary([len(s) for s in triple_sets]),
        "canonical_name_consistency": cnc,
        "entity_type_agreement": eta,
        "entity_name_jaccard": entity_jaccard,
        "relationship_triple_jaccard": triple_jaccard,
        "relationship_connection_jaccard": connection_jaccard,
        "summary_word_overlap": summary_overlap,
        "overall_score": round(overall, 1) if overall is not None else None,
        "grade": grade_letter,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(report):
    print("\n=== Consistency report ===")
    print(f"document:            {report['document']}")
    print(f"runs:                {report['runs']}  "
          f"(passed {report['validity']['passed']}, failed {report['validity']['failed']})")
    print(f"entity count:        mean {report['entity_count']['mean']}  stdev {report['entity_count']['stdev']}")
    print(f"relationship count:  mean {report['relationship_count']['mean']}  stdev {report['relationship_count']['stdev']}")
    print(f"entity name jaccard: mean {report['entity_name_jaccard']['mean']}  min {report['entity_name_jaccard']['min']}")
    print(f"connection jaccard:  mean {report['relationship_connection_jaccard']['mean']}  min {report['relationship_connection_jaccard']['min']}")
    print(f"canonical-name rate: mean {report['canonical_name_consistency']['mean']}  min {report['canonical_name_consistency']['min']}")
    print(f"entity type agree:   {report['entity_type_agreement']}")
    print(f"overall score:       {report['overall_score']}  (grade {report['grade']})")


def write_markdown_report(report, path):
    def fmt(stat):
        if stat is None or stat.get("mean") is None:
            return "n/a"
        return f"mean **{stat['mean']}** · min {stat['min']} · max {stat['max']} · stdev {stat['stdev']} (n={stat['n']})"

    lines = [
        f"# Consistency report — {report['document']}",
        "",
        f"- model: `{report['model']}`",
        f"- runs: {report['runs']} (passed {report['validity']['passed']}, failed {report['validity']['failed']})",
        f"- generated: {report['generated_at']}",
        "",
        f"## Overall score: {report['overall_score']} ({report['grade']})",
        "",
        "| metric | value |",
        "|---|---|",
        f"| entity count | {fmt(report['entity_count'])} |",
        f"| relationship count | {fmt(report['relationship_count'])} |",
        f"| entity name jaccard (cross-run) | {fmt(report['entity_name_jaccard'])} |",
        f"| relationship triple jaccard (exact, cross-run) | {fmt(report['relationship_triple_jaccard'])} |",
        f"| relationship connection jaccard (subject/object only, cross-run) | {fmt(report['relationship_connection_jaccard'])} |",
        f"| summary word overlap (cross-run) | {fmt(report['summary_word_overlap'])} |",
        f"| canonical-name consistency (intra-run) | {fmt(report['canonical_name_consistency'])} |",
        f"| entity type agreement | {report['entity_type_agreement']} |",
        "",
        "Jaccard scores are pairwise similarity across all run pairs "
        "(1.0 = identical sets, 0.0 = no overlap). Canonical-name consistency is the "
        "fraction of relationship subject/object values that exactly match an entity "
        "name found in that same run — low values mean relationships are using aliases "
        "instead of the canonical entity name (fragmenting the graph).",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", nargs="?", help=f"File to evaluate (defaults to the first file in {INPUT_DIR}/)")
    p.add_argument("--runs", type=int, default=DEFAULT_RUNS, help=f"Number of extraction runs (default: {DEFAULT_RUNS})")
    p.add_argument("--out", default=None, help="Output directory for run JSON + report (default: eval/<doc-stem>/)")
    return p.parse_args()


def resolve_path(path):
    if path:
        return path
    if not os.path.isdir(INPUT_DIR):
        print(f"Error: folder '{INPUT_DIR}' not found and no path given.")
        sys.exit(1)
    candidates = sorted(
        name
        for name in os.listdir(INPUT_DIR)
        if os.path.isfile(os.path.join(INPUT_DIR, name)) and not name.startswith(".")
    )
    if not candidates:
        print(f"No files in '{INPUT_DIR}' and no path given.")
        sys.exit(1)
    chosen = os.path.join(INPUT_DIR, candidates[0])
    print(f"No path given, using first file in '{INPUT_DIR}': {chosen}\n")
    return chosen


def main():
    args = parse_args()
    path = resolve_path(args.path)
    if not os.path.isfile(path):
        print(f"Error: '{path}' is not a file.")
        sys.exit(1)

    stem = os.path.splitext(os.path.basename(path))[0]
    out_dir = args.out or os.path.join("eval", stem)

    results = run_evaluation(path, args.runs, out_dir)
    report = grade(results, path, args.runs)

    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    write_markdown_report(report, os.path.join(out_dir, "report.md"))

    print_summary(report)
    print(f"\nSaved {args.runs} run(s) + report.json + report.md to {out_dir}/")


if __name__ == "__main__":
    main()
