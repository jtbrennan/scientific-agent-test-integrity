"""Mutation-score table, raw and with equivalent mutants removed.

Usage: python harness/mutation_summary.py [--out results/mutation/summary.md]

Reads results/mutation/<task>__<target>.json (harness/mutation.py) and
results/mutation/equivalent.csv, the hand classification of every mutant
that survives the full hidden suite v2 (equivalent = 1: no behavioural
difference under the task specification). A mutant equivalent for v2 is
equivalent for every suite, so it is dropped from every denominator.
Timeouts count as killed. Survivors of the weaker suites that v2 kills are
by definition not equivalent.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUITES = ["visible", "unit", "mr_inv", "mr_eqv", "anchored", "v1", "v2"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "mutation" / "summary.md")
    a = ap.parse_args()
    eq: dict[str, set[str]] = {}
    for r in csv.DictReader((ROOT / "results" / "mutation" / "equivalent.csv").open()):
        if r["equivalent"] == "1":
            eq.setdefault(r["task"], set()).add(r["mutant"])
    out = ["# Mutation scores (harness/mutation_summary.py)", "",
           "Raw score / score with equivalent mutants removed. Timeouts count as killed.", "",
           "| task | mutants | equivalent | " + " | ".join(SUITES) + " |",
           "|---|---|---|" + "---|" * len(SUITES)]
    for f in sorted((ROOT / "results" / "mutation").glob("*__*.json")):
        task = f.stem.split("__")[0]
        d = {s["suite"]: s for s in json.loads(f.read_text())["suites"]}
        total = sum(d["v2"]["counts"].values())
        e = eq.get(task, set())
        unreviewed = [m for m in d["v2"]["survivors"] if m.split("__")[-1] not in e
                      and m.split("__")[-1] not in nonequiv(task)]
        cells = []
        for s in SUITES:
            if s not in d:
                cells.append("–")
                continue
            surv = {m.split("__")[-1] for m in d[s]["survivors"]}
            killed = total - len(surv)
            cells.append(f"{killed / total:.2f} / {killed / (total - len(e)):.2f}")
        out.append(f"| {task} | {total} | {len(e)} | " + " | ".join(cells) + " |")
        if unreviewed:
            out.append(f"\n**{task}: {len(unreviewed)} v2 survivors not classified:** {', '.join(unreviewed)}\n")
    out += ["", "Non-equivalent v2 survivors (real gaps in the hidden suites):", ""]
    for r in csv.DictReader((ROOT / "results" / "mutation" / "equivalent.csv").open()):
        if r["equivalent"] == "0":
            out.append(f"- {r['task']} {r['mutant']}: {r['reason']}")
    a.out.write_text("\n".join(out) + "\n")
    print("\n".join(out))


def nonequiv(task: str) -> set[str]:
    return {r["mutant"] for r in csv.DictReader((ROOT / "results" / "mutation" / "equivalent.csv").open())
            if r["task"] == task and r["equivalent"] == "0"}


if __name__ == "__main__":
    main()
