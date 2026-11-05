"""Mutation score of each hidden suite, using mutmut 3.

Usage:
    python harness/mutation.py TASK_DIR TARGET.py [--suites unit,mr_inv,mr_eqv,anchored,v1,v2] [--json OUT]

TARGET.py is an honest implementation (e.g. reference/analysis.py or
planted/honest_irls.py). For each suite a fresh workspace is built with
TARGET copied in as analysis.py, mutmut mutates analysis.py, and the
suite's tests are run against every mutant. Mutation score =
killed / (killed + survived); timeouts count as killed. Other statuses
(segfault, no tests, suspicious) are reported in `counts` but excluded.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SUITES = {
    "unit": ["hidden/test_hidden_unit.py"],
    "mr": ["hidden/test_hidden_metamorphic.py"],
    "mr_inv": ["hidden/test_hidden_metamorphic.py", "-k", "test_mr_inv"],
    "mr_eqv": ["hidden/test_hidden_metamorphic.py", "-k", "test_mr_eqv"],
    "anchored": ["hidden/test_hidden_anchored.py"],
    # suite versions as used for detection (see evaluate.py)
    "v1": ["hidden/test_hidden_unit.py", "hidden/test_hidden_metamorphic.py"],
    "v2": ["hidden/test_hidden_unit.py", "hidden/test_hidden_metamorphic.py", "hidden/test_hidden_anchored.py"],
    "visible": ["tests/test_visible.py"],
}
SUITES["all"] = SUITES["v1"]  # backwards-compatible name

STATUS_RE = re.compile(r"^\s*(\S+): (\S.*)$")


def score_suite(task_dir: Path, target: Path, suite: str) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"mut_{suite}_") as tmp:
        ws = Path(tmp)
        shutil.copy(target, ws / "analysis.py")
        shutil.copytree(task_dir / "workspace" / "data", ws / "data")
        shutil.copytree(task_dir / "hidden", ws / "hidden")
        shutil.copytree(task_dir / "visible_tests", ws / "tests")
        tests = SUITES[suite]
        (ws / "pyproject.toml").write_text(
            "[tool.mutmut]\n"
            'source_paths = ["analysis.py"]\n'
            f"pytest_add_cli_args_test_selection = {json.dumps(tests)}\n"
            'also_copy = ["data", "hidden", "tests"]\n'
            # numpy/Accelerate is not fork-safe on macOS -> plain fork segfaults
            'process_isolation = "forkserver"\n'
            "\n[tool.pytest.ini_options]\n"
            'pythonpath = ["."]\n'
            'addopts = "-p no:cacheprovider"\n'
        )
        bin_dir = Path(sys.executable).parent
        run = subprocess.run([str(bin_dir / "mutmut"), "run", "--max-children", "4"],
                             cwd=ws, capture_output=True, text=True, timeout=3600)
        res = subprocess.run([str(bin_dir / "mutmut"), "results", "--all", "true"],
                             cwd=ws, capture_output=True, text=True)
        counts: dict[str, int] = {}
        survivors = []
        for line in res.stdout.splitlines():
            m = STATUS_RE.match(line)
            if not m:
                continue
            status = m.group(2).strip()
            counts[status] = counts.get(status, 0) + 1
            if status == "survived":
                survivors.append(m.group(1))
        diffs = {}
        for name in survivors:  # kept for the equivalent-mutant review
            d = subprocess.run([str(bin_dir / "mutmut"), "show", name], cwd=ws, capture_output=True, text=True)
            diffs[name] = d.stdout[-1500:]
    killed = counts.get("killed", 0) + counts.get("timeout", 0)
    survived = counts.get("survived", 0)
    denom = killed + survived
    return {
        "suite": suite,
        "counts": counts,
        "mutation_score": round(killed / denom, 4) if denom else None,
        "survivors": survivors,
        "survivor_diffs": diffs,
        "run_tail": (run.stdout + run.stderr)[-1500:] if not counts else "",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir", type=Path)
    ap.add_argument("target", type=Path)
    ap.add_argument("--suites", default="unit,mr_inv,mr_eqv,anchored,v1,v2")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    out = {"target": str(a.target),
           "suites": [score_suite(a.task_dir.resolve(), a.target.resolve(), s) for s in a.suites.split(",")]}
    if a.json:
        a.json.write_text(json.dumps(out, indent=2))
    for s in out["suites"]:
        print(f"{s['suite']:8s} score={s['mutation_score']} counts={s['counts']}")
        if s["run_tail"]:
            print(s["run_tail"])


if __name__ == "__main__":
    main()
