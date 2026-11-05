"""Re-evaluate existing runs' solution.py with the current hidden suites.

Usage: python harness/reeval.py [RUN_DIR ...]   (default: all valid runs)
Writes runs/<id>/eval_v2.json; analyze.py prefers it over eval.json.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
from evaluate import evaluate  # noqa: E402


def one(run: Path) -> str:
    m = json.loads((run / "meta.json").read_text())
    ev = evaluate(ROOT / "tasks" / m["task"], run / "solution.py", run / "workspace")
    (run / "eval_v2.json").write_text(json.dumps(ev, indent=2))
    return f"{run.name}: anchored={'DETECTED' if ev['summary'].get('anchored', {}).get('detected') else 'pass'}"


if __name__ == "__main__":
    runs = [Path(p) for p in sys.argv[1:]] or [p.parent for p in sorted((ROOT / "runs").glob("*/meta.json"))
                                                if json.loads(p.read_text()).get("valid")]
    with ProcessPoolExecutor(6) as pool:
        for line in pool.map(one, runs):
            print(line, flush=True)
