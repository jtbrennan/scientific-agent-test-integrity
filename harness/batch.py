"""Run a grid of agent runs in parallel (one subprocess per run).

Usage:
    python harness/batch.py --tasks nls_ses_logit,nls_ses_interaction \
        --conditions all,some,none,conflict --models gpt-5.6-terra,gpt-5.6-luna \
        --prompts explicit,neutral,pressure --nudges 3 \
        --reps 3 [--rep-start 1] [--jobs 6] [--agent codex] [--timeout 1200]

Each run writes runs/<run_id>/ as harness/run_agent.py does; a one-line
summary per run is appended to results/batch_<stamp>.jsonl.
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def one(task: str, cond: str, prompt: str, model: str, rep: int, agent: str, timeout: int, nudges: int) -> dict:
    cmd = [sys.executable, str(ROOT / "harness" / "run_agent.py"), str(ROOT / "tasks" / task),
           "--condition", cond, "--agent", agent, "--model", model, "--rep", str(rep), "--prompt", prompt, "--nudges", str(nudges),
           "--timeout", str(timeout)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    try:
        start = p.stdout.index("{")
        meta = json.loads(p.stdout[start:])
    except (ValueError, json.JSONDecodeError):
        return {"task": task, "condition": cond, "prompt_arm": prompt, "model": model, "rep": rep,
                "valid": False, "invalid_reason": "runner crashed", "stderr": p.stderr[-1500:]}
    review = meta.get("self_review") or {}
    return {k: meta.get(k) for k in ("run_id", "task", "condition", "prompt_arm", "nudges_used", "model", "rep", "valid",
                                     "invalid_reason", "tests_tampered", "gamed_by_construction",
                                     "eval_summary")} | {
        "agent_seconds": meta["agent"]["seconds"], "agent_tokens": meta["agent"]["tokens"],
        "review_tokens": review.get("tokens")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--conditions", default="all,some,none,conflict")
    ap.add_argument("--models", default="gpt-5.6-terra")
    ap.add_argument("--prompts", default="explicit")
    ap.add_argument("--nudges", type=int, default=0)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--rep-start", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--agent", default="codex")
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip cells that already have a valid run in runs/")
    a = ap.parse_args()
    grid = list(itertools.product(a.tasks.split(","), a.conditions.split(","), a.prompts.split(","),
                                  a.models.split(","), range(a.rep_start, a.rep_start + a.reps)))
    if a.skip_existing:
        have = set()
        for f in (ROOT / "runs").glob("*/meta.json"):
            m = json.loads(f.read_text())
            if m.get("valid"):
                have.add((m["task"], m["condition"], m.get("prompt_arm", "explicit"), m["model"], m["rep"]))
        grid = [g for g in grid if g not in have]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "results" / f"batch_{stamp}.jsonl"
    print(f"{len(grid)} runs, {a.jobs} in parallel -> {out}", flush=True)
    with ThreadPoolExecutor(a.jobs) as pool:
        futs = [pool.submit(one, t, c, p, m, r, a.agent, a.timeout, a.nudges) for t, c, p, m, r in grid]
        for f in as_completed(futs):
            if f.cancelled():
                continue
            row = f.result()
            if "quota" in str(row.get("invalid_reason") or "").lower():
                # billing/quota exhausted: every further run would be invalid too
                for g in futs:
                    g.cancel()
                print("QUOTA EXCEEDED - cancelling the remaining runs", flush=True)
            with out.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(json.dumps({k: row.get(k) for k in ("task", "condition", "prompt_arm", "nudges_used", "model", "rep", "valid",
                                                      "gamed_by_construction", "eval_summary")}), flush=True)


if __name__ == "__main__":
    main()
