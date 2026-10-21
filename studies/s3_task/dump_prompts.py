"""Render the public Forcebench tasks' prompts with Forcebench's own code, for study 3.

Run from a copy of Forcebench at the graded run's commit (its ``git_sha``), with an environment that has
Forcebench's dependencies; public suites only, never the private pool:

    git -C "$FORCEBENCH_DIR" archive <git_sha> src suites | tar -x -C <copy>
    cd <copy> && PYTHONPATH=src <forcebench venv>/bin/python <this repo>/studies/s3_task/dump_prompts.py \
        --run-cases <run>/cases.jsonl --out <this repo>/outputs/s3/prompts.jsonl

Each line: task id, suite, version, the system prompt and the rendered user message, and how the rendering
was checked: ``run`` when it matches the ``prompt_sha`` the graded run recorded for the task, ``manifest``
when the run recorded none and it matches that commit's ``suites/prompt-hashes.json``, else ``mismatch``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forcebench.answers import SYSTEM_PROMPT, prompt_sha, render_prompt
from forcebench.tasks import load_suite


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-cases", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cases = {c["task_id"]: c for c in map(json.loads, args.run_cases.read_text().splitlines())}
    manifest = json.loads(Path("suites/prompt-hashes.json").read_text())
    tasks = {}
    for suite_dir in sorted(p for p in Path("suites").iterdir() if (p / "suite.yaml").exists()):
        for task in load_suite(suite_dir).tasks:
            tasks[task.id] = task

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for task_id, case in cases.items():
            task = tasks.get(task_id)
            row = {"task_id": task_id, "suite": case["suite"], "run_version": case["task_version"]}
            if task is None:
                row["missing"] = True
            else:
                sha, recorded = prompt_sha(task), case["prompt_sha"]
                listed = manifest.get(task_id, {})
                if recorded:
                    check = "run" if sha == recorded else "mismatch"
                else:
                    ok = listed.get("prompt_sha") == sha and listed.get("version") == case["task_version"]
                    check = "manifest" if ok else "mismatch"
                row |= {"version": task.version, "system": SYSTEM_PROMPT, "user": render_prompt(task), "check": check}
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
