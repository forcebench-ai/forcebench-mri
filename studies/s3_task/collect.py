"""Study 3: does the model like the task at hand, and does it know it'll fail? Collect what the model holds.

For every public Forcebench task a graded run answered (the same weights, reasoning effort low), five reads:

1. Prompt end: the residual stream at the last position of the generation prompt, every layer, built the way
   the run's server built it: the model's own chat template with the system prompt, the task, reasoning effort
   low and thinking on (so the prompt ends with ``<think>\\n``).
2. Answer end: the same conversation with the model's graded answer as its reply, at the answer's last token.
   Only the final answer is used; the reasoning that came before it is private and left out.
3. Stated confidence and 4. stated enjoyment, thinking off: the task with a question appended, and the
   probability of each digit 0-9 as the reply's first token.
5. Prompt end with thinking off: the same task at the start of a reply without reasoning, the position study 2
   reads its valence anchors at, so the two can share a scale.

Forward passes only, prefilled in chunks through the model's cache so that long answers fit in memory. Reads 3-5
share their system prompt and task, which are prefilled once. Each task is saved as it finishes (tasks/<id>.npz),
in a fixed shuffled order so that a partial run is a fair sample, and skipped on a rerun; ``--assemble`` then
gathers them into arrays.

    MRI_MODEL_PATH=... uv run python studies/s3_task/collect.py --prompts outputs/s3/prompts.jsonl \
        --run <forcebench>/results/runs/<run_id> --out outputs/s3
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import make_prompt_cache

from forcebench_mri.model import Hooks, Subject, load

EFFORT = "low"  # the graded run's reasoning effort, which the chat template turns into a system instruction
CHUNK = 2048
MIN_FREE_GB = 45
CACHE_LIMIT = 4 << 30  # MLX keeps freed buffers for reuse; sequences of every length would otherwise pile them up
QUESTIONS = {
    "confidence": "Don't solve it yet. How likely is it that your answer would pass the grader's hidden tests? "
    "Reply with one digit, 0 (certain to fail) to 9 (certain to pass).",
    "enjoyment": "Don't solve it yet. How much would you enjoy doing this task? "
    "Reply with one digit, 0 (would hate it) to 9 (would love it).",
}
SITES = ("prompt_end", "answer_end", "prompt_end_nothink")


def free_gb() -> float:
    """Free, inactive and purgeable memory, from vm_stat."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    page = int(re.search(r"page size of (\d+)", out).group(1))
    pages = sum(int(m) for m in re.findall(r"Pages (?:free|inactive|purgeable):\s+(\d+)", out))
    return pages * page / 1e9


def wait_for_memory() -> None:
    while (gb := free_gb()) < MIN_FREE_GB:
        print(f"waiting for memory: {gb:.0f} GB free, need {MIN_FREE_GB}", flush=True)
        time.sleep(60)


def template(subject: Subject, messages: list[dict], **kwargs) -> str:
    return subject.tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)


def turn(task: dict, user: str | None = None) -> list[dict]:
    return [{"role": "system", "content": task["system"]}, {"role": "user", "content": user or task["user"]}]


def prompt_end_text(subject: Subject, task: dict) -> str:
    """The generation prompt exactly as the run's server built it (thinking on, effort low)."""
    return template(subject, turn(task), add_generation_prompt=True, reasoning_effort=EFFORT)


def answer_end_text(subject: Subject, task: dict, answer: str) -> str:
    """The conversation with the model's answer as its reply, ending on the answer's last character."""
    messages = [*turn(task), {"role": "assistant", "content": answer}]
    text = template(subject, messages, add_generation_prompt=False, reasoning_effort=EFFORT)
    end = text.rindex("<|im_end|>")
    if not text[:end].rstrip().endswith(answer.strip()[-200:]):
        raise ValueError("the answer isn't where the template should have put it")
    return text[:end].rstrip()


def no_thinking_text(subject: Subject, task: dict) -> str:
    """The generation prompt with thinking off: the reply starts straight after an empty ``<think>`` block."""
    return template(subject, turn(task), add_generation_prompt=True, enable_thinking=False)


def question_text(subject: Subject, task: dict, question: str) -> str:
    """The task with ``question`` appended, thinking off: the reply's first token is the answer."""
    return template(
        subject, turn(task, task["user"] + "\n\n" + question), add_generation_prompt=True, enable_thinking=False
    )


def shared_prefix_text(subject: Subject, task: dict) -> str:
    """What reads 3-5 have in common: the system prompt and the task, up to the end of the task's text."""
    text = no_thinking_text(subject, task)
    return text[: text.rindex("<|im_end|>")]


def prefill(subject: Subject, tokens: list[int], cache: list) -> tuple[np.ndarray, mx.array]:
    """Run ``tokens`` on from ``cache``, ``CHUNK`` at a time. Returns every layer's residual at the last position
    ([layers, d_model], float16) and the next-token logits."""
    last: dict[int, mx.array] = {}

    def keep(index: int, out: mx.array) -> mx.array:
        last[index] = out[:, -1]
        return out

    subject._hooks = Hooks(fn=keep)  # Subject.run has no cache; the hooks apply to every chunk alike
    try:
        for start in range(0, len(tokens), CHUNK):
            subject.text.model(mx.array(tokens[start : start + CHUNK])[None], cache)
            mx.eval([c.state for c in cache], list(last.values()))
    finally:
        subject._hooks = None
    resid = mx.concatenate([last[i] for i in range(subject.n_layers)], axis=0)
    logits = subject.logit_lens(resid[-1:])[0].astype(mx.float32)
    mx.eval(resid, logits)
    return np.array(resid.astype(mx.float16)), logits


def digit_probs(subject: Subject, logits: mx.array) -> tuple[np.ndarray, float]:
    """P(digit d | reply starts with a digit) for d = 0..9, and the total probability of a digit."""
    ids = [subject.tokenizer.encode(str(d), add_special_tokens=False)[0] for d in range(10)]
    logp = logits - mx.logsumexp(logits)
    p = np.exp(np.array(logp[mx.array(ids)], dtype=np.float64))
    return p / p.sum(), float(p.sum())


def read_task(subject: Subject, task: dict, answer: str) -> dict[str, np.ndarray]:
    enc = lambda text: subject.tokenizer.encode(text, add_special_tokens=False)
    out: dict[str, np.ndarray] = {}
    for site, text in (
        ("prompt_end", prompt_end_text(subject, task)),
        ("answer_end", answer_end_text(subject, task, answer)),
    ):
        tokens = enc(text)
        out[site], _ = prefill(subject, tokens, make_prompt_cache(subject.model))
        out[f"n_{site}"] = np.array(len(tokens))

    prefix = enc(shared_prefix_text(subject, task))
    shared = make_prompt_cache(subject.model)
    prefill(subject, prefix, shared)
    reads = {"prompt_end_nothink": no_thinking_text(subject, task)}
    reads |= {name: question_text(subject, task, q) for name, q in QUESTIONS.items()}
    for name, text in reads.items():
        tokens = enc(text)
        if tokens[: len(prefix)] == prefix:
            resid, logits = prefill(subject, tokens[len(prefix) :], copy.deepcopy(shared))
        else:  # the prefix doesn't end on a token boundary of this text: no sharing
            resid, logits = prefill(subject, tokens, make_prompt_cache(subject.model))
        if name in QUESTIONS:
            out[f"{name}_probs"], mass = digit_probs(subject, logits)
            out[f"{name}_mass"] = np.array(mass)
        else:
            out[name] = resid
            out[f"n_{name}"] = np.array(len(tokens))
    return out


def assemble(rows: list[dict], cases: dict, out: Path, run: str, partial: bool) -> None:
    """Gather the per-task files into arrays, in the prompts' order: acts_<site>.npy, stated.json, index.json."""
    done = [r for r in rows if (out / "tasks" / f"{r['task_id']}.npz").exists()]
    if len(done) < len(rows) and not partial:
        raise SystemExit(f"{len(rows) - len(done)} tasks still to collect (or pass --partial)")
    data = [dict(np.load(out / "tasks" / f"{r['task_id']}.npz")) for r in done]
    for site in SITES:
        np.save(out / f"acts_{site}.npy", np.stack([d[site] for d in data]))
    stated = {
        name: {
            "question": q,
            "digit_probs": [d[f"{name}_probs"].round(6).tolist() for d in data],
            "digit_mass": [round(float(d[f"{name}_mass"]), 6) for d in data],
        }
        for name, q in QUESTIONS.items()
    }
    (out / "stated.json").write_text(json.dumps(stated))
    index = {
        "task_ids": [r["task_id"] for r in done],
        "suites": [r["suite"] for r in done],
        "difficulty": [cases[r["task_id"]]["difficulty"] for r in done],
        "passed": [bool(cases[r["task_id"]]["passed"]) for r in done],
        "run": run,
        "effort": EFFORT,
        "complete": len(done) == len(rows),
        **{f"{site}_tokens": [int(d[f"n_{site}"]) for d in data] for site in SITES},
    }
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"assembled {len(done)}/{len(rows)} tasks into {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=Path, required=True, help="prompts.jsonl from dump_prompts.py")
    ap.add_argument("--run", type=Path, required=True, help="the graded run's directory (labels and answers)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, help="stop after N tasks of the shuffled order (a pilot)")
    ap.add_argument("--assemble", action="store_true", help="only gather the per-task files")
    ap.add_argument("--partial", action="store_true", help="with --assemble: gather what's done so far")
    args = ap.parse_args()
    (args.out / "tasks").mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in args.prompts.read_text().splitlines()]
    bad = [r["task_id"] for r in rows if r.get("check") not in ("run", "manifest")]
    if bad:
        raise SystemExit(f"prompts not verified against the run: {bad}")
    cases = {c["task_id"]: c for c in map(json.loads, (args.run / "cases.jsonl").read_text().splitlines())}
    if args.assemble:
        assemble(rows, cases, args.out, args.run.name, args.partial)
        return

    order = [rows[i] for i in np.random.default_rng(0).permutation(len(rows))][: args.limit]
    todo = [r for r in order if not (args.out / "tasks" / f"{r['task_id']}.npz").exists()]
    wait_for_memory()
    started = time.perf_counter()
    mx.set_cache_limit(CACHE_LIMIT)
    subject = load()
    for i, task in enumerate(todo):
        t0 = time.perf_counter()
        data = read_task(subject, task, cases[task["task_id"]]["output"])
        tmp = args.out / "tasks" / f"{task['task_id']}.tmp.npz"
        np.savez(tmp, **data)
        tmp.rename(args.out / "tasks" / f"{task['task_id']}.npz")
        mx.clear_cache()
        print(
            f"{i + 1}/{len(todo)} {task['task_id']} {int(data['n_answer_end'])} tok "
            f"{time.perf_counter() - t0:.1f}s (total {time.perf_counter() - started:.0f}s, "
            f"peak {mx.get_peak_memory() / 1e9:.1f} GB)",
            flush=True,
        )
    if not args.limit:
        assemble(rows, cases, args.out, args.run.name, partial=False)


if __name__ == "__main__":
    main()
