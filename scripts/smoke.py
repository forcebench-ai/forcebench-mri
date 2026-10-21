"""Smoke test of the instrument: load, logit lens on an Apex vs Java prefix, speed, memory, gradients.

A pilot only: one pair of prompts proves the plumbing, not a finding. The prefixes end at `System`, not
`System.`: this tokenizer has single tokens `.debug` and `.out`, so a prompt ending in a bare `.` would be
an unnatural token boundary.

    MRI_MODEL_PATH=... uv run python scripts/smoke.py --prompts apex_prompts.json --out outputs/smoke
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import stream_generate

from forcebench_mri.model import Hooks, load

CODE = "```{lang}\npublic class Logger {{\n    public static void log(String message) {{\n        System"
ASK = "Write a {name} class with a static method that logs a message."
LANGS = {"apex": "Apex", "java": "Java"}


def first_token(tokenizer, word: str) -> int:
    return tokenizer.encode(word, add_special_tokens=False)[0]


def lens_table(subject, lang: str, ids: dict[str, int], layers: list[int]) -> dict:
    tokens = subject.chat_tokens(ASK.format(name=LANGS[lang]), assistant_prefix=CODE.format(lang=lang))
    subject.run(tokens, Hooks(capture=layers))
    rows = []
    for layer in layers:
        logits = subject.logit_lens(subject.captured[layer][:, -1]).astype(mx.float32)
        probs = mx.softmax(logits, axis=-1)[0]
        top = mx.argsort(-probs)[:5].tolist()
        rows.append(
            {
                "layer": layer,
                "top5": [subject.tokenizer.decode([t]) for t in top],
                **{f"p_{w}": probs[i].item() for w, i in ids.items()},
            }
        )
    return {"lang": lang, "n_tokens": len(tokens), "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=Path, required=True, help="JSON from Forcebench's renderer: system + tasks")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    start = time.perf_counter()
    subject = load()
    mx.eval(subject.model.parameters())
    report["load_s"] = round(time.perf_counter() - start, 1)
    report["n_layers"] = subject.n_layers
    tok = subject.tokenizer

    # Logit lens: Apex vs Java prefix ending in `System`; next token `.debug` (Apex) or `.out` (Java)
    ids = {"debug": first_token(tok, ".debug"), "out": first_token(tok, ".out")}
    report["ids"] = {w: [i, tok.decode([i])] for w, i in ids.items()}
    layers = sorted({*range(0, subject.n_layers, 4), subject.n_layers - 1})
    report["lens"] = [lens_table(subject, lang, ids, layers) for lang in LANGS]

    # Prefill speed on real Forcebench prompts (the longest public Apex task, and the two longest together)
    data = json.loads(args.prompts.read_text())
    longest = sorted(data["tasks"].items(), key=lambda kv: len(kv[1]["prompt"]))[-2:]
    prefill = []
    for label, user in [
        (longest[-1][0], longest[-1][1]["prompt"]),
        ("two longest", longest[-1][1]["prompt"] + "\n\n" + longest[-2][1]["prompt"]),
    ]:
        tokens = subject.chat_tokens(user, system=data["system"])
        mx.eval(subject.run(tokens[:64]))  # warm-up
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        mx.eval(subject.run(tokens)[:, -1])
        dt = time.perf_counter() - t0
        prefill.append(
            {
                "prompt": label,
                "tokens": len(tokens),
                "s": round(dt, 2),
                "tok_per_s": round(len(tokens) / dt),
                "peak_gb": round(mx.get_peak_memory() / 1e9, 1),
            }
        )
    report["prefill"] = prefill

    # Decode speed: ~200 tokens of an answer to the longest task, thinking off
    tokens = subject.chat_tokens(longest[-1][1]["prompt"], system=data["system"])
    last = None
    for last in stream_generate(subject.model, tok, tokens, max_tokens=200):
        pass
    report["decode"] = {
        "tokens": last.generation_tokens,
        "tok_per_s": round(last.generation_tps, 1),
        "prompt_tok_per_s": round(last.prompt_tps),
        "peak_gb": round(last.peak_memory, 1),
    }

    # Gradient of a logit difference w.r.t. a residual-stream perturbation (attribution patching needs this)
    tokens = subject.chat_tokens(ASK.format(name="Apex"), assistant_prefix=CODE.format(lang="apex"))
    layer = subject.n_layers // 2
    subject.run(tokens, Hooks(capture=[layer]))
    shape, dtype = subject.captured[layer].shape, subject.captured[layer].dtype

    def diff(delta: mx.array) -> mx.array:
        logits = subject.run(tokens, Hooks(add={layer: (slice(None), delta)}))
        return logits[0, -1, ids["debug"]] - logits[0, -1, ids["out"]]

    grads = {}
    for mode in ("eval (Metal kernel)", "train (ops fallback)"):
        subject.model.train(mode.startswith("train"))
        try:
            t0 = time.perf_counter()
            g = mx.grad(lambda d: diff(d.reshape(shape)))(mx.zeros(shape, dtype=dtype))
            mx.eval(g)
            grads[mode] = {
                "ok": True,
                "s": round(time.perf_counter() - t0, 1),
                "grad_norm_last_pos": round(mx.linalg.norm(g[0, -1].astype(mx.float32)).item(), 4),
            }
        except Exception as e:  # noqa: BLE001 - report, don't stop: this is what the smoke test is for
            grads[mode] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    subject.model.eval()
    report["grad"] = {"layer": layer, "tokens": len(tokens), **grads}

    (args.out / "smoke.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "lens"}, indent=1))
    for table in report["lens"]:
        print(f"\n{table['lang']} ({table['n_tokens']} tokens)  layer: P(debug) P(out) | top-5")
        for r in table["rows"]:
            print(f"  L{r['layer']:>2}: {r['p_debug']:.3f} {r['p_out']:.3f} | {r['top5']}")


if __name__ == "__main__":
    main()
