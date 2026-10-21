"""Study 2: clicks or code. Does the model reach for code when clicks would do?

Every requirement in stimuli.yaml has a textbook answer: a standard declarative feature, a flow, or Apex. With
reasoning off, the model is asked how it would build each one, three ways, by three askers (nobody named, an
admin, a developer):

- feature: name the one Salesforce feature it would use (greedy, no options offered)
- sentence: say in one sentence how it would build it (greedy, no options offered)
- choice: clicks or code, in one word, in both orders. The chance of each answer is read from the next-token
  distribution and the two orders are averaged, so a lean towards whichever is named first cancels out.

A reply counts as clicks, code, both (a flow that calls Apex, say) or neither by the features it names; it reaches
for code when it names any. For the one-word choice, the logit lens reads the log-odds of "code" over "clicks"
after every layer.

    uv run python studies/s2_clicks/run.py --out outputs/s2
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import matplotlib
import mlx.core as mx
import numpy as np
import yaml

from forcebench_mri.model import Hooks, Subject, load

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
GROUPS = ("standard", "flow", "apex")
COLOURS = {"standard": "#2a9d6e", "flow": "#2c66d4", "apex": "#d23f5c"}
MAX_TOKENS = {"feature": 24, "sentence": 80}

# Features by kind. "No code" and the like are clicks; "record-triggered" and a flow that "triggers on" are not a
# trigger. Platform events and the Bulk API are neither: a flow or Apex can use them.
NO_CODE_RE = re.compile(r"no[- ](?:custom )?code|without (?:\w+ ){0,3}code|low[- ]code", re.IGNORECASE)
CLICKS_RE = re.compile(
    r"declarative|\bclicks?\b|\bflows?\b|validation rule|formula|roll-?up summary|approval (?:process|step)|"
    r"assignment rule|escalation rule|duplicate (?:rule|management)|matching rule|sharing rule|sharing setting|"
    r"organi[sz]ation-wide default|territory management|required field|page layout|field-level|workflow|"
    r"process builder|auto-?response rule|email alert|lookup filter|dynamic form|record type|scheduled path",
    re.IGNORECASE,
)
CODE_RE = re.compile(
    r"\bapex\b|(?<!that )(?<!which )(?<!flow )(?<!builder )(?<!it )\btriggers?\b|lightning web component|"
    r"\blwc\b|\bbatch(?:able)?\b|queueable|schedulable|\bclass\b|\bcode\b|custom rest|rest resource",
    re.IGNORECASE,
)
# Tools Salesforce has retired in favour of Flow (no support since the end of 2025).
RETIRED_RE = re.compile(r"process builder|workflow rules?|time-dependent workflow", re.IGNORECASE)


def sort_reply(text: str) -> str:
    """'clicks', 'code', 'both' or 'neither', by the features the reply names."""
    text = NO_CODE_RE.sub(" clicks ", text)
    clicks, code = CLICKS_RE.search(text) is not None, CODE_RE.search(text) is not None
    return "both" if clicks and code else "code" if code else "clicks" if clicks else "neither"


def chat_text(subject: Subject, user: str) -> str:
    """The chat template's text up to the start of the assistant's reply, thinking off."""
    return subject.tokenizer.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def encode(subject: Subject, text: str) -> list[int]:
    return subject.tokenizer.encode(text, add_special_tokens=False)


def first_token(subject: Subject, base: str, continuation: str) -> int | None:
    """The first token ``continuation`` adds after ``base``, or None when ``base`` doesn't end on a token boundary."""
    pre, full = encode(subject, base), encode(subject, base + continuation)
    if full[: len(pre)] != pre or len(full) == len(pre):
        return None
    return full[len(pre)]


def answer_ids(subject: Subject, words: dict[str, str]) -> dict[str, list[int]]:
    """For each option, the first tokens of its one-word answer as the reply may start it: as given, capitalised,
    in capitals. The two options must share none."""
    base = chat_text(subject, "Answer with one word.")
    ids: dict[str, list[int]] = {}
    for k, w in words.items():
        found = (first_token(subject, base, v) for v in (w, w.capitalize(), w.upper()))
        ids[k] = sorted({t for t in found if t is not None})
    a, b = ids.values()
    assert a and b and not set(a) & set(b), f"answer tokens overlap: {ids}"
    return ids


def read_last(subject: Subject, tokens: list[int]) -> tuple[mx.array, mx.array]:
    """The residual stream at the last position after every layer ([layers, d_model]) and the log-probabilities of
    the next token ([vocab])."""
    last: dict[int, mx.array] = {}

    def keep(index: int, out: mx.array) -> mx.array:
        last[index] = out[:, -1]
        return out

    logits = subject.run(tokens, Hooks(fn=keep))
    resid = mx.concatenate([last[i] for i in range(subject.n_layers)], axis=0)
    final = logits[0, -1].astype(mx.float32)
    return resid, final - mx.logsumexp(final)


def boot_ci(values, rng: np.random.Generator, n: int = 2000, stat=np.mean) -> tuple:
    """95% percentile bootstrap interval of ``stat`` over the first axis."""
    values = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    draws = np.stack([stat(values[i], axis=0) for i in idx])
    return np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)


def summary(values, rng: np.random.Generator) -> dict:
    values = np.asarray(values, dtype=float)
    lo, hi = boot_ci(values, rng)
    return {"n": len(values), "mean": float(values.mean()), "ci": [float(lo), float(hi)]}


# ------------------------------------------------------------------------------------------------ asking
def reply(subject: Subject, user: str, max_tokens: int, stop: list[int]) -> str:
    tokens = encode(subject, chat_text(subject, user))
    return subject.tokenizer.decode(subject.generate(tokens, max_tokens=max_tokens, stop=stop)).strip()


def choose(subject: Subject, template: str, fields: dict, options: dict, ids: dict[str, list[int]]) -> dict:
    """Code over clicks in one word, in both orders: the log-odds at the output and through the logit lens."""
    orders = []
    for a, b in (("clicks", "code"), ("code", "clicks")):
        user = template.format(
            a=options[a]["text"], b=options[b]["text"], a_word=options[a]["word"], b_word=options[b]["word"], **fields
        )
        resid, logp = read_last(subject, encode(subject, chat_text(subject, user)))
        lens = subject.logit_lens(resid).astype(mx.float32)
        lens = lens - mx.logsumexp(lens, axis=-1, keepdims=True)
        code, clicks = mx.array(ids["code"]), mx.array(ids["clicks"])
        layers = mx.logsumexp(lens[:, code], axis=-1) - mx.logsumexp(lens[:, clicks], axis=-1)
        final = mx.logsumexp(logp[code]) - mx.logsumexp(logp[clicks])
        mass = mx.exp(mx.logsumexp(logp[mx.concatenate([code, clicks])]))
        mx.eval(layers, final, mass)
        orders.append({"first": a, "lo_code": final.item(), "mass": mass.item(), "lens": np.array(layers).tolist()})
    lo = float(np.mean([o["lo_code"] for o in orders]))
    return {
        "lo_code": lo,
        "p_code": float(np.mean([1 / (1 + np.exp(-o["lo_code"])) for o in orders])),
        "lens": np.mean([o["lens"] for o in orders], axis=0).round(4).tolist(),
        "orders": [{k: v for k, v in o.items() if k != "lens"} for o in orders],
    }


def run(subject: Subject, stim: dict, limit: int | None) -> list[dict]:
    tok = subject.tokenizer
    stop = [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
    ids = answer_ids(subject, {k: v["word"] for k, v in stim["options"].items()})
    items = []
    for group in GROUPS:
        for req in stim["requirements"][group][:limit]:
            item: dict = {"group": group, "req": req, "by_asker": {}}
            for asker, line in stim["askers"].items():
                fields = {"asker": line, "req": req}
                got: dict = {}
                for prompt in ("feature", "sentence"):
                    text = reply(subject, stim["prompts"][prompt].format(**fields), MAX_TOKENS[prompt], stop)
                    got[prompt] = {"text": text, "sorted": sort_reply(text)}
                got["choice"] = choose(subject, stim["prompts"]["choice"], fields, stim["options"], ids)
                item["by_asker"][asker] = got
            items.append(item)
            print(f"{group:8} {item['by_asker']['none']['feature']['text'][:50]!r:54} {req[:60]}", flush=True)
    return items


# ------------------------------------------------------------------------------------------------ summaries
def feature_name(text: str) -> str:
    """A reply to the feature prompt as a short name: its first line, without markdown or a final full stop."""
    line = text.strip().splitlines()[0] if text.strip() else ""
    return re.sub(r"[*_`#]", "", line).strip().rstrip(".").strip()


def summarise(items: list[dict], askers: list[str], rng: np.random.Generator) -> dict:
    def of(group: str) -> list[dict]:
        return [i for i in items if group in ("all", i["group"]) or (group == "clicks" and i["group"] != "apex")]

    out: dict = {"reach": {}, "choice": {}, "lens": {}, "shift": {}, "reach_shift": {}, "features": {}, "agreement": {}}
    for prompt in ("feature", "sentence"):
        for asker in askers:
            for group in (*GROUPS, "clicks", "all"):
                got = [i["by_asker"][asker][prompt] for i in of(group)]
                sorted_ = [g["sorted"] for g in got]
                out["reach"].setdefault(prompt, {}).setdefault(asker, {})[group] = {
                    "code": summary([s in ("code", "both") for s in sorted_], rng),
                    "code_only": float(np.mean([s == "code" for s in sorted_])),
                    **{k: float(np.mean([s == k for s in sorted_])) for k in ("both", "clicks", "neither")},
                    "retired": float(np.mean([RETIRED_RE.search(g["text"]) is not None for g in got])),
                }
    for asker in askers:
        for group in (*GROUPS, "clicks", "all"):
            ch = [i["by_asker"][asker]["choice"] for i in of(group)]
            out["choice"].setdefault(asker, {})[group] = {
                "p_code": summary([c["p_code"] for c in ch], rng),
                "lo_code": summary([c["lo_code"] for c in ch], rng),
                "share_code": float(np.mean([c["lo_code"] > 0 for c in ch])),
            }
            lens = np.array([c["lens"] for c in ch])
            lo, hi = boot_ci(lens, rng, stat=np.median)
            out["lens"].setdefault(asker, {})[group] = {
                "median": np.median(lens, axis=0).round(4).tolist(),
                "lo": np.round(lo, 4).tolist(),
                "hi": np.round(hi, 4).tolist(),
                "share_code_ahead": (lens > 0).mean(axis=0).round(4).tolist(),
            }
    # Who is asking: paired over requirements, the developer's answer minus the admin's, and each minus nobody's.
    for group in (*GROUPS, "clicks", "all"):
        p = {a: np.array([i["by_asker"][a]["choice"]["p_code"] for i in of(group)]) for a in askers}
        out["shift"][group] = {
            "developer_minus_admin": summary(p["developer"] - p["admin"], rng),
            "admin_minus_none": summary(p["admin"] - p["none"], rng),
            "developer_minus_none": summary(p["developer"] - p["none"], rng),
        }
    # The same for the unprompted answers: did it reach for code (1) or not (0), paired over requirements.
    for prompt in ("feature", "sentence"):
        for group in (*GROUPS, "clicks", "all"):
            r = {
                a: np.array([i["by_asker"][a][prompt]["sorted"] in ("code", "both") for i in of(group)]) for a in askers
            }
            out["reach_shift"].setdefault(prompt, {})[group] = {
                "developer_minus_admin": summary(r["developer"] * 1.0 - r["admin"], rng),
                "admin_minus_none": summary(r["admin"] * 1.0 - r["none"], rng),
                "developer_minus_none": summary(r["developer"] * 1.0 - r["none"], rng),
            }
    for group in GROUPS:
        names = Counter(feature_name(i["by_asker"]["none"]["feature"]["text"]) for i in of(group))
        out["features"][group] = names.most_common()
    # How often each reading picks the textbook side (clicks for standard and flow, code for apex).
    for asker in askers:
        right = {
            prompt: [
                (i["by_asker"][asker][prompt]["sorted"] in ("code", "both")) == (i["group"] == "apex") for i in items
            ]
            for prompt in ("feature", "sentence")
        }
        right["choice"] = [(i["by_asker"][asker]["choice"]["lo_code"] > 0) == (i["group"] == "apex") for i in items]
        out["agreement"][asker] = {k: float(np.mean(v)) for k, v in right.items()}
    orders = [o for i in items for a in askers for o in i["by_asker"][a]["choice"]["orders"]]
    out["min_mass"] = float(min(o["mass"] for o in orders))
    out["median_mass"] = float(np.median([o["mass"] for o in orders]))
    # The lean towards whichever option is named first (it cancels in the averaged answers).
    out["first_named_lo"] = float(np.mean([o["lo_code"] if o["first"] == "code" else -o["lo_code"] for o in orders]))
    return out


# ------------------------------------------------------------------------------------------------ figures
def plot(results: dict, out: Path) -> list[Path]:
    s = results["summary"]
    paths = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    reads = [("feature", "names one feature"), ("sentence", "one sentence"), ("choice", "clicks or code, one word")]
    for ax, (read, title) in zip(axes, reads, strict=True):
        for a_i, asker in enumerate(("none", "admin", "developer")):
            for g_i, group in enumerate(GROUPS):
                d = s["choice"][asker][group]["p_code"] if read == "choice" else s["reach"][read][asker][group]["code"]
                x = g_i * 4 + a_i
                ax.bar(x, d["mean"], color=COLOURS[group], alpha=(0.95, 0.6, 0.35)[a_i], width=0.9)
                ax.errorbar(
                    x, d["mean"], yerr=[[d["mean"] - d["ci"][0]], [d["ci"][1] - d["mean"]]], color="k", capsize=2
                )
        ax.set_xticks([1, 5, 9], [f"textbook: {g}" for g in GROUPS], fontsize=8)
        ax.set_title(f"{title}\n(bars: nobody, admin, developer)", fontsize=9)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("share reaching for code (choice: P(code))")
    fig.suptitle("Clicks or code, by textbook answer and asker (95% CI over requirements)", fontsize=10)
    fig.tight_layout()
    paths.append(out / "s2_reach.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.4))
    for group in GROUPS:
        d = s["lens"]["none"][group]
        xs = np.arange(len(d["median"]))
        ax.plot(xs, d["median"], color=COLOURS[group], label=f"textbook: {group}")
        ax.fill_between(xs, d["lo"], d["hi"], color=COLOURS[group], alpha=0.15)
    ax.axhline(0, color="#555", lw=0.8)
    ax.set_xlabel("layer")
    ax.set_ylabel("log-odds of 'code' over 'clicks' (median)")
    ax.set_title("Logit lens on the one-word answer, nobody named (95% CI of the median)", fontsize=10)
    ax.legend()
    fig.tight_layout()
    paths.append(out / "s2_lens.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimuli", type=Path, default=HERE / "stimuli.yaml")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, help="the first N requirements of each group only (a smoke test)")
    ap.add_argument("--plot-only", action="store_true", help="redraw the figures from s2_results.json")
    ap.add_argument("--resort", action="store_true", help="sort the saved replies again, then redo the summary")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "s2_results.json"
    stim = yaml.safe_load(args.stimuli.read_text())
    if args.plot_only or args.resort:
        results = json.loads(path.read_text())
        if args.resort:
            for item in results["items"]:
                for got in item["by_asker"].values():
                    for prompt in ("feature", "sentence"):
                        got[prompt]["sorted"] = sort_reply(got[prompt]["text"])
            results["summary"] = summarise(results["items"], list(stim["askers"]), np.random.default_rng(0))
            path.write_text(json.dumps(results, indent=1))
        for p in plot(results, args.out):
            print(p)
        return
    started = time.perf_counter()
    subject = load()
    items = run(subject, stim, args.limit)
    results = {
        "model": "Qwen3.8 27B (MLX 8-bit)",
        "n_layers": subject.n_layers,
        "runtime_s": round(time.perf_counter() - started, 1),
        "items": items,
        "summary": summarise(items, list(stim["askers"]), np.random.default_rng(0)),
    }
    path.write_text(json.dumps(results, indent=1))
    for p in plot(results, args.out):
        print(p)
    print(path, f"{results['runtime_s']} s")


if __name__ == "__main__":
    main()
