"""Study 1, phase B: is the Java lead "thinking in Java", or the common idiom drafted first?

Phase A found that in Apex code the logit lens favours the Java token at layers 49-55, and Apex wins only from
layer ~58. Four checks on the same minimal pairs tell the readings apart:

B1 neutral: the request names no language and the code fence is plain. Where the Apex context's mid layers read
   like the neutral one, the middle of the network drafts the idiom without regard to the language.
B2 probe: a linear probe on the last position's residual stream, Apex context against Java context, at every layer
   (pairs held out). It shows when the language can be read there, which phase A's lens can't: knowing it is not
   the same as using it for the next token.
B3 patching: one context's residual stream written into the other's run at one layer, at the last position only or
   over the whole code span, both ways. The share of the final log-odds that moves says where the language
   decision lives at that depth.
B4 steering: the mean Apex-minus-Java difference at the last position, added at one layer, on the pairs and on
   generated Apex tests (Java-isms counted), against a random direction of the same length.

    MRI_MODEL_PATH=... uv run python studies/s1_latent_java/phase_b.py pairs --out outputs/s1b
    MRI_MODEL_PATH=... uv run python studies/s1_latent_java/phase_b.py generation --out outputs/s1b
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import matplotlib
import mlx.core as mx
import numpy as np
import run as phase_a
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from forcebench_mri.model import Hooks, Subject, load

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
CONTEXTS = ("apex", "java", "neutral")
PATCH_LAYERS = sorted({*range(0, 64, 4), 50, 54, 58, 62, 63})
STEER_LAYERS = [32, 44, 50, 56]
STEER_SCALES = [1.0, 2.0]

# Things an Apex compiler rejects or that only Java writes, counted in generated Apex code.
JAVA_ISMS = {
    "System.out": r"System\.out\.",
    "JUnit Assert.assert*": r"\bAssert\.assert[A-Z]\w*",
    "JUnit Assertions.": r"\bAssertions\.",
    "@Test": r"@Test\b",
    "@Override": r"@Override\b",
    "import": r"(?m)^\s*import\s",
    "int": r"\bint\s+\w+\s*[=;:]",
    "double-quoted string": r'"[^"\n]*"',
    "throws": r"\bthrows\b",
    "ArrayList/HashMap/HashSet": r"\b(?:ArrayList|HashMap|HashSet)\b",
    "main": r"public\s+static\s+void\s+main",
}


def neutral_ask(ask: str) -> str:
    """The request with no language named ("Write an Apex class" becomes "Write a class")."""
    return ask.replace("{a_lang}", "a")


def prepare(subject: Subject, pair: dict) -> tuple[dict | None, str]:
    """Phase A's two contexts plus the neutral one (None when its token boundaries differ)."""
    prep, why = phase_a.prepare(subject, pair)
    if prep is None:
        return None, why
    tok = subject.tokenizer
    base = phase_a.chat_text(subject, neutral_ask(pair["ask"])) + "```\n" + pair["prefix"]
    got = {s: {phase_a.first_token(tok, base, c) for c in pair[s]} for s in ("apex", "java")}
    same = all(got[s] == set(prep["ids"][s]) for s in got)
    prep["tokens"]["neutral"] = tok.encode(base, add_special_tokens=False) if same else None
    return prep, ""


def run_all(subject: Subject, tokens: list[int]) -> tuple[list[mx.array], mx.array]:
    """Every layer's output over the whole sequence, and the final log-probabilities at the last position."""
    outs: dict[int, mx.array] = {}

    def keep(index: int, out: mx.array) -> mx.array:
        outs[index] = out
        return out

    logits = subject.run(tokens, Hooks(fn=keep))
    final = logits[0, -1].astype(mx.float32)
    layers = [outs[i] for i in range(subject.n_layers)]
    mx.eval(layers, final)
    return layers, final - mx.logsumexp(final)


def final_log_odds(logp: mx.array, ids: dict) -> float:
    """log P(Apex continuation) - log P(Java continuation) from final log-probabilities."""
    a = mx.logsumexp(logp[mx.array(ids["apex"])])
    j = mx.logsumexp(logp[mx.array(ids["java"])])
    return float((a - j).item())


def run_log_odds(subject: Subject, tokens: list[int], ids: dict, hooks: Hooks) -> float:
    logits = subject.run(tokens, hooks)[0, -1].astype(mx.float32)
    return final_log_odds(logits - mx.logsumexp(logits), ids)


def lens_log_odds(subject: Subject, last: mx.array, ids: dict) -> np.ndarray:
    """The logit lens's log-odds at every layer, from the last position's residuals ([layers, d_model])."""
    lens = subject.logit_lens(last).astype(mx.float32)
    logp = lens - mx.logsumexp(lens, axis=-1, keepdims=True)
    a = mx.logsumexp(logp[:, mx.array(ids["apex"])], axis=-1)
    j = mx.logsumexp(logp[:, mx.array(ids["java"])], axis=-1)
    return np.array(a - j)


def common_suffix(a: list[int], b: list[int]) -> int:
    """How many trailing tokens two sequences share: the code span, since only the request and the fence differ."""
    n = 0
    while n < min(len(a), len(b)) and a[-1 - n] == b[-1 - n]:
        n += 1
    return n


def patch_sweep(subject: Subject, tokens: dict, layers_by_ctx: dict, ids: dict, base: dict) -> dict:
    """For each direction (target <- source) and layer, the final log-odds after patching the last position or
    the code span, as a share of the gap between the two unpatched runs (0: no effect, 1: fully the source's)."""
    span = common_suffix(tokens["apex"], tokens["java"])
    out = {}
    for target, source in (("apex", "java"), ("java", "apex")):
        gap = base[source] - base[target]
        n_t = len(tokens[target])
        rows = {"last": [], "span": []}
        for layer in PATCH_LAYERS:
            src = layers_by_ctx[source][layer]
            last = Hooks(replace={layer: ([n_t - 1], src[:, -1:])})
            code = Hooks(replace={layer: (list(range(n_t - span, n_t)), src[:, -span:])})
            for name, hooks in (("last", last), ("span", code)):
                rows[name].append((run_log_odds(subject, tokens[target], ids, hooks) - base[target]) / gap)
        out[f"{target}<-{source}"] = rows
    out["span_tokens"] = span
    return out


def probe_by_layer(apex: np.ndarray, java: np.ndarray, neutral: np.ndarray, neutral_mask: np.ndarray) -> dict:
    """Held-out accuracy of a linear probe for the context's language at each layer (8 folds, a pair's two
    contexts always in the same fold), and the probe's mean P(Apex) on the neutral context."""
    n_pairs, n_layers, _ = apex.shape
    groups = np.concatenate([np.arange(n_pairs), np.arange(n_pairs)])
    y = np.concatenate([np.ones(n_pairs), np.zeros(n_pairs)])
    acc, p_neutral = [], []
    for layer in range(n_layers):
        x = np.concatenate([apex[:, layer], java[:, layer]])
        right = 0
        for train, test in GroupKFold(n_splits=min(8, n_pairs)).split(x, y, groups):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000))
            clf.fit(x[train], y[train])
            right += int((clf.predict(x[test]) == y[test]).sum())
        acc.append(right / len(y))
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000)).fit(x, y)
        p_neutral.append(float(clf.predict_proba(neutral[neutral_mask, layer])[:, 1].mean()))
    return {"accuracy": acc, "neutral_p_apex": p_neutral}


def code_blocks(text: str) -> str:
    """The fenced code in a reply (all of it when there is no fence), comments removed."""
    blocks = re.findall(r"```[a-zA-Z]*\n(.*?)(?:```|$)", text, flags=re.DOTALL)
    code = "\n".join(blocks) if blocks else text
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", code)


def java_isms(text: str) -> dict[str, int]:
    code = code_blocks(text)
    return {name: len(re.findall(pattern, code)) for name, pattern in JAVA_ISMS.items()}


def clean_nan(x):
    """NaN (an undefined correlation, e.g. where a layer reads the same for every pair) as JSON null."""
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, dict):
        return {k: clean_nan(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean_nan(v) for v in x]
    return x


def plot(results: dict, out: Path) -> list[Path]:
    paths = []
    layers = np.arange(results["n_layers"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    for ctx, colour in (("apex", "#2c66d4"), ("java", "#d23f5c"), ("neutral", "#666666")):
        s = results["b1"]["median"][ctx]
        ax.plot(layers, s, color=colour, label=f"{ctx} context (median)")
    ax.axhline(0, color="#555", lw=0.8)
    ax.set_xlabel("layer")
    ax.set_ylabel("log P(Apex token) - log P(Java token)")
    ax.set_title(f"B1: logit lens, three contexts (n={results['b1']['n_pairs']} pairs)", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(layers, results["b2"]["accuracy"], color="black", label="probe: held-out accuracy (Apex vs Java context)")
    ax.plot(layers, results["b2"]["neutral_p_apex"], color="#666666", ls="--", label="probe on neutral: mean P(Apex)")
    share = results["b1"]["apex_leads_share"]["apex"]
    ax.plot(layers, share, color="#2c66d4", alpha=0.7, label="lens, Apex context: share where Apex leads")
    ax.axhline(0.5, color="#999", lw=0.8)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("layer")
    ax.set_title("B2: knowing the language vs using it for the next token", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    paths.append(out / "s1b_neutral_and_probe.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, key in zip(axes, ("apex<-java", "java<-apex"), strict=True):
        for where, colour in (("last", "#d23f5c"), ("span", "#2c66d4")):
            med = results["b3"]["median"][key][where]
            lo, hi = results["b3"]["median_ci"][key][where]
            ax.plot(PATCH_LAYERS, med, marker="o", ms=3, color=colour, label=f"patch {where}")
            ax.fill_between(PATCH_LAYERS, lo, hi, color=colour, alpha=0.15)
        ax.axhline(0, color="#555", lw=0.8)
        ax.axhline(1, color="#555", lw=0.8, ls=":")
        ax.set_title(f"B3: {key} (share of the final log-odds gap moved)", fontsize=10)
        ax.set_xlabel("layer patched")
    axes[0].set_ylabel("median normalised effect")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    paths.append(out / "s1b_patching.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)
    return paths


def measure(args: argparse.Namespace) -> None:
    """B1-B4a over the minimal pairs."""
    rng = np.random.default_rng(0)
    started = time.perf_counter()

    subject = load()
    pairs = yaml.safe_load(args.pairs.read_text())["pairs"]
    n_layers = subject.n_layers

    rows, last, preps = [], {c: [] for c in CONTEXTS}, {}
    dropped = []
    for pair in pairs:
        prep, why = prepare(subject, pair)
        if prep is None:
            dropped.append({"id": pair["id"], "why": why})
            continue
        preps[pair["id"]] = prep
        tokens, ids = prep["tokens"], prep["ids"]
        layers_by_ctx, base, lens = {}, {}, {}
        for ctx in CONTEXTS:
            if tokens[ctx] is None:
                continue
            layers, logp = run_all(subject, tokens[ctx])
            layers_by_ctx[ctx] = layers
            base[ctx] = final_log_odds(logp, ids)
            stack = mx.concatenate([layer[:, -1] for layer in layers], axis=0)
            lens[ctx] = lens_log_odds(subject, stack, ids).round(4).tolist()
            last[ctx].append(np.array(stack.astype(mx.float32)))
        if tokens["neutral"] is None:
            last["neutral"].append(np.zeros_like(last["apex"][-1]))
        row = {
            "id": pair["id"],
            "kind": pair["kind"],
            "has_neutral": tokens["neutral"] is not None,
            "final_log_odds": base,
            "lens_log_odds": lens,
            "patch": patch_sweep(subject, tokens, layers_by_ctx, ids, base),
        }
        rows.append(row)
        print(f"{pair['id']}: patched ({time.perf_counter() - started:.0f}s)", flush=True)

    apex, java, neutral = (np.stack(last[c]) for c in CONTEXTS)
    has_neutral = np.array([r["has_neutral"] for r in rows])

    # B1: the three contexts' lens readings, over the pairs that have all three.
    b1_rows = [r for r in rows if r["has_neutral"]]
    lo = {c: np.array([r["lens_log_odds"][c] for r in b1_rows]) for c in CONTEXTS}
    b1 = {
        "n_pairs": len(b1_rows),
        "median": {c: np.median(lo[c], axis=0).tolist() for c in CONTEXTS},
        "apex_leads_share": {c: (lo[c] > 0).mean(axis=0).tolist() for c in CONTEXTS},
        "apex_minus_neutral_median": np.median(lo["apex"] - lo["neutral"], axis=0).tolist(),
        "java_minus_neutral_median": np.median(lo["java"] - lo["neutral"], axis=0).tolist(),
        # Across pairs, how well the neutral reading predicts the Apex context's at each layer.
        "corr_apex_neutral": [float(np.corrcoef(lo["apex"][:, i], lo["neutral"][:, i])[0, 1]) for i in range(n_layers)],
        "neutral_final_prefers_java_share": float(np.mean([r["final_log_odds"]["neutral"] < 0 for r in b1_rows])),
        "by_kind_neutral_final_median": {
            k: float(np.median([r["final_log_odds"]["neutral"] for r in b1_rows if r["kind"] == k]))
            for k in dict.fromkeys(r["kind"] for r in b1_rows)
        },
    }

    # B2: the probe.
    b2 = probe_by_layer(apex, java, neutral, has_neutral)

    # B3: patching, median normalised effect with bootstrap intervals over pairs.
    b3 = {"layers": PATCH_LAYERS, "median": {}, "median_ci": {}}
    for key in ("apex<-java", "java<-apex"):
        b3["median"][key], b3["median_ci"][key] = {}, {}
        for where in ("last", "span"):
            m = np.array([r["patch"][key][where] for r in rows])
            b3["median"][key][where] = np.median(m, axis=0).tolist()
            ci = phase_a.boot_ci(m, np.median, args.boot, rng)
            b3["median_ci"][key][where] = [ci[0].tolist(), ci[1].tolist()]

    # B4a: steering on the pairs. The direction leaves out the pair it is tested on.
    diff = apex - java  # [pairs, layers, d_model]
    total = diff.sum(axis=0)
    np.savez(args.out / "s1b_apex_minus_java.npz", mean=total / len(rows))
    big = max(STEER_SCALES)
    b4_pairs = {"layers": STEER_LAYERS, "scales": STEER_SCALES, "rows": []}
    for i, row in enumerate(rows):
        tokens, ids = preps[row["id"]]["tokens"], preps[row["id"]]["ids"]
        entry = {"id": row["id"], "kind": row["kind"], "effects": {}}
        for layer in STEER_LAYERS:
            d = (total[layer] - diff[i, layer]) / (len(rows) - 1)
            noise = rng.standard_normal(d.shape)
            noise *= np.linalg.norm(d) / np.linalg.norm(noise)
            runs = [(ctx, f"apex_dir/x{s:g}", s * d) for ctx in ("apex", "java") for s in STEER_SCALES]
            runs += [(ctx, f"random/x{big:g}", big * noise) for ctx in ("apex", "java")]
            # Subtracting the direction in the Apex context: does the Java idiom come back?
            runs.append(("apex", f"minus_apex_dir/x{big:g}", -big * d))
            for ctx, name, vec in runs:
                hooks = Hooks(add={layer: ([len(tokens[ctx]) - 1], mx.array(vec))})
                moved = run_log_odds(subject, tokens[ctx], ids, hooks) - row["final_log_odds"][ctx]
                entry["effects"][f"{ctx}/L{layer}/{name}"] = moved
        b4_pairs["rows"].append(entry)
    keys = list(b4_pairs["rows"][0]["effects"])
    b4_pairs["median_effect"] = {k: float(np.median([e["effects"][k] for e in b4_pairs["rows"]])) for k in keys}
    b4_pairs["apex_wins"] = {}
    for k in keys:
        ctx = k.split("/")[0]
        before = np.array([r["final_log_odds"][ctx] for r in rows])
        after = before + np.array([e["effects"][k] for e in b4_pairs["rows"]])
        b4_pairs["apex_wins"][k] = {"before": int((before > 0).sum()), "after": int((after > 0).sum())}

    results = {
        "model": "Qwen3.8 27B (MLX 8-bit)",
        "n_layers": n_layers,
        "n_pairs": len(rows),
        "dropped": dropped,
        "b1": b1,
        "b2": b2,
        "b3": b3,
        "b4_pairs": b4_pairs,
        "pairs": rows,
        "runtime_s": round(time.perf_counter() - started, 1),
    }
    (args.out / "s1b_results.json").write_text(json.dumps(clean_nan(results), indent=1))
    for p in plot(results, args.out):
        print(p)
    print(args.out / "s1b_results.json")


def generation(args: argparse.Namespace) -> None:
    """B4b: generated Apex tests, with the saved direction added at every position of prompt and reply."""
    started = time.perf_counter()
    rng = np.random.default_rng(1)
    subject = load()
    tok = subject.tokenizer
    gen = yaml.safe_load(args.gen.read_text())
    layer = gen["layer"]
    d = np.load(args.out / "s1b_apex_minus_java.npz")["mean"][layer]
    noise = rng.standard_normal(d.shape)
    noise *= np.linalg.norm(d) / np.linalg.norm(noise)
    conditions = {"none": None, "+apex_dir": d, "-apex_dir": -d, "random": noise}
    stop = [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
    b4_gen = {"layer": layer, "scale": gen["scale"], "prompts": []}
    for item in gen["prompts"]:
        tokens = subject.chat_tokens(item["ask"])
        entry = {"id": item["id"], "conditions": {}}
        for name, vec in conditions.items():
            hooks = None if vec is None else Hooks(add={layer: (slice(None), mx.array(gen["scale"] * vec))})
            text = tok.decode(subject.generate(tokens, hooks, max_tokens=args.max_tokens, stop=stop))
            entry["conditions"][name] = {"java_isms": java_isms(text), "text": text}
        b4_gen["prompts"].append(entry)
        print(f"generated {item['id']} ({time.perf_counter() - started:.0f}s)", flush=True)
    b4_gen["totals"] = {
        name: {
            "java_isms": int(sum(sum(p["conditions"][name]["java_isms"].values()) for p in b4_gen["prompts"])),
            "replies_with_any": int(sum(any(p["conditions"][name]["java_isms"].values()) for p in b4_gen["prompts"])),
            "by_kind": {
                k: int(sum(p["conditions"][name]["java_isms"][k] for p in b4_gen["prompts"])) for k in JAVA_ISMS
            },
        }
        for name in conditions
    }

    b4_gen["runtime_s"] = round(time.perf_counter() - started, 1)
    (args.out / "s1b_generation.json").write_text(json.dumps(b4_gen, indent=1))
    print(args.out / "s1b_generation.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["pairs", "generation"])
    ap.add_argument("--pairs", type=Path, default=HERE / "pairs.yaml")
    ap.add_argument("--gen", type=Path, default=HERE / "generation.yaml")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--max-tokens", type=int, default=320)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (measure if args.part == "pairs" else generation)(args)


if __name__ == "__main__":
    main()
