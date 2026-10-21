"""Study 1, phase A: does the model think in Java while writing Apex? The logit lens over minimal pairs.

For each pair in pairs.yaml the same request and code prefix are shown as Apex and as Java (thinking off).
At the last position of the prefix, every layer's residual stream is decoded with the logit lens and two
probabilities are read: the Apex continuation's first token and the Java continuation's (summed over the
alternatives a side lists). The per-layer log-odds log P(apex) - log P(java) says which language's idiom
leads at that layer. Forward passes only; nothing here generates text.

    MRI_MODEL_PATH=... uv run python studies/s1_latent_java/run.py --out outputs/s1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
import mlx.core as mx
import numpy as np
import yaml

from forcebench_mri.model import Hooks, Subject, load

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
LANGS = {"apex": ("Apex", "an Apex", "apex"), "java": ("Java", "a Java", "java")}


def chat_text(subject: Subject, user: str) -> str:
    """The chat template's text up to the start of the assistant's reply, thinking off."""
    return subject.tokenizer.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def first_token(tok, base: str, continuation: str) -> int | None:
    """The first token ``continuation`` adds after ``base``, or None when ``base`` doesn't end on a token boundary
    (its own tokens aren't a prefix of the tokens of ``base + continuation``)."""
    pre = tok.encode(base, add_special_tokens=False)
    full = tok.encode(base + continuation, add_special_tokens=False)
    if full[: len(pre)] != pre or len(full) == len(pre):
        return None
    return full[len(pre)]


def prepare(subject: Subject, pair: dict) -> tuple[dict | None, str]:
    """Token ids for both contexts and the compared first tokens; (None, reason) when the pair fails a check."""
    tok = subject.tokenizer
    contexts, ids = {}, {}
    for lang, (name, a_lang, fence) in LANGS.items():
        user = pair["ask"].format(lang=name, a_lang=a_lang)
        base = chat_text(subject, user) + f"```{fence}\n" + pair["prefix"]
        side_ids = {}
        for side in ("apex", "java"):
            got = [first_token(tok, base, c) for c in pair[side]]
            if None in got:
                bad = pair[side][got.index(None)]
                return None, f"{lang} context: prefix doesn't end on a token boundary before {bad!r}"
            side_ids[side] = sorted(set(got))
        if ids and side_ids != ids:
            return None, f"first tokens differ between contexts: {ids} vs {side_ids}"
        ids = side_ids
        contexts[lang] = tok.encode(base, add_special_tokens=False)
    overlap = set(ids["apex"]) & set(ids["java"])
    if overlap:
        return None, f"Apex and Java share first token(s) {[tok.decode([i]) for i in overlap]}"
    return {
        "tokens": contexts,
        "ids": ids,
        "decoded": {s: [tok.decode([i]) for i in v] for s, v in ids.items()},
    }, ""


def lens_logprobs(subject: Subject, tokens: list[int]) -> tuple[mx.array, int, float]:
    """Log-probabilities over the vocabulary at the last position for every layer ([layers, vocab]), the model's
    own top-1 next token, and the largest gap between the last layer's lens and the real output (a sanity check)."""
    last: dict[int, mx.array] = {}

    def keep(index: int, out: mx.array) -> mx.array:
        last[index] = out[:, -1]
        return out

    logits = subject.run(tokens, Hooks(fn=keep))
    resid = mx.concatenate([last[i] for i in range(subject.n_layers)], axis=0)  # [layers, d_model]
    lens = subject.logit_lens(resid).astype(mx.float32)
    logp = lens - mx.logsumexp(lens, axis=-1, keepdims=True)
    final = logits[0, -1].astype(mx.float32)
    final_logp = final - mx.logsumexp(final)
    gap = mx.max(mx.abs(final_logp - logp[-1]))
    mx.eval(logp, final, gap)
    return logp, int(mx.argmax(final).item()), float(gap.item())


def side_logp(logp: mx.array, ids: list[int]) -> np.ndarray:
    """log P(any of ``ids``) per layer."""
    return np.array(mx.logsumexp(logp[:, mx.array(ids)], axis=-1))


def boot_ci(values: np.ndarray, stat, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """95% percentile bootstrap interval over pairs (rows) of ``stat`` applied per layer (columns)."""
    idx = rng.integers(0, values.shape[0], size=(n, values.shape[0]))
    draws = np.stack([stat(values[i], axis=0) for i in idx])
    return np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)


def summarise(lo: np.ndarray, kinds: list[str], boot: int, rng: np.random.Generator) -> dict:
    """Per-layer summaries of log-odds (pairs x layers) for one context."""
    apex_leads = (lo > 0).astype(float)
    java_leads = (lo < 0).astype(float)
    out = {
        "median": np.median(lo, axis=0).tolist(),
        "q25": np.percentile(lo, 25, axis=0).tolist(),
        "q75": np.percentile(lo, 75, axis=0).tolist(),
        "java_leads_share": java_leads.mean(axis=0).tolist(),
        "apex_leads_share": apex_leads.mean(axis=0).tolist(),
    }
    lo_lo, lo_hi = boot_ci(lo, np.median, boot, rng)
    jl_lo, jl_hi = boot_ci(java_leads, np.mean, boot, rng)
    al_lo, al_hi = boot_ci(apex_leads, np.mean, boot, rng)
    out["median_ci"] = [lo_lo.tolist(), lo_hi.tolist()]
    out["java_leads_ci"] = [jl_lo.tolist(), jl_hi.tolist()]
    out["apex_leads_ci"] = [al_lo.tolist(), al_hi.tolist()]
    out["by_kind"] = {
        k: {
            "n": int(sum(1 for x in kinds if x == k)),
            "median": np.median(lo[[x == k for x in kinds]], axis=0).tolist(),
            "java_leads_share": java_leads[[x == k for x in kinds]].mean(axis=0).tolist(),
            "apex_leads_share": apex_leads[[x == k for x in kinds]].mean(axis=0).tolist(),
        }
        for k in dict.fromkeys(kinds)
    }
    return out


def context_gap(rows: list[dict], boot: int, rng: np.random.Generator) -> dict:
    """Per pair and layer, Apex-context log-odds minus Java-context log-odds: how far the language named in the
    request has moved the lens's reading at that layer. Zero means the layer reads the same in both contexts."""
    gap = np.array([np.array(r["contexts"]["apex"]["log_odds"]) - r["contexts"]["java"]["log_odds"] for r in rows])
    lo, hi = boot_ci(gap, np.median, boot, rng)
    return {
        "median": np.median(gap, axis=0).tolist(),
        "median_ci": [lo.tolist(), hi.tolist()],
        "share_positive": (gap > 0).mean(axis=0).tolist(),
    }


def java_band(summary: dict) -> list[int]:
    """Layers where, in the Apex context, the median log-odds is below zero and its 95% interval excludes zero."""
    return [i for i, hi in enumerate(summary["median_ci"][1]) if hi < 0]


def plot(results: dict, out: Path) -> list[Path]:
    layers = np.arange(results["n_layers"])
    paths = []
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for lang, colour in (("apex", "#2c66d4"), ("java", "#d23f5c")):
        s = results["summary"][lang]
        ax.plot(layers, s["median"], color=colour, label=f"{lang.capitalize()} context: median")
        ax.fill_between(layers, s["q25"], s["q75"], color=colour, alpha=0.18, label=f"{lang.capitalize()} context: IQR")
    ax.plot(
        layers,
        results["context_gap"]["median"],
        color="black",
        ls="--",
        lw=1.2,
        label="paired gap: Apex context minus Java context (median)",
    )
    ax.axhline(0, color="#555", lw=0.8)
    ax.set_xlabel("layer")
    ax.set_ylabel("log P(Apex token) - log P(Java token)")
    ax.set_title(f"Logit lens over {results['n_pairs']} Apex/Java minimal pairs ({results['model']})")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    paths.append(out / "s1_log_odds_by_layer.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, lang, key, title in (
        (axes[0], "apex", "java_leads_share", "Apex context: share of pairs where the Java token leads"),
        (axes[1], "java", "apex_leads_share", "Java context (control): share where the Apex token leads"),
    ):
        s = results["summary"][lang]
        for kind, ks in s["by_kind"].items():
            ax.plot(layers, ks[key], lw=1, alpha=0.75, label=f"{kind} (n={ks['n']})")
        ax.plot(layers, s[key], color="black", lw=2.2, label="all pairs")
        ci = s["java_leads_ci" if key == "java_leads_share" else "apex_leads_ci"]
        ax.fill_between(layers, ci[0], ci[1], color="black", alpha=0.12)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("layer")
    axes[0].set_ylabel("share of pairs")
    axes[1].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    paths.append(out / "s1_java_bump_by_kind.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=HERE / "pairs.yaml")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    started = time.perf_counter()
    subject = load()
    tok = subject.tokenizer
    pairs = yaml.safe_load(args.pairs.read_text())["pairs"]

    kept, dropped, rows = [], [], []
    max_gap = 0.0
    for pair in pairs:
        prep, why = prepare(subject, pair)
        if prep is None:
            dropped.append({"id": pair["id"], "kind": pair["kind"], "why": why})
            continue
        row = {"id": pair["id"], "kind": pair["kind"], "compared": prep["decoded"], "contexts": {}}
        for lang in LANGS:
            logp, top1, gap = lens_logprobs(subject, prep["tokens"][lang])
            max_gap = max(max_gap, gap)
            la, lj = side_logp(logp, prep["ids"]["apex"]), side_logp(logp, prep["ids"]["java"])
            expected = prep["ids"][lang]
            row["contexts"][lang] = {
                "n_tokens": len(prep["tokens"][lang]),
                "logp_apex": la.round(4).tolist(),
                "logp_java": lj.round(4).tolist(),
                "log_odds": (la - lj).round(4).tolist(),
                "final_top1": tok.decode([top1]),
                "final_top1_is_expected": top1 in expected,
            }
        rows.append(row)
        kept.append(pair["id"])

    kinds = [r["kind"] for r in rows]
    results = {
        "model": "Qwen3.8 27B (MLX 8-bit)",
        "n_layers": subject.n_layers,
        "n_pairs": len(rows),
        "dropped": dropped,
        "lens_vs_output_max_abs_logp_gap": max_gap,
        "summary": {
            lang: summarise(np.array([r["contexts"][lang]["log_odds"] for r in rows]), kinds, args.boot, rng)
            for lang in LANGS
        },
        "context_gap": context_gap(rows, args.boot, rng),
        "wrong_final": [
            {"id": r["id"], "context": lang, "top1": r["contexts"][lang]["final_top1"], "expected": r["compared"][lang]}
            for r in rows
            for lang in LANGS
            if not r["contexts"][lang]["final_top1_is_expected"]
        ],
        "pairs": rows,
    }
    # Same summaries over only the pairs whose final answer was the context's own token, in both contexts.
    clean = [r for r in rows if all(r["contexts"][lang]["final_top1_is_expected"] for lang in LANGS)]
    results["summary_correct_only"] = {
        "n_pairs": len(clean),
        **{
            lang: summarise(
                np.array([r["contexts"][lang]["log_odds"] for r in clean]), [r["kind"] for r in clean], args.boot, rng
            )
            for lang in LANGS
        },
    }
    results["java_band_apex_context"] = java_band(results["summary"]["apex"])
    results["java_band_apex_context_correct_only"] = java_band(results["summary_correct_only"]["apex"])
    results["runtime_s"] = round(time.perf_counter() - started, 1)
    (args.out / "s1_results.json").write_text(json.dumps(results, indent=1))
    for p in plot(results, args.out):
        print(p)
    print(args.out / "s1_results.json")


if __name__ == "__main__":
    main()
