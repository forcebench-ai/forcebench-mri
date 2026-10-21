"""Study 3, analysis: does the model know it'll fail, and does it like the task? No model needed.

Reads what collect.py saved, the graded run's labels (one sample per task, temperature 1.0) and a soft label:
each task's pass rate over every graded sample of the same base model at another quantisation.

- Pass/fail probes per layer, at the prompt end (before any reasoning) and after the answer: standardised L2
  logistic regression, its strength picked by inner cross-validation, scored out of fold (stratified 10-fold,
  repeated 5 times). An L2 solution lies in the span of its training rows, so each fold fits on the training
  rows' singular-vector coordinates: the same model, far cheaper.
- A label-permutation null for the best layer's AUROC, so picking the best of 64 layers isn't flattering.
- A baseline from what's known without looking inside (suite, difficulty, prompt length); the probe added to
  it (nested stacking); a within-suite AUROC (only pairs of tasks from the same suite); probes on activations
  with each suite's mean removed; ridge regression on the soft label.
- Stated confidence and stated enjoyment (expected digit), against pass/fail, the probes and each other.
- If study 2 has saved a valence direction, the prompt-end activations projected onto it.

    uv run python studies/s3_task/analyse.py --out outputs/s3 --runs <forcebench>/results/runs
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import matplotlib
import numpy as np
from joblib import Parallel, delayed, parallel_config
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SOFT_MODEL = "qwen3.8-27b-awq-int4"
SITES = ("prompt_end", "answer_end")
FIXED_LAYER = 40  # fixed before any result was seen: comparisons at it don't get to pick the best layer
CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
ALPHAS = np.logspace(0, 6, 13)
REPEATS, FOLDS, INNER = 5, 10, 5
DIFFICULTY = {"easy": 0, "medium": 1, "hard": 2}


def quiet() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)


def probe(seed: int) -> LogisticRegressionCV:
    return LogisticRegressionCV(
        Cs=CS, cv=StratifiedKFold(INNER, shuffle=True, random_state=seed), scoring="roc_auc", max_iter=5000
    )


def project(xtr: np.ndarray, xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardise on the training rows, then express both sets in the training rows' singular vectors."""
    mu, sd = xtr.mean(0), xtr.std(0)
    sd = np.maximum(sd, 1e-3 * np.median(sd))
    a, b = (xtr - mu) / sd, (xte - mu) / sd
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    return u * s, b @ vt.T


def remove_suite_means(xtr, xte, str_, ste):
    """Subtract each suite's mean activation, as estimated on the training rows."""
    means = {s: xtr[str_ == s].mean(0) for s in np.unique(str_)}
    return xtr - np.stack([means[s] for s in str_]), xte - np.stack([means[s] for s in ste])


def layer_acts(path: Path, layer: int) -> np.ndarray:
    return np.load(path, mmap_mode="r")[:, layer].astype(np.float32)


def splits(y: np.ndarray) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    return [list(StratifiedKFold(FOLDS, shuffle=True, random_state=r).split(y, y)) for r in range(REPEATS)]


def oof_layer(path: Path, layer: int, y, folds, suites=None) -> np.ndarray:
    """Out-of-fold probe scores ([repeats, tasks]) at one layer; with ``suites``, on suite-centred activations."""
    quiet()
    x = layer_acts(path, layer)
    out = np.zeros((len(folds), len(y)))
    for r, repeat in enumerate(folds):
        for tr, te in repeat:
            xtr, xte = (x[tr], x[te]) if suites is None else remove_suite_means(x[tr], x[te], suites[tr], suites[te])
            ztr, zte = project(xtr, xte)
            out[r, te] = probe(r).fit(ztr, y[tr]).decision_function(zte)
    return out


def null_layer(path: Path, layer: int, perms: np.ndarray, folds) -> np.ndarray:
    """AUROC at one layer for each permuted label vector (one repeat, the first repeat's folds)."""
    quiet()
    x = layer_acts(path, layer)
    scores = np.zeros(perms.shape)
    for tr, te in folds:
        ztr, zte = project(x[tr], x[te])
        for p, yp in enumerate(perms):
            scores[p, te] = probe(p).fit(ztr, yp[tr]).decision_function(zte)
    return np.array([roc_auc_score(yp, s) for yp, s in zip(perms, scores, strict=True)])


def ridge_layer(path: Path, layer: int, soft: np.ndarray, seed_folds) -> np.ndarray:
    """Out-of-fold ridge predictions of the soft label at one layer (mean over repeats)."""
    quiet()
    x = layer_acts(path, layer)
    out = np.zeros((len(seed_folds), len(soft)))
    for r, repeat in enumerate(seed_folds):
        for tr, te in repeat:
            ztr, zte = project(x[tr], x[te])
            out[r, te] = RidgeCV(alphas=ALPHAS).fit(ztr, soft[tr]).predict(zte)
    return out.mean(0)


def stacked_layer(path: Path, layer: int, y, base: np.ndarray, folds) -> np.ndarray:
    """Out-of-fold scores of a logistic model on ``base`` features plus the probe's score, nested: the probe's
    score on each training row comes from an inner fold that didn't see it. Mean over repeats."""
    quiet()
    x = layer_acts(path, layer)
    out = np.zeros((len(folds), len(y)))
    for r, repeat in enumerate(folds):
        for tr, te in repeat:
            inner = np.zeros(len(tr))
            for itr, ite in StratifiedKFold(INNER, shuffle=True, random_state=r).split(tr, y[tr]):
                ztr, zte = project(x[tr][itr], x[tr][ite])
                inner[ite] = probe(r).fit(ztr, y[tr][itr]).decision_function(zte)
            ztr, zte = project(x[tr], x[te])
            outer = probe(r).fit(ztr, y[tr]).decision_function(zte)
            ftr, fte = np.column_stack([base[tr], inner]), np.column_stack([base[te], outer])
            mu, sd = ftr.mean(0), ftr.std(0) + 1e-9
            out[r, te] = (
                LogisticRegression(max_iter=5000).fit((ftr - mu) / sd, y[tr]).decision_function((fte - mu) / sd)
            )
    return out.mean(0)


def oof_plain(f: np.ndarray, y, folds) -> np.ndarray:
    """Out-of-fold scores of a logistic model on a few plain features (mean over repeats)."""
    quiet()
    out = np.zeros((len(folds), len(y)))
    for r, repeat in enumerate(folds):
        for tr, te in repeat:
            mu, sd = f[tr].mean(0), f[tr].std(0) + 1e-9
            clf = LogisticRegressionCV(
                Cs=CS + [10.0, 100.0],
                cv=StratifiedKFold(INNER, shuffle=True, random_state=r),
                scoring="roc_auc",
                max_iter=5000,
            )
            out[r, te] = clf.fit((f[tr] - mu) / sd, y[tr]).decision_function((f[te] - mu) / sd)
    return out.mean(0)


def within_suite_auc(score: np.ndarray, y: np.ndarray, suites: np.ndarray) -> float:
    """AUROC over pairs (passed, failed) from the same suite only; suites with one class drop out."""
    wins = pairs = 0.0
    for s in np.unique(suites):
        pos, neg = score[(suites == s) & (y == 1)], score[(suites == s) & (y == 0)]
        if len(pos) and len(neg):
            d = pos[:, None] - neg[None, :]
            wins += (d > 0).sum() + 0.5 * (d == 0).sum()
            pairs += d.size
    return float(wins / pairs) if pairs else float("nan")


def boot(stat, n: int, rng: np.random.Generator, reps: int = 2000) -> tuple[float, float]:
    """95% percentile bootstrap interval of ``stat(idx)`` over tasks resampled with replacement."""
    vals = []
    for _ in range(reps):
        v = stat(rng.integers(0, n, n))
        if np.isfinite(v):
            vals.append(v)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def safe_auc(y, s) -> float:
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def auc_summary(score, y, suites, rng) -> dict:
    return {
        "auroc": round(safe_auc(y, score), 4),
        "auroc_ci": [round(v, 4) for v in boot(lambda i: safe_auc(y[i], score[i]), len(y), rng)],
        "within_suite": round(within_suite_auc(score, y, suites), 4),
        "within_suite_ci": [
            round(v, 4) for v in boot(lambda i: within_suite_auc(score[i], y[i], suites[i]), len(y), rng)
        ],
    }


def delta_auc(a, b, y, rng) -> dict:
    """AUROC(a) - AUROC(b) on the same tasks, with a paired bootstrap interval."""
    return {
        "delta": round(safe_auc(y, a) - safe_auc(y, b), 4),
        "ci": [round(v, 4) for v in boot(lambda i: safe_auc(y[i], a[i]) - safe_auc(y[i], b[i]), len(y), rng)],
    }


def corr(a, b, rng) -> dict:
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    return {
        "spearman": round(float(spearmanr(a, b).statistic), 4),
        "spearman_ci": [round(v, 4) for v in boot(lambda i: spearmanr(a[i], b[i]).statistic, len(a), rng)],
        "pearson": round(float(pearsonr(a, b).statistic), 4),
        "n": len(a),
    }


def soft_labels(runs: Path, model_id: str, task_ids: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Each task's pass rate over every graded full-set sample of ``model_id`` (skipped/stale/infra excluded)."""
    passes: dict[str, list[bool]] = {t: [] for t in task_ids}
    used = []
    for run_dir in sorted(runs.iterdir()):
        meta, cases = run_dir / "run.json", run_dir / "cases.jsonl"
        if not (meta.exists() and cases.exists()):
            continue
        run = json.loads(meta.read_text())
        if run.get("model", {}).get("id") != model_id or run.get("subset") != "full":
            continue
        used.append(run_dir.name)
        for c in map(json.loads, cases.read_text().splitlines()):
            if c["task_id"] in passes and not (c.get("skipped") or c.get("stale") or c.get("infra_error")):
                passes[c["task_id"]].append(bool(c["passed"]))
    rate = np.array([np.mean(passes[t]) if passes[t] else np.nan for t in task_ids])
    count = np.array([len(passes[t]) for t in task_ids])
    return rate, count, used


def expected_digit(probs: list[list[float]]) -> np.ndarray:
    return np.array(probs) @ np.arange(10)


def valence(out: Path, valence_path: Path, suites, y, conf, enjoy, rng) -> dict | None:
    """Each task's start of reply (thinking off, the position study 2 reads its anchors at) projected onto study
    2's valence direction at study 2's chosen layer. On the scale of study 2's coding anchors when it saved them
    (+1 = its pleasant coding activities' mean, -1 = its unpleasant ones', 0 = midway), else of whatever anchors
    it saved, else raw. Positive is pleasant: the direction is positive minus negative statements."""
    if not valence_path.exists():
        return None
    v = np.load(valence_path)
    layer = int(v["layer"])
    unit = v["unit"][layer] if "unit" in v.files else v["delta"][layer] / np.linalg.norm(v["delta"][layer])
    site = "prompt_end_nothink" if (out / "acts_prompt_end_nothink.npy").exists() else "prompt_end"
    proj = layer_acts(out / f"acts_{site}.npy", layer) @ unit.astype(np.float32)
    if "coding_pleasant_mean" in v.files:
        hi, lo = float(v["coding_pleasant_mean"][layer]), float(v["coding_unpleasant_mean"][layer])
        mid, half, scale = (hi + lo) / 2, (hi - lo) / 2, "study 2 coding anchors (+1 pleasant, -1 unpleasant)"
    elif "anchor_mid" in v.files:
        mid, half, scale = float(v["anchor_mid"][layer]), float(v["anchor_half"][layer]), "study 2 anchors"
    else:
        mid, half, scale = 0.0, 1.0, "raw projection"
    score = (proj - mid) / half
    return {
        "site": site,
        "layer": layer,
        "scale": scale,
        "valence_file_keys": list(v.files),
        "mean": round(float(score.mean()), 3),
        "vs_pass": auc_summary(score, y, suites, rng),
        "vs_stated_enjoyment": corr(score, enjoy, rng),
        "vs_stated_confidence": corr(score, conf, rng),
        "by_suite": {
            s: {
                "mean": round(float(score[suites == s].mean()), 3),
                "ci": [
                    round(c, 3)
                    for c in boot(lambda i, m=score[suites == s]: m[i].mean(), int((suites == s).sum()), rng)
                ],
                "pass_rate": round(float(y[suites == s].mean()), 3),
                "stated_enjoyment": round(float(enjoy[suites == s].mean()), 3),
            }
            for s in np.unique(suites)
        },
        "score": score.round(4).tolist(),
    }


def plot_valence(val: dict, out: Path) -> Path:
    by_suite = val["by_suite"]
    order = sorted(by_suite, key=lambda s: by_suite[s]["mean"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [3, 2]})
    means = [by_suite[s]["mean"] for s in order]
    err = [
        [by_suite[s]["mean"] - by_suite[s]["ci"][0] for s in order],
        [by_suite[s]["ci"][1] - by_suite[s]["mean"] for s in order],
    ]
    axes[0].barh(order, means, xerr=err, color="#c27c0e", alpha=0.85, capsize=3)
    axes[0].axvline(0, color="#555", lw=0.8)
    axes[0].set_xlabel(f"internal valence at layer {val['layer']} ({val['scale']})", fontsize=8)
    axes[0].set_title("Internal valence at the start of the reply, by suite (95% CI)", fontsize=10)
    xs = [by_suite[s]["stated_enjoyment"] for s in order]
    axes[1].scatter(xs, means, color="#c27c0e")
    for s, x_, y_ in zip(order, xs, means, strict=True):
        axes[1].annotate(s, (x_, y_), fontsize=7, textcoords="offset points", xytext=(3, 3))
    r = val["vs_stated_enjoyment"]
    axes[1].set_xlabel("stated enjoyment (suite mean, 0-9)")
    axes[1].set_ylabel("internal valence (suite mean)")
    axes[1].set_title(
        f"Inside vs stated, per task Spearman {r['spearman']:.2f} "
        f"[{r['spearman_ci'][0]:.2f}, {r['spearman_ci'][1]:.2f}]",
        fontsize=9,
    )
    fig.tight_layout()
    path = out / "s3_valence_by_suite.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_auroc(res: dict, out: Path) -> Path:
    layers = np.array(res["layers"])
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for site, colour, label in (
        ("prompt_end", "#2c66d4", "prompt end (before reasoning)"),
        ("answer_end", "#d23f5c", "after its own answer"),
    ):
        p = res["probe"][site]
        ax.plot(layers, p["auroc_mean"], color=colour, label=f"probe: {label}")
        ax.fill_between(layers, p["auroc_ci"][0], p["auroc_ci"][1], color=colour, alpha=0.15)
        ax.fill_between(
            layers, p["null"]["per_layer_lo"], p["null"]["per_layer_hi"], color=colour, alpha=0.07, hatch="..", lw=0
        )
        ax.axhline(p["null"]["max_p95"], color=colour, ls=":", lw=1)
    ax.axhline(res["baseline"]["auroc"], color="black", ls="--", lw=1.2, label="baseline: suite + difficulty + length")
    ax.axhline(
        res["stated"]["confidence"]["vs_pass"]["auroc"],
        color="#1a8a5a",
        ls="-.",
        lw=1.2,
        label="stated confidence (0-9)",
    )
    ax.axhline(0.5, color="#888", lw=0.8)
    ax.set_xlabel("layer")
    ax.set_ylabel("AUROC, pass vs fail (out of fold)")
    ax.set_title(f"Does it know it'll fail? {res['n_tasks']} Forcebench tasks, {res['model']}", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.text(
        0.01,
        0.02,
        "shaded dots: permutation null, 95% per layer; dotted: 95th pct of the null's best layer",
        transform=ax.transAxes,
        fontsize=7,
        color="#555",
    )
    fig.tight_layout()
    path = out / "s3_auroc_by_layer.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_calibration(conf: np.ndarray, y: np.ndarray, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    edges = np.unique(np.quantile(conf, np.linspace(0, 1, 7)))
    which = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, len(edges) - 2)
    xs, ys, ns = [], [], []
    for b in range(len(edges) - 1):
        m = which == b
        if m.any():
            xs.append(conf[m].mean() / 9)
            ys.append(y[m].mean())
            ns.append(int(m.sum()))
    se = [np.sqrt(p * (1 - p) / n) for p, n in zip(ys, ns, strict=True)]
    axes[0].errorbar(xs, ys, yerr=1.96 * np.array(se), fmt="o-", color="#1a8a5a", capsize=3)
    for x_, y_, n in zip(xs, ys, ns, strict=True):
        axes[0].annotate(f"n={n}", (x_, y_), textcoords="offset points", xytext=(4, 6), fontsize=7)
    axes[0].plot([0, 1], [0, 1], color="#888", lw=0.8, ls="--")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_xlabel("stated confidence (expected digit / 9), sextile bins")
    axes[0].set_ylabel("share that passed")
    axes[0].set_title("Stated confidence vs what happened", fontsize=10)
    bins = np.linspace(0, 9, 19)
    axes[1].hist(conf[y == 1], bins=bins, alpha=0.6, color="#2c66d4", label=f"passed (n={int(y.sum())})")
    axes[1].hist(conf[y == 0], bins=bins, alpha=0.6, color="#d23f5c", label=f"failed (n={int((1 - y).sum())})")
    axes[1].set_xlabel("stated confidence (expected digit, 0-9)")
    axes[1].set_title("Stated confidence by outcome", fontsize=10)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = out / "s3_confidence_calibration.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_enjoyment(by_suite: dict, out: Path) -> Path:
    order = sorted(by_suite, key=lambda s: by_suite[s]["mean"])
    fig, ax = plt.subplots(figsize=(8, 5.2))
    means = [by_suite[s]["mean"] for s in order]
    lo = [by_suite[s]["mean"] - by_suite[s]["ci"][0] for s in order]
    hi = [by_suite[s]["ci"][1] - by_suite[s]["mean"] for s in order]
    ax.barh(order, means, xerr=[lo, hi], color="#7a5cd2", alpha=0.8, capsize=3)
    for i, s in enumerate(order):
        ax.text(0.1, i, f"passed {by_suite[s]['pass_rate']:.0%}", va="center", fontsize=7, color="white")
    ax.set_xlim(0, 9)
    ax.set_xlabel("stated enjoyment (expected digit, 0 = would hate it, 9 = would love it)")
    ax.set_title("How much would it enjoy the task? Mean by suite, 95% CI", fontsize=10)
    fig.tight_layout()
    path = out / "s3_enjoyment_by_suite.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="collect.py's output directory")
    ap.add_argument("--runs", type=Path, required=True, help="Forcebench results/runs, for the soft label")
    ap.add_argument("--perms", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--valence", type=Path, default=Path("outputs/s2/valence.npz"))
    ap.add_argument("--valence-only", action="store_true", help="redo only the valence read in s3_results.json")
    args = ap.parse_args()
    quiet()
    started = time.perf_counter()
    rng = np.random.default_rng(0)

    index = json.loads((args.out / "index.json").read_text())
    stated = json.loads((args.out / "stated.json").read_text())
    task_ids = index["task_ids"]
    y = np.array(index["passed"], dtype=int)
    suites = np.array(index["suites"])
    diff = np.array([DIFFICULTY[d] for d in index["difficulty"]])
    n = len(y)
    paths = {site: args.out / f"acts_{site}.npy" for site in SITES}  # the probes' two sites
    n_layers = np.load(paths["prompt_end"], mmap_mode="r").shape[1]
    layers = list(range(n_layers))
    folds = splits(y)

    soft, soft_n, soft_runs = soft_labels(args.runs, SOFT_MODEL, task_ids)
    if np.isnan(soft).any():
        raise SystemExit(f"no {SOFT_MODEL} samples for {[t for t, s in zip(task_ids, soft) if np.isnan(s)]}")
    conf = expected_digit(stated["confidence"]["digit_probs"])
    enjoy = expected_digit(stated["enjoyment"]["digit_probs"])
    if args.valence_only:
        results_path = args.out / "s3_results.json"
        res = json.loads(results_path.read_text())
        res["valence"] = valence(args.out, args.valence, suites, y, conf, enjoy, rng)
        results_path.write_text(json.dumps(res, indent=1))
        if res["valence"]:
            print(plot_valence(res["valence"], args.out))
        print(results_path)
        return
    onehot = (suites[:, None] == np.unique(suites)[None, :]).astype(float)
    base = np.column_stack([onehot, np.eye(3)[diff], np.log(index["prompt_end_tokens"])])
    base_score = oof_plain(base, y, folds)

    res: dict = {
        "model": "Qwen3.8 27B (MLX 8-bit)",
        "run": index["run"],
        "n_tasks": n,
        "n_passed": int(y.sum()),
        "layers": layers,
        "fixed_layer": FIXED_LAYER,
        "soft_label": {
            "model": SOFT_MODEL,
            "runs": soft_runs,
            "samples_per_task_median": float(np.median(soft_n)),
            "samples_per_task_min": int(soft_n.min()),
            "mean_pass_rate": round(float(np.nanmean(soft)), 4),
            # How well other samples' pass rate predicts this sample's outcome: a reference for any predictor.
            "vs_pass": auc_summary(soft, y, suites, rng),
        },
        "baseline": auc_summary(base_score, y, suites, rng),
        "probe": {},
        "incremental": {},
        "ridge_soft": {},
    }

    with parallel_config(backend="loky", inner_max_num_threads=1):
        par = Parallel(n_jobs=args.jobs)
        perms = np.array([rng.permutation(y) for _ in range(args.perms)])
        oof: dict[str, np.ndarray] = {}
        for site, path in paths.items():
            t0 = time.perf_counter()
            scores = par(delayed(oof_layer)(path, layer, y, folds) for layer in layers)
            mean_scores = np.stack([s.mean(0) for s in scores])  # [layers, tasks]
            auroc_mean = [float(np.mean([roc_auc_score(y, s[r]) for r in range(REPEATS)])) for s in scores]
            cis = [boot(lambda i, s=s: safe_auc(y[i], s[i]), n, rng, reps=500) for s in mean_scores]
            within = [within_suite_auc(s, y, suites) for s in mean_scores]
            null = np.stack(par(delayed(null_layer)(path, layer, perms, folds[0]) for layer in layers), axis=1)
            resid = par(delayed(oof_layer)(path, layer, y, folds, suites) for layer in layers)
            resid_mean = np.stack([s.mean(0) for s in resid])
            best = int(np.argmax(auroc_mean))
            oof[site] = mean_scores
            res["probe"][site] = {
                "auroc_mean": [round(v, 4) for v in auroc_mean],
                "auroc_ci": [[round(c[0], 4) for c in cis], [round(c[1], 4) for c in cis]],
                "within_suite": [round(v, 4) for v in within],
                "best_layer": best,
                "best": auc_summary(mean_scores[best], y, suites, rng),
                "fixed": auc_summary(mean_scores[FIXED_LAYER], y, suites, rng),
                "null": {
                    "perms": args.perms,
                    "per_layer_lo": np.percentile(null, 2.5, axis=0).round(4).tolist(),
                    "per_layer_hi": np.percentile(null, 97.5, axis=0).round(4).tolist(),
                    "max_p95": round(float(np.percentile(null.max(1), 95)), 4),
                    "p_best": round(float((1 + (null.max(1) >= auroc_mean[best]).sum()) / (args.perms + 1)), 4),
                },
                "suite_centred": {
                    "auroc": [round(safe_auc(y, s), 4) for s in resid_mean],
                    "within_suite": [round(within_suite_auc(s, y, suites), 4) for s in resid_mean],
                    "fixed": auc_summary(resid_mean[FIXED_LAYER], y, suites, rng),
                },
            }
            stack_layers = sorted({*range(7, n_layers, 8), FIXED_LAYER, best})
            stacked = par(delayed(stacked_layer)(path, layer, y, base, folds) for layer in stack_layers)
            res["incremental"][site] = {
                str(layer): {
                    "combined": auc_summary(s, y, suites, rng),
                    "vs_baseline": delta_auc(s, base_score, y, rng),
                }
                for layer, s in zip(stack_layers, stacked, strict=True)
            }
            ridge = par(delayed(ridge_layer)(path, layer, soft, folds[:2]) for layer in layers)
            res["ridge_soft"][site] = {
                "spearman": [round(float(spearmanr(p, soft).statistic), 4) for p in ridge],
                "fixed": corr(ridge[FIXED_LAYER], soft, rng),
            }
            print(
                f"{site}: best layer {best} AUROC {auroc_mean[best]:.3f}, fixed L{FIXED_LAYER} "
                f"{auroc_mean[FIXED_LAYER]:.3f}, baseline {res['baseline']['auroc']:.3f} "
                f"({time.perf_counter() - t0:.0f}s)",
                flush=True,
            )

        # Stated confidence and the probe: does the inside know more than the model says?
        conf_base = np.column_stack([conf])
        pe = paths["prompt_end"]
        conf_plus_probe = stacked_layer(pe, FIXED_LAYER, y, conf_base, folds)
        conf_score = oof_plain(conf_base, y, folds)
        base_conf = np.column_stack([base, conf])
        base_conf_score = oof_plain(base_conf, y, folds)
        base_conf_probe = stacked_layer(pe, FIXED_LAYER, y, base_conf, folds)

    ridge_base = np.zeros(n)
    for tr, te in KFold(FOLDS, shuffle=True, random_state=0).split(base):
        ridge_base[te] = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(base[tr], soft[tr]).predict(base[te])
    res["ridge_soft"]["baseline"] = corr(ridge_base, soft, rng)

    probe_fixed = oof["prompt_end"][FIXED_LAYER]
    res["stated"] = {
        "confidence": {
            "question": stated["confidence"]["question"],
            "digit_mass_median": round(float(np.median(stated["confidence"]["digit_mass"])), 4),
            "mean": round(float(conf.mean()), 3),
            "mean_passed": round(float(conf[y == 1].mean()), 3),
            "mean_failed": round(float(conf[y == 0].mean()), 3),
            "vs_pass": auc_summary(conf, y, suites, rng),
            "vs_soft": corr(conf, soft, rng),
            "vs_probe_fixed": corr(conf, probe_fixed, rng),
        },
        "enjoyment": {
            "question": stated["enjoyment"]["question"],
            "digit_mass_median": round(float(np.median(stated["enjoyment"]["digit_mass"])), 4),
            "mean": round(float(enjoy.mean()), 3),
            "vs_pass": auc_summary(enjoy, y, suites, rng),
            "vs_soft": corr(enjoy, soft, rng),
            "vs_difficulty": corr(enjoy, diff.astype(float), rng),
            "vs_confidence": corr(enjoy, conf, rng),
            "vs_probe_fixed": corr(enjoy, probe_fixed, rng),
            "by_suite": {
                s: {
                    "n": int((suites == s).sum()),
                    "mean": round(float(enjoy[suites == s].mean()), 3),
                    "ci": [
                        round(v, 3)
                        for v in boot(lambda i, m=enjoy[suites == s]: m[i].mean(), int((suites == s).sum()), rng)
                    ],
                    "pass_rate": round(float(y[suites == s].mean()), 3),
                    "confidence": round(float(conf[suites == s].mean()), 3),
                }
                for s in np.unique(suites)
            },
        },
    }
    res["inside_vs_says"] = {
        "layer": FIXED_LAYER,
        "probe_minus_stated": delta_auc(probe_fixed, conf, y, rng),
        "stated_plus_probe_vs_stated": delta_auc(conf_plus_probe, conf_score, y, rng),
        "baseline_plus_stated": auc_summary(base_conf_score, y, suites, rng),
        "baseline_plus_stated_plus_probe_vs_baseline_plus_stated": delta_auc(base_conf_probe, base_conf_score, y, rng),
        "stated_vs_baseline": delta_auc(conf, base_score, y, rng),
    }
    res["valence"] = valence(args.out, args.valence, suites, y, conf, enjoy, rng)
    res["per_task"] = [
        {
            "task_id": t,
            "suite": str(suites[i]),
            "difficulty": index["difficulty"][i],
            "passed": bool(y[i]),
            "soft_pass_rate": None if np.isnan(soft[i]) else round(float(soft[i]), 3),
            "stated_confidence": round(float(conf[i]), 3),
            "stated_enjoyment": round(float(enjoy[i]), 3),
            "probe_prompt_end_fixed": round(float(probe_fixed[i]), 4),
            "probe_answer_end_fixed": round(float(oof["answer_end"][FIXED_LAYER][i]), 4),
        }
        for i, t in enumerate(task_ids)
    ]
    res["analysis_runtime_s"] = round(time.perf_counter() - started, 1)
    (args.out / "s3_results.json").write_text(json.dumps(res, indent=1))
    for path in (
        plot_auroc(res, args.out),
        plot_calibration(conf, y, args.out),
        plot_enjoyment(res["stated"]["enjoyment"]["by_suite"], args.out),
    ):
        print(path)
    if res["valence"]:
        print(plot_valence(res["valence"], args.out))
    print(args.out / "s3_results.json")


if __name__ == "__main__":
    main()
