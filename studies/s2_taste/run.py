"""Study 2: taste. Which languages, and clicks or code, does the model lean towards, and what does its inside say?

Four measurements, all with thinking off:

1. Forced choices (stated preference): log P("A") vs log P("B") at the start of the reply, for pairs of languages
   and for clicks (Flow Builder) vs code (Apex), framed as "which would you enjoy" (taste) and "which is the
   better choice" (judgement). Every choice is asked in both orders and the two are averaged.
2. Taste lens: log P(positive adjective) - log P(negative adjective) after "Writing {X} feels", at every layer.
3. Valence direction: the mean difference between positive and negative first-person statements about everyday
   activities, checked on held-out activities and templates, then on anchor activities (pleasant vs unpleasant)
   at the start of the reply. Items are projected onto it and reported on the anchors' scale. A steering check
   adds it to one option's tokens in a forced choice.
4. A few greedy replies, for colour.

    MRI_MODEL_PATH=... uv run python studies/s2_taste/run.py --out outputs/s2
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
from mlx_lm import stream_generate
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from forcebench_mri.model import Hooks, Subject, load

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
STAGES = ("choices", "lens", "valence", "steering", "quotes")
COLOURS = {"Apex": "#2c66d4", "Java": "#d23f5c", "Python": "#2a9d6e"}


def cap(s: str) -> str:
    return s[:1].upper() + s[1:]


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


def read_last(subject: Subject, tokens: list[int], add: dict | None = None) -> tuple[mx.array, mx.array]:
    """The residual stream at the last position after every layer ([layers, d_model], float32) and the
    log-probabilities of the next token ([vocab])."""
    last: dict[int, mx.array] = {}

    def keep(index: int, out: mx.array) -> mx.array:
        last[index] = out[:, -1]
        return out

    logits = subject.run(tokens, Hooks(fn=keep, add=add or {}))
    resid = mx.concatenate([last[i] for i in range(subject.n_layers)], axis=0).astype(mx.float32)
    final = logits[0, -1].astype(mx.float32)
    logp = final - mx.logsumexp(final)
    mx.eval(resid, logp)
    return resid, logp


def next_logp(subject: Subject, tokens: list[int], add: dict | None = None) -> mx.array:
    """Log-probabilities of the next token only."""
    final = subject.run(tokens, Hooks(add=add) if add else None)[0, -1].astype(mx.float32)
    logp = final - mx.logsumexp(final)
    mx.eval(logp)
    return logp


def boot_ci(values: np.ndarray, rng: np.random.Generator, n: int = 2000, stat=np.mean) -> tuple:
    """95% percentile bootstrap interval of ``stat`` over the first axis."""
    values = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    draws = np.stack([stat(values[i], axis=0) for i in idx])
    return np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)


def summary(values, rng: np.random.Generator) -> dict:
    values = np.asarray(values, dtype=float)
    lo, hi = boot_ci(values, rng)
    return {"n": len(values), "mean": float(values.mean()), "ci": [float(lo), float(hi)]}


# ------------------------------------------------------------------------------------------------ forced choices
def letter_ids(subject: Subject, user: str) -> tuple[int, int]:
    """The reply's first token for "A" and for "B", checked to be single tokens after the template."""
    base = chat_text(subject, user)
    ids = tuple(first_token(subject, base, letter) for letter in ("A", "B"))
    assert None not in ids and encode(subject, base + "A")[-1] == ids[0], "letters must be single tokens"
    return ids  # type: ignore[return-value]


def choose(subject: Subject, template: str, fields: dict, x: str, y: str, ids, add_for=None) -> dict:
    """Preference for option ``x`` over ``y``: the log-odds of choosing x, in both orders and averaged.
    ``add_for(text, option_text)`` may return steering hooks for a prompt."""
    orders = []
    for a, b in ((x, y), (y, x)):
        user = template.format(a=a, b=b, **fields)
        tokens = encode(subject, chat_text(subject, user))
        logp = next_logp(subject, tokens, add_for(user, tokens) if add_for else None)
        la, lb = logp[ids[0]].item(), logp[ids[1]].item()
        lo_x = la - lb if a == x else lb - la
        orders.append(
            {
                "x_is": "A" if a == x else "B",
                "lo_x": lo_x,
                "p_x": 1 / (1 + np.exp(-lo_x)),  # P(x | the reply is A or B)
                "mass_ab": float(np.exp(la) + np.exp(lb)),
            }
        )
    return {
        "lo_x": float(np.mean([o["lo_x"] for o in orders])),
        "p_x": float(np.mean([o["p_x"] for o in orders])),
        "orders": orders,
    }


def run_choices(subject: Subject, stim: dict, rng: np.random.Generator) -> dict:
    ids = letter_ids(subject, "Answer with the letter only.")
    out: dict = {"letter_ids": list(ids), "languages": {}, "clicks": {}}
    for framing, template in stim["language_prompts"].items():
        for x, y in stim["language_pairs"]:
            rows = []
            for group, tasks in stim["tasks"].items():
                for task in tasks:
                    rows.append({"group": group, "task": task, **choose(subject, template, {"task": task}, x, y, ids)})
            out["languages"].setdefault(framing, {})[f"{x} vs {y}"] = rows
    opts = stim["clicks_options"]
    for framing, template in stim["clicks_prompts"].items():
        rows = []
        for req in stim["requirements"]:
            got = choose(subject, template, {"req": req["text"]}, opts["code"], opts["clicks"], ids)
            rows.append({"textbook": req["textbook"], "req": req["text"], **got})
        out["clicks"][framing] = rows

    # Summaries: P(first-named option), its mean log-odds, and enjoy minus fit (paired over tasks)
    summ: dict = {"languages": {}, "clicks": {}}
    for pair in out["languages"]["enjoy"]:
        for group in [*stim["tasks"], "all"]:
            e = [r for r in out["languages"]["enjoy"][pair] if group in ("all", r["group"])]
            f = [r for r in out["languages"]["fit"][pair] if group in ("all", r["group"])]
            summ["languages"].setdefault(pair, {})[group] = {
                "enjoy_p_first": summary([r["p_x"] for r in e], rng),
                "fit_p_first": summary([r["p_x"] for r in f], rng),
                "enjoy_lo_first": summary([r["lo_x"] for r in e], rng),
                "fit_lo_first": summary([r["lo_x"] for r in f], rng),
                "enjoy_minus_fit_p": summary([a["p_x"] - b["p_x"] for a, b in zip(e, f, strict=True)], rng),
                "enjoy_share_first": float(np.mean([r["lo_x"] > 0 for r in e])),
                "fit_share_first": float(np.mean([r["lo_x"] > 0 for r in f])),
            }
    for textbook in ("flow", "apex", "all"):
        e = [r for r in out["clicks"]["enjoy"] if textbook in ("all", r["textbook"])]
        f = [r for r in out["clicks"]["fit"] if textbook in ("all", r["textbook"])]
        summ["clicks"][textbook] = {
            "enjoy_p_code": summary([r["p_x"] for r in e], rng),
            "fit_p_code": summary([r["p_x"] for r in f], rng),
            "enjoy_minus_fit_p_code": summary([a["p_x"] - b["p_x"] for a, b in zip(e, f, strict=True)], rng),
            "enjoy_share_code": float(np.mean([r["lo_x"] > 0 for r in e])),
            "fit_share_code": float(np.mean([r["lo_x"] > 0 for r in f])),
        }
    allrows = [o for fr in out["languages"].values() for rows in fr.values() for r in rows for o in r["orders"]]
    allrows += [o for rows in out["clicks"].values() for r in rows for o in r["orders"]]
    summ["min_mass_ab"] = float(min(o["mass_ab"] for o in allrows))
    summ["median_mass_ab"] = float(np.median([o["mass_ab"] for o in allrows]))
    # Letter bias: mean log-odds of "A" over "B" across every prompt (cancels in the averaged preferences)
    summ["mean_lo_A_minus_B"] = float(np.mean([o["lo_x"] if o["x_is"] == "A" else -o["lo_x"] for o in allrows]))
    out["summary"] = summ
    return out


# ------------------------------------------------------------------------------------------------ taste lens
def run_lens(subject: Subject, stim: dict, rng: np.random.Generator) -> dict:
    rows, dropped = [], []
    for item in stim["items"]:
        for t_i, (user_t, prefix_t) in enumerate(stim["lens_templates"]):
            user = user_t.format(act=item["act"])
            base = chat_text(subject, user) + prefix_t.format(act=item["act"], Act=cap(item["act"]))
            pos = [first_token(subject, base, " " + w) for w in stim["positive"]]
            neg = [first_token(subject, base, " " + w) for w in stim["negative"]]
            if None in pos + neg or set(pos) & set(neg):
                dropped.append({"item": item["name"], "template": t_i})
                continue
            resid, logp_final = read_last(subject, encode(subject, base))
            lens = subject.logit_lens(resid).astype(mx.float32)
            logp = lens - mx.logsumexp(lens, axis=-1, keepdims=True)
            lp = mx.logsumexp(logp[:, mx.array(sorted(set(pos)))], axis=-1)
            ln = mx.logsumexp(logp[:, mx.array(sorted(set(neg)))], axis=-1)
            top = mx.argsort(-logp_final)[:5].tolist()
            rows.append(
                {
                    "item": item["name"],
                    "template": t_i,
                    "log_odds": np.array(lp - ln).round(4).tolist(),
                    "final_mass_pos_neg": float(np.exp(np.array(lp[-1])) + np.exp(np.array(ln[-1]))),
                    "final_top5": [subject.tokenizer.decode([i]) for i in top],
                }
            )
    names = [i["name"] for i in stim["items"]]
    n_t = len(stim["lens_templates"])
    grid = np.full((len(names), n_t, subject.n_layers), np.nan)
    for r in rows:
        grid[names.index(r["item"]), r["template"]] = r["log_odds"]
    # Centred: each template's mean over items is subtracted, so a template's overall tone cancels
    centred = grid - np.nanmean(grid, axis=0, keepdims=True)
    summ = {}
    for i, name in enumerate(names):
        g, c = grid[i][~np.isnan(grid[i, :, 0])], centred[i][~np.isnan(centred[i, :, 0])]
        lo, hi = boot_ci(g, rng)
        clo, chi = boot_ci(c, rng)
        summ[name] = {
            "n_templates": len(g),
            "mean": g.mean(0).round(4).tolist(),
            "ci": [lo.round(4).tolist(), hi.round(4).tolist()],
            "centred_mean": c.mean(0).round(4).tolist(),
            "centred_ci": [clo.round(4).tolist(), chi.round(4).tolist()],
        }
    return {"rows": rows, "dropped": dropped, "summary": summ}


# ------------------------------------------------------------------------------------------------ valence
def cv_direction(acts: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    """Leave-one-group-out check of the mean-difference direction at every layer: accuracy (threshold at the
    midpoint of the training classes' projections) and Cohen's d on the held-out statements."""
    acc, d = [], []
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        delta = acts[tr & (y == 1)].mean(0) - acts[tr & (y == 0)].mean(0)  # [layers, d_model]
        u = delta / np.linalg.norm(delta, axis=-1, keepdims=True)
        proj_tr = np.einsum("nld,ld->nl", acts[tr], u)
        thr = (proj_tr[y[tr] == 1].mean(0) + proj_tr[y[tr] == 0].mean(0)) / 2
        proj = np.einsum("nld,ld->nl", acts[te], u)
        acc.append(((proj > thr) == (y[te][:, None] == 1)).mean(0))
        p, q = proj[y[te] == 1], proj[y[te] == 0]
        d.append((p.mean(0) - q.mean(0)) / np.sqrt((p.var(0, ddof=1) + q.var(0, ddof=1)) / 2))
    return {"acc": np.mean(acc, 0), "d": np.mean(d, 0)}


def anchor_separation(pos: np.ndarray, neg: np.ndarray) -> dict:
    """AUROC and Cohen's d between two anchor sets ([anchors, templates, layers], template-averaged) per layer."""
    a, b = pos.mean(1), neg.mean(1)
    labels = np.array([1] * len(a) + [0] * len(b))
    both = np.concatenate([a, b])
    auroc = np.array([roc_auc_score(labels, both[:, k]) for k in range(both.shape[1])])
    d = (a.mean(0) - b.mean(0)) / np.sqrt((a.var(0, ddof=1) + b.var(0, ddof=1)) / 2)
    return {"auroc": auroc, "d": d}


def anchor_scale(x: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """``x`` ([n, templates, layers]) per template on the anchors' scale: 0 is their midpoint, +1 the pleasant
    anchors' mean and -1 the unpleasant anchors'."""
    mid = (pos.mean(0) + neg.mean(0)) / 2
    half = (pos.mean(0) - neg.mean(0)) / 2
    return (x - mid[None]) / half[None]


def run_valence(subject: Subject, stim: dict, rng: np.random.Generator) -> tuple[dict, dict]:
    # Statements: every activity with every positive and negative template, read at the last token
    acts, y, act_i, tmpl_i = [], [], [], []
    for a_i, act in enumerate(stim["activities"]):
        for label, templates in (
            (1, stim["statement_templates"]["positive"]),
            (0, stim["statement_templates"]["negative"]),
        ):
            for t_i, t in enumerate(templates):
                resid, _ = read_last(subject, encode(subject, t.format(act=act, Act=cap(act))))
                acts.append(np.array(resid))
                y.append(label)
                act_i.append(a_i)
                tmpl_i.append(t_i)
    acts, y, act_i, tmpl_i = np.stack(acts), np.array(y), np.array(act_i), np.array(tmpl_i)
    folds = rng.permutation(len(stim["activities"])) % 5  # five folds of activities
    by_activity = cv_direction(acts, y, folds[act_i])
    by_template = cv_direction(acts, y, tmpl_i)
    delta = acts[y == 1].mean(0) - acts[y == 0].mean(0)
    unit = delta / np.linalg.norm(delta, axis=-1, keepdims=True)
    stmt_proj = np.einsum("nld,ld->nl", acts, unit)
    resid_norm = np.linalg.norm(acts, axis=-1).mean(0)

    # The same templates at the start of the reply, for anchors (known valence) and for the items
    def project(phrases: list[str]) -> np.ndarray:
        out = np.zeros((len(phrases), len(stim["projection_templates"]), subject.n_layers))
        for p_i, phrase in enumerate(phrases):
            for t_i, t in enumerate(stim["projection_templates"]):
                resid, _ = read_last(subject, subject.chat_tokens(t.format(act=phrase)))
                out[p_i, t_i] = np.einsum("ld,ld->l", np.array(resid), unit)
        return out

    everyday = project(stim["anchors"]["pleasant"]), project(stim["anchors"]["unpleasant"])
    coding = project(stim["coding_anchors"]["pleasant"]), project(stim["coding_anchors"]["unpleasant"])
    items = project([i["act"] for i in stim["items"]])
    ev, co = anchor_separation(*everyday), anchor_separation(*coding)
    # The layer: where the direction best separates the everyday anchors at the start of the reply, among
    # mid-to-late layers. The coding anchors stay out of the choice: they are its in-domain check and set the scale.
    layer = int(16 + np.argmax(ev["d"][16:57]))

    item_summary = {}
    scaled = {"coding": anchor_scale(items, *coding), "everyday": anchor_scale(items, *everyday)}
    centred = scaled["coding"] - scaled["coding"].mean(0, keepdims=True)  # relative to the items' mean
    for i, item in enumerate(stim["items"]):
        row = {}
        for name, values in (*scaled.items(), ("coding_centred", centred)):
            lo, hi = boot_ci(values[i, :, layer], rng)
            row[name] = {"score": float(values[i, :, layer].mean()), "ci": [float(lo), float(hi)]}
        row["coding_by_template"] = scaled["coding"][i, :, layer].round(3).tolist()
        row["coding_by_layer"] = scaled["coding"][i].mean(0).round(3).tolist()
        item_summary[item["name"]] = row
    on_coding = {}  # every anchor on the coding anchors' scale, template-averaged
    for group, key, sets in (("coding", "coding_anchors", coding), ("everyday", "anchors", everyday)):
        values = anchor_scale(np.concatenate(sets), *coding)[:, :, layer].mean(1)
        names = stim[key]["pleasant"] + stim[key]["unpleasant"]
        on_coding[group] = {name: float(v) for name, v in zip(names, values, strict=True)}
    res = {
        "n_statements": len(y),
        "layer": layer,
        "heldout_activity_acc": by_activity["acc"].round(4).tolist(),
        "heldout_activity_d": by_activity["d"].round(3).tolist(),
        "heldout_template_acc": by_template["acc"].round(4).tolist(),
        "heldout_template_d": by_template["d"].round(3).tolist(),
        "everyday_anchor_auroc": ev["auroc"].round(4).tolist(),
        "everyday_anchor_d": ev["d"].round(3).tolist(),
        "coding_anchor_auroc": co["auroc"].round(4).tolist(),
        "coding_anchor_d": co["d"].round(3).tolist(),
        "delta_norm": np.linalg.norm(delta, axis=-1).round(3).tolist(),
        "mean_resid_norm_statements": resid_norm.round(2).tolist(),
        "items": item_summary,
        "anchors_on_coding_scale": on_coding,
    }
    saved = {
        "delta": delta.astype(np.float32),
        "unit": unit.astype(np.float32),
        "layer": np.array(layer),
        "statement_proj_pos_mean": stmt_proj[y == 1].mean(0).astype(np.float32),
        "statement_proj_neg_mean": stmt_proj[y == 0].mean(0).astype(np.float32),
    }
    for name, (pos, neg), sep in (("coding", coding, co), ("everyday", everyday, ev)):
        # Raw projections of each anchor set at the start of the reply, template-averaged ([layers])
        saved[f"{name}_pleasant_mean"] = pos.mean((0, 1)).astype(np.float32)
        saved[f"{name}_unpleasant_mean"] = neg.mean((0, 1)).astype(np.float32)
        saved[f"{name}_auroc"] = sep["auroc"].astype(np.float32)
    return res, saved


def option_positions(subject: Subject, user: str, option: str, tokens: list[int]) -> list[int]:
    """Token positions of ``option``'s text where it is listed (after "A) " or "B) ") in the chat prompt."""
    text = chat_text(subject, user)
    enc = subject.tokenizer._tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    assert enc["input_ids"] == tokens
    for letter in ("A) ", "B) "):
        start = text.find(letter + option)
        if start >= 0:
            s, e = start + len(letter), start + len(letter) + len(option)
            return [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < e and b > s]
    raise ValueError(f"option {option!r} not listed")


def run_steering(
    subject: Subject, stim: dict, saved: dict, rng: np.random.Generator, layers: list[int] | None = None
) -> dict:
    """Add alpha * (valence mean difference) at the "Apex" option's tokens in the "enjoy" Apex-vs-Python choice.
    By default at the chosen layer alone and across a band of eight layers around it; with ``layers``, at each of
    those layers alone. Random directions of the same norms are the control."""
    layer = int(saved["layer"])
    deltas = saved["delta"]
    rand = rng.standard_normal(deltas.shape).astype(np.float32)
    rand *= np.linalg.norm(deltas, axis=-1, keepdims=True) / np.linalg.norm(rand, axis=-1, keepdims=True)
    ids = letter_ids(subject, "Answer with the letter only.")
    template = stim["language_prompts"]["enjoy"]
    tasks = [t for ts in stim["tasks"].values() for t in ts]
    if layers:
        designs = {f"L{k}": ([k], [-2, 0, 2]) for k in layers}
    else:
        designs = {f"L{layer}": ([layer], [-4, 0, 4]), "band": (list(range(layer - 4, layer + 4)), [-2, -1, 0, 1, 2])}
    out: dict = {}
    for design, (steer_at, alphas) in designs.items():
        out[design] = {"layers": steer_at, "valence": {}, "random": {}}
        for kind, vecs in (("valence", deltas), ("random", rand)):
            for alpha in alphas if kind == "valence" else [-max(alphas), max(alphas)]:

                def add_for(user: str, tokens: list[int], a=alpha, v=vecs, ls=steer_at) -> dict:
                    pos = option_positions(subject, user, "Apex", tokens)
                    return {k: (pos, mx.array(a * v[k])) for k in ls}

                los = [choose(subject, template, {"task": t}, "Apex", "Python", ids, add_for)["lo_x"] for t in tasks]
                out[design][kind][str(alpha)] = {"lo_apex": summary(los, rng), "per_task": np.round(los, 4).tolist()}
    return out


# ------------------------------------------------------------------------------------------------ quotes
def run_quotes(subject: Subject, stim: dict, max_tokens: int = 120) -> list[dict]:
    out = []
    for q in stim["quotes"]:
        text = "".join(
            r.text
            for r in stream_generate(subject.model, subject.tokenizer, subject.chat_tokens(q), max_tokens=max_tokens)
        )
        out.append({"prompt": q, "reply": text})
    return out


# ------------------------------------------------------------------------------------------------ contrasts
CONTRASTS = [
    ("Python", "Apex"),
    ("Apex", "Java"),
    ("Python", "Java"),
    ("Flow Builder", "Apex"),
    ("Flow Builder", "Process Builder"),
    ("Lightning Web Components", "Visualforce"),
]


def contrasts(results: dict, rng: np.random.Generator) -> dict:
    """Paired differences between items, template by template (taste lens at the output, valence on the coding
    scale), and how far the two methods agree on the items' order."""
    lens = {}
    for r in results["lens"]["rows"]:
        lens.setdefault(r["item"], {})[r["template"]] = r["log_odds"][-1]
    val = {n: v["coding_by_template"] for n, v in results["valence"]["items"].items()}
    out: dict = {}
    for a, b in CONTRASTS:
        lens_diff = [lens[a][t] - lens[b][t] for t in sorted(lens[a]) if t in lens[b]]
        val_diff = np.subtract(val[a], val[b])
        out[f"{a} - {b}"] = {"taste_lens_final": summary(lens_diff, rng), "valence": summary(val_diff, rng)}
    names = list(val)
    lens_c = [results["lens"]["summary"][n]["centred_mean"][-1] for n in names]
    val_c = [results["valence"]["items"][n]["coding"]["score"] for n in names]
    rho = spearmanr(lens_c, val_c)
    out["lens_vs_valence_spearman"] = {"rho": float(rho.statistic), "p": float(rho.pvalue), "n_items": len(names)}
    return out


def steering_effects(steering: dict, rng: np.random.Generator) -> dict:
    """Per design, the paired effect over tasks of steering: log-odds of Apex at the largest alpha minus at the
    smallest, for the valence direction and for the random control, and valence minus random."""
    out = {}
    for design, x in steering.items():
        per = {}
        for kind in ("valence", "random"):
            hi, lo = max(x[kind], key=float), min(x[kind], key=float)
            per[kind] = np.subtract(x[kind][hi]["per_task"], x[kind][lo]["per_task"])
            out.setdefault(design, {"layers": x["layers"], "alpha": float(hi)})[kind] = summary(per[kind], rng)
        out[design]["valence_minus_random"] = summary(per["valence"] - per["random"], rng)
        out[design]["valence_share_positive"] = float(np.mean(per["valence"] > 0))
    return out


# ------------------------------------------------------------------------------------------------ figures
def plot(results: dict, stim: dict, out: Path) -> list[Path]:
    paths = []
    ch = results["choices"]["summary"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    for ax, (pair, groups) in zip(axes, ch["languages"].items(), strict=True):
        first, second = pair.split(" vs ")
        xs, labels = [], []
        for g_i, group in enumerate(["generic", "salesforce"]):
            for f_i, framing in enumerate(["enjoy", "fit"]):
                s = groups[group][f"{framing}_p_first"]
                x = g_i * 2.6 + f_i
                ax.bar(x, s["mean"], color=COLOURS[first], alpha=0.9 if framing == "enjoy" else 0.45, width=0.85)
                ax.errorbar(
                    x, s["mean"], yerr=[[s["mean"] - s["ci"][0]], [s["ci"][1] - s["mean"]]], color="k", capsize=3
                )
                xs.append(x)
                labels.append(f"{group}\n{framing}")
        ax.axhline(0.5, color="#555", lw=0.8, ls="--")
        ax.set_xticks(xs, labels, fontsize=8)
        ax.set_title(f"{first} vs {second}: P(choose {first})", fontsize=10)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("P(first-named option), both orders averaged")
    fig.suptitle(
        "Forced choices: 'which would you enjoy writing it in?' vs 'which is the better choice?' (n=8 tasks per group, 95% CI)",
        fontsize=10,
    )
    fig.tight_layout()
    paths.append(out / "s2_language_preferences.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for t_i, textbook in enumerate(["flow", "apex"]):
        for f_i, framing in enumerate(["enjoy", "fit"]):
            s = ch["clicks"][textbook][f"{framing}_p_code"]
            x = t_i * 2.6 + f_i
            ax.bar(
                x,
                s["mean"],
                color="#2c66d4" if framing == "enjoy" else "#9db8ea",
                width=0.85,
                label=framing if t_i == 0 else None,
            )
            ax.errorbar(x, s["mean"], yerr=[[s["mean"] - s["ci"][0]], [s["ci"][1] - s["mean"]]], color="k", capsize=3)
            rows = [r for r in results["choices"]["clicks"][framing] if r["textbook"] == textbook]
            ax.scatter(
                np.full(len(rows), x) + rng_jitter(len(rows)),
                [r["p_x"] for r in rows],
                s=10,
                color="k",
                alpha=0.5,
                zorder=3,
            )
    ax.axhline(0.5, color="#555", lw=0.8, ls="--")
    ax.set_xticks([0.5, 3.1], ["textbook answer: Flow (n=8)", "textbook answer: Apex (n=8)"])
    ax.set_ylabel("P(choose the Apex trigger), both orders averaged")
    ax.set_ylim(0, 1)
    ax.legend(title="framing")
    ax.set_title("Clicks vs code: Flow Builder or an Apex trigger?", fontsize=10)
    fig.tight_layout()
    paths.append(out / "s2_clicks_vs_code.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    lens = results["lens"]["summary"]
    layers = np.arange(len(next(iter(lens.values()))["mean"]))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.3, 1]})
    for name, s in lens.items():
        if name in COLOURS:
            axes[0].plot(layers, s["centred_mean"], color=COLOURS[name], lw=2, label=name)
            axes[0].fill_between(layers, *s["centred_ci"], color=COLOURS[name], alpha=0.15)
        else:
            axes[0].plot(layers, s["centred_mean"], color="#999", lw=0.8, alpha=0.6)
    axes[0].axhline(0, color="#555", lw=0.8)
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("log-odds positive vs negative adjective,\nminus the template's mean over items")
    axes[0].set_title("Taste lens by layer (grey: the other items)", fontsize=10)
    axes[0].legend(fontsize=8)
    order = sorted(lens, key=lambda n: lens[n]["centred_mean"][-1])
    for k, name in enumerate(order):
        s = lens[name]
        m, lo, hi = s["centred_mean"][-1], s["centred_ci"][0][-1], s["centred_ci"][1][-1]
        axes[1].errorbar(m, k, xerr=[[m - lo], [hi - m]], fmt="o", color=COLOURS.get(name, "#555"), capsize=3)
    axes[1].set_yticks(range(len(order)), order, fontsize=8)
    axes[1].axvline(0, color="#555", lw=0.8)
    axes[1].set_xlabel("final-layer log-odds, template-centred (95% CI over 7 templates)")
    axes[1].set_title("Taste lens at the output", fontsize=10)
    fig.tight_layout()
    paths.append(out / "s2_taste_lens.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    v = results["valence"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1, 1.25]})
    L = np.arange(len(v["heldout_activity_acc"]))
    axes[0].plot(L, v["heldout_activity_acc"], label="statements, held-out activities: accuracy")
    axes[0].plot(L, v["heldout_template_acc"], label="statements, held-out templates: accuracy")
    axes[0].plot(L, v["everyday_anchor_auroc"], label="everyday anchors at reply start: AUROC")
    axes[0].plot(L, v["coding_anchor_auroc"], label="coding anchors at reply start: AUROC (held out)")
    axes[0].axvline(v["layer"], color="k", lw=0.8, ls="--")
    axes[0].set_xlabel("layer")
    axes[0].set_ylim(0.3, 1.02)
    axes[0].set_title(f"Valence direction: validation (chosen layer {v['layer']})", fontsize=10)
    axes[0].legend(fontsize=8, loc="lower right")
    order = sorted(v["items"], key=lambda n: v["items"][n]["coding"]["score"])
    for k, name in enumerate(order):
        s = v["items"][name]["coding"]
        axes[1].errorbar(
            s["score"],
            k,
            xerr=[[s["score"] - s["ci"][0]], [s["ci"][1] - s["score"]]],
            fmt="o",
            color=COLOURS.get(name, "#555"),
            capsize=3,
        )
    anchors = v["anchors_on_coding_scale"]["coding"]
    pl = [anchors[n] for n in stim["coding_anchors"]["pleasant"]]
    un = [anchors[n] for n in stim["coding_anchors"]["unpleasant"]]
    top = len(order) + 0.6
    axes[1].scatter(pl, np.full(len(pl), top), marker="|", color="#2a9d6e", s=120, label="pleasant coding anchors")
    axes[1].scatter(un, np.full(len(un), top), marker="|", color="#d23f5c", s=120, label="unpleasant coding anchors")
    axes[1].set_yticks(range(len(order)), order, fontsize=8)
    axes[1].axvline(0, color="#555", lw=0.8)
    for x in (-1, 1):
        axes[1].axvline(x, color="#bbb", lw=0.8, ls=":")
    axes[1].set_xlabel("valence at the start of the reply (-1 = unpleasant coding anchors' mean, +1 = pleasant)")
    axes[1].set_title("Items on the valence direction (95% CI over 6 templates)", fontsize=10)
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    paths.append(out / "s2_valence.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    if "steering_effects" in results:
        eff = results["steering_effects"]
        designs = sorted(eff, key=lambda k: (len(eff[k]["layers"]), eff[k]["layers"][0]))
        fig, ax = plt.subplots(figsize=(8, 4.4))
        for offset, kind, colour in ((-0.15, "valence", "#2c66d4"), (0.15, "random", "#999")):
            m = np.array([eff[d][kind]["mean"] for d in designs])
            ci = np.array([eff[d][kind]["ci"] for d in designs])
            x = np.arange(len(designs)) + offset
            ax.errorbar(x, m, yerr=[m - ci[:, 0], ci[:, 1] - m], fmt="o", color=colour, capsize=3, label=kind)
        labels = []
        for d in designs:
            ls = eff[d]["layers"]
            where = f"layer {ls[0]}" if len(ls) == 1 else f"layers {ls[0]}-{ls[-1]}"
            labels.append(f"{where}\nalpha +/-{eff[d]['alpha']:g}")
        ax.set_xticks(range(len(designs)), labels, fontsize=8)
        ax.axhline(0, color="#555", lw=0.8)
        ax.set_ylabel("change in log-odds of choosing Apex\n(+alpha minus -alpha, paired over 16 tasks)")
        ax.set_title("Steering the 'Apex' option's tokens in 'which would you enjoy?' (Apex vs Python)", fontsize=10)
        ax.legend(title="direction added")
        fig.tight_layout()
        paths.append(out / "s2_steering.png")
        fig.savefig(paths[-1], dpi=150)
        plt.close(fig)
    return paths


def rng_jitter(n: int) -> np.ndarray:
    return np.random.default_rng(n).uniform(-0.18, 0.18, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimuli", type=Path, default=HERE / "stimuli.yaml")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--stages",
        nargs="+",
        default=list(STAGES),
        choices=list(STAGES),
        help="stages to run; the others are kept from an earlier s2_results.json",
    )
    ap.add_argument("--steer-layers", type=int, nargs="+", help="steer at each of these layers alone instead")
    ap.add_argument("--plot-only", action="store_true", help="redraw the figures from s2_results.json")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    stim = yaml.safe_load(args.stimuli.read_text())
    path = args.out / "s2_results.json"
    if args.plot_only:
        results = json.loads(path.read_text())
        results["contrasts"] = contrasts(results, rng)
        results["steering_effects"] = steering_effects(results["steering"], rng)
        path.write_text(json.dumps(results, indent=1))
        for p in plot(results, stim, args.out):
            print(p)
        return

    started = time.perf_counter()
    subject = load()
    results: dict = json.loads(path.read_text()) if path.exists() and set(args.stages) != set(STAGES) else {}
    results.update({"model": "Qwen3.8 27B (MLX 8-bit)", "n_layers": subject.n_layers})
    results.setdefault("timings_s", {})

    def timed(name: str, fn):
        t0 = time.perf_counter()
        value = fn()
        results["timings_s"][name] = round(time.perf_counter() - t0, 1)
        print(f"{name}: {results['timings_s'][name]} s", flush=True)
        return value

    if "choices" in args.stages:
        results["choices"] = timed("choices", lambda: run_choices(subject, stim, rng))
    if "lens" in args.stages:
        results["lens"] = timed("lens", lambda: run_lens(subject, stim, rng))
    if "valence" in args.stages:
        results["valence"], saved = timed("valence", lambda: run_valence(subject, stim, rng))
        np.savez(args.out / "valence.npz", **saved)
    else:
        saved = dict(np.load(args.out / "valence.npz"))
    if "steering" in args.stages:
        steering = timed("steering", lambda: run_steering(subject, stim, saved, rng, args.steer_layers))
        kept = {k: v for k, v in results.get("steering", {}).items() if isinstance(v, dict)}
        results["steering"] = {**kept, **steering}
    if "quotes" in args.stages:
        results["quotes"] = timed("quotes", lambda: run_quotes(subject, stim))
    if "lens" in results and "valence" in results:
        results["contrasts"] = contrasts(results, rng)
    if "steering" in results:
        results["steering_effects"] = steering_effects(results["steering"], rng)
    results["runtime_s"] = round(sum(results["timings_s"].values()), 1)  # model compute over every stage
    print(f"this run: {time.perf_counter() - started:.0f} s")
    (args.out / "s2_results.json").write_text(json.dumps(results, indent=1))
    for p in plot(results, stim, args.out):
        print(p)
    print(args.out / "s2_results.json")


if __name__ == "__main__":
    main()
