"""The study file forcebench.ai's Inside the model page reads: aggregates only, from the three studies' outputs.

Every number the page shows or charts comes from here. Per-pair and per-task detail stays in each study's own
results; nothing here quotes a task prompt or a model's answer to one.

    uv run python scripts/export_site.py --outputs outputs --out results/inside-the-model.json
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

REPO = "https://github.com/leo-dcfa/forcebench-mri"
BOOT = 2000
GEN = Path(__file__).parent.parent / "studies" / "s1_latent_java" / "generation.yaml"
EXAMPLE = "test-discount"


def head(text: str, lines: int = 9) -> str:
    """The first lines of a reply's first code block."""
    body = text.split("```", 2)[1].split("\n", 1)[1] if text.lstrip().startswith("```") else text
    return "\n".join(body.splitlines()[:lines])


def r(x, nd: int = 4):
    """Rounded for the file: floats to ``nd`` places, recursively; NaN as null."""
    if isinstance(x, float | np.floating):
        return None if np.isnan(x) else round(float(x), nd)
    if isinstance(x, dict):
        return {k: r(v, nd) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [r(v, nd) for v in x]
    if isinstance(x, np.ndarray):
        return r(x.tolist(), nd)
    if isinstance(x, np.integer):
        return int(x)
    return x


def median_band(values: np.ndarray, rng: np.random.Generator) -> dict:
    """Per column (layer) median of rows (pairs), with a 95% percentile bootstrap interval over rows."""
    idx = rng.integers(0, values.shape[0], size=(BOOT, values.shape[0]))
    draws = np.stack([np.median(values[i], axis=0) for i in idx])
    return {
        "median": np.median(values, axis=0),
        "lo": np.percentile(draws, 2.5, axis=0),
        "hi": np.percentile(draws, 97.5, axis=0),
    }


def share_band(flags: np.ndarray, rng: np.random.Generator) -> dict:
    idx = rng.integers(0, flags.shape[0], size=(BOOT, flags.shape[0]))
    draws = np.stack([flags[i].mean(axis=0) for i in idx])
    return {
        "share": flags.mean(axis=0),
        "lo": np.percentile(draws, 2.5, axis=0),
        "hi": np.percentile(draws, 97.5, axis=0),
    }


def q1(outputs: Path, rng: np.random.Generator) -> dict:
    a = json.loads((outputs / "s1" / "s1_results.json").read_text())
    b = json.loads((outputs / "s1b" / "s1b_results.json").read_text())
    pairs = [p for p in b["pairs"] if p["has_neutral"]]
    lens = {c: np.array([p["lens_log_odds"][c] for p in pairs]) for c in ("apex", "java", "neutral")}
    final = {c: np.array([p["final_log_odds"][c] for p in b["pairs"]]) for c in ("apex", "java")}
    steer = b["b4_pairs"]
    wins = steer["apex_wins"]
    kinds = dict.fromkeys(p["kind"] for p in a["pairs"])
    out = {
        "n_pairs": b["n_pairs"],
        "n_layers": b["n_layers"],
        "kinds": len(kinds),
        "lens": {c: median_band(lens[c], rng) for c in lens},
        "lens_n_pairs": len(pairs),
        "apex_leads": {c: share_band(lens[c] > 0, rng) for c in lens},
        "java_band": a["java_band_apex_context"],
        "neutral_final_prefers_java": b["b1"]["neutral_final_prefers_java_share"],
        "corr_apex_neutral": b["b1"]["corr_apex_neutral"],
        "apex_minus_neutral": b["b1"]["apex_minus_neutral_median"],
        "by_kind": [
            {
                "kind": k,
                "n": a["summary"]["apex"]["by_kind"][k]["n"],
                "java_leads_l54": a["summary"]["apex"]["by_kind"][k]["java_leads_share"][54],
                "final_apex_median": a["summary"]["apex"]["by_kind"][k]["median"][-1],
                "neutral_final_median": b["b1"]["by_kind_neutral_final_median"][k],
            }
            for k in kinds
        ],
        "wrong_final_apex": [w for w in a["wrong_final"] if w["context"] == "apex"],
        "final_apex_wins": int((final["apex"] > 0).sum()),
        "final_java_wins_in_java": int((final["java"] < 0).sum()),
        "probe": {"accuracy": b["b2"]["accuracy"], "neutral_p_apex": b["b2"]["neutral_p_apex"]},
        "patching": {
            "layers": b["b3"]["layers"],
            **{
                key: {
                    where: {
                        "median": b["b3"]["median"][key][where],
                        "lo": b["b3"]["median_ci"][key][where][0],
                        "hi": b["b3"]["median_ci"][key][where][1],
                    }
                    for where in ("last", "span")
                }
                for key in ("apex<-java", "java<-apex")
            },
        },
        "steering": {
            "layers": steer["layers"],
            "median_effect": steer["median_effect"],
            "apex_wins": wins,
        },
    }
    gen = outputs / "s1b" / "s1b_generation.json"
    if gen.exists():
        g = json.loads(gen.read_text())
        fences = {
            name: {
                lang: sum(p["conditions"][name]["text"].lstrip().startswith(f"```{lang}") for p in g["prompts"])
                for lang in ("apex", "java")
            }
            for name in g["totals"]
        }
        # One request's opening lines, unsteered and with the direction taken away (a request of the study's
        # own, not a Forcebench task).
        ex = next(p for p in g["prompts"] if p["id"] == EXAMPLE)
        out["generation"] = {
            "layer": g["layer"],
            "scale": g["scale"],
            "n_prompts": len(g["prompts"]),
            "totals": g["totals"],
            "fences": fences,
            "example": {
                "ask": next(x["ask"] for x in yaml.safe_load(GEN.read_text())["prompts"] if x["id"] == EXAMPLE),
                **{k: head(ex["conditions"][k]["text"]) for k in ("none", "-apex_dir")},
            },
        }
    return out


def q2(outputs: Path) -> dict:
    s = json.loads((outputs / "s2" / "s2_results.json").read_text())
    ch = s["choices"]["summary"]
    v = s["valence"]
    lay = v["layer"]
    return {
        "languages": {
            pair: {
                group: {
                    frame: {"p": d[f"{frame}_p_first"]["mean"], "ci": d[f"{frame}_p_first"]["ci"]}
                    for frame in ("enjoy", "fit")
                }
                | {"n": d["enjoy_p_first"]["n"], "enjoy_minus_fit": d["enjoy_minus_fit_p"]}
                for group, d in groups.items()
                if group in ("generic", "salesforce")
            }
            for pair, groups in ch["languages"].items()
        },
        "clicks": {
            group: {
                frame: {"p": d[f"{frame}_p_code"]["mean"], "ci": d[f"{frame}_p_code"]["ci"]}
                for frame in ("enjoy", "fit")
            }
            | {"n": d["enjoy_p_code"]["n"]}
            for group, d in ch["clicks"].items()
            if group in ("flow", "apex")
        },
        "letters_mass_min": ch["min_mass_ab"],
        "letters_mass_median": ch["median_mass_ab"],
        "valence": {
            "layer": lay,
            "n_statements": v["n_statements"],
            "heldout_activity_acc": v["heldout_activity_acc"][lay],
            "heldout_template_acc": v["heldout_template_acc"][lay],
            "everyday_anchor_auroc": v["everyday_anchor_auroc"][lay],
            "coding_anchor_auroc": v["coding_anchor_auroc"][lay],
            "coding_anchor_d": v["coding_anchor_d"][lay],
            "items": {k: {"score": d["coding"]["score"], "ci": d["coding"]["ci"]} for k, d in v["items"].items()},
            "n_templates": len(next(iter(v["items"].values()))["coding_by_template"]),
            "anchors": v["anchors_on_coding_scale"],
        },
        "contrasts": s["contrasts"],
        "steering": s["steering_effects"],
        # The taste lens at the last layer, each template's mean over items subtracted.
        "lens_final": {
            k: {"mean": d["centred_mean"][-1], "ci": [d["centred_ci"][0][-1], d["centred_ci"][1][-1]]}
            for k, d in s["lens"]["summary"].items()
        },
        "quotes": s["quotes"],
    }


def q3(outputs: Path, rng: np.random.Generator) -> dict:
    s = json.loads((outputs / "s3" / "s3_results.json").read_text())
    pe, ae = s["probe"]["prompt_end"], s["probe"]["answer_end"]
    fixed = str(s["fixed_layer"])
    conf, enj, val = s["stated"]["confidence"], s["stated"]["enjoyment"], s["valence"]
    tasks = s["per_task"]
    # Stated confidence in six bins of equal size, and the share of each that passed.
    order = sorted(tasks, key=lambda t: t["stated_confidence"])
    bins = []
    for chunk in np.array_split(np.arange(len(order)), 6):
        group = [order[i] for i in chunk]
        passed = np.array([t["passed"] for t in group], dtype=float)
        draws = passed[rng.integers(0, len(passed), size=(BOOT, len(passed)))].mean(axis=1)
        bins.append(
            {
                "n": len(group),
                "confidence": float(np.mean([t["stated_confidence"] for t in group])),
                "passed": passed.mean(),
                "ci": [np.percentile(draws, 2.5), np.percentile(draws, 97.5)],
            }
        )
    suites = sorted({t["suite"] for t in tasks})
    by_suite = [
        {
            "suite": su,
            "n": enj["by_suite"][su]["n"],
            "pass_rate": enj["by_suite"][su]["pass_rate"],
            "confidence": enj["by_suite"][su]["confidence"],
            "enjoyment": enj["by_suite"][su]["mean"],
            "enjoyment_ci": enj["by_suite"][su]["ci"],
            "valence": val["by_suite"][su]["mean"],
            "valence_ci": val["by_suite"][su]["ci"],
        }
        for su in suites
    ]
    ivs = s["inside_vs_says"]
    return {
        "run_id": s["run"],
        "n_tasks": s["n_tasks"],
        "n_passed": s["n_passed"],
        "fixed_layer": s["fixed_layer"],
        "auroc_by_layer": {
            "prompt_end": {"mean": pe["auroc_mean"], "lo": pe["auroc_ci"][0], "hi": pe["auroc_ci"][1]},
            "answer_end": {"mean": ae["auroc_mean"], "lo": ae["auroc_ci"][0], "hi": ae["auroc_ci"][1]},
        },
        "null_best_layer_p95": pe["null"]["max_p95"],
        "null_p_best": pe["null"]["p_best"],
        "null_perms": pe["null"]["perms"],
        "probe": {
            "fixed": pe["fixed"],
            "best_layer": pe["best_layer"],
            "best": pe["best"],
            "answer_fixed": ae["fixed"],
            "answer_best_layer": ae["best_layer"],
            "answer_best": ae["best"],
            "plus_baseline_vs_baseline": s["incremental"]["prompt_end"][fixed]["vs_baseline"],
        },
        "baseline": s["baseline"],
        "soft": {
            "model": s["soft_label"]["model"],
            "runs": len(s["soft_label"]["runs"]),
            "samples_per_task_min": s["soft_label"]["samples_per_task_min"],
            "auroc": s["soft_label"]["vs_pass"]["auroc"],
            "auroc_ci": s["soft_label"]["vs_pass"]["auroc_ci"],
            "probe_spearman": s["ridge_soft"]["prompt_end"]["fixed"],
            "baseline_spearman": s["ridge_soft"]["baseline"],
        },
        "confidence": {
            "question": conf["question"],
            "mean": conf["mean"],
            "mean_passed": conf["mean_passed"],
            "mean_failed": conf["mean_failed"],
            "vs_pass": conf["vs_pass"],
            "vs_probe": conf["vs_probe_fixed"],
            "bins": bins,
        },
        "probe_minus_stated": ivs["probe_minus_stated"],
        "stated_minus_baseline": ivs["stated_vs_baseline"],
        "probe_over_baseline_and_stated": ivs["baseline_plus_stated_plus_probe_vs_baseline_plus_stated"],
        "enjoyment": {
            "question": enj["question"],
            "mean": enj["mean"],
            "sd": float(np.std([t["stated_enjoyment"] for t in tasks])),
            "vs_pass": enj["vs_pass"],
            "vs_difficulty": enj["vs_difficulty"],
            "vs_confidence": enj["vs_confidence"],
        },
        "valence": {
            "layer": val["layer"],
            "mean": val["mean"],
            "vs_pass": val["vs_pass"],
            "vs_enjoyment": val["vs_stated_enjoyment"],
            "vs_confidence": val["vs_stated_confidence"],
        },
        "by_suite": by_suite,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", type=Path, default=Path("outputs"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    data = {
        "study": "inside-the-model",
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repo": REPO,
        "model": {
            "name": "Qwen3.8 27B",
            "weights": "MLX 8-bit",
            "config_id": "qwen3.8-27b-mlx-8bit",
            "model_id": "qwen3-8-27b",
            "layers": 64,
        },
        "q1": q1(args.outputs, rng),
        "q2": q2(args.outputs),
        "q3": q3(args.outputs, rng),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(r(data), indent=1) + "\n")
    print(args.out, f"{args.out.stat().st_size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
