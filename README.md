# forcebench-mri

An MRI of an open-weights model doing Salesforce work. [Forcebench](https://forcebench.ai) grades what
models write; this repo looks inside one while it works, with the logit lens, linear probes, activation
patching and steering.

The write-up, with every chart: **[forcebench.ai/studies/inside-the-model](https://forcebench.ai/studies/inside-the-model/)**.
The tools are explained on [Interpretability 101](https://forcebench.ai/interpretability-101/).

The model is Qwen3.8 27B (64 layers) in an MLX 8-bit build, the same weights Forcebench grades as
`qwen3.8-27b-mlx-8bit`. Everything runs with [MLX](https://github.com/ml-explore/mlx) on Apple silicon;
the 8-bit model needs about 30 GB of memory.

## Three questions

| Study | Question | Code |
|---|---|---|
| 1 | Does it think in Java while writing Apex? | [`studies/s1_latent_java`](studies/s1_latent_java) |
| 2 | Apex, Java or Python? Clicks or code? Taste against judgement | [`studies/s2_taste`](studies/s2_taste) |
| 3 | Does it like the task at hand, and does it know it'll fail? | [`studies/s3_task`](studies/s3_task) |

**1. Apex and Java.** 64 minimal pairs of code, each asked for as Apex, as Java and with no language named.
The model knows the language from the first layers (a linear probe reads it at 96% or better from layer 1),
but until about layer 50 of 64 its next-token guess follows the one it makes with no language named, and
that default is Java's in 92% of the pairs. Apex is applied late: patching the last position at layer 52
moves 76% of the decision. Taking the mean Apex-minus-Java direction away at layer 50 makes the model answer
Apex requests in Java (10 of 12 replies in a Java code block, 98 Java-isms against none unsteered); a random
direction of the same length changes nothing.

**2. Taste.** Asked which is the better choice for Salesforce work, it picks Apex essentially every time;
asked which it would enjoy, Apex over Python drops to 56%. A pleasant–unpleasant direction, built from
statements about everyday activities and validated on coding experiences it never saw, puts Python highest
of 15 languages and tools, Apex in the middle, Java just below it (not clearly different) and Process
Builder lowest. Adding the direction to "Apex" at middle layers nudges its stated preference up.

**3. The task at hand.** The 272 public Forcebench tasks, shown exactly as in a graded run of the same
weights (78 passed). A linear probe on the activations at the end of the prompt, before any reasoning,
tells passes from fails with an AUROC of 0.81 at a layer fixed in advance (95% interval 0.76–0.86), above
what the suite, difficulty and prompt length give (0.75); shuffled labels reach 0.62. Asked, the model
rates its chance of passing 7.6 out of 9 on average against a 29% pass rate, and that stated confidence
does worse than the suite (AUROC 0.64). It says it would enjoy almost every task the same (7.9 ± 0.4 out
of 9), passed or failed. Study 2's pleasant–unpleasant direction doesn't carry over to these long prompts.

## Running it

```bash
uv sync
uv run pytest                      # the hooks, on a tiny random model of the same family (no downloads)
export MRI_MODEL_PATH=<an MLX build of Qwen3.8 27B>

uv run python studies/s1_latent_java/run.py --out outputs/s1               # phase A: the logit lens
uv run python studies/s1_latent_java/phase_b.py pairs --out outputs/s1b    # neutral baseline, probe, patching, steering
uv run python studies/s1_latent_java/phase_b.py generation --out outputs/s1b
uv run python studies/s2_taste/run.py --out outputs/s2
uv run python scripts/export_site.py --out results/inside-the-model.json   # the aggregates the page reads
```

Study 3 reads Forcebench's public tasks and one graded run from a checkout of
[leo-dcfa/forcebench](https://github.com/leo-dcfa/forcebench); its scripts say how.

`src/forcebench_mri/model.py` is the instrument: it wraps every decoder layer so a run can read, add to or
overwrite the residual stream after any layer, decodes residuals with the logit lens, and generates
greedily with hooks applied to every new token. Gradients need `model.train()`: the linear-attention
kernel has no backward pass in eval mode.

## What's here

| Path | What it is |
|---|---|
| `studies/*/` | Each study's stimuli (YAML) and scripts |
| `results/` | Each study's results (aggregates and per-item numbers) and its figures |
| `results/inside-the-model.json` | The aggregates [forcebench.ai](https://forcebench.ai/studies/inside-the-model/) reads |
| `src/forcebench_mri/` | The hooked model |
| `tests/` | The hooks, tested on a tiny random model |

Activations and other bulky intermediates go to `outputs/`, which isn't committed. Nothing here comes from
Forcebench's private task pool, and no model reasoning traces are published: study 3 uses the public tasks
and the graded answers Forcebench already publishes.

## Licence

Apache-2.0.
