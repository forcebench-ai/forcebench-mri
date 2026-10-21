"""The hooks on a tiny, randomly initialised model of the same family (no downloads)."""

import mlx.core as mx
import pytest
from mlx_lm.models import qwen3_5

from forcebench_mri.model import Hooks, Subject

TEXT = {
    "model_type": "qwen3_5_text",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "vocab_size": 100,
    "linear_num_value_heads": 4,
    "linear_num_key_heads": 2,
    "linear_key_head_dim": 32,
    "linear_value_head_dim": 16,
    "head_dim": 16,
}


@pytest.fixture(scope="module")
def subject():
    mx.random.seed(0)
    model = qwen3_5.Model(qwen3_5.ModelArgs(model_type="qwen3_5", text_config=TEXT))
    mx.eval(model.parameters())
    return Subject(model, tokenizer=None)


TOKENS = [1, 5, 9, 2, 7, 3]


def test_no_hooks_matches_the_plain_model(subject):
    plain = subject.model(mx.array([TOKENS]))
    assert mx.allclose(subject.run(TOKENS), plain).item()


def test_capture_and_logit_lens_of_the_last_layer_give_the_logits(subject):
    last = subject.n_layers - 1
    logits = subject.run(TOKENS, Hooks(capture=[0, last]))
    assert set(subject.captured) == {0, last}
    assert subject.captured[0].shape == (1, len(TOKENS), TEXT["hidden_size"])
    assert mx.allclose(subject.logit_lens(subject.captured[last]), logits, atol=1e-4).item()


def test_add_changes_only_later_positions(subject):
    base = subject.run(TOKENS)
    vector = mx.ones((TEXT["hidden_size"],)) * 5
    steered = subject.run(TOKENS, Hooks(add={1: ([3], vector)}))
    # Attention is causal, so positions before the edit are untouched and the edited one changes.
    assert mx.allclose(steered[:, :3], base[:, :3], atol=1e-5).item()
    assert not mx.allclose(steered[:, 3], base[:, 3], atol=1e-3).item()


def test_replace_patches_one_run_into_another(subject):
    other = [1, 5, 9, 2, 8, 3]
    subject.run(other, Hooks(capture=[2]))
    saved = subject.captured[2]
    # Overwriting every position at layer 2 with the other run's values makes the rest of the run identical.
    patched = subject.run(TOKENS, Hooks(replace={2: (slice(None), saved)}))
    assert mx.allclose(patched, subject.run(other), atol=1e-4).item()
    # Patching only position 4 at layer 0 (where the inputs differ) makes positions 0-4 match the other run.
    subject.run(other, Hooks(capture=[0]))
    patched4 = subject.run(TOKENS, Hooks(replace={0: ([4], subject.captured[0][:, 4:5])}))
    assert mx.allclose(patched4[:, :5], subject.run(other)[:, :5], atol=1e-4).item()
