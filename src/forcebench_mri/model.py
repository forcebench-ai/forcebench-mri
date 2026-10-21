"""Load a model with MLX and look inside it: capture, steer and patch the residual stream.

Every decoder layer is wrapped so that, during ``Subject.run``, its output (the residual stream after
that layer) can be read, added to or overwritten. ``logit_lens`` decodes any residual vector the way
the model decodes its last one: final norm, then the unembedding.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx_lm import load as mlx_load
from mlx_lm.models.cache import make_prompt_cache

# Positions along the sequence: a slice or explicit indices.
Positions = slice | Sequence[int]
LayerFn = Callable[[int, mx.array], mx.array]


@dataclass
class Hooks:
    """What to do to the residual stream at each layer during one run.

    ``capture``: layers whose output is saved. ``add``: layer -> (positions, vector) added there.
    ``replace``: layer -> (positions, values) written there. ``fn``: called last on every layer's output.
    """

    capture: Sequence[int] = ()
    add: dict[int, tuple[Positions, mx.array]] = field(default_factory=dict)
    replace: dict[int, tuple[Positions, mx.array]] = field(default_factory=dict)
    fn: LayerFn | None = None


class HookedLayer(nn.Module):
    """A decoder layer whose output passes through the subject's active hooks."""

    def __init__(self, layer: nn.Module, index: int, subject: Subject):
        super().__init__()
        self.layer = layer
        self.index = index
        self._subject = subject

    @property
    def is_linear(
        self,
    ) -> bool:  # read by the qwen3_5 forward pass to pick each layer's mask
        return getattr(self.layer, "is_linear", False)

    def __call__(self, x, mask=None, cache=None):
        return self._subject._after_layer(self.index, self.layer(x, mask=mask, cache=cache))


def _positions(out: mx.array, pos: Positions) -> mx.array:
    """A [1, seq, 1] mask that is 1 at ``pos``."""
    seq = out.shape[1]
    idx = range(seq)[pos] if isinstance(pos, slice) else pos
    mask = mx.zeros((seq,), dtype=out.dtype)
    mask[mx.array(list(idx))] = 1
    return mask[None, :, None]


class Subject:
    """A language model with hooks on every decoder layer."""

    def __init__(self, model: nn.Module, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        text = model.language_model if hasattr(model, "language_model") else model
        self.text = text  # embed_tokens, layers, norm and lm_head live under text.model / text
        layers = text.model.layers
        for i, layer in enumerate(layers):
            if not isinstance(layer, HookedLayer):
                layers[i] = HookedLayer(layer, i, self)
        self.n_layers = len(layers)
        self._hooks: Hooks | None = None
        self.captured: dict[int, mx.array] = {}

    def _after_layer(self, index: int, out: mx.array) -> mx.array:
        hooks = self._hooks
        if hooks is None:
            return out
        if index in hooks.replace:
            pos, values = hooks.replace[index]
            mask = _positions(out, pos)
            out = out * (1 - mask) + _spread(values, out, pos) * mask
        if index in hooks.add:
            pos, vector = hooks.add[index]
            out = out + _positions(out, pos) * vector.astype(out.dtype)
        if hooks.fn is not None:
            out = hooks.fn(index, out)
        if index in hooks.capture:
            self.captured[index] = out
        return out

    def run(self, tokens: mx.array | Sequence[int], hooks: Hooks | None = None) -> mx.array:
        """Logits for ``tokens`` ([seq] or [batch, seq]) with ``hooks`` applied; captures go to ``captured``."""
        tokens = mx.array(tokens)
        if tokens.ndim == 1:
            tokens = tokens[None]
        self._hooks = hooks
        self.captured = {}
        try:
            return self.model(tokens)
        finally:
            self._hooks = None

    def generate(
        self, tokens: Sequence[int], hooks: Hooks | None = None, *, max_tokens: int = 256, stop: Sequence[int] = ()
    ) -> list[int]:
        """Greedy continuation of ``tokens`` with ``hooks`` applied to the prompt and to every new token.

        Each step runs only the new token through the cache, so a hook's positions are relative to that step:
        ``slice(None)`` reaches every position, prompt and continuation alike.
        """
        cache = make_prompt_cache(self.model)
        self._hooks = hooks
        out: list[int] = []
        try:
            logits = self.model(mx.array(tokens)[None], cache=cache)
            for _ in range(max_tokens):
                nxt = int(mx.argmax(logits[0, -1]).item())
                if nxt in stop:
                    break
                out.append(nxt)
                logits = self.model(mx.array([[nxt]]), cache=cache)
        finally:
            self._hooks = None
        return out

    def logit_lens(self, hidden: mx.array) -> mx.array:
        """Decode residual vectors as if they were the last layer's: final norm, then unembedding."""
        normed = self.text.model.norm(hidden)
        if getattr(self.text.args, "tie_word_embeddings", False):
            return self.text.model.embed_tokens.as_linear(normed)
        return self.text.lm_head(normed)

    def chat_tokens(self, user: str, *, system: str | None = None, assistant_prefix: str = "") -> list[int]:
        """Token ids of a chat turn with thinking off, ending with the start of the assistant's reply."""
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return self.tokenizer.encode(text + assistant_prefix, add_special_tokens=False)


def _spread(values: mx.array, out: mx.array, pos: Positions) -> mx.array:
    """``values`` for the positions ``pos``, laid out at full sequence length (zeros elsewhere)."""
    seq = out.shape[1]
    idx = list(range(seq)[pos] if isinstance(pos, slice) else pos)
    values = values.reshape(out.shape[0], len(idx), out.shape[-1]).astype(out.dtype)
    full = mx.zeros(out.shape, dtype=out.dtype)
    full[:, mx.array(idx)] = values
    return full


def model_path() -> Path:
    """The model directory, from ``MRI_MODEL_PATH`` (no default: weights live outside the repo)."""
    path = os.environ.get("MRI_MODEL_PATH")
    if not path:
        raise RuntimeError("Set MRI_MODEL_PATH to the model's directory (an MLX model, e.g. a Qwen3.8 27B pack).")
    return Path(path).expanduser()


def load(path: str | Path | None = None) -> Subject:
    """Load an MLX model and its tokenizer, and hook its decoder layers."""
    model, tokenizer = mlx_load(str(path or model_path()))
    model.eval()
    return Subject(model, tokenizer)
