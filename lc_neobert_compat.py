"""Compatibility shims for chandar-lab/NeoBERT under transformers 5 + PEFT.

NeoBERT ships its model definition as remote code (trust_remote_code=True), and
two things break when that code meets current transformers and PEFT. Neither is
NeoBERT's fault exactly, and only one of them announces itself.

Import and call `patch_neobert(model)` on the model returned by
`from_pretrained`, BEFORE `get_peft_model`.

Both fixes are also needed on the INFERENCE path, not just for training: a saved
adapter re-loaded without `patch_neobert` scores like a coin flip.

---------------------------------------------------------------------------
1. inputs_embeds  (loud: TypeError on the first forward)

`PeftModelForSequenceClassification.forward` passes `inputs_embeds=inputs_embeds`
unconditionally, but NeoBERT's forward signature is

    (input_ids, position_ids, max_seqlen, cu_seqlens, attention_mask,
     output_hidden_states, output_attentions, labels, return_dict)

with no `inputs_embeds`, so every forward dies with

    TypeError: NeoBERTForSequenceClassification.forward() got an unexpected
    keyword argument 'inputs_embeds'

Everything else PEFT passes IS in that signature, so swallowing this one
argument is the whole fix. It is rejected rather than ignored when actually
supplied, because NeoBERT has no embeds-input path at all and silently dropping
real embeddings would train on the wrong thing.

The `functools.wraps` is NOT cosmetic and must not be dropped. Trainer decides
which dataset columns to keep by inspecting the signature of the base model's
forward (it unwraps PeftModel via `get_base_model()` specifically to do this). A
bare wrapper advertises `(self, *args, inputs_embeds=None, **kwargs)`, so
`input_ids` and `attention_mask` are not "in the signature", get silently
dropped as unused columns, and training dies at the first batch with

    ValueError: You should supply an encoding ... but you provided ['label']

`wraps()` sets `__wrapped__`, which `inspect.signature` follows, so the original
NeoBERT signature is what everything downstream sees.

---------------------------------------------------------------------------
2. freqs_cis  (silent: the model trains fine and loses all positional info)

NeoBERT computes `freqs_cis` in `NeoBERT.__init__` and registers it with
`persistent=False`, so it is not in the checkpoint. transformers 5 builds the
model under a meta device and materializes it from the state dict; a
non-persistent buffer has nothing to materialize FROM, so it comes back as
whatever was in the allocation. Measured across repeated loads of the same
checkpoint, same input, same device:

    |freqs_cis| absmax = 1.658e-39   (denormal noise)
    |freqs_cis| absmax = 3.251e+35   (huge)
    ... and sometimes NaN, which is the only case that announces itself

`torch.polar(ones_like(freqs), freqs)` guarantees `|freqs_cis| == 1.0` exactly,
so any other magnitude is corruption. The NaN loads (2 of 6 in one sample) are
the LUCKY ones: they crash. The rest run happily with rotary embeddings
multiplied by garbage — the model silently loses all positional information and
still returns plausible activations and a trainable loss. A run that merely
scored badly would look like "NeoBERT is not that good".

The buffer is recomputed with the model's own function, taken from the
dynamically loaded remote module so it cannot drift from the model definition,
and then asserted unit-magnitude so this can never regress quietly.
"""

import functools
import sys

import torch

__all__ = ["patch_neobert"]

# Marker attribute so repeated calls (one per Optuna trial, all sharing the same
# dynamically-loaded class object) do not stack wrappers 20 deep.
_PATCHED = "_lc_inputs_embeds_shim"


def patch_neobert(model, verbose=True):
    """Apply both NeoBERT shims in place. Returns the model, for chaining."""
    cls = type(model)
    if not getattr(cls, _PATCHED, False):
        _neobert_forward = cls.forward

        @functools.wraps(_neobert_forward)
        def _forward_without_inputs_embeds(self, *args, inputs_embeds=None, **kwargs):
            if inputs_embeds is not None:
                raise ValueError("NeoBERT accepts no inputs_embeds; pass input_ids")
            return _neobert_forward(self, *args, **kwargs)

        cls.forward = _forward_without_inputs_embeds
        setattr(cls, _PATCHED, True)

    backbone = model.model  # NeoBERT inside NeoBERTForSequenceClassification
    rotary = sys.modules[type(backbone).__module__]
    backbone.freqs_cis = rotary.precompute_freqs_cis(
        model.config.hidden_size // model.config.num_attention_heads,
        model.config.max_length,
    ).to(backbone.freqs_cis.device)

    mag = torch.view_as_real(backbone.freqs_cis).pow(2).sum(-1).sqrt()
    assert torch.allclose(mag, torch.ones_like(mag), atol=1e-5), (
        f"freqs_cis is not unit-magnitude (min {mag.min():.3e}, max {mag.max():.3e}) "
        "— the rotary buffer is corrupt and this model would run on garbage positions")
    if verbose:
        print(f"freqs_cis recomputed: |z| in [{mag.min():.6f}, {mag.max():.6f}]")
    return model
