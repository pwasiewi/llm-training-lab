"""
LESSON 4 — Where does the VRAM go? Training memory, by hand.

No GPU, no torch — just arithmetic. Being able to run this math on a napkin
tells you in advance whether a training run fits on a 16 GB card or needs
gradient checkpointing / a smaller batch / QLoRA. The estimator models the
lesson-2/3 encoder, but the formulas are the same for any transformer.

The four consumers of training memory:

  1. WEIGHTS         n_params * bytes_per_param.
  2. GRADIENTS       one more copy of every weight, same dtype as training.
  3. OPTIMIZER       AdamW keeps TWO fp32 statistics per weight (momentum m
                     and variance v). In mixed precision it also keeps an
                     fp32 MASTER copy of the weights, because accumulating
                     tiny updates directly into bf16 weights loses them
                     (bf16 has ~3 significant digits).
                     => bf16 training: 2 (weights) + 2 (grads)
                        + 4+4 (m, v) + 4 (master) = 16 bytes per parameter.
                     "My 8 GB card fits a 7B model" — for inference, yes;
                     for full AdamW training you need ~112 GB. This factor-16
                     is why LoRA exists.
  4. ACTIVATIONS     every intermediate tensor saved for backward. Scales
                     with batch * seq_length, and the attention matrix piece
                     scales with batch * heads * seq_length^2 — the ONLY
                     quadratic term, and the reason long context is hard.

Sliding-window attention (lesson 2's local layers) replaces seq^2 with
seq * window per local layer — that one substitution is what the
`--window` comparison below quantifies. FlashAttention removes the
materialized matrix altogether; the printout notes what that changes.

Run:  python simple_04_vram_math.py
      python simple_04_vram_math.py --dtype fp32 --seq 8192 --batch 8
"""

import argparse

MB = 1024 * 1024
GB = 1024 * MB

DTYPE_BYTES = {"fp32": 4, "bf16": 2}


class EncoderShape:
    """The lesson-2 encoder, plus CLI-overridable size knobs."""

    def __init__(self, vocab_size=30000, hidden_size=256, num_heads=4,
                 num_layers=6, seq_length=512, batch_size=32,
                 local_window=128, global_every=3):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.local_window = local_window
        self.global_every = global_every


def count_parameters(shape):
    """
    Parameter counts follow directly from the layer shapes (biases and
    layer norms are <0.1% here and are folded into a small constant):
      embeddings : vocab * hidden  (+ seq * hidden if learned positions)
      attention  : 4 * hidden^2 per layer (qkv is 3, output proj is 1)
      ffn        : 8 * hidden^2 per layer (up 4x + down 4x)
    """
    h = shape.hidden_size
    parts = {
        "token embeddings": shape.vocab_size * h,
        "position embeddings": shape.seq_length * h,
        "attention (all layers)": shape.num_layers * 4 * h * h,
        "ffn (all layers)": shape.num_layers * 8 * h * h,
        "norms + classifier": shape.num_layers * 4 * h + h * (h // 2) + h // 2,
    }
    return parts, sum(parts.values())


def attention_matrix_bytes(shape, dtype_bytes, window=None):
    """
    The materialized softmax(QK^T) matrix, per layer:
      global : batch * heads * seq * seq      entries
      local  : batch * heads * seq * window   entries (each query only
               scores `window` keys — the seq^2 term is GONE)
    """
    keys_per_query = shape.seq_length if window is None else min(window, shape.seq_length)
    entries = shape.batch_size * shape.num_heads * shape.seq_length * keys_per_query
    return entries * dtype_bytes


def estimate(shape, dtype="bf16", optimizer="adamw_mixed", window=None):
    """
    Returns a dict of MB figures. `window=None` = all layers global;
    otherwise lesson 2's layout: every `global_every`-th layer global,
    the rest sliding-window.
    """
    act_bytes = DTYPE_BYTES[dtype]
    _, n_params = count_parameters(shape)

    if optimizer == "adamw_mixed" and dtype == "bf16":
        # bf16 weights+grads, fp32 m+v+master: the 16-bytes-per-param rule.
        weights = n_params * 2
        grads = n_params * 2
        opt_states = n_params * (4 + 4 + 4)
    else:
        # plain fp32 training: 4 (w) + 4 (g) + 8 (m+v) = 16 too — mixed
        # precision saves ACTIVATION memory and speed, not optimizer memory.
        weights = n_params * 4
        grads = n_params * 4
        opt_states = n_params * 8

    # Residual-stream activations kept for backward: input to every
    # sublayer, ~2 per block (attention in, ffn in) + ffn's 4x inner tensor.
    tokens = shape.batch_size * shape.seq_length
    per_layer = (2 * tokens * shape.hidden_size          # sublayer inputs
                 + tokens * 4 * shape.hidden_size)       # ffn inner (4x)
    stream_acts = shape.num_layers * per_layer * act_bytes

    # Attention matrices, the quadratic (or windowed-linear) term:
    if window is None:
        n_global, n_local = shape.num_layers, 0
    else:
        n_global = len([i for i in range(shape.num_layers)
                        if i % shape.global_every == 0])
        n_local = shape.num_layers - n_global
    attn_acts = (n_global * attention_matrix_bytes(shape, act_bytes)
                 + n_local * attention_matrix_bytes(shape, act_bytes, window))

    total = weights + grads + opt_states + stream_acts + attn_acts
    return {
        "weights": weights / MB,
        "gradients": grads / MB,
        "optimizer states": opt_states / MB,
        "activations (residual stream)": stream_acts / MB,
        "activations (attention matrices)": attn_acts / MB,
        "TOTAL": total / MB,
    }, n_params


def print_estimate(title, breakdown):
    print(f"\n{title}")
    print("-" * 62)
    for key, mb in breakdown.items():
        marker = "=" if key == "TOTAL" else " "
        print(f" {marker} {key:<34} {mb:>10.1f} MB  ({mb/1024:.2f} GB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--window", type=int, default=128,
                        help="sliding-window width for the local-attention comparison")
    args = parser.parse_args()

    shape = EncoderShape(hidden_size=args.hidden, num_heads=args.heads,
                         num_layers=args.layers, seq_length=args.seq,
                         batch_size=args.batch, local_window=args.window)

    parts, n_params = count_parameters(shape)
    print(f"Model: hidden {shape.hidden_size}, {shape.num_layers} layers, "
          f"{shape.num_heads} heads, vocab {shape.vocab_size}")
    print(f"Parameters: {n_params/1e6:.2f}M")
    for name, n in parts.items():
        print(f"    {name:<24} {n/1e6:6.2f}M")
    print(f"\nTraining shape: batch {shape.batch_size}, seq {shape.seq_length}, "
          f"dtype {args.dtype}")
    print(f"Rule of thumb: full AdamW training costs ~16 bytes/param "
          f"regardless of dtype\n  -> {n_params*16/MB:.0f} MB before any "
          f"activations. Mixed precision buys speed and\n  activation "
          f"memory, NOT optimizer memory.")

    breakdown, _ = estimate(shape, dtype=args.dtype)
    print_estimate(f"All {shape.num_layers} layers GLOBAL attention "
                   f"(seq {shape.seq_length})", breakdown)

    breakdown_local, _ = estimate(shape, dtype=args.dtype, window=shape.local_window)
    print_estimate(f"Alternating global/local, window {shape.local_window} "
                   f"(seq {shape.seq_length})", breakdown_local)

    # The punchline: same model at long context. At seq 8192 the global
    # attention matrices dominate EVERYTHING else; the windowed version's
    # attention term grows only linearly.
    long_shape = EncoderShape(hidden_size=shape.hidden_size,
                              num_heads=shape.num_heads,
                              num_layers=shape.num_layers,
                              seq_length=8192, batch_size=shape.batch_size,
                              local_window=shape.local_window)
    long_global, _ = estimate(long_shape, dtype=args.dtype)
    long_local, _ = estimate(long_shape, dtype=args.dtype,
                             window=long_shape.local_window)
    print_estimate("Same model, seq 8192, all-global", long_global)
    print_estimate(f"Same model, seq 8192, window {long_shape.local_window}",
                   long_local)

    ratio = long_global["TOTAL"] / long_local["TOTAL"]
    print(f"\nAt seq 8192 the windowed layout needs {ratio:.1f}x less total "
          f"memory —\nthat is ModernBERT's long-context trick, quantified.")
    print("\nCaveat: FlashAttention (used by every serious framework) never")
    print("materializes the attention matrix at all, trading memory for")
    print("recomputation — the 'attention matrices' rows above drop to ~0,")
    print("but compute TIME still scales with seq^2 for global layers.")
    print("Lesson 7 measures that time scaling empirically.")


if __name__ == "__main__":
    main()
