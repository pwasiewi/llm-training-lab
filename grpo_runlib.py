"""Run hygiene for the grpo_* training scripts (2026-10-20).

Implements the "run hygiene" block of llm/research/grpo-improvements-20261020.md, ideas taken from
Google's Kauldron trainer (Apache-2.0; re-implemented, no code copied):
  H1  write_manifest / update_manifest  -> <run>/run.json: config, overrides, env, git, versions, hashes
  H2  MetricsLog + metrics_callback      -> <run>/metrics.jsonl: one JSON row per log/tripwire/deveval event
  H3  check_env                          -> refuse unknown GRPO_* variables (a typo used to be silent)
  M2  derive_seed                        -> independent seed streams from one root seed
  H6  bf16_update_stats                  -> how much of the LoRA update would survive a bf16 merge, live
  H4  SmokeCheck                         -> post-run assertions for GRPO_SMOKE=1, non-zero exit on failure
Pure stdlib + torch (torch only inside bf16_update_stats). No network.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from importlib import metadata

# ---------------------------------------------------------------- H3: strict knob check

def check_env(known: set[str], prefix: str = "GRPO_") -> None:
    """Exit with a clear message when the environment carries an unknown PREFIX* variable."""
    unknown = sorted(k for k in os.environ if k.startswith(prefix) and k not in known)
    if unknown:
        close = {u: [k for k in known if _similar(u, k)] for u in unknown}
        hints = "; ".join(f"{u} (did you mean {', '.join(c)}?)" if c else u for u, c in close.items())
        sys.exit(f"[runlib] unknown {prefix}* variable(s): {hints}. Known: {', '.join(sorted(known))}")


def _similar(a: str, b: str) -> bool:
    a, b = a.rstrip("_"), b
    return a == b or a in b or b in a


def overrides(values: dict, defaults: dict) -> dict:
    """The knobs whose value differs from the script default."""
    return {k: v for k, v in values.items() if defaults.get(k, object()) != v}

# ---------------------------------------------------------------- M2: seed streams

def derive_seed(root, *names) -> int:
    """Stateless seed for one purpose: same (root, names) -> same seed, different names -> independent."""
    key = "/".join(str(x) for x in (root, *names))
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little") & 0x7FFFFFFF

# ---------------------------------------------------------------- H1: manifest

_PKGS = ("torch", "transformers", "trl", "peft", "unsloth", "unsloth_zoo", "vllm", "datasets",
         "accelerate", "bitsandbytes", "math-verify", "numpy", "flashinfer-python", "triton")


def _sha256(path: str | None) -> str | None:
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _cmd(*argv) -> str | None:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=20).stdout.strip() or None
    except Exception:
        return None


_LIVE_EBUILDS = ("sci-ml/caffe2", "dev-python/vllm", "sci-ml/unsloth", "sci-ml/unsloth-zoo", "sci-ml/trl", "sci-ml/peft",
                 "sci-ml/torchvision", "sci-ml/torchaudio")


def _live_commits() -> dict:
    """Git commit of each installed Gentoo live (-9999) ebuild: their Python metadata only says 9999."""
    import bz2, re
    out = {}
    for cp in _LIVE_EBUILDS:
        try:
            with bz2.open(f"/var/db/pkg/{cp}-9999/environment.bz2", "rt", errors="replace") as fh:
                m = re.search(r'EGIT_VERSION="([0-9a-f]+)"', fh.read())
            out[cp] = m.group(1) if m else None
        except OSError:
            continue
    return out


def environment_snapshot(repo_dir: str) -> dict:
    versions = {}
    for p in _PKGS:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = None
    gpu = _cmd("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader")
    cuda = torch_version = None
    try:
        import torch
        cuda, torch_version = torch.version.cuda, torch.__version__   # the live ebuild's metadata says 9999
    except Exception:
        pass
    return {
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "kernel": platform.release(),
        "packages": versions,
        "torch_version": torch_version,
        "torch_cuda": cuda,
        "live_commits": _live_commits(),
        "gpu": gpu,
        "git_head": _cmd("git", "-C", repo_dir, "rev-parse", "HEAD"),
        "git_dirty": bool(_cmd("git", "-C", repo_dir, "status", "--porcelain", "--untracked-files=no")),
    }


def write_manifest(run_dir: str, *, script: str, config: dict, defaults: dict, extra: dict | None = None,
                   hashed_files: dict | None = None, env_prefixes=("GRPO_", "CUDA", "PYTORCH", "VLLM_")) -> str:
    """Create <run_dir>/run.json once per launch. A resumed run keeps the first launch under 'launches'."""
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "run.json")
    launch = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "argv": sys.argv,
        "pid": os.getpid(),
        "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith(env_prefixes)},
        "environment": environment_snapshot(os.path.dirname(os.path.abspath(script))),
        "script_sha256": _sha256(script),
        "files_sha256": {k: _sha256(v) for k, v in (hashed_files or {}).items()},
    }
    doc = {}
    if os.path.isfile(path):
        with open(path) as fh:
            doc = json.load(fh)
    doc.setdefault("launches", []).append(launch)
    doc.update({"script": os.path.abspath(script), "config": config,
                "overrides": overrides(config, defaults), "status": "running"})
    doc.update(extra or {})
    _atomic_write(path, doc)
    return path


def update_manifest(run_dir: str, **fields) -> None:
    path = os.path.join(run_dir, "run.json")
    doc = {}
    if os.path.isfile(path):
        with open(path) as fh:
            doc = json.load(fh)
    doc.update(fields)
    doc["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _atomic_write(path, doc)


def _atomic_write(path: str, doc: dict) -> None:
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=1, default=str)
        fh.write("\n")
    os.replace(tmp, path)

# ---------------------------------------------------------------- H2: metrics stream

class MetricsLog:
    """Append-only JSON lines: {"t": epoch_s, "kind": ..., "step": ..., ...}. Queue gates read this, not stdout."""

    def __init__(self, run_dir: str):
        os.makedirs(run_dir, exist_ok=True)
        self.path = os.path.join(run_dir, "metrics.jsonl")

    def write(self, kind: str, step, /, **values) -> None:
        # kind/step are positional-only so a Trainer log dict may carry the same names; those and "t" land as log_<name>
        row = {"t": round(time.time(), 1), "kind": kind, "step": step}
        row.update({(f"log_{k}" if k in row else k): _plain(v) for k, v in values.items()})
        with open(self.path, "a") as fh:
            fh.write(json.dumps(row) + "\n")


def _plain(v):
    try:
        import torch
        if isinstance(v, torch.Tensor):
            return v.item() if v.numel() == 1 else v.tolist()
    except Exception:
        pass
    return v


def metrics_callback(log: MetricsLog):
    """A transformers TrainerCallback that copies every Trainer log dict into metrics.jsonl."""
    from transformers import TrainerCallback

    class _MetricsCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                row = dict(logs)
                row.setdefault("epoch", state.epoch)   # Trainer logs already carry epoch; passing both was a TypeError
                log.write("train", state.global_step, **row)

    return _MetricsCallback()

# ---------------------------------------------------------------- H6: live bf16 representability

def _bf16_half_ulp(x):
    """Half of the bf16 spacing at |x| (8 significant bits): 2^(floor(log2|x|) - 8)."""
    import torch
    ax = x.abs().float().clamp_min(2.0 ** -126)
    return torch.exp2(torch.floor(torch.log2(ax)) - 8)


def bf16_update_stats(model, max_modules: int = 8) -> dict | None:
    """For a few LoRA-wrapped Linear layers: what fraction of the update B@A*scale would survive
    rounding W + dW to bf16 (|dW| >= half a bf16 ULP of W), and the median |dW| / half-ULP ratio.
    On g1 (2026-09) only ~6% survived a merge; a live curve shows that during training."""
    import torch
    mods = [m for _, m in model.named_modules()
            if hasattr(m, "lora_A") and hasattr(m, "lora_B") and hasattr(m, "base_layer")
            and "default" in getattr(m, "lora_A", {})]
    if not mods:
        return None
    step = max(1, len(mods) // max_modules)
    survive, ratios, n_el = [], [], 0
    with torch.no_grad():
        for m in mods[::step][:max_modules]:
            a = m.lora_A["default"].weight.float()
            b = m.lora_B["default"].weight.float()
            scale = float(m.scaling["default"]) if hasattr(m, "scaling") else 1.0
            w = m.base_layer.weight
            if w.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                continue                      # quantised base: not meaningful
            dw = (b @ a) * scale
            half = _bf16_half_ulp(w)
            survive.append((dw.abs() >= half).float().mean().item())
            ratios.append((dw.abs() / half).median().item())
            n_el += dw.numel()
    if not survive:
        return None
    return {"bf16_survive_frac": sum(survive) / len(survive),
            "bf16_ratio_median": sorted(ratios)[len(ratios) // 2],
            "bf16_modules": len(survive), "bf16_elements": n_el}

# ---------------------------------------------------------------- H4: smoke assertions

def merged_rollout_mode() -> str | None:
    """unsloth-zoo's merged-weight rollout mode, or None when rollouts use vLLM's own LoRA (what we want).

    zoo main (after 2026-10-19) folds s*B@A into the bf16 base weights by default; at lr ~4e-6 the delta is
    below half a bf16 ULP for most weights, so rollouts silently come from the base model. Scripts set
    UNSLOTH_VLLM_MERGED_ROLLOUT=0 before the unsloth import; this reads back what zoo actually decided."""
    vu = sys.modules.get("unsloth_zoo.vllm_utils")
    fn = getattr(vu, "_merged_rollout_mode", None) if vu else None
    if fn is None:
        return None if vu else "unknown (unsloth_zoo.vllm_utils not loaded)"
    return fn()


class SmokeCheck:
    """Collects the facts a GRPO_SMOKE=1 run must establish; verdict() exits non-zero on any failure."""

    def __init__(self):
        self.nonfinite_max = 0
        self.b_max_last = 0.0
        self.tripwire_calls = 0
        self.lora_calls = None          # None = rollout wrapper not armed

    def tripwire(self, nonfinite: int, b_max: float) -> None:
        self.tripwire_calls += 1
        self.nonfinite_max = max(self.nonfinite_max, nonfinite)
        self.b_max_last = b_max

    def failures(self) -> list[str]:
        out = []
        if self.tripwire_calls == 0:
            out.append("no optimizer step was observed (tripwire never ran)")
        if self.nonfinite_max:
            out.append(f"non-finite LoRA-B gradients seen (max {self.nonfinite_max} tensors in one step)")
        if self.b_max_last <= 0.0:
            out.append("|B|max is 0 after training: the adapter never moved")
        if self.lora_calls is not None and self.lora_calls == 0:
            out.append("the rollout engine never received the live adapter (0 wrapped generate calls)")
        if (mode := merged_rollout_mode()) is not None:
            out.append(f"merged-weight rollouts are on ({mode}): set UNSLOTH_VLLM_MERGED_ROLLOUT=0")
        return out

    def verdict(self, run_dir: str | None = None) -> None:
        fails = self.failures()
        facts = {"tripwire_calls": self.tripwire_calls, "nonfinite_max": self.nonfinite_max,
                 "b_max_last": self.b_max_last, "lora_calls": self.lora_calls,
                 "merged_rollout": merged_rollout_mode()}
        if run_dir:
            update_manifest(run_dir, smoke={"passed": not fails, "failures": fails, **facts})
        if fails:
            print("[SMOKE] FAILED: " + " | ".join(fails), flush=True)
            sys.stdout.flush()
            os._exit(3)
        print(f"[SMOKE] PASSED {facts}", flush=True)
