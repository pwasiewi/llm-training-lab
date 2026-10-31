#!/usr/bin/env python3
"""bench_08 helper: chain trials against a raw llama-server, nanoeuler vote rows,
and the report (p, p(fill), L_eff, vote-recoverability).

Subcommands
  chain   run dependent-step chain trials over an (N x fill) grid -> JSONL
  voterow turn one verified nanoeuler attempt into per-stage JSONL rows
  report  read chain/vote JSONL files and print the bench_08 tables

Only the standard library is used. Every trial row is a
(proposal, verified_outcome) record usable as distillation/GRPO data.
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

# ------------------------------------------------------------------ http ----

class ServerDead(Exception):
    pass


def http_json(url, payload=None, timeout=60):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300]
        raise RuntimeError(f"HTTP {e.code} from {url}: {body}") from e
    except (urllib.error.URLError, ConnectionError) as e:
        raise ServerDead(f"{url}: {e}") from e


def tokenize(base, text):
    return http_json(base + "/tokenize", {"content": text}, timeout=300)["tokens"]


def detokenize(base, toks):
    return http_json(base + "/detokenize", {"tokens": toks}, timeout=300)["content"]


def server_ctx(base):
    props = http_json(base + "/props", timeout=30)
    dgs = props.get("default_generation_settings") or {}
    n = dgs.get("n_ctx") or props.get("n_ctx")
    if not n:
        raise RuntimeError("cannot read n_ctx from /props")
    return int(n)

# ----------------------------------------------------------------- chain ----

MOD = 1000


def apply_op(op, x):
    kind = op[0]
    if kind == "add":
        return (x + op[1]) % MOD
    if kind == "sub":
        return (x - op[1]) % MOD
    if kind == "mul":
        return (x * op[1]) % MOD
    if kind == "rev":
        return int(f"{x:03d}"[::-1])
    if kind == "swp":
        s = f"{x:03d}"
        return int(s[0] + s[2] + s[1])
    if kind == "par":
        return x // 2 if x % 2 == 0 else (x + op[1]) % MOD
    if kind == "cmp":
        return (x - op[2]) % MOD if x > op[1] else (x + op[3]) % MOD
    if kind == "dig":
        return (sum(int(c) for c in str(x)) * op[1]) % MOD
    raise ValueError(kind)


def op_text(op):
    kind = op[0]
    return {
        "add": lambda: f"Add {op[1]}.",
        "sub": lambda: f"Subtract {op[1]}.",
        "mul": lambda: f"Multiply by {op[1]}.",
        "rev": lambda: "Reverse the three digits of the value written with leading zeros (e.g. 7 -> 007 -> 700, 120 -> 021 -> 21).",
        "swp": lambda: "Swap the tens digit and the units digit of the value written with leading zeros (e.g. 45 -> 045 -> 054 -> 54).",
        "par": lambda: f"If the value is even, halve it; otherwise add {op[1]}.",
        "cmp": lambda: f"If the value is greater than {op[1]}, subtract {op[2]}; otherwise add {op[3]}.",
        "dig": lambda: f"Replace the value with the sum of its decimal digits multiplied by {op[1]}.",
    }[kind]()


def gen_chain(n, seed):
    rng = random.Random(seed)
    x0 = rng.randrange(100, 1000)
    ops = []
    for _ in range(n):
        kind = rng.choice(["add", "sub", "mul", "rev", "swp", "par", "cmp", "dig"])
        if kind in ("add", "sub"):
            ops.append((kind, rng.randrange(11, 500)))
        elif kind == "mul":
            ops.append((kind, rng.randrange(2, 10)))
        elif kind == "par":
            ops.append((kind, rng.randrange(11, 200)))
        elif kind == "cmp":
            ops.append((kind, rng.randrange(200, 800), rng.randrange(11, 300), rng.randrange(11, 300)))
        elif kind == "dig":
            ops.append((kind, rng.randrange(2, 31)))
        else:
            ops.append((kind,))
    truth, x = [], x0
    for op in ops:
        x = apply_op(op, x)
        truth.append(x)
    return x0, ops, truth


def task_text(x0, ops):
    lines = [
        "TASK. Start with the value x0 = %d. Apply the steps below in order; each step works on the result of the previous one." % x0,
        "All values are integers from 0 to 999: after every step reduce the result modulo 1000 (a negative result wraps around, e.g. -5 -> 995).",
        "",
    ]
    lines += [f"Step {i}: {op_text(op)}" for i, op in enumerate(ops, 1)]
    lines += [
        "",
        "Answer format (required): list the value after each step, one per line, exactly as `S1 = <value>`, `S2 = <value>`, ..., "
        "then a last line `FINAL = <value>`.",
    ]
    return "\n".join(lines)


BALLAST_HEAD = "The following material is unrelated background text. Ignore it.\n-----BEGIN MATERIAL-----\n"
BALLAST_TAIL = "\n-----END MATERIAL-----\n\n"
START_TRAILER = "\n\nNow carry out the TASK stated at the top of this message, in the required answer format."


def distractor_corpus(seed, approx_chars):
    """Synthetic worked chains with other numbers: a distractor, not neutral."""
    rng = random.Random(seed)
    out, size = [], 0
    while size < approx_chars:
        n = rng.randrange(3, 12)
        x0, ops, truth = gen_chain(n, rng.randrange(1 << 30))
        block = [f"Worked example. Start with x0 = {x0}."]
        block += [f"Step {i}: {op_text(op)} -> S{i} = {v}" for i, (op, v) in enumerate(zip(ops, truth), 1)]
        block.append(f"FINAL = {truth[-1]}\n")
        s = "\n".join(block)
        out.append(s)
        size += len(s)
    return "\n".join(out)


def parse_answer(content, n):
    vals = {}
    for m in re.finditer(r"(?m)^[\s*`>-]*S(\d+)\s*[=:]\s*\**\s*(-?\d+)", content or ""):
        i = int(m.group(1))
        if 1 <= i <= n:
            vals[i] = int(m.group(2))            # last occurrence wins
    fm = re.findall(r"(?m)FINAL\s*[=:]\s*\**\s*(-?\d+)", content or "")
    final = int(fm[-1]) if fm else None
    return [vals.get(i) for i in range(1, n + 1)], final


def score_chain(x0, ops, truth, prop, final):
    """Per-step conditional correctness: step i is judged against the model's
    OWN previous value, so every step is an independent trial even after an
    earlier slip. None = unscorable (missing value / missing predecessor)."""
    step_ok, prev = [], x0
    for op, v in zip(ops, prop):
        if prev is None or v is None:
            step_ok.append(None)
        else:
            step_ok.append(apply_op(op, prev) == v)
        prev = v
    chain_ok = all(p == t for p, t in zip(prop, truth)) and final == truth[-1]
    return step_ok, chain_ok


def cmd_chain(a):
    base = a.base.rstrip("/")
    try:
        n_ctx = server_ctx(base)
    except ServerDead as e:
        print(f"SERVER DEAD before start: {e}", file=sys.stderr)
        return 3
    corpus_cache = {}

    def ballast_tokens(kind):
        if kind not in corpus_cache:
            if kind == "readme":
                with open(a.corpus, encoding="utf-8", errors="replace") as f:
                    text = f.read()
            elif kind == "distractor":
                text = distractor_corpus(a.seed_base + 7, 4 * max(a.fills) + 4096)
            else:
                raise SystemExit(f"unknown ballast type {kind}")
            corpus_cache[kind] = tokenize(base, text)
        return corpus_cache[kind]

    ns = a.ns
    fills = a.fills
    lost = 0
    with open(a.out, "a", encoding="utf-8") as out:
        for kind in a.ballast:
            for fill in fills:
                if fill + a.max_tokens + 256 > n_ctx:
                    print(f"--- skip fill {fill}: + max_tokens {a.max_tokens} exceeds n_ctx {n_ctx}")
                    continue
                for n in ns:
                    for rep in range(1, a.reps + 1):
                        cseed = a.seed_base * 1000003 + n * 1009 + rep    # same chain at every fill (paired design)
                        x0, ops, truth = gen_chain(n, cseed)
                        task = task_text(x0, ops)
                        btxt, btok = "", 0
                        if fill > 0:
                            fixed = len(tokenize(base, BALLAST_HEAD + BALLAST_TAIL + task
                                                 + (START_TRAILER if a.position == "start" else "")))
                            need = max(0, fill - fixed - 24)            # ~chat-template overhead
                            toks = ballast_tokens(kind)
                            if need > len(toks):
                                reps_needed = need // max(1, len(toks)) + 1
                                toks = (toks * reps_needed)
                            btxt = detokenize(base, toks[:need])
                            btok = need
                        if fill == 0:
                            prompt = task
                        elif a.position == "end":
                            prompt = BALLAST_HEAD + btxt + BALLAST_TAIL + task
                        else:
                            prompt = task + "\n\n" + BALLAST_HEAD + btxt + BALLAST_TAIL + START_TRAILER
                        payload = {
                            "model": a.model,
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": a.max_tokens,
                            "seed": cseed + fill,
                            "cache_prompt": True,
                        }
                        if a.temperature is not None:
                            payload["temperature"] = a.temperature
                        t0 = time.time()
                        row = {
                            "bench": "bench08", "part": "chain", "model": a.model, "tag": a.tag,
                            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "N": n, "fill": fill,
                            "ballast": kind, "ballast_tokens": btok, "position": a.position,
                            "rep": rep, "seed": cseed, "n_ctx": n_ctx, "x0": x0,
                            "ops": [list(o) for o in ops], "task": task, "truth": truth,
                        }
                        try:
                            r = http_json(base + "/v1/chat/completions", payload, timeout=a.req_timeout)
                            msg = r["choices"][0]["message"]
                            content = msg.get("content") or ""
                            prop, final = parse_answer(content, n)
                            step_ok, chain_ok = score_chain(x0, ops, truth, prop, final)
                            row.update({
                                "proposal": prop, "final": final, "step_ok": step_ok,
                                "verified_outcome": "PASS" if chain_ok else "FAIL",
                                "finish_reason": r["choices"][0].get("finish_reason"),
                                "prompt_tokens": r.get("usage", {}).get("prompt_tokens"),
                                "completion_tokens": r.get("usage", {}).get("completion_tokens"),
                                "response": content,
                                "reasoning": msg.get("reasoning_content") or "",
                            })
                        except ServerDead as e:
                            row.update({"verified_outcome": "LOST", "error": f"server died: {e}"})
                            lost += 1
                        except Exception as e:           # timeout, bad JSON: a scored FAIL, not lost
                            row.update({"verified_outcome": "FAIL", "proposal": [None] * n, "final": None,
                                        "step_ok": [None] * n, "error": str(e)[:300]})
                        row["dur_s"] = round(time.time() - t0, 1)
                        out.write(json.dumps(row) + "\n")
                        out.flush()
                        ok = sum(1 for s in row.get("step_ok") or [] if s)
                        print(f"=== [{a.model}/chain {kind} fill={fill} N={n} rep {rep}/{a.reps}] "
                              f"{row['verified_outcome']} ({row['dur_s']}s) steps {ok}/{n} "
                              f"ptok={row.get('prompt_tokens')} ctok={row.get('completion_tokens')}"
                              f"{' finish=' + str(row.get('finish_reason')) if row.get('finish_reason') not in (None, 'stop') else ''}"
                              f"{' err=' + row['error'][:80] if row.get('error') else ''}", flush=True)
                        if row["verified_outcome"] == "LOST":
                            print("!!! llama-server died — remaining chain trials of this model are SKIPPED", file=sys.stderr)
                            return 3
    return 0

# ------------------------------------------------------------------ vote ----

def ebuild_funcs(text):
    """Return {func_name: body} for top-level shell functions of an ebuild."""
    funcs, cur, depth, buf = {}, None, 0, []
    for line in text.splitlines():
        if cur is None:
            m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(\)\s*\{", line)
            if m:
                cur, depth, buf = m.group(1), line.count("{") - line.count("}"), []
                if depth <= 0:
                    funcs[cur] = ""
                    cur = None
            continue
        depth += line.count("{") - line.count("}")
        if depth <= 0:
            funcs[cur] = "\n".join(buf)
            cur = None
        else:
            buf.append(line)
    return funcs


def strip_comments(s):
    return "\n".join(re.sub(r"\s+#.*$", "", l) for l in s.splitlines() if not l.strip().startswith("#"))


def features(text):
    """Coarse per-stage APPROACH labels — the unit majority voting acts on.
    Exact text never repeats across samples; the approach does."""
    lines = text.splitlines()
    f = ebuild_funcs(text)
    body = strip_comments(text)
    hdr = [re.sub(r"\b(19|20)\d\d(-(19|20)\d\d)?\b", "YYYY", l.strip()) for l in lines[:2]]
    inh = sorted(set(" ".join(re.findall(r"(?m)^inherit\s+(.*)$", body)).split()))
    kw = re.findall(r"(?m)^KEYWORDS=.*$", body)
    pin = []
    for var in ("EGIT_COMMIT", "EGIT_BRANCH", "EGIT_REPO_URI", "SRC_URI"):
        m = re.search(rf"(?m)^{var}=\"?([^\"\n]*)", body)
        if m:
            v = m.group(1)
            if var == "EGIT_COMMIT":
                v = v[:7]
            elif var in ("EGIT_REPO_URI", "SRC_URI"):
                v = "set"
            pin.append(f"{var}={v}")
    prep = strip_comments(f.get("src_prepare", "")) + "\n" + strip_comments(f.get("src_configure", ""))
    comp = strip_comments(f.get("src_compile", ""))
    both = prep + "\n" + comp
    removes = bool(re.search(r"\brm\b[^\n]*nanoeuler|\b(emake|make)\b[^\n]*\bclean\b|\bemake\s+-B\b|\bmake\s+-B\b|--always-make", both))
    if not comp.strip():
        clabel = "default-src_compile"
    else:
        clabel = "emake" if re.search(r"\bemake\b", comp) else ("direct-cc" if re.search(r"\$\(tc-getCC\)|\$\{?CC\}?|\bgcc\b|\bcc\b", comp) else "other")
    clabel += "+prebuilt-removed" if removes else "+prebuilt-kept"
    test = strip_comments(f.get("src_test", ""))
    if not test.strip():
        tlabel = "no-src_test" if "src_test" not in f else "empty-src_test"
    elif "gradcheck" in test or "nanoeuler_check" in test:
        tlabel = "runs-nanoeuler_check"
    elif re.search(r"\bemake\b[^\n]*\b(check|test)\b", test):
        tlabel = "emake-check"
    else:
        tlabel = "other"
    inst = strip_comments(f.get("src_install", ""))
    ilabel = " ".join(sorted(set(re.findall(r"\b(dobin|newbin|dosbin|doexe|dodoc|einstalldocs|emake install|default)\b", inst)))) or "default-src_install"
    lic = re.findall(r'(?m)^LICENSE="?([^"\n]*)', body)
    return {
        "header": " | ".join(hdr),
        "inherit": " ".join(inh) or "(none)",
        "keywords": kw[0] if kw else "(no KEYWORDS)",
        "pin": " ".join(pin) or "(none)",
        "compile": clabel,
        "test": tlabel,
        "install": ilabel,
        "license": lic[0] if lic else "(none)",   # no gate: reported, never scored
    }


# stage -> oracle gates whose AND is the stage's verified outcome
STAGE_GATES = {
    "header": ["header1", "header2"],
    "inherit": ["inherit"],
    "keywords": ["keywords"],
    "pin": ["pin"],
    "compile": ["build", "noop", "realcompile"],
    "test": ["testran"],
    "install": ["installed"],
}


def read_gates(path):
    g = {}
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("|", 2)
                if len(parts) >= 2:
                    g[parts[0]] = (parts[1], parts[2] if len(parts) > 2 else "")
    return g


def cmd_voterow(a):
    text = open(a.ebuild, encoding="utf-8", errors="replace").read() if a.ebuild and os.path.exists(a.ebuild) else ""
    feats = features(text) if text else {k: "(no ebuild)" for k in list(STAGE_GATES) + ["license"]}
    gates = read_gates(a.gates)
    sha = hashlib.sha256(text.encode()).hexdigest() if text else None
    allpass = bool(gates) and all(v[0] in ("PASS", "SKIP") for v in gates.values())
    common = {"bench": "bench08", "part": "vote", "model": a.model, "tag": a.tag,
              "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "attempt": a.attempt, "iter": a.iter,
              "status": a.status, "ebuild_path": a.ebuild, "ebuild_sha256": sha}
    with open(a.out, "a", encoding="utf-8") as out:
        for stage, gl in STAGE_GATES.items():
            if a.status != "scored":
                outcome = "LOST" if a.status == "lost" else "FAIL"
            elif not text:
                outcome = "FAIL"
            else:
                res = [gates.get(g, ("MISSING", ""))[0] for g in gl]
                outcome = "PASS" if all(r in ("PASS", "SKIP") for r in res) else "FAIL"
                # install is only observable when the build got that far: an
                # earlier build death is the compile/test stage's failure
                if stage == "install" and gates.get("build", ("",))[0] == "FAIL":
                    outcome = "NA"
            notes = "; ".join(gates[g][1] for g in gl if g in gates and gates[g][0] == "FAIL" and gates[g][1])
            out.write(json.dumps({**common, "stage": stage, "proposal": feats[stage],
                                  "verified_outcome": outcome, "gates": {g: gates.get(g, ("MISSING", ""))[0] for g in gl},
                                  "note": notes[:300]}) + "\n")
        out.write(json.dumps({**common, "stage": "ALL", "proposal": sha, "features": feats,
                              "verified_outcome": ("PASS" if allpass and a.status == "scored" else
                                                   ("LOST" if a.status == "lost" else "FAIL")),
                              "gates": {k: v[0] for k, v in gates.items()},
                              "ebuild": text}) + "\n")
    return 0

# ---------------------------------------------------------------- report ----

def load(paths):
    rows = []
    for p in paths:
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                rows += [json.loads(l) for l in fh if l.strip()]
    return rows


def mle_p(pairs):
    """MLE of per-step p from chain outcomes (N, success): P(success|N)=p^N."""
    if not pairs:
        return None
    if all(s for _, s in pairs):
        return 1.0          # no failure observed: p is censored at 1, N_max unbounded
    best, bp = -1e300, None
    for i in range(1, 1000):
        p = i / 1000.0
        ll = 0.0
        for n, s in pairs:
            q = p ** n
            ll += math.log(q) if s else math.log(max(1e-300, 1 - q))
        if ll > best:
            best, bp = ll, p
    return bp


def se(p, n):
    return math.sqrt(p * (1 - p) / n) if n else float("nan")


def cmd_report(a):
    rows = load(a.chain)
    out = []
    pr = out.append
    if rows:
        pr("==================== bench_08 CHAIN: horizon p and fill decay ====================")
        groups = {}
        for r in rows:
            key = (r["model"], r["ballast"], r["position"])
            groups.setdefault(key, []).append(r)
        for (model, kind, pos), rs in groups.items():
            scored = [r for r in rs if r["verified_outcome"] != "LOST"]
            lost = len(rs) - len(scored)
            pr(f"\nMODEL {model}   ballast={kind}   position={pos}   trials={len(scored)}  lost={lost}")
            pr(f"{'FILL':>7} {'N':>3} {'n':>3} {'CHAIN_OK':>8} {'STEP_OK':>9} {'p_step':>7}")
            fills = sorted({r["fill"] for r in scored})
            ns = sorted({r["N"] for r in scored})
            per_fill = {}
            for fill in fills:
                fr = [r for r in scored if r["fill"] == fill]
                for n in ns:
                    cr = [r for r in fr if r["N"] == n]
                    if not cr:
                        continue
                    ok = sum(r["verified_outcome"] == "PASS" for r in cr)
                    st = [s for r in cr for s in r["step_ok"] if s is not None]
                    sk = sum(st)
                    pr(f"{fill:>7} {n:>3} {len(cr):>3} {ok:>4}/{len(cr):<3} {sk:>4}/{len(st):<4} "
                       f"{(sk / len(st)) if st else float('nan'):>7.3f}")
                st_all = [s for r in fr for s in r["step_ok"] if s is not None]
                # an unscorable step (no value printed) is a failed step for p
                miss = sum(1 for r in fr for s in r["step_ok"] if s is None)
                n_tr = len(st_all) + miss
                p_dir = sum(st_all) / n_tr if n_tr else None
                p_fit = mle_p([(r["N"], r["verified_outcome"] == "PASS") for r in fr])
                per_fill[fill] = (p_dir, n_tr, p_fit, len(fr))
            pr(f"\n  {'FILL':>7} {'p_step(direct)':>15} {'±SE':>6} {'steps':>6} {'p_chain(MLE)':>13} {'chains':>6} {'N_max(S=0.5)':>12}")
            for fill, (pd, nt, pf, nc) in per_fill.items():
                nmax = (math.log(0.5) / math.log(pf)) if pf and 0 < pf < 1 else (float("inf") if pf == 1 else float("nan"))
                pr(f"  {fill:>7} {pd if pd is not None else float('nan'):>15.3f} {se(pd or 0, nt):>6.3f} {nt:>6} "
                   f"{pf if pf is not None else float('nan'):>13.3f} {nc:>6} {nmax:>12.1f}")
            # p0 / L_eff / post-knee slope (refinement 2026-07-18c)
            fl = [f for f in per_fill if per_fill[f][0] is not None]
            if fl:
                p0 = per_fill[fl[0]][0]
                knee = next((f for f in fl[1:] if per_fill[f][0] < p0 - a.delta), None)
                if knee is None:
                    pr(f"  p0={p0:.3f} (fill {fl[0]})   L_eff > {fl[-1]} (no drop of {a.delta} below p0 in range — flat = valid null)")
                else:
                    i = fl.index(knee)
                    prev = fl[i - 1]
                    last = fl[-1]
                    slope = ((per_fill[last][0] - per_fill[prev][0]) / ((last - prev) / 10000.0)) if last > prev else float("nan")
                    pr(f"  p0={p0:.3f} (fill {fl[0]})   L_eff = {prev}..{knee} tokens (first fill with p < p0-{a.delta})   "
                       f"post-knee slope {slope:+.3f} p per 10k tokens")
        pr("\nRead: p_step(direct) = per-step correctness judged against the model's own previous value;")
        pr("p_chain(MLE) = p fitted to chain success ~ p^N at fixed fill. p_chain < p_step = steps are not")
        pr("independent (an early slip derails more than one step). Resolve 0.90->0.80 needs ~100 steps per fill.")
    vrows = [r for r in load(a.vote) if r.get("stage") != "ALL"]
    if vrows:
        pr("\n==================== bench_08 VOTE: nanoeuler ebuild, majority-of-k ====================")
        pr(f"{'MODEL':<22} {'STAGE':<9} {'k':>3} {'p_stage':>7} {'MODAL PROPOSAL (count)':<46} {'MODAL_OK':>8} CLASS")
        by = {}
        for r in vrows:
            if r["iter"] != 0 or r["verified_outcome"] in ("LOST", "NA"):
                continue      # vote = first proposals only; arbiter iterations are arm C
            by.setdefault((r["model"], r["stage"]), []).append(r)
        for (model, stage), rs in by.items():
            k = len(rs)
            ps = sum(r["verified_outcome"] == "PASS" for r in rs) / k
            counts = {}
            for r in rs:
                counts.setdefault(r["proposal"], []).append(r["verified_outcome"] == "PASS")
            ranked = sorted(counts.items(), key=lambda kv: -len(kv[1]))
            top = ranked[0]
            tie = len(ranked) > 1 and len(ranked[1][1]) == len(top[1])
            modal_ok = sum(top[1]) / len(top[1]) >= 0.5
            if ps == 1.0:
                cls = "solid (p=1)"
            elif tie:
                cls = "tie — no majority"
            elif modal_ok:
                cls = "A recoverable (vote helps)"
            else:
                cls = "B modal-wrong (vote amplifies)"
            label = (top[0] or "")[:40]
            pr(f"{model:<22} {stage:<9} {k:>3} {ps:>7.2f} {label + ' (' + str(len(top[1])) + ')':<46} {'yes' if modal_ok else 'NO':>8} {cls}")
        arb = [r for r in load(a.vote) if r.get("stage") == "ALL"]
        if any(r["iter"] > 0 for r in arb):
            pr("\nARBITER (arm C): iterations-to-pass per attempt (iter 0 = first proposal)")
            att = {}
            for r in arb:
                att.setdefault((r["model"], r["attempt"]), []).append(r)
            for (model, at), rs in sorted(att.items()):
                rs.sort(key=lambda r: r["iter"])
                first = next((r["iter"] for r in rs if r["verified_outcome"] == "PASS"), None)
                pr(f"  {model:<22} attempt {at}: {'PASS at iter ' + str(first) if first is not None else 'no pass in ' + str(rs[-1]['iter']) + ' iters'}")
        pr("\nRead: CLASS A = errors are sampling noise, majority-of-k lands on a verified-correct approach;")
        pr("CLASS B = the most common approach FAILS the oracle — voting would lock the wrong answer in.")
        pr("Proposal = coarse approach label per stage (b08.py features); MODAL_OK = attempts carrying the")
        pr("modal label pass that stage's gates in >= half the cases.")
    txt = "\n".join(out)
    print(txt)
    return 0

# ------------------------------------------------------------------ main ----

def ints(s):
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("chain")
    c.add_argument("--base", required=True)
    c.add_argument("--model", required=True)
    c.add_argument("--tag", default="")
    c.add_argument("--ns", type=ints, required=True)
    c.add_argument("--fills", type=ints, required=True)
    c.add_argument("--reps", type=int, required=True)
    c.add_argument("--ballast", type=lambda s: s.split(","), default=["readme"])
    c.add_argument("--position", choices=["end", "start"], default="end")
    c.add_argument("--corpus", required=True)
    c.add_argument("--max-tokens", type=int, default=8192)
    c.add_argument("--req-timeout", type=int, default=900)
    c.add_argument("--temperature", type=float, default=None)
    c.add_argument("--seed-base", type=int, default=8)
    c.add_argument("--out", required=True)
    v = sub.add_parser("voterow")
    for k in ("model", "tag", "ebuild", "gates", "out"):
        v.add_argument("--" + k, default="")
    v.add_argument("--attempt", type=int, required=True)
    v.add_argument("--iter", type=int, default=0)
    v.add_argument("--status", choices=["scored", "lost", "noebuild"], default="scored")
    f = sub.add_parser("features")
    f.add_argument("ebuild")
    r = sub.add_parser("report")
    r.add_argument("--chain", nargs="*", default=[])
    r.add_argument("--vote", nargs="*", default=[])
    r.add_argument("--delta", type=float, default=0.05)
    a = ap.parse_args()
    if a.cmd == "chain":
        return cmd_chain(a)
    if a.cmd == "voterow":
        return cmd_voterow(a)
    if a.cmd == "features":
        print(json.dumps(features(open(a.ebuild, encoding="utf-8").read()), indent=1))
        return 0
    return cmd_report(a)


if __name__ == "__main__":
    sys.exit(main())
