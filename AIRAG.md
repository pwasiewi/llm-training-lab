# AIRAG — local hybrid RAG index (aillama + airag)

Local retrieval that substitutes for a missing domain LoRA on the small local
models. Document/script chunks are embedded on the CPU and stored in one SQLite
file (vectors as float32 BLOBs + FTS5 for BM25), queried with hybrid
vector+BM25 retrieval fused by Reciprocal Rank Fusion, with an optional CPU
cross-encoder rerank stage. Every query is logged — the corpus + query log seed
future SFT/GRPO data.

Scripts: `~/Claude/bin/aillama` (server manager), `~/Claude/bin/airag`
(indexer/query, Python 3), `~/Claude/bin/qwen-yolo` (headless named-entity
launcher for qwen-code). **All three are mirrored to `/usr/local/bin`, which
shadows `~/Claude/bin` in PATH — after editing, `sudo cp -f` all three, or the
old version keeps running.**

## Components and ports

| Process | Where | Port | GPU/CPU | Role |
|---|---|---|---|---|
| **llama-server (chat)** | host | **8092** `/v1` | GPU (16 GB) | chat model, one at a time |
| **llama-server (embed)** | host | **8093** `/v1/embeddings` | **CPU** (`-ngl 0`; `--gpu` → `-ngl 99`) | backend for `airag` (Qwen3-Embedding-0.6B Q8_0) |
| **llama-server (rerank)** | host | **8094** `/v1/rerank` | **CPU** | optional `airag query -r` (Qwen3-Reranker-0.6B Q8_0) |
| **Open WebUI** | docker | **3001** | — | GUI, talks to `:8092/v1` |
| **SearXNG** | docker | **8081** | — | web search for WebUI (shared with `airun`) |

**Bind:** `LLAMA_HOST=auto` → the **docker0** gateway IP (reachable from
containers *and* the host, invisible on the LAN). If docker is down → falls back
to `127.0.0.1` (CLI still works, containers cannot reach it).

## Where things are stored

| Path | Content |
|---|---|
| `~/models/` | GGUF files (`MODEL_ROOT`) |
| `~/.aillama/models.conf` | profiles `name \| gguf \| args` (`aillama init` writes defaults) |
| `~/.aillama/current-profile` | last-used profile (the default for `start` with no arg) |
| `~/.aillama/{llama-server,embed-server,rerank-server}.{pid,log}` | PIDs + logs of the three servers |
| **`~/.airag/index.db`** | the whole RAG DB: vectors (float32 BLOB) + FTS5/BM25 + `queries` + `feedback` |
| docker volume `open-webui-llama` | Open WebUI data (`/app/backend/data`) |
| docker volume `searxng` | SearXNG config |

**Embeddings are NOT separate files** — they live as BLOBs inside
`~/.airag/index.db`. The embedder model name is recorded on first `index`;
changing the embedder requires a full reindex (delete the DB).

## Index schema (v2 content/locations split + v3 scope/sessions)

Content is separated from location so identical bytes under several paths embed
**once**; `files.scope` additionally partitions content into knowledge scopes,
and `sessions` gives each launch a stable per-entity number:

```
files(id, sha256 UNIQUE, mtime, scope)        -- content, embedded once; scope: NULL/'' = common
locations(path PRIMARY KEY, file_id → files)  -- N paths → 1 content (dedup)
chunks.file_id → files.id                     -- chunks hang off content, not path
chunks_fts (FTS5)                             -- BM25 lexical side
queries(..., source, session_id)              -- query log + curation
sessions(session_id PK, seq, source, first_ts) -- session UUID -> per-source seq
feedback                                      -- curation of logged queries
```

`index_one` returns: `-1` unchanged · `-2` linked to already-indexed content
(dedup, nothing embedded) · `>=0` chunks embedded for new content. `gc_content`
drops a content row (chunks + FTS + `files`) only once **no** location points to
it — so deleting one of two identical copies changes nothing but the path.
`roots` (in `meta`) is a `{root_path: scope}` map (a bare list from a
pre-2026-07-18 index reads back as all-`None`/common — no rebuild needed).

Reranking is **not** stored in the DB — it is a runtime stage (port 8094) that
reorders the top-20 RRF candidates on the fly.

### Knowledge scope (`--scope`) — partitions the corpus, not the log

`files.scope` tags **content**: `NULL`/`''` = common knowledge, visible to
every scope; any other string = a private partition.

- `airag index --scope NAME PATH...` — new content under these roots is tagged
  `NAME`. Existing content already indexed keeps whatever scope it has; a
  `reindex` re-walks each recorded root with **its own remembered scope**, not
  a scope you pass on the command line (there is no `--scope` on `reindex`).
- `airag query --scope NAME TEXT` — retrieves `NAME`'s own content **plus**
  all common content. A different scope's private content is invisible, full
  stop (not just deprioritized).
- `airag query TEXT` (no `--scope`) — searches **everything**, unfiltered.
  This is the default and the correct behavior for ad-hoc CLI lookups.
- Until you actually index something with `--scope`, every content row is
  common — so scoping is fully opt-in and backward compatible.

Verified end-to-end (2026-07-18): an `agent2` query for a fact stored under
`--scope agent1` returns nothing from that private doc, but still returns
common-scope hits; the same query without `--scope` returns everything.

### Per-entity session numbering (`AIRAG_SESSION`) — scheme A

`AIRAG_SESSION` (a session UUID) plus `AIRAG_SOURCE` give each **launch** of a
named entity a stable, human-readable number — scheme A: per-entity, not one
global counter. First query of a new `session_id` under a given `source`
allocates the next `seq` for *that source*; every later query with the same
`session_id` reuses it. `log`/`export-queries`/the query's own stderr note
render it as `source#seq` (e.g. `agent1#3`); a source with no `AIRAG_SESSION`
set just shows the bare source name (no `#`).

Verified: two queries in one session → same `agent1#1`; a second session of
the same entity → `agent1#2`; a different entity's session starts its own
counter at `#1`. The counter lives in `sessions`, keyed by the raw UUID, so
the number is stable across the whole session's lifetime regardless of query
count.

## Startup order

**Full stack (chat + WebUI):**
```
aillama up [profile]     # chat(8092) + embed(8093 if GGUF present) + webui(3001) + searxng(8081)
aillama status           # who is alive + VRAM
aillama down             # teardown in reverse
```
Default profile: `qwen36-128k`. Change model: `aillama switch <profile>` (one
model fits 16 GB VRAM at a time).

**RAG only (no WebUI) — minimal path:**
```
aillama embed-start                 # 8093 CPU — REQUIRED by airag
airag index ~/Claude ~/QwenCode     # incremental (sha256) → ~/.airag/index.db
airag query -k 6 "your question"    # hybrid vector+BM25 (RRF)
# optional rerank:
aillama rerank-start                # 8094 CPU
airag query -r "your question"      # reorder top-20 with the cross-encoder
```

**Bulk reindex on GPU (large corpus):** the embedder is CPU-only by default so
the chat model can own all 16 GB. For a big one-off index, free the card and
offload the embedder:
```
aillama stop                # release VRAM from the chat model
aillama embed-start --gpu   # -ngl 99; the 0.6B Q8_0 needs ~0.6 GB VRAM
airag index ~/Claude ...    # far faster than CPU
aillama embed-stop
aillama switch <profile>    # bring the chat model back
```
`EMBED_NGL=99 aillama embed-start` is the env-var equivalent. GPU vs CPU only
changes vectors by float noise, and the embedder-version guard keys on the model
name (not backend), so switching CPU↔GPU does **not** force a reindex. You do
not strictly need `aillama stop` (0.6 GB fits alongside the chat model) — do it
for throughput, to give the whole card to the batch.

**For CLI agents (qwen-code / aider / codex):**
```
eval "$(aillama env)"    # OPENAI_BASE_URL=:8092/v1, AIRAG_EMBED_URL=:8093, model=profile alias
```

## Dependencies

```
airag index/query ──needs──> aillama embed-start (8093)
airag query -r     ──needs──> aillama rerank-start (8094)
Open WebUI (3001)  ──talks──> chat llama-server (8092) ──web search──> SearXNG (8081)
```

## Command reference (airag)

| Command | What it does |
|---|---|
| `index PATH... [--scope NAME]` | incremental index; identical content under many paths embeds once; roots accumulate; `--scope` tags new content (default: common) |
| `query [-k N] [-s] [-r] [--scope NAME] TEXT` | hybrid vector+BM25 search; `-s` sources only, `-r` rerank stage, `--scope` restricts to `NAME ∪ common` (default: unfiltered) |
| `reindex` | re-walk recorded roots with each root's own remembered scope: update changed, **drop deleted** (index alone never deletes) |
| `stats` | DB size, embed model, roots (with scope), `N paths (M unique contents)`, chunks per scope, queries, sessions numbered, per-source query counts |
| `log [-n N]` | recent logged queries: id, `source#seq` (or bare source), feedback, text |
| `feedback ID good\|bad [NOTE]` | curate a logged query (LoRA-data seed) |
| `export-queries` | dump query log + retrieved chunks as JSONL, incl. `session_id`/`session_seq` |

Query logging carries a `source` column (`AIRAG_SOURCE` env; qwen-code tags
`qwen` via `~/.qwen/.env`) and an optional `session_id` (`AIRAG_SESSION` env,
a UUID) — `stats`/`log` measure whether an agent actually consulted the index
(skill-compliance) and let individual launches of one entity be told apart,
without trusting the agent's own report.

## Named entities: qwen-yolo launcher + the airag-inject hook

`~/Claude/bin/qwen-yolo NAME [prompt-source]` launches qwen-code headless as
its own named entity:
```
qwen-yolo agent1 -f task.md            # prompt from a file
qwen-yolo agent1 -e                    # compose in $EDITOR
qwen-yolo agent1 "step one" "step two" # each trailing arg = one line
git log --oneline | qwen-yolo agent1   # piped/heredoc stdin
```
It sets `AIRAG_SOURCE=NAME` and execs `qwen --yolo -p "$prompt"` (stdin →
`/dev/null`; multi-line prompt assembled from exactly one source, in that
priority order). `--yolo` is a hidden-but-working flag (`tools.approvalMode
"yolo"`); do **not** add `--safe-mode` — it disables hooks, including the one
below. Optional gate: `-k "kw1 kw2"` (or `$QWEN_YOLO_REQUIRE`) refuses to
launch unless the assembled prompt contains every keyword.

The `~/.qwen/hooks/airag-inject.sh` UserPromptSubmit hook reads the launch
identity back out of `$AIRAG_SOURCE` (default `qwen` for a plain session) and
queries **with `--scope <entity>`** — i.e. that entity's own indexed memory
plus common knowledge, never another entity's private scope. It tags its own
queries `<entity>-hook` (vs. `<entity>` for model-initiated calls) and, when
the hook payload carries a `session_id`, exports it as `AIRAG_SESSION` so the
injected queries get numbered too. One hook script serves every entity name —
no per-name hook copies needed.

## Gotchas

- **Embed server is mandatory** for `index`/`query`. Without `aillama
  embed-start` both error out; `airag query -r` additionally needs `aillama
  rerank-start`.
- **`index` never deletes.** Files removed from disk stay in the index until a
  `reindex` (which walks the roots and GCs vanished paths).
- **Dedup is content-level (sha256), across paths.** Same bytes at two paths →
  one `files` row, two `locations`; different content (different sha) → updated
  as a new version. `query`/`export` surface one canonical path per content.
- **After editing the scripts:** `sudo cp -f ~/Claude/bin/{aillama,airag,qwen-yolo}
  /usr/local/bin/` — the `/usr/local/bin` copies shadow `~/Claude/bin` in PATH.
- **Rebuild from a pre-v2 index:** the old (v1) DB cannot be read; `airag` refuses
  it with the exact rebuild command. Do:
  ```
  rm -f ~/.airag/index.db ~/.airag/index.db-wal ~/.airag/index.db-shm
  aillama embed-start
  airag index ~/.claude/skills ~/Claude/info ~/Claude/llm/BENCH.md ~/Claude/llm/README.md
  ```
