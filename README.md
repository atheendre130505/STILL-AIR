# Still Air: an idle loop for self-initiated AI thought

CIE 3 project · Abhishek Nair, Atheendre Ramesh, Kushi A Kumar

Today's language models only compute when prompted. Still Air gives an LLM-based agent a continuous **idle loop**:
small internal perturbations nudge it into self-initiated chains of thought, a **selector** keeps the connections
that check out in **persistent memory** and prunes the rest, and **grounding** stops the drift from collapsing into
repetition or nonsense. This repo is a working prototype of roadmap steps 1-4 in the concept document
(`Still Air An Idle Loop for Self-Initiated AI Thought.pdf`): a prototype loop, a perturbation study, a selector with
consolidation, and the three-condition evaluation.

Pure Python 3, **no dependencies** (`matplotlib` only for the optional chart script). Runs offline.

## Run it

```bash
python3 -m stillair demo --fast        # narrated terminal demo (drop --fast for paced output)
python3 -m stillair serve              # live dashboard at http://127.0.0.1:8765
python3 -m stillair eval --seeds 20    # three-condition experiment -> results/results.{json,md}  (~1 min)
python3 -m unittest discover -s tests  # 28 tests
```

Demo variants: `--seed 3` shows a run where a confabulated link slips past the verifier; `--no-selector` shows
condition B (nothing filters the drift); `--backend claude` uses the real Claude API (see below).

## How the code maps to the paper

| Concept document | Code | Notes |
|---|---|---|
| Perturbation source (noise, internal state) | `stillair/perturb.py` | a nudge is a memory query mixing *drive* (own focus, or an open question) with *noise*; `curiosity` and `uncertainty` drift |
| Persistent memory | `stillair/memory.py` | facts, connections, open questions, self-model; TF-IDF retrieval; JSON save/load; unreinforced connections decay and are pruned |
| Selector | `stillair/selector.py` | scores validity and novelty, then **keep / reinforce / discard** |
| Grounding | `SimLLM.judge`, `ClaudeLLM.judge` | checks a candidate against the memories it cites |
| The loop (perturb → associate → evaluate → consolidate → repeat) | `stillair/loop.py` | token budget, rate, every thought logged to JSONL |
| Mode switching | `stillair/live.py` (`ask`) | a prompt takes the loop's lock; afterwards the loop resumes seeded with the question |
| Task mode | `stillair/agent.py` | answers under a test-time budget (hops × memories read); failures become open questions |
| Evaluation plan (A / B / C, ablations, metrics) | `stillair/evaluate.py`, `stillair/report.py` | ground truth lives in the world generator, never in the agent |
| Oversight ("log every thought and memory write", thought-only) | `loop.py` event log; no action tools exist | |

## What the experiment found (20 worlds, simulated associator)

Full tables: [`results/results.md`](results/results.md). Charts: `results/charts/`.

| | A: no idle loop | B: noise only, no selector | C: full Still Air |
|---|---|---|---|
| accuracy, 1 retrieval round | 0% | 40% | **50%** |
| accuracy, 2 retrieval rounds | **100%** | 83% | 92% |
| memory precision | n/a | 7% | **74%** |
| one-hop control questions | 99% | 78% | **96%** |

Idle compute is matched in tokens (≈12.2k each for B and C). The honest reading:

- **Supported:** idle thinking lets a tight test-time budget answer questions it otherwise cannot (+50 points at one retrieval round), and the selector is what keeps memory usable (precision 74% vs 7%; B also damages questions that used to work).
- **Not supported:** at a *generous* test-time budget the no-idle baseline is best (100% vs 92%), because false connections that slip past an imperfect verifier shadow correct answers. With a perfect verifier (miss rate 0) C ties A at 100%. More idle compute does not rescue this: with an imperfect verifier it slowly makes it worse.
- **Total compute:** the idle loop only breaks even after roughly 23 questions per fact base. Its value here is *shifting* compute from test time to idle time (latency, per-question cost), not saving compute outright.
- **Spontaneity vs constraint is real:** a strong internal drive recovers 46% of questions it was asked but gains 0% elsewhere; high noise gains ~18% everywhere but only ~20% on asked questions.

## Limits you should state out loud

- The associator is **simulated** (`SimLLM`). Its hallucination rate (0.3) and the judge's miss rate (0.15) are *assumptions*; the robustness table sweeps both. Results show the mechanism works in a controlled world, not that real LLMs behave this way.
- The knowledge world is synthetic (two-hop chains over invented names, templated sentences). The simulator and the world share the templates.
- The real-LLM backend (`stillair/claude_backend.py`) is implemented and unit-tested against a fake client, but **has not been run against the live API** (no credentials were available when it was built). Treat it as untested until you run it once. It uses `claude-opus-5-5` by default; set `STILLAIR_MODEL` to change it, and add your own refusal-fallback policy if you need one.
- Design additions that are *not* in the concept document: habituation (inhibition of return), follow-up memory lookups inside a step, and preferring the better-supported of conflicting connections at answer time. See `docs/` for why each was added.

## Layout

```
stillair/        the package (memory, perturb, selector, loop, agent, llm, claude_backend, world, evaluate, report, live, cli)
web/index.html   the dashboard (single file, no CDN)
tests/           28 unit tests
scripts/         make_charts.py (needs matplotlib)
results/         results.json, results.md, charts/
docs/            VIVA_PREP.md, PRESENTATION_AND_DEMO.md (roles, demo steps, checklist)
```
