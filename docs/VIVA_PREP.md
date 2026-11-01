# Still Air: viva preparation

Everyone in the team must be able to answer **every** question below. The viva mark (and "team work") is judged per person. Read it out loud to each other tonight.

## 0. The 20-second and 60-second answers

**20 seconds.** "Today's language models only compute when you prompt them. Still Air gives an agent an idle loop: small internal nudges make it think on its own, a selector keeps the connections that check out in persistent memory and drops the rest. We built a prototype and tested whether idle thinking helps later questions, compared with no idle loop and with a noise-only loop."

**60 seconds.** Add: "We built the memory, the perturbation source, the selector with grounding, the loop, and a task mode that answers under a budget. We ran three conditions with the idle compute matched in tokens, over 20 simulated worlds. With a tight test budget the idle loop answers 50% of two-hop questions against 0% for no idle loop. With a generous budget the no-idle baseline wins, 100% against 92%, because a few false connections slip past an imperfect verifier. The selector is essential: without it memory is 93% junk. So the result is partial support, and it depends on how good the verifier is."

## 1. Numbers to know cold

| Thing | Value |
|---|---|
| Worlds (independent seeds) | 20 (seeds 0-19); mean ± 95% CI |
| World | 24 two-hop chains, 5 sessions, 12 dangling relation facts, 12 near-name decoy property facts, 12 one-hop control questions |
| Idle budget | 2,400 tokens per session (≈12,200 total) for B and C: matched in tokens |
| Strict test budget | 1 retrieval round × 3 memories. Generous: 2 rounds × 3 |
| A (no idle) | 0% strict, **100%** generous; 40 / 70 tokens per question |
| B (noise only, no selector) | 40% strict, 83% generous; precision **7%**; control questions 78% |
| C (full Still Air) | **50%** strict, 92% generous; precision **74%**; coverage 56%; control 96% |
| Paired differences | C−A strict +50 ±4; C−A generous −8 ±3; C−B strict +10 ±7; C−B generous +8 ±5 |
| Break-even | ≈ 558 questions ≈ 23 per chain (12,246 idle tokens ÷ 22 tokens saved per question) |
| Simulator assumptions | hallucination rate 0.30, judge miss rate 0.15, retrieval noise σ 0.04 |
| Perfect judge (miss rate 0) | C = 100% at generous budget for every hallucination rate |
| Targeted consolidation | drive only (amplitude 0): recovers 46% of asked questions, 0% elsewhere; blind noise: ~20% / ~23% |
| Tests | 28 unit tests, all passing |

## 2. Concept questions

**What is Still Air in one line?** An architecture that gives an LLM agent a continuous idle loop, so it thinks between prompts and consolidates useful connections into memory.

**How is this different from just running an agent in a loop (AutoGPT-style)?** Those loops are goal-driven and act on the world. Still Air is *thought-only* (no external actions), is driven by internal perturbation rather than a task, and has a selector that decides what gets written to memory. The question we study is whether unprompted thinking, filtered properly, improves later performance.

**How is it different from sleep-time compute (Lin, Snell et al. 2025)?** Sleep-time compute lets a model reason offline about a *given* context before queries arrive, and reports about 5× less test-time compute at equal accuracy on their benchmarks. Still Air is continuous and perturbation-driven, with persistent memory writes gated by a selector, and we test whether the selector matters (B vs C). Our setting is also much smaller and simulated; do not claim our numbers are comparable to theirs.

**How is it different from Generative Agents' reflection (Park et al. 2023)?** Reflection there is periodic and summarises memories into insights. Ours is continuous, nudged by internal state, and every candidate is scored and may be discarded.

**What was the motivation for Still Air?** Abhishek's starting point: there is no obvious reason a language model should only reason or infer when it is prompted. People also think on their own: they consolidate what they learned, notice connections and chase loose ends. The hypothesis is that self-initiated thought is one missing ingredient of general intelligence, so we built a small, testable version of it.

**Is this AGI? How does it move towards AGI?** No, and the concept document says so. We do not claim it is AGI, and we did not measure generality. Our hypothesis is that self-initiated thought (initiative, incubation, consolidation) is one missing ingredient. We built a small testable version and found partial support: it helps when the test budget is tight, it needs a good checker, and it is not a general solution. A second lesson bears on autonomy: free thinking without a selector filled memory with junk (precision 7% against 74% with it), so letting a model think on its own is not enough; the checking step decides whether it works.

**Is the brain analogy justified?** It is an analogy, not a claim of equivalence. Token-level generation may be a coarse analogue of neural dynamics (the document lists this as a risk). We use the brain as motivation: the default mode network (Raichle 2001) is active at rest, and spontaneous thought (Christoff 2016) is thought freed from constraints. Perturbation supplies spontaneity; selector and grounding supply constraints.

**What does "no input means no prompt" mean?** A deterministic model with nothing perturbing it replays a single path. Cognition has to be initiated from inside (noise, curiosity, open questions), not by a user.

**What are the four jobs of idle activity?** Incubation, consolidation, rehearsal, initiative. Our experiment tests mainly consolidation (connecting facts learned apart) and initiative (working on open questions).

**What is the hypothesis?** An agent with an idle loop outperforms a compute-matched agent without one on later tasks that reward reusing and recombining earlier knowledge.

**Is the hypothesis supported?** *Partly.* Supported at a tight test budget (+50 points). Not supported at a generous budget or at equal total compute in our testbed (A is 100%, C is 92%). It depends on verifier quality: with a perfect judge C ties A. Say this with confidence; a mixed result with a reason is stronger than a clean claim.

## 3. Architecture questions

**Walk me through one idle step.** (`loop.py: IdleLoop.step`)
1. *Perturb:* `perturb.draw` builds a query = (1−a)·drive + a·noise. Drive is the loop's own focus, or an open question. Noise is a mix of two random memories.
2. *Retrieve seeds:* top-4 memories for that query, minus a habituation penalty on recently visited ones; plus one follow-up lookup ("what else do I know about this entity?").
3. *Associate:* the model produces a thought from the seeds.
4. *Evaluate:* the judge scores validity; novelty checks whether the claim (or near-identical text) is already stored.
5. *Consolidate:* keep (write a connection), reinforce (strengthen the existing one) or discard. Then focus and curiosity update, and unreinforced connections decay and may be pruned.
6. Log the whole event to a JSONL file.

**What does the selector do?** Scores validity (grounding) and novelty. Valid and novel → keep. Valid but already known → reinforce. Not valid → discard. Thresholds are 0.5 each.

**What is grounding?** Checking a candidate against the memories it cites. In the simulator it is a structural check: the linking entity must be *exactly* the same entity in both parent memories. In the Claude backend it is an LLM judge told to be strict. The concept document also mentions trusted sources; version 1 grounds against memory only.

**What are the perturbation knobs?** Amplitude (how much noise versus own drive), loop frequency, memory-write threshold, compute budget. In code: `Perturbation(amplitude)`, `live.py` rate, `Selector(validity_min, novelty_min)`, `run(token_budget=...)`.

**What are curiosity and uncertainty?** Internal state. Curiosity rises 0.05 each step that nothing is accepted and drops ×0.6 after a success. Uncertainty grows with the number of open questions. Both raise the chance the drive points at an open question.

**What is an open question?** A question the agent could not answer in task mode; stored as a node with status open. It becomes a drive for the idle loop (initiative).

**How does mode switching work?** `live.py: LiveSession.ask` takes the loop's lock, so idle thinking pauses while the task runs; afterwards the loop resumes, seeded with the question (`perturb.seed`). The dashboard badge flips IDLE to TASK.

**Why does memory fade?** Connections lose 0.002 strength per step and are pruned below 0.2, unless reinforced (+0.5 each time they are re-derived, +0.2 when used to answer). This is "the rest fade", and it also lets true links (re-derived often) outlast confabulations (rarely repeated).

**What is stored in memory?** Facts (from the user), connections (from the idle loop), open questions, and a self-model dictionary (counts of kept, reinforced, discarded, pruned, curiosity, uncertainty). It saves to and loads from JSON.

## 4. Code questions (Atheendre should lead, everyone should skim)

**Which file does what?** `memory.py` memory and retrieval; `perturb.py` nudges; `selector.py` keep/reinforce/discard; `loop.py` the loop and the factory for conditions; `agent.py` task mode; `llm.py` the simulated associator; `claude_backend.py` real-LLM backend; `world.py` synthetic worlds; `evaluate.py` and `report.py` the experiment; `live.py` and `web/index.html` the dashboard; `cli.py` commands.

**How does retrieval work?** TF-IDF cosine similarity over word tokens, pure Python. tf = 1 + log(count); idf = log((N+1)/(df+0.5)); vectors are L2-normalised. Why not embeddings? To keep the demo dependency-free and fully explainable; the interface (`retrieve`) could take an embedding model without other changes. Weakness: bag-of-words cannot match paraphrases.

**What is `SimLLM`, and is it cheating?** A deterministic, seeded stand-in for a language model. For two memories that chain, it writes the connection; with probability `hallucination_rate` (0.30) it instead joins a look-alike entity's fact (a confabulation); with no chain available it sometimes makes a loose analogy or restates a memory. Its judge checks the claim structurally and is fooled with probability `verifier_miss` (0.15). It is not cheating if we say what it is: it tests the *mechanism* under stated error rates, not real language ability. We sweep both rates.

**Is the experiment circular, since the simulator and the world share templates?** Partly, yes, and we say so. The simulator parses sentences using the same templates the world generates. What the experiment can show is how the loop behaves *given* a noisy associator and verifier: whether selection protects memory, what idle compute buys, how the budget behaves. It cannot show that real LLMs confabulate at 30%.

**How do you keep ground truth away from the agent?** `world.py` holds `valid_derived`. The agent only sees fact sentences. `evaluate.py` compares stored connections against `world.valid_derived` afterwards. The agent and the selector never read it.

**What is a "token" in your accounting?** For the simulator, the number of whitespace-separated words in prompts and outputs (a proxy). For the Claude backend it is `usage.input_tokens + output_tokens`.

**How is compute matched?** Each session, B and C each get 2,400 idle tokens (`loop.run(token_budget=...)`). B takes more steps because it makes no judge calls. Idle totals are ≈12,184 and ≈12,246. A gets no idle compute; instead we also report A at several test-time budgets.

**How is the loop deterministic?** Every random draw uses a seeded `random.Random`. A test checks that two runs give identical thought logs.

**Is the Claude backend tested?** Only against a fake client (shape of calls, parsing, refusal handling). It has **not** been run against the live API. Never claim otherwise.

**How would you change the amplitude live?** The dashboard slider calls `/api/config`, which sets `loop.perturb.amplitude`.

**How do you log thoughts?** `IdleLoop.step` appends a JSON line per step (`log_path`) and keeps `events` in memory; the dashboard stream shows the latest.

## 5. Evaluation questions (Kushi should lead)

**Why a synthetic world with invented names?** Real LLMs could have seen real facts in training. Invented names mean anything the agent knows must come from memory. Two-hop chains whose facts arrive in different sessions mean the answer cannot be found by retrieving one memory.

**What are your conditions?** A: no idle loop. B: idle loop with random noise only and no selector (everything kept). C: full Still Air. Plus ablations: remove the selector, remove memory writes, remove the internal-state drive; and an amplitude sweep.

**What are the metrics?** Two-hop accuracy; test tokens per question; memory precision (stored connections that are true, ground-truthed); chain coverage (fraction of true chains connected); repetition and incoherence rates from the thought log; control accuracy (one-hop questions, to detect harm).

**Why does A reach 100% with two rounds?** Because the world is solvable by plain two-step retrieval: find the relation fact, then look up the bridge entity. The idle loop's value is therefore speed and per-question cost, not a higher ceiling. That is why the strong form of the hypothesis fails here.

**Why does C lose 8 points to A at the generous budget?** False connections that slip past the judge (miss rate 0.15) sit in memory. At test time a false connection can outrank the true two-step path and the answer is confident and wrong. With a perfect judge C ties A (the heatmap's miss-rate-0 column is all 100).

**Why does more idle compute make the generous case worse?** Each extra idle step is another chance for a false link to pass the judge, so precision degrades as compute grows (99% to 89% over our sweep). This is the cost risk of an always-on system, and the reason a verify-on-use step is the first next step.

**Why does B look decent at the strict budget but is "bad"?** B does connect many chains (coverage 53%), so strict accuracy is 40%. But its memory is 93% junk, it wrecks questions that used to work (control accuracy 78% against 96%), and its strict accuracy falls as idle compute grows (33% at the biggest budget).

**What about "C minus selector" collapsing (2% coverage)?** We checked the mechanism on one world: with the selector off, the loop wrote 162 notes in 162 steps, of which 86 were restatements and 75 were loose analogies, and only 1 was a real derivation. 95% of the memories it then retrieved as seeds were its own junk notes, so it kept thinking about its own output. With the selector on (same world) only 21 notes were written, all derivations, and 20% of seed slots were connections. Without a selector, drift feeds on its own output. B avoids the collapse only because its random jumps keep re-exploring, at the cost of 93% junk.

**Is the internal drive useful?** In the untargeted headline benchmark, no (blind noise with a selector scored 55% against 50%). Its value is *targeted* consolidation of open questions: 46% recovery against ~20%, at the price of no collateral gain. That is the spontaneity-versus-constraint dial.

**How did you get confidence intervals?** Mean ± t-based 95% interval over the 20 worlds (`evaluate.mean_ci`); for differences we use paired differences on the same worlds.

**Were results tuned on the test seeds?** Be honest. The design additions (habituation, follow-up lookups, support-weighted answers) were made while looking at small runs, some on seeds that overlap the reported ones; development sweeps used seeds 100+ for the amplitude and budget choices. The default amplitude (0.5) and idle budget were set by judgement, and the full sweeps are reported. So treat the results as a demonstration of a mechanism under stated assumptions, not as a held-out benchmark.

**What would a real test look like?** Run A, B, C with a real LLM and judge on multi-session tasks with held-out questions, count tokens from API usage, and have a model or humans rate novelty and validity of stored connections.

**What is break-even?** Idle tokens spent ÷ tokens saved per question = 12,246 ÷ 22 ≈ 558 questions, about 23 per chain. Below that, the idle loop costs more in total.

## 6. Hard and critical questions

**Why should anyone care if A already gets 100%?** In real use compute at test time costs latency and money per query; idle compute can be spent while nobody is waiting and shared across many questions. The honest claim is shifting compute, not saving it, and it only pays if many questions reuse the same connections.

**What if the model just hallucinates connections?** That is the main risk, and the reason for the selector and grounding. Our experiment measures it: precision and accuracy fall as hallucination rises and the judge weakens (robustness table).

**Is an always-on system safe?** Version 1 is thought-only (no tools, no external actions) and logs every thought and memory write. Auditability and cost remain open risks.

**How do you define "useful thought"?** Hard, and open. We use proxies: later-task accuracy, stored-connection validity against ground truth (in simulation), and novelty. A real evaluation would need human or model judgement.

**Why not reinforcement learning?** Out of scope for version 1. The selector is a hand-designed scoring rule; learning it is future work.

**What would you do with more time?** Real-LLM run; verify-on-use at test time; compare noise-only, internal-state-driven and sensory-stream perturbations (roadmap step 2); latent or recurrent variants.

**What did you add that is not in the concept document?** Habituation, follow-up lookups, support-weighted answers, and the open-question experiment. We list them in the README and on slide 6.

## 7. Things not to claim

- That the real Claude backend works or was tested live. It was not.
- That the result "confirms the hypothesis". It partly supports it.
- That 5× comes from our work. It is the sleep-time compute paper's number on their benchmarks.
- That the simulated hallucination or judge rates match real models. They are assumptions.
- **Check before presenting:** the concept document says Hu & Ying (*Unified Mind Model*, arXiv 2503.03459) frames spontaneous and goal-directed thought as exploration and exploitation and calls spontaneous-thought modelling an open problem as of 2023. The arXiv abstract we could read describes a global-workspace-based agent architecture and does not mention either point. We left that claim off the slides. Abhishek should check the full text, or just describe the paper as a global-workspace-based cognitive architecture for LLM agents.
- Dong et al. 2025 (arXiv 2507.19495) is the *Psychological-mechanism Agent* (feeling, thought, action; a triple-network model for goal-directed and spontaneous thinking).

## 8. If you do not know an answer

Say what you do know, say what you would check, and point to where it is measured ("our robustness table shows how that moves"). Do not guess a number. It is fine to say "that is outside what we tested".

## 9. Cross-training: each person must be able to...

- [ ] Draw the loop (perturb, associate, evaluate, consolidate, repeat) and name the four components.
- [ ] Explain keep, reinforce, discard with an example, and say why a false link can still get through.
- [ ] State the three conditions and the headline result with the −8 included.
- [ ] Explain why the simulator is simulated, and what that limits.
- [ ] Run `python3 -m stillair demo --fast` and `python3 -m stillair serve` and narrate them.
- [ ] Say what a CI is and read one row of `results/results.md`.
