# Still Air: experiment results

Simulated associator (`SimLLM`), **20 independent worlds** (seeds), mean ± 95% CI. 24 two-hop chains, 5 sessions, idle budget 2400 tokens/session, hallucination rate 0.3, verifier miss rate 0.15, retrieval noise σ=0.04.

> These numbers test the *mechanism* in a controlled simulation. They are not evidence about real language models: the confabulation and verifier-error rates are assumptions, and the robustness table shows how the conclusions move when they change.

## 1. Headline: three conditions

Test budget 'strict' = 1 retrieval round reading 3 memories; 'generous' = 2 rounds x 3.

| Condition | idle tokens | connections stored | memory precision | chain coverage | acc (strict) | acc (generous) | wrong-answer rate (strict) | control acc | test tokens/q (strict / generous) |
|---|---|---|---|---|---|---|---|---|---|
| A  no idle loop | 0 ±0 | 0 ±0 | n/a | 0% ±0 | 0% ±0 | 100% ±0 | 0% ±0 | 99% ±1 | 40 ±0 / 70 ±0 |
| B  noise only, no selector | 12184 ±16 | 176 ±2 | 7% ±1 | 53% ±4 | 40% ±5 | 83% ±3 | 10% ±3 | 78% ±5 | 49 ±1 / 67 ±1 |
| C  full Still Air | 12246 ±27 | 18 ±1 | 74% ±4 | 56% ±4 | 50% ±4 | 92% ±3 | 7% ±2 | 96% ±3 | 48 ±1 / 62 ±1 |

Paired accuracy differences (same worlds), mean ± 95% CI:

- **C-A@strict**: 50% ±4 points
- **C-B@strict**: 10% ±7 points
- **C-A@generous**: -8% ±3 points
- **C-B@generous**: 8% ±5 points

## 2. Compute: where does the idle loop pay off?

- C spent **12246** idle tokens up front. At test time, A needs 70 tokens per question (2 hops) while C answers in 48 (1 hop when a connection exists).
- Per-question saving ≈ 22 tokens, so total compute breaks even after about **558 questions** (≈ 23.3 per chain). Below that, the idle loop costs more in total than doing the second hop at test time; above it, it is cheaper, *if* accuracy holds.

## 3. Ablations

| Variant | memory precision | chain coverage | acc (strict) | acc (generous) | control acc |
|---|---|---|---|---|---|
| A  no idle loop | n/a | 0% ±0 | 0% ±0 | 100% ±0 | 99% ±1 |
| C  full Still Air | 74% ±4 | 56% ±4 | 50% ±4 | 92% ±3 | 96% ±3 |
| B  noise only, no selector | 7% ±1 | 53% ±4 | 40% ±5 | 83% ±3 | 78% ±5 |
| C minus selector | 0% ±0 | 2% ±1 | 1% ±1 | 96% ±1 | 100% ±1 |
| C minus memory writes | n/a | 0% ±0 | 0% ±0 | 100% ±0 | 99% ±1 |
| C minus internal-state drive | 75% ±4 | 61% ±4 | 55% ±4 | 93% ±2 | 95% ±3 |

## 4. Idle budget sweep

| idle tokens/session | mode | chain coverage | memory precision | acc (strict) | acc (generous) | control acc |
|---|---|---|---|---|---|---|
| 0 | B  noise only, no selector | 0% ±0 | n/a | 0% ±0 | 100% ±0 | 99% ±1 |
| 0 | C  full Still Air | 0% ±0 | n/a | 0% ±0 | 100% ±0 | 99% ±1 |
| 300 | B  noise only, no selector | 21% ±3 | 20% ±3 | 19% ±4 | 90% ±3 | 93% ±3 |
| 300 | C  full Still Air | 17% ±3 | 94% ±5 | 16% ±2 | 99% ±1 | 99% ±1 |
| 600 | B  noise only, no selector | 34% ±4 | 17% ±2 | 29% ±3 | 88% ±4 | 86% ±4 |
| 600 | C  full Still Air | 25% ±3 | 84% ±5 | 23% ±3 | 97% ±2 | 98% ±2 |
| 1200 | B  noise only, no selector | 40% ±4 | 10% ±1 | 32% ±4 | 83% ±3 | 84% ±6 |
| 1200 | C  full Still Air | 36% ±3 | 79% ±5 | 34% ±3 | 95% ±2 | 98% ±2 |
| 2400 | B  noise only, no selector | 53% ±4 | 7% ±1 | 40% ±5 | 83% ±3 | 78% ±5 |
| 2400 | C  full Still Air | 56% ±4 | 74% ±4 | 50% ±4 | 92% ±3 | 96% ±3 |
| 4800 | B  noise only, no selector | 53% ±5 | 4% ±0 | 33% ±5 | 81% ±5 | 74% ±7 |
| 4800 | C  full Still Air | 74% ±3 | 67% ±3 | 65% ±4 | 89% ±2 | 92% ±3 |

## 5. Perturbation amplitude (drift failure vs coverage)

| amplitude | repetition rate | incoherence rate | chain coverage | acc (strict) |
|---|---|---|---|---|
| 0.0 | 55% ±3 | 40% ±3 | 31% ±2 | 27% ±3 |
| 0.1 | 53% ±2 | 40% ±2 | 31% ±3 | 28% ±3 |
| 0.25 | 48% ±2 | 45% ±2 | 35% ±3 | 29% ±3 |
| 0.5 | 40% ±3 | 49% ±3 | 56% ±4 | 50% ±4 |
| 0.75 | 39% ±2 | 48% ±2 | 65% ±3 | 57% ±4 |
| 1.0 | 40% ±2 | 48% ±3 | 63% ±4 | 57% ±4 |

## 6. Spontaneity vs constraint: targeted consolidation of open questions

Half the chain questions are asked (one hop); failures become open questions; the loop then runs extra idle budget. 'Asked' = recovery on those questions, 'unasked' = collateral gain on questions never asked.

| perturbation | asked: before -> after | asked gain | unasked gain |
|---|---|---|---|
| drive a=0.0 | 6% ±2 -> 52% ±4 | 46% ±3 | 0% ±0 |
| drive a=0.25 | 8% ±3 -> 47% ±6 | 38% ±5 | 5% ±3 |
| drive a=0.5 | 16% ±3 -> 33% ±5 | 17% ±4 | 20% ±4 |
| drive a=0.75 | 16% ±3 -> 40% ±5 | 24% ±5 | 18% ±5 |
| drive a=1.0 | 20% ±3 -> 40% ±7 | 21% ±6 | 18% ±5 |
| blind noise (no state) | 19% ±4 -> 39% ±6 | 20% ±5 | 23% ±5 |

## 7. Robustness to the simulator's assumptions

Rows vary the associator's hallucination rate (h) and the grounding judge's miss rate (m). Cells: memory precision / acc (generous) for B (no selector) and C (full).

| h | m | B precision | C precision | B acc (generous) | C acc (generous) | A acc (generous) | C acc (strict) |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.0 | 7% ±1 | 100% ±0 | 93% ±2 | 100% ±0 | 100% ±0 | 63% ±3 |
| 0.0 | 0.15 | 7% ±1 | 100% ±0 | 93% ±2 | 100% ±0 | 100% ±0 | 60% ±4 |
| 0.0 | 0.4 | 7% ±1 | 100% ±0 | 93% ±2 | 100% ±0 | 100% ±0 | 57% ±4 |
| 0.15 | 0.0 | 7% ±1 | 100% ±0 | 90% ±3 | 100% ±0 | 100% ±0 | 60% ±4 |
| 0.15 | 0.15 | 7% ±1 | 83% ±4 | 90% ±3 | 95% ±2 | 100% ±0 | 54% ±4 |
| 0.15 | 0.4 | 7% ±1 | 68% ±4 | 90% ±3 | 90% ±3 | 100% ±0 | 46% ±5 |
| 0.3 | 0.0 | 7% ±1 | 100% ±0 | 83% ±3 | 100% ±0 | 100% ±0 | 57% ±3 |
| 0.3 | 0.15 | 7% ±1 | 74% ±4 | 83% ±3 | 92% ±3 | 100% ±0 | 50% ±4 |
| 0.3 | 0.4 | 7% ±1 | 55% ±4 | 83% ±3 | 84% ±3 | 100% ±0 | 42% ±3 |
| 0.5 | 0.0 | 6% ±1 | 100% ±0 | 80% ±4 | 100% ±0 | 100% ±0 | 51% ±4 |
| 0.5 | 0.15 | 6% ±1 | 66% ±4 | 80% ±4 | 91% ±3 | 100% ±0 | 44% ±5 |
| 0.5 | 0.4 | 6% ±1 | 40% ±4 | 80% ±4 | 79% ±3 | 100% ±0 | 35% ±4 |
