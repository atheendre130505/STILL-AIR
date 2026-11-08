# Still Air: experiment results

Simulated associator (`SimLLM`), **20 independent worlds** (seeds), mean ± 95% CI. 24 two-hop chains, 5 sessions, idle budget 2400 tokens/session, hallucination rate 0.3, verifier miss rate 0.15, retrieval noise σ=0.04.

> These numbers test the *mechanism* in a controlled simulation. They are not evidence about real language models: the confabulation and verifier-error rates are assumptions, and the robustness table shows how the conclusions move when they change.

## 1. Headline: three conditions

Test budget 'strict' = 1 retrieval round reading 3 memories; 'generous' = 2 rounds x 3.

| Condition | idle tokens | connections stored | memory precision | chain coverage | acc (strict) | acc (generous) | wrong-answer rate (strict) | control acc | test tokens/q (strict / generous) |
|---|---|---|---|---|---|---|---|---|---|
| A  no idle loop | 0 ±0 | 0 ±0 | n/a | 0% ±0 | 0% ±0 | 100% ±0 | 0% ±0 | 99% ±1 | 40 ±0 / 70 ±0 |
| B  noise only, no selector | 12184 ±16 | 176 ±2 | 7% ±1 | 53% ±4 | 40% ±5 | 83% ±3 | 10% ±3 | 78% ±5 | 49 ±1 / 67 ±1 |
| C  full Still Air | 12232 ±31 | 19 ±1 | 73% ±3 | 57% ±4 | 51% ±4 | 92% ±2 | 7% ±3 | 95% ±4 | 48 ±1 / 63 ±1 |

Paired accuracy differences (same worlds), mean ± 95% CI:

- **C-A@strict**: 51% ±4 points
- **C-B@strict**: 11% ±8 points
- **C-A@generous**: -8% ±2 points
- **C-B@generous**: 9% ±4 points

## 2. Compute: where does the idle loop pay off?

- C spent **12232** idle tokens up front.
- At the same test budget (2 rounds), A uses 70 tokens per question (accuracy 100%) and C uses 63 (accuracy 92%).
- Per-question saving ≈ 8 tokens, so total compute breaks even after about **1602 questions** (≈ 66.8 per chain).
- That break-even ignores C's lower accuracy at this budget. A cheaper one-round answer costs 48 tokens, but only reaches 51% accuracy.

## 3. Ablations

| Variant | memory precision | chain coverage | acc (strict) | acc (generous) | control acc |
|---|---|---|---|---|---|
| A  no idle loop | n/a | 0% ±0 | 0% ±0 | 100% ±0 | 99% ±1 |
| C  full Still Air | 73% ±3 | 57% ±4 | 51% ±4 | 92% ±2 | 95% ±4 |
| B  noise only, no selector | 7% ±1 | 53% ±4 | 40% ±5 | 83% ±3 | 78% ±5 |
| C minus selector | 0% ±0 | 2% ±1 | 1% ±1 | 96% ±1 | 100% ±1 |
| C minus memory writes | n/a | 0% ±0 | 0% ±0 | 100% ±0 | 99% ±1 |
| C minus internal-state drive | 75% ±4 | 63% ±4 | 59% ±4 | 93% ±2 | 92% ±4 |

## 4. Idle budget sweep

| idle tokens/session | mode | chain coverage | memory precision | acc (strict) | acc (generous) | control acc |
|---|---|---|---|---|---|---|
| 0 | B  noise only, no selector | 0% ±0 | n/a | 0% ±0 | 100% ±0 | 99% ±1 |
| 0 | C  full Still Air | 0% ±0 | n/a | 0% ±0 | 100% ±0 | 99% ±1 |
| 300 | B  noise only, no selector | 21% ±3 | 20% ±3 | 19% ±4 | 90% ±3 | 93% ±3 |
| 300 | C  full Still Air | 17% ±2 | 93% ±5 | 16% ±2 | 99% ±1 | 99% ±1 |
| 600 | B  noise only, no selector | 34% ±4 | 17% ±2 | 29% ±3 | 88% ±4 | 86% ±4 |
| 600 | C  full Still Air | 25% ±3 | 84% ±5 | 23% ±3 | 97% ±2 | 98% ±2 |
| 1200 | B  noise only, no selector | 40% ±4 | 10% ±1 | 32% ±4 | 83% ±3 | 84% ±6 |
| 1200 | C  full Still Air | 39% ±3 | 80% ±4 | 37% ±3 | 95% ±2 | 95% ±3 |
| 2400 | B  noise only, no selector | 53% ±4 | 7% ±1 | 40% ±5 | 83% ±3 | 78% ±5 |
| 2400 | C  full Still Air | 57% ±4 | 73% ±3 | 51% ±4 | 92% ±2 | 95% ±4 |
| 4800 | B  noise only, no selector | 53% ±5 | 4% ±0 | 33% ±5 | 81% ±5 | 74% ±7 |
| 4800 | C  full Still Air | 75% ±3 | 67% ±3 | 66% ±4 | 91% ±2 | 90% ±4 |

## 5. Perturbation amplitude (drift failure vs coverage)

| amplitude | repetition rate | incoherence rate | chain coverage | acc (strict) |
|---|---|---|---|---|
| 0.0 | 56% ±2 | 39% ±2 | 32% ±2 | 29% ±3 |
| 0.1 | 56% ±3 | 39% ±3 | 33% ±3 | 30% ±4 |
| 0.25 | 49% ±2 | 45% ±2 | 35% ±3 | 31% ±3 |
| 0.5 | 40% ±3 | 49% ±3 | 57% ±4 | 51% ±4 |
| 0.75 | 39% ±2 | 49% ±2 | 65% ±4 | 58% ±4 |
| 1.0 | 41% ±2 | 48% ±3 | 62% ±3 | 58% ±4 |

## 6. Spontaneity vs constraint: targeted consolidation of open questions

Half the chain questions are asked (one hop); failures become open questions; the loop then runs extra idle budget. 'Asked' = recovery on those questions, 'unasked' = collateral gain on questions never asked.

| perturbation | asked: before -> after | asked gain | unasked gain |
|---|---|---|---|
| drive a=0.0 | 6% ±2 -> 53% ±4 | 47% ±4 | 0% ±0 |
| drive a=0.25 | 9% ±3 -> 49% ±5 | 40% ±5 | 3% ±2 |
| drive a=0.5 | 17% ±3 -> 36% ±4 | 19% ±4 | 20% ±3 |
| drive a=0.75 | 16% ±3 -> 41% ±5 | 25% ±5 | 19% ±5 |
| drive a=1.0 | 20% ±3 -> 40% ±7 | 21% ±6 | 18% ±5 |
| blind noise (no state) | 20% ±4 -> 40% ±6 | 20% ±5 | 22% ±5 |

## 7. Robustness to the simulator's assumptions

Rows vary the associator's hallucination rate (h) and the grounding judge's miss rate (m). Cells: memory precision / acc (generous) for B (no selector) and C (full).

| h | m | B precision | C precision | B acc (generous) | C acc (generous) | A acc (generous) | C acc (strict) |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.0 | 7% ±1 | 100% ±0 | 93% ±2 | 100% ±0 | 100% ±0 | 63% ±3 |
| 0.0 | 0.15 | 7% ±1 | 100% ±0 | 93% ±2 | 100% ±0 | 100% ±0 | 63% ±3 |
| 0.0 | 0.4 | 7% ±1 | 100% ±0 | 93% ±2 | 100% ±0 | 100% ±0 | 59% ±3 |
| 0.15 | 0.0 | 7% ±1 | 100% ±0 | 90% ±3 | 100% ±0 | 100% ±0 | 60% ±4 |
| 0.15 | 0.15 | 7% ±1 | 83% ±4 | 90% ±3 | 95% ±2 | 100% ±0 | 54% ±4 |
| 0.15 | 0.4 | 7% ±1 | 69% ±4 | 90% ±3 | 92% ±3 | 100% ±0 | 51% ±4 |
| 0.3 | 0.0 | 7% ±1 | 100% ±0 | 83% ±3 | 100% ±0 | 100% ±0 | 57% ±3 |
| 0.3 | 0.15 | 7% ±1 | 73% ±3 | 83% ±3 | 92% ±2 | 100% ±0 | 51% ±4 |
| 0.3 | 0.4 | 7% ±1 | 55% ±3 | 83% ±3 | 84% ±3 | 100% ±0 | 45% ±4 |
| 0.5 | 0.0 | 6% ±1 | 100% ±0 | 80% ±4 | 100% ±0 | 100% ±0 | 51% ±4 |
| 0.5 | 0.15 | 6% ±1 | 65% ±4 | 80% ±4 | 92% ±3 | 100% ±0 | 44% ±4 |
| 0.5 | 0.4 | 6% ±1 | 41% ±3 | 80% ±4 | 78% ±2 | 100% ±0 | 37% ±4 |

## 8. Memory and background-thinking upgrades

Condition C with each upgrade on its own, then all together, at the same idle token budget. The simulator's judge errors are independent between checks, so 'check link on use' is an optimistic estimate: a real judge may repeat its own mistake.

| Variant | memory precision | chain coverage | acc (strict) | acc (generous) | wrong (generous) | control acc | idle steps | links expired or pruned |
|---|---|---|---|---|---|---|---|---|
| C as reported | 73% ±3 | 57% ±4 | 51% ±4 | 92% ±2 | 8% ±2 | 95% ±4 | 134 ±2 | 0.0 ±0.0 |
| + trial period for new links | 73% ±3 | 57% ±4 | 27% ±4 | 96% ±2 | 4% ±2 | 99% ±1 | 134 ±2 | 0.0 ±0.0 |
| + check link on use | 73% ±3 | 57% ±4 | 49% ±4 | 99% ±1 | 1% ±1 | 93% ±4 | 134 ±2 | 0.0 ±0.0 |
| + expire unconfirmed links | 77% ±3 | 57% ±4 | 50% ±4 | 92% ±2 | 8% ±2 | 96% ±3 | 134 ±2 | 1.4 ±0.7 |
| + skip unlinked seeds | 88% ±4 | 72% ±3 | 66% ±3 | 98% ±1 | 2% ±1 | 92% ±4 | 3124 ±569 | 0.0 ±0.0 |
| + adaptive noise | 76% ±4 | 65% ±4 | 58% ±4 | 92% ±2 | 8% ±2 | 91% ±4 | 135 ±2 | 0.0 ±0.0 |
| all upgrades | 94% ±2 | 69% ±4 | 38% ±3 | 100% ±0 | 0% ±0 | 99% ±1 | 3115 ±571 | 2.5 ±1.2 |

Targeted recovery of open questions (setup as in section 6):

| perturbation | asked gain | unasked gain |
|---|---|---|
| fixed amplitude 0.5 | 19% ±4 | 20% ±3 |
| adaptive noise | 46% ±5 | 7% ±3 |
