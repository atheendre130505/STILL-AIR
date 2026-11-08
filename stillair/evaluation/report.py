"""Turn results.json into results.md."""

from __future__ import annotations

from .config import HEADLINE_GENEROUS, HEADLINE_STRICT
from .stats import fmt

LABEL = {
    "A": "A  no idle loop",
    "B": "B  noise only, no selector",
    "C": "C  full Still Air",
    "noise_only": "B  noise only, no selector",
    "full": "C  full Still Air",
    "no_selector": "C minus selector",
    "no_writes": "C minus memory writes",
    "noise_selector": "C minus internal-state drive",
}


def _k(cfg):
    return f"H{cfg[0]}k{cfg[1]}"


S, G = _k(HEADLINE_STRICT), _k(HEADLINE_GENEROUS)


def _intro(res):
    d = res["defaults"]
    return [
        "# Still Air: experiment results\n",
        f"Simulated associator (`SimLLM`), **{len(res['seeds'])} independent worlds** (seeds), mean ± 95% CI. "
        f"{d['n_chains']} two-hop chains, {d['n_sessions']} sessions, idle budget "
        f"{d['idle_tokens_per_session']} tokens/session, hallucination rate {d['hallucination_rate']}, "
        f"verifier miss rate {d['verifier_miss']}, retrieval noise σ={d['retrieval_noise']}.\n",
        "> These numbers test the *mechanism* in a controlled simulation. They are not evidence about "
        "real language models: the confabulation and verifier-error rates are assumptions, and the "
        "robustness table shows how the conclusions move when they change.\n",
    ]


def _headline(res):
    L = [
        "## 1. Headline: three conditions\n",
        f"Test budget 'strict' = {HEADLINE_STRICT[0]} retrieval round reading {HEADLINE_STRICT[1]} memories; "
        f"'generous' = {HEADLINE_GENEROUS[0]} rounds x {HEADLINE_GENEROUS[1]}.\n",
        "| Condition | idle tokens | connections stored | memory precision | chain coverage | acc (strict) | acc (generous) | wrong-answer rate (strict) | control acc | test tokens/q (strict / generous) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in ("A", "B", "C"):
        r = res["headline"][m]
        L.append(f"| {LABEL[m]} | {fmt(r.get('idle_tokens'), digits=0)} | {fmt(r.get('connections'), digits=0)} | "
                 f"{fmt(r.get('precision'), True)} | {fmt(r.get('coverage'), True)} | {fmt(r[S+'.acc'], True)} | "
                 f"{fmt(r[G+'.acc'], True)} | {fmt(r[S+'.wrong'], True)} | {fmt(r[G+'.control_acc'], True)} | "
                 f"{fmt(r[S+'.tokens'], digits=0)} / {fmt(r[G+'.tokens'], digits=0)} |")
    L += ["", "Paired accuracy differences (same worlds), mean ± 95% CI:\n"]
    L += [f"- **{k}**: {fmt(v, True)} points" for k, v in res["paired"].items()]
    L.append("")
    return L


def _compute(res):
    d = res["defaults"]
    A, C = res["headline"]["A"], res["headline"]["C"]
    idle = C["idle_tokens"][0]
    t_a, t_c = A[G + ".tokens"][0], C[G + ".tokens"][0]
    acc_a, acc_c = A[G + ".acc"][0], C[G + ".acc"][0]
    L = [
        "## 2. Compute: where does the idle loop pay off?\n",
        f"- C spent **{idle:.0f}** idle tokens up front.",
        f"- At the same test budget ({HEADLINE_GENEROUS[0]} rounds), A uses {t_a:.0f} tokens per question "
        f"(accuracy {100*acc_a:.0f}%) and C uses {t_c:.0f} (accuracy {100*acc_c:.0f}%).",
    ]
    saving = t_a - t_c
    if saving > 0:
        be = idle / saving
        L.append(f"- Per-question saving ≈ {saving:.0f} tokens, so total compute breaks even after about "
                 f"**{be:.0f} questions** (≈ {be / d['n_chains']:.1f} per chain).")
        if acc_c < acc_a:
            L.append("- That break-even ignores C's lower accuracy at this budget. A cheaper one-round answer costs "
                     f"{C[S + '.tokens'][0]:.0f} tokens, but only reaches {100 * C[S + '.acc'][0]:.0f}% accuracy.")
    else:
        L.append("- C saves no test-time tokens at this budget, so the idle compute never breaks even.")
    L.append("")
    return L


def _ablations(res):
    L = [
        "## 3. Ablations\n",
        "| Variant | memory precision | chain coverage | acc (strict) | acc (generous) | control acc |",
        "|---|---|---|---|---|---|",
    ]
    for m in ("A", "C", "noise_only", "no_selector", "no_writes", "noise_selector"):
        r = res["ablations"][m]
        L.append(f"| {LABEL.get(m, m)} | {fmt(r.get('precision'), True)} | {fmt(r.get('coverage'), True)} | "
                 f"{fmt(r[S+'.acc'], True)} | {fmt(r[G+'.acc'], True)} | {fmt(r[G+'.control_acc'], True)} |")
    L.append("")
    return L


def _budget_sweep(res):
    L = [
        "## 4. Idle budget sweep\n",
        "| idle tokens/session | mode | chain coverage | memory precision | acc (strict) | acc (generous) | control acc |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in res["budget_sweep"]:
        r = row["summary"]
        L.append(f"| {row['budget']} | {LABEL[row['mode']]} | {fmt(r.get('coverage'), True)} | {fmt(r.get('precision'), True)} | "
                 f"{fmt(r[S+'.acc'], True)} | {fmt(r[G+'.acc'], True)} | {fmt(r[G+'.control_acc'], True)} |")
    L.append("")
    return L


def _amplitude_sweep(res):
    L = [
        "## 5. Perturbation amplitude (drift failure vs coverage)\n",
        "| amplitude | repetition rate | incoherence rate | chain coverage | acc (strict) |",
        "|---|---|---|---|---|",
    ]
    for row in res["amplitude_sweep"]:
        r = row["summary"]
        L.append(f"| {row['amplitude']} | {fmt(r.get('repetition_rate'), True)} | {fmt(r.get('incoherence_rate'), True)} | "
                 f"{fmt(r.get('coverage'), True)} | {fmt(r[S+'.acc'], True)} |")
    L.append("")
    return L


def _focus(res):
    L = [
        "## 6. Spontaneity vs constraint: targeted consolidation of open questions\n",
        "Half the chain questions are asked (one hop); failures become open questions; the loop then runs "
        "extra idle budget. 'Asked' = recovery on those questions, 'unasked' = collateral gain on questions never asked.\n",
        "| perturbation | asked: before -> after | asked gain | unasked gain |",
        "|---|---|---|---|",
    ]
    for r in res["focus"]:
        L.append(f"| {r['label']} | {fmt(r['asked_before'], True)} -> {fmt(r['asked_after'], True)} | "
                 f"{fmt(r['asked_gain'], True)} | {fmt(r['unasked_gain'], True)} |")
    L.append("")
    return L


def _robustness(res):
    L = [
        "## 7. Robustness to the simulator's assumptions\n",
        "Rows vary the associator's hallucination rate (h) and the grounding judge's miss rate (m). "
        "Cells: memory precision / acc (generous) for B (no selector) and C (full).\n",
        "| h | m | B precision | C precision | B acc (generous) | C acc (generous) | A acc (generous) | C acc (strict) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    a_gen = fmt(res["headline"]["A"][G + ".acc"], True)
    for row in res["robustness"]:
        b, c = row["B"], row["C"]
        L.append(f"| {row['h']} | {row['m']} | {fmt(b.get('precision'), True)} | {fmt(c.get('precision'), True)} | "
                 f"{fmt(b[G+'.acc'], True)} | {fmt(c[G+'.acc'], True)} | {a_gen} | {fmt(c[S+'.acc'], True)} |")
    L.append("")
    return L


def _upgrades(res):
    if "upgrades" not in res:
        return []
    L = [
        "## 8. Memory and background-thinking upgrades\n",
        "Condition C with each upgrade on its own, then all together, at the same idle token budget. "
        "The simulator's judge errors are independent between checks, so 'check link on use' is an optimistic estimate: "
        "a real judge may repeat its own mistake.\n",
        "| Variant | memory precision | chain coverage | acc (strict) | acc (generous) | wrong (generous) | control acc | idle steps | links expired or pruned |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in res["upgrades"]["rows"]:
        r = row["summary"]
        L.append(f"| {row['label']} | {fmt(r.get('precision'), True)} | {fmt(r.get('coverage'), True)} | "
                 f"{fmt(r[S+'.acc'], True)} | {fmt(r[G+'.acc'], True)} | {fmt(r[G+'.wrong'], True)} | "
                 f"{fmt(r[G+'.control_acc'], True)} | {fmt(r.get('idle_steps'), digits=0)} | {fmt(r.get('pruned'), digits=1)} |")
    L += ["", "Targeted recovery of open questions (setup as in section 6):\n",
          "| perturbation | asked gain | unasked gain |", "|---|---|---|"]
    for r in res["upgrades"]["focus"]:
        L.append(f"| {r['label']} | {fmt(r['asked_gain'], True)} | {fmt(r['unasked_gain'], True)} |")
    L.append("")
    return L


def to_markdown(res: dict) -> str:
    sections = (_intro, _headline, _compute, _ablations, _budget_sweep, _amplitude_sweep, _focus, _robustness, _upgrades)
    return "\n".join(line for section in sections for line in section(res))
