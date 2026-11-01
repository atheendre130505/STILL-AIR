"""Turn results.json into results.md (tables the slides and viva notes quote directly)."""

from __future__ import annotations

from .evaluate import HEADLINE_GENEROUS, HEADLINE_STRICT, fmt

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


def to_markdown(res: dict) -> str:
    d = res["defaults"]
    n = len(res["seeds"])
    S, G = _k(HEADLINE_STRICT), _k(HEADLINE_GENEROUS)
    L = []
    w = L.append
    w("# Still Air: experiment results\n")
    w(f"Simulated associator (`SimLLM`), **{n} independent worlds** (seeds), mean ± 95% CI. "
      f"{d['n_chains']} two-hop chains, {d['n_sessions']} sessions, idle budget "
      f"{d['idle_tokens_per_session']} tokens/session, hallucination rate {d['hallucination_rate']}, "
      f"verifier miss rate {d['verifier_miss']}, retrieval noise σ={d['retrieval_noise']}.\n")
    w("> These numbers test the *mechanism* in a controlled simulation. They are not evidence about "
      "real language models: the confabulation and verifier-error rates are assumptions, and the "
      "robustness table shows how the conclusions move when they change.\n")

    w("## 1. Headline: three conditions\n")
    w(f"Test budget 'strict' = {HEADLINE_STRICT[0]} retrieval round reading {HEADLINE_STRICT[1]} memories; "
      f"'generous' = {HEADLINE_GENEROUS[0]} rounds x {HEADLINE_GENEROUS[1]}.\n")
    w("| Condition | idle tokens | connections stored | memory precision | chain coverage | acc (strict) | acc (generous) | wrong-answer rate (strict) | control acc | test tokens/q (strict / generous) |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for m in ("A", "B", "C"):
        r = res["headline"][m]
        w(f"| {LABEL[m]} | {fmt(r.get('idle_tokens'), digits=0)} | {fmt(r.get('connections'), digits=0)} | "
          f"{fmt(r.get('precision'), True)} | {fmt(r.get('coverage'), True)} | {fmt(r[S+'.acc'], True)} | "
          f"{fmt(r[G+'.acc'], True)} | {fmt(r[S+'.wrong'], True)} | {fmt(r[G+'.control_acc'], True)} | "
          f"{fmt(r[S+'.tokens'], digits=0)} / {fmt(r[G+'.tokens'], digits=0)} |")
    w("")
    w("Paired accuracy differences (same worlds), mean ± 95% CI:\n")
    for k, v in res["paired"].items():
        w(f"- **{k}**: {fmt(v, True)} points")
    w("")

    w("## 2. Compute: where does the idle loop pay off?\n")
    A, C = res["headline"]["A"], res["headline"]["C"]
    idle = C["idle_tokens"][0]
    tA, tC = A[G + ".tokens"][0], C[S + ".tokens"][0]
    nq = d["n_chains"]
    saving = tA - tC
    w(f"- C spent **{idle:.0f}** idle tokens up front. At test time, A needs {tA:.0f} tokens per question "
      f"(2 hops) while C answers in {tC:.0f} (1 hop when a connection exists).")
    if saving > 0:
        be = idle / saving
        w(f"- Per-question saving ≈ {saving:.0f} tokens, so total compute breaks even after about "
          f"**{be:.0f} questions** (≈ {be / nq:.1f} per chain). Below that, the idle loop costs more in total than "
          f"doing the second hop at test time; above it, it is cheaper, *if* accuracy holds.")
    w("")

    w("## 3. Ablations\n")
    w("| Variant | memory precision | chain coverage | acc (strict) | acc (generous) | control acc |")
    w("|---|---|---|---|---|---|")
    for m in ("A", "C", "noise_only", "no_selector", "no_writes", "noise_selector"):
        r = res["ablations"][m]
        w(f"| {LABEL.get(m, m)} | {fmt(r.get('precision'), True)} | {fmt(r.get('coverage'), True)} | "
          f"{fmt(r[S+'.acc'], True)} | {fmt(r[G+'.acc'], True)} | {fmt(r[G+'.control_acc'], True)} |")
    w("")

    w("## 4. Idle budget sweep\n")
    w("| idle tokens/session | mode | chain coverage | memory precision | acc (strict) | acc (generous) | control acc |")
    w("|---|---|---|---|---|---|---|")
    for row in res["budget_sweep"]:
        r = row["summary"]
        w(f"| {row['budget']} | {LABEL[row['mode']]} | {fmt(r.get('coverage'), True)} | {fmt(r.get('precision'), True)} | "
          f"{fmt(r[S+'.acc'], True)} | {fmt(r[G+'.acc'], True)} | {fmt(r[G+'.control_acc'], True)} |")
    w("")

    w("## 5. Perturbation amplitude (drift failure vs coverage)\n")
    w("| amplitude | repetition rate | incoherence rate | chain coverage | acc (strict) |")
    w("|---|---|---|---|---|")
    for row in res["amplitude_sweep"]:
        r = row["summary"]
        w(f"| {row['amplitude']} | {fmt(r.get('repetition_rate'), True)} | {fmt(r.get('incoherence_rate'), True)} | "
          f"{fmt(r.get('coverage'), True)} | {fmt(r[S+'.acc'], True)} |")
    w("")

    w("## 6. Spontaneity vs constraint: targeted consolidation of open questions\n")
    w("Half the chain questions are asked (one hop); failures become open questions; the loop then runs "
      "extra idle budget. 'Asked' = recovery on those questions, 'unasked' = collateral gain on questions never asked.\n")
    w("| perturbation | asked: before -> after | asked gain | unasked gain |")
    w("|---|---|---|---|")
    for r in res["focus"]:
        w(f"| {r['label']} | {fmt(r['asked_before'], True)} -> {fmt(r['asked_after'], True)} | "
          f"{fmt(r['asked_gain'], True)} | {fmt(r['unasked_gain'], True)} |")
    w("")

    w("## 7. Robustness to the simulator's assumptions\n")
    w("Rows vary the associator's hallucination rate (h) and the grounding judge's miss rate (m). "
      "Cells: memory precision / acc (generous) for B (no selector) and C (full).\n")
    w("| h | m | B precision | C precision | B acc (generous) | C acc (generous) | A acc (generous) | C acc (strict) |")
    w("|---|---|---|---|---|---|---|---|")
    a_gen = fmt(res["headline"]["A"][G + ".acc"], True)
    for row in res["robustness"]:
        b, c = row["B"], row["C"]
        w(f"| {row['h']} | {row['m']} | {fmt(b.get('precision'), True)} | {fmt(c.get('precision'), True)} | "
          f"{fmt(b[G+'.acc'], True)} | {fmt(c[G+'.acc'], True)} | {a_gen} | {fmt(c[S+'.acc'], True)} |")
    w("")
    return "\n".join(L)
