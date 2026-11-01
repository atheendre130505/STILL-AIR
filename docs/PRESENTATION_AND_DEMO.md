# Presentation and demo plan (CIE 3, 2nd or 3rd November)

Marks: scientific and technical content 5 · presentation 5 · team work 5 · code execution and demo 5 · class activity and viva 5.

## Who says what (about 12 minutes, then questions)

| Slides | Topic | Speaker | Time |
|---|---|---|---|
| 1 Cover, 2 Why, 3 Core idea, 4 Architecture, 5 Related work | Concept | **Abhishek** | 3.5 min |
| 6 What we built, 7 Live demo, 8 How a thought is judged, 9 Experiment design | Prototype and demo | **Atheendre** | 4.5 min (demo ≈ 2.5) |
| 10 Result 1, 11 Result 2, 12 Result 3, 13 Verdict, 14 Limits | Evaluation | **Kushi** | 4 min |
| 15 Takeaways | Close | **Abhishek** | 0.5 min |

Viva guide as a phone-friendly page (same content as `docs/VIVA_PREP.md`, questions collapsible): https://claude.ai/artifact/GZH92aVjA7rWKCZjxyx1CQ (also private until shared).

Speaker notes are in the deck (press the notes view). For questions: Abhishek takes concept and related work, Atheendre code and demo, Kushi experiment and statistics, but **anyone should answer anything**: the viva is individual. Hand off with a sentence ("Atheendre will show it running").

The deck: https://claude.ai/artifact/FvGxiTcCXwNAeZAmT5xvku (private to Abhishek; use the page's Share menu so Atheendre and Kushi can open it, and the download menu for a .pptx or PDF to take to the class machine).

## Demo: exact steps

Run from the project folder. Nothing to install: it uses only the Python standard library. (Tested on Python 3.14; other recent Python 3 versions should work but we have not tried them, so check on the presentation machine.)

```bash
python3 -m stillair serve
```

Open http://127.0.0.1:8765 and use dark or light as the projector needs (the ◐ button).

1. **Teach.** Click *Teach next session* four times. Dots appear (blue = facts). Say: "Two related facts arrive in different sessions. Dotted lines show which facts share an entity. Nobody has stated the combined fact."
2. **Ask before thinking.** Click a suggested question, then *Ask*. Both cards say "I do not know" (✗ unknown). An amber diamond appears: an open question.
3. **Idle.** Click *Start idle loop*. Let it run 20-30 seconds at 4-12 steps/s. Point at the thought stream (KEEP, REINFORCE, DISCARD badges) and at white rings (what it is thinking about now). Green nodes join dotted pairs.
4. **Ground truth.** Tick *show ground truth*. Say: "Red would be a false connection; the loop only sees green and the selector's scores." Show the precision bar.
5. **Ask again.** Click the same question and *Ask*. The right card ("With Still Air memory") answers in one hop; the left ("Without idle thinking") still does not. Point out the ✓.
6. **Mode switching.** Say: "While answering, the loop pauses; the badge reads TASK, then goes back to IDLE, and the loop resumes seeded with the question."
7. **Failure mode (30 seconds, optional but impressive).** Drag *hallucination rate* and *grounding miss rate* up to about 0.5, let it run for 20 seconds, and with *show ground truth* ticked point at any red node (a false connection that got through). Then untick *selector + grounding* and let it run another 20 seconds: memory fills with junk. Say: "This is why the selector exists." (Press *Reset* first if you want a clean slate; it also resets the sessions, so teach them again.)

**If the dashboard fails or the browser blocks it:** use the terminal.

```bash
python3 -m stillair demo --fast          # success run (about 30 seconds of output)
python3 -m stillair demo --seed 3 --fast # a run where a confabulated link slips through
python3 -m stillair demo --no-selector --fast  # condition B: nothing filters the drift
```

Drop `--fast` for paced output suitable for narration.

**If asked to show results:** `python3 -m stillair eval --seeds 20` (about a minute) or open `results/results.md`.

## Night-before checklist

- [ ] Everyone clones/copies the project folder to their own laptop and runs `python3 -m unittest discover -s tests` (28 tests should pass) and `python3 -m stillair demo --fast`.
- [ ] Check `python3 --version` on the presentation machine. If it is old, use the terminal demo; the package needs only the standard library.
- [ ] Download the deck as .pptx and PDF from the deck page and copy both to a USB stick or email them to yourselves.
- [ ] Start the server once and open the page, so you know it works on that machine (port 8765; use `--port 9000` if it is busy).
- [ ] Rehearse the 12 minutes once, out loud, with a timer. Practise the hand-offs.
- [ ] Each person reads `docs/VIVA_PREP.md` section 1 (numbers) and section 7 (things not to claim).
- [ ] Decide who runs the keyboard during the demo (Atheendre) and who watches the clock.

## During the presentation

- Say the honest result: "partly supported". Examiners reward a result you can defend and explain.
- Name the limits before anyone asks: simulated associator, synthetic world, Claude backend not run live.
- Do not read the slides. Each slide has one idea.
