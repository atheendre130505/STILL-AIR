# Still Air

**An assistant that keeps working on your problems after you stop typing.**

CIE 3 project · Abhishek Nair, Atheendre Ramesh, Kushi A Kumar

You know when you can't solve something, so you take a break, and later the answer just comes to you? Or you learn a fact, and days later realise it connects to something else? That is called **incubation**. Normal chatbots can't do it, because they do nothing between messages. Still Air can.

- A **small model on your own computer** keeps thinking whenever you are not typing.
- A **large model on an API** does almost no work. It answers when you speak to it, and briefly checks anything the small model believes it has solved.

**Example.** You tell it you moved your plants next to the radiator, and that the heating is on all day. Later you say your plants keep dying and you can't work out why. You leave. On its own, it reasons that the radiator is probably overheating them, and tells you.

---

## Setting it up

You need Python 3, a free Groq API key, and Ollama. Python needs no extra packages.

**1. Get a free key** at [console.groq.com](https://console.groq.com).

**2. Save the key** in a new file called `.env` in this folder. Git ignores this file, so your key stays private.

```text
STILLAIR_API_KEY=your-key-here
```

**3. Install [Ollama](https://ollama.com)** and download the two small models (about 2.5 GB in total):

```bash
ollama pull phi4-mini
ollama pull all-minilm
```

**4. Start it:**

```bash
python -m stillair
```

Open http://127.0.0.1:8770 in your browser. For a live view of its thinking, open http://127.0.0.1:8770/backend beside it.

---

## How it works

```text
 You ──────► LARGE MODEL ──────► reply
              ▲    (only answers)  │
              │                    │ your words + its reply
     reads    │                    ▼
   what's     │             SMALL MODEL DIGESTS
   ready      │             notes + unsolved problems
              │                    │
              │                    ▼
         ┌────┴─────────  MEMORY  ◄──────────────┐
         │                    │                  │ conclusion
         │                    ▼                  │
         │        INCUBATION (while you're away) │
         │        pick a problem or something    │
         │        maths: what comes to mind?     │
         │        small model: next thought ─────┘
         │        repeat, or try a new angle
         │
         └── something looks solved → large model checks it → tells you
```

### 1. Digest

After each exchange, the small model reads **both your message and the large model's reply**. It pulls out clear notes, and anything you are stuck on becomes an **unsolved problem**.

### 2. Incubate

Whenever you are not typing, it starts a **chain of thought**:

1. **Pick what to work on.** Two out of three chains go to unsolved problems. The rest go to things you have learned, asking "does this connect to anything?"
2. **What comes to mind.** Maths finds related memories. Sometimes an unrelated memory is added on purpose, like a distraction, because that is where new angles come from.
3. **Think one step further.** The small model sees the goal, its train of thought so far, and what came to mind, then writes its next thought.
4. **Keep going, or stop.** It continues for up to 5 steps, until it reaches a conclusion or a dead end.
5. **Remember what failed, then let go.** A dead end is noted so the next attempt differs. After three failures the problem sleeps for a couple of minutes and those attempts are forgotten, so it returns with a clean slate, the way a break helps you.

**New information wakes things up.** Anything new that relates to an open problem sends it straight to the front of the queue, even if it was asleep, with that new thing guaranteed to come to mind.

**It reads a little.** Every few chains it fetches one short Wikipedia summary: usually on an open problem, sometimes completely at random. Only about three keywords leave your computer. Reading is treated purely as material to think about, fades after a few chains unless it led to a conclusion, and is never shown to the large model as fact.

Only conclusions enter memory. The scratch work disappears. Then it starts another chain, again and again, for free on your computer.

### 3. Telling you

When it believes it has solved a problem, or noticed something urgent, the large model checks the reasoning. That is the only time the large model is used between messages. If the conclusion holds up, it appears in your chat without being asked. If not, it is deleted and the problem reopens with a note about what went wrong.

Any other conclusion just waits in memory. When a later message relates to it, the large model sees it marked as uncertain, and either uses it or discards it as part of answering.

### The maths

Two memories count as related if they share rare words, or if a small embedding model finds they mean similar things, or if one was concluded from the other. Starting from one memory, activation spreads along those links and fades at each step, so it can reach things only indirectly related:

```text
0.5 × sender's activation × (this link's strength ÷ total strength of the sender's links)
```

### Keeping memory lean

Conclusions weaken with every reply unless they are used, and are dropped after about 16 unused replies. Only the 40 strongest are kept. A note or conclusion that means the same as an existing one never enters.

---

## On the page

| Control | What it does |
|---|---|
| **Conscious switch** | Turns the large model off, so you can watch the small model solve things alone. Nothing is sent to the API, and its conclusions appear marked as unchecked. Switching it back on makes the large model check everything that happened while it was off. |
| **Think dropdown** | How long it thinks after you stop: until your next message, or 5, 15 or 60 minutes. |
| **Still working on** | The problems it is incubating. |
| **Worked out while idle** | Its conclusions so far. |
| **New chat / Clear chat** | A blank conversation, or empty this one. Long-term memory is kept. |
| **Backend view** | Watch the current train of thought and the memory network live. It also has **Clear memory**, which permanently forgets all memories, conclusions and problems. |

---

## Limits

- Its ideas are only as good as a small model running on a CPU. It can reach wrong conclusions, which is why the large model checks them, and it will struggle with hard problems.
- A chain takes about a minute on a laptop CPU, and thinking keeps the CPU busy. Use the dropdown to limit it.
- It has not been measured against a version without incubation, so how much it helps is not yet known.
- It only talks in the chat, and takes no other actions.
- Your conversations and memories are saved to `stillair_memory.json` on your computer. What you type is also sent to Groq so the large model can answer.

---

## Settings

Put these in `.env` to change anything.

| Setting | What it changes | Default |
|---|---|---|
| `STILLAIR_API_KEY` | Your API key (required for Groq) | none |
| `STILLAIR_MAIN_MODEL` | The large model | `openai/gpt-oss-120b` |
| `STILLAIR_SMALL_MODEL` | The small model | `phi4-mini` |
| `STILLAIR_EMBED_MODEL` | The meaning model | `all-minilm` |
| `STILLAIR_BASE_URL` | Any OpenAI-compatible API for the large model | Groq |
| `STILLAIR_LOCAL_URL` | Where Ollama is running | `http://127.0.0.1:11434/v1` |

---

## Research behind it

- [Sleep-time Compute](https://arxiv.org/abs/2504.13171): thinking ahead while idle.
- [Synapse](https://arxiv.org/abs/2601.02744): spreading activation over memory.
- [Hippo](https://arxiv.org/abs/2610.24453): learning from what failed.
- [HaluMem](https://arxiv.org/abs/2511.03506): false memories start at the moment of writing.

The original simulation experiment for this project is preserved in the git history at commit `66fe977`.

---

## What is in this folder

```text
stillair/
  subconscious.py   digesting and incubating: the small model's work
  conscious.py      answering and checking: the large model's work
  links.py          the maths: how related memories are, spreading, search
  memory.py         notes, problems and conclusions, and how they fade
  session.py        runs the two together
  state.py          chats and everything saved to disk
  model.py          talks to Groq and Ollama
  server.py         starts everything and serves the pages
web/                chat.html and backend.html
tests/              automated tests: python -m unittest discover -s tests
```
