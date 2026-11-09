# Still Air

**An assistant that collects what you tell it, connects it while you are away, and reminds you without being asked.**

CIE 3 project · Abhishek Nair, Atheendre Ramesh, Kushi A Kumar

Normal chatbots do nothing between messages. Still Air keeps going:

- A **large model on an API** does very little. It answers when you speak to it, and sets any reminders in that same call.
- A **small model on your own computer** notes what you said, reads about what you mentioned, and looks for connections whenever you are not typing.
- **Maths** decides which memories are worth comparing. Nothing is random.
- **A clock** delivers reminders, with no model involved.

**Example.** You mention you collected your blood test report. Later you say your doctor's appointment about it is today, and ask for a reminder. While you are away, the small model connects the two. When the reminder arrives, the connection comes with it.

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
 You ──────► LARGE MODEL ──────► reply + any reminders ──► CLOCK ──► reminds you
              ▲                    │                                     ▲
              │ reads what's       │ your words + its reply              │ adds linked
              │ relevant           ▼                                     │ connections
              │             SMALL MODEL DIGESTS                          │
              │             notes + topics you mentioned                 │
              │                    │                                     │
              │                    ▼                                     │
              └───────────  MEMORY ◄──────────────┬──────────────────────┘
                              │    ▲              │
             topics you       ▼    │ page         │ connection
             mentioned ──► WIKIPEDIA              │
                              │                   │
                              ▼                   │
                 maths picks the closest pair ──► small model: do these connect?
```

### 1. Collect

After each exchange, the small model pulls out lasting notes, such as appointments, plans, people and preferences. It also picks out up to two topics you mentioned.

### 2. Read about what you mentioned

Each topic is looked up on Wikipedia, and the result closest in meaning to that topic is kept. A topic only counts if every word of it was in your message, so reading always stays on what you talked about. Only those short topics leave your computer. Pages are never shown to the large model directly.

### 3. Connect

Maths finds the two most related memories that have not been compared yet, and at least one must be your own words. The small model says whether they really connect. A connection must mention something from both sides, or it is thrown away. Each pair is judged once. When no pairs are left it waits for you to say something new.

### 4. Remind

While replying, the large model sets a reminder for anything upcoming: the exact time you asked for, otherwise the evening before, or a few hours before for something today. When the time comes the clock posts it in your chat, along with anything the small model connected to that memory. Reminders missed while the app was closed are delivered when it starts again.

### The maths

Two memories count as related if they share rare words, or if a small embedding model finds they mean similar things, or if one is a connection made from the other. When searching for a reply, activation spreads along those links and fades at each step, so it can reach things only indirectly related:

```text
0.5 × sender's activation × (this link's strength ÷ total strength of the sender's links)
```

### Keeping memory lean

Connections weaken with every reply unless they are used, and are dropped after about 16 unused replies. Only the 40 strongest are kept, and only the newest 25 Wikipedia pages. A note or connection that means the same as an existing one never enters.

---

## On the page

| Control | What it does |
|---|---|
| **Conscious switch** | Turns the large model off. Your messages are still collected and nothing is sent to the API. Switching it back on makes the large model check every connection made and respond to what you said. |
| **Think dropdown** | How long it works after you stop: until your next message, or 5, 15 or 60 minutes. |
| **Reminders coming up** | What it will bring up, and when. |
| **Connections it made** | What the small model has linked so far. |
| **New chat / Clear chat** | A blank conversation, or empty this one. Long-term memory is kept. |
| **Backend view** | Watch the small model and the memory network live. It also has **Clear memory**, which permanently forgets all memories, connections and reminders. |

---

## Limits

- The small model running on a CPU can make weak or wrong connections. The large model sees them marked as uncertain and can discard them. It cannot solve hard problems by itself, so that is not what this is for.
- Comparing pairs keeps the CPU busy for a while after you stop. Use the dropdown to limit it.
- A reminder is only set while the large model is on, or when it catches up after being switched back on.
- It has not been measured against a version without connections, so how much they help is not yet known.
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
- [HaluMem](https://arxiv.org/abs/2511.03506): false memories start at the moment of writing.

The original simulation experiment for this project is preserved in the git history at commit `66fe977`.

---

## What is in this folder

```text
stillair/
  subconscious.py   collecting, reading, connecting: the small model's work
  conscious.py      answering and checking: the large model's work
  reminders.py      setting and delivering reminders
  links.py          the maths: how related memories are, spreading, search
  memory.py         notes, pages and connections, and how they fade
  session.py        runs the two together
  state.py          chats and everything saved to disk
  model.py          talks to Groq and Ollama
  server.py         starts everything and serves the pages
web/                chat.html and backend.html
tests/              automated tests: python -m unittest discover -s tests
```
