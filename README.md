# Still Air

**An AI assistant that keeps thinking after you stop typing.**

CIE 3 project · Abhishek Nair, Atheendre Ramesh, Kushi A Kumar

Normal chatbots do nothing between your messages. Still Air has two parts that share one memory:

- A **conscious** mind: a larger model that answers you.
- A **subconscious**: a small model on your own computer that quietly looks for connections in what you have told it, whenever you are not typing.

If it notices something useful, it tells you without being asked.

**Example.** You mention a dentist appointment on Friday at 3pm. Later, you mention your sister's wedding rehearsal on Friday afternoon. You stop typing. A little later, this appears in the chat by itself:

> Hey, just a heads-up: your dentist appointment at 3 PM on Friday might overlap with your sister's wedding rehearsal later that afternoon.

---

## Setting it up

You need three things: Python 3, a free Groq API key, and Ollama. Nothing else needs installing.

**1. Get a free key** at [console.groq.com](https://console.groq.com).

**2. Save the key** in a new file called `.env` in this folder. Git ignores this file, so your key stays private.

```text
STILLAIR_API_KEY=your-key-here
```

**3. Install [Ollama](https://ollama.com)**, then download the two small models (about 2.5 GB in total):

```bash
ollama pull phi4-mini
ollama pull all-minilm
```

**4. Start it:**

```bash
python -m stillair chat
```

Open http://127.0.0.1:8770 in your browser.

---

## Using it

| On the page | What it does |
|---|---|
| **Message box** | Talk to it normally. It remembers what you tell it. |
| **Subconscious switch** | Turns background thinking on or off. |
| **Think dropdown** | How long it thinks after you stop: until your next message, or 5, 15 or 60 minutes. |
| **Reminders coming up** | Heads-ups it has scheduled from things you mentioned. |
| **Hunches it formed** | Connections it found by itself. |
| **New chat** | Starts a blank conversation. Old chats stay in the list, and memory is shared across all of them. |
| **Clear chat** | Empties the open conversation. Long-term memory and hunches are kept. |
| **Backend view** | Opens a live picture of what is happening inside (below). |

### Seeing what goes on inside

Click **Backend view**, or open http://127.0.0.1:8770/backend, and keep it beside the chat. You will see:

- **The memory network.** Each dot is a memory. Blue dots are your own words and green dots are hunches.
- **Lines between dots** show why two memories are related: shared words, similar meaning, close in time, or a stored link. A thicker line is a path that has proved useful.
- **Glow** shows activation spreading. **White rings** mark the two memories being compared. **Amber rings** mark what the main model is reading.
- **The five steps** of the subconscious, lit up as they happen.
- **The thought stream**, a record of every thought and how it ended.

### Things to try

| Say this | Then | You should see |
|---|---|---|
| "My brother Ravi lives in Pune and works as a vet." | Click New chat, ask "What does my brother do?" | It remembers. |
| "My tooth has been hurting badly." | New chat, ask "Anything medical I should deal with?" | It connects them by meaning, with no shared words. |
| "We are adopting a puppy next week." then "Our landlord does not allow pets." | Stop typing for a minute or two | A hunch forms, and it may warn you unprompted. |
| "I have a job interview tomorrow at 10am." | Look at Reminders coming up | A reminder is scheduled. |
| (Nothing about a dog) | Ask "What's my dog's name?" | It should say you never told it. |

---

## How it works

```text
 You ──> CONSCIOUS (larger model, on the API) ──> reply
            │ saves your words      ▲ reads hunches
            ▼                       │
         SHARED MEMORY  (your words + hunches)
            │ finds a pair          ▲ stores a checked link
            ▼                       │
         IDLE ALGORITHM ──> SUBCONSCIOUS (small model, on your computer)
         (no AI, free)
```

### When you send a message

The main model looks up related memories, reads any new hunches and replies. Your message is saved in your exact words. It is never summarised or rewritten, because research found that heavy summarising makes memory much worse.

### After you stop typing

About 3 seconds after the reply, this loop begins:

1. **Pick a memory.** It starts with things left unresolved, then recent things, then things it has not revisited much.
2. **Spread outward.** Activation flows from that memory to related ones, fading as it goes, so it can reach something only indirectly connected.
3. **Choose a partner.** It takes the most strongly activated memory it has not already tried.
4. **Ask the small model:** "Do these two connect in a way that neither says alone?"
5. **Check.** The small model checks its own answer.
6. **Store it** as a hunch if it passes.

Steps 1 to 3 are plain maths, with no AI involved. When every pair has been tried, it goes round again and reaches further. All of this runs on your computer, so it uses no API quota.

### How it decides two memories are related

It scores each pair four ways and uses whichever is strongest:

| Reason | In simple words |
|---|---|
| **Shared words** | Both mention a rare word such as "Friday" or "dentist". Each memory becomes a list of word scores, and rarer words count more. |
| **Similar meaning** | A small embedding model turns each sentence into numbers based on meaning, so "tooth" and "dentist" land close together. |
| **Close in time** | Things you said around the same time get a weak link that fades over time. |
| **Stored link** | A hunch is firmly connected to the two memories it came from. |

Paths the main model has actually used get up to twice as strong, so the subconscious learns what helps.

### How activation spreads

The starting memory gets 1.0. Each link passes on:

```text
0.5 × sender's activation × (this link's strength ÷ total strength of the sender's links)
```

The 0.5 makes it fade at each step. The division shares it out, so a memory linked to everything cannot dominate.

### How it speaks first

- **Reminders.** When you mention an upcoming event, the main model schedules a heads-up: a few hours before for something today, otherwise the evening before.
- **Speaking up.** Small models sometimes invent details, so the main model checks new hunches against your own words, at most every 3 minutes. Invented ones are deleted, and anything worth raising appears in the chat, labelled "Brought up on its own".

After two messages you have not answered, it stays quiet until you reply. This stops an idle night from using up your API quota.

### Keeping memory healthy

A hunch the main model uses gets stronger. One that goes unused slowly fades as the conversation continues, and is eventually deleted. Your own words never fade.

---

## Limits

- Whether this makes the assistant more useful has not been measured yet.
- It only talks in the chat. It cannot send phone or desktop notifications or take actions.
- Reminders only fire while the app is running. Missed ones appear the next time you start it.
- Old memories are never overwritten, so if a plan changes, it sees both versions and has to work out which is current.
- Background thinking keeps your CPU busy. Use the switch or the dropdown to limit it.
- Your conversations and memories are saved to `stillair_memory.json` on your computer. What you type is also sent to Groq so the main model can answer.

---

## Settings

Put these in `.env` if you want to change anything.

| Setting | What it changes | Default |
|---|---|---|
| `STILLAIR_API_KEY` | Your API key (required) | none |
| `STILLAIR_MAIN_MODEL` | The conscious model | `openai/gpt-oss-120b` |
| `STILLAIR_SMALL_MODEL` | The subconscious model | `phi4-mini` |
| `STILLAIR_EMBED_MODEL` | The meaning model | `all-minilm` |
| `STILLAIR_BASE_URL` | Any OpenAI-compatible API for the main model | Groq |
| `STILLAIR_LOCAL_URL` | Where Ollama is running | `http://127.0.0.1:11434/v1` |
| `STILLAIR_RPM`, `STILLAIR_TPM` | Requests and tokens per minute | 25, 6000 |

---

## The research behind it

Before the chat assistant, this project tested the idea in a controlled simulation. A simulated model stands in for a real one, with error rates we set ourselves. The worlds are made of invented facts, split so that answering needs two facts learned at different times. These numbers test the mechanism, and say nothing about how real models behave.

```bash
python -m stillair demo --fast          # narrated run in the terminal
python -m stillair serve                # the original simulation dashboard
python -m stillair eval --seeds 20      # the full experiment, about 2 minutes
python -m unittest discover -s tests    # all tests
```

Three setups were compared, using the same amount of thinking for B and C, over 20 random worlds:

| | A: no idle thinking | B: random thinking, no filter | C: Still Air |
|---|---|---|---|
| Accuracy with 1 lookup | 0% | 40% | **51%** |
| Accuracy with 2 lookups | **100%** | 83% | 92% |
| Stored links that are true | n/a | 7% | **73%** |

**What this showed:**

- Idle thinking lets the agent answer questions it otherwise could not, when it only gets one lookup.
- The filter is essential. Without it, 93% of stored links were junk.
- With two lookups, no idle thinking is still best, because a few false links get through.
- Extra upgrades closed that gap in the simulator. On 20 fresh worlds, checking a link before using it, skipping pointless thinking, adaptive noise and expiring unconfirmed links together reached 100% with two lookups and 60% with one, with 93% of stored links true. Checking does cost more per question (85 tokens against 70), and the simulator's checker is more forgiving than a real one would be.

Full tables and charts are in [`results/`](results/results.md).

**Papers that shaped the design:**

- [Sleep-time Compute](https://arxiv.org/abs/2504.13171): thinking ahead while idle.
- [Synapse](https://arxiv.org/abs/2601.02744): spreading activation over memory.
- [Hippo](https://arxiv.org/abs/2610.24453): replaying surprises, and adding or deleting memories explicitly.
- [Human-Inspired Memory Architecture](https://arxiv.org/html/2605.08538v1): deduplicate memories, do not summarise them.
- [LightMem](https://arxiv.org/abs/2604.07798): pairing small models with a larger one.
- [HaluMem](https://arxiv.org/abs/2511.03506): false memories start at the moment of writing.

---

## What is in this folder

```text
stillair/
  assistant/    the chat assistant: conscious, subconscious, memory links, reminders, server
  core/         memory, spreading activation, the original idle loop, task-mode agent
  backends/     connections to models: simulator, OpenAI-compatible APIs, Claude
  world/        the invented facts used by the experiment
  evaluation/   the experiment and its reports
  live/         the original simulation dashboard
  cli/          the commands you run
web/            chat.html, backend.html, and the simulation dashboard (index.html)
tests/          automated tests
scripts/        makes the charts (needs matplotlib)
results/        experiment results and charts
```
