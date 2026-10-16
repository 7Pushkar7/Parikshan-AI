# Demo guide: showing Parikshan AI to your mentor

Keep this open on your phone or a second screen during the demo.

## Before the demo (do this the day before)

```bash
git clone https://github.com/7Pushkar7/Parikshan-AI.git
cd Parikshan-AI
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then paste your key into GEMINI_API_KEY
python scripts/check_setup.py # must show 4 x OK
uvicorn app.main:app --reload
```

Open http://localhost:8000.

1. Run the first sample question once, so the local embedding model (about 80 MB) downloads before the demo.
2. Run 2 or 3 questions and keep those threads. If wifi or the free-tier limit fails during the demo, you can open an old thread and show the saved run and report.
3. Download one PDF report and keep it on your desktop as a backup.
4. Optional: run `python -m eval.run_eval --limit 5 --delay 20` and note the numbers.

## The 10-minute demo flow

### 1. The basics (1 min, terminal)

Run `python scripts/llm_call_demo.py "What is pharmacovigilance?"`

Say: *"This is one LLM call. One request, one response. It also shows the input tokens, output tokens and cost. No tools, no memory."*

### 2. An agent with no framework (1 min, terminal)

Run `python scripts/agent_from_scratch.py`

Say: *"This is an agent in 60 lines. The LLM asks for a tool, my code runs it, and the result goes back to the LLM. It loops until the LLM stops asking for tools. LangGraph does this same loop in the main app."*

### 3. LLM vs agent (2 min, "LLM vs agent" tab)

Click **Run both** with the pembrolizumab question.

Say: *"On the left, a plain LLM answers from memory. It has no sources and may be outdated. On the right, the agent called ClinicalTrials.gov and found real recruiting trials with NCT IDs. The agent costs more because it makes several LLM calls and resends the history every step."*

### 4. The full agent (4 min, "Research agent" tab)

Click the first sample: *"Review the efficacy and safety of semaglutide for obesity and write a report"*.

While it runs, point at:
- **Think** (purple): the agent's reasoning before each action.
- **Act** (blue): which tool it chose and the arguments it passed. Gemini decided this, not my code.
- **Observe** (green): what came back, how long it took, how many characters went back to the LLM.
- **Stage tracker** (top of the run): Plan, Trials, Literature, FDA safety, Extract, Report, Answer light up as the agent works.
- **Right panel**: total cost in $ and ₹, split into input and output, plus "Tokens per LLM call": one bar per call with its purpose (agent reasoning, entity extraction, report generation). Point out that the report call has the biggest output (amber) part.
- **Report**: click **Open report**. Every claim has a numbered citation linking to the reference list. Blue = trial or paper, amber = FDA data, red "?" = cited but never fetched (hallucination check). Click **Download PDF**.
- **Dark mode**: the moon/sun button at the top right.

### 5. Memory (1 min)

In the same thread, ask: *"Now compare it with liraglutide"*.

Say: *"I never said semaglutide again. The agent remembers because LangGraph saves the full state to SQLite after every step, keyed by the thread id."* Click another thread on the left to show older conversations.

### 6. Under the hood (1 min)

- Open http://localhost:8000/docs: *"FastAPI auto-generates this. Every endpoint can be tried here."*
- Open the **How it works** tab and walk through the loop diagram.
- Optional: show `app/agent/graph.py`. It has 2 nodes, the agent and the tools, plus one conditional edge.

## Questions your mentor may ask

| Question | Short answer |
|---|---|
| What is the difference between an LLM and an agent? | An LLM gives one input and one output. An agent is an LLM in a loop with tools and memory that decides its next step itself. |
| What is an LLM call? | One API request with the model name, messages and settings. The response has the text plus token usage. |
| Which framework? | LangGraph, LangChain's agent runtime. It uses the ReAct pattern: agent node, tools node, conditional edge. |
| Why not AgentExecutor? | It is legacy. LangGraph is the current recommended way and has memory, streaming and step limits built in. |
| Which LLM, and why? | Gemini 3.8 Flash. It has a free tier, strong tool calling for agents, is fast, and costs $0.75 input / $3.75 output per 1M tokens on the paid tier (until 31 Dec 2026). |
| Input vs output cost? | Input is everything sent (prompt, history, tool results). Output is what the model writes, including thinking. Output is about 5 times pricier. A full report run costs about $0.02 to $0.04 on the paid tier, and $0 on the free tier. |
| Where is memory stored? | Conversation: SQLite (`data/memory.sqlite`, LangGraph checkpointer). Evidence: ChromaDB (`data/chroma`). Logs and cost: SQLite (`data/app.sqlite`). |
| Which database? | SQLite for memory and logs. ChromaDB as the vector database for RAG. |
| External APIs? | PubMed E-utilities, ClinicalTrials.gov API v2, openFDA (FAERS + labels). All free, no key needed. |
| What is FastAPI used for? | It is the web server. It exposes the agent as HTTP endpoints, validates input with Pydantic, streams steps to the UI with Server-Sent Events, serves the UI, and gives docs at /docs. |
| How do you stop hallucination? | The agent uses live data, RAG over fetched evidence only, and mandatory citations. Any cited ID that was never fetched is flagged automatically. |
| How do you stop infinite loops? | A step limit (MAX_AGENT_STEPS), a prompt that says when to stop, and tool errors returned as messages instead of crashing. |
| How did you test it? | 10 offline tests with a fake LLM and mocked APIs, plus an evaluation script over 16 drug queries. |

## If something breaks during the demo

| Problem | Fix |
|---|---|
| "Gemini rate limit reached" or 503 "high demand" | The app already switches to the backup model by itself. If both fail, wait 1 to 2 minutes and run again. Point it out as resilience: "if the main model is busy, it falls back to a lighter model". |
| Banner says agent is offline | `GEMINI_API_KEY` is missing in `.env`. Add it and restart uvicorn. |
| A tool shows red (error) | That API was slow or down. The agent retries or continues with other sources. Point this out as error handling. |
| No internet | Open an old thread from the left panel, or show your saved PDF. |
| Port already in use | `uvicorn app.main:app --port 8001` |
