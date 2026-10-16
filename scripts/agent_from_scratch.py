"""Learning script 2: an agent in ~60 lines, with NO framework.

    python scripts/agent_from_scratch.py "How many FAERS reports mention semaglutide?"

This is exactly what LangGraph does for us in app/agent/graph.py, written out by hand:
  loop:
    1. call the LLM with the conversation + tool descriptions      (THINK)
    2. if it asked for tools, run them in Python                    (ACT)
    3. append the results to the conversation                       (OBSERVE)
    4. if it did not ask for tools, that's the final answer -> stop
"""
import json
import os
import sys

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


# ---- tools: plain Python functions. The docstring tells Gemini when to use them. ----
def count_faers_reports(drug: str) -> dict:
    """Return how many FDA FAERS adverse event reports mention a drug (generic name)."""
    r = httpx.get("https://api.fda.gov/drug/event.json",
                  params={"search": f'patient.drug.openfda.generic_name:"{drug}"', "limit": 1}, timeout=20)
    return {"drug": drug, "reports": r.json().get("meta", {}).get("results", {}).get("total", 0)}


def count_trials(drug: str) -> dict:
    """Return how many trials of a drug are registered on ClinicalTrials.gov."""
    r = httpx.get("https://clinicaltrials.gov/api/v2/studies",
                  params={"query.intr": drug, "countTotal": "true", "pageSize": 1}, timeout=20)
    return {"drug": drug, "trials": r.json().get("totalCount", 0)}


TOOLS = {f.__name__: f for f in (count_faers_reports, count_trials)}

question = " ".join(sys.argv[1:]) or "Compare how many trials and FAERS reports exist for semaglutide vs liraglutide."
contents = [types.Content(role="user", parts=[types.Part(text=question)])]
config = types.GenerateContentConfig(
    tools=list(TOOLS.values()),
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),  # we run the loop ourselves
)

for step in range(1, 8):
    resp = client.models.generate_content(model=MODEL, contents=contents, config=config)   # THINK
    print(f"\n--- step {step}: LLM call ({resp.usage_metadata.prompt_token_count} in / "
          f"{resp.usage_metadata.candidates_token_count} out tokens)")
    if not resp.function_calls:
        print("FINAL ANSWER:\n", resp.text)
        break
    contents.append(resp.candidates[0].content)          # keep the model's tool request in the history
    for call in resp.function_calls:                      # ACT
        result = TOOLS[call.name](**call.args)
        print(f"tool {call.name}({json.dumps(call.args)}) -> {result}")
        contents.append(types.Content(role="user", parts=[  # OBSERVE
            types.Part.from_function_response(name=call.name, response={"result": result})]))
else:
    print("Stopped: step limit reached")
