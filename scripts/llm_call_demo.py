"""Learning script 1: what ONE LLM call looks like, and what it costs.

    python scripts/llm_call_demo.py "What is pharmacovigilance?"

No framework, just Google's official SDK (google-genai). This is the smallest unit everything
else is built from: request in -> text + token counts out.
"""
import os
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
PRICE_IN, PRICE_OUT = 0.75, 3.75  # USD per 1M tokens, Gemini 3.8 Flash paid tier (until 31 Dec 2026)

question = " ".join(sys.argv[1:]) or "What is pharmacovigilance? Answer in 2 lines."
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# ---- the LLM call ----
response = client.models.generate_content(
    model=MODEL,
    contents=question,                                   # the user message
    config=types.GenerateContentConfig(
        system_instruction="You are a concise pharma research assistant.",
        max_output_tokens=400,
    ),
)

u = response.usage_metadata
inp = u.prompt_token_count or 0
out = (u.candidates_token_count or 0) + (u.thoughts_token_count or 0)  # thinking is billed as output
cost = inp / 1e6 * PRICE_IN + out / 1e6 * PRICE_OUT

print("ANSWER:\n", response.text)
print(f"\nInput tokens : {inp}")
print(f"Output tokens: {out}  (of which thinking: {u.thoughts_token_count or 0})")
print(f"Cost on paid tier: ${cost:.6f}  (free tier: $0)")
