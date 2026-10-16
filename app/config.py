"""All settings in one place. Values come from environment variables or a .env file.

Example: GEMINI_API_KEY=... in .env  ->  settings.gemini_api_key
"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    # ---- LLM (Gemini) ----
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    # Backup model used automatically when the main one is overloaded (503) or rate-limited (429).
    gemini_fallback_model: str = "gemini-3.1-flash-lite"
    temperature: float = 0.1
    # Optional: cap "thinking" tokens on Gemini 2.5 models (thinking tokens are billed as output).
    thinking_budget: int | None = None
    # "free" = AI Studio free tier (you are billed $0, cost shown is the paid-tier equivalent)
    # "paid" = pay-as-you-go
    gemini_tier: str = "free"
    # Optional manual price override (USD per 1M tokens). Leave empty to use app/cost.py table.
    price_input_per_m: float | None = None
    price_output_per_m: float | None = None
    usd_to_inr: float = 84.0

    # ---- Agent ----
    max_agent_steps: int = 12          # max Think->Act->Observe loops per question
    history_messages_kept: int = 30    # short-term memory window sent to the LLM

    # ---- External APIs ----
    ncbi_api_key: str = ""             # optional, raises PubMed limit from 3 to 10 req/s
    ncbi_email: str = "parikshan-ai@example.com"
    http_timeout: float = 20.0

    # ---- Storage ----
    data_dir: Path = BASE_DIR / "data"
    # "local" = free on-device embeddings (Chroma's built-in MiniLM model)
    # "gemini" = Gemini embedding API
    embeddings: str = "local"
    gemini_embedding_model: str = "gemini-embedding-001"
    rag_top_k: int = 12

    @property
    def memory_db(self) -> Path:
        return self.data_dir / "memory.sqlite"

    @property
    def app_db(self) -> Path:
        return self.data_dir / "app.sqlite"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    for d in (s.data_dir, s.chroma_dir, s.reports_dir):
        d.mkdir(parents=True, exist_ok=True)
    return s
