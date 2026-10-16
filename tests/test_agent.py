import json

import pytest

from app.agent.graph import trim_history
from app.agent.service import AgentService
from app.clients import clinical_trials, openfda, pubmed
from app.cost import cost_usd, price_for
from app.tools.analysis import verify_citations
from langchain_core.messages import AIMessage, HumanMessage


# ---------------- API clients parse real-shaped payloads ----------------

async def test_pubmed_parsing():
    res = await pubmed.search("semaglutide", 1)
    a = res["articles"][0]
    assert a["pmid"] == "33567185" and "Semaglutide" in a["title"]
    assert "RESULTS:" in a["abstract"] and a["year"] == "2021"


async def test_clinical_trials_parsing():
    res = await clinical_trials.search("semaglutide", "obesity")
    t = res["trials"][0]
    assert t["nct_id"] == "NCT03548935" and t["phases"] == ["PHASE3"] and t["enrollment"] == 1961
    d = await clinical_trials.get_study("NCT03548935")
    assert d["adverse_events"]["other"][0]["term"] == "Nausea"
    assert d["adverse_events"]["other"][0]["pct"] == pytest.approx(44.2, 0.1)


async def test_openfda_parsing():
    ae = await openfda.adverse_events("semaglutide")
    assert ae["total_reports"] == 25000 and ae["top_reactions"][0]["reaction"] == "Nausea"
    lbl = await openfda.label("semaglutide")
    assert "THYROID" in lbl["boxed_warning"]


# ---------------- cost math ----------------

def test_cost_formula():
    assert price_for("gemini-2.5-flash") == (0.30, 2.50)
    assert price_for("gemini-2.5-flash-lite") == (0.10, 0.40)  # longest prefix wins
    # 1M input + 1M output on 2.5 Flash = $0.30 + $2.50
    assert cost_usd("gemini-2.5-flash", 1_000_000, 1_000_000) == pytest.approx(2.80)


# ---------------- citation checker catches hallucinated IDs ----------------

def test_verify_citations():
    text = "Works [NCT03548935]. Safe-ish [PMID:1; FAERS:Semaglutide]. Fake [NCT99999999]."
    ok, bad = verify_citations(text, {"NCT03548935", "PMID:1", "FAERS:semaglutide"})
    assert ok == ["NCT03548935", "PMID:1", "FAERS:semaglutide"] and bad == ["NCT99999999"]


def test_trim_history_keeps_current_turn():
    msgs = [HumanMessage("q1"), AIMessage("a1"), HumanMessage("q2"), AIMessage("a2"), HumanMessage("q3")]
    assert trim_history(msgs, 3) == msgs[2:]
    assert trim_history(msgs, 1) == msgs[4:]


# ---------------- full agent loop, offline ----------------

async def test_full_agent_run(tmp_path):
    async with AgentService.create(memory_path=tmp_path / "mem.sqlite") as svc:
        out = await svc.ask("Review efficacy and safety of semaglutide for obesity", "t-1")
        assert out["status"] == "completed", out
        assert out["tools_used"][:3] == ["search_clinical_trials", "search_pubmed", "get_drug_safety_data"]
        assert "write_report" in out["tools_used"] and not out["tool_errors"]
        assert out["usage"]["llm_calls"] == 6  # 4 agent steps + extraction + report
        purposes = {c["purpose"] for c in out["usage"]["calls"]}
        assert purposes == {"agent_reasoning", "entity_extraction", "report_generation"}
        assert out["usage"]["equivalent_cost_usd"] > 0 and out["usage"]["billed_cost_usd"] == 0
        rep = out["report"]
        assert rep and "NCT99999999" in rep["unverified_citations"]
        assert {c["id"] for c in rep["citations"]} >= {"NCT03548935", "PMID:33567185", "FAERS:semaglutide"}
        thoughts = [e for e in out["trace"] if e["type"] == "thought"]
        assert thoughts and "trial" in thoughts[0]["text"]

        # memory: second question in same thread sees the first
        out2 = await svc.ask("And what about long-term safety?", "t-1")
        hist = await svc.history("t-1")
        assert [h["role"] for h in hist][:2] == ["user", "assistant"] and len(hist) == 4
        assert out2["status"] == "completed"

        await svc.forget("t-1")
        assert await svc.history("t-1") == []


async def test_tool_result_is_compact_json():
    from app.tools import search_pubmed
    out = await search_pubmed.ainvoke({"query": "semaglutide"}, {"configurable": {"thread_id": "x"}})
    data = json.loads(out)
    assert data["articles"][0]["cite_as"] == "PMID:33567185"


async def test_fallback_model_used_when_primary_overloaded(monkeypatch, tmp_path):
    """Main model returns 503 -> agent switches to GEMINI_FALLBACK_MODEL and still finishes."""
    from app import llm
    from app.config import get_settings
    from tests.fakes import FakeGemini

    class Overloaded(FakeGemini):
        def _generate(self, *a, **k):
            raise RuntimeError("503 UNAVAILABLE: model is experiencing high demand")

    backup = FakeGemini()
    monkeypatch.setattr(llm, "_override", None)
    monkeypatch.setattr(llm, "get_chat_model", lambda name=None: backup if name else Overloaded())
    monkeypatch.setattr(get_settings(), "gemini_fallback_model", "backup-model")
    async with AgentService.create(memory_path=tmp_path / "mem.sqlite") as svc:
        out = await svc.ask("Review semaglutide safety and write a report", "fb-1")
    assert out["status"] == "completed" and out["report"] and backup.calls >= 4
