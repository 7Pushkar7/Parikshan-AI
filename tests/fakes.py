"""Test doubles: a scripted fake Gemini, a fake embedding function and sample API payloads.
They let the full agent loop run offline, with no API key and no network."""
from __future__ import annotations

import hashlib
import math
from typing import Any

from chromadb import Documents, EmbeddingFunction, Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda


class HashEmbedding(EmbeddingFunction):
    """Bag-of-words hashing into 64 dims. Good enough for retrieval tests."""

    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002
        out = []
        for text in input:
            v = [0.0] * 64
            for w in text.lower().split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
            n = math.sqrt(sum(x * x for x in v)) or 1
            out.append([x / n for x in v])
        return out

    @staticmethod
    def name() -> str:
        return "hash-test"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config):
        return HashEmbedding()


REPORT_MD = """# Semaglutide 2.4 mg in Obesity: Efficacy and Safety Evidence Review

## Executive summary
- **Efficacy:** Once-weekly semaglutide 2.4 mg produced a mean body-weight change of -14.9% vs -2.4% with placebo at week 68 in adults without diabetes [PMID:33567185, NCT03548935].
- **Safety:** Gastrointestinal events (nausea, diarrhoea, vomiting) were the most common, mostly mild-to-moderate and transient; discontinuation due to GI events was 4.5% vs 0.8% [PMID:33567185].
- **Key warning:** The FDA label carries a boxed warning for thyroid C-cell tumours seen in rodents; contraindicated with MTC or MEN 2 [LABEL:semaglutide].
- **Conclusion:** Benefit is consistent across STEP trials; GI tolerability and gallbladder events need monitoring [PMID:33667417, PMID:36216945, NCT99999999].

## Trial evidence
The STEP programme tested semaglutide 2.4 mg across populations with overweight or obesity, with and without type 2 diabetes, over 68 to 104 weeks [PMID:33567185, PMID:33667417].

| Trial | Phase | N | Population | Key result | Source |
|---|---|---|---|---|---|
| STEP 1 | 3 | 1,961 | Adults with BMI >= 30, no diabetes | -14.9% vs -2.4% weight at week 68 | [NCT03548935] |
| STEP 2 | 3 | 1,210 | Overweight/obesity with type 2 diabetes | -9.6% vs -3.4%; treatment difference -6.2 points | [PMID:33667417] |
| STEP 5 | 3 | 304 | Adults without diabetes, 104 weeks | Sustained weight loss over two years | [PMID:36216945] |
| SELECT | 3 | 17,604 | Age >= 45, established CVD, no diabetes | Cardiovascular outcomes vs placebo | [PMID:37952131] |

## Safety profile
### In clinical trials
- **GI events:** Nausea was the most frequent adverse event, reported by about 44% of participants on semaglutide [NCT03548935].
- **Gallbladder:** Cholelithiasis occurred more often with semaglutide than placebo [NCT03548935].
### Real-world reports (FAERS and FDA label)
- **Report volume:** FAERS holds 73,001 reports mentioning semaglutide; counts reflect reporting, not incidence [FAERS:semaglutide].
- **Top reactions:** Nausea, vomiting and off-label use are among the most reported terms [FAERS:semaglutide].

## Signals to watch
- **Pancreatitis:** Listed as a warning on the label [LABEL:semaglutide].
- **Delayed gastric emptying:** Appears in post-marketing reports [FAERS:semaglutide].

## Evidence gaps
- **Long-term data:** Few trials beyond 2 years in the fetched evidence.
- **Head-to-head:** No direct comparison with tirzepatide was retrieved.
"""


class FakeGemini(BaseChatModel):
    """Behaves like a tool-calling LLM following a fixed research plan."""

    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-gemini"

    def bind_tools(self, tools: Any, **kwargs: Any):
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any):
        def _parse(_msg):
            return schema.model_validate({"sources": [{
                "source_id": "NCT03548935", "drugs": ["semaglutide"], "dosages": ["2.4 mg once weekly"],
                "population": "adults with obesity", "outcomes": [{"measure": "weight change", "result": "-14.9%"}],
                "adverse_events": ["nausea"], "safety_signals": ["gallbladder events"]}]})
        return self | RunnableLambda(_parse)  # real LLM call first, so usage is tracked

    def _reply(self, messages: list) -> AIMessage:
        system = next((m for m in messages if isinstance(m, SystemMessage)), None)
        sys_text = system.content if system else ""
        if "pharmacovigilance and clinical evidence analyst" in sys_text:
            return AIMessage(content=REPORT_MD)
        if sys_text.startswith("You are a biomedical information extraction"):
            return AIMessage(content='{"sources": []}')
        if "Parikshan AI" not in sys_text:
            return AIMessage(content="Semaglutide is a GLP-1 receptor agonist.")
        last_human = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
        done = [m.name for m in messages[last_human:] if isinstance(m, ToolMessage)]
        if not done:
            return AIMessage(content="I need trial, literature and FDA safety data first.", tool_calls=[
                {"name": "search_clinical_trials", "args": {"drug": "semaglutide", "condition": "obesity"}, "id": "c1"},
                {"name": "search_pubmed", "args": {"query": "semaglutide obesity trial"}, "id": "c2"},
                {"name": "get_drug_safety_data", "args": {"drug": "semaglutide"}, "id": "c3"},
            ])
        if "get_trial_details" not in done:
            return AIMessage(content="Checking posted adverse events for the main trial.", tool_calls=[
                {"name": "get_trial_details", "args": {"nct_id": "NCT03548935"}, "id": "c4"},
                {"name": "extract_entities", "args": {"source_ids": ["NCT03548935"]}, "id": "c5"}])
        if "write_report" not in done:
            return AIMessage(content="Enough evidence. Writing the cited report.", tool_calls=[
                {"name": "write_report", "args": {"request": "semaglutide efficacy and safety review"}, "id": "c6"}])
        return AIMessage(content="- **Efficacy:** semaglutide 2.4 mg gave -14.9% body weight vs -2.4% with placebo "
                                 "at week 68 [NCT03548935, PMID:33567185].\n- **Safety:** nausea is the most reported "
                                 "reaction in FAERS [FAERS:semaglutide]; the label carries a thyroid C-cell tumour boxed "
                                 "warning [LABEL:semaglutide].\n\nThe full cited report is ready to open or download.")

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls += 1
        msg = self._reply(messages)
        n_in = sum(len(str(m.content)) for m in messages) // 4
        n_out = max(1, len(str(msg.content)) // 4)
        msg.usage_metadata = {"input_tokens": n_in, "output_tokens": n_out, "total_tokens": n_in + n_out}
        return ChatResult(generations=[ChatGeneration(message=msg)])


# ---------------- sample API payloads (shapes match the real APIs) ----------------

ESEARCH = {"esearchresult": {"count": "1234", "idlist": ["33567185"]}}
EFETCH_XML = """<?xml version="1.0"?><PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>33567185</PMID>
<Article><Journal><JournalIssue><PubDate><Year>2021</Year></PubDate></JournalIssue>
<Title>The New England journal of medicine</Title></Journal>
<ArticleTitle>Once-Weekly Semaglutide in Adults with Overweight or Obesity.</ArticleTitle>
<Abstract><AbstractText Label="BACKGROUND">Obesity is a global health challenge.</AbstractText>
<AbstractText Label="RESULTS">Mean change in body weight was -14.9% with semaglutide vs -2.4% with placebo.
Gastrointestinal disorders, typically nausea and diarrhea, were more frequent with semaglutide.</AbstractText></Abstract>
<AuthorList><Author><LastName>Wilding</LastName><Initials>JPH</Initials></Author></AuthorList>
<PublicationTypeList><PublicationType>Randomized Controlled Trial</PublicationType></PublicationTypeList>
</Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"""

STUDY = {
    "protocolSection": {
        "identificationModule": {"nctId": "NCT03548935", "briefTitle": "STEP 1: Semaglutide 2.4 mg in Obesity"},
        "statusModule": {"overallStatus": "COMPLETED", "startDateStruct": {"date": "2018-06"},
                         "completionDateStruct": {"date": "2020-03"}},
        "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Novo Nordisk A/S"}},
        "conditionsModule": {"conditions": ["Obesity"]},
        "designModule": {"phases": ["PHASE3"], "studyType": "INTERVENTIONAL", "enrollmentInfo": {"count": 1961}},
        "armsInterventionsModule": {"interventions": [{"type": "DRUG", "name": "Semaglutide"}]},
        "outcomesModule": {"primaryOutcomes": [{"measure": "Change in body weight", "timeFrame": "Week 68"}]},
        "eligibilityModule": {"eligibilityCriteria": "Inclusion: BMI >= 30", "sex": "ALL", "minimumAge": "18 Years"},
        "descriptionModule": {"briefSummary": "Tests semaglutide for weight management."},
    },
    "hasResults": True,
    "resultsSection": {"adverseEventsModule": {
        "eventGroups": [{"id": "EG000", "title": "Semaglutide", "seriousNumAffected": 128, "seriousNumAtRisk": 1306}],
        "seriousEvents": [{"term": "Cholelithiasis", "stats": [{"groupId": "EG000", "numAffected": 8, "numAtRisk": 1306}]}],
        "otherEvents": [{"term": "Nausea", "stats": [{"groupId": "EG000", "numAffected": 577, "numAtRisk": 1306}]}],
    }},
}
CT_SEARCH = {"studies": [STUDY], "totalCount": 42}
FDA_TOTAL = {"meta": {"results": {"total": 25000}}, "results": [{}]}
FDA_SERIOUS = {"meta": {"results": {"total": 9000}}, "results": [{}]}
FDA_COUNT = {"results": [{"term": "NAUSEA", "count": 4000}, {"term": "VOMITING", "count": 2500}]}
FDA_LABEL = {"results": [{"openfda": {"brand_name": ["WEGOVY"], "generic_name": ["SEMAGLUTIDE"]},
                          "boxed_warning": ["WARNING: RISK OF THYROID C-CELL TUMORS"],
                          "warnings_and_cautions": ["Acute pancreatitis. Acute gallbladder disease."],
                          "adverse_reactions": ["Nausea, diarrhea, vomiting"], "set_id": "abc"}]}
