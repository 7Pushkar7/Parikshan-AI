"""The agent's toolbox. Gemini sees each tool's name + docstring + argument schema and decides
which one to call (this is called "function calling" or "tool calling")."""
from app.tools.analysis import extract_entities, write_report
from app.tools.research import (get_drug_safety_data, get_trial_details, search_clinical_trials,
                                search_pubmed)

ALL_TOOLS = [
    search_clinical_trials,
    get_trial_details,
    search_pubmed,
    get_drug_safety_data,
    extract_entities,
    write_report,
]

TOOL_LABELS = {
    "search_clinical_trials": "ClinicalTrials.gov search",
    "get_trial_details": "Trial details + adverse events",
    "search_pubmed": "PubMed literature search",
    "get_drug_safety_data": "FDA safety data (FAERS + label)",
    "extract_entities": "Entity extraction (NLP)",
    "write_report": "RAG report writer",
}
