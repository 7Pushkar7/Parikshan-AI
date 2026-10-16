import os
import tempfile

import pytest
import respx
from httpx import Response

_tmp = tempfile.mkdtemp(prefix="parikshan-test-")
os.environ["DATA_DIR"] = _tmp
os.environ["GEMINI_API_KEY"] = ""
os.environ["GEMINI_MODEL"] = "gemini-2.5-flash"

from app.config import get_settings  # noqa: E402
from app.llm import set_model_override  # noqa: E402
from app.rag import vector_store  # noqa: E402
from tests import fakes  # noqa: E402

get_settings.cache_clear()


@pytest.fixture(autouse=True)
def offline_world():
    set_model_override(fakes.FakeGemini())
    vector_store.set_embedding_override(fakes.HashEmbedding())
    with respx.mock(assert_all_called=False) as mock:
        mock.get(url__startswith="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch").mock(
            return_value=Response(200, json=fakes.ESEARCH))
        mock.get(url__startswith="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch").mock(
            return_value=Response(200, text=fakes.EFETCH_XML))
        mock.get(url__startswith="https://clinicaltrials.gov/api/v2/studies/NCT").mock(
            return_value=Response(200, json=fakes.STUDY))
        mock.get(url__startswith="https://clinicaltrials.gov/api/v2/studies").mock(
            return_value=Response(200, json=fakes.CT_SEARCH))
        mock.get(url__regex=r"https://api\.fda\.gov/drug/event\.json.*count=").mock(
            return_value=Response(200, json=fakes.FDA_COUNT))
        mock.get(url__regex=r"https://api\.fda\.gov/drug/event\.json.*serious").mock(
            return_value=Response(200, json=fakes.FDA_SERIOUS))
        mock.get(url__regex=r"https://api\.fda\.gov/drug/event\.json.*").mock(
            return_value=Response(200, json=fakes.FDA_TOTAL))
        mock.get(url__startswith="https://api.fda.gov/drug/label.json").mock(
            return_value=Response(200, json=fakes.FDA_LABEL))
        yield mock
    set_model_override(None)
