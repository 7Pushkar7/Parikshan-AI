"""Developer tool: run the full app with the scripted fake LLM and canned API responses.
Lets you click through the UI with no API key and no network. NOT real data.

    python -m tests.offline_server     # then open http://localhost:8000
"""
import os
import tempfile
import time

os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="parikshan-offline-"))

import respx  # noqa: E402
import uvicorn  # noqa: E402
from httpx import Response  # noqa: E402

from app.llm import set_model_override  # noqa: E402
from app.rag import vector_store  # noqa: E402
from tests import fakes  # noqa: E402


class SlowFake(fakes.FakeGemini):
    def _generate(self, *a, **k):
        time.sleep(0.8)  # look like a real network call
        return super()._generate(*a, **k)


def main():
    set_model_override(SlowFake())
    vector_store.set_embedding_override(fakes.HashEmbedding())
    router = respx.mock(assert_all_called=False)
    router.route(host="127.0.0.1").pass_through()
    router.route(host="localhost").pass_through()
    router.get(url__startswith="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch").mock(return_value=Response(200, json=fakes.ESEARCH))
    router.get(url__startswith="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch").mock(return_value=Response(200, text=fakes.EFETCH_XML))
    router.get(url__startswith="https://clinicaltrials.gov/api/v2/studies/NCT").mock(return_value=Response(200, json=fakes.STUDY))
    router.get(url__startswith="https://clinicaltrials.gov/api/v2/studies").mock(return_value=Response(200, json=fakes.CT_SEARCH))
    router.get(url__regex=r"https://api\.fda\.gov/drug/event\.json.*count=").mock(return_value=Response(200, json=fakes.FDA_COUNT))
    router.get(url__regex=r"https://api\.fda\.gov/drug/event\.json.*serious").mock(return_value=Response(200, json=fakes.FDA_SERIOUS))
    router.get(url__regex=r"https://api\.fda\.gov/drug/event\.json.*").mock(return_value=Response(200, json=fakes.FDA_TOTAL))
    router.get(url__startswith="https://api.fda.gov/drug/label.json").mock(return_value=Response(200, json=fakes.FDA_LABEL))
    router.start()
    from app.main import app
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))


if __name__ == "__main__":
    main()
