import json

from fastapi.testclient import TestClient

from app.main import app


def test_api_end_to_end():
    with TestClient(app) as client:
        h = client.get("/api/health").json()
        assert h["status"] == "ok" and len(h["tools"]) == 6

        # streaming endpoint: parse Server-Sent Events
        events = []
        with client.stream("POST", "/api/chat/stream",
                           json={"message": "Review semaglutide safety and write a report"}) as r:
            assert r.status_code == 200
            etype = None
            for line in r.iter_lines():
                if line.startswith("event:"):
                    etype = line[6:].strip()
                elif line.startswith("data:"):
                    events.append((etype, json.loads(line[5:].strip())))
        types = [t for t, _ in events]
        assert types[0] == "run_started" and types[-1] == "run_finished"
        assert {"thought", "tool_call", "tool_result", "usage", "report", "final"} <= set(types)
        thread_id = events[0][1]["thread_id"]
        report_id = next(e["report_id"] for t, e in events if t == "report")

        assert client.get(f"/api/reports/{report_id}").json()["title"].startswith("Semaglutide")
        pdf = client.get(f"/api/reports/{report_id}/pdf")
        assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"

        assert client.get("/api/threads").json()[0]["thread_id"] == thread_id
        hist = client.get(f"/api/threads/{thread_id}").json()
        assert hist["messages"][0]["role"] == "user" and hist["runs"][0]["status"] == "completed"
        assert client.get("/api/stats").json()["runs"] >= 1

        cmp = client.post("/api/compare", json={"message": "What is semaglutide?"}).json()
        assert cmp["usage"]["llm_calls"] == 1 and cmp["answer"]

        assert client.delete(f"/api/threads/{thread_id}").status_code == 200
        assert client.get(f"/api/threads/{thread_id}").json()["messages"] == []
        assert client.get("/").status_code == 200


async def test_eval_harness_runs(capsys):
    from eval import run_eval
    await run_eval.main(limit=1, delay=0, only="q01")
    out = capsys.readouterr().out
    assert "SUMMARY" in out and '"task_completion": 1.0' in out
