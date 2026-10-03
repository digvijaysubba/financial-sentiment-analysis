from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_companies():
    rows = client.get("/companies").json()
    assert {r["company"] for r in rows} >= {"AAPL", "MSFT"}
    assert all(r["n_transcripts"] > 0 for r in rows)


def test_transcripts_filter_and_detail():
    rows = client.get("/transcripts", params={"company": "aapl"}).json()
    assert rows and all(r["company"] == "AAPL" for r in rows)

    detail = client.get(f"/transcripts/{rows[0]['id']}").json()
    assert "prepared_text" not in detail
    assert {s["model_name"] for s in detail["sentiment"]} == {"vader", "textblob"}

    with_text = client.get(f"/transcripts/{rows[0]['id']}", params={"include_text": True}).json()
    assert with_text["prepared_text"]


def test_transcript_404():
    assert client.get("/transcripts/999999").status_code == 404


def test_sentiment_filters():
    rows = client.get("/sentiment", params={"model": "vader", "section": "qa"}).json()
    assert rows and all(r["model_name"] == "vader" and r["section"] == "qa" for r in rows)


def test_sentiment_rejects_bad_model():
    assert client.get("/sentiment", params={"model": "bert"}).status_code == 422


def test_results_and_summary():
    rows = client.get("/results", params={"company": "NVDA"}).json()
    assert rows and {"vader_shift", "ret_1d"} <= rows[0].keys()
    assert client.get("/results", params={"company": "ZZZZ"}).status_code == 404

    summary = client.get("/summary").json()
    assert {"company", "n_calls", "vader_shift"} <= summary[0].keys()


def test_correlations():
    body = client.get("/correlations", params={"model": "textblob"}).json()
    assert set(body["correlations"]) == {"gap_ret", "ret_1d", "ret_3d", "ret_5d"}


def test_analyze():
    pos = client.post("/analyze", json={"text": "Revenue grew strongly and we are very pleased with record results."}).json()
    neg = client.post("/analyze", json={"text": "Demand weakened sharply and we saw disappointing, terrible losses this quarter."}).json()
    assert pos["vader"] > neg["vader"]
    assert pos["textblob"] > neg["textblob"]
    assert client.post("/analyze", json={"text": ""}).status_code == 422
