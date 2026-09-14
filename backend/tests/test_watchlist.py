import json

import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.infrastructure.watchlist_repository import JsonWatchlistRepository


@pytest.fixture(autouse=True)
def isolated_watchlist(tmp_path, monkeypatch):
    """La imagen de test lleva datos reales dentro; el seguimiento se escribe en un tmp_path."""
    repository = JsonWatchlistRepository(tmp_path / "watchlist.json")
    monkeypatch.setattr(main, "watchlist_repository", repository)
    return repository


def analysis(ticker="KO", score=82.0, rpd=3.1):
    return {
        "ticker": ticker,
        "name": "Coca-Cola",
        "score": score,
        "price": 60.0,
        "sector": "Consumer Defensive",
        "recommendation": "Comprar",
        "flags": [],
        "exchange": {"currency": "USD"},
        "metrics": {"rpd_ttm": rpd, "payout": 70.0},
    }


def test_added_ticker_survives_with_its_score(isolated_watchlist):
    isolated_watchlist.add("ko", analysis())

    entries = JsonWatchlistRepository(isolated_watchlist.path).load()

    assert entries[0]["ticker"] == "KO"
    assert entries[0]["name"] == "Coca-Cola"
    assert entries[0]["last_score"] == 82.0
    assert entries[0]["history"][0]["rpd_ttm"] == 3.1


def test_analyzing_twice_the_same_day_keeps_one_point_but_updates_it(isolated_watchlist):
    isolated_watchlist.add("KO", analysis(score=82.0))
    isolated_watchlist.record_analysis(analysis(score=79.0))

    entry = isolated_watchlist.load()[0]

    assert len(entry["history"]) == 1
    assert entry["history"][0]["score"] == 79.0
    assert entry["last_score"] == 79.0


def test_analysis_of_an_untracked_ticker_is_not_stored(isolated_watchlist):
    isolated_watchlist.record_analysis(analysis(ticker="JNJ"))

    assert isolated_watchlist.load() == []


def test_watchlist_is_ordered_by_score_with_unscored_entries_last(isolated_watchlist):
    isolated_watchlist.add("KO", analysis(ticker="KO", score=60.0))
    isolated_watchlist.add("JNJ", analysis(ticker="JNJ", score=88.0))
    isolated_watchlist.add("AAA", None)

    assert [entry["ticker"] for entry in isolated_watchlist.load()] == ["JNJ", "KO", "AAA"]


def test_watchlist_endpoints_add_list_and_remove():
    client = TestClient(main.app)

    assert client.get("/api/v1/investments/watchlist").json()["result"] == []

    added = client.post("/api/v1/investments/watchlist", json={"ticker": "ko", "analysis": analysis()})
    assert added.status_code == 200
    assert added.json()["result"][0]["ticker"] == "KO"

    removed = client.delete("/api/v1/investments/watchlist/KO")
    assert removed.json()["result"] == []


def test_watchlist_endpoint_rejects_an_empty_ticker():
    client = TestClient(main.app)

    response = client.post("/api/v1/investments/watchlist", json={"ticker": "  "})

    assert response.status_code == 400


def test_analyzing_a_followed_ticker_records_its_score(monkeypatch, isolated_watchlist):
    isolated_watchlist.add("KO", None)
    monkeypatch.setattr(main, "analyze_ticker", lambda ticker, refresh=False: analysis())
    client = TestClient(main.app)

    client.get("/api/v1/investments/analyze?ticker=KO")

    assert isolated_watchlist.load()[0]["last_score"] == 82.0


def test_corrupted_watchlist_file_reports_a_clear_error(isolated_watchlist):
    isolated_watchlist.path.write_text("{no es json", encoding="utf-8")
    client = TestClient(main.app)

    response = client.get("/api/v1/investments/watchlist")

    assert response.status_code == 500
    assert "lista de seguimiento" in response.json()["detail"]


def test_legacy_file_with_unexpected_shape_is_ignored(isolated_watchlist):
    isolated_watchlist.path.write_text(json.dumps(["KO"]), encoding="utf-8")

    assert isolated_watchlist.load() == []
