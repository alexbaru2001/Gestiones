import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.infrastructure.portfolio_review_repository import JsonPortfolioReviewRepository


@pytest.fixture(autouse=True)
def isolated_review(tmp_path, monkeypatch):
    repository = JsonPortfolioReviewRepository(tmp_path / "portfolio_review.json")
    monkeypatch.setattr(main, "portfolio_review_repository", repository)
    return repository


def analysis(ticker="KO", score=82.0, flags=None, rpd=3.1, avg5=3.3):
    return {
        "ticker": ticker,
        "name": "Coca-Cola",
        "score": score,
        "price": 60.0,
        "sector": "Consumer Defensive",
        "recommendation": "Comprar",
        "flags": flags or [],
        "exchange": {"currency": "USD"},
        "metrics": {"rpd_ttm": rpd, "rpd_avg5": avg5, "payout": 70.0},
    }


def position(weight=12.0, value=1200.0, name="Coca-Cola Co"):
    return {"weight": weight, "value": value, "broker": "Trade Republic", "name": name}


def test_review_stores_the_weight_from_the_portfolio_not_from_the_analysis(isolated_review):
    result = isolated_review.record("KO", analysis(), position())

    entry = result["positions"][0]
    assert entry["weight"] == 12.0
    assert entry["value"] == 1200.0
    assert entry["portfolio_name"] == "Coca-Cola Co"
    assert result["last_review"] is not None


def test_worst_scored_positions_come_first_because_that_is_what_needs_reviewing(isolated_review):
    isolated_review.record("KO", analysis(ticker="KO", score=82.0), position())
    isolated_review.record("MC.PA", analysis(ticker="MC.PA", score=48.0), position(value=7.3))

    assert [entry["ticker"] for entry in isolated_review.load()["positions"]] == ["MC.PA", "KO"]


def test_each_review_adds_a_dated_point_with_the_flag_count(isolated_review):
    isolated_review.record("KO", analysis(score=82.0), position())
    result = isolated_review.record("KO", analysis(score=74.0, flags=["payout alto"]), position())

    history = result["positions"][0]["history"]
    assert len(history) == 1  # misma fecha: la toma del día se actualiza, no se duplica
    assert history[0]["score"] == 74.0
    assert history[0]["flags"] == 1
    assert history[0]["rpd_avg5"] == 3.3


def test_sold_positions_stop_being_reviewed(isolated_review):
    isolated_review.record("KO", analysis(ticker="KO"), position())
    isolated_review.record("MC.PA", analysis(ticker="MC.PA"), position())

    result = isolated_review.forget_missing(["KO"])

    assert [entry["ticker"] for entry in result["positions"]] == ["KO"]


def test_review_survives_a_restart_because_it_lives_on_disk(isolated_review):
    isolated_review.record("KO", analysis(), position())

    reloaded = JsonPortfolioReviewRepository(isolated_review.path).load()

    assert reloaded["positions"][0]["ticker"] == "KO"
    assert reloaded["positions"][0]["history"][0]["score"] == 82.0


def test_endpoint_records_a_position_and_lists_it():
    client = TestClient(main.app)

    assert client.get("/api/v1/investments/portfolio-review").json()["result"]["positions"] == []

    response = client.post(
        "/api/v1/investments/portfolio-review",
        json={"ticker": "ko", "analysis": analysis(), "position": position()},
    )

    assert response.status_code == 200
    assert response.json()["result"]["positions"][0]["ticker"] == "KO"


def test_endpoint_requires_the_analysis_payload():
    client = TestClient(main.app)

    response = client.post("/api/v1/investments/portfolio-review", json={"ticker": "KO"})

    assert response.status_code == 400
    assert "análisis" in response.json()["detail"]


def test_prune_endpoint_requires_a_ticker_list():
    client = TestClient(main.app)

    assert client.post("/api/v1/investments/portfolio-review/prune", json={}).status_code == 400


def test_corrupted_review_file_reports_a_clear_error(isolated_review):
    isolated_review.path.write_text("{roto", encoding="utf-8")
    client = TestClient(main.app)

    response = client.get("/api/v1/investments/portfolio-review")

    assert response.status_code == 500
    assert "revisión de cartera" in response.json()["detail"]
