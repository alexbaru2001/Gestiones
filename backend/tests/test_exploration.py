import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.domain import exploration, universes
from backend.infrastructure.exploration_repository import JsonExplorationRepository


@pytest.fixture(autouse=True)
def isolated_files(tmp_path, monkeypatch):
    monkeypatch.setattr(universes, "get_universes_path", lambda: tmp_path / "universes.json")
    monkeypatch.setattr(main, "exploration_repository", JsonExplorationRepository(tmp_path / "exploraciones.json"))
    return tmp_path


# --- Universos ---------------------------------------------------------------------------------

def test_indices_carry_their_exact_member_list():
    assert len(universes.get_members("sp500")) > 490
    assert "KO" in universes.get_members("sp500")
    assert all(ticker.endswith(".MC") for ticker in universes.get_members("ibex35"))


def test_msci_universes_are_flagged_as_approximate():
    catalog = {item["key"]: item for item in universes.list_universes()}

    assert catalog["msci_world"]["approximate"] is True
    assert catalog["sp500"]["approximate"] is False


def test_a_hand_corrected_list_wins_over_the_seed(isolated_files):
    universes.save_overrides({"ibex35": ["SAN.MC", "BBVA.MC"]})

    assert universes.get_members("ibex35") == ["SAN.MC", "BBVA.MC"]


def test_unknown_universe_is_rejected():
    with pytest.raises(ValueError, match="No conozco el universo"):
        universes.get_universe("nikkei")


def test_refresh_keeps_the_current_list_if_the_page_returns_almost_nothing(monkeypatch, isolated_files):
    """Si Wikipedia cambia de formato, es mejor quedarse con la lista buena que con tres tickers."""

    class FakeResponse:
        text = '<table id="constituents"><tr><td>KO</td></tr></table>'

        def raise_for_status(self):
            return None

    import requests

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: FakeResponse())

    with pytest.raises(RuntimeError, match="no se sustituye"):
        universes.refresh_members("sp500")


def test_only_indices_with_a_published_table_can_be_refreshed():
    with pytest.raises(ValueError, match="a mano"):
        universes.refresh_members("eurostoxx50")


# --- Criba -------------------------------------------------------------------------------------

def frame(prices, dividends):
    index = pd.to_datetime([f"{year}-06-30" for year in prices])
    return pd.DataFrame({"Close": list(prices.values()), "Dividends": [dividends.get(year, 0.0) for year in prices]}, index=index)


def test_screen_row_computes_yield_growth_and_streak():
    prices = {year: 100.0 for year in range(2015, 2027)}
    dividends = {year: 1.0 + (year - 2015) * 0.1 for year in range(2015, 2027)}

    row = exploration._screen_row("TEST", frame(prices, dividends))

    assert row["ticker"] == "TEST"
    assert row["rpd_ttm"] > 0
    assert row["streak_years"] >= 10
    assert row["dgr5"] > 0


def test_a_company_without_price_is_dropped_from_the_screen():
    empty = pd.DataFrame({"Close": [float("nan")], "Dividends": [0.0]}, index=pd.to_datetime(["2026-01-01"]))

    assert exploration._screen_row("TEST", empty) is None


def test_a_company_without_dividend_scores_zero_in_the_screen():
    assert exploration.prescore({"rpd_ttm": 0.0, "dgr5": 40.0, "streak_years": 0}) == 0.0


def test_an_absurd_yield_is_not_rewarded_as_if_it_were_good():
    """Una RPD del 15 % suele ser una trampa, no una oportunidad: no debe ganar a una sana."""
    trap = exploration.prescore({"rpd_ttm": 15.0, "dgr5": 0.0, "streak_years": 3, "streak_growth": 0})
    healthy = exploration.prescore({"rpd_ttm": 4.0, "dgr5": 8.0, "streak_years": 12, "streak_growth": 10})

    assert healthy > trap


def test_the_screen_rewards_trading_above_its_own_historical_yield():
    cheap = exploration.prescore({"rpd_ttm": 4.0, "rpd_avg5": 3.0, "dgr5": 5.0, "streak_years": 12, "streak_growth": 8})
    expensive = exploration.prescore({"rpd_ttm": 4.0, "rpd_avg5": 5.0, "dgr5": 5.0, "streak_years": 12, "streak_growth": 8})

    assert cheap > expensive


def test_ranking_keeps_only_the_requested_finalists():
    rows = [{"ticker": f"T{index}", "rpd_ttm": 3.0 + index, "dgr5": 5.0, "streak_years": 10} for index in range(10)]

    ranked = exploration.rank_candidates(rows, finalists=3)

    assert len(ranked) == 3
    assert ranked[0]["prescore"] >= ranked[1]["prescore"] >= ranked[2]["prescore"]


# --- Endpoints ---------------------------------------------------------------------------------

def test_candidates_endpoint_returns_the_index_members():
    client = TestClient(main.app)

    response = client.post("/api/v1/investments/explore/candidates", json={"universe": "ibex35"})

    assert response.status_code == 200
    assert len(response.json()["result"]["candidates"]) == 35


def test_candidates_endpoint_rejects_an_unknown_universe():
    client = TestClient(main.app)

    assert client.post("/api/v1/investments/explore/candidates", json={"universe": "nikkei"}).status_code == 400


def test_screen_endpoint_refuses_a_batch_bigger_than_the_chunk(monkeypatch):
    client = TestClient(main.app)

    response = client.post(
        "/api/v1/investments/explore/screen",
        json={"tickers": [f"T{index}" for index in range(exploration.DOWNLOAD_CHUNK + 1)]},
    )

    assert response.status_code == 400


def test_screen_endpoint_fills_the_prescore(monkeypatch):
    monkeypatch.setattr(
        main, "screen_chunk", lambda tickers: [{"ticker": "KO", "rpd_ttm": 3.0, "dgr5": 6.0, "streak_years": 12, "streak_growth": 10}]
    )
    client = TestClient(main.app)

    response = client.post("/api/v1/investments/explore/screen", json={"tickers": ["KO"]})

    assert response.json()["result"][0]["prescore"] > 0


def test_exploration_is_saved_and_read_back(isolated_files):
    client = TestClient(main.app)

    client.post(
        "/api/v1/investments/explore/save",
        json={"universe": "sp500", "label": "S&P 500", "screened": 503, "analyzed": 40, "results": [{"ticker": "KO", "score": 73.8}]},
    )

    saved = client.get("/api/v1/investments/universes").json()["result"]["explorations"]
    assert saved["sp500"]["results"][0]["ticker"] == "KO"
    assert saved["sp500"]["screened"] == 503
    assert saved["sp500"]["finished_at"]


def test_saving_requires_a_result_list():
    client = TestClient(main.app)

    assert client.post("/api/v1/investments/explore/save", json={"universe": "sp500"}).status_code == 400
