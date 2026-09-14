from datetime import datetime, timedelta, timezone
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.domain import investments


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(investments, "get_analysis_cache_path", lambda: tmp_path / "analysis_cache")


def fake_analysis(ticker="KO", score=80.0):
    return {"ticker": ticker, "name": "Coca-Cola", "score": score, "price": 60.0,
            "metrics": {"rpd_ttm": 3.0}, "rules": {}, "breakdown": {}, "sector": "Consumer Defensive",
            "recommendation": "Comprar", "flags": [], "exchange": {"currency": "USD"}}


# --- Caché en disco con caducidad -------------------------------------------------

def test_second_analysis_is_served_from_disk_without_calling_yahoo(monkeypatch):
    llamadas = []
    monkeypatch.setattr(investments, "build_analysis", lambda ticker: llamadas.append(ticker) or fake_analysis())

    investments.analyze_ticker("KO")
    result = investments.analyze_ticker("KO")

    assert llamadas == ["KO"]
    assert result["ticker"] == "KO"
    assert result["cached_at"]  # el frontend enseña de cuándo son los datos


def test_expired_cache_is_ignored(monkeypatch):
    """La caché anterior vivía en memoria y no caducaba: devolvía el precio de días atrás."""
    llamadas = []
    monkeypatch.setattr(investments, "build_analysis", lambda ticker: llamadas.append(ticker) or fake_analysis())
    investments.analyze_ticker("KO")

    stale = datetime.now(timezone.utc) - timedelta(hours=investments.ANALYSIS_CACHE_TTL_HOURS + 1)
    path = investments.analysis_cache_file("KO")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["cached_at"] = stale.isoformat(timespec="seconds")
    path.write_text(json.dumps(payload), encoding="utf-8")

    investments.analyze_ticker("KO")
    assert llamadas == ["KO", "KO"]


def test_refresh_skips_the_cache(monkeypatch):
    llamadas = []
    monkeypatch.setattr(investments, "build_analysis", lambda ticker: llamadas.append(ticker) or fake_analysis())

    investments.analyze_ticker("KO")
    investments.analyze_ticker("KO", refresh=True)

    assert llamadas == ["KO", "KO"]


def test_unreadable_cache_file_falls_back_to_a_fresh_analysis(monkeypatch):
    monkeypatch.setattr(investments, "build_analysis", lambda ticker: fake_analysis())
    investments.analyze_ticker("KO")
    investments.analysis_cache_file("KO").write_text("{roto", encoding="utf-8")

    assert investments.analyze_ticker("KO")["ticker"] == "KO"


def test_cache_filename_is_safe_for_tickers_with_market_suffix():
    assert investments.analysis_cache_file("ROVI.MC").name == "ROVI.MC.json"
    assert "/" not in investments.analysis_cache_file("BRK/B").name


# --- Valoración por RPD media ------------------------------------------------------

def test_yield_history_uses_the_average_price_of_each_closed_year():
    dates = pd.to_datetime(["2023-01-02", "2023-07-02", "2024-01-02", "2024-07-02"])
    history = pd.DataFrame({"Close": [90.0, 110.0, 45.0, 55.0]}, index=dates)
    by_year = pd.Series({2023: 2.0, 2024: 2.5})

    yields = investments.dividend_yield_history(by_year, history)

    assert yields[2023] == pytest.approx(2.0)  # 2 / 100
    assert yields[2024] == pytest.approx(5.0)  # 2.5 / 50


def test_yield_history_skips_the_current_year_because_it_is_incomplete():
    current_year = pd.Timestamp.today().year
    history = pd.DataFrame({"Close": [100.0]}, index=pd.to_datetime([f"{current_year}-01-02"]))

    assert investments.dividend_yield_history(pd.Series({current_year: 1.0}), history).empty


def test_average_of_last_years_ignores_older_points():
    series = pd.Series({2019: 1.0, 2020: 2.0, 2021: 3.0, 2022: 4.0, 2023: 5.0, 2024: 6.0})

    assert investments.average_of_last(series, 5) == pytest.approx(4.0)


def test_average_of_last_returns_nan_without_data():
    import math

    assert math.isnan(investments.average_of_last(pd.Series(dtype="float64"), 5))


# --- Comparador --------------------------------------------------------------------

def test_compare_returns_one_column_per_ticker_without_the_long_series(monkeypatch):
    monkeypatch.setattr(investments, "analyze_ticker", lambda ticker, refresh=False: {
        **fake_analysis(ticker), "price_history": [{"date": "2026-01-01", "close": 1.0}], "ai_analysis": {"text": "x"},
    })

    comparison = investments.compare_tickers(["KO", "JNJ"])

    assert [column["ticker"] for column in comparison["columns"]] == ["KO", "JNJ"]
    assert "price_history" not in comparison["columns"][0]
    assert "ai_analysis" not in comparison["columns"][0]
    assert comparison["metrics"][0]["key"] == "rpd_ttm"


def test_compare_deduplicates_and_normalizes_tickers(monkeypatch):
    monkeypatch.setattr(investments, "analyze_ticker", lambda ticker, refresh=False: fake_analysis(ticker))

    comparison = investments.compare_tickers([" ko ", "KO", "jnj"])

    assert [column["ticker"] for column in comparison["columns"]] == ["KO", "JNJ"]


def test_compare_reports_a_failing_ticker_without_losing_the_rest(monkeypatch):
    def analyze(ticker, refresh=False):
        if ticker == "XXXX":
            raise ValueError("No se pudo obtener precio para XXXX.")
        return fake_analysis(ticker)

    monkeypatch.setattr(investments, "analyze_ticker", analyze)

    comparison = investments.compare_tickers(["KO", "XXXX"])

    assert [column["ticker"] for column in comparison["columns"]] == ["KO"]
    assert comparison["errors"][0]["ticker"] == "XXXX"


def test_compare_rejects_too_many_tickers():
    with pytest.raises(ValueError, match="como mucho"):
        investments.compare_tickers(["A", "B", "C", "D", "E"])


# --- Búsqueda por nombre -----------------------------------------------------------

def test_search_returns_equities_by_name(monkeypatch):
    class FakeSearch:
        def __init__(self, query, max_results=10):
            self.quotes = [
                {"symbol": "IBE.MC", "longname": "Iberdrola, S.A.", "quoteType": "EQUITY",
                 "exchDisp": "Madrid", "sectorDisp": "Utilities"},
                {"symbol": "IBE.MC", "longname": "duplicado", "quoteType": "EQUITY"},
                {"symbol": "IBDRY", "shortname": "Iberdrola S.A.", "quoteType": "EQUITY", "exchDisp": "OID"},
                {"symbol": "XYZ", "longname": "Un futuro", "quoteType": "FUTURE"},
            ]

    import yfinance
    monkeypatch.setattr(yfinance, "Search", FakeSearch)

    results = investments.search_tickers("iberdrola")

    assert [item["ticker"] for item in results] == ["IBE.MC", "IBDRY"]
    assert results[0]["name"] == "Iberdrola, S.A."
    assert results[0]["exchange"] == "Madrid"


def test_search_ignores_queries_that_are_too_short():
    assert investments.search_tickers("i") == []


def test_search_endpoint_surfaces_a_yahoo_failure(monkeypatch):
    monkeypatch.setattr(main, "search_tickers", lambda query: (_ for _ in ()).throw(RuntimeError("Yahoo no responde")))
    client = TestClient(main.app)

    response = client.get("/api/v1/investments/search?q=iberdrola")

    assert response.status_code == 502
    assert "Yahoo no responde" in response.json()["detail"]


def test_compare_endpoint_splits_the_ticker_list(monkeypatch):
    monkeypatch.setattr(main, "compare_tickers", lambda tickers, refresh=False: {"columns": [{"ticker": t} for t in tickers]})
    client = TestClient(main.app)

    response = client.get("/api/v1/investments/compare?tickers=KO,JNJ,")

    assert [column["ticker"] for column in response.json()["result"]["columns"]] == ["KO", "JNJ"]
