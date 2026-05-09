from fastapi.testclient import TestClient

from backend import main


def test_analyze_investment_returns_payload(monkeypatch):
    def fake_analyze_ticker(ticker: str):
        return {
            "ticker": ticker,
            "name": "Coca-Cola",
            "score": 82.0,
            "recommendation": "Comprar / añadir con DCA normal",
        }

    monkeypatch.setattr(main, "analyze_ticker", fake_analyze_ticker)
    client = TestClient(main.app)

    response = client.get("/api/v1/investments/analyze?ticker=KO")

    assert response.status_code == 200
    assert response.json()["result"]["ticker"] == "KO"
    assert response.json()["result"]["score"] == 82.0


def test_analyze_investment_reports_invalid_ticker(monkeypatch):
    def fake_analyze_ticker(_ticker: str):
        raise ValueError("Ticker no válido")

    monkeypatch.setattr(main, "analyze_ticker", fake_analyze_ticker)
    client = TestClient(main.app)

    response = client.get("/api/v1/investments/analyze?ticker=%20")

    assert response.status_code == 400
    assert response.json()["detail"] == "Ticker no válido"
