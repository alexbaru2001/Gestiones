from fastapi.testclient import TestClient

from backend import main
from backend.domain import investments
from backend.domain.investments import build_ai_analysis, normalize_ticker


def test_normalize_ticker_keeps_market_suffix_dot():
    assert normalize_ticker(" rovi.mc ") == "ROVI.MC"


def test_ai_analysis_is_optional_without_groq_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    analysis = build_ai_analysis(metrics=None, rules=None, total_score=0, breakdown={}, flags=[])

    assert analysis["configured"] is False
    assert "GROQ_API_KEY" in analysis["error"]


def test_ai_analysis_uses_groq_when_key_is_configured(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(investments, "build_groq_prompt", lambda *_args: "prompt")
    monkeypatch.setattr(investments, "request_groq_analysis", lambda *_args: "Análisis generado")

    analysis = build_ai_analysis(metrics=None, rules=None, total_score=0, breakdown={}, flags=[])

    assert analysis["configured"] is True
    assert analysis["text"] == "Análisis generado"
    assert analysis["error"] is None


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
