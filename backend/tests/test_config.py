from pathlib import Path

from backend.config import DEFAULT_CORS_ORIGINS, DEFAULT_OBJECTIVES_PATH, get_cors_origins, get_objectives_path


def test_cors_origins_default_to_local_frontend(monkeypatch):
    monkeypatch.delenv("GESTIONES_CORS_ORIGINS", raising=False)

    assert get_cors_origins() == list(DEFAULT_CORS_ORIGINS)


def test_cors_origins_can_be_configured(monkeypatch):
    monkeypatch.setenv("GESTIONES_CORS_ORIGINS", "https://app.example.com, http://localhost:5173 ,")

    assert get_cors_origins() == ["https://app.example.com", "http://localhost:5173"]


def test_objectives_path_defaults_to_legacy_data_file(monkeypatch):
    monkeypatch.delenv("GESTIONES_OBJECTIVES_PATH", raising=False)

    assert get_objectives_path() == DEFAULT_OBJECTIVES_PATH


def test_objectives_path_can_be_configured(monkeypatch, tmp_path):
    path = tmp_path / "objetivos.json"
    monkeypatch.setenv("GESTIONES_OBJECTIVES_PATH", str(path))

    assert get_objectives_path() == Path(path)
