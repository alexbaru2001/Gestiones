from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import math
import os
from typing import Any

import pandas as pd


EXCHANGE_NAMES = {
    "NYQ": "NYSE",
    "NMS": "NASDAQ",
    "NGM": "NASDAQ GM",
    "NCM": "NASDAQ CM",
    "ASE": "NYSE American",
    "MCE": "BME Madrid",
    "LSE": "London Stock Exchange",
    "AMS": "Euronext Amsterdam",
    "EPA": "Euronext Paris",
    "FRA": "Frankfurt",
    "GER": "Xetra",
    "MIL": "Borsa Italiana",
}


@dataclass(frozen=True)
class SectorRules:
    rpd_min: float
    rpd_max: float
    dgr5_min: float
    payout_min: float
    payout_max: float
    per_min: float
    per_max: float
    de_max: float
    roe_min: float
    streak_min: int


BASELINE = SectorRules(3.0, 6.0, 5.0, 30.0, 70.0, 8.0, 18.0, 1.0, 10.0, 10)
SECTOR_TWEAKS = {
    "Utilities": SectorRules(3.5, 7.0, 3.0, 40.0, 80.0, 8.0, 20.0, 2.0, 8.0, 8),
    "Communication Services": SectorRules(4.0, 8.0, 2.0, 40.0, 85.0, 7.0, 16.0, 1.5, 8.0, 5),
    "Consumer Defensive": SectorRules(3.0, 6.0, 5.0, 30.0, 70.0, 10.0, 20.0, 1.0, 10.0, 10),
    "Healthcare": SectorRules(2.5, 5.5, 5.0, 30.0, 65.0, 10.0, 22.0, 1.0, 10.0, 8),
    "Industrials": SectorRules(2.5, 5.5, 5.0, 25.0, 60.0, 9.0, 18.0, 1.0, 10.0, 7),
    "Technology": SectorRules(1.5, 4.0, 7.0, 20.0, 55.0, 12.0, 25.0, 0.8, 12.0, 7),
    "Energy": SectorRules(4.0, 9.0, 0.0, 30.0, 70.0, 6.0, 12.0, 1.0, 8.0, 5),
    "Financial Services": SectorRules(3.0, 7.0, 3.0, 30.0, 60.0, 7.0, 12.0, 2.0, 10.0, 7),
    "Consumer Cyclical": SectorRules(2.5, 5.5, 5.0, 25.0, 60.0, 8.0, 18.0, 1.0, 10.0, 7),
}


@dataclass
class TickerMetrics:
    ticker: str
    name: str
    sector: str
    price: float
    currency: str
    exchange_code: str
    exchange_name: str
    country: str
    rpd_ttm: float
    rpd_forward: float
    dgr5: float
    dgr10: float
    payout: float
    payout_fcf: float
    per_ttm: float
    de_ratio: float
    roe: float
    ev_ebitda: float
    fcf_yield: float
    streak_years: int
    streak_growth: int
    fcf_pos_years: int
    dgr_reliable: bool


def analyze_ticker(ticker: str) -> dict[str, Any]:
    normalized = normalize_ticker(ticker)
    return _analyze_ticker_cached(normalized)


def normalize_ticker(ticker: str) -> str:
    normalized = ticker.strip().upper()
    if not normalized:
        raise ValueError("Indica un ticker para analizar.")
    if len(normalized) > 16 or not all(char.isalnum() or char in ".-=^" for char in normalized):
        raise ValueError("El ticker contiene caracteres no válidos.")
    return normalized


@lru_cache(maxsize=64)
def _analyze_ticker_cached(ticker: str) -> dict[str, Any]:
    metrics, dividends, prices = fetch_metrics(ticker)
    rules = rules_for_sector(metrics.sector)
    flags = red_flags(metrics)
    total_score, breakdown, details = score_company(metrics, rules)
    verdict = recommendation(total_score, flags)

    return clean_payload(
        {
            "ticker": metrics.ticker,
            "name": metrics.name,
            "sector": metrics.sector,
            "exchange": {
                "code": metrics.exchange_code,
                "name": metrics.exchange_name,
                "country": metrics.country,
                "currency": metrics.currency,
            },
            "price": metrics.price,
            "score": total_score,
            "recommendation": verdict,
            "flags": flags,
            "metrics": asdict(metrics),
            "rules": asdict(rules),
            "breakdown": breakdown,
            "details": details,
            "ai_analysis": build_ai_analysis(metrics, rules, total_score, breakdown, flags),
            "dividends_by_year": series_to_records(dividends, "year", "amount"),
            "price_history": series_to_records(prices.tail(1260), "date", "close"),
        }
    )


def fetch_metrics(ticker: str) -> tuple[TickerMetrics, pd.Series, pd.Series]:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Falta instalar yfinance en el backend.") from exc

    asset = yf.Ticker(ticker)
    info = safe_info(asset)
    history = safe_history(asset)
    price = latest_price(history, info)
    if math.isnan(price):
        raise ValueError(f"No se pudo obtener precio para {ticker}.")

    dividends = safe_dividends(asset)
    by_year_raw = dividends_by_year(dividends)
    by_year_clean = dividends_by_year_clean(dividends)
    cashflow = safe_cashflow(asset)
    payout_fcf, fcf_pos_years, fcf_series = cashflow_metrics(cashflow)

    dgr5 = cagr_from_annual(by_year_clean, 5)
    dgr10 = cagr_from_annual(by_year_clean, 10)
    ttm = dividends_ttm(dividends)
    forward_div = info.get("forwardAnnualDividendRate") or info.get("dividendRate")
    exchange_code = str(info.get("exchange") or "").strip()

    metrics = TickerMetrics(
        ticker=ticker,
        name=str(info.get("shortName") or info.get("longName") or ticker),
        sector=str(info.get("sector") or "Unknown"),
        price=price,
        currency=str(info.get("currency") or ""),
        exchange_code=exchange_code,
        exchange_name=EXCHANGE_NAMES.get(exchange_code, exchange_code or "Unknown"),
        country=str(info.get("country") or ""),
        rpd_ttm=(ttm / price * 100.0) if price > 0 else math.nan,
        rpd_forward=(float(forward_div) / price * 100.0) if is_number(forward_div) and price > 0 else math.nan,
        dgr5=(dgr5 * 100.0) if not math.isnan(dgr5) else math.nan,
        dgr10=(dgr10 * 100.0) if not math.isnan(dgr10) else math.nan,
        payout=ratio_to_percent(info.get("payoutRatio")),
        payout_fcf=payout_fcf,
        per_ttm=number_or_nan(info.get("trailingPE")),
        de_ratio=debt_to_equity(info.get("debtToEquity")),
        roe=ratio_to_percent(info.get("returnOnEquity")),
        ev_ebitda=ev_ebitda(info),
        fcf_yield=fcf_yield(fcf_series, info.get("marketCap")),
        streak_years=dividend_streak_years(by_year_raw),
        streak_growth=dividend_growth_streak(by_year_clean),
        fcf_pos_years=fcf_pos_years,
        dgr_reliable=not dgr_not_reliable(by_year_clean, by_year_raw),
    )
    prices = history["Close"].dropna() if history is not None and "Close" in history else pd.Series(dtype="float64")
    return metrics, by_year_raw, prices


def safe_info(asset: Any) -> dict[str, Any]:
    try:
        return asset.info or {}
    except Exception:
        return {}


def safe_history(asset: Any) -> pd.DataFrame:
    try:
        history = asset.history(period="max", interval="1d")
        if history is None or history.empty:
            return asset.history(period="20y", interval="1d")
        return history
    except Exception:
        return pd.DataFrame()


def safe_dividends(asset: Any) -> pd.Series:
    try:
        dividends = asset.dividends
        if dividends is not None:
            return dividends
    except Exception:
        pass
    return pd.Series(dtype="float64")


def safe_cashflow(asset: Any) -> pd.DataFrame:
    try:
        cashflow = asset.cashflow
        if cashflow is not None:
            return cashflow
    except Exception:
        pass
    return pd.DataFrame()


def latest_price(history: pd.DataFrame, info: dict[str, Any]) -> float:
    if history is not None and not history.empty and "Close" in history:
        close = history["Close"].dropna()
        if not close.empty:
            return float(close.iloc[-1])
    return number_or_nan(info.get("currentPrice") or info.get("regularMarketPrice"))


def to_naive_index(index: Any) -> pd.DatetimeIndex:
    return pd.to_datetime(index, utc=True, errors="coerce").tz_localize(None)


def dividends_ttm(dividends: pd.Series, ref_date: pd.Timestamp | None = None) -> float:
    if dividends is None or dividends.empty:
        return 0.0
    series = dividends.copy()
    series.index = to_naive_index(series.index)
    ref = pd.to_datetime(ref_date) if ref_date is not None else pd.to_datetime("today").tz_localize(None)
    return float(series[series.index > ref - pd.Timedelta(days=365)].sum())


def dividends_by_year(dividends: pd.Series) -> pd.Series:
    if dividends is None or dividends.empty:
        return pd.Series(dtype="float64")
    series = dividends.copy()
    series.index = to_naive_index(series.index)
    return series.groupby(series.index.year).sum()


def dividends_by_year_clean(dividends: pd.Series) -> pd.Series:
    if dividends is None or dividends.empty:
        return pd.Series(dtype="float64")
    series = dividends.copy()
    series.index = to_naive_index(series.index)
    grouped = series.groupby(series.index.year)
    counts = grouped.size()
    expected_payments = int(counts.mode().iloc[0]) if not counts.empty else 4
    cleaned = {}

    for year, group in grouped:
        values = group.sort_index().astype(float).values
        if len(values) == 0:
            continue
        median = float(pd.Series(values).median())
        regulars = [value for value in values if value <= 1.5 * median + 1e-12]
        if len(regulars) > expected_payments:
            regulars = sorted(regulars, key=lambda value: abs(value - median))[:expected_payments]
        if len(regulars) == expected_payments:
            cleaned[year] = float(sum(regulars))

    return pd.Series(cleaned).sort_index() if cleaned else pd.Series(dtype="float64")


def cagr_from_annual(series_by_year: pd.Series, years: int) -> float:
    series = series_by_year.dropna().sort_index()
    if len(series) < 2:
        return math.nan
    end_year = int(series.index.max())
    start_year = end_year - years
    if start_year not in series.index or end_year not in series.index:
        return math.nan
    start = float(series.loc[start_year])
    end = float(series.loc[end_year])
    if start <= 0 or end <= 0:
        return math.nan
    return (end / start) ** (1.0 / years) - 1.0


def dividend_streak_years(series_by_year: pd.Series) -> int:
    if series_by_year is None or series_by_year.empty:
        return 0
    streak = 0
    for year in sorted(series_by_year.index.tolist(), reverse=True):
        if series_by_year.loc[year] > 0:
            streak += 1
        else:
            break
    return streak


def dividend_growth_streak(series_by_year: pd.Series) -> int:
    if series_by_year is None or series_by_year.empty or len(series_by_year) < 2:
        return 0
    series = series_by_year.dropna().sort_index()
    streak = 0
    for index in range(len(series) - 1, 0, -1):
        if series.iloc[index] > series.iloc[index - 1]:
            streak += 1
        else:
            break
    return streak


def dgr_not_reliable(clean: pd.Series, raw: pd.Series) -> bool:
    if clean is None or len(clean) < 5:
        return True
    clean_5 = cagr_from_annual(clean, 5)
    raw_5 = cagr_from_annual(raw, 5)
    if not math.isnan(clean_5) and not math.isnan(raw_5):
        return abs(clean_5 - raw_5) > 0.05
    return False


def cashflow_metrics(cashflow: pd.DataFrame) -> tuple[float, int, pd.Series]:
    if cashflow is None or cashflow.empty:
        return math.nan, 0, pd.Series(dtype="float64")
    ordered = cashflow.loc[:, sorted(cashflow.columns)]
    fcf = ordered.loc["Free Cash Flow"].dropna() if "Free Cash Flow" in ordered.index else pd.Series(dtype="float64")
    positive_years = int((fcf > 0).sum()) if not fcf.empty else 0
    payout_fcf = math.nan
    if "Dividends Paid" in ordered.index and not fcf.empty:
        dividends_paid = ordered.loc["Dividends Paid"].dropna()
        if not dividends_paid.empty and abs(float(fcf.iloc[-1])) > 0:
            payout_fcf = abs(float(dividends_paid.iloc[-1])) / abs(float(fcf.iloc[-1])) * 100.0
    return payout_fcf, positive_years, fcf


def rules_for_sector(sector_name: str | None) -> SectorRules:
    return SECTOR_TWEAKS.get(sector_name or "", BASELINE)


def band_score(value: float, minimum: float | None = None, maximum: float | None = None, prefer_mid: bool = False) -> float:
    if math.isnan(number_or_nan(value)):
        return 0.0
    value = float(value)
    if minimum is not None and maximum is not None:
        if value < minimum:
            return 0.0 if minimum == 0 else max(0.0, 100.0 * (value / minimum))
        if value > maximum:
            return 0.0 if value == 0 else max(0.0, 100.0 * (maximum / value))
        if prefer_mid:
            middle = (minimum + maximum) / 2.0
            half_range = (maximum - minimum) / 2.0 or 1.0
            return max(0.0, 100.0 - 100.0 * abs(value - middle) / half_range)
        return 100.0
    if minimum is not None:
        return 100.0 if value >= minimum else 0.0
    if maximum is not None:
        return 100.0 if value <= maximum else 0.0
    return 50.0


def score_company(metrics: TickerMetrics, rules: SectorRules) -> tuple[float, dict[str, float], dict[str, list[dict[str, Any]]]]:
    details = {"Dividendo": [], "Solidez": [], "Valoración": [], "Historial": []}
    rpd_value = metrics.rpd_forward if not math.isnan(metrics.rpd_forward) else metrics.rpd_ttm
    dgr5_score = 50.0 if not metrics.dgr_reliable else band_score(metrics.dgr5, rules.dgr5_min)
    payout_value = metrics.payout_fcf if not math.isnan(metrics.payout_fcf) else metrics.payout

    dividend_inputs = [
        ("RPD", rpd_value, rules.rpd_min, rules.rpd_max, True, 0.35),
        ("DGR 5 años", metrics.dgr5, rules.dgr5_min, None, False, 0.30),
        ("DGR 10 años", metrics.dgr10, 0.0, None, False, 0.15),
        ("Payout", payout_value, rules.payout_min, rules.payout_max, True, 0.20),
    ]
    dividend_scores = []
    for label, value, minimum, maximum, prefer_mid, weight in dividend_inputs:
        score = dgr5_score if label == "DGR 5 años" else band_score(value, minimum, maximum, prefer_mid)
        dividend_scores.append(weight * score)
        details["Dividendo"].append(detail_row(label, value, minimum, maximum, weight, score, "%"))

    solid_inputs = [
        ("Deuda/Patrimonio", metrics.de_ratio, None, rules.de_max, False, 0.35, "x"),
        ("ROE", metrics.roe, rules.roe_min, None, False, 0.30, "%"),
        ("Racha de pagos", metrics.streak_years, rules.streak_min, None, False, 0.20, "a"),
        ("Racha de crecimiento", metrics.streak_growth, 5.0, None, False, 0.15, "a"),
    ]
    solid_scores = []
    for label, value, minimum, maximum, prefer_mid, weight, unit in solid_inputs:
        score = band_score(value, minimum, maximum, prefer_mid)
        solid_scores.append(weight * score)
        details["Solidez"].append(detail_row(label, value, minimum, maximum, weight, score, unit))

    valuation_score = band_score(metrics.per_ttm, rules.per_min, rules.per_max, True)
    details["Valoración"].append(detail_row("PER", metrics.per_ttm, rules.per_min, rules.per_max, 1.0, valuation_score, ""))

    history_inputs = [
        ("Racha de crecimiento", metrics.streak_growth, 5.0, None, 0.5, "a"),
        ("Años con FCF positivo", metrics.fcf_pos_years, 3.0, None, 0.5, ""),
    ]
    history_scores = []
    for label, value, minimum, maximum, weight, unit in history_inputs:
        score = band_score(value, minimum, maximum)
        history_scores.append(weight * score)
        details["Historial"].append(detail_row(label, value, minimum, maximum, weight, score, unit))

    breakdown = {
        "Dividendo": 0.40 * sum(dividend_scores),
        "Solidez": 0.25 * sum(solid_scores),
        "Valoración": 0.15 * valuation_score,
        "Historial": 0.20 * sum(history_scores),
    }
    total = max(0.0, min(100.0, sum(breakdown.values())))
    return total, breakdown, details


def detail_row(label: str, value: float, minimum: float | None, maximum: float | None, weight: float, score: float, unit: str) -> dict[str, Any]:
    return {
        "metric": label,
        "value": None if math.isnan(number_or_nan(value)) else value,
        "range": range_label(minimum, maximum, unit),
        "weight": weight,
        "subscore": score,
    }


def range_label(minimum: float | None, maximum: float | None, unit: str) -> str:
    suffix = f" {unit}" if unit else ""
    if minimum is None and maximum is None:
        return "-"
    if minimum is None:
        return f"<= {maximum}{suffix}"
    if maximum is None:
        return f">= {minimum}{suffix}"
    return f"{minimum}-{maximum}{suffix}"


def red_flags(metrics: TickerMetrics) -> list[str]:
    flags = []
    if not math.isnan(metrics.dgr5) and metrics.dgr5 < 0 and not math.isnan(metrics.payout) and metrics.payout > 80:
        flags.append("DGR 5a negativo con payout alto")
    if not math.isnan(metrics.payout_fcf) and metrics.payout_fcf > 100:
        flags.append("Payout FCF superior al 100%")
    if metrics.streak_growth == 0 and not math.isnan(metrics.dgr5) and metrics.dgr5 < 0:
        flags.append("Recortes o ausencia de crecimiento del dividendo")
    return flags


def recommendation(total_score: float, flags: list[str]) -> str:
    if flags:
        return "Pausa / revisar banderas rojas"
    if total_score >= 75:
        return "Comprar / añadir con DCA normal"
    if total_score >= 60:
        return "Vigilar o DCA prudente"
    return "Mantenerse al margen"


def build_ai_analysis(
    metrics: TickerMetrics,
    rules: SectorRules,
    total_score: float,
    breakdown: dict[str, float],
    flags: list[str],
) -> dict[str, Any]:
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key")
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    if not api_key:
        return {
            "configured": False,
            "model": model,
            "text": None,
            "error": "Configura GROQ_API_KEY para activar el análisis automático.",
        }

    try:
        text = request_groq_analysis(api_key, model, build_groq_prompt(metrics, rules, total_score, breakdown, flags))
        return {"configured": True, "model": model, "text": text, "error": None}
    except Exception as exc:
        return {"configured": True, "model": model, "text": None, "error": f"No se pudo generar análisis Groq: {exc}"}


def build_groq_prompt(
    metrics: TickerMetrics,
    rules: SectorRules,
    total_score: float,
    breakdown: dict[str, float],
    flags: list[str],
) -> str:
    return f"""
Eres un analista conservador de inversión a largo plazo por dividendos crecientes.
Usa únicamente los datos proporcionados. Si falta un dato, dilo claramente como dato no disponible.

Empresa:
- Ticker: {metrics.ticker}
- Nombre: {metrics.name}
- Sector: {metrics.sector}
- País: {metrics.country}
- Divisa: {metrics.currency}
- Precio: {metrics.price}

Ratios:
- RPD TTM (%): {metrics.rpd_ttm}
- RPD forward (%): {metrics.rpd_forward}
- DGR 5a (%): {metrics.dgr5}
- DGR 10a (%): {metrics.dgr10}
- Payout (%): {metrics.payout}
- Payout FCF (%): {metrics.payout_fcf}
- PER TTM: {metrics.per_ttm}
- Deuda/Patrimonio (x): {metrics.de_ratio}
- ROE (%): {metrics.roe}
- EV/EBITDA (x): {metrics.ev_ebitda}
- FCF Yield (%): {metrics.fcf_yield}
- Racha de pagos: {metrics.streak_years}
- Racha de crecimiento: {metrics.streak_growth}
- Años FCF positivo: {metrics.fcf_pos_years}
- DGR fiable: {metrics.dgr_reliable}

Umbrales sectoriales:
- RPD: {rules.rpd_min}-{rules.rpd_max} %
- DGR 5a mínimo: {rules.dgr5_min} %
- Payout: {rules.payout_min}-{rules.payout_max} %
- PER: {rules.per_min}-{rules.per_max}
- Deuda máxima: {rules.de_max} x
- ROE mínimo: {rules.roe_min} %
- Racha mínima: {rules.streak_min} años

Score:
- Total: {total_score}
- Desglose: {breakdown}
- Banderas rojas: {flags if flags else "Ninguna"}

Devuelve en español:
1. Resumen ejecutivo en 5 líneas.
2. Diagnóstico del dividendo.
3. Solidez financiera.
4. Valoración frente a los umbrales.
5. Conclusión: Apta, Dudosa o No apta para dividendos crecientes, con 3 acciones prácticas.
""".strip()


def request_groq_analysis(api_key: str, model: str, prompt: str) -> str:
    import requests

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Responde como analista financiero conservador. No inventes cifras."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 900,
    }
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        json=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Gestiones/0.1",
        },
        timeout=60,
    )

    if response.status_code >= 400:
        detail = response.text.strip()
        try:
            parsed = response.json()
            detail = parsed.get("error", {}).get("message") or parsed.get("message") or detail
        except ValueError:
            pass
        raise RuntimeError(f"HTTP {response.status_code}: {detail[:500]}")

    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def series_to_records(series: pd.Series, index_key: str, value_key: str) -> list[dict[str, Any]]:
    records = []
    if series is None or series.empty:
        return records
    for index, value in series.dropna().items():
        if isinstance(index, pd.Timestamp):
            index_value = index.strftime("%Y-%m-%d")
        else:
            index_value = int(index) if isinstance(index, int) or str(index).isdigit() else str(index)
        records.append({index_key: index_value, value_key: float(value)})
    return records


def is_number(value: Any) -> bool:
    try:
        return not math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def number_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def ratio_to_percent(value: Any) -> float:
    number = number_or_nan(value)
    if math.isnan(number):
        return math.nan
    return number * 100.0 if abs(number) <= 1 else number


def debt_to_equity(value: Any) -> float:
    number = number_or_nan(value)
    if math.isnan(number):
        return math.nan
    return number / 100.0 if 10 < number < 10000 else number


def ev_ebitda(info: dict[str, Any]) -> float:
    ev = number_or_nan(info.get("enterpriseValue"))
    ebitda = number_or_nan(info.get("ebitda"))
    return ev / ebitda if not math.isnan(ev) and not math.isnan(ebitda) and ebitda > 0 else math.nan


def fcf_yield(fcf_series: pd.Series, market_cap: Any) -> float:
    cap = number_or_nan(market_cap)
    if fcf_series.empty or math.isnan(cap) or cap <= 0:
        return math.nan
    return float(fcf_series.iloc[-1]) / cap * 100.0


def clean_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clean_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_payload(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value
