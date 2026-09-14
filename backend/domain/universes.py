"""Universos de exploración: los conjuntos de empresas que se pueden cribar de una vez.

Hay dos clases de universo, y la diferencia importa al leer los resultados:

- Los índices (S&P 500, IBEX 35, Euro Stoxx 50) llevan su lista de miembros exacta, extraída de
  Wikipedia y guardada como semilla para que la app funcione sin red. Se pueden refrescar y
  corregir a mano en `universes.json`, igual que el catálogo de activos.
- MSCI World y MSCI Emergentes no publican su lista de miembros de forma libre, así que se
  aproximan cogiendo las mayores empresas de cada país que compone el índice. Es una aproximación
  razonable, no el índice, y la pantalla lo dice.

No se filtra por capitalización mínima porque Yahoo la devuelve en moneda local: un umbral de
5.000 millones dejaba entrar a cualquier empresa indonesia (5.000 M de rupias son 300.000 dólares)
y llenaba el universo emergente de valores diminutos. En su lugar se cogen las N mayores de cada
país, que además reparte el universo por regiones como hace un índice de verdad.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any

from backend.config import get_universes_path


@dataclass(frozen=True)
class Universe:
    key: str
    label: str
    kind: str  # "index" o "screen"
    description: str
    members: tuple[str, ...] = ()
    regions: tuple[str, ...] = ()
    per_region: int = 0
    source_url: str = ""
    approximate: bool = False
    size: int = 0


# Regiones tal y como las nombra el buscador de Yahoo.
DEVELOPED_REGIONS = ("us", "gb", "jp", "ca", "fr", "de", "ch", "au", "nl", "se", "es", "it", "dk", "hk", "sg", "fi", "be", "no", "ie", "at", "pt", "nz", "il")
# Argentina y Chile quedan fuera a propósito: lo que Yahoo lista allí son casi todo CEDEAR
# (Apple, Alibaba, Tesla cotizando en local), que no son empresas emergentes sino duplicados de
# megacaps globales, y por capitalización se comían el universo entero.
EMERGING_REGIONS = ("cn", "in", "br", "tw", "kr", "za", "mx", "id", "th", "my", "tr", "pl", "gr", "ph", "pe", "hu", "cz", "co", "eg", "qa", "ae", "sa", "kw")

UNIVERSES: dict[str, Universe] = {}


def _register(universe: Universe) -> None:
    UNIVERSES[universe.key] = universe


def load_overrides() -> dict[str, list[str]]:
    """Listas corregidas a mano o refrescadas desde Wikipedia; pisan a la semilla del código."""
    path = get_universes_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        str(key): [str(item).strip().upper() for item in value if str(item).strip()]
        for key, value in payload.items()
        if isinstance(value, list)
    }


def save_overrides(overrides: dict[str, list[str]]) -> None:
    path = get_universes_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(overrides, ensure_ascii=False, indent=1), encoding="utf-8")


def get_universe(key: str) -> Universe:
    universe = UNIVERSES.get(key)
    if universe is None:
        raise ValueError(f"No conozco el universo '{key}'.")
    return universe


def get_members(key: str) -> list[str]:
    universe = get_universe(key)
    if universe.kind != "index":
        return []
    return load_overrides().get(key, list(universe.members))


def list_universes() -> list[dict[str, Any]]:
    overrides = load_overrides()
    catalog = []
    for universe in UNIVERSES.values():
        members = overrides.get(universe.key, list(universe.members))
        catalog.append(
            {
                "key": universe.key,
                "label": universe.label,
                "kind": universe.kind,
                "description": universe.description,
                "approximate": universe.approximate,
                "size": len(members) if universe.kind == "index" else universe.size,
                "source_url": universe.source_url,
            }
        )
    return catalog


# --- Refresco de las listas de índices desde Wikipedia -----------------------------------------

WIKIPEDIA_SOURCES = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "ibex35": "https://es.wikipedia.org/wiki/IBEX_35",
}


def refresh_members(key: str) -> list[str]:
    """Vuelve a leer la lista de miembros de Wikipedia y la guarda.

    Solo para los índices con una tabla de tickers publicada; el Euro Stoxx 50 no la tiene (su
    tabla solo trae nombres), así que su lista se mantiene a mano en `universes.json`.
    """
    url = WIKIPEDIA_SOURCES.get(key)
    if not url:
        raise ValueError("Ese universo no se puede refrescar automáticamente; edítalo a mano en universes.json.")

    import requests

    try:
        response = requests.get(url, headers={"User-Agent": "Gestiones/1.0"}, timeout=25)
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"No se pudo leer la lista de {key}: {exc}") from exc

    members = _parse_sp500(response.text) if key == "sp500" else _parse_ibex35(response.text)
    if len(members) < 20:
        raise RuntimeError(f"La página de {key} devolvió solo {len(members)} tickers; no se sustituye la lista actual.")

    overrides = load_overrides()
    overrides[key] = members
    save_overrides(overrides)
    return members


def _cells(row: str) -> list[str]:
    return [re.sub(r"<[^>]+>", "", cell).strip() for cell in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)]


def _parse_sp500(html: str) -> list[str]:
    match = re.search(r'id="constituents"(.*?)</table>', html, re.S)
    if not match:
        return []
    members = []
    for row in re.split(r"<tr", match.group(1))[1:]:
        cells = _cells(row)
        if cells and re.fullmatch(r"[A-Z][A-Z.\-]{0,6}", cells[0]):
            members.append(cells[0].replace(".", "-"))  # Yahoo escribe BRK-B donde Wikipedia pone BRK.B
    return members


def _parse_ibex35(html: str) -> list[str]:
    tables = re.findall(r'<table[^>]*class="[^"]*wikitable[^"]*"[^>]*>(.*?)</table>', html, re.S)
    if not tables:
        return []
    members = []
    for row in re.split(r"<tr", tables[0])[2:]:
        cells = _cells(row)
        if cells and re.fullmatch(r"[A-Z]{2,5}", cells[0]):
            members.append(f"{cells[0]}.MC")
    return members


# --- Semillas ----------------------------------------------------------------------------------

SP500_MEMBERS = (
    "MMM", "AOS", "ABT", "ABBV", "ACN", "ADBE", "AMD", "AES", "AFL", "A", "APD", "ABNB", "AKAM", "ALB",
    "ARE", "ALGN", "ALLE", "LNT", "ALL", "GOOGL", "GOOG", "MO", "AMZN", "AMCR", "AEE", "AEP", "AXP",
    "AIG", "AMT", "AWK", "AMP", "AME", "AMGN", "APH", "ADI", "AON", "APA", "APO", "AAPL", "AMAT", "APP",
    "APTV", "ACGL", "ADM", "ARES", "ANET", "AJG", "AIZ", "T", "ATO", "ADSK", "ADP", "AZO", "AVY",
    "AXON", "BKR", "BALL", "BAC", "BAX", "BDX", "BRK-B", "BBY", "TECH", "BIIB", "BLK", "BX", "XYZ",
    "BNY", "BA", "BKNG", "BSX", "BMY", "AVGO", "BR", "BRO", "BF-B", "BLDR", "BG", "BXP", "CHRW", "CDNS",
    "CPT", "COF", "CAH", "CCL", "CARR", "CVNA", "CASY", "CAT", "CBOE", "CBRE", "CDW", "COR", "CNC",
    "CNP", "CF", "CRL", "SCHW", "CHTR", "CVX", "CMG", "CB", "CHD", "CIEN", "CI", "CINF", "CTAS", "CSCO",
    "C", "CFG", "CLX", "CME", "CMS", "KO", "CTSH", "COHR", "COIN", "CL", "CMCSA", "FIX", "COP", "ED",
    "STZ", "CEG", "COO", "CPRT", "GLW", "CPAY", "CTVA", "CSGP", "COST", "CRH", "CRWD", "CCI", "CSX",
    "CMI", "CVS", "DHR", "DRI", "DDOG", "DVA", "DECK", "DE", "DELL", "DAL", "DVN", "DXCM", "FANG",
    "DLR", "DG", "DLTR", "D", "DPZ", "DASH", "DOV", "DOW", "DHI", "DTE", "DUK", "DD", "ETN", "EBAY",
    "ECHO", "ECL", "EIX", "EW", "ELV", "EME", "EMR", "ETR", "EOG", "EQT", "EFX", "EQIX", "ERIE", "ESS",
    "EL", "EG", "EVRG", "ES", "EXC", "EXE", "EXPE", "EXPD", "EXR", "XOM", "FFIV", "FDS", "FICO", "FAST",
    "FRT", "FDX", "FDXF", "FERG", "FIS", "FITB", "FSLR", "FE", "FISV", "FLEX", "F", "FTNT", "FTV",
    "FOXA", "FOX", "BEN", "FCX", "GRMN", "IT", "GE", "GEHC", "GEV", "GEN", "GNRC", "GD", "GIS", "GM",
    "GPC", "GILD", "GPN", "GL", "GDDY", "GS", "HAL", "HIG", "HAS", "HCA", "DOC", "HSIC", "HSY", "HPE",
    "HLT", "HD", "HONA", "HON", "HRL", "HST", "HWM", "HPQ", "HUBB", "HUM", "HBAN", "HII", "IBM", "IEX",
    "IDXX", "ITW", "INCY", "IR", "PODD", "INTC", "IBKR", "ICE", "IFF", "IP", "INTU", "ISRG", "IVZ",
    "INVH", "IQV", "IRM", "JBHT", "JBL", "JKHY", "J", "JNJ", "JCI", "JPM", "KVUE", "KDP", "KEY", "KEYS",
    "KMB", "KIM", "KMI", "KKR", "KLAC", "KHC", "KR", "LHX", "LH", "LRCX", "LVS", "LDOS", "LEN", "LII",
    "LLY", "LIN", "LYV", "LMT", "L", "LOW", "LULU", "LITE", "LYB", "MTB", "MPC", "MAR", "MRSH", "MLM",
    "MRVL", "MAS", "MA", "MKC", "MCD", "MCK", "MDT", "MRK", "META", "MET", "MTD", "MGM", "MCHP", "MU",
    "MSFT", "MAA", "MRNA", "TAP", "MDLZ", "MPWR", "MNST", "MCO", "MS", "MOS", "MSI", "MSCI", "NDAQ",
    "NTAP", "NFLX", "NEM", "NWSA", "NWS", "NEE", "NKE", "NI", "NDSN", "NSC", "NTRS", "NOC", "NCLH",
    "NRG", "NUE", "NVDA", "NVR", "NXPI", "ORLY", "OXY", "ODFL", "OMC", "ON", "OKE", "ORCL", "OTIS",
    "PCAR", "PKG", "PLTR", "PANW", "PSKY", "PH", "PAYX", "PYPL", "PNR", "PEP", "PFE", "PCG", "PM",
    "PSX", "PNW", "PNC", "PPG", "PPL", "PFG", "PG", "PGR", "PLD", "PRU", "PEG", "PTC", "PSA", "PHM",
    "PWR", "QCOM", "DGX", "Q", "RL", "RJF", "RDDT", "RTX", "O", "REG", "REGN", "RF", "RSG", "RMD",
    "RVTY", "HOOD", "ROK", "ROL", "ROP", "ROST", "RCL", "SPGI", "CRM", "SNDK", "SBAC", "SLB", "STX",
    "SRE", "NOW", "SHW", "SPG", "SWKS", "SJM", "SW", "SNA", "SOLV", "SO", "LUV", "SWK", "SBUX", "STT",
    "STLD", "STE", "SYK", "SMCI", "SYF", "SNPS", "SYY", "TMUS", "TROW", "TTWO", "TPR", "TRGP", "TGT",
    "TEL", "TDY", "TER", "TSLA", "TXN", "TPL", "TXT", "TMO", "TJX", "TKO", "TTD", "TSCO", "TT", "TDG",
    "TRV", "TRMB", "TFC", "TYL", "TSN", "USB", "UBER", "UDR", "ULTA", "UNP", "UAL", "UPS", "URI", "UNH",
    "UHS", "VLO", "VEEV", "VTR", "VLTO", "VRSN", "VRSK", "VZ", "VRTX", "VRT", "VTRS", "VICI", "V",
    "VST", "VMRK", "VMC", "WRB", "GWW", "WAB", "WMT", "DIS", "WBD", "WM", "WAT", "WEC", "WFC", "WELL",
    "WST", "WDC", "WY", "WSM", "WMB", "WTW", "WDAY", "WYNN", "XEL", "XYL", "YUM", "ZBRA", "ZBH", "ZTS"
)

IBEX35_MEMBERS = (
    "ANA.MC", "ANE.MC", "ACX.MC", "ACS.MC", "AENA.MC", "AMS.MC", "MTS.MC", "SAB.MC", "BKT.MC",
    "BBVA.MC", "CABK.MC", "CLNX.MC", "ENG.MC", "ELE.MC", "FER.MC", "FDR.MC", "GRF.MC", "IAG.MC",
    "IBE.MC", "ITX.MC", "IDR.MC", "COL.MC", "LOG.MC", "MAP.MC", "MEL.MC", "MRL.MC", "NTGY.MC", "RED.MC",
    "REP.MC", "ROVI.MC", "SCYR.MC", "SAN.MC", "SLR.MC", "TEF.MC", "UNI.MC"
)

EUROSTOXX50_MEMBERS = (
    "ABI.BR", "AD.AS", "ADS.DE", "ADYEN.AS", "AI.PA", "AIR.PA", "ALV.DE", "ARGX.BR", "ASML.AS",
    "BAS.DE", "BBVA.MC", "BMW.DE", "BN.PA", "BNP.PA", "CS.PA", "DB1.DE", "DBK.DE", "DG.PA", "DHL.DE",
    "DTE.DE", "EL.PA", "ENEL.MI", "ENI.MI", "ENR.DE", "IBE.MC", "IFX.DE", "INGA.AS", "ISP.MI",
    "IXD1.DE", "MBG.DE", "MC.PA", "MUV2.DE", "NDA-FI.HE", "OR.PA", "PRX.AS", "RACE.MI", "RHM.DE",
    "RMS.PA", "SAF.PA", "SAN.MC", "SAN.PA", "SAP.DE", "SGO.PA", "SU.PA", "TTE.PA", "UCG.MI", "VOW3.DE",
    "WKL.AS"
)


_register(Universe(
    key="sp500",
    label="S&P 500",
    kind="index",
    description="Las 500 mayores cotizadas de Estados Unidos. Lista exacta de miembros.",
    members=SP500_MEMBERS,
    source_url=WIKIPEDIA_SOURCES["sp500"],
))
_register(Universe(
    key="eurostoxx50",
    label="Euro Stoxx 50",
    kind="index",
    description="Las mayores cotizadas de la zona euro. Lista mantenida a mano en universes.json, porque Wikipedia no publica sus tickers.",
    members=EUROSTOXX50_MEMBERS,
))
_register(Universe(
    key="ibex35",
    label="IBEX 35",
    kind="index",
    description="Las 35 mayores cotizadas españolas. Lista exacta de miembros.",
    members=IBEX35_MEMBERS,
    source_url=WIKIPEDIA_SOURCES["ibex35"],
))
_register(Universe(
    key="msci_world",
    label="MSCI World (aproximado)",
    kind="screen",
    description="Las mayores cotizadas de cada país desarrollado. Aproximación por país y tamaño, no la lista oficial del índice.",
    regions=DEVELOPED_REGIONS,
    per_region=22,
    approximate=True,
    size=22 * len(DEVELOPED_REGIONS),
))
_register(Universe(
    key="msci_em",
    label="MSCI Emergentes (aproximado)",
    kind="screen",
    description="Las mayores cotizadas de cada país emergente. Aproximación por país y tamaño; excluye Argentina y Chile, cuyos listados en Yahoo son casi todo CEDEAR de empresas extranjeras.",
    regions=EMERGING_REGIONS,
    per_region=18,
    approximate=True,
    size=18 * len(EMERGING_REGIONS),
))
