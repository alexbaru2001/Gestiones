from pathlib import Path
import pandas as pd

hoy = str(pd.Timestamp.today().to_period('M'))

REGISTROS_DIR = Path(__file__).resolve().parent / "Data" / "Registros"
ARCHIVO_BASE_REGISTROS = REGISTROS_DIR / "Inicio.xlsx"
HOJAS_REGISTRO = ["Gastos", "Ingresos", "Transferencias"]
OBJETIVOS_VISTA_CONFIG_PATH = Path(__file__).resolve().parent / "Data" / "objetivos_vista.json"

saldos_iniciales = {
    'Principal': 482.99000000000007,
    'Cobee': 0,
    'Metálico': 54.999999999999986,
    'Revolut': 0.0,
    'Ahorro': 4680.999999999999
}
