#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 01:13:27 2025

@author: alex
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 12:02:06 2025

@author: abarragan1
"""
import json
import os
import re
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import unicodedata
import plotly.express as px
# from financier import Account, Transaction, Budget
import warnings
warnings.filterwarnings('ignore')
import plotly.graph_objects as go
from datetime import datetime
from collections import defaultdict
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union
import streamlit as st
from config import (
    hoy,
    REGISTROS_DIR,
    ARCHIVO_BASE_REGISTROS,
    HOJAS_REGISTRO,
    OBJETIVOS_VISTA_CONFIG_PATH,
    saldos_iniciales,
)
from domain import Account, Transaction, Budget

from io_data import (
    cargar_objetivos_vista,
    guardar_objetivos_vista,
    fusionar_archivos_registro,
    fusionar_archivos_registro_seguro,
    guardar_historial,
    fondo_reserva_general,
)

from logic import (
    clasificar_ingreso,
    procesar_ingresos,
    calcular_gastos_netos,
    resumen_ingresos,
    resumen_gastos,
    resumen_mensual,
    eliminar_tildes,
    agregar_fila_ingresos,
    cargar_saldos_iniciales,
    cargar_transacciones,
    invertido_en_mes,
    crear_historial_cuentas_virtuales,
    resumen_global,
)

from dateutil.relativedelta import relativedelta


from plots import *







