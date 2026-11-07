from backend.domain.myinvestor_movements import (
    classify,
    extract_contributions,
    extract_fees,
    extract_fund_operations,
)


def mov(fecha, texto, importe, saldo=0.0):
    return {"Fecha Operación": fecha, "Movimiento": texto, "Importe": importe, "Saldo": saldo}


def test_cada_tipo_de_linea_del_extracto_se_reconoce():
    assert classify("COM. GESTION 3T 2025") == "comision"
    assert classify("IVA COM. GESTION 1T 2026") == "comision"
    assert classify("Aportacion automatica cartera") == "aportacion"
    assert classify("PERIODO 05/09/2025 05/10/2025") == "interes"
    assert classify("EFECTIVO-EUR @ 0") == "efectivo"
    assert classify("INDEX MSCI EM IE AC EUR @ 0.14") == "fondo"
    assert classify("VANGUARD US 500 STOCK INDEX EUR") == "fondo"


def test_las_comisiones_salen_en_positivo_con_su_mes():
    """En el extracto son cargos; aguas abajo se reparten como coste y mezclar signos confunde."""
    fees = extract_fees([mov("27/01/2026", "COM. GESTION 4T 2025", -1.37),
                         mov("27/01/2026", "IVA COM. GESTION 4T 2025", -0.29)])

    assert [(f["Mes"], f["Importe"]) for f in fees] == [("2026-01", 1.37), ("2026-01", 0.29)]
    assert all(f["Broker"] == "MyInvestor" for f in fees)


def test_el_iva_de_la_comision_cuenta_como_comision():
    """Es coste real: dejarlo fuera subestimaría lo que cuesta mantener los fondos."""
    assert len(extract_fees([mov("29/04/2026", "IVA COM. GESTION 1T 2026", -0.46)])) == 1


def test_las_aportaciones_se_suman_por_mes():
    aportaciones = extract_contributions([mov("13/10/2025", "Aportacion automatica cartera", 150),
                                          mov("27/10/2025", "Aportacion automatica cartera", 150),
                                          mov("10/11/2025", "Aportacion automatica cartera", 150)])

    assert [(a["Mes"], a["Importe"]) for a in aportaciones] == [("2025-10", 300.0), ("2025-11", 150.0)]


def test_una_compra_de_fondo_separa_el_nombre_de_las_participaciones():
    operaciones = extract_fund_operations([mov("09/10/2025", "INDEX MSCI EM IE AC EUR @ 0.14", -30.09)])

    assert operaciones[0]["Nombre"] == "INDEX MSCI EM IE AC EUR"
    assert operaciones[0]["participaciones"] == 0.14
    assert operaciones[0]["Importe"] == 30.09  # el cargo sale de la cuenta, lo invertido es positivo


def test_una_compra_sin_arroba_conserva_el_importe_en_vez_de_descartarse():
    """El extracto a veces escribe el nombre largo y se queda sin sitio para el '@'."""
    operaciones = extract_fund_operations([mov("15/10/2025", "VANGUARD US 500 STOCK INDEX EUR", -52.14)])

    assert operaciones[0]["participaciones"] is None
    assert operaciones[0]["Importe"] == 52.14


def test_ni_los_intereses_ni_el_efectivo_se_cuelan_como_compra_de_fondo():
    filas = [mov("06/10/2025", "PERIODO 05/09/2025 05/10/2025", 0.04),
             mov("08/10/2025", "EFECTIVO-EUR @ 0", -0.16)]

    assert extract_fund_operations(filas) == []
    assert extract_fees(filas) == []
