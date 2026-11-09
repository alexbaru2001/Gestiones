import pytest

from backend.domain.portfolio_positions import (
    accumulate_costs,
    allocate_fees,
    classify,
    derive_movements,
    merge_movements,
    merge_positions,
    position_value,
    read_positions,
    summarize_load,
)


def pos(mes, broker, isin, nombre, cantidad, precio=0.0, moneda="EUR", valor=None, aportado=None):
    return {
        "Mes": mes, "Broker": broker, "ISIN": isin, "Nombre": nombre,
        "Cantidad": cantidad, "Precio cierre": precio, "Moneda": moneda, "Valor EUR": valor,
        "Aportado EUR": aportado,
    }


# --- Lectura -------------------------------------------------------------------------------------

def test_una_fila_sin_mes_broker_o_isin_no_es_una_posicion():
    filas = [pos("2026-01", "Trade Republic", "DE0007074007", "KWS", 3), pos("", "X", "Y", "Z", 1),
             pos("2026-01", "", "Y", "Z", 1), pos("2026-01", "X", "", "Z", 1)]

    assert len(read_positions(filas)) == 1


def test_el_mes_se_recorta_aunque_venga_con_dia():
    assert read_positions([pos("2026-01-31", "TR", "ISIN", "X", 1)])[0]["Mes"] == "2026-01"


def test_manda_el_valor_del_extracto_y_no_el_producto_de_cantidad_por_precio():
    """Caso real: Iberdrola en agosto de 2026. El precio del extracto viene redondeado."""
    posicion = pos("2026-08", "Trade Republic", "ES0144580Y14", "Iberdrola", 26.994544, 19.81, valor=534.63)

    assert position_value(posicion) == 534.63
    assert round(26.994544 * 19.81, 2) == 534.76  # lo que daría recalcular


def test_sin_valor_en_el_extracto_se_calcula_solo_si_esta_en_euros():
    en_euros = pos("2026-01", "Degiro", "ES0157261019", "ROVI", 3, 62.25, "EUR")
    en_dolares = pos("2025-12", "Degiro", "US7427181091", "PG", 5, 142.885, "USD")

    assert position_value(read_positions([en_euros])[0]) == 186.75
    # Convertir divisa exige un tipo de cambio que el archivo no trae: mejor nada que inventarlo.
    assert position_value(read_positions([en_dolares])[0]) is None


# --- Movimientos deducidos -----------------------------------------------------------------------

def test_una_subida_de_cantidad_es_una_compra_y_una_bajada_una_venta():
    posiciones = [
        pos("2026-01", "TR", "A", "Alfa", 3, 10.0), pos("2026-02", "TR", "A", "Alfa", 5, 10.0),
        pos("2026-03", "TR", "A", "Alfa", 4, 10.0),
    ]

    movimientos = derive_movements(read_positions(posiciones))

    assert [(m["Mes"], m["tipo"], m["cantidad"]) for m in movimientos] == [
        ("2026-02", "compra", 2.0), ("2026-03", "venta", -1.0)]


def test_una_posicion_nueva_se_marca_como_alta_y_una_que_desaparece_como_cierre():
    posiciones = [
        pos("2026-01", "TR", "A", "Alfa", 3, 10.0),
        pos("2026-02", "TR", "A", "Alfa", 3, 10.0), pos("2026-02", "TR", "B", "Beta", 2, 50.0),
        pos("2026-03", "TR", "B", "Beta", 2, 50.0),
    ]

    movimientos = derive_movements(read_positions(posiciones))

    assert ("2026-02", "Beta", "alta") in [(m["Mes"], m["Nombre"], m["tipo"]) for m in movimientos]
    assert ("2026-03", "Alfa", "cierre") in [(m["Mes"], m["Nombre"], m["tipo"]) for m in movimientos]


def test_un_mes_que_solo_trae_un_broker_no_vende_la_cartera_de_los_demas():
    """El caso real: los meses de 2026 solo traen Trade Republic y DeGiro no debe desaparecer."""
    posiciones = [
        pos("2025-12", "Degiro", "ROVI", "ROVI", 3, 60.0), pos("2025-12", "TR", "A", "Alfa", 1, 10.0),
        pos("2026-01", "TR", "A", "Alfa", 2, 10.0),
    ]

    movimientos = derive_movements(read_positions(posiciones))

    assert all(m["Broker"] != "Degiro" for m in movimientos), "DeGiro no estaba en la foto, no se ha vendido"
    assert [(m["Nombre"], m["tipo"]) for m in movimientos] == [("Alfa", "compra")]


def test_una_diferencia_de_redondeo_no_se_cuenta_como_operacion():
    posiciones = [pos("2026-01", "TR", "A", "Alfa", 3.0000000), pos("2026-02", "TR", "A", "Alfa", 3.0000001)]

    assert derive_movements(read_positions(posiciones)) == []


def test_el_importe_estimado_sale_del_precio_de_cierre_del_mes():
    posiciones = [pos("2026-01", "TR", "A", "Alfa", 1, 100.0), pos("2026-02", "TR", "A", "Alfa", 3, 120.0)]

    assert derive_movements(read_positions(posiciones))[0]["importe_estimado"] == 240.0


# --- Reparto de comisiones -----------------------------------------------------------------------

def test_la_comision_se_reparte_entre_las_posiciones_del_broker_a_prorrata_del_valor():
    posiciones = read_positions([
        pos("2026-01", "MyInvestor", "F1", "Fondo 1", 1, valor=750.0),
        pos("2026-01", "MyInvestor", "F2", "Fondo 2", 1, valor=250.0),
    ])

    reparto = allocate_fees(posiciones, [{"Mes": "2026-01", "Broker": "MyInvestor", "Importe": 8.0}])

    assert reparto[("2026-01", "MyInvestor", "F1")] == 6.0
    assert reparto[("2026-01", "MyInvestor", "F2")] == 2.0


def test_la_comision_de_un_broker_no_toca_las_posiciones_de_otro():
    posiciones = read_positions([
        pos("2026-01", "MyInvestor", "F1", "Fondo 1", 1, valor=500.0),
        pos("2026-01", "Degiro", "A", "Alfa", 1, valor=500.0),
    ])

    reparto = allocate_fees(posiciones, [{"Mes": "2026-01", "Broker": "Degiro", "Importe": 2.5}])

    assert reparto == {("2026-01", "Degiro", "A"): 2.5}


def test_varias_comisiones_del_mismo_mes_se_acumulan():
    posiciones = read_positions([pos("2026-01", "MyInvestor", "F1", "Fondo 1", 1, valor=100.0)])
    comisiones = [{"Mes": "2026-01", "Broker": "MyInvestor", "Importe": 2.17},
                  {"Mes": "2026-01", "Broker": "MyInvestor", "Importe": 0.46}]

    assert allocate_fees(posiciones, comisiones)[("2026-01", "MyInvestor", "F1")] == 2.63


def test_una_comision_de_un_mes_sin_posiciones_no_rompe_el_reparto():
    posiciones = read_positions([pos("2026-01", "MyInvestor", "F1", "Fondo 1", 1, valor=100.0)])

    assert allocate_fees(posiciones, [{"Mes": "2026-05", "Broker": "MyInvestor", "Importe": 2.0}]) == {}


# --- Carga incremental ---------------------------------------------------------------------------

def test_subir_dos_veces_el_mismo_archivo_no_duplica_nada():
    filas = [pos("2026-01", "TR", "A", "Alfa", 3), pos("2026-01", "TR", "B", "Beta", 1)]

    assert len(merge_positions(filas, filas)) == 2


def test_un_mes_nuevo_se_añade_y_los_antiguos_se_conservan():
    guardadas = [pos("2026-01", "TR", "A", "Alfa", 3)]
    nuevas = [pos("2026-02", "TR", "A", "Alfa", 5)]

    combinadas = merge_positions(guardadas, nuevas)

    assert [(p["Mes"], p["Cantidad"]) for p in combinadas] == [("2026-01", 3.0), ("2026-02", 5.0)]


def test_un_mes_corregido_sustituye_al_guardado_en_vez_de_duplicarse():
    guardadas = [pos("2026-01", "TR", "A", "Alfa", 3, valor=100.0)]
    nuevas = [pos("2026-01", "TR", "A", "Alfa", 3, valor=123.45)]

    combinadas = merge_positions(guardadas, nuevas)

    assert len(combinadas) == 1
    assert combinadas[0]["Valor EUR"] == 123.45


def test_quitar_una_posicion_de_un_mes_ya_cargado_la_elimina_de_verdad():
    """El mes recién subido manda entero: si no, una posición borrada se quedaría colgando."""
    guardadas = [pos("2026-01", "TR", "A", "Alfa", 3), pos("2026-01", "TR", "B", "Beta", 1)]
    nuevas = [pos("2026-01", "TR", "A", "Alfa", 3)]

    combinadas = merge_positions(guardadas, nuevas)

    assert [p["ISIN"] for p in combinadas] == ["A"]


def test_corregir_un_broker_de_un_mes_no_borra_el_otro_broker_de_ese_mes():
    guardadas = [pos("2026-01", "TR", "A", "Alfa", 3), pos("2026-01", "Degiro", "C", "Gamma", 2)]
    nuevas = [pos("2026-01", "TR", "A", "Alfa", 4)]

    combinadas = merge_positions(guardadas, nuevas)

    assert sorted((p["Broker"], p["Cantidad"]) for p in combinadas) == [("Degiro", 2.0), ("TR", 4.0)]


def mov(fecha, texto, importe, saldo):
    return {"Fecha Operación": fecha, "Movimiento": texto, "Importe": importe, "Saldo": saldo}


def test_los_movimientos_repetidos_entre_exportaciones_no_se_duplican():
    guardados = [mov("06/10/2025", "Aportacion automatica cartera", 150, 159.29)]
    nuevos = [mov("06/10/2025", "Aportacion automatica cartera", 150, 159.29),
              mov("13/10/2026", "COM. GESTION 3T 2026", -2.91, 12.0)]

    assert len(merge_movements(guardados, nuevos)) == 2


def test_dos_movimientos_identicos_del_mismo_dia_no_se_colapsan_en_uno():
    """El saldo corriente es lo que los distingue: sin él, una compra repetida se perdería."""
    nuevos = [mov("16/10/2025", "INDEX MSCI EM IE AC EUR @ 0.11", -23.87, 38.56),
              mov("16/10/2025", "INDEX MSCI EM IE AC EUR @ 0.11", -23.87, 14.69)]

    assert len(merge_movements([], nuevos)) == 2


def test_el_resumen_dice_que_meses_entran_y_cuales_se_actualizan():
    guardadas = [pos("2026-01", "TR", "A", "Alfa", 3)]
    nuevas = [pos("2026-01", "TR", "A", "Alfa", 4), pos("2026-02", "TR", "A", "Alfa", 5)]
    combinadas = merge_positions(guardadas, nuevas)

    resumen = summarize_load(guardadas, combinadas, [], [], nuevas)

    assert resumen["meses_nuevos"] == ["2026-02"]
    assert resumen["meses_actualizados"] == ["2026-01 (TR)"]
    assert resumen["posiciones"] == 2
    assert resumen["posiciones_nuevas"] == 1


def test_anadir_un_mes_no_dice_haber_actualizado_todo_el_historico():
    """Cargar solo octubre listaba como actualizados los 14 meses ya guardados."""
    guardadas = [
        pos("2026-01", "TR", "A", "Alfa", 3),
        pos("2026-02", "TR", "A", "Alfa", 4),
        pos("2026-02", "MI", "B", "Beta", 9),
    ]
    nuevas = [pos("2026-03", "TR", "A", "Alfa", 5)]

    resumen = summarize_load(guardadas, merge_positions(guardadas, nuevas), [], [], nuevas)

    assert resumen["meses_nuevos"] == ["2026-03"]
    assert resumen["meses_actualizados"] == []
    assert resumen["posiciones_nuevas"] == 1


def test_un_mes_nuevo_para_un_broker_y_repetido_para_otro_se_separan():
    guardadas = [pos("2026-02", "TR", "A", "Alfa", 3)]
    nuevas = [pos("2026-02", "TR", "A", "Alfa", 6), pos("2026-02", "MI", "B", "Beta", 2)]

    resumen = summarize_load(guardadas, merge_positions(guardadas, nuevas), [], [], nuevas)

    # El mes ya existía para TR, pero para MI entra por primera vez.
    assert resumen["meses_nuevos"] == ["2026-02"]
    assert resumen["meses_actualizados"] == ["2026-02 (TR)"]


# --- Coste acumulado ------------------------------------------------------------------------------

def test_el_coste_se_acumula_con_cada_compra():
    posiciones = read_positions([
        pos("2026-01", "TR", "A", "Alfa", 2, 10.0),
        pos("2026-02", "TR", "A", "Alfa", 5, 12.0),
    ])

    costes = accumulate_costs(posiciones)

    assert costes[("2026-01", "TR", "A")] == 20.0            # 2 × 10
    assert costes[("2026-02", "TR", "A")] == 56.0            # 20 + 3 × 12


def test_mantener_la_posicion_no_cambia_el_coste_aunque_suba_el_precio():
    posiciones = read_positions([
        pos("2026-01", "TR", "A", "Alfa", 2, 10.0),
        pos("2026-02", "TR", "A", "Alfa", 2, 50.0),
    ])

    costes = accumulate_costs(posiciones)

    assert costes[("2026-02", "TR", "A")] == 20.0


def test_una_venta_parcial_retira_coste_medio_no_valor_de_mercado():
    posiciones = read_positions([
        pos("2026-01", "TR", "A", "Alfa", 4, 10.0),   # coste 40
        pos("2026-02", "TR", "A", "Alfa", 2, 80.0),   # vende la mitad
    ])

    costes = accumulate_costs(posiciones)

    assert costes[("2026-02", "TR", "A")] == 20.0, "la mitad del coste, no la mitad del valor"


def test_sin_precio_en_el_archivo_se_usa_el_que_den_de_fuera():
    """Los fondos no traen precio: el valor liquidativo lo pone el mercado."""
    posiciones = read_positions([pos("2026-01", "MyInvestor", "F1", "Fondo", 10)])

    costes = accumulate_costs(posiciones, price_of=lambda isin, mes: 25.0)

    assert costes[("2026-01", "MyInvestor", "F1")] == 250.0


def test_sin_precio_de_ninguna_fuente_el_coste_se_queda_a_cero_en_vez_de_romper():
    posiciones = read_positions([pos("2026-01", "MyInvestor", "F1", "Fondo", 10)])

    assert accumulate_costs(posiciones)[("2026-01", "MyInvestor", "F1")] == 0.0


def test_lo_aportado_manda_sobre_el_precio_de_cierre():
    """El archivo dice que se pagaron 300 €; valorar la compra al cierre daría 200 € y mentiría."""
    posiciones = read_positions([pos("2026-01", "TR", "A", "Alfa", 10, 20.0, aportado=300.0)])

    assert accumulate_costs(posiciones)[("2026-01", "TR", "A")] == 300.0


def test_una_posicion_heredada_no_arrastra_el_valor_del_primer_mes_como_coste():
    """El caso que motivó la columna: Repsol ya estaba comprada antes de que empiece el archivo.

    Sin lo aportado, su coste sería su cotización de enero y la revalorización posterior se leería
    entera como ganancia del inversor.
    """
    posiciones = read_positions([
        pos("2026-01", "TR", "R", "Repsol", 6, 16.44, aportado=99.54),
        pos("2026-09", "TR", "R", "Repsol", 6, 29.75, aportado=0.0),
    ])

    costes = accumulate_costs(posiciones)

    assert costes[("2026-09", "TR", "R")] == 99.54


def test_un_mes_sin_aportacion_no_mueve_el_coste_aunque_crezca_la_cantidad():
    """Un scrip dividend entrega títulos sin poner dinero: 0 € aportados es un dato, no un hueco."""
    posiciones = read_positions([
        pos("2026-01", "TR", "I", "Iberdrola", 6, 20.0, aportado=120.0),
        pos("2026-02", "TR", "I", "Iberdrola", 7, 20.0, aportado=0.0),
    ])

    assert accumulate_costs(posiciones)[("2026-02", "TR", "I")] == 120.0


def test_sin_la_columna_se_sigue_deduciendo_el_coste_de_los_meses():
    """Los fondos y DeGiro no traen aportaciones: el camino viejo tiene que seguir funcionando."""
    posiciones = read_positions([
        pos("2026-01", "Degiro", "A", "Alfa", 2, 10.0),
        pos("2026-02", "Degiro", "A", "Alfa", 5, 12.0),
    ])

    assert accumulate_costs(posiciones)[("2026-02", "Degiro", "A")] == 56.0


def test_el_coste_de_cada_broker_va_por_su_cuenta():
    posiciones = read_positions([
        pos("2026-01", "TR", "A", "Alfa", 1, 10.0),
        pos("2026-01", "Degiro", "A", "Alfa", 1, 50.0),
    ])

    costes = accumulate_costs(posiciones)

    assert costes[("2026-01", "TR", "A")] == 10.0
    assert costes[("2026-01", "Degiro", "A")] == 50.0


# --- Clasificación --------------------------------------------------------------------------------

CATALOGO = {
    "ES0144580Y14": {"region": "Europa", "sector": "Utilities", "focus": "Dividendos"},
    "IE0032126645": {"region": "Norteamerica", "sector": "Renta variable diversificada", "focus": "Indexados"},
}


def test_una_accion_toma_region_sector_y_foco_del_catalogo():
    clase = classify(pos("2026-01", "TR", "ES0144580Y14", "Iberdrola", 6), CATALOGO)

    assert clase == {"region": "Europa", "sector": "Utilities", "foco": "Dividendos", "tipo": "stock"}


def test_un_isin_del_mapa_de_fondos_se_marca_como_fondo():
    clase = classify(pos("2026-01", "MyInvestor", "IE0032126645", "Vanguard US 500", 10), CATALOGO, {"IE0032126645"})

    assert clase["tipo"] == "fund"


def test_bitcoin_se_reconoce_por_el_nombre_porque_no_trae_isin():
    """En el archivo se apunta como 'BTC', que no está en el catálogo de ISIN."""
    clase = classify(pos("2026-01", "Degiro", "BTC", "Bitcoin", 0.005), CATALOGO)

    assert clase["tipo"] == "crypto"
    assert clase["sector"] == "Criptoactivo"


def test_un_activo_desconocido_queda_sin_clasificar_en_vez_de_romper():
    clase = classify(pos("2026-01", "TR", "XX0000000000", "Nuevo", 1), CATALOGO)

    assert clase == {"region": "Sin clasificar", "sector": "Sin clasificar", "foco": "Sin clasificar", "tipo": "stock"}
