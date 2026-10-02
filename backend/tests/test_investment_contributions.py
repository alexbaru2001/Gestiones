from backend.domain.investment_contributions import build_investment_contributions, merge_income_rows


def summary(rows):
    return [
        {"Mes": mes, "Dinero Invertido": invertido, "🧾 Presupuesto Disponible": disponible, "💳 Gasto del mes": gasto}
        for mes, invertido, disponible, gasto in rows
    ]


def test_la_parte_del_sueldo_es_el_porcentaje_configurado():
    rows = build_investment_contributions(
        summary([("2025-01", 0.0, 0.0, 0.0)]),
        [{"Mes": "2025-01", "cantidad": 2000.0, "tipo_logico": "Ingreso Real", "etiquetas": ""}],
        0.1,
    )

    assert rows[0]["sueldo"] == 200.0


def test_los_dividendos_se_separan_de_los_demas_intereses():
    """Ambos son 'Rendimiento Financiero': sin separarlos por etiqueta se contarían dos veces."""
    rows = build_investment_contributions(
        summary([("2025-01", 0.0, 0.0, 0.0)]),
        [
            {"Mes": "2025-01", "cantidad": 30.0, "tipo_logico": "Rendimiento Financiero", "etiquetas": "Interes"},
            {"Mes": "2025-01", "cantidad": 12.0, "tipo_logico": "Rendimiento Financiero", "etiquetas": "Dividendos, IB"},
        ],
    )

    assert rows[0]["intereses"] == 30.0
    assert rows[0]["dividendos"] == 12.0


def test_la_etiqueta_de_dividendos_se_reconoce_sin_tildes_ni_mayusculas():
    rows = build_investment_contributions(
        summary([("2025-01", 0.0, 0.0, 0.0)]),
        [{"Mes": "2025-01", "cantidad": 5.0, "tipo_logico": "rendimiento financiero", "etiquetas": "DIVIDENDOS"}],
    )

    assert rows[0]["dividendos"] == 5.0


def test_el_extra_por_ahorrar_es_un_tercio_del_sobrante_que_ya_trae_el_resumen():
    """La columna "🧾 Presupuesto Disponible" guarda `presupuesto_efectivo`, que YA es lo que sobra
    tras el gasto (logic.py:814): restarle otra vez el gasto dejaba el extra casi siempre a cero."""
    rows = build_investment_contributions(summary([("2025-01", 0.0, 600.0, 300.0)]), [])

    assert rows[0]["extra"] == 200.0


def test_sin_sobrante_no_hay_extra():
    rows = build_investment_contributions(summary([("2025-01", 0.0, 0.0, 900.0)]), [])

    assert rows[0]["extra"] == 0.0


def test_lo_invertido_del_mes_es_lo_que_sube_el_acumulado():
    rows = build_investment_contributions(
        summary([("2025-01", 1000.0, 0.0, 0.0), ("2025-02", 1450.0, 0.0, 0.0), ("2025-03", 1450.0, 0.0, 0.0)]), []
    )

    assert [row["invertido"] for row in rows] == [1000.0, 450.0, 0.0]


def test_el_desglose_cuadra_con_el_movimiento_de_la_bolsa():
    """La bolsa sube lo aportado y baja lo invertido: esta igualdad es la que valida el cálculo."""
    rows = build_investment_contributions(
        summary([("2025-01", 0.0, 600.0, 300.0), ("2025-02", 500.0, 0.0, 0.0)]),
        [
            {"Mes": "2025-01", "cantidad": 2000.0, "tipo_logico": "Ingreso Real", "etiquetas": ""},
            {"Mes": "2025-01", "cantidad": 50.0, "tipo_logico": "Rendimiento Financiero", "etiquetas": "Interes"},
            {"Mes": "2025-02", "cantidad": 1800.0, "tipo_logico": "Ingreso Real", "etiquetas": ""},
            {"Mes": "2025-02", "cantidad": 20.0, "tipo_logico": "Rendimiento Financiero", "etiquetas": "Dividendos"},
        ],
        0.1,
    )

    enero, febrero = rows
    # 200 € de sueldo + 50 € de intereses + 200 € de sobrante (600/3) = 450 €, sin nada invertido.
    assert enero["sueldo"] + enero["intereses"] + enero["dividendos"] + enero["extra"] - enero["invertido"] == 450.0
    assert febrero["sueldo"] + febrero["intereses"] + febrero["dividendos"] + febrero["extra"] - febrero["invertido"] == -300.0


def test_un_ingreso_sin_mes_se_puede_fechar_por_su_fecha():
    rows = build_investment_contributions(
        summary([("2025-01", 0.0, 0.0, 0.0)]),
        [{"fecha": "2025-01-14", "cantidad": 1000.0, "tipo_logico": "Ingreso Real"}],
        0.1,
    )

    assert rows[0]["sueldo"] == 100.0


def test_un_mes_sin_ingresos_no_inventa_aportaciones():
    rows = build_investment_contributions(summary([("2025-01", 0.0, 0.0, 0.0)]), [])

    assert rows[0] == {"Mes": "2025-01", "sueldo": 0.0, "intereses": 0.0, "dividendos": 0.0, "extra": 0.0, "invertido": 0.0}


def test_sin_resumen_no_hay_desglose():
    assert build_investment_contributions([], [{"Mes": "2025-01", "cantidad": 10.0, "tipo_logico": "Ingreso Real"}]) == []


def test_un_mes_presente_en_el_historico_y_en_el_tramo_no_se_cuenta_dos_veces():
    """Al reprocesar un Excel que ya incluía meses guardados, el desglose salía al doble."""
    historico = [{"Mes": "2025-01", "cantidad": 2000.0, "tipo_logico": "Ingreso Real"}]
    tramo = [{"Mes": "2025-01", "cantidad": 2000.0, "tipo_logico": "Ingreso Real"}]

    rows = build_investment_contributions(summary([("2025-01", 0.0, 0.0, 0.0)]), merge_income_rows(historico, tramo), 0.1)

    assert rows[0]["sueldo"] == 200.0


def test_los_meses_que_solo_estan_en_el_historico_se_conservan():
    historico = [{"Mes": "2024-12", "cantidad": 1000.0, "tipo_logico": "Ingreso Real"}]
    tramo = [{"Mes": "2025-01", "cantidad": 2000.0, "tipo_logico": "Ingreso Real"}]

    merged = merge_income_rows(historico, tramo)

    assert {row["Mes"] for row in merged} == {"2024-12", "2025-01"}


def test_el_extra_y_la_deuda_acumulada_no_pueden_coexistir_en_el_mismo_mes():
    """Regla de `logic.py`: el presupuesto disponible es `max(0, presupuesto - deuda arrastrada)`,
    así que para que sobre algo el presupuesto ha tenido que cubrir el gasto DESPUÉS de pagar la
    deuda; en ese caso el mes cierra con deuda cero. Si una gráfica enseña extra y deuda a la vez
    en el mismo mes, el fallo está en el cálculo, no en los datos."""
    con_deuda = build_investment_contributions(
        [
            {
                "Mes": "2025-06",
                "Dinero Invertido": 0.0,
                "🧾 Presupuesto Disponible": 0.0,  # no sobró nada: la deuda se comió el presupuesto
                "📉 Deuda Presupuestaria acumulada": 1140.89,
            }
        ],
        [],
    )
    con_sobrante = build_investment_contributions(
        [
            {
                "Mes": "2025-07",
                "Dinero Invertido": 0.0,
                "🧾 Presupuesto Disponible": 300.0,
                "📉 Deuda Presupuestaria acumulada": 0.0,
            }
        ],
        [],
    )

    assert con_deuda[0]["extra"] == 0.0
    assert con_sobrante[0]["extra"] == 100.0
