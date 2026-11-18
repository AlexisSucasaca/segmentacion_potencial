"""Tabla de segmentación de la capa gold: une las 3 segmentaciones (transaccional,
estratégica y potencial) por cliente-periodo, aplica las reglas de negocio y la inserta en
BigQuery.

Reproduce el notebook `notebook.ipynb` ("Carga carga de toda la información" → "Tratamiento
de la info" → "Carga a capa gold"), con estas diferencias deliberadas, todas pedidas
explícitamente:

1. La clave de la tabla es (`cod_interlocutor`, `periodo`) — un cliente puede tener una
   fila por cada periodo cargado (es una tabla de HISTORIAL, no "estado actual por
   cliente"). Esto se confirmó después de un incidente en producción: una versión anterior
   de este módulo asumía `cod_interlocutor` como clave única de toda la tabla y pisaba,
   con cada corrida, el `periodo` de clientes que en realidad pertenecían a un periodo
   histórico ya cerrado.
2. La carga a un periodo que todavía no existe en gold es un `INSERT INTO ... SELECT` desde una
   tabla temporal del sandbox (no un *load job* directo a prod, que pide permiso para crear
   tablas) y no hace falta `MERGE`: con la clave (cliente, periodo), un periodo nuevo nunca puede chocar con
   una fila existente). **Nunca se toca un periodo que ya está cargado** — si `PERIODO` ya
   existe en gold, `cargar_periodo_gold` se niega a insertar y hay que decidir a mano qué
   hacer (por ejemplo, borrar ese periodo primero si de verdad hay que recargarlo).
3. Se quitó el parche puntual para el cliente `1265373` del notebook original (un
   `drop_duplicates` hardcodeado para un solo id): `aplicar_reglas_negocio` ya deduplica
   por cliente dentro de un mismo periodo de forma genérica.
4. Una `des_segmento_estrategica` VACÍA (NaN, '' o 'Sin asignar') queda como **"Por definir"**
   en todas las bocas (incluidas PA, GA y LV). Antes se rellenaba con un valor por defecto de la
   boca (Clásica, Abarrotero, Otros Gastronomía, ...; en PA, ''), lo que tapaba el hueco y hacía
   imposible distinguir "sin segmento" de "segmento Clásica". Esos defaults solo se usan ahora
   para valores INVÁLIDOS (no vacíos) de la boca.
5. El reemplazo de la boca `GO` por su boca real se hace ANTES de las reglas por boca (antes se
   hacía después de la regla de AB/BO/MM, y un ex-GO que pasaba a BO/AB/MM con segmento vacío
   no recibía ninguna regla).
6. Los clientes con `cod_boca_salida == "Sin Asignar"` ya no se eliminan en silencio: se
   conservan y salen en `validar_combinaciones` (`conservar_sin_asignar=False` reproduce el
   filtro anterior).
7. Al deduplicar por cliente se ordena antes (`_desempatar_duplicados`), así que el registro que
   se conserva es siempre el mismo y no depende del orden de los joins.

Dos modos de ejecución (`armar_df_final` + `aplicar_reglas_negocio` + `preparar_carga_periodo`
+ `cargar_periodo_gold`):

completo
    Inserta la fila fresca de cada cliente tal cual sale del cálculo de este periodo, sin
    mirar el historial.

incremental
    La base es el gold del periodo INMEDIATAMENTE anterior (no se busca más atrás). El
    emparejamiento es SOLO por `cod_interlocutor`; el cálculo fresco de este periodo aporta
    únicamente lo que el periodo anterior no tenía resuelto:

    - Cliente del periodo anterior que hoy no viene en el cálculo fresco: se arrastra su fila
      anterior (nadie sale).
    - Cliente NUEVO (su código no estaba): se agrega con el cálculo fresco. Si su transaccional
      es "Por definir", queda sin estratégica ("Por definir") ni potencial (NULL).
    - Cliente que ya estaba pero con OTRA boca: se desestima el cálculo fresco y conserva su
      fila anterior (boca, transaccional, estratégica y potencial).
    - Cliente que ya estaba con la misma boca: solo cambia lo que estaba "Por definir". Si ya
      tenía segmento, se conservan transaccional, estratégica y potencial (incluido
      `flg_potential = False`), aunque el cálculo fresco diga otra cosa: nunca regresa a "Por
      definir". Si estaba "Por definir" y hoy tiene segmento, toma los valores frescos (es el
      único caso en que cambia la potencialidad); si sigue "Por definir", se queda así.

    `validar_incremental` cuenta cada uno de estos grupos para revisarlo antes de insertar.

En los dos modos se aplica siempre, al final, el invariante de negocio: si
`flg_potential = False`, `des_potencial_predic` tiene que ser `NULL` (no puede haber una
etiqueta de potencial para un cliente que no es potencial). Es un chequeo estructural, no
un parche puntual — cualquier combinación de fuentes que lo viole se corrige acá.
"""

from __future__ import annotations

import calendar
import warnings
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

UMBRAL_PROBA_POTENCIAL_DEFAULT = 0.65

# (cod_boca_salida, des_segmento_estrategica) válidos, más 'Por definir' por boca (celda 44
# del notebook original). '' representa vacío (panadería sin segmento).
COMBINACIONES_VALIDAS = {
    ("AB", "Clásica"), ("AB", "Por definir"),
    ("BO", "Alejada"), ("BO", "Clásica"), ("BO", "Estrella"), ("BO", "Golosinera"), ("BO", "Por definir"),
    ("GA", "Cebichería"), ("GA", "Chifa"), ("GA", "Otros Gastronomía"), ("GA", "Comida Casera Criolla"),
    ("GA", "Pollería"), ("GA", "Sanguchería"), ("GA", "Por definir"),
    ("LV", "Hotel/Casino"), ("LV", "Lavanderia"), ("LV", "Otros Limpieza"), ("LV", "Por definir"),
    ("MM", "Clásica"), ("MM", "Estrella"), ("MM", "Golosinera"), ("MM", "Alejada"), ("MM", "Por definir"),
    ("PA", ""), ("PA", "Estadio 1"), ("PA", "Estadio 2"), ("PA", "Estadio 3"), ("PA", "Estadio 4"), ("PA", "Por definir"),
    ("PM", "Abarrotero"), ("PM", "Confitero"), ("PM", "Lavandero"), ("PM", "Por definir"),
}

COLUMNAS_GOLD = [
    "periodo", "cod_interlocutor", "cod_boca_salida", "des_segmento_transaccional",
    "des_segmento_estrategica", "flg_encuesta", "flg_mercado_potencial", "flg_potential",
    "des_potencial_predic", "des_oficina_venta", "fec_proceso",
]
# Las 3 segmentaciones que puede faltarle a un cliente (para decidir qué completar en modo incremental).
COLUMNAS_SEGMENTACION = ["des_segmento_estrategica", "flg_potential", "des_potencial_predic"]
# Para decidir si a un cliente "le falta algo", solo importan des_segmento_estrategica y
# flg_potential: des_potencial_predic en NULL con flg_potential=False es un estado válido
# (sin potencial no hay etiqueta que predecir), no un dato pendiente de completar.


def ultimo_dia_mes(periodo) -> pd.Timestamp:
    """El último día del mes de `periodo` (fec_proceso = último día del mes de periodo)."""
    p = pd.Timestamp(periodo)
    return pd.Timestamp(p.year, p.month, calendar.monthrange(p.year, p.month)[1])


def union_potencial_clientes(
    salidas_vertical: Dict[str, pd.DataFrame], umbral_proba: float = UMBRAL_PROBA_POTENCIAL_DEFAULT,
) -> pd.DataFrame:
    """Une las hojas "Potencial Estimaciones" de lavandería/hoteles/panadería/gastro/bodegas.

    `salidas_vertical`: {nombre_vertical: hoja de Excel ya leída}. Se queda con las filas
    con `proba_potential > umbral_proba` (igual que el notebook original).
    """
    partes = []
    for df in salidas_vertical.values():
        faltan = [c for c in ("Cod Cliente DEX", "predicted", "flag_potential", "proba_potential") if c not in df.columns]
        if faltan:
            raise KeyError(f"Faltan columnas en una salida de vertical: {faltan}")
        partes.append(df.loc[df["proba_potential"] > umbral_proba,
                             ["Cod Cliente DEX", "predicted", "flag_potential", "proba_potential"]])
    out = pd.concat(partes, ignore_index=True).drop_duplicates().rename(
        columns={"Cod Cliente DEX": "cod_cliente_dex", "predicted": "des_potencial_predic"}
    )
    out["flag_potential"] = True
    return out.reset_index(drop=True)


def union_potencial_mercado_pdm(hojas_pdm: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Une las 3 hojas del Excel final de PDM (Abarroteros, Lavanderos, Impulso)."""
    partes = []
    for df in hojas_pdm.values():
        if not {"cod_cliente_dex", "Potencial"} <= set(df.columns):
            raise KeyError("Cada hoja de PDM debe traer 'cod_cliente_dex' y 'Potencial'.")
        partes.append(df[["cod_cliente_dex", "Potencial"]])
    out = pd.concat(partes, ignore_index=True).drop_duplicates().rename(columns={"Potencial": "des_potencial_predic"})
    out["flag_potential"] = out["des_potencial_predic"] != "No Potencial"
    return out.reset_index(drop=True)


def combinar_potencial(potencial_clientes: pd.DataFrame, potencial_mercado_pdm: pd.DataFrame) -> pd.DataFrame:
    """Junta el potencial de cliente (5 verticales) con el de PDM (PM), sin duplicar filas."""
    return pd.concat([potencial_clientes, potencial_mercado_pdm], ignore_index=True).drop_duplicates().reset_index(drop=True)


def flag_mercado_potencial(ids_mercado: pd.DataFrame, potencial_pm: pd.DataFrame) -> pd.DataFrame:
    """`flg_mercado_potencial` por cliente: si el mercado de PM al que pertenece es potencial.

    `ids_mercado`: hoja "IDs mercado" de PM (`Cod Cliente DEX`, `mercado_id`).
    `potencial_pm`: hoja "Potencial" de PM (`mercado_id`, `flag_potential`).
    """
    out = ids_mercado.merge(potencial_pm[["mercado_id", "flag_potential"]], on="mercado_id", how="left")
    out["flag_potential"] = out["flag_potential"].fillna(False)
    return out.rename(columns={"Cod Cliente DEX": "cod_cliente_dex", "flag_potential": "flg_mercado_potencial"})


def union_encuestas(encuestas_por_vertical: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Un cliente tiene `flg_encuesta = True` si aparece en cualquiera de las encuestas."""
    partes = []
    for df in encuestas_por_vertical.values():
        if "external_store_id" not in df.columns:
            raise KeyError("Cada encuesta debe traer 'external_store_id'.")
        partes.append(df[["external_store_id"]])
    out = pd.concat(partes, ignore_index=True).drop_duplicates().reset_index(drop=True)
    out = out.rename(columns={"external_store_id": "cod_cliente_dex"})
    out["flg_encuesta"] = True
    return out


def armar_df_final(
    df_tranx: pd.DataFrame, df_estrategico: pd.DataFrame, df_encuestas: pd.DataFrame,
    df_potencial_mercado: pd.DataFrame, df_potencial: pd.DataFrame, last_values: pd.DataFrame,
    periodo,
) -> pd.DataFrame:
    """Universo = `df_tranx`; todo lo demás se cruza con LEFT JOIN por `cod_cliente_dex`.

    `df_estrategico` trae `id_interlocutor_origen` (se renombra a `cod_cliente_dex`),
    `segmento` (-> `des_segmentacion_estrategica`) y `cod_boca_salida` (-> `cod_boca_salida_vs2`,
    para la Regla 2 de `aplicar_reglas_negocio`). `last_values` trae `cod_cliente_dex` y
    `des_oficina_venta_alicorp`.
    """
    est = df_estrategico.rename(columns={
        "id_interlocutor_origen": "cod_cliente_dex", "segmento": "des_segmentacion_estrategica",
        "cod_boca_salida": "cod_boca_salida_vs2",
    })
    lv = last_values.copy()
    lv["cod_cliente_dex"] = pd.to_numeric(lv["cod_cliente_dex"], errors="coerce").astype("Int64")
    lv = lv[["cod_cliente_dex", "des_oficina_venta_alicorp"]]

    df = df_tranx.merge(est, on="cod_cliente_dex", how="left")
    df = df.merge(df_encuestas, on="cod_cliente_dex", how="left")
    df["flg_encuesta"] = df["flg_encuesta"].fillna(False)
    df = df.merge(df_potencial_mercado[["cod_cliente_dex", "flg_mercado_potencial"]], on="cod_cliente_dex", how="left")
    df["flg_mercado_potencial"] = df["flg_mercado_potencial"].fillna(False)
    df = df.merge(df_potencial, on="cod_cliente_dex", how="left")
    df["flag_potential"] = df["flag_potential"].fillna(False)
    df = df.merge(lv, on="cod_cliente_dex", how="left")

    df["periodo"] = pd.Timestamp(periodo)
    df["fec_corrida"] = ultimo_dia_mes(periodo)
    return df


POR_DEFINIR = "Por definir"
# Textos que cuentan como "sin segmento" (se comparan en minúsculas y sin espacios en los bordes).
_TEXTOS_PENDIENTES = {"", "por definir", "sin asignar"}

# des_segmento_estrategica admitidos por boca; cualquier otro valor NO vacío es inválido.
_VALIDOS_AB_BO_MM = ["Alejada", "Clásica", "Estrella"]
VALIDOS_POR_BOCA = {
    "AB": _VALIDOS_AB_BO_MM, "BO": _VALIDOS_AB_BO_MM, "MM": _VALIDOS_AB_BO_MM,
    "PM": ["Abarrotero", "Confitero", "Lavandero"],
    "PA": ["Estadio 1", "Estadio 2", "Estadio 3", "Estadio 4"],
    "LV": ["Hotel/Casino", "Lavanderia", "Otros Limpieza"],
    "GA": ["Cebichería", "Chifa", "Clásica", "Comida Casera Criolla", "Otros Gastronomía",
           "Pollería", "Sanguchería"],
}
# A qué se reasigna un valor INVÁLIDO (no vacío). Un valor vacío NUNCA recibe estos defaults:
# queda "Por definir".
DEFAULT_INVALIDO_POR_BOCA = {
    "AB": "Clásica", "BO": "Clásica", "MM": "Clásica", "PM": "Abarrotero",
    "PA": POR_DEFINIR, "LV": "Otros Limpieza", "GA": "Otros Gastronomía",
}


def _es_pendiente(serie: pd.Series) -> pd.Series:
    """True donde el valor es vacío / NaN / 'Por definir' / 'Sin asignar' (sin segmento)."""
    texto = serie.astype("string").str.strip().str.lower()
    return (texto.isna() | texto.isin(_TEXTOS_PENDIENTES)).fillna(False).astype(bool)


def _reglas_boca_salida(df: pd.DataFrame) -> pd.DataFrame:
    """Reglas 1-6: cada `cod_boca_salida` solo admite ciertos `des_segmentacion_estrategica`.

    1. `GO` se reemplaza por su boca real (`cod_boca_salida_vs2`) ANTES de todo lo demás, para
       que el ex-GO pase por las reglas de su boca real.
    2. Un segmento vacío (NaN, '', 'Sin asignar') queda "Por definir" en TODAS las bocas.
    3. Un segmento no vacío que la boca no admite (p. ej. 'Estadio 9' en PA) pasa al default de
       su boca (`DEFAULT_INVALIDO_POR_BOCA`).
    """
    df = df.copy()
    m = df["cod_boca_salida"] == "GO"
    df.loc[m, "cod_boca_salida"] = df.loc[m, "cod_boca_salida_vs2"]
    df = df.drop(columns=["cod_boca_salida_vs2"], errors="ignore")

    col = "des_segmentacion_estrategica"
    df[col] = df[col].astype(object)
    pendiente = _es_pendiente(df[col])
    df.loc[pendiente, col] = POR_DEFINIR

    for boca, validos in VALIDOS_POR_BOCA.items():
        invalido = (df["cod_boca_salida"] == boca) & ~pendiente & ~df[col].isin(validos)
        df.loc[invalido, col] = DEFAULT_INVALIDO_POR_BOCA[boca]
    return df


def _convertir_booleano(serie: pd.Series) -> pd.Series:
    return serie.replace({"FALSO": False, "VERDADERO": True, "False": False, "True": True})


def _reglas_potencial_por_encuesta(df: pd.DataFrame) -> pd.DataFrame:
    """Reglas 1-4 de validación potencial/encuesta/PM (celda 30)."""
    df = df.copy()
    df["flag_potential"] = df["flag_potential"].astype(object)  # admite NaN/True/False mezclados
    if df["flag_potential"].dtype == object:
        df["flag_potential"] = _convertir_booleano(df["flag_potential"])
    if df["flg_mercado_potencial"].dtype == object:
        df["flg_mercado_potencial"] = _convertir_booleano(df["flg_mercado_potencial"])

    no_pm = df["cod_boca_salida"] != "PM"
    df.loc[(df["flg_encuesta"] == False) & no_pm, "flag_potential"] = np.nan  # noqa: E712
    df.loc[(df["flg_encuesta"] == True) & no_pm & (df["flag_potential"] == False), "des_potencial_predic"] = np.nan  # noqa: E712

    validos = ["High", "Low", "Medium"]
    cond_pm_true = (df["cod_boca_salida"] == "PM") & (df["flag_potential"] == True)  # noqa: E712
    invalido = cond_pm_true & (df["des_potencial_predic"].isna() | ~df["des_potencial_predic"].isin(validos))
    df.loc[invalido, ["des_potencial_predic", "flag_potential"]] = [np.nan, False]

    df.loc[(df["cod_boca_salida"] == "PM") & (df["flag_potential"] == False), "des_potencial_predic"] = np.nan  # noqa: E712
    return df


def _reglas_por_definir(df: pd.DataFrame, conservar_sin_asignar: bool = True) -> pd.DataFrame:
    """"Sin asignar"/"Por definir" (celdas 31, 32, 34, 35)."""
    df = df.copy()
    df["des_segmentacion_transaccional"] = df["des_segmentacion_transaccional"].replace("Sin asignar", "Por definir")

    cond1 = (df["des_segmentacion_transaccional"] == "Por definir") & df["cod_boca_salida"].isin(["BO", "AB", "MM", "PM"])
    df["des_segmentacion_estrategica"] = np.where(cond1, "Por definir", df["des_segmentacion_estrategica"])

    cond2 = (df["cod_boca_salida"] == "GA") & (df["des_segmentacion_estrategica"] == "Clásica")
    df["des_segmentacion_estrategica"] = np.where(cond2, "Otros Gastronomía", df["des_segmentacion_estrategica"])

    if not conservar_sin_asignar:
        df = df[df["cod_boca_salida"] != "Sin Asignar"].copy()

    df["flag_potential"] = np.where(
        df["des_segmentacion_transaccional"] == "Por definir", np.nan, df["flag_potential"].fillna(False)
    )
    df["flag_potential"] = df["flag_potential"].astype("boolean")
    return df


def validar_combinaciones(
    df: pd.DataFrame, boca_col: str = "cod_boca_salida", seg_col: str = "des_segmento_estrategica",
) -> pd.DataFrame:
    """Filas cuya (boca_salida, segmento) no está en `COMBINACIONES_VALIDAS` — para revisar, no se tocan."""
    combo = list(zip(df[boca_col], df[seg_col].fillna("")))
    invalidas = ~pd.Series(combo, index=df.index).isin(COMBINACIONES_VALIDAS)
    return df.loc[invalidas, [boca_col, seg_col]]


_RANGO_POTENCIAL = {"High": 0, "Medium": 1, "Low": 2}


def _desempatar_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    """Deja UNA fila por `cod_interlocutor`, siempre la misma para los mismos datos.

    Un cliente puede repetirse (p. ej. sale en dos verticales con distinto potencial). Se
    conserva primero el que es potencial (`flg_potential` True), luego el de mayor etiqueta
    (High > Medium > Low) y, a igualdad, el de menor valor alfabético en las demás columnas,
    para que el resultado no dependa del orden de los joins.
    """
    tmp = df.assign(
        _r_flg=(~df["flg_potential"].fillna(False).astype(bool)).astype(int),
        _r_pred=df["des_potencial_predic"].map(_RANGO_POTENCIAL).fillna(len(_RANGO_POTENCIAL)),
    )
    claves = ["cod_interlocutor", "_r_flg", "_r_pred", "des_potencial_predic",
              "des_segmento_estrategica", "des_segmento_transaccional", "cod_boca_salida"]
    tmp = tmp.sort_values(claves, kind="mergesort", na_position="last")
    tmp = tmp.drop_duplicates(subset=["cod_interlocutor"], keep="first")
    return tmp.drop(columns=["_r_flg", "_r_pred"]).reset_index(drop=True)


def aplicar_reglas_negocio(df_final: pd.DataFrame, conservar_sin_asignar: bool = True) -> pd.DataFrame:
    """Todas las reglas de negocio + limpieza + el renombre final a los nombres de la tabla gold.

    `conservar_sin_asignar=True` (por defecto) NO elimina los clientes con
    `cod_boca_salida == "Sin Asignar"`: salen en `validar_combinaciones`. Con `False` se
    reproduce el filtro del notebook original.

    No incluye el parche puntual del cliente `1265373` del notebook original: la
    deduplicación de acá ya cubre ese caso de forma genérica, para cualquier cliente.
    """
    df = _reglas_boca_salida(df_final)
    df = _reglas_potencial_por_encuesta(df)
    df = _reglas_por_definir(df, conservar_sin_asignar=conservar_sin_asignar)

    df = df.rename(columns={
        "cod_cliente_dex": "cod_interlocutor",
        "des_segmentacion_transaccional": "des_segmento_transaccional",
        "des_segmentacion_estrategica": "des_segmento_estrategica",
        "flag_potential": "flg_potential",
        "des_oficina_venta_alicorp": "des_oficina_venta",
        "fec_corrida": "fec_proceso",
    })
    df["cod_interlocutor"] = df["cod_interlocutor"].astype(str).str.zfill(10)

    faltan = [c for c in COLUMNAS_GOLD if c not in df.columns]
    if faltan:
        raise KeyError(f"Faltan columnas para la tabla gold: {faltan}")
    df = _desempatar_duplicados(df)
    return df[COLUMNAS_GOLD]


def aplicar_invariante_potencial(df: pd.DataFrame) -> pd.DataFrame:
    """Si `flg_potential` es `False`, `des_potencial_predic` tiene que ser `NULL`.

    Es una regla estructural (no puede existir una etiqueta de potencial para un cliente
    que no es potencial), y se aplica siempre al final, sin importar de dónde salió cada
    columna — cálculo fresco, copiado del periodo anterior, o una mezcla de ambos.
    """
    df = df.copy()
    df["flg_potential"] = df["flg_potential"].map(_a_booleano).astype("boolean")
    df.loc[df["flg_potential"].eq(False).fillna(False).astype(bool), "des_potencial_predic"] = None
    return df


def _cruzar_con_anterior(df_final: pd.DataFrame, gold_periodo_anterior: pd.DataFrame) -> pd.DataFrame:
    """`df_final` + las columnas `*_ant` de la fila del cliente en el periodo anterior.

    El emparejamiento es SOLO por `cod_interlocutor`: un cliente es "nuevo" únicamente si su
    código no estaba en el periodo anterior. Agrega `_en_anterior` (bool) y las columnas
    `cod_boca_salida_ant`, `des_segmento_transaccional_ant`, `des_segmento_estrategica_ant`,
    `flg_potential_ant` y `des_potencial_predic_ant`. Nunca agrega ni quita filas de `df_final`.
    """
    cols = ["cod_interlocutor", "cod_boca_salida", "des_segmento_transaccional", *COLUMNAS_SEGMENTACION]
    faltan = [c for c in cols if c not in gold_periodo_anterior.columns]
    if faltan:
        raise KeyError(f"Faltan columnas en gold_periodo_anterior: {faltan}")

    anterior = gold_periodo_anterior[cols].drop_duplicates(subset=["cod_interlocutor"])
    anterior = anterior.rename(columns={c: f"{c}_ant" for c in cols[1:]})
    anterior["_en_anterior"] = True
    out = df_final.merge(anterior, on="cod_interlocutor", how="left")
    out["_en_anterior"] = out["_en_anterior"].fillna(False).astype(bool)
    return out


def _boca_distinta(cruce: pd.DataFrame) -> pd.Series:
    """El cliente estaba en el periodo anterior pero con otra `cod_boca_salida`."""
    misma = cruce["cod_boca_salida"].fillna("") == cruce["cod_boca_salida_ant"].fillna("")
    return cruce["_en_anterior"] & ~misma


def _estaba_por_definir(cruce: pd.DataFrame) -> pd.Series:
    """Estaba 'Por definir' en el periodo anterior: su transaccional o su estratégica no tenían segmento."""
    return (_es_pendiente(cruce["des_segmento_transaccional_ant"])
            | _es_pendiente(cruce["des_segmento_estrategica_ant"]))


def completar_incremental(df_final: pd.DataFrame, gold_periodo_anterior: pd.DataFrame) -> pd.DataFrame:
    """Modo incremental: la base es el gold del periodo INMEDIATAMENTE anterior (no se busca
    más atrás) y `df_final` (cálculo fresco de este periodo, ya con `aplicar_reglas_negocio`)
    solo aporta lo que el periodo anterior no tenía resuelto.

    Nadie sale: devuelve una fila por cada cliente de `df_final` MÁS una fila por cada cliente
    del periodo anterior que hoy no viene en el cálculo fresco (ver abajo). El emparejamiento
    es SOLO por `cod_interlocutor`:

    - **Cliente del periodo anterior que no está en `df_final`**: se arrastra su fila anterior
      tal cual (boca, segmentos, potencial y demás columnas), con el `periodo` y el
      `fec_proceso` de este periodo.

    - **Cliente nuevo** (su código no estaba en el periodo anterior): se agrega con el cálculo
      fresco. Si su transaccional es "Por definir", queda sin segmento estratégico ("Por
      definir") ni potencial (`flg_potential` y `des_potencial_predic` NULL), en todas las bocas.
    - **Cliente que ya estaba pero con otra `cod_boca_salida`**: se desestima el cálculo fresco;
      conserva la boca, la transaccional, la estratégica y el potencial del periodo anterior.
    - **Cliente que ya estaba con la misma boca**: solo cambia lo que estaba "Por definir":
        - `des_segmento_transaccional`: fresca si la anterior era "Por definir"/vacía; si ya
          tenía segmento, se conserva la anterior (aunque la fresca diga otro segmento).
        - `des_segmento_estrategica`: ídem, según la estratégica anterior.
        - `flg_potential` y `des_potencial_predic`: frescos si el cliente estaba "Por definir"
          (transaccional o estratégica) o su `flg_potential` anterior era NULL; si no, se
          conservan los anteriores, incluido `False`.
      Un cliente con segmento nunca regresa a "Por definir".
    """
    cruce = _cruzar_con_anterior(df_final, gold_periodo_anterior)
    en_ant = cruce["_en_anterior"]
    boca_distinta = _boca_distinta(cruce)
    misma_boca = en_ant & ~boca_distinta

    transac_ant_pend = _es_pendiente(cruce["des_segmento_transaccional_ant"])
    estrat_ant_pend = _es_pendiente(cruce["des_segmento_estrategica_ant"])
    potencial_ant_pend = transac_ant_pend | estrat_ant_pend | cruce["flg_potential_ant"].isna()

    out = cruce
    # Valor fresco: transaccional "Por definir" => sin estratégica ni potencial (todas las bocas).
    fresco_pend = _es_pendiente(out["des_segmento_transaccional"])
    out["des_segmento_estrategica"] = out["des_segmento_estrategica"].astype(object)
    out["flg_potential"] = out["flg_potential"].astype(object)
    out["des_potencial_predic"] = out["des_potencial_predic"].astype(object)
    out.loc[fresco_pend, "des_segmento_estrategica"] = POR_DEFINIR
    out.loc[fresco_pend, "flg_potential"] = pd.NA
    out.loc[fresco_pend, "des_potencial_predic"] = None

    for col, heredar in (
        ("cod_boca_salida", boca_distinta),
        ("des_segmento_transaccional", boca_distinta | (misma_boca & ~transac_ant_pend)),
        ("des_segmento_estrategica", boca_distinta | (misma_boca & ~estrat_ant_pend)),
        ("flg_potential", boca_distinta | (misma_boca & ~potencial_ant_pend)),
        ("des_potencial_predic", boca_distinta | (misma_boca & ~potencial_ant_pend)),
    ):
        out[col] = out[col].astype(object).where(~heredar, out[f"{col}_ant"].astype(object))

    out = out.drop(columns=[c for c in out.columns if c.endswith("_ant")] + ["_en_anterior"])

    # Nadie sale: los clientes del periodo anterior que hoy no vienen en el cálculo fresco
    # se arrastran con su fila anterior, solo con el periodo y la fecha de proceso de hoy.
    faltan_ant = [c for c in COLUMNAS_GOLD if c not in gold_periodo_anterior.columns]
    if faltan_ant:
        raise KeyError(f"Faltan columnas en gold_periodo_anterior: {faltan_ant}")
    ausentes = gold_periodo_anterior.loc[
        ~gold_periodo_anterior["cod_interlocutor"].isin(out["cod_interlocutor"]), COLUMNAS_GOLD
    ].drop_duplicates(subset=["cod_interlocutor"]).copy()
    if not ausentes.empty:
        ausentes["periodo"] = out["periodo"].iloc[0]
        ausentes["fec_proceso"] = out["fec_proceso"].iloc[0]
        out = pd.concat([out, ausentes[COLUMNAS_GOLD]], ignore_index=True)
    return aplicar_invariante_potencial(out)


def validar_incremental(df_final: pd.DataFrame, gold_periodo_anterior: pd.DataFrame) -> dict:
    """Cuenta cada grupo del modo incremental, para revisar ANTES de insertar.

    `df_final` es el cálculo fresco (salida de `aplicar_reglas_negocio`). Devuelve un dict:

    - `clientes_periodo_anterior`, `clientes_periodo_actual`
    - `nuevos`: su código no estaba en el periodo anterior (se agregan)
    - `arrastrados`: estaban en el periodo anterior y hoy no vienen en el cálculo fresco; se
      arrastra su fila anterior (nadie sale)
    - `boca_distinta_desestimada`: estaban con otra boca; se desestima el cálculo fresco y
      conservan su fila anterior
    - `por_definir_resueltos`: estaban "Por definir" y hoy su transaccional tiene segmento
      (cambian transaccional, estratégica y potencial)
    - `siguen_por_definir`: estaban "Por definir" y hoy también
    - `conservados`: tenían segmento y misma boca; se conserva todo lo anterior
    - `cambios_ignorados`: de los conservados, cuántos traían una transaccional fresca distinta
      (el cambio no se aplica; conviene revisarlos)
    """
    cruce = _cruzar_con_anterior(df_final, gold_periodo_anterior)
    en_ant = cruce["_en_anterior"]
    boca_distinta = _boca_distinta(cruce)
    misma_boca = en_ant & ~boca_distinta
    pendiente = _estaba_por_definir(cruce)
    hoy_pendiente = _es_pendiente(cruce["des_segmento_transaccional"])
    transac_ant_pend = _es_pendiente(cruce["des_segmento_transaccional_ant"])
    conservado = misma_boca & ~pendiente
    transac_distinta = (
        cruce["des_segmento_transaccional"].fillna("") != cruce["des_segmento_transaccional_ant"].fillna("")
    )

    ids_hoy = set(df_final["cod_interlocutor"])
    ids_ant = set(gold_periodo_anterior["cod_interlocutor"])
    return {
        "clientes_periodo_anterior": len(ids_ant),
        "clientes_periodo_actual": len(ids_hoy),
        "nuevos": len(ids_hoy - ids_ant),
        "arrastrados": len(ids_ant - ids_hoy),
        "boca_distinta_desestimada": int(boca_distinta.sum()),
        "por_definir_resueltos": int((misma_boca & pendiente & ~hoy_pendiente).sum()),
        "siguen_por_definir": int((misma_boca & pendiente & hoy_pendiente).sum()),
        "conservados": int(conservado.sum()),
        "cambios_ignorados": int((conservado & ~transac_ant_pend & transac_distinta).sum()),
    }


def resumen_universo(df_armado: pd.DataFrame, df_reglas: pd.DataFrame) -> dict:
    """Entrada vs. salida de `aplicar_reglas_negocio`: cuántos clientes entran, cuántas filas
    duplicadas se descartan y si algún cliente se perdió (no debería con
    `conservar_sin_asignar=True`)."""
    ids_in = set(df_armado["cod_cliente_dex"].astype(str).str.zfill(10))
    ids_out = set(df_reglas["cod_interlocutor"])
    return {
        "filas_entrada": len(df_armado),
        "clientes_entrada": len(ids_in),
        "filas_duplicadas_descartadas": len(df_armado) - len(ids_in),
        "clientes_salida": len(ids_out),
        "clientes_perdidos": len(ids_in - ids_out),
    }


def preparar_carga_periodo(df_final: pd.DataFrame, modo: str, gold_periodo_anterior: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """`modo='completo'`: `df_final` tal cual (solo el invariante de potencial).
    `modo='incremental'`: requiere `gold_periodo_anterior` y aplica `completar_incremental`.
    En ambos casos, listo para pasarlo a `cargar_periodo_gold`."""
    if modo == "completo":
        return aplicar_invariante_potencial(df_final)
    if modo == "incremental":
        if gold_periodo_anterior is None:
            raise ValueError("modo='incremental' necesita gold_periodo_anterior (la fila de gold del periodo anterior).")
        return completar_incremental(df_final, gold_periodo_anterior)
    raise ValueError(f"modo debe ser 'completo' o 'incremental', recibió {modo!r}")


COLUMNAS_BOOLEANAS = ["flg_encuesta", "flg_mercado_potencial", "flg_potential"]
COLUMNAS_FECHA = ["periodo", "fec_proceso"]
COLUMNAS_TEXTO = [c for c in COLUMNAS_GOLD if c not in COLUMNAS_BOOLEANAS + COLUMNAS_FECHA]

# Tipos de la tabla gold según el notebook original (`CAST(periodo AS DATE)`,
# `CAST(fec_proceso AS DATETIME)`, `CAST(... AS BOOL)`, `CAST(... AS STRING)`). OJO: `periodo`
# es DATE pero `fec_proceso` es DATETIME — no son el mismo tipo. Se usan solo si no se puede
# leer el esquema real de la tabla (ver `tipos_de_tabla`).
TIPOS_BQ_POR_DEFECTO = {
    **{c: "STRING" for c in COLUMNAS_TEXTO},
    **{c: "BOOLEAN" for c in COLUMNAS_BOOLEANAS},
    "periodo": "DATE",
    "fec_proceso": "DATETIME",
}


def _a_booleano(v):
    """Un valor suelto -> True / False / pd.NA. Acepta bool, 0/1 (int o float), y los textos
    'True'/'False'/'VERDADERO'/'FALSO'. Cualquier otra cosa levanta ValueError (mejor fallar
    acá, con el valor a la vista, que dentro de Arrow con un mensaje críptico)."""
    if pd.isna(v):
        return pd.NA
    if isinstance(v, str):
        t = v.strip().lower()
        if t in ("true", "verdadero", "1", "1.0"):
            return True
        if t in ("false", "falso", "0", "0.0"):
            return False
        raise ValueError(f"No se puede interpretar {v!r} como booleano.")
    return bool(v)


def _a_texto(v):
    return None if pd.isna(v) else str(v)


def tipos_de_tabla(client, tabla_gold: str) -> Dict[str, str]:
    """Los tipos de BigQuery de las columnas de `COLUMNAS_GOLD`, leídos del esquema REAL de
    `tabla_gold` (`client.get_table`) — para no depender de lo que se supone que es cada columna.

    Si no se puede leer (el cliente no tiene `get_table`, o falla), usa `TIPOS_BQ_POR_DEFECTO`
    y avisa con un `warnings.warn`, en vez de seguir en silencio con tipos supuestos.
    """
    try:
        esquema = client.get_table(tabla_gold).schema
        reales = {campo.name: campo.field_type.upper() for campo in esquema}
    except Exception as exc:  # noqa: BLE001 — cualquier falla acá cae al esquema documentado
        warnings.warn(
            f"No se pudo leer el esquema de {tabla_gold} ({type(exc).__name__}: {exc}); "
            "se usan los tipos documentados del notebook original.",
            stacklevel=2,
        )
        return dict(TIPOS_BQ_POR_DEFECTO)
    return {c: reales.get(c, TIPOS_BQ_POR_DEFECTO[c]) for c in COLUMNAS_GOLD}


def _convertir_columna(serie: pd.Series, tipo_bq: str) -> pd.Series:
    t = tipo_bq.upper()
    if t == "DATE":
        return pd.to_datetime(serie).dt.date
    if t == "DATETIME":                      # sin zona horaria
        out = pd.to_datetime(serie)
        return out.dt.tz_convert("UTC").dt.tz_localize(None) if out.dt.tz is not None else out
    if t == "TIMESTAMP":                     # con zona horaria (UTC)
        return pd.to_datetime(serie, utc=True)
    if t in ("BOOLEAN", "BOOL"):
        return serie.map(_a_booleano).astype("boolean")
    if t == "STRING":
        return serie.map(_a_texto).astype(object)
    return serie                             # otros tipos (INT64, NUMERIC, ...): no se tocan


def preparar_tipos_bigquery(df: pd.DataFrame, tipos: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Deja cada columna de `COLUMNAS_GOLD` con el tipo que espera la tabla gold.

    `tipos`: {columna: tipo de BigQuery} (lo normal es pasar `tipos_de_tabla(client, tabla)`);
    si no se pasa, se usa `TIPOS_BQ_POR_DEFECTO`. Conversiones:

    - `DATE` -> `datetime.date`; `DATETIME` -> `datetime64` sin zona horaria; `TIMESTAMP` -> con
      zona UTC. (Un `date` donde la tabla espera `DATETIME`/`TIMESTAMP` falla en Arrow con
      "object of type <class 'datetime.date'> cannot be converted to int".)
    - `BOOLEAN` -> `boolean` nullable. Una columna `object` que mezcla `True`/`False` con `0.0`
      o con textos falla en Arrow con "tried to convert to boolean".
    - `STRING` -> texto o `None`, nunca `NaN` ni números sueltos.
    """
    faltan = [c for c in COLUMNAS_GOLD if c not in df.columns]
    if faltan:
        raise KeyError(f"Faltan columnas para la tabla gold: {faltan}")
    tipos = {**TIPOS_BQ_POR_DEFECTO, **(tipos or {})}
    out = df[COLUMNAS_GOLD].copy()
    for c in COLUMNAS_GOLD:
        out[c] = _convertir_columna(out[c], tipos[c])
    return out


def periodo_ya_cargado(client, tabla_gold: str, periodo: str) -> bool:
    """¿Ya hay alguna fila en `tabla_gold` con `periodo = periodo`?

    Filtra por `periodo` (obligatorio: la tabla tiene `require_partition_filter`), así que
    sirve también como el chequeo que necesita `cargar_periodo_gold` antes de insertar.
    """
    query = f"SELECT COUNT(*) AS n FROM `{tabla_gold}` WHERE periodo = DATE('{periodo}')"
    resultado = client.query(query).to_dataframe()
    return bool(resultado["n"].iloc[0] > 0)


_TIPO_SQL = {"BOOLEAN": "BOOL", "BOOL": "BOOL"}


def construir_insert_gold(tabla_gold: str, tabla_staging: str, tipos: Optional[Dict[str, str]] = None) -> str:
    """El `INSERT INTO <gold> (...) SELECT CAST(...) FROM <staging>` de la carga.

    Es el mismo mecanismo del notebook original (celda "Carga a capa gold"): una consulta SQL
    de escritura, que solo necesita permiso para escribir datos en la tabla — no para crear
    tablas en el dataset de prod, que es lo que pide un *load job* con `CREATE_IF_NEEDED`
    (el valor por defecto de `pandas_gbq`) y fallaba con "bigquery.tables.create denied".
    """
    tipos = {**TIPOS_BQ_POR_DEFECTO, **(tipos or {})}
    columnas = ", ".join(COLUMNAS_GOLD)
    casts = ",\n    ".join(
        f"CAST({c} AS {_TIPO_SQL.get(tipos[c].upper(), tipos[c].upper())}) AS {c}" for c in COLUMNAS_GOLD
    )
    return f"INSERT INTO `{tabla_gold}` ({columnas})\nSELECT\n    {casts}\nFROM `{tabla_staging}`"


def cargar_periodo_gold(df: pd.DataFrame, client, tabla_gold: str, tabla_staging: str) -> Optional[int]:
    """Inserta en `tabla_gold` las filas de UN periodo (`df`), pasando por `tabla_staging`.

    1. Se niega (`ValueError`) si `df` tiene más de un periodo, o si ese periodo YA tiene filas
       en gold — se consulta acá mismo, justo antes de insertar, así que volver a ejecutar solo
       esta celda tras una carga exitosa no duplica nada. Nunca se toca un periodo ya cargado.
    2. Sube `df` a `tabla_staging` (`project.dataset.tabla`, en un dataset donde sí se pueden
       crear tablas, p. ej. el sandbox), reemplazándola, con los tipos del esquema real de gold.
    3. Hace `INSERT INTO <gold> SELECT ... FROM <staging>` (`construir_insert_gold`).

    Devuelve las filas insertadas según BigQuery, y lanza `RuntimeError` si no coinciden con
    `len(df)`. No hay `MERGE`: con la clave (cliente, periodo), un periodo que todavía no
    existe nunca choca con una fila existente.
    """
    if df.empty:
        raise ValueError("df está vacío: no hay nada que insertar.")
    periodos = pd.to_datetime(df["periodo"]).dt.date.unique()
    if len(periodos) != 1:
        raise ValueError(f"Una carga es de un solo periodo, pero df trae {len(periodos)}: {sorted(map(str, periodos))}")
    periodo = periodos[0].isoformat()
    if periodo_ya_cargado(client, tabla_gold, periodo):
        raise ValueError(
            f"Ya existe al menos una fila en la tabla gold para periodo={periodo}. No se inserta nada "
            "(nunca se toca un periodo ya cargado) — si de verdad hay que recargarlo, hay que "
            "borrar esas filas primero, a mano."
        )
    from google.cloud import bigquery

    tipos = tipos_de_tabla(client, tabla_gold)          # `periodo` DATE, `fec_proceso` DATETIME, ...
    df = preparar_tipos_bigquery(df, tipos)

    config = bigquery.LoadJobConfig(
        schema=[bigquery.SchemaField(c, tipos[c]) for c in COLUMNAS_GOLD],
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,   # el staging se reemplaza en cada carga
    )
    client.load_table_from_dataframe(df, tabla_staging, job_config=config).result()

    trabajo = client.query(construir_insert_gold(tabla_gold, tabla_staging, tipos))
    trabajo.result()
    insertadas = getattr(trabajo, "num_dml_affected_rows", None)
    if insertadas is not None and insertadas != len(df):
        raise RuntimeError(
            f"BigQuery reporta {insertadas:,} filas insertadas, pero se esperaban {len(df):,}. "
            "Revisa la tabla gold antes de volver a correr nada."
        )
    return insertadas











