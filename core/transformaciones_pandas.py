"""
Paso 2 de la guia - Implementacion secuencial con pandas.

Las cinco transformaciones se ejecutan sobre un unico hilo (pandas no paraleliza
estas operaciones), lo que las convierte en la linea base T_secuencial del
experimento. Cada transformacion parte SIEMPRE del DataFrame base, de modo que
sus tiempos son independientes entre si, tal como exige la guia.

Las operaciones responden a consultas reales del dominio SCLI:

  T1 : solicitudes aprobadas de grupos grandes en jornada vespertina
  T2 : indicadores de uso agregados por laboratorio y estado
  T3 : vista analitica completa (hechos + 3 dimensiones, join por UUID)
  T4 : metricas derivadas de ocupacion y duracion
  T5 : ranking de las solicitudes de mayor demanda
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from . import config


def construir_transformaciones(
    solicitudes: pd.DataFrame,
    reservas: pd.DataFrame,
    laboratorios: pd.DataFrame,
    perfiles: pd.DataFrame,
    materias: pd.DataFrame,
) -> dict[str, Callable[[], pd.DataFrame]]:
    """Devuelve {clave: callable} con las cinco transformaciones pandas."""

    def t1() -> pd.DataFrame:
        # Filtrado por condicion compuesta (cuatro predicados)
        # Dominio: solicitudes aprobadas, de grupos grandes, en jornada vespertina
        return solicitudes[
            (solicitudes["estado_solicitud"] == "APROBADA")
            & (solicitudes["numero_participantes"] >= 20)
            & (solicitudes["hora_inicio"] >= 14)
            & (solicitudes["hora_fin"] - solicitudes["hora_inicio"] >= 2)
        ]

    def t2() -> pd.DataFrame:
        # Agrupacion por dos claves + cinco funciones de agregacion
        # Dominio: indicadores de uso por laboratorio y estado de solicitud
        return solicitudes.groupby(
            ["laboratorio_id", "estado_solicitud"], as_index=False
        ).agg(
            total_solicitudes=("solicitud_id", "count"),
            participantes_tot=("numero_participantes", "sum"),
            participantes_prom=("numero_participantes", "mean"),
            hora_inicio_min=("hora_inicio", "min"),
            hora_fin_max=("hora_fin", "max"),
        )

    def t3() -> pd.DataFrame:
        # Join de 4 DataFrames por claves UUID
        # Dominio: vista analitica de solicitudes con laboratorio, solicitante y materia
        return (
            solicitudes.merge(laboratorios, on="laboratorio_id", how="inner")
            .merge(
                perfiles.rename(columns={"perfil_id": "solicitante_id"}),
                on="solicitante_id",
                how="inner",
            )
            .merge(materias, on="materia_id", how="inner")
        )

    def t4() -> pd.DataFrame:
        # Columna derivada compleja: aritmetica temporal + condicional + categorizacion
        # Dominio: duracion real, franja horaria y clasificacion de la demanda
        df = solicitudes.copy()

        df["horas_laboratorio"] = (df["hora_fin"] - df["hora_inicio"]).astype(float)

        df["franja"] = np.select(
            [df["hora_inicio"] < 12, df["hora_inicio"] < 18],
            ["MATUTINA", "VESPERTINA"],
            default="NOCTURNA",
        )

        # Peso operativo: participantes-hora, con recargo por franja nocturna
        recargo = np.where(
            df["hora_inicio"] >= 18, 1.25, np.where(df["hora_inicio"] < 9, 0.90, 1.00)
        )
        df["participantes_hora"] = np.round(
            df["numero_participantes"] * df["horas_laboratorio"] * recargo, 4
        )

        df["nivel_demanda"] = np.select(
            [
                df["participantes_hora"] < 30,
                df["participantes_hora"] < 80,
                df["participantes_hora"] < 150,
            ],
            ["BAJA", "MEDIA", "ALTA"],
            default="CRITICA",
        )
        return df

    def t5() -> pd.DataFrame:
        # Ordenamiento multiclave descendente + top-100
        # Dominio: ranking de solicitudes de mayor demanda
        return solicitudes.sort_values(
            ["numero_participantes", "hora_fin", "fecha_reserva"],
            ascending=[False, False, False],
        ).head(100)

    return {"T1": t1, "T2": t2, "T3": t3, "T4": t4, "T5": t5}


def persistir_resultados(
    transformaciones: dict[str, Callable[[], pd.DataFrame]], verbose: bool = True
) -> dict[str, pd.DataFrame]:
    """Ejecuta cada transformacion una vez y guarda el resultado en /data/pandas/.

    Va aparte de la medicion: la escritura a disco no debe contar en el tiempo.
    """
    salidas: dict[str, pd.DataFrame] = {}
    for clave, fn in transformaciones.items():
        df = fn()
        salidas[clave] = df
        ruta = config.DIR_PANDAS / f"{clave}_resultado.csv"
        df.to_csv(ruta, index=False)
        if verbose:
            print(f"[pandas] {clave}: {df.shape[0]:>8,} filas x {df.shape[1]:>2} cols -> {ruta}")
    return salidas


def firma(df: pd.DataFrame, clave: str) -> float:
    """Suma de la columna representativa; se usa para verificar equivalencia."""
    columna = config.COLUMNA_FIRMA[clave]
    return round(float(df[columna].sum()), 2)


__all__ = ["construir_transformaciones", "persistir_resultados", "firma"]
