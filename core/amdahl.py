"""
Paso 4 de la guia - Analisis cuantitativo con la Ley de Amdahl.

FORMULACION
-----------
Ley de Amdahl (1967), con p = fraccion PARALELIZABLE del programa:

        S(N) = 1 / ( (1 - p) + p/N )

Despejando p a partir de un speedup medido S con N procesadores (formula
inversa que aparece en la guia, donde el resultado es la fraccion SERIAL):

        1 - p = ( 1/S - 1/N ) / ( 1 - 1/N )

Limite cuando N -> infinito:

        S_max = 1 / (1 - p)

Numero de procesadores para alcanzar el 90 % de S_max:

        S(N) = 0.9 * S_max
        (1-p) + p/N = (1-p)/0.9
        p/N = (1-p) * (1/0.9 - 1) = (1-p)/9
        N_90 = 9p / (1-p)

DECISION METODOLOGICA IMPORTANTE
--------------------------------
El speedup que alimenta la formula inversa es el speedup de ESCALADO INTERNO de
Spark, S(N) = T_spark(1) / T_spark(N), y NO T_pandas / T_spark.

Amdahl modela como escala UN MISMO programa al anadir procesadores. Usar pandas
como denominador mezcla dos motores distintos (interprete de CPython vs. JVM +
Catalyst) e introduce el coste fijo de arranque de Spark disfrazado de fraccion
serial, lo que sobreestima 1-p. Se reportan ambos speedups, pero la estimacion
de p usa la base N=1 de Spark.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from . import config


def speedup_por_transformacion(
    resultados_pandas: dict[str, dict[str, Any]],
    resultados_spark: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Tabla de speedup S = T_pandas / T_spark por transformacion."""
    filas = []
    for clave in resultados_pandas:
        t_pandas = resultados_pandas[clave]["mediana_s"]
        t_spark = resultados_spark[clave]["mediana_s"]
        filas.append(
            {
                "TX": clave,
                "Descripcion": config.DESCRIPCION[clave],
                "T_pandas_s": round(t_pandas, 6),
                "T_spark_s": round(t_spark, 6),
                "T_pandas_us": round(t_pandas * 1e6, 1),
                "T_spark_us": round(t_spark * 1e6, 1),
                "Speedup": round(t_pandas / t_spark, 4),
                "CV_pandas_pct": round(resultados_pandas[clave]["cv_pct"], 2),
                "CV_spark_pct": round(resultados_spark[clave]["cv_pct"], 2),
                "Ganancia": "SI" if t_pandas / t_spark > 1 else "NO",
            }
        )
    return pd.DataFrame(filas)


def tabla_escalado(escalado_t3: dict[int, dict[str, Any]]) -> pd.DataFrame:
    """Tabla de escalado de T3 con la fraccion serial/paralela estimada por N."""
    t_base = escalado_t3[1]["mediana_s"]
    filas = []
    for n in sorted(escalado_t3):
        t_n = escalado_t3[n]["mediana_s"]
        s = t_base / t_n
        if n == 1:
            frac_serial = np.nan
        else:
            frac_serial = ((1 / s) - (1 / n)) / (1 - (1 / n))
        filas.append(
            {
                "N": n,
                "T_spark_s": round(t_n, 6),
                "S_medido": round(s, 4),
                "Eficiencia": round(s / n, 4),
                "frac_serial": round(frac_serial, 4) if not np.isnan(frac_serial) else np.nan,
                "frac_paralela_p": round(1 - frac_serial, 4)
                if not np.isnan(frac_serial)
                else np.nan,
            }
        )
    return pd.DataFrame(filas)


def estimar_parametros(tabla: pd.DataFrame) -> dict[str, float]:
    """Estima p, S_max y N_90 promediando las estimaciones de p con N > 1."""
    valores = [v for v in tabla["frac_paralela_p"].tolist() if not np.isnan(v)]
    p_bruto = float(np.mean(valores)) if valores else 0.0
    p = float(np.clip(p_bruto, 1e-4, 1 - 1e-4))

    return {
        "p_bruto": p_bruto,
        "p_paralelizable": p,
        "fraccion_serial": 1 - p,
        "S_max": 1 / (1 - p),
        "N_90": 9 * p / (1 - p),
        "N_90_entero": float(math.ceil(9 * p / (1 - p))),
        "escala": bool(p_bruto > 0),
    }


def curva_amdahl(p: float, n: np.ndarray) -> np.ndarray:
    """Evalua S(N) = 1 / ((1-p) + p/N) sobre un vector de N."""
    return 1.0 / ((1 - p) + p / n)


def imprimir_resumen(parametros: dict[str, float], tabla_speedup: pd.DataFrame) -> None:
    """Resumen legible del analisis para la consola."""
    p = parametros["p_paralelizable"]
    print("\n=== ANALISIS DE AMDAHL (transformacion T3) ===")
    print(f"Fraccion paralelizable  p      = {p:.4f}  ({p * 100:.2f} %)")
    print(f"Fraccion serial         1-p    = {1 - p:.4f}  ({(1 - p) * 100:.2f} %)")
    print(f"Speedup teorico maximo  S_max  = {parametros['S_max']:.3f}x")
    print(
        f"Procesadores para 90%   N_90   = {parametros['N_90']:.2f} "
        f"-> {int(parametros['N_90_entero'])} procesadores"
    )

    s_t3 = tabla_speedup.loc[tabla_speedup["TX"] == "T3", "Speedup"].iloc[0]
    print(f"Speedup de T3 frente a pandas   = {s_t3:.3f}x")

    if not parametros["escala"]:
        print(
            "\n[!] p <= 0: T3 no escala al anadir executors en este entorno.\n"
            "    Interpretacion valida: el shuffle y el coste fijo de la JVM dominan\n"
            "    sobre el trabajo paralelizable a este volumen de datos.\n"
            "    Documentalo como HALLAZGO experimental, no como error."
        )
