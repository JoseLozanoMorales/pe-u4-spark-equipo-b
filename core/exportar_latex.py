"""
Exportacion de resultados a LaTeX (booktabs) y a JSON.

Los .tex generados se incluyen en el informe con \\input{} o se pegan tal cual.
Los numeros quedan anclados al experimento real: no hay transcripcion manual,
por lo que no puede haber discrepancias entre el codigo y el documento.

Requiere en el preambulo del informe:
    \\usepackage{booktabs}
    \\usepackage{siunitx}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from . import config


def tabla_booktabs(
    cuerpo: Iterable[Sequence[Any]],
    encabezado: Sequence[str],
    formato: str,
    caption: str,
    label: str,
    nota: str = "",
) -> str:
    """Construye un entorno table de LaTeX con reglas booktabs."""
    lineas = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\small",
        rf"\begin{{tabular}}{{{formato}}}",
        r"\toprule",
        " & ".join(encabezado) + r" \\",
        r"\midrule",
    ]
    lineas += [" & ".join(str(celda) for celda in fila) + r" \\" for fila in cuerpo]
    lineas += [r"\bottomrule", r"\end{tabular}"]
    if nota:
        lineas.append(rf"\\[2pt]\footnotesize {nota}")
    lineas += [r"\end{table}", ""]
    return "\n".join(lineas)


def tex_tiempos(tabla_speedup: pd.DataFrame) -> str:
    """Tabla 1: tiempos medianos y speedup por transformacion."""
    cuerpo = [
        [
            f["TX"],
            f["Descripcion"],
            f"{f['T_pandas_s']:.4f}",
            f"{f['T_spark_s']:.4f}",
            f"{f['Speedup']:.3f}",
            f"{f['CV_spark_pct']:.1f}",
        ]
        for _, f in tabla_speedup.iterrows()
    ]
    return tabla_booktabs(
        cuerpo,
        [r"\textbf{TX}", r"\textbf{Transformacion}", r"$T_{pandas}$ (s)",
         r"$T_{spark}$ (s)", r"$S$", r"CV (\%)"],
        "llrrrr",
        "Tiempos medianos de ejecucion "
        f"({config.REPETICIONES} repeticiones) y speedup obtenido con PySpark "
        f"en local[{config.CORES_BASE}].",
        "tab:tiempos",
        "CV = coeficiente de variacion de las mediciones en PySpark.",
    )


def tex_amdahl(tabla_escalado: pd.DataFrame, parametros: dict[str, float]) -> str:
    """Tabla 2: escalado de T3 y estimacion de la fraccion paralelizable."""
    cuerpo = [
        [
            int(f["N"]),
            f"{f['T_spark_s']:.4f}",
            f"{f['S_medido']:.3f}",
            f"{f['Eficiencia']:.3f}",
            "---" if np.isnan(f["frac_paralela_p"]) else f"{f['frac_paralela_p']:.4f}",
        ]
        for _, f in tabla_escalado.iterrows()
    ]
    nota = (
        rf"$p$ promedio = {parametros['p_paralelizable']:.4f}; "
        rf"$S_{{max}}$ = {parametros['S_max']:.3f}; "
        rf"$N_{{90}}$ = {int(parametros['N_90_entero'])}."
    )
    return tabla_booktabs(
        cuerpo,
        [r"$N$", r"$T_{spark}$ (s)", r"$S(N)$", r"$E=S/N$", r"$p$ estimada"],
        "rrrrr",
        "Escalado de la transformacion T3 (join) con 1, 2 y 4 executors y "
        "estimacion de la fraccion paralelizable.",
        "tab:amdahl",
        nota,
    )


def tex_verificacion(verificacion: dict[str, dict[str, Any]]) -> str:
    """Tabla 3: equivalencia de resultados pandas vs. PySpark."""
    cuerpo = [
        [
            clave,
            f"{v['filas_pandas']:,}",
            f"{v['filas_spark']:,}",
            f"{v['firma_pandas']:,.2f}",
            f"{v['firma_spark']:,.2f}",
            "Si" if v["identico"] else "No",
        ]
        for clave, v in verificacion.items()
    ]
    return tabla_booktabs(
        cuerpo,
        [r"\textbf{TX}", "Filas pandas", "Filas Spark", "Firma pandas",
         "Firma Spark", "Identico"],
        "lrrrrc",
        "Verificacion de equivalencia de resultados entre pandas y PySpark.",
        "tab:verificacion",
        "Firma = suma de la columna representativa de cada transformacion.",
    )


def exportar_todo(
    tabla_speedup: pd.DataFrame,
    tabla_escalado: pd.DataFrame,
    parametros: dict[str, float],
    verificacion: dict[str, dict[str, Any]],
    resultados_pandas: dict[str, Any],
    resultados_spark: dict[str, Any],
    escalado_t3: dict[int, Any],
    version_spark: str = "3.5.0",
) -> Path:
    """Escribe tablas.tex, los CSV de resultados y resultados.json."""
    contenido = "\n".join(
        [
            tex_tiempos(tabla_speedup),
            tex_amdahl(tabla_escalado, parametros),
            tex_verificacion(verificacion),
        ]
    )
    ruta_tex = config.DIR_TEX / "tablas.tex"
    ruta_tex.write_text(contenido, encoding="utf-8")

    tabla_speedup.to_csv(config.BASE / "tabla_speedup.csv", index=False)
    tabla_escalado.to_csv(config.BASE / "tabla_amdahl.csv", index=False)

    resumen = {
        "configuracion": {
            "n_reservas": config.N_RESERVAS,
            "n_laboratorios": config.N_LABORATORIOS,
            "n_usuarios": config.N_USUARIOS,
            "semilla": config.SEMILLA,
            "repeticiones": config.REPETICIONES,
            "cores_base": config.CORES_BASE,
            "cores_escalado": list(config.CORES_TEST),
            "spark_version": version_spark,
        },
        "pandas": resultados_pandas,
        "spark": resultados_spark,
        "escalado_T3": {str(k): v for k, v in escalado_t3.items()},
        "amdahl": parametros,
        "verificacion": verificacion,
    }
    ruta_json = config.BASE / "resultados.json"
    ruta_json.write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[export] {ruta_tex}")
    print(f"[export] {ruta_json}")
    return ruta_tex
