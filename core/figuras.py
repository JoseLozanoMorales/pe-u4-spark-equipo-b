"""
Paso 4 de la guia - Generacion de las tres figuras exigidas, en PNG a 300 DPI.

  (a) fig_a_tiempos.png          barras: tiempos pandas vs. PySpark por transformacion
  (b) fig_b_speedup_amdahl.png   speedup vs. N con la curva teorica de Amdahl superpuesta
  (c) fig_c_eficiencia.png       eficiencia E = S/N vs. N

Las curvas de referencia para p = 0.5, 0.75, 0.9 y 0.95 se incluyen en (b) porque
la guia las pide explicitamente en el fundamento teorico (seccion 4.1).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: funciona en Colab y en servidor

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402
from .amdahl import curva_amdahl  # noqa: E402

COLOR_PANDAS = "#2E5E8C"
COLOR_SPARK = "#D96C3B"
COLOR_TEORICO = "#4B4B4B"

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.axisbelow": True,
    }
)


def figura_tiempos(tabla_speedup: pd.DataFrame) -> Path:
    """(a) Barras comparativas de tiempos, en escala logaritmica."""
    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(tabla_speedup))
    ancho = 0.38

    ax.bar(x - ancho / 2, tabla_speedup["T_pandas_s"], ancho,
           label="pandas (secuencial)", color=COLOR_PANDAS)
    ax.bar(x + ancho / 2, tabla_speedup["T_spark_s"], ancho,
           label=f"PySpark (local[{config.CORES_BASE}])", color=COLOR_SPARK)

    for i, fila in tabla_speedup.iterrows():
        altura = max(fila["T_pandas_s"], fila["T_spark_s"]) * 1.12
        ax.text(i, altura, f"S={fila['Speedup']:.2f}x", ha="center", fontsize=9,
                fontweight="bold",
                color="green" if fila["Speedup"] > 1 else "firebrick")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(tabla_speedup["TX"])
    ax.set_xlabel("Transformacion")
    ax.set_ylabel("Tiempo mediano (s, escala logaritmica)")
    ax.set_title("Tiempos de ejecucion: pandas vs. PySpark\n"
                 f"(mediana de {config.REPETICIONES} repeticiones)")
    ax.legend()
    fig.tight_layout()

    ruta = config.DIR_FIGS / "fig_a_tiempos.png"
    fig.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return ruta


def figura_speedup(tabla_escalado: pd.DataFrame, parametros: dict[str, float]) -> Path:
    """(b) Speedup medido vs. curva teorica de Amdahl."""
    p = parametros["p_paralelizable"]
    s_max = parametros["S_max"]
    n_continuo = np.linspace(1, 32, 400)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(n_continuo, curva_amdahl(p, n_continuo), "-", color=COLOR_TEORICO, lw=2,
            label=f"Amdahl teorico (p={p:.4f})")

    for p_ref, estilo in [(0.50, ":"), (0.75, "-."), (0.90, "--"),
                          (0.95, (0, (3, 1, 1, 1)))]:
        ax.plot(n_continuo, curva_amdahl(p_ref, n_continuo), ls=estilo, lw=1.1,
                alpha=0.55, label=f"p = {p_ref}")

    ax.plot(tabla_escalado["N"], tabla_escalado["S_medido"], "o-", color=COLOR_SPARK,
            ms=9, lw=2.2, label="Speedup medido (T3)")
    ax.plot(n_continuo, n_continuo, color="gray", lw=0.8, alpha=0.5,
            label="Speedup ideal (lineal)")
    ax.axhline(s_max, color="crimson", ls="--", lw=1.2, alpha=0.8)
    ax.text(1.4, s_max * 1.02, f"$S_{{max}}$ = {s_max:.2f}x", va="bottom", ha="left",
            color="crimson", fontsize=9)

    ax.set_xlabel("Numero de executors / procesadores $N$")
    ax.set_ylabel("Speedup $S(N)$")
    ax.set_title("Speedup experimental de T3 (join) vs. Ley de Amdahl")
    ax.set_xlim(1, 32)
    ax.set_ylim(0, max(6.0, min(20.0, s_max * 1.35)))
    ax.legend(fontsize=8.5, ncol=2)
    fig.tight_layout()

    ruta = config.DIR_FIGS / "fig_b_speedup_amdahl.png"
    fig.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return ruta


def figura_eficiencia(tabla_escalado: pd.DataFrame, parametros: dict[str, float]) -> Path:
    """(c) Eficiencia E = S/N medida vs. teorica."""
    p = parametros["p_paralelizable"]
    n_continuo = np.linspace(1, 32, 400)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(n_continuo, curva_amdahl(p, n_continuo) / n_continuo, "-",
            color=COLOR_TEORICO, lw=2, label="Eficiencia teorica")
    ax.plot(tabla_escalado["N"], tabla_escalado["Eficiencia"], "o-", color=COLOR_SPARK,
            ms=9, lw=2.2, label="Eficiencia medida (T3)")
    ax.axhline(1.0, color="gray", ls="--", lw=0.9, label="Eficiencia ideal (E=1)")
    ax.axhline(0.5, color="firebrick", ls=":", lw=1.0,
               label="Umbral de escalabilidad util (E=0.5)")

    ax.set_xlabel("Numero de executors / procesadores $N$")
    ax.set_ylabel("Eficiencia $E = S/N$")
    ax.set_title("Eficiencia del paralelismo vs. numero de procesadores")
    ax.set_xlim(1, 32)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9)
    fig.tight_layout()

    ruta = config.DIR_FIGS / "fig_c_eficiencia.png"
    fig.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return ruta


def generar_todas(
    tabla_speedup: pd.DataFrame,
    tabla_escalado: pd.DataFrame,
    parametros: dict[str, float],
    verbose: bool = True,
) -> list[Path]:
    """Genera las tres figuras y devuelve sus rutas."""
    rutas = [
        figura_tiempos(tabla_speedup),
        figura_speedup(tabla_escalado, parametros),
        figura_eficiencia(tabla_escalado, parametros),
    ]
    if verbose:
        for r in rutas:
            print(f"[figuras] {r}  (300 DPI)")
    return rutas
