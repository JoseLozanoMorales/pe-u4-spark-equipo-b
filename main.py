"""
Orquestador del experimento PE-U4 - Ley de Amdahl con Apache Spark.

Ejecuta de forma secuencial los cuatro pasos de la guia de practica:

    Paso 1 : generacion / carga del dataset y verificacion de esquema
    Paso 2 : cinco transformaciones secuenciales con pandas
    Paso 3 : las mismas cinco con PySpark + escalado de T3 (1, 2, 4 executors)
    Paso 4 : analisis de Amdahl, figuras a 300 DPI y tablas LaTeX

USO
---
    python main.py                    # corrida completa
    python main.py --solo-dataset     # regenera unicamente los CSV de entrada
    python main.py --sin-escalado     # omite el barrido de executors

Variables de entorno:
    PFC_SPARK_BASE    directorio de salida (por defecto ./salida_pe_u4)
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from core import amdahl, benchmark, config, dataset, exportar_latex, figuras
from core import transformaciones_pandas as tpandas
from core import transformaciones_spark as tspark


def separador(titulo: str) -> None:
    print(f"\n{'=' * 72}\n{titulo}\n{'=' * 72}")


def verificar_equivalencia(
    salidas_pandas: dict[str, Any],
    transformaciones_spark: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Compara filas, columnas y firma numerica entre ambos motores.

    La comparacion de firmas usa error RELATIVO, no absoluto. Al sumar cientos
    de miles de valores de punto flotante, pandas (orden secuencial) y Spark
    (agregacion por particiones) obtienen resultados que difieren en los ultimos
    digitos, porque la suma en coma flotante no es asociativa. Una tolerancia
    absoluta penalizaria injustamente a las columnas de mayor magnitud.
    """
    print(f"{'TX':<5}{'filas pandas':>15}{'filas spark':>14}{'cols':>11}"
          f"{'firma pandas':>18}{'firma spark':>18}{'err.rel':>12}   estado")

    verificacion: dict[str, dict[str, Any]] = {}
    for clave, df_pandas in salidas_pandas.items():
        df_spark = transformaciones_spark[clave]()

        filas_p, filas_s = len(df_pandas), df_spark.count()
        cols_p, cols_s = len(df_pandas.columns), len(df_spark.columns)
        firma_p = tpandas.firma(df_pandas, clave)
        firma_s = tspark.firma(df_spark, clave)

        escala = max(abs(firma_p), abs(firma_s), 1.0)
        error_rel = abs(firma_p - firma_s) / escala

        identico = (
            filas_p == filas_s
            and cols_p == cols_s
            and error_rel < config.TOLERANCIA_RELATIVA
        )
        verificacion[clave] = {
            "filas_pandas": filas_p,
            "filas_spark": filas_s,
            "columnas_pandas": cols_p,
            "columnas_spark": cols_s,
            "firma_pandas": firma_p,
            "firma_spark": firma_s,
            "error_relativo": error_rel,
            "identico": bool(identico),
        }
        print(f"{clave:<5}{filas_p:>15,}{filas_s:>14,}{cols_p:>6}/{cols_s:<4}"
              f"{firma_p:>18,.2f}{firma_s:>18,.2f}{error_rel:>12.2e}   "
              f"{'IDENTICO' if identico else 'REVISAR'}")
    return verificacion


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Experimento PE-U4 (Ley de Amdahl)")
    parser.add_argument("--solo-dataset", action="store_true",
                        help="genera unicamente los CSV de entrada y termina")
    parser.add_argument("--sin-escalado", action="store_true",
                        help="omite el barrido de 1/2/4 executors")
    args = parser.parse_args(argv)

    config.preparar_directorios()

    # ---------------------------------------------------------------- Paso 1
    separador("PASO 1 - Dataset y carga")
    tablas = dataset.cargar_pandas()
    if args.solo_dataset:
        return 0

    spark = tspark.crear_sesion(config.CORES_BASE)
    dfs_spark = tspark.cargar_spark(spark, config.CORES_BASE)

    print("\nVerificacion de carga pandas vs. Spark:")
    for nombre in config.RUTAS:
        n_p, n_s = len(tablas[nombre]), dfs_spark[nombre].count()
        estado = "OK" if n_p == n_s else "DIFIERE"
        print(f"  {nombre:<14} pandas={n_p:>8,}  spark={n_s:>8,}  {estado}")
    print()
    dfs_spark["solicitudes"].printSchema()

    # ---------------------------------------------------------------- Paso 2
    separador("PASO 2 - Implementacion secuencial con pandas")
    tx_pandas = tpandas.construir_transformaciones(
        tablas["solicitudes"], tablas["reservas"], tablas["laboratorios"],
        tablas["perfiles"], tablas["materias"],
    )
    resultados_pandas = benchmark.medir_conjunto(tx_pandas, prefijo="pandas")
    print()
    salidas_pandas = tpandas.persistir_resultados(tx_pandas)

    # ---------------------------------------------------------------- Paso 3
    separador(f"PASO 3 - Implementacion distribuida con PySpark (local[{config.CORES_BASE}])")
    tx_spark = tspark.construir_transformaciones(dfs_spark)
    resultados_spark = benchmark.medir_conjunto(tx_spark, prefijo="spark ")
    print()
    tspark.persistir_resultados(tx_spark)

    separador("PASO 3.5 - Verificacion de equivalencia de resultados")
    verificacion = verificar_equivalencia(salidas_pandas, tx_spark)

    separador("PASO 3.6 - Escalado de T3 con 1, 2 y 4 executors")
    if args.sin_escalado:
        print("Omitido por --sin-escalado (no se podra estimar p).")
        return 0
    escalado_t3 = tspark.medir_escalado_t3(benchmark.medir)

    # ---------------------------------------------------------------- Paso 4
    separador("PASO 4 - Analisis cuantitativo (Ley de Amdahl)")
    tabla_speedup = amdahl.speedup_por_transformacion(resultados_pandas, resultados_spark)
    tabla_esc = amdahl.tabla_escalado(escalado_t3)
    parametros = amdahl.estimar_parametros(tabla_esc)

    print(tabla_speedup[["TX", "Descripcion", "T_pandas_s", "T_spark_s",
                         "Speedup", "Ganancia"]].to_string(index=False))
    print()
    print(tabla_esc.to_string(index=False))
    amdahl.imprimir_resumen(parametros, tabla_speedup)

    separador("PASO 4.3 - Figuras y exportacion")
    figuras.generar_todas(tabla_speedup, tabla_esc, parametros)
    exportar_latex.exportar_todo(
        tabla_speedup, tabla_esc, parametros, verificacion,
        resultados_pandas, resultados_spark, escalado_t3,
        version_spark=spark.version,
    )

    print(f"\nExperimento completado. Salida en: {config.BASE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
