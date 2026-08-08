"""
Configuracion central del experimento PE-U4 (Ley de Amdahl con Apache Spark).

Todos los parametros del experimento viven aqui para garantizar reproducibilidad:
cambiar un valor en este archivo reconfigura la corrida completa.

Asignatura : Aplicaciones Distribuidas [20701] - 7mo Nivel A
Dominio    : SCLI - Sistema de Control de Laboratorios Institucional
Esquema    : replica el modelo relacional real de los microservicios del PFC
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Reproducibilidad
# --------------------------------------------------------------------------
SEMILLA: int = 20260808

# --------------------------------------------------------------------------
# Volumetria del dataset sintetico
# La guia exige >= 500,000 registros en la tabla de hechos.
# --------------------------------------------------------------------------
N_RESERVAS: int = 520_000      # filas de solicitudes_reserva (tabla de hechos)
N_LABORATORIOS: int = 120
N_USUARIOS: int = 15_000       # perfiles

# --------------------------------------------------------------------------
# Protocolo de medicion
# --------------------------------------------------------------------------
REPETICIONES: int = 5   # exigido por la guia -> se reporta la MEDIANA
WARMUP: int = 1         # ejecucion de calentamiento NO medida (JIT, cache)

# --------------------------------------------------------------------------
# Paralelismo
# --------------------------------------------------------------------------
CORES_BASE: int = 4
CORES_TEST: tuple[int, ...] = (1, 2, 4)

# CONTROL EXPERIMENTAL CRITICO
# ----------------------------
# El numero de particiones se mantiene CONSTANTE en todo el barrido de
# executors. Si las particiones variaran con N, la configuracion N=1 quedaria
# con una sola particion y resolveria el join SIN shuffle, mientras que N=2 y
# N=4 si lo pagarian: se estarian comparando algoritmos distintos, no el mismo
# programa con mas procesadores, que es lo que exige la Ley de Amdahl.
PARTICIONES_FIJAS: int = 8

# --------------------------------------------------------------------------
# Tolerancia para verificar equivalencia pandas <-> Spark
# Se usa error RELATIVO: la suma de cientos de miles de dobles difiere en los
# ultimos digitos entre ambos motores porque la suma de punto flotante no es
# asociativa y Spark agrega por particiones en orden distinto a pandas.
# --------------------------------------------------------------------------
TOLERANCIA_RELATIVA: float = 1e-6

# --------------------------------------------------------------------------
# Rutas de salida
# Sobrescribibles con la variable de entorno PFC_SPARK_BASE
# --------------------------------------------------------------------------
BASE = Path(os.environ.get("PFC_SPARK_BASE", "./salida_pe_u4")).resolve()

DIR_DATA = BASE / "data"
DIR_PANDAS = DIR_DATA / "pandas"
DIR_SPARK = DIR_DATA / "spark"
DIR_FIGS = BASE / "figs"
DIR_TEX = BASE / "tex"

RUTAS: dict[str, Path] = {
    "solicitudes": DIR_DATA / "solicitudes_reserva.csv",
    "reservas": DIR_DATA / "reservas.csv",
    "laboratorios": DIR_DATA / "laboratorios.csv",
    "perfiles": DIR_DATA / "perfiles.csv",
    "materias": DIR_DATA / "materias.csv",
}

# --------------------------------------------------------------------------
# Metadatos de las transformaciones
# --------------------------------------------------------------------------
DESCRIPCION: dict[str, str] = {
    "T1": "Filtrado por condicion compuesta",
    "T2": "Groupby + agregacion (5 funciones)",
    "T3": "Join de 4 DataFrames por UUID",
    "T4": "Columna derivada compleja",
    "T5": "Ordenamiento y top-100",
}

# Columna representativa usada para verificar equivalencia pandas <-> Spark
COLUMNA_FIRMA: dict[str, str] = {
    "T1": "numero_participantes",
    "T2": "participantes_tot",
    "T3": "numero_participantes",
    "T4": "horas_laboratorio",
    "T5": "numero_participantes",
}


def preparar_directorios() -> None:
    """Crea el arbol de directorios de salida si no existe."""
    for d in (DIR_DATA, DIR_PANDAS, DIR_SPARK, DIR_FIGS, DIR_TEX):
        d.mkdir(parents=True, exist_ok=True)
