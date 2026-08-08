"""
Paso 3 de la guia - Implementacion distribuida con PySpark.

DECISIONES DE CONFIGURACION (deben justificarse en el informe)
--------------------------------------------------------------
1. Esquema explicito: se evita `inferSchema`, que lanza un job adicional de
   lectura completa del CSV y contaminaria las mediciones.

2. `spark.sql.adaptive.enabled=false`: con AQE activo Spark reajusta en tiempo
   de ejecucion el numero de particiones de shuffle, ignorando el valor que
   fijamos por configuracion. Eso invalidaria el barrido de 1/2/4 executors.

3. `spark.sql.autoBroadcastJoinThreshold=-1`: sin esto Spark difundiria las
   dimensiones pequenas (broadcast join) y T3 dejaria de ejercitar el shuffle,
   que es justamente la parte no paralelizable que queremos medir.

4. PARTICIONADO CONSTANTE (`config.PARTICIONES_FIJAS`): el numero de
   particiones NO varia con N. Si variara, la configuracion N=1 quedaria con
   una sola particion y resolveria el join sin shuffle, mientras que N=2 y N=4
   si lo pagarian: se compararian algoritmos distintos en vez del mismo
   programa con mas procesadores, que es lo que modela la Ley de Amdahl.

5. Materializacion con el sink `noop`: la evaluacion perezosa exige una accion.
   NO se usa `count()` porque el optimizador Catalyst PODA las columnas no
   referenciadas, con lo que T4 (columnas derivadas) mediria practicamente
   cero. `write.format("noop")` evalua todas las columnas sin escribir a disco.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DateType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from . import config

# --------------------------------------------------------------------------
# Esquemas explicitos (reflejan el modelo relacional del PFC)
# --------------------------------------------------------------------------
ESQUEMA_SOLICITUDES = StructType([
    StructField("solicitud_id", StringType(), False),
    StructField("solicitante_id", StringType(), False),
    StructField("docente_id", StringType(), False),
    StructField("laboratorio_id", StringType(), False),
    StructField("materia_id", StringType(), False),
    StructField("periodo_lectivo_id", StringType(), False),
    StructField("fecha_reserva", DateType(), True),
    StructField("hora_inicio", IntegerType(), True),
    StructField("hora_fin", IntegerType(), True),
    StructField("numero_participantes", IntegerType(), True),
    StructField("motivo", StringType(), True),
    StructField("estado_solicitud", StringType(), True),
])

ESQUEMA_RESERVAS = StructType([
    StructField("reserva_id", StringType(), False),
    StructField("solicitud_id", StringType(), False),
    StructField("laboratorio_id", StringType(), False),
    StructField("responsable_id", StringType(), False),
    StructField("fecha_reserva", DateType(), True),
    StructField("hora_inicio", IntegerType(), True),
    StructField("hora_fin", IntegerType(), True),
    StructField("codigo_reserva", StringType(), True),
    StructField("estado_reserva", StringType(), True),
])

ESQUEMA_LABORATORIOS = StructType([
    StructField("laboratorio_id", StringType(), False),
    StructField("piso_id", StringType(), True),
    StructField("codigo_laboratorio", StringType(), True),
    StructField("nombre_laboratorio", StringType(), True),
    StructField("tipo_laboratorio", StringType(), True),
    StructField("capacidad", IntegerType(), True),
    StructField("estado_laboratorio", StringType(), True),
    StructField("campus", StringType(), True),
    StructField("bloque", StringType(), True),
    StructField("numero_piso", IntegerType(), True),
    StructField("facultad_id", StringType(), True),
])

ESQUEMA_PERFILES = StructType([
    StructField("perfil_id", StringType(), False),
    StructField("identificacion", StringType(), True),
    StructField("nombres", StringType(), True),
    StructField("apellidos", StringType(), True),
    StructField("email_institucional", StringType(), True),
    StructField("rol", StringType(), True),
    StructField("carrera_id", StringType(), True),
    StructField("semestre", IntegerType(), True),
    StructField("departamento", StringType(), True),
    StructField("tipo_contrato", StringType(), True),
    StructField("dedicacion", StringType(), True),
])

ESQUEMA_MATERIAS = StructType([
    StructField("materia_id", StringType(), False),
    StructField("carrera_materia_id", StringType(), True),
    StructField("codigo_materia", StringType(), True),
    StructField("nombre_materia", StringType(), True),
    StructField("numero_horas", IntegerType(), True),
])

ESQUEMAS = {
    "solicitudes": ESQUEMA_SOLICITUDES,
    "reservas": ESQUEMA_RESERVAS,
    "laboratorios": ESQUEMA_LABORATORIOS,
    "perfiles": ESQUEMA_PERFILES,
    "materias": ESQUEMA_MATERIAS,
}


# --------------------------------------------------------------------------
# Sesion
# --------------------------------------------------------------------------
def crear_sesion(n_cores: int, app: str = "PE-U4-SCLI") -> SparkSession:
    """Detiene la sesion activa y crea una nueva con `local[n_cores]`.

    En modo local el driver aloja los hilos de ejecucion, por lo que N equivale
    funcionalmente a N executors de un core. Debe declararse como limitacion
    experimental en el informe.

    Notese que `shuffle.partitions` usa PARTICIONES_FIJAS y no n_cores: lo unico
    que varia entre configuraciones es cuantas particiones se procesan a la vez.
    """
    activa = SparkSession.getActiveSession()
    if activa is not None:
        activa.stop()
        time.sleep(3)

    sesion = (
        SparkSession.builder.master(f"local[{n_cores}]")
        .appName(f"{app}-{n_cores}c")
        .config("spark.executor.instances", str(n_cores))
        .config("spark.default.parallelism", str(config.PARTICIONES_FIJAS))
        .config("spark.sql.shuffle.partitions", str(config.PARTICIONES_FIJAS))
        .config("spark.driver.memory", "10g")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .getOrCreate()
    )
    sesion.sparkContext.setLogLevel("ERROR")
    return sesion


def cargar_spark(spark: SparkSession, n_cores: int) -> dict[str, DataFrame]:
    """Lee los CSV con esquema explicito, reparticiona a un numero FIJO y cachea.

    El parametro `n_cores` se conserva por simetria con la API anterior, pero el
    particionado deliberadamente NO depende de el (ver decision 4 del modulo).
    """
    dfs: dict[str, DataFrame] = {}
    for nombre, ruta in config.RUTAS.items():
        df = (
            spark.read.schema(ESQUEMAS[nombre])
            .option("header", True)
            .csv(str(ruta))
            .repartition(config.PARTICIONES_FIJAS)
            .cache()
        )
        df.count()  # materializa la cache antes de medir
        dfs[nombre] = df
    return dfs


def materializar(df: DataFrame) -> None:
    """Fuerza la evaluacion completa de todas las columnas sin escribir a disco."""
    df.write.mode("overwrite").format("noop").save()


# --------------------------------------------------------------------------
# Transformaciones
# --------------------------------------------------------------------------
def construir_transformaciones(dfs: dict[str, DataFrame]) -> dict[str, Callable[[], DataFrame]]:
    """Devuelve {clave: callable} con las cinco transformaciones PySpark.

    Son las mismas operaciones de `transformaciones_pandas`, expresadas con la
    DataFrame API. Cada callable dispara una accion para que el tiempo medido
    corresponda a la ejecucion real y no a la construccion del plan logico.
    """
    solicitudes = dfs["solicitudes"]
    laboratorios = dfs["laboratorios"]
    perfiles = dfs["perfiles"]
    materias = dfs["materias"]

    def t1() -> DataFrame:
        df = solicitudes.filter(
            (F.col("estado_solicitud") == "APROBADA")
            & (F.col("numero_participantes") >= 20)
            & (F.col("hora_inicio") >= 14)
            & ((F.col("hora_fin") - F.col("hora_inicio")) >= 2)
        )
        materializar(df)
        return df

    def t2() -> DataFrame:
        df = solicitudes.groupBy("laboratorio_id", "estado_solicitud").agg(
            F.count("solicitud_id").alias("total_solicitudes"),
            F.sum("numero_participantes").alias("participantes_tot"),
            F.avg("numero_participantes").alias("participantes_prom"),
            F.min("hora_inicio").alias("hora_inicio_min"),
            F.max("hora_fin").alias("hora_fin_max"),
        )
        df.collect()  # resultado pequeno (~720 filas)
        return df

    def t3() -> DataFrame:
        perfiles_sol = perfiles.withColumnRenamed("perfil_id", "solicitante_id")
        df = (
            solicitudes.join(laboratorios, on="laboratorio_id", how="inner")
            .join(perfiles_sol, on="solicitante_id", how="inner")
            .join(materias, on="materia_id", how="inner")
        )
        materializar(df)
        return df

    def t4() -> DataFrame:
        recargo = (
            F.when(F.col("hora_inicio") >= 18, F.lit(1.25))
            .when(F.col("hora_inicio") < 9, F.lit(0.90))
            .otherwise(F.lit(1.00))
        )
        df = (
            solicitudes.withColumn(
                "horas_laboratorio",
                (F.col("hora_fin") - F.col("hora_inicio")).cast("double"),
            )
            .withColumn(
                "franja",
                F.when(F.col("hora_inicio") < 12, "MATUTINA")
                .when(F.col("hora_inicio") < 18, "VESPERTINA")
                .otherwise("NOCTURNA"),
            )
            .withColumn(
                "participantes_hora",
                F.round(
                    F.col("numero_participantes") * F.col("horas_laboratorio") * recargo,
                    4,
                ),
            )
            .withColumn(
                "nivel_demanda",
                F.when(F.col("participantes_hora") < 30, "BAJA")
                .when(F.col("participantes_hora") < 80, "MEDIA")
                .when(F.col("participantes_hora") < 150, "ALTA")
                .otherwise("CRITICA"),
            )
        )
        materializar(df)
        return df

    def t5() -> DataFrame:
        df = solicitudes.orderBy(
            F.col("numero_participantes").desc(),
            F.col("hora_fin").desc(),
            F.col("fecha_reserva").desc(),
        ).limit(100)
        df.collect()
        return df

    return {"T1": t1, "T2": t2, "T3": t3, "T4": t4, "T5": t5}


def persistir_resultados(
    transformaciones: dict[str, Callable[[], DataFrame]], verbose: bool = True
) -> None:
    """Escribe el resultado de cada transformacion en /data/spark/."""
    for clave, fn in transformaciones.items():
        df = fn()
        ruta = config.DIR_SPARK / f"{clave}_resultado"
        df.coalesce(1).write.mode("overwrite").option("header", True).csv(str(ruta))
        if verbose:
            print(f"[spark ] {clave}: {df.count():>8,} filas -> {ruta}")


def firma(df: DataFrame, clave: str) -> float:
    """Suma de la columna representativa; se compara contra la firma de pandas."""
    columna = config.COLUMNA_FIRMA[clave]
    valor = df.agg(F.sum(columna)).collect()[0][0]
    return round(float(valor), 2)


def medir_escalado_t3(medir_fn: Callable[..., dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Mide T3 con 1, 2 y 4 executors recreando la sesion en cada configuracion.

    Experimento nucleo del analisis de Amdahl: la sesion debe destruirse y
    recrearse porque `local[N]` queda fijado al construir el SparkContext.
    El particionado se mantiene constante en las tres configuraciones.
    """
    escalado: dict[int, dict[str, Any]] = {}
    for n in config.CORES_TEST:
        spark = crear_sesion(n)
        dfs = cargar_spark(spark, n)
        transformaciones = construir_transformaciones(dfs)
        escalado[n] = medir_fn(
            transformaciones["T3"], etiqueta=f"[spark ] T3 con N={n} executor(s)"
        )
        print(
            f"         particiones={spark.conf.get('spark.sql.shuffle.partitions')} "
            f"(fijas) | master={spark.sparkContext.master}"
        )
    return escalado
