"""
Paso 1 de la guia - Generacion y carga del dataset del dominio SCLI.

FIDELIDAD AL MODELO DEL PFC
---------------------------
El esquema replica el modelo relacional real de los microservicios del PFC,
tomado de sus scripts de migracion Flyway:

  reservas-solicitudes-service/.../V1__crear_esquema_inicial.sql
      -> solicitudes_reserva, reservas, historial_solicitudes
  academico-laboratorios-service/.../V1__crear_esquema_academico_laboratorios.sql
      -> facultades, carreras, materias, periodos_lectivos, campus, bloques,
         pisos, laboratorios
  usuarios-service/.../V1__crear_esquema_usuarios.sql
      -> perfiles, docentes, estudiantes, tecnicos

Se conservan los identificadores UUID, los estados exactos definidos en las
restricciones CHECK, los nombres de columna originales y las relaciones entre
entidades. El ciclo de vida del dominio tambien se respeta: una `reserva` solo
existe cuando su solicitud fue APROBADA (relacion 1:1, uq_reservas_solicitud_id).

FUENTE Y LICENCIA DE LOS DATOS
------------------------------
El PFC no dispone de datos operativos con el volumen requerido (>= 500,000
registros) y no existe un dataset publico abierto de reservas de laboratorios
universitarios en Kaggle, datos.gob.ec ni INEC. La guia autoriza el uso de datos
sinteticos generados con Faker documentando fuente y licencia.

  - Generador  : Faker (locale es_MX, licencia MIT) para catalogos nominales
                 + NumPy default_rng para variables numericas y temporales.
  - Semilla    : config.SEMILLA -> reproducible bit a bit.
  - Privacidad : datos sinteticos, sin informacion personal real (LOPDP).

NOTA SOBRE LOS UUID
-------------------
Las claves son UUID en texto de 36 caracteres, como en el PFC. Esto encarece
los joins frente a claves enteras (36 bytes por clave contra 4), lo que resulta
pertinente para el experimento: es el tipo de carga donde el procesamiento
distribuido tiene mas margen para compensar su overhead.
"""

from __future__ import annotations

import time
import uuid

import numpy as np
import pandas as pd
from faker import Faker

from . import config

# --------------------------------------------------------------------------
# Catalogos del dominio (valores exactos de las restricciones CHECK del PFC)
# --------------------------------------------------------------------------
ESTADOS_SOLICITUD = ["PENDIENTE", "EN_REVISION", "APROBADA", "RECHAZADA",
                     "CANCELADA", "EXPIRADA"]
ESTADOS_RESERVA = ["PROGRAMADA", "EN_CURSO", "FINALIZADA", "CANCELADA",
                   "NO_ASISTIDA"]
ESTADOS_LABORATORIO = ["DISPONIBLE", "OCUPADO", "MANTENIMIENTO", "INACTIVO"]

FACULTADES = [
    ("FCI", "Facultad de Ciencias de la Ingenieria"),
    ("FCP", "Facultad de Ciencias Pecuarias"),
    ("FCA", "Facultad de Ciencias Agrarias"),
    ("FCE", "Facultad de Ciencias Empresariales"),
    ("FCS", "Facultad de Ciencias de la Salud"),
]
CARRERAS = ["Software", "Telematica", "Agroindustria", "Veterinaria",
            "Agronomia", "Administracion", "Contabilidad", "Enfermeria",
            "Economia", "Marketing"]
MATERIAS = ["Aplicaciones Distribuidas", "Base de Datos", "Redes de Computadoras",
            "Inteligencia Artificial", "Calidad de Software", "Sistemas Operativos",
            "Bioquimica", "Microbiologia", "Estadistica", "Fisiologia Animal",
            "Nutricion", "Contabilidad de Costos"]
MOTIVOS = ["Practica de laboratorio de la unidad",
           "Clase regular de la asignatura",
           "Desarrollo de proyecto de titulacion",
           "Actividad de investigacion formativa",
           "Capacitacion docente",
           "Examen practico de fin de unidad",
           "Refuerzo academico para estudiantes"]
TIPOS_CONTRATO = ["TITULAR", "OCASIONAL", "CONTRATADO"]
DEDICACIONES = ["TIEMPO_COMPLETO", "MEDIO_TIEMPO", "TIEMPO_PARCIAL"]
TIPOS_LAB = ["COMPUTO", "REDES", "ELECTRONICA", "QUIMICA", "BIOTECNOLOGIA",
             "SIMULACION"]
CAMPUS = ["Campus Central", "Campus La Maria", "Campus Finca Experimental"]
BLOQUES = ["Bloque A", "Bloque B", "Bloque C", "Bloque D", "Bloque E"]


def _uuids(n: int, rng: np.random.Generator) -> np.ndarray:
    """Genera n UUID v4 en texto de forma DETERMINISTA a partir de `rng`.

    Se construyen desde bytes del generador con semilla en vez de usar
    uuid.uuid4(), que no seria reproducible entre corridas.
    """
    crudos = rng.integers(0, 256, size=(n, 16), dtype=np.uint8)
    return np.array([str(uuid.UUID(bytes=bytes(fila))) for fila in crudos])


# --------------------------------------------------------------------------
# Dimensiones: academico-laboratorios-service
# --------------------------------------------------------------------------
def generar_dimensiones_academicas(rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    """facultades, carreras, materias y periodos_lectivos."""
    n_fac = len(FACULTADES)
    facultades = pd.DataFrame({
        "facultad_id": _uuids(n_fac, rng),
        "codigo_facultad": [c for c, _ in FACULTADES],
        "nombre_facultad": [n for _, n in FACULTADES],
    })

    n_car = len(CARRERAS)
    carreras = pd.DataFrame({
        "carrera_id": _uuids(n_car, rng),
        "facultad_id": rng.choice(facultades["facultad_id"].to_numpy(), n_car),
        "codigo_carrera": [f"CAR-{i:03d}" for i in range(1, n_car + 1)],
        "nombre_carrera": CARRERAS,
    })

    n_mat = len(MATERIAS)
    materias = pd.DataFrame({
        "materia_id": _uuids(n_mat, rng),
        # Nombre distinto de perfiles.carrera_id: evita columnas ambiguas en T3
        "carrera_materia_id": rng.choice(carreras["carrera_id"].to_numpy(), n_mat),
        "codigo_materia": [f"MAT-{i:03d}" for i in range(1, n_mat + 1)],
        "nombre_materia": MATERIAS,
        "numero_horas": rng.choice([48, 64, 80, 96], n_mat).astype(np.int32),
    })

    codigos_periodo = ["2023-1", "2023-2", "2024-1", "2024-2", "2025-1", "2025-2"]
    periodos = pd.DataFrame({
        "periodo_lectivo_id": _uuids(len(codigos_periodo), rng),
        "codigo_periodo": codigos_periodo,
        "nombre_periodo": [f"Periodo Academico {c}" for c in codigos_periodo],
        "estado_periodo": ["FINALIZADO"] * 5 + ["ACTIVO"],
    })

    return {"facultades": facultades, "carreras": carreras,
            "materias": materias, "periodos_lectivos": periodos}


def generar_laboratorios(rng: np.random.Generator,
                         facultades: pd.DataFrame) -> pd.DataFrame:
    """Laboratorios con la jerarquia campus -> bloques -> pisos desnormalizada.

    La jerarquia se aplana en columnas porque el experimento mide operaciones
    analiticas, no navegacion relacional.
    """
    n = config.N_LABORATORIOS
    tipos = rng.choice(TIPOS_LAB, n)
    return pd.DataFrame({
        "laboratorio_id": _uuids(n, rng),
        "piso_id": _uuids(n, rng),
        "codigo_laboratorio": [f"LAB-{t[:3]}-{i:03d}"
                               for i, t in zip(range(1, n + 1), tipos)],
        "nombre_laboratorio": [f"Laboratorio de {t.capitalize()} {i:03d}"
                               for i, t in zip(range(1, n + 1), tipos)],
        "tipo_laboratorio": tipos,
        "capacidad": rng.integers(15, 61, n).astype(np.int32),
        "estado_laboratorio": rng.choice(ESTADOS_LABORATORIO, n,
                                         p=[0.78, 0.10, 0.08, 0.04]),
        "campus": rng.choice(CAMPUS, n, p=[0.70, 0.20, 0.10]),
        "bloque": rng.choice(BLOQUES, n),
        "numero_piso": rng.integers(1, 5, n).astype(np.int32),
        "facultad_id": rng.choice(facultades["facultad_id"].to_numpy(), n),
    })


# --------------------------------------------------------------------------
# Dimension: usuarios-service
# --------------------------------------------------------------------------
def generar_perfiles(rng: np.random.Generator, fake: Faker,
                     carreras: pd.DataFrame) -> pd.DataFrame:
    """Perfiles con el rol resuelto y los atributos propios de cada tipo.

    En el PFC los roles viven en tablas hijas (docentes, estudiantes, tecnicos,
    administradores) con FK a perfiles. Aqui se aplanan en una tabla unica con
    la columna `rol`, conservando los atributos especificos de cada tipo.
    """
    n = config.N_USUARIOS
    nombres = [fake.first_name() for _ in range(n)]
    apellidos = [fake.last_name() for _ in range(n)]
    roles = rng.choice(["ESTUDIANTE", "DOCENTE", "TECNICO", "ADMINISTRADOR"],
                       n, p=[0.78, 0.16, 0.04, 0.02])

    es_estudiante = roles == "ESTUDIANTE"
    es_docente = roles == "DOCENTE"
    ids_carrera = rng.choice(carreras["carrera_id"].to_numpy(), n)

    return pd.DataFrame({
        "perfil_id": _uuids(n, rng),
        "identificacion": rng.integers(1_000_000_000, 2_399_999_999, n).astype(str),
        "nombres": nombres,
        "apellidos": apellidos,
        "email_institucional": [
            f"{no.split()[0].lower()}.{ap.split()[0].lower()}{i}@uteq.edu.ec"
            for i, (no, ap) in enumerate(zip(nombres, apellidos))
        ],
        "rol": roles,
        "carrera_id": np.where(es_estudiante, ids_carrera, "SIN_CARRERA"),
        "semestre": np.where(es_estudiante, rng.integers(1, 11, n), 0).astype(np.int32),
        "departamento": np.where(es_docente,
                                 rng.choice([nm for _, nm in FACULTADES], n),
                                 "NO_APLICA"),
        "tipo_contrato": np.where(es_docente, rng.choice(TIPOS_CONTRATO, n),
                                  "NO_APLICA"),
        "dedicacion": np.where(es_docente, rng.choice(DEDICACIONES, n),
                               "NO_APLICA"),
    })


# --------------------------------------------------------------------------
# Hechos: reservas-solicitudes-service
# --------------------------------------------------------------------------
def generar_solicitudes(rng: np.random.Generator, laboratorios: pd.DataFrame,
                        perfiles: pd.DataFrame, materias: pd.DataFrame,
                        periodos: pd.DataFrame) -> pd.DataFrame:
    """Tabla de hechos `solicitudes_reserva` (>= 500,000 filas).

    Respeta ck_solicitudes_reserva_horas (hora_fin > hora_inicio) y
    ck_solicitudes_reserva_participantes_positivos.
    """
    n = config.N_RESERVAS

    docentes = perfiles.loc[perfiles["rol"] == "DOCENTE", "perfil_id"].to_numpy()
    if len(docentes) == 0:
        docentes = perfiles["perfil_id"].to_numpy()

    # Franja horaria academica de 07:00 a 22:00
    hora_inicio = rng.integers(7, 19, n).astype(np.int32)
    duracion = rng.choice([1, 2, 3, 4], n, p=[0.30, 0.42, 0.20, 0.08])
    hora_fin = np.minimum(hora_inicio + duracion, 22).astype(np.int32)
    hora_fin = np.where(hora_fin <= hora_inicio, hora_inicio + 1, hora_fin)

    inicio = np.datetime64("2023-01-01")
    dias = rng.integers(0, 3 * 365, n)

    return pd.DataFrame({
        "solicitud_id": _uuids(n, rng),
        "solicitante_id": rng.choice(perfiles["perfil_id"].to_numpy(), n),
        "docente_id": rng.choice(docentes, n),
        "laboratorio_id": rng.choice(laboratorios["laboratorio_id"].to_numpy(), n),
        "materia_id": rng.choice(materias["materia_id"].to_numpy(), n),
        "periodo_lectivo_id": rng.choice(periodos["periodo_lectivo_id"].to_numpy(), n),
        "fecha_reserva": inicio + dias.astype("timedelta64[D]"),
        "hora_inicio": hora_inicio,
        "hora_fin": hora_fin,
        "numero_participantes": rng.integers(5, 51, n).astype(np.int32),
        "motivo": rng.choice(MOTIVOS, n),
        "estado_solicitud": rng.choice(ESTADOS_SOLICITUD, n,
                                       p=[0.10, 0.07, 0.55, 0.12, 0.11, 0.05]),
    })


def generar_reservas(rng: np.random.Generator,
                     solicitudes: pd.DataFrame) -> pd.DataFrame:
    """Tabla `reservas`: solo para solicitudes APROBADAS (relacion 1:1).

    Reproduce la restriccion uq_reservas_solicitud_id del PFC.
    """
    aprobadas = solicitudes.loc[
        solicitudes["estado_solicitud"] == "APROBADA",
        ["solicitud_id", "laboratorio_id", "docente_id",
         "fecha_reserva", "hora_inicio", "hora_fin"],
    ].reset_index(drop=True)

    n = len(aprobadas)
    return pd.DataFrame({
        "reserva_id": _uuids(n, rng),
        "solicitud_id": aprobadas["solicitud_id"],
        "laboratorio_id": aprobadas["laboratorio_id"],
        "responsable_id": aprobadas["docente_id"],
        "fecha_reserva": aprobadas["fecha_reserva"],
        "hora_inicio": aprobadas["hora_inicio"],
        "hora_fin": aprobadas["hora_fin"],
        "codigo_reserva": [f"RES-{i:08d}" for i in range(1, n + 1)],
        "estado_reserva": rng.choice(ESTADOS_RESERVA, n,
                                     p=[0.12, 0.03, 0.68, 0.10, 0.07]),
    })


# --------------------------------------------------------------------------
# Orquestacion
# --------------------------------------------------------------------------
def generar_dataset(verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Genera todas las tablas del dominio y las persiste en CSV."""
    config.preparar_directorios()

    rng = np.random.default_rng(config.SEMILLA)
    fake = Faker("es_MX")
    Faker.seed(config.SEMILLA)

    t0 = time.perf_counter()
    academicas = generar_dimensiones_academicas(rng)
    laboratorios = generar_laboratorios(rng, academicas["facultades"])
    perfiles = generar_perfiles(rng, fake, academicas["carreras"])
    solicitudes = generar_solicitudes(rng, laboratorios, perfiles,
                                      academicas["materias"],
                                      academicas["periodos_lectivos"])
    reservas = generar_reservas(rng, solicitudes)
    elapsed = time.perf_counter() - t0

    tablas = {
        "solicitudes": solicitudes,
        "reservas": reservas,
        "laboratorios": laboratorios,
        "perfiles": perfiles,
        "materias": academicas["materias"],
    }

    for nombre, df in tablas.items():
        df.to_csv(config.RUTAS[nombre], index=False)

    if verbose:
        mb = config.RUTAS["solicitudes"].stat().st_size / 1e6
        print(f"[dataset] generado en {elapsed:.2f} s")
        for nombre, df in tablas.items():
            extra = f"   ({mb:.1f} MB)" if nombre == "solicitudes" else ""
            print(f"[dataset] {nombre:<14} {str(df.shape):>16}{extra}")

    return tablas


def cargar_pandas(verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Carga las tablas desde CSV en pandas (las genera si no existen)."""
    if not config.RUTAS["solicitudes"].exists():
        return generar_dataset(verbose=verbose)

    con_fecha = ("solicitudes", "reservas")
    tablas = {
        nombre: pd.read_csv(
            ruta,
            parse_dates=["fecha_reserva"] if nombre in con_fecha else None,
        )
        for nombre, ruta in config.RUTAS.items()
    }
    if verbose:
        for nombre, df in tablas.items():
            print(f"[pandas] {nombre:<14} {df.shape}")
    return tablas