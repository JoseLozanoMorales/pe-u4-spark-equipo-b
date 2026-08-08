# PE-U4 — Procesamiento distribuido con Apache Spark

Comprobación experimental de la **Ley de Amdahl** mediante la implementación y
medición de un pipeline de datos ejecutado dos veces: de forma secuencial con
pandas y de forma distribuida con PySpark.

**Asignatura:** Aplicaciones Distribuidas [20701] — 7mo Nivel A
**Docente:** Ing. Guerrero Ulloa Gleiston Cicerón
**Dominio:** SCLI — Sistema de Control de Laboratorios e Infraestructura

## Equipo

| Integrante | PFC de origen | Rol |
|---|---|---|
| Lozano Morales José Alejandro | AGLS - TiendaTech | Coordinador de experimentación y reproducibilidad |
| Sánchez Pilaloa Andy Paul | AGLS - TiendaTech | Responsable de datos y documentación científica |
| Urbina Romero Isaías Abraham | FUVV - Laboratorios Informáticos | Responsable de PySpark y verificación |

## Decisión de negocio

El pipeline resume la demanda y utilización de los laboratorios para apoyar la
asignación de horarios, detectar periodos de alta ocupación y priorizar
ampliaciones de capacidad. Se usan datos sintéticos porque no se dispone de una
base institucional anonimizada y con licencia pública. El generador mantiene
relaciones consistentes, utiliza una semilla fija y queda publicado para
garantizar la reproducibilidad.

## Dataset

El esquema replica el **modelo relacional real** del PFC, tomado de sus scripts
de migración Flyway:

| Microservicio | Entidades replicadas |
|---|---|
| `reservas-solicitudes-service` | `solicitudes_reserva`, `reservas` |
| `academico-laboratorios-service` | `laboratorios`, `materias` |
| `usuarios-service` | `perfiles` |

Se conservan las claves UUID, los estados definidos en las restricciones CHECK
(`PENDIENTE`, `EN_REVISION`, `APROBADA`, `RECHAZADA`, `CANCELADA`, `EXPIRADA`
para solicitudes; `PROGRAMADA`, `EN_CURSO`, `FINALIZADA`, `CANCELADA`,
`NO_ASISTIDA` para reservas) y el ciclo de vida del dominio: una reserva solo
existe cuando su solicitud fue aprobada, con relación 1:1.

| Tabla | Filas | Rol |
|---|---|---|
| `solicitudes_reserva` | 520 000 | Hechos |
| `reservas` | 286 216 | Hechos derivados |
| `perfiles` | 15 000 | Dimensión |
| `laboratorios` | 120 | Dimensión |
| `materias` | 12 | Dimensión |

**Generador:** Faker (licencia MIT) para catálogos nominales y NumPy
`default_rng` para variables numéricas y temporales. **Semilla fija 20260808**,
por lo que el dataset es reproducible bit a bit en cualquier máquina. No
contiene información personal real, en cumplimiento de la LOPDP del Ecuador.

Por su tamaño (~148 MB) los CSV no se versionan: se regeneran ejecutando el
proyecto, y el procedimiento exacto queda documentado en el código.

## Protocolo experimental

- Cinco repeticiones cronometradas por transformación.
- Una ejecución de calentamiento descartada y declarada, que amortiza la
  compilación JIT de la JVM y el llenado de caché.
- Medición con `time.perf_counter()`, resolución de nanosegundos.
- **Materialización explícita en PySpark mediante el sink `noop`.** No se usa
  `count()` porque el optimizador Catalyst poda las columnas no referenciadas:
  en la transformación de columnas derivadas, `count()` descartaría el cálculo
  y registraría un tiempo prácticamente nulo.
- Mediana como estadístico principal, acompañada del coeficiente de variación.
- Escalado con 1, 2 y 4 unidades locales de procesamiento sobre T3 (join), que
  es la transformación con mayor componente de shuffle.
- **Número de particiones constante durante todo el barrido.** Si las
  particiones variaran con el número de unidades, la configuración N=1 quedaría
  con una sola partición y resolvería el join sin shuffle: se compararían
  algoritmos distintos en vez del mismo programa con más procesadores, que es lo
  que modela la Ley de Amdahl.
- Verificación por cardinalidad, número de columnas y huellas agregadas de
  control, con tolerancia relativa. La suma en coma flotante no es asociativa y
  Spark agrega por particiones en orden distinto a pandas, por lo que una
  tolerancia absoluta penalizaría a las columnas de mayor magnitud.

## Transformaciones

| ID | Operación | Consulta del dominio |
|---|---|---|
| T1 | Filtrado por condición compuesta | Solicitudes aprobadas de grupos grandes en jornada vespertina |
| T2 | Agrupación + 5 funciones de agregación | Indicadores de uso por laboratorio y estado |
| T3 | Join de 4 DataFrames por UUID | Vista analítica con laboratorio, solicitante y materia |
| T4 | Columnas derivadas | Duración, franja horaria y nivel de demanda |
| T5 | Ordenamiento multiclave + top-N | Ranking de solicitudes de mayor demanda |

## Ejecución en Google Colab

El entorno recomendado es Google Colab, que provee Java y Hadoop ya
configurados. El notebook `notebooks/EJECUTAR_EN_COLAB.ipynb` organiza el flujo
completo en celdas y debe entregarse con todas sus salidas visibles.

1. Comprimir `core/`, `main.py` y `requirements.txt` en un archivo ZIP.
2. Abrir el notebook en Colab y ejecutar las celdas en orden.
3. La celda 2 solicita el ZIP del proyecto.
4. La celda 5 ejecuta el experimento completo (aproximadamente 9 minutos).
5. La celda 7 exporta los resultados.

### Ejecución local

Requiere Python 3.11 y Java 11 o superior.

```bash
pip install -r requirements.txt
python main.py
```

| Argumento | Efecto |
|---|---|
| `--solo-dataset` | Regenera únicamente los CSV de entrada |
| `--sin-escalado` | Omite el barrido de unidades de procesamiento |

En Windows, la escritura de resultados requiere `winutils.exe` y la variable
`HADOOP_HOME`; por ese motivo se recomienda Colab.

## Estructura

```
pe-u4-spark/
├── core/
│   ├── config.py                    Parámetros del experimento
│   ├── dataset.py                   Generación y carga del dataset
│   ├── benchmark.py                 Instrumentación de medición
│   ├── transformaciones_pandas.py   Implementación secuencial
│   ├── transformaciones_spark.py    Implementación distribuida
│   ├── amdahl.py                    Análisis cuantitativo
│   ├── figuras.py                   Figuras en PNG a 300 DPI
│   └── exportar_latex.py            Tablas booktabs y resumen JSON
├── notebooks/                       Notebook de ejecución en Colab
├── evidencia/                       Capturas y registro de la corrida
├── salida_pe_u4/
│   ├── figs/                        Figuras del informe
│   ├── tex/tablas.tex               Tablas listas para LaTeX
│   └── resultados.json              Tiempos medidos y parámetros
├── main.py                          Orquestador
└── requirements.txt
```

## Resultados

Las tablas del informe se generan automáticamente a partir de los tiempos
medidos, sin transcripción manual, de modo que los valores documentados
coinciden por construcción con los que produjo la ejecución.

```latex
\usepackage{booktabs}
\input{tex/tablas.tex}
```

Las figuras se exportan en PNG a 300 DPI:

- `fig_a_tiempos.png` — comparación de tiempos pandas frente a PySpark
- `fig_b_speedup_amdahl.png` — speedup medido frente a la curva teórica
- `fig_c_eficiencia.png` — eficiencia del paralelismo

## Limitaciones experimentales

1. **Modo local, no clúster distribuido.** El driver aloja los hilos de
   ejecución, por lo que N corresponde al número de unidades de procesamiento
   concurrentes. No existe latencia de red entre nodos, de modo que la fracción
   serial medida constituye un límite inferior respecto de un clúster real.
2. **Dos núcleos físicos en la máquina virtual asignada.** La configuración N=4
   implica sobresuscripción: cuatro hilos compitiendo por dos CPU. El punto N=4
   no representa paralelismo adicional real.
3. **Volumen moderado.** A 520 000 registros el costo fijo de Spark resulta
   comparable al trabajo útil, lo que explica los valores de speedup inferiores
   a la unidad en las transformaciones ligeras.
4. **Datos sintéticos con distribuciones uniformes**, que no reproducen el sesgo
   presente en datos operativos reales, uno de los factores que degrada el
   escalado en producción.

## Compilación del informe

Desde `docs/`:

```bash
pdflatex PE_U4_Informe.tex
biber PE_U4_Informe
pdflatex PE_U4_Informe.tex
pdflatex PE_U4_Informe.tex
```

Antes de entregar deben reemplazarse todos los marcadores PENDIENTE con
resultados reales y comprobarse la compilación desde una clonación limpia.