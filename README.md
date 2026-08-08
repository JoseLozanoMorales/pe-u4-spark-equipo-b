# PE-U4 - Procesamiento distribuido con Apache Spark

Comprobación experimental de la Ley de Amdahl aplicada al PFC **FUVV - Sistema de Control de Laboratorios e Infraestructura**.

## Equipo

| Integrante | PFC de origen | Rol |
|---|---|---|
| Lozano Morales José Alejandro | AGLS - TiendaTech | Coordinador de experimentación y reproducibilidad |
| Sánchez Pilaloa Andy Paul | AGLS - TiendaTech | Responsable de datos y documentación científica |
| Urbina Romero Isaías Abraham | FUVV - Laboratorios Informáticos | Responsable de PySpark y verificación |

## Decisión de negocio

El pipeline resume la demanda y utilización de los laboratorios para apoyar la asignación de horarios, detectar periodos de alta ocupación y priorizar ampliaciones de capacidad. Se usan datos sintéticos porque no se dispone de una base institucional anonimizada y con licencia pública. El generador mantiene relaciones consistentes, utiliza una semilla fija y queda publicado para garantizar la reproducibilidad.

## Inicio rápido en Google Colab

1. Subir o clonar el proyecto.
2. Instalar las dependencias: `pip install -r requirements.txt`.
3. Generar los datos: `python src/generar_dataset.py`.
4. Ejecutar el experimento: `python src/ejecutar_experimento.py`.
5. Generar las figuras: `python src/graficas.py`.

El notebook `notebooks/PE_U4_pipeline_spark.ipynb` organiza el mismo flujo en celdas y debe entregarse con todas sus salidas visibles.

## Protocolo experimental

- Cinco repeticiones cronometradas por transformación.
- Una ejecución de calentamiento descartada y declarada.
- Medición con `time.perf_counter()`.
- Materialización explícita en PySpark mediante `count()`.
- Mediana como estadístico principal.
- Escalado con 1, 2 y 4 unidades locales de procesamiento para T1, T2 y T3; T3 es el mínimo obligatorio.
- Verificación por cardinalidad y huellas agregadas de control.

## Compilación del informe

Desde `docs/`:

```text
pdflatex PE_U4_Informe.tex
biber PE_U4_Informe
pdflatex PE_U4_Informe.tex
pdflatex PE_U4_Informe.tex
```

Antes de entregar deben reemplazarse todos los marcadores `PENDIENTE` con resultados reales y comprobarse la compilación desde una clonación limpia.

## Datos sintéticos

Los archivos generados no se presentan como datos reales. La semilla, el volumen, el esquema y las reglas de generación se documentan en `data/README_dataset.md`. Por su tamaño, los CSV pueden regenerarse y no es obligatorio versionarlos si el repositorio documenta el procedimiento exacto.
