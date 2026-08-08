"""
Instrumentacion de medicion de tiempos.

PROTOCOLO
---------
`time.perf_counter()` es el reloj monotono de mayor resolucion disponible en
Python (nanosegundos); los resultados se reportan en microsegundos.

Por cada transformacion se ejecuta:
    1 corrida de calentamiento NO medida (amortiza JIT de la JVM, llenado de
      cache de Spark y page cache del sistema operativo)
  + config.REPETICIONES corridas medidas

De las corridas medidas se reporta la MEDIANA, que la guia exige explicitamente
y que es robusta frente a valores atipicos causados por el recolector de basura
o por el planificador del sistema operativo. Se acompana del coeficiente de
variacion (CV) como indicador de estabilidad de la medicion.
"""

from __future__ import annotations

import statistics
import time
from typing import Any, Callable

from . import config


def medir(
    fn: Callable[[], Any],
    repeticiones: int | None = None,
    warmup: int | None = None,
    etiqueta: str = "",
) -> dict[str, Any]:
    """Ejecuta `fn` y devuelve las estadisticas de tiempo de ejecucion."""
    repeticiones = config.REPETICIONES if repeticiones is None else repeticiones
    warmup = config.WARMUP if warmup is None else warmup

    for _ in range(warmup):
        fn()

    tiempos: list[float] = []
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        fn()
        tiempos.append(time.perf_counter() - t0)

    media = statistics.mean(tiempos)
    resultado: dict[str, Any] = {
        "mediana_s": statistics.median(tiempos),
        "media_s": media,
        "min_s": min(tiempos),
        "max_s": max(tiempos),
        "desv_s": statistics.pstdev(tiempos),
        "tiempos_s": [round(t, 6) for t in tiempos],
        "repeticiones": repeticiones,
    }
    resultado["cv_pct"] = (100 * resultado["desv_s"] / media) if media else 0.0

    if etiqueta:
        print(
            f"{etiqueta:<40s} mediana={resultado['mediana_s'] * 1e6:>12,.0f} us "
            f"({resultado['mediana_s']:.4f} s)  CV={resultado['cv_pct']:.1f}%"
        )
    return resultado


def medir_conjunto(
    transformaciones: dict[str, Callable[[], Any]], prefijo: str = ""
) -> dict[str, dict[str, Any]]:
    """Aplica `medir` a un diccionario de transformaciones {clave: callable}."""
    resultados: dict[str, dict[str, Any]] = {}
    for clave, fn in transformaciones.items():
        etiqueta = f"[{prefijo}] {clave} {config.DESCRIPCION.get(clave, '')}"
        resultados[clave] = medir(fn, etiqueta=etiqueta)
    return resultados
