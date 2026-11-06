"""
Paso 1 del modelado: máscaras ground truth (vegetación = 1 / no vegetación = 0) por índice espectral.

Para cada escena de dataset/*.tiff:
  1. Se calculan NDVI, EVI, NDRE, CIre y NDMI (mismas fórmulas que analisis_preliminar.py).
  2. Se umbraliza cada índice: 1 si índice >= umbral (intervalo cerrado [umbral, máx]), 0 si índice < umbral, 255 si el píxel no es válido (SCL: nube,
     sombra, nieve, sin dato). El 255 se ignora en el entrenamiento y en las métricas.
     Clases SCL excluidas (criterio de analisis_preliminar.py): 0 sin dato, 1 saturado/defectuoso, 2 zonas oscuras,
     3 sombra de nube, 8–10 nube/cirro, 11 nieve. OJO: 5 escenas (ago-2026, todas de prueba) no traen SCL;
     en ellas la invalidez se estima con reglas espectrales (mascara_calidad).
  3. Se guarda la entrada de la red (RGB = B02, B03, B04 en uint8) y las 5 máscaras en un .npz por escena.

Umbrales:
  * NDVI = 0.5 fijo: sugerencia de la asesora (mencionó 0.5 y 0.6 como opciones). Es el mismo corte que la
    Sección III-D eligió entre los rangos de referencia de [17] (arbustos ≈0.2–0.3, bosque ≈0.6–0.8);
    [17] no propone 0.5 como tal. Como contraste, el Otsu del NDVI en entrenamiento se reporta en umbrales.json.
  * Resto de índices: umbral de Otsu calculado SOLO con píxeles de las escenas de entrenamiento (2015–2021),
    para no usar información del periodo de prueba. También se reporta el Otsu del NDVI como referencia.

Partición temporal (Sección III-E): entrenamiento 2015–2021, validación 2022, prueba 2023–2026.

Escenas sin banda SCL (12 bandas en lugar de 13) se descartan automáticamente y se listan en
<salida>/escenas_excluidas.txt (en la descarga usada: 5 escenas de agosto de 2026 → quedan 308).

Uso:  python generar_mascaras.py [--dataset dataset] [--salida datos]
      (salida: datos/escenas/*.npz, datos/umbrales.json, datos/escenas.csv, datos/escenas_excluidas.txt)
"""
import argparse
import glob
import json
import os
import zlib
from multiprocessing import Pool

import numpy as np
import pandas as pd

import rasterio

from analisis_preliminar import leer_escena, calcular_indices, mascara_calidad, fecha_de_archivo

INDICES = ["NDVI", "EVI", "NDRE", "CIre", "NDMI"]
UMBRAL_FIJO = {"NDVI": 0.5}
RGB_MAX = 0.4            # reflectancia que se mapea a 255 (escala lineal fija para todas las escenas)
IGNORAR = 255
SALIDA = "datos"
MUESTRA_OTSU = 20000     # píxeles válidos por escena de entrenamiento para estimar Otsu
SEMILLA = 0


def particion(fecha):
    if fecha.year <= 2021:
        return "train"
    if fecha.year == 2022:
        return "val"
    return "test"


def otsu(x, bins=512):
    """Umbral de Otsu sobre una muestra 1-D (maximiza la varianza entre clases)."""
    lo, hi = np.percentile(x, [0.5, 99.5])
    hist, bordes = np.histogram(np.clip(x, lo, hi), bins=bins, range=(lo, hi))
    centros = (bordes[:-1] + bordes[1:]) / 2
    p = hist / hist.sum()
    w0 = np.cumsum(p)
    mu = np.cumsum(p * centros)
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mu_t * w0 - mu) ** 2 / (w0 * (1 - w0))
    return float(centros[np.nanargmax(sb)])


def _cargar(path):
    r, sin_dato, _ = leer_escena(path)
    valido, *_ = mascara_calidad(r, sin_dato)
    idx = calcular_indices(r)
    for v in idx.values():
        valido &= np.isfinite(v)
    return r, valido, idx


def _muestra(path):
    _, valido, idx = _cargar(path)
    rng = np.random.default_rng(SEMILLA + zlib.crc32(os.path.basename(path).encode()))
    pos = np.flatnonzero(valido)
    pos = rng.choice(pos, size=min(MUESTRA_OTSU, pos.size), replace=False)
    return {k: idx[k].ravel()[pos] for k in INDICES}


def tiene_scl(path):
    """Las escenas válidas traen 13 bandas (12 espectrales + SCL). Sin SCL no hay máscara de nubes homogénea."""
    with rasterio.open(path) as src:
        return src.count == 13


def _procesar(args):
    path, umbrales, salida = args
    fecha = fecha_de_archivo(path)
    r, valido, idx = _cargar(path)
    rgb = np.stack([r["B04"], r["B03"], r["B02"]])  # orden R, G, B
    rgb = np.clip(np.nan_to_num(rgb) / RGB_MAX * 255, 0, 255).astype(np.uint8)
    mascaras, fila = {}, {"fecha": fecha.date().isoformat(), "particion": particion(fecha),
                         "frac_valido": float(valido.mean())}
    for k in INDICES:
        m = np.full(valido.shape, IGNORAR, np.uint8)
        m[valido] = (idx[k][valido] >= umbrales[k]).astype(np.uint8)  # vegetación = [umbral, máx]; sin veg. = [mín, umbral)
        mascaras[k] = m
        fila[f"veg_{k}"] = float(m[valido].mean()) if valido.any() else np.nan
    nombre = os.path.join(salida, "escenas", os.path.basename(path).replace(".tiff", ".npz"))
    np.savez_compressed(nombre, rgb=rgb, **mascaras)
    return fila


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset", help="carpeta con los GeoTIFF descargados")
    ap.add_argument("--salida", default=SALIDA, help="carpeta de salida (máscaras, umbrales, escenas.csv)")
    a = ap.parse_args()
    salida = a.salida
    os.makedirs(os.path.join(salida, "escenas"), exist_ok=True)
    todos = sorted(glob.glob(os.path.join(a.dataset, "*.tiff")))
    archivos = [p for p in todos if tiene_scl(p)]
    excluidas = sorted(set(todos) - set(archivos))
    print(f"{len(todos)} escenas encontradas; {len(excluidas)} excluidas por no traer banda SCL:",
          [os.path.basename(p) for p in excluidas])
    entreno = [p for p in archivos if particion(fecha_de_archivo(p)) == "train"]
    print(f"{len(archivos)} escenas; {len(entreno)} de entrenamiento para estimar umbrales")

    with Pool(6) as pool:
        muestras = pool.map(_muestra, entreno)
    umbrales, otsu_ref = {}, {}
    for k in INDICES:
        x = np.concatenate([m[k] for m in muestras])
        otsu_ref[k] = otsu(x)
        umbrales[k] = UMBRAL_FIJO.get(k, otsu_ref[k])
    # muestra guardada para graficar los histogramas con su umbral
    np.savez_compressed(os.path.join(salida, "muestra_indices_train.npz"),
                        **{k: np.concatenate([m[k] for m in muestras])[::5] for k in INDICES})
    with open(os.path.join(salida, "umbrales.json"), "w") as f:
        json.dump({"umbral_usado": umbrales, "otsu_train": otsu_ref,
                   "particion": {"train": "2015-2021", "val": "2022", "test": "2023-2026"}}, f, indent=2)
    print("Umbrales:", {k: round(v, 3) for k, v in umbrales.items()})
    print("Otsu ref:", {k: round(v, 3) for k, v in otsu_ref.items()})

    with Pool(6) as pool:
        filas = pool.map(_procesar, [(p, umbrales, salida) for p in archivos])
    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(salida, "escenas.csv"), index=False)
    with open(os.path.join(salida, "escenas_excluidas.txt"), "w") as f:
        f.write("# Escenas descartadas por no incluir la banda SCL (13 bandas esperadas)\n")
        f.writelines(os.path.basename(p) + "\n" for p in excluidas)
    print(df.groupby("particion").size().to_string())
    print(df[[f"veg_{k}" for k in INDICES]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
