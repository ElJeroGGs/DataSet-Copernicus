"""
Comprueba empíricamente que los GeoTIFF descargados ya vienen armonizados respecto al cambio de línea base de
procesamiento 04.00 de Sentinel-2 (25-ene-2022, offset radiométrico de −1000 DN en los productos L2A).

Si el offset NO estuviera corregido, después de esa fecha el DN mínimo de los píxeles válidos subiría a ~1000 y el
percentil 1 de reflectancia aumentaría ~0.1. Se comparan escenas de temporada seca (dic–abr) antes y después.

Uso:  python verificaciones/verificar_armonizacion.py [carpeta_dataset]      (por defecto: dataset)
Resultado observado (308 escenas con SCL): DN mínimo = 1 en ambos grupos; percentil 1 de B02/B04 ≈ 0.016–0.020
en ambos → sin salto, no hace falta corregir.
"""
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import rasterio

BANDAS = {"B02": 2, "B04": 4, "B08": 8, "B11": 11}   # índice 1-based en el GeoTIFF (orden B01..B12, SCL)
SCL_VALIDAS = [4, 5, 6, 7]                           # vegetación, suelo, agua, sin clasificar
CAMBIO = pd.Timestamp("2022-01-25")


def resumen(path):
    with rasterio.open(path) as s:
        if s.count < 13:
            return None
        a = s.read(list(BANDAS.values()) + [13]).astype(np.float32)
    scl, a = a[-1], a[:-1]
    v = np.isin(scl, SCL_VALIDAS) & np.all(a > 0, axis=0)
    fila = {"fecha": pd.Timestamp(os.path.basename(path).split("_")[1][:10]), "min_DN_B02": float(a[0][a[0] > 0].min())}
    for k, b in enumerate(BANDAS):
        fila[f"p1_{b}"], fila[f"p50_{b}"] = np.percentile(a[k][v], [1, 50]) / 1e4
    return fila


def main():
    carpeta = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    with Pool(6) as pool:
        df = pd.DataFrame([r for r in pool.map(resumen, sorted(glob.glob(os.path.join(carpeta, "*.tiff")))) if r])
    df["despues_del_cambio"] = df.fecha >= CAMBIO
    seca = df[df.fecha.dt.month.isin([12, 1, 2, 3, 4])]
    cols = [c for c in df.columns if c.startswith(("p1_", "p50_"))] + ["min_DN_B02"]
    print(f"{len(df)} escenas con SCL; temporada seca: {len(seca)}")
    print("Mediana por grupo (temporada seca):")
    print(seca.groupby("despues_del_cambio")[cols].median().round(4).T.to_string())
    salto = (seca.groupby("despues_del_cambio")["p1_B04"].median().diff().iloc[-1])
    print(f"\nSalto del percentil 1 de B04 en el cambio: {salto:+.4f} (un offset sin corregir daría ≈ +0.1)")


if __name__ == "__main__":
    main()
