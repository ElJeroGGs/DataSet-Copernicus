"""
Recalcula las cifras del reporte a partir de los archivos de métricas de cada modelo
(<carpeta>/<modelo>/{metricas.json, historial.csv, test_por_escena.csv}).

  * Tabla de prueba 2023–2026 por modelo: IoU, F1, precisión, recall, MCC, prevalencia.
  * Pruebas pareadas por escena (Wilcoxon bilateral, n = escenas de prueba):
      - mejor U-Net ResNet34 (por MCC) frente a las demás máscaras (MCC por escena);
      - U-Net frente a la línea base por píxel en NDVI (IoU por escena);
      - con frente a sin preentreno en NDVI (MCC por escena).
  * Diferencias entre encoders (EfficientNet-B0 − ResNet34).
  * Reproducibilidad entre dos carpetas de resultados (p. ej. GPU frente a CPU).

Uso:
  python analisis_resultados.py resultados/gpu [--comparar resultados/cpu] [--salida resultados]
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

COLS = ["tp", "fp", "fn", "tn"]


def mcc(c):
    tp, fp, fn, tn = (float(c[k]) for k in COLS)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / den if den else np.nan


def iou(c):
    return c["tp"] / max(c["tp"] + c["fp"] + c["fn"], 1)


def cargar(carpeta):
    modelos = {}
    for f in sorted(glob.glob(os.path.join(carpeta, "*", "metricas.json"))):
        d = os.path.dirname(f)
        modelos[os.path.basename(d)] = dict(
            metricas=json.load(open(f)),
            escenas=pd.read_csv(os.path.join(d, "test_por_escena.csv")),
            historial=pd.read_csv(os.path.join(d, "historial.csv")))
    return modelos


def tabla(modelos):
    filas = []
    for nombre, m in modelos.items():
        tot = m["escenas"][COLS].sum()
        t = m["metricas"]["test"]
        filas.append(dict(modelo=nombre, IoU=t["IoU"], F1=t["F1"], precision=t["precision"], recall=t["recall"],
                          MCC=mcc(tot), prevalencia=(tot.tp + tot.fn) / tot.sum(),
                          mejor_epoca=int(m["historial"].val_IoU.idxmax()) + 1, epocas=len(m["historial"])))
    return pd.DataFrame(filas).sort_values("MCC", ascending=False).reset_index(drop=True)


def por_escena(m, f):
    return m["escenas"].apply(f, axis=1).to_numpy()


def pareada(nombre_a, a, nombre_b, b, metrica, f):
    assert (a["escenas"].fecha.values == b["escenas"].fecha.values).all(), "las escenas de prueba no coinciden"
    xa, xb = por_escena(a, f), por_escena(b, f)
    d = xa - xb
    return dict(a=nombre_a, b=nombre_b, metrica=metrica, n=len(d), mediana_dif=float(np.median(d)),
                a_gana=int((d > 0).sum()), p_wilcoxon=float(wilcoxon(xa, xb).pvalue))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("carpeta")
    ap.add_argument("--comparar", help="otra carpeta de resultados para medir reproducibilidad")
    ap.add_argument("--salida", default=None, help="carpeta donde guardar los CSV (por defecto, la misma)")
    a = ap.parse_args()
    salida = a.salida or a.carpeta
    os.makedirs(salida, exist_ok=True)
    mod = cargar(a.carpeta)
    tab = tabla(mod)
    pd.set_option("display.width", 160)
    print("== Prueba 2023–2026 ==")
    print(tab.round(4).to_string(index=False))
    tab.to_csv(os.path.join(salida, "tabla_modelos.csv"), index=False)

    pruebas = []
    r34 = [n for n in tab.modelo if n.endswith("_unet_resnet34")]
    if r34:
        mejor = r34[0]
        for n in r34[1:]:
            pruebas.append(pareada(mejor, mod[mejor], n, mod[n], "MCC", mcc))
    if "NDVI_unet_resnet34" in mod and "NDVI_pixel" in mod:
        pruebas.append(pareada("NDVI_unet_resnet34", mod["NDVI_unet_resnet34"], "NDVI_pixel", mod["NDVI_pixel"], "IoU", iou))
    if "NDVI_unet_resnet34_sin_preentreno" in mod and "NDVI_unet_resnet34" in mod:
        pruebas.append(pareada("NDVI_unet_resnet34_sin_preentreno", mod["NDVI_unet_resnet34_sin_preentreno"],
                               "NDVI_unet_resnet34", mod["NDVI_unet_resnet34"], "MCC", mcc))
    if pruebas:
        pr = pd.DataFrame(pruebas)
        print("\n== Pruebas pareadas por escena (Wilcoxon bilateral) ==")
        print(pr.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        pr.to_csv(os.path.join(salida, "pruebas_pareadas.csv"), index=False)

    enc = []
    for ind in ["NDVI", "EVI", "NDRE", "CIre", "NDMI"]:
        x, y = f"{ind}_unet_resnet34", f"{ind}_unet_efficientnet-b0"
        if x in mod and y in mod:
            fx, fy = tab.set_index("modelo").loc[x], tab.set_index("modelo").loc[y]
            enc.append(dict(indice=ind, dMCC=fy.MCC - fx.MCC, dIoU=fy.IoU - fx.IoU))
    if enc:
        print("\n== EfficientNet-B0 − ResNet34 ==")
        print(pd.DataFrame(enc).round(4).to_string(index=False))

    if a.comparar:
        otro = cargar(a.comparar)
        rep = [dict(modelo=n, IoU_a=mod[n]["metricas"]["test"]["IoU"], IoU_b=otro[n]["metricas"]["test"]["IoU"])
               for n in mod if n in otro]
        rep = pd.DataFrame(rep)
        rep["diferencia"] = rep.IoU_a - rep.IoU_b
        print(f"\n== Reproducibilidad: IoU {a.carpeta} − {a.comparar} ==")
        print(rep.round(4).to_string(index=False))
        rep.to_csv(os.path.join(salida, "reproducibilidad.csv"), index=False)


if __name__ == "__main__":
    main()
