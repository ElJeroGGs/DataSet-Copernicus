"""
Figuras de las máscaras ground truth y de los resultados de los modelos (salidas/mascaras/).

  figM1_histogramas_umbral.png   Distribución de cada índice (entrenamiento) y umbral usado.
  figM2_ejemplo_mascaras.png     Una escena: RGB + las 5 máscaras.
  figM3_fraccion_vegetacion.png  Fracción de píxeles con vegetación por año según cada máscara.
  figM4_predicciones_<m>_<origen>.png  RGB | máscara real | predicción | errores (por modelo entrenado).

Uso:  python graficas_mascaras.py modelos_gpu     (o modelos_cpu, o resultados/gpu: sin figuras M4)
  figM5_metricas.png             IoU/F1 en prueba por modelo y por año.
  metricas_modelos.csv           Tabla resumen de prueba.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from analisis_preliminar import AZUL, NARANJA, AQUA, AMARILLO, TINTA2, MUTED, GRID, CRITICO  # aplica el mismo estilo

INDICES = ["NDVI", "EVI", "NDRE", "CIre", "NDMI"]
COLOR = {"NDVI": AQUA, "EVI": AMARILLO, "NDRE": AZUL, "CIre": NARANJA, "NDMI": "#7a5bd6"}
OUT = os.path.join("salidas", "mascaras")
# 0 = sin vegetación, 1 = vegetación, 2 = inválido (nube/sombra/nieve)
CMAP_MASK = ListedColormap(["#f3efe4", "#2e8b3a", "#b9b8b1"])
LEYENDA_MASK = [Patch(color="#2e8b3a", label="Vegetación (1)"), Patch(color="#f3efe4", label="Sin vegetación (0)"),
                Patch(color="#b9b8b1", label="Inválido (nube/sombra)")]


def a_cmap(m):
    m = m.astype(np.int16)
    m[m == 255] = 2
    return m


def rgb_vis(rgb):
    x = np.moveaxis(rgb, 0, -1).astype(np.float32) / 255.0
    return np.clip(x * 2.2, 0, 1) ** 0.8  # realce sólo para visualizar


def fig_histogramas(umb):
    z = np.load("datos/muestra_indices_train.npz")
    fig, axs = plt.subplots(1, 5, figsize=(15, 2.8))
    for ax, k in zip(axs, INDICES):
        x = z[k]
        lo, hi = np.percentile(x, [0.5, 99.5])
        ax.hist(x, bins=120, range=(lo, hi), color=COLOR[k], alpha=0.85)
        ax.axvline(umb["umbral_usado"][k], color=CRITICO, lw=1.4)
        fuente = "fijo" if k == "NDVI" else "Otsu"
        ax.set_title(f"{k}  (umbral {fuente} = {umb['umbral_usado'][k]:.2f})")
        ax.set_yticks([])
        ax.text(0.98, 0.95, f"{(x > umb['umbral_usado'][k]).mean()*100:.0f} % > umbral", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color=TINTA2)
    fig.suptitle("Distribución de los índices en píxeles válidos de entrenamiento (2015–2021) y umbral de la máscara",
                 x=0.01, y=1.08, ha="left", fontweight="bold")
    fig.savefig(os.path.join(OUT, "figM1_histogramas_umbral.png"))
    plt.close(fig)


def fig_ejemplo(esc):
    # escena de prueba de temporada seca con más píxeles válidos
    cand = esc[(esc.particion == "test") & esc.fecha.str[5:7].isin(["02", "03", "04"])]
    fecha = cand.sort_values("frac_valido", ascending=False).iloc[0].fecha
    z = np.load(f"datos/escenas/toluca_{fecha}.npz")
    fig, axs = plt.subplots(1, 6, figsize=(16, 4.6))
    axs[0].imshow(rgb_vis(z["rgb"]))
    axs[0].set_title(f"RGB (entrada)  {fecha}")
    for ax, k in zip(axs[1:], INDICES):
        m = z[k]
        ax.imshow(a_cmap(m), cmap=CMAP_MASK, vmin=0, vmax=2, interpolation="nearest")
        ax.set_title(f"Máscara {k}  ({(m[m != 255] == 1).mean()*100:.0f} % veg.)")
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.legend(handles=LEYENDA_MASK, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.savefig(os.path.join(OUT, "figM2_ejemplo_mascaras.png"))
    plt.close(fig)


def fig_fraccion(esc):
    esc = esc.copy()
    esc["anio"] = esc.fecha.str[:4].astype(int)
    g = esc.groupby("anio")[[f"veg_{k}" for k in INDICES]].median() * 100
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.axvspan(2021.5, 2022.5, color=GRID, alpha=0.6, lw=0)
    ax.axvspan(2022.5, g.index.max() + 0.5, color=GRID, alpha=0.3, lw=0)
    for k in INDICES:
        ax.plot(g.index, g[f"veg_{k}"], marker="o", ms=3.5, color=COLOR[k], label=k)
    y = ax.get_ylim()[1]
    for x, t in [(2018, "entrenamiento"), (2022, "val."), (2024.7, "prueba")]:
        ax.text(x, y, t, ha="center", va="top", fontsize=8, color=MUTED)
    ax.set_ylabel("% de píxeles con vegetación\n(mediana de las escenas del año)")
    ax.set_xticks(g.index)
    ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    ax.set_title("Fracción de vegetación según cada máscara")
    fig.savefig(os.path.join(OUT, "figM3_fraccion_vegetacion.png"))
    plt.close(fig)


def etiqueta(nombre):
    """NDVI_unet_resnet34 -> 'NDVI · U-Net R34'; NDVI_pixel -> 'NDVI · por píxel'."""
    partes = nombre.split("_")
    txt = partes[0] + " · " + ("por píxel" if partes[1] == "pixel" else "U-Net " +
                               partes[2].replace("resnet", "R").replace("efficientnet-", "Eff-"))
    return txt + (" (sin preentreno)" if nombre.endswith("sin_preentreno") else "")


def mcc_de(c):
    tp, fp, fn, tn = (float(c[k]) for k in ("tp", "fp", "fn", "tn"))
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / den if den else np.nan


def fig_predicciones(carpeta, sufijo):
    nombre = os.path.basename(carpeta)
    z = np.load(os.path.join(carpeta, "ejemplos_test.npz"))
    n = len(z["fechas"])
    fig, axs = plt.subplots(n, 4, figsize=(9.6, 4.1 * n))
    fig.subplots_adjust(wspace=0.04, hspace=0.12, top=0.95, bottom=0.04)
    cmap_err = ListedColormap(["#f3efe4", "#2e8b3a", NARANJA, AZUL, "#b9b8b1"])  # TN, TP, FP, FN, inválido
    for i in range(n):
        y, p = z["y"][i], z["prob"][i] > 0.5
        err = np.zeros(y.shape, np.int16)
        err[(y == 1) & p] = 1
        err[(y == 0) & p] = 2
        err[(y == 1) & ~p] = 3
        err[y == 255] = 4
        pm = p.astype(np.int16)
        pm[y == 255] = 2
        axs[i, 0].imshow(rgb_vis(z["rgb"][i])); axs[i, 0].set_title(f"RGB {z['fechas'][i]}")
        axs[i, 1].imshow(a_cmap(y), cmap=CMAP_MASK, vmin=0, vmax=2, interpolation="nearest"); axs[i, 1].set_title("Máscara real")
        axs[i, 2].imshow(pm, cmap=CMAP_MASK, vmin=0, vmax=2, interpolation="nearest"); axs[i, 2].set_title("Predicción")
        axs[i, 3].imshow(err, cmap=cmap_err, vmin=0, vmax=4, interpolation="nearest"); axs[i, 3].set_title("Errores")
    for ax in axs.ravel():
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.legend(handles=[Patch(color="#2e8b3a", label="Acierto vegetación"), Patch(color="#f3efe4", label="Acierto sin veg."),
                        Patch(color=NARANJA, label="Falso positivo"), Patch(color=AZUL, label="Falso negativo"),
                        Patch(color="#b9b8b1", label="Inválido")], loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{etiqueta(nombre)}: escenas de prueba (2023–2026)", x=0.02, y=0.985, ha="left", fontweight="bold")
    fig.savefig(os.path.join(OUT, f"figM4_predicciones_{nombre}{sufijo}.png"))
    plt.close(fig)


def fig_metricas(res, sufijo, carpeta):
    filas = []
    for r in res:
        t = pd.read_csv(os.path.join(carpeta, r["modelo"], "test_por_escena.csv"))
        tot = t[["tp", "fp", "fn", "tn"]].sum()
        filas.append(dict(modelo=r["modelo"], **{k: round(v, 4) for k, v in r["test"].items()},
                          MCC=round(mcc_de(tot), 4), prevalencia=round((tot.tp + tot.fn) / tot.sum(), 4),
                          **{f"MCC_{a}": round(mcc_de(g[["tp", "fp", "fn", "tn"]].sum()), 4) for a, g in t.groupby("anio")}))
    tabla = pd.DataFrame(filas)
    tabla.to_csv(os.path.join(OUT, f"metricas_modelos{sufijo}.csv"), index=False)
    print(tabla.drop(columns=[c for c in tabla if c.startswith("MCC_")]).to_string(index=False))
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.2), gridspec_kw=dict(width_ratios=[1.4, 1, 1]))
    x = np.arange(len(tabla))
    for k, (m, c) in enumerate([("IoU", AZUL), ("F1", AQUA), ("MCC", NARANJA)]):
        axs[0].bar(x + (k - 1) * 0.27, tabla[m], 0.27, label=m, color=c)
    axs[0].set_xticks(x, [etiqueta(m) for m in tabla.modelo], rotation=25, ha="right")
    axs[0].set_ylim(0, 1.12); axs[0].set_yticks(np.arange(0, 1.01, 0.2))
    axs[0].legend(ncol=3, loc="upper center"); axs[0].set_title("Prueba 2023–2026 (todas las escenas)")
    for r, (_, f) in zip(res, tabla.iterrows()):
        anios = sorted(int(a) for a in r["test_por_anio"])
        estilo = dict(marker="o", ls="--" if "pixel" in r["modelo"] else ("-." if "efficientnet" in r["modelo"] else "-"),
                      color=COLOR[r["indice"]], mfc="white" if "pixel" in r["modelo"] or "sin_preentreno" in r["modelo"] else None)
        axs[1].plot(anios, [r["test_por_anio"][str(a)]["IoU"] for a in anios], label=etiqueta(r["modelo"]), **estilo)
        axs[2].plot(anios, [f[f"MCC_{a}"] for a in anios], label=etiqueta(r["modelo"]), **estilo)
    for ax, m in [(axs[1], "IoU"), (axs[2], "MCC")]:
        ax.set_xticks(anios); ax.set_ylabel(m); ax.set_title(f"{m} por año de prueba")
    axs[2].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.savefig(os.path.join(OUT, f"figM5_metricas{sufijo}.png"))
    plt.close(fig)


def fig_encoders(tabla, sufijo):
    """M6: misma tarea, dos encoders preentrenados (ResNet34 vs EfficientNet-B0), MCC e IoU por índice."""
    filas = []
    for k in INDICES:
        a = tabla[tabla.modelo == f"{k}_unet_resnet34"]
        b = tabla[tabla.modelo == f"{k}_unet_efficientnet-b0"]
        if len(a) and len(b):
            filas.append((k, a.iloc[0], b.iloc[0]))
    if not filas:
        return
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    x = np.arange(len(filas))
    for ax, m in zip(axs, ["MCC", "IoU"]):
        ax.bar(x - 0.2, [f[1][m] for f in filas], 0.4, label="ResNet34", color=AZUL)
        ax.bar(x + 0.2, [f[2][m] for f in filas], 0.4, label="EfficientNet-B0", color=NARANJA)
        for i, f in enumerate(filas):
            ax.text(i, max(f[1][m], f[2][m]) + 0.008, f"Δ {round(f[2][m] - f[1][m], 3) + 0.0:+.3f}", ha="center", fontsize=8, color=TINTA2)
        ax.set_xticks(x, [f[0] for f in filas]); ax.set_ylim(0.6, 0.85); ax.set_ylabel(m)
        ax.set_title(f"{m} en prueba 2023–2026")
    axs[0].legend(loc="upper right")
    fig.suptitle("Comparación de encoder preentrenado (misma U-Net, misma configuración)", x=0.01, ha="left", fontweight="bold", y=1.03)
    fig.savefig(os.path.join(OUT, f"figM6_encoders{sufijo}.png"))
    plt.close(fig)


def fig_ablacion(carpeta, sufijo):
    """M7: curvas de validación con y sin pesos de ImageNet (NDVI, ResNet34)."""
    a = os.path.join(carpeta, "NDVI_unet_resnet34", "historial.csv")
    b = os.path.join(carpeta, "NDVI_unet_resnet34_sin_preentreno", "historial.csv")
    if not (os.path.exists(a) and os.path.exists(b)):
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for f, lab, c in [(a, "Con pesos de ImageNet", AZUL), (b, "Sin preentreno (pesos aleatorios)", NARANJA)]:
        h = pd.read_csv(f)
        ax.plot(h.epoca, h.val_IoU, marker="o", color=c, label=lab)
    ax.set_xlabel("Época"); ax.set_ylabel("IoU validación (2022)"); ax.set_xticks(range(1, 11))
    ax.set_title("NDVI · U-Net R34: efecto del preentreno"); ax.legend()
    fig.savefig(os.path.join(OUT, f"figM7_ablacion_preentreno{sufijo}.png"))
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    umb = json.load(open("datos/umbrales.json"))
    esc = pd.read_csv("datos/escenas.csv")
    fig_histogramas(umb)
    fig_ejemplo(esc)
    fig_fraccion(esc)
    carpeta = sys.argv[1] if len(sys.argv) > 1 else "modelos"   # modelos_gpu, modelos_cpu, resultados/gpu…
    base = os.path.basename(os.path.normpath(carpeta))
    sufijo = "_" + (base.split("_", 1)[1] if base.startswith("modelos_") else base) if base != "modelos" else ""
    res = []
    for c in sorted(glob.glob(os.path.join(carpeta, "*"))):
        if os.path.exists(os.path.join(c, "metricas.json")):
            res.append(json.load(open(os.path.join(c, "metricas.json"))))
            if os.path.exists(os.path.join(c, "ejemplos_test.npz")):   # no se versiona (pesa ~40 MB por modelo)
                fig_predicciones(c, sufijo)
    if res:
        principal = [r for r in res if r["modelo"].endswith("_unet_resnet34") or r["modelo"].endswith("_pixel")]
        fig_metricas(res, sufijo + "_todos", carpeta)        # tabla CSV completa (y figura de referencia)
        fig_metricas(principal, sufijo, carpeta)             # M5: comparativa principal
        fig_encoders(pd.read_csv(os.path.join(OUT, f"metricas_modelos{sufijo}_todos.csv")), sufijo)
        fig_ablacion(carpeta, sufijo)
    print("figuras en", OUT)


if __name__ == "__main__":
    main()
