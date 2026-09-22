"""
analisis_preliminar.py
======================
Análisis exploratorio y resultados preliminares del dataset Sentinel-2 L2A de Toluca
(archivos generados por download_timeseries.py -> dataset/toluca_YYYY-MM-DD.tiff).

Qué produce (carpeta de salida, por defecto ./salidas):
  inventario_escenas.csv      1 fila por escena: calidad (píxeles válidos, nubes, nieve) + estadísticos de índices
  resumen.json / resumen.txt  Números clave para la presentación
  fig01_inventario.png        Escenas por año/mes y fracción de píxeles limpios
  fig02_firmas_espectrales.png Firma espectral media por clase de cobertura
  fig03_correlacion.png       Correlación entre bandas e índices (redundancia)
  fig04_serie_temporal.png    Series NDVI / NDRE / CIre / NDMI por clase de cobertura
  fig05_estacionalidad.png    Climatología mensual (fenología) de NDVI y NDRE
  fig06_anomalias.png         Serie desestacionalizada + tendencia de Sen / Mann-Kendall
  fig07_mapas_ndvi.png        NDVI temporada seca: primer año vs último año y diferencia
  fig08_mapa_tendencia.png    Pendiente de Sen por píxel (NDVI/año) + significancia MK
  fig09_ci_vs_ndvi.png        Trayectorias CIre(t1)/CIre(t2) vs NDVI(t1)/NDVI(t2) (Zarco-Tejada et al., 2018)
  fig10_puntuacion_salud.png  Puntuación preliminar de salud (anomalía estandarizada) por año
  fig11_rgb.png               Color verdadero de la primera y última escena más limpias

Uso:
  pip install rasterio numpy pandas matplotlib scipy
  python analisis_preliminar.py --dataset dataset --out salidas
  (opcional)  --factor 4   factor de submuestreo para las pilas por píxel (4 ≈ 110 m; 2 ≈ 55 m, usa más RAM)

Supuestos (coinciden con download_timeseries.py):
  * 12 bandas INT16 en orden B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12, unidades DN (reflectancia = DN/10000,
    armonizadas por Sentinel Hub, sin offset de la línea base 04.00).
  * Valor 0 en todas las bandas = sin dato.
  * Si el TIFF trae 13 bandas, la 13 es SCL y se usa para enmascarar nubes/sombras/nieve;
    con 12 bandas (descarga antigua) se usan reglas espectrales simples.
"""

import argparse
import glob
import json
import os
import re
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.stats import norm

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import rasterio
except ImportError:
    raise SystemExit("Falta rasterio:  pip install rasterio")

# ----------------------------------------------------------------------------- estilo
AZUL, NARANJA, AQUA, AMARILLO = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
TINTA, TINTA2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BUENO, AVISO, SERIO, CRITICO = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
DIVERGENTE = LinearSegmentedColormap.from_list(
    "div", ["#d03b3b", "#ec835a", "#f0efec", "#6da7ec", "#1c5cab"])  # rojo = pérdida, azul = ganancia
SECUENCIAL_VEG = LinearSegmentedColormap.from_list(
    "veg", ["#f3efe4", "#cfe3b5", "#7fbf6a", "#2e8b3a", "#0e4f1c"])

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.family": "sans-serif", "font.size": 9,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": TINTA2, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlecolor": TINTA,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
    "lines.linewidth": 1.6,
})

BANDAS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
LONG_ONDA = [443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1610, 2190]
INDICES = ["NDVI", "EVI", "NDRE", "CIre", "NDMI"]
CLASES = {  # clase de cobertura según NDVI mediano de toda la serie (por píxel)
    "Urbano / suelo": (-1.0, 0.25),
    "Vegetación dispersa": (0.25, 0.50),
    "Vegetación densa / bosque": (0.50, 1.01),
}
COLOR_CLASE = {"Urbano / suelo": NARANJA, "Vegetación dispersa": AMARILLO, "Vegetación densa / bosque": AQUA}
FECHA_LB04 = pd.Timestamp("2022-01-25")  # cambio de línea base de procesamiento 04.00 (offset -1000)


# ----------------------------------------------------------------------------- utilidades
def fecha_de_archivo(path):
    m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(path))
    return pd.Timestamp(m.group(1)) if m else None


def leer_escena(path):
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        perfil = dict(width=src.width, height=src.height, count=src.count, bounds=tuple(src.bounds))
    scl = None
    if arr.shape[0] == 13:  # banda 13 = SCL (clasificación de escena), versión nueva de download_timeseries.py
        scl, arr = arr[12].astype(np.int16), arr[:12]
    if arr.shape[0] != 12:
        raise ValueError(f"{path}: se esperaban 12 o 13 bandas, hay {arr.shape[0]}")
    sin_dato = np.all(arr == 0, axis=0)
    refl = arr / 10000.0
    r = {b: refl[i] for i, b in enumerate(BANDAS)}
    if scl is not None:
        r["SCL"] = scl
    return r, sin_dato, perfil


def dividir(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = a / b
    r[~np.isfinite(r)] = np.nan
    return r


def calcular_indices(r):
    return {
        "NDVI": dividir(r["B08"] - r["B04"], r["B08"] + r["B04"]),
        "EVI": 2.5 * dividir(r["B08"] - r["B04"], r["B08"] + 6 * r["B04"] - 7.5 * r["B02"] + 1),
        "NDRE": dividir(r["B8A"] - r["B05"], r["B8A"] + r["B05"]),
        "CIre": dividir(r["B07"], r["B05"]) - 1,
        "NDMI": dividir(r["B08"] - r["B11"], r["B08"] + r["B11"]),
    }


def mascara_calidad(r, sin_dato):
    """Con SCL: nubes (8, 9, 10), sombras (3), nieve (11), sin dato/defectuoso (0, 1), zonas oscuras (2).
    Sin SCL: reglas espectrales simples (nube brillante, sombra/oscuro, nieve del Nevado, saturación)."""
    if "SCL" in r:
        scl = r["SCL"]
        nube = np.isin(scl, [8, 9, 10])
        nieve = scl == 11
        oscuro = np.isin(scl, [2, 3])
        malo = np.isin(scl, [0, 1])
        valido = ~(sin_dato | nube | nieve | oscuro | malo)
        return valido, nube & ~sin_dato, nieve & ~sin_dato, oscuro & ~sin_dato
    ndsi = dividir(r["B03"] - r["B11"], r["B03"] + r["B11"])
    nube = (r["B02"] > 0.18) & (ndsi < 0.4) | (r["B02"] > 0.12) & (r["B09"] > 0.25) & (r["B11"] > 0.25)
    nieve = (ndsi > 0.42) & (r["B03"] > 0.20) & (r["B08"] > 0.11)
    oscuro = (r["B08"] < 0.05) & (r["B11"] < 0.05)  # sombras de nube/terreno y agua
    fuera_rango = (r["B04"] > 1.0) | (r["B08"] > 1.0)
    valido = ~(sin_dato | nube | nieve | oscuro | fuera_rango)
    return valido, nube & ~sin_dato, nieve & ~sin_dato, oscuro & ~sin_dato


def submuestrear(a, f):
    """Media por bloques f×f ignorando NaN (bloque vale NaN si <50 % de píxeles válidos)."""
    if f == 1:
        return a
    h, w = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    b = a[:h, :w].reshape(h // f, f, w // f, f)
    n = np.sum(np.isfinite(b), axis=(1, 3))
    m = np.nanmean(b, axis=(1, 3))
    m[n < 0.5 * f * f] = np.nan
    return m


def mann_kendall_sen(t, y):
    """Mann-Kendall (sin corrección por empates) + pendiente de Sen. y puede tener NaN en eje 0."""
    y = np.asarray(y, dtype=np.float32)
    t = np.asarray(t, dtype=np.float64)
    n_t = len(t)
    i, j = np.triu_indices(n_t, k=1)
    dy = y[j] - y[i]
    dt = (t[j] - t[i]).reshape((-1,) + (1,) * (y.ndim - 1))
    s = np.nansum(np.sign(dy), axis=0)
    n = np.sum(np.isfinite(y), axis=0).astype(np.float64)
    var = n * (n - 1) * (2 * n + 5) / 18.0
    z = np.where(s > 0, (s - 1) / np.sqrt(var), np.where(s < 0, (s + 1) / np.sqrt(var), 0.0))
    p = 2 * (1 - norm.cdf(np.abs(z)))
    pendiente = np.nanmedian(dy / dt, axis=0)
    p = np.where(n >= 4, p, np.nan)
    pendiente = np.where(n >= 4, pendiente, np.nan)
    return pendiente, p, n


def guardar(fig, out, nombre):
    fig.savefig(os.path.join(out, nombre))
    plt.close(fig)
    print(f"   figura -> {nombre}")


# ----------------------------------------------------------------------------- análisis
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--out", default="salidas")
    ap.add_argument("--factor", type=int, default=4)
    ap.add_argument("--min-validos", type=float, default=0.30,
                    help="fracción mínima de píxeles válidos para usar la escena en series/pilas")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    archivos = sorted(glob.glob(os.path.join(args.dataset, "*.tif*")), key=lambda p: fecha_de_archivo(p) or 0)
    archivos = [a for a in archivos if fecha_de_archivo(a) is not None]
    if not archivos:
        raise SystemExit(f"No hay TIFF con fecha en {args.dataset}/")
    print(f"[1/6] {len(archivos)} escenas encontradas. Leyendo...")

    filas, pilas, fechas_pila, rgb_candidatas = [], {k: [] for k in INDICES}, [], []
    firmas = []
    perfil0 = None
    muestra_corr = None
    for k, path in enumerate(archivos):
        fecha = fecha_de_archivo(path)
        try:
            r, sin_dato, perfil = leer_escena(path)
        except Exception as e:
            print(f"   [omitida] {os.path.basename(path)}: {e}")
            continue
        perfil0 = perfil0 or perfil
        valido, nube, nieve, oscuro = mascara_calidad(r, sin_dato)
        idx = calcular_indices(r)
        total = sin_dato.size
        fila = {
            "fecha": fecha.date().isoformat(), "archivo": os.path.basename(path),
            "ancho": perfil["width"], "alto": perfil["height"],
            "frac_sin_dato": sin_dato.sum() / total, "frac_nube": nube.sum() / total,
            "frac_nieve": nieve.sum() / total, "frac_oscuro": oscuro.sum() / total,
            "frac_validos": valido.sum() / total,
        }
        for nombre, v in idx.items():
            vv = v[valido]
            vv = vv[np.isfinite(vv)]
            if vv.size:
                fila.update({f"{nombre}_media": float(vv.mean()), f"{nombre}_mediana": float(np.median(vv)),
                             f"{nombre}_p10": float(np.percentile(vv, 10)), f"{nombre}_p90": float(np.percentile(vv, 90))})
        nd = idx["NDVI"][valido]
        if nd.size:
            fila["frac_veg_densa"] = float(np.mean(nd > 0.5))
            fila["frac_veg"] = float(np.mean(nd > 0.25))
        filas.append(fila)

        if fila["frac_validos"] >= args.min_validos:
            for nombre in INDICES:
                a = np.where(valido, idx[nombre], np.nan)
                pilas[nombre].append(submuestrear(a, args.factor).astype(np.float16))
            fechas_pila.append(fecha)
            rgb_candidatas.append((fila["frac_validos"], fecha, path))
            # firma espectral por clase instantánea (muestra) y muestra para correlación
            if len(firmas) < 25:
                for clase, (lo, hi) in CLASES.items():
                    m = valido & (idx["NDVI"] >= lo) & (idx["NDVI"] < hi)
                    if m.sum() > 100:
                        firmas.append({"clase": clase, **{b: float(np.median(r[b][m])) for b in BANDAS}})
            if muestra_corr is None and fila["frac_validos"] > 0.8:
                rng = np.random.default_rng(0)
                yy, xx = np.nonzero(valido)
                sel = rng.choice(len(yy), size=min(20000, len(yy)), replace=False)
                muestra_corr = pd.DataFrame({**{b: r[b][yy[sel], xx[sel]] for b in BANDAS},
                                             **{n: idx[n][yy[sel], xx[sel]] for n in INDICES}})
        if (k + 1) % 20 == 0:
            print(f"   {k + 1}/{len(archivos)}")

    inv = pd.DataFrame(filas)
    inv["fecha"] = pd.to_datetime(inv["fecha"])
    inv["anio"], inv["mes"] = inv.fecha.dt.year, inv.fecha.dt.month
    inv.to_csv(os.path.join(args.out, "inventario_escenas.csv"), index=False, float_format="%.4f")
    print(f"[2/6] Inventario guardado ({len(inv)} escenas; {len(fechas_pila)} pasan el umbral de calidad).")
    if len(fechas_pila) < 6:
        raise SystemExit("Muy pocas escenas utilizables para el análisis temporal.")

    P = {k: np.stack(v).astype(np.float32) for k, v in pilas.items()}  # (t, y, x)
    fp = pd.DatetimeIndex(fechas_pila)
    t_dec = (fp.year + (fp.dayofyear - 1) / 365.25).values
    meses = fp.month.values
    anios = fp.year.values

    # --- clases de cobertura (NDVI mediano de toda la serie, por píxel)
    ndvi_med = np.nanmedian(P["NDVI"], axis=0)
    clase_px = np.full(ndvi_med.shape, "", dtype=object)
    for c, (lo, hi) in CLASES.items():
        clase_px[(ndvi_med >= lo) & (ndvi_med < hi)] = c
    frac_clase = {c: float(np.mean(clase_px[np.isfinite(ndvi_med)] == c)) for c in CLASES}

    # ============================================================ FIG 1 inventario
    print("[3/6] Figuras de exploración...")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.1, 1.4]})
    tabla = inv.pivot_table(index="anio", columns="mes", values="fecha", aggfunc="count").reindex(columns=range(1, 13)).fillna(0)
    im = ax[0].imshow(tabla.values, cmap=LinearSegmentedColormap.from_list("s", ["#fcfcfb", "#86b6ef", "#1c5cab"]), aspect="auto")
    ax[0].set_xticks(range(12), list("EFMAMJJASOND"))
    ax[0].set_yticks(range(len(tabla.index)), tabla.index)
    ax[0].grid(False)
    for (i, j), v in np.ndenumerate(tabla.values):
        if v:
            ax[0].text(j, i, int(v), ha="center", va="center", fontsize=7, color=TINTA if v < tabla.values.max() * 0.6 else "white")
    ax[0].set_title("Escenas por año y mes")
    ax[1].scatter(inv.fecha, inv.frac_validos * 100, s=12, color=AZUL, edgecolor="white", linewidth=0.5, zorder=3)
    ax[1].axhline(args.min_validos * 100, color=CRITICO, lw=1, ls="--")
    ax[1].text(inv.fecha.min(), args.min_validos * 100 + 2, f"umbral {args.min_validos:.0%}", color=CRITICO, fontsize=7)
    ax[1].set_ylabel("% píxeles válidos (sin nube/sombra/nieve)")
    ax[1].set_title("Calidad por escena")
    ax[1].set_ylim(0, 100)
    guardar(fig, args.out, "fig01_inventario.png")

    # ============================================================ FIG 2 firmas
    if firmas:
        fdf = pd.DataFrame(firmas).groupby("clase")[BANDAS].median()
        fig, ax = plt.subplots(figsize=(7, 3.6))
        for c in CLASES:
            if c in fdf.index:
                ax.plot(LONG_ONDA, fdf.loc[c].values, marker="o", ms=4, color=COLOR_CLASE[c], label=c)
        ax.axvspan(700, 790, color=AQUA, alpha=0.08)
        ax.text(705, ax.get_ylim()[1] * 0.95, "borde rojo\n(B05–B07)", fontsize=7, color=TINTA2, va="top")
        ax.set_xlabel("Longitud de onda central (nm)")
        ax.set_ylabel("Reflectancia (BOA)")
        ax.set_title("Firma espectral mediana por clase de cobertura")
        ax.legend()
        guardar(fig, args.out, "fig02_firmas_espectrales.png")

    # ============================================================ FIG 3 correlación
    if muestra_corr is not None:
        c = muestra_corr.corr(method="spearman")
        fig, ax = plt.subplots(figsize=(7.5, 6.3))
        im = ax.imshow(c.values, cmap=DIVERGENTE.reversed(), vmin=-1, vmax=1)
        ax.set_xticks(range(len(c)), c.columns, rotation=90)
        ax.set_yticks(range(len(c)), c.columns)
        ax.grid(False)
        for (i, j), v in np.ndenumerate(c.values):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5, color=TINTA)
        fig.colorbar(im, ax=ax, shrink=0.75, label="ρ de Spearman")
        ax.set_title("Correlación entre bandas e índices (20 000 píxeles)")
        guardar(fig, args.out, "fig03_correlacion.png")

    # ============================================================ series por clase
    serie = pd.DataFrame({"fecha": fp})
    for nombre in ["NDVI", "NDRE", "CIre", "NDMI"]:
        for c in CLASES:
            m = clase_px == c
            serie[f"{nombre}|{c}"] = np.nanmedian(P[nombre][:, m], axis=1) if m.any() else np.nan
    serie.to_csv(os.path.join(args.out, "series_por_clase.csv"), index=False, float_format="%.4f")

    fig, axs = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for ax, nombre in zip(axs, ["NDVI", "NDRE", "CIre", "NDMI"]):
        for c in CLASES:
            s = serie.set_index("fecha")[f"{nombre}|{c}"]
            ax.plot(s.index, s.values, ".", ms=3, color=COLOR_CLASE[c], alpha=0.5)
            ax.plot(s.rolling("90D", min_periods=2).median(), color=COLOR_CLASE[c], lw=1.6, label=c)
        if fp.min() < FECHA_LB04 < fp.max():
            ax.axvline(FECHA_LB04, color=MUTED, lw=0.8, ls=":")
        ax.set_ylabel(nombre)
    fig.legend(*axs[0].get_legend_handles_labels(), ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    axs[0].set_title("Serie temporal (mediana espacial por clase; línea = mediana móvil de 90 días)")
    if fp.min() < FECHA_LB04 < fp.max():
        axs[-1].text(FECHA_LB04, axs[-1].get_ylim()[0], " línea base 04.00", fontsize=7, color=MUTED, va="bottom")
    guardar(fig, args.out, "fig04_serie_temporal.png")

    # ============================================================ FIG 5 estacionalidad
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, nombre in zip(axs, ["NDVI", "NDRE"]):
        for k, c in enumerate(CLASES):
            s = serie[f"{nombre}|{c}"]
            datos = [s[meses == mm].dropna().values for mm in range(1, 13)]
            pos = np.arange(1, 13) + (k - 1) * 0.26
            bp = ax.boxplot([d if len(d) else [np.nan] for d in datos], positions=pos, widths=0.22,
                            patch_artist=True, showfliers=False, medianprops=dict(color=TINTA, lw=1))
            for b in bp["boxes"]:
                b.set(facecolor=COLOR_CLASE[c], edgecolor=COLOR_CLASE[c], alpha=0.8)
            for w in bp["whiskers"] + bp["caps"]:
                w.set(color=COLOR_CLASE[c])
        ax.axvspan(5.5, 10.5, color=AZUL, alpha=0.06)
        ax.text(5.6, ax.get_ylim()[1], "temporada de lluvias", fontsize=7, color=AZUL, va="top")
        ax.set_xticks(range(1, 13), list("EFMAMJJASOND"))
        ax.set_title(f"Climatología mensual de {nombre}")
    axs[0].legend([plt.Rectangle((0, 0), 1, 1, color=COLOR_CLASE[c]) for c in CLASES], CLASES.keys(), loc="lower left", fontsize=7)
    guardar(fig, args.out, "fig05_estacionalidad.png")

    # ============================================================ FIG 6 anomalías + tendencia (escena)
    print("[4/6] Tendencias...")
    fig, axs = plt.subplots(len(CLASES), 1, figsize=(11, 7), sharex=True)
    tendencias = {}
    for ax, c in zip(axs, CLASES):
        for nombre, col in [("NDVI", AZUL), ("NDRE", NARANJA)]:
            s = serie[f"{nombre}|{c}"].values
            clim = np.array([np.nanmedian(s[meses == mm]) if np.any(meses == mm) else np.nan for mm in range(1, 13)])
            anom = s - clim[meses - 1]
            ok = np.isfinite(anom)
            if ok.sum() < 6:
                continue
            pend, p, _ = mann_kendall_sen(t_dec[ok], anom[ok])
            tendencias[f"{nombre}|{c}"] = {"sen_por_anio": float(pend), "p_valor": float(p), "n": int(ok.sum())}
            ax.plot(fp[ok], anom[ok], ".", ms=3, color=col, alpha=0.45)
            x0 = t_dec[ok].mean()
            yfit = np.nanmedian(anom[ok] - pend * (t_dec[ok] - x0)) + pend * (t_dec[ok] - x0)
            ax.plot(fp[ok], yfit, color=col, lw=2,
                    label=f"{nombre}: {pend*1000:+.2f}×10⁻³/año (p={p:.3f})")
        ax.axhline(0, color="#c3c2b7", lw=0.8)
        ax.set_ylabel("anomalía")
        ax.set_title(c, loc="left", fontsize=9)
        ax.legend(loc="lower left", fontsize=7)
    fig.suptitle("Anomalías desestacionalizadas y tendencia de Sen (Mann-Kendall)", fontweight="bold", x=0.02, ha="left")
    guardar(fig, args.out, "fig06_anomalias.png")

    # ============================================================ compuestos anuales temporada seca
    seca = np.isin(meses, [11, 12, 1, 2, 3, 4])
    anio_seco = np.where(meses >= 11, anios + 1, anios)  # nov-dic se asignan a la temporada seca del año siguiente
    anios_u = sorted(set(anio_seco[seca]))
    comp = {n: [] for n in ["NDVI", "CIre", "NDRE"]}
    anios_ok = []
    for a in anios_u:
        sel = seca & (anio_seco == a)
        if sel.sum() >= 2:
            anios_ok.append(a)
            for n in comp:
                comp[n].append(np.nanmedian(P[n][sel], axis=0))
    comp = {n: np.stack(v) for n, v in comp.items()} if anios_ok else {}
    extent = None
    if perfil0:
        b = perfil0["bounds"]
        extent = [b[0], b[2], b[1], b[3]]

    resumen_pix = {}
    if len(anios_ok) >= 2:
        # FIG 7 mapas primer/último año
        a0, a1 = anios_ok[0], anios_ok[-1]
        d = comp["NDVI"][-1] - comp["NDVI"][0]
        fig, axs = plt.subplots(1, 3, figsize=(12, 6.2))
        for ax, img, tit in [(axs[0], comp["NDVI"][0], f"NDVI temporada seca {a0}"),
                             (axs[1], comp["NDVI"][-1], f"NDVI temporada seca {a1}")]:
            im = ax.imshow(img, cmap=SECUENCIAL_VEG, vmin=0, vmax=0.85, extent=extent)
            ax.set_title(tit); ax.grid(False)
        fig.colorbar(im, ax=axs[:2], shrink=0.6, orientation="horizontal", label="NDVI", pad=0.06)
        im = axs[2].imshow(d, cmap=DIVERGENTE, norm=TwoSlopeNorm(0, -0.3, 0.3), extent=extent)
        axs[2].set_title(f"Diferencia {a1} − {a0}"); axs[2].grid(False)
        fig.colorbar(im, ax=axs[2], shrink=0.6, orientation="horizontal", label="ΔNDVI (rojo = pérdida)", pad=0.06)
        guardar(fig, args.out, "fig07_mapas_ndvi.png")

        # FIG 8 tendencia por píxel
        pend, p, n = mann_kendall_sen(np.array(anios_ok, float), comp["NDVI"])
        veg = np.isin(clase_px, ["Vegetación dispersa", "Vegetación densa / bosque"]) & np.isfinite(pend)
        sig = (p < 0.05) & veg
        resumen_pix = {
            "anios_temporada_seca": [int(a) for a in anios_ok],
            "pix_vegetados": int(veg.sum()),
            "pct_veg_declive_significativo": float(100 * np.mean((pend[veg] < 0) & (p[veg] < 0.05))) if veg.any() else None,
            "pct_veg_aumento_significativo": float(100 * np.mean((pend[veg] > 0) & (p[veg] < 0.05))) if veg.any() else None,
            "sen_mediana_veg_ndvi_por_anio": float(np.nanmedian(pend[veg])) if veg.any() else None,
            "tamano_pixel_aprox_m": None,
        }
        fig, axs = plt.subplots(1, 2, figsize=(11, 6), gridspec_kw={"width_ratios": [1.2, 1]})
        vis = np.where(veg, pend, np.nan)
        im = axs[0].imshow(vis, cmap=DIVERGENTE, norm=TwoSlopeNorm(0, -0.03, 0.03), extent=extent)
        yy, xx = np.nonzero(sig)
        if extent and len(yy):
            h, w = pend.shape
            axs[0].scatter(extent[0] + (xx + 0.5) / w * (extent[1] - extent[0]),
                           extent[3] - (yy + 0.5) / h * (extent[3] - extent[2]), s=0.15, c=TINTA, alpha=0.35, lw=0)
        axs[0].set_title("Pendiente de Sen del NDVI (por año), sólo vegetación\npuntos = p < 0,05 (Mann-Kendall)")
        axs[0].grid(False)
        fig.colorbar(im, ax=axs[0], shrink=0.7, label="ΔNDVI / año")
        v = pend[veg]
        axs[1].hist(v[np.isfinite(v)], bins=60, color=AZUL, edgecolor="white", linewidth=0.3)
        axs[1].axvline(0, color=TINTA, lw=0.8)
        axs[1].set_title("Distribución de pendientes (píxeles vegetados)")
        axs[1].set_xlabel("ΔNDVI / año")
        txt = (f"declive significativo: {resumen_pix['pct_veg_declive_significativo']:.1f} %\n"
               f"aumento significativo: {resumen_pix['pct_veg_aumento_significativo']:.1f} %")
        axs[1].text(0.02, 0.97, txt, transform=axs[1].transAxes, va="top", fontsize=8, color=TINTA2)
        guardar(fig, args.out, "fig08_mapa_tendencia.png")

        # FIG 9 CIre vs NDVI (Zarco-Tejada 2018): razones t1/t2 (t1 = primer año, t2 = último)
        r_ci = comp["CIre"][0] / comp["CIre"][-1]
        r_nd = comp["NDVI"][0] / comp["NDVI"][-1]
        ok = veg & np.isfinite(r_ci) & np.isfinite(r_nd) & (comp["CIre"][-1] > 0.05) & (comp["NDVI"][-1] > 0.1)
        x, y = r_nd[ok], r_ci[ok]
        clorosis = (y > 1.15) & (x < 1.05)
        defol = (y > 1.15) & (x >= 1.05)
        resumen_pix.update({
            "pct_veg_perdida_clorofila_sin_perdida_estructura": float(100 * clorosis.mean()) if ok.any() else None,
            "pct_veg_perdida_clorofila_y_estructura": float(100 * defol.mean()) if ok.any() else None,
        })
        fig, ax = plt.subplots(figsize=(6.2, 5.6))
        hb = ax.hexbin(x, y, gridsize=70, extent=(0.6, 1.6, 0.4, 2.2), mincnt=1, bins="log",
                       cmap=LinearSegmentedColormap.from_list("h", ["#cde2fb", "#3987e5", "#0d366b"]))
        ax.plot([0.6, 1.6], [0.6, 1.6], color=TINTA, lw=1, ls="--")
        ax.axhline(1.15, color=CRITICO, lw=0.8, ls=":")
        ax.axvline(1.05, color=MUTED, lw=0.8, ls=":")
        ax.text(0.62, 2.1, f"pérdida de clorofila sin\npérdida estructural: {100*clorosis.mean():.1f} %", fontsize=7.5, color=CRITICO, va="top")
        ax.text(1.07, 2.1, f"clorofila + estructura\n(defoliación): {100*defol.mean():.1f} %", fontsize=7.5, color=TINTA2, va="top")
        ax.set_xlabel(f"NDVI({anios_ok[0]}) / NDVI({anios_ok[-1]})")
        ax.set_ylabel(f"CIre({anios_ok[0]}) / CIre({anios_ok[-1]})")
        ax.set_title("Trayectoria clorofila vs. estructura (temporada seca)")
        fig.colorbar(hb, ax=ax, label="píxeles (log)")
        guardar(fig, args.out, "fig09_ci_vs_ndvi.png")

    # ============================================================ FIG 10 puntuación de salud
    print("[5/6] Puntuación preliminar de salud...")
    # z = anomalía respecto a la climatología del propio píxel (ventana de ±1 mes), promediada en NDVI, NDRE y NDMI.
    z_total = np.zeros_like(P["NDVI"])
    cuenta = np.zeros_like(P["NDVI"])
    for n in ["NDVI", "NDRE", "NDMI"]:
        X = P[n]
        Z = np.full_like(X, np.nan)
        for mm in range(1, 13):
            vent = np.isin(meses, [(mm - 2) % 12 + 1, mm, mm % 12 + 1])
            if vent.sum() < 3:
                continue
            med = np.nanmedian(X[vent], axis=0)
            mad = 1.4826 * np.nanmedian(np.abs(X[vent] - med), axis=0)
            mad = np.where(mad < 0.01, 0.01, mad)
            tgt = meses == mm
            Z[tgt] = (X[tgt] - med) / mad
        ok = np.isfinite(Z)
        z_total[ok] += np.clip(Z[ok], -5, 5)
        cuenta[ok] += 1
    z = np.where(cuenta > 0, z_total / np.maximum(cuenta, 1), np.nan)
    puntuacion = 100 * norm.cdf(z)  # 0-100, 50 = condición típica para ese píxel y época
    vegm = np.isin(clase_px, ["Vegetación dispersa", "Vegetación densa / bosque"])
    cats = [("Saludable", -0.5, np.inf, BUENO), ("Estrés leve", -1.0, -0.5, AVISO),
            ("Estrés moderado", -2.0, -1.0, SERIO), ("Estrés severo", -np.inf, -2.0, CRITICO)]
    reg = []
    for ti, f in enumerate(fp):
        zz = z[ti][vegm]
        zz = zz[np.isfinite(zz)]
        if zz.size < 50:
            continue
        reg.append({"fecha": f, "anio": f.year, "puntuacion_media": float(np.mean(100 * norm.cdf(zz))),
                    **{c: float(np.mean((zz >= lo) & (zz < hi))) for c, lo, hi, _ in cats}})
    salud = pd.DataFrame(reg)
    salud.to_csv(os.path.join(args.out, "puntuacion_salud_por_escena.csv"), index=False, float_format="%.4f")
    anual = salud.groupby("anio")[[c for c, *_ in cats] + ["puntuacion_media"]].mean()
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.3, 1]})
    base = np.zeros(len(anual))
    for c, lo, hi, col in cats:
        axs[0].bar(anual.index, anual[c] * 100, bottom=base, color=col, label=c, edgecolor="white", linewidth=1)
        base += anual[c].values * 100
    axs[0].set_ylabel("% de píxeles vegetados")
    axs[0].set_title("Clases de salud por año (anomalía estandarizada)")
    axs[0].legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    axs[0].set_ylim(0, 100)
    ultimo = int(np.nanargmax([np.isfinite(z[i][vegm]).mean() if vegm.any() else 0 for i in range(len(fp))][-10:])) + max(0, len(fp) - 10)
    im = axs[1].imshow(np.where(vegm, puntuacion[ultimo], np.nan), cmap=DIVERGENTE, vmin=0, vmax=100, extent=extent)
    axs[1].set_title(f"Puntuación de salud 0–100, {fp[ultimo].date()}")
    axs[1].grid(False)
    fig.colorbar(im, ax=axs[1], shrink=0.8, label="puntuación (50 = condición típica)")
    guardar(fig, args.out, "fig10_puntuacion_salud.png")

    # ============================================================ FIG 11 RGB
    def rgb(path):
        r, sd, _ = leer_escena(path)
        img = np.dstack([r["B04"], r["B03"], r["B02"]])
        return np.clip(img / 0.3, 0, 1) ** (1 / 1.2)
    if rgb_candidatas:
        cand = sorted(rgb_candidatas, key=lambda x: x[1])
        primero = max(cand[: max(1, len(cand) // 10)], key=lambda x: x[0])
        ultimo_c = max(cand[-max(1, len(cand) // 10):], key=lambda x: x[0])
        fig, axs = plt.subplots(1, 2, figsize=(9, 8))
        for ax, (_, f, pth) in zip(axs, [primero, ultimo_c]):
            ax.imshow(rgb(pth), extent=extent)
            ax.set_title(f"Color verdadero {f.date()}")
            ax.grid(False)
        guardar(fig, args.out, "fig11_rgb.png")

    # ============================================================ resumen
    print("[6/6] Resumen...")
    ancho_m = alto_m = None
    if perfil0:
        b = perfil0["bounds"]
        lat = np.deg2rad((b[1] + b[3]) / 2)
        ancho_m = (b[2] - b[0]) * 111320 * np.cos(lat) / perfil0["width"]
        alto_m = (b[3] - b[1]) * 110574 / perfil0["height"]
    if resumen_pix:
        resumen_pix["tamano_pixel_aprox_m"] = round(float(ancho_m * args.factor), 1) if ancho_m else None
    n_train = int((fp.year <= 2022).sum()); n_val = int((fp.year == 2023).sum()); n_test = int((fp.year >= 2024).sum())
    resumen = {
        "escenas_total": int(len(inv)),
        "escenas_utilizables": int(len(fp)),
        "rango_fechas": [str(inv.fecha.min().date()), str(inv.fecha.max().date())],
        "escenas_por_anio": {int(k): int(v) for k, v in inv.groupby("anio").size().items()},
        "escenas_temporada_lluvias_jun_oct_pct": float(100 * inv.mes.between(6, 10).mean()),
        "frac_validos_mediana": float(inv.frac_validos.median()),
        "frac_nube_mediana": float(inv.frac_nube.median()),
        "frac_nieve_max": float(inv.frac_nieve.max()),
        "dimensiones_px": [int(perfil0["width"]), int(perfil0["height"])] if perfil0 else None,
        "resolucion_efectiva_m": [round(float(ancho_m), 1), round(float(alto_m), 1)] if ancho_m else None,
        "fraccion_area_por_clase": frac_clase,
        "tendencia_anomalias_por_clase": tendencias,
        "analisis_por_pixel": resumen_pix,
        "salud_media_por_anio": {int(k): round(float(v), 1) for k, v in anual["puntuacion_media"].items()},
        "salud_pct_estres_mod_sev_por_anio": {int(k): round(float(100 * (anual.loc[k, "Estrés moderado"] + anual.loc[k, "Estrés severo"])), 1) for k in anual.index},
        "particion_temporal_sugerida": {"entrenamiento_<=2022": n_train, "validacion_2023": n_val, "prueba_>=2024": n_test},
        "estadisticos_indices_globales": {n: {"mediana": float(inv[f"{n}_mediana"].median()), "p10": float(inv[f"{n}_p10"].median()),
                                              "p90": float(inv[f"{n}_p90"].median())} for n in INDICES if f"{n}_mediana" in inv},
    }
    with open(os.path.join(args.out, "resumen.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out, "resumen.txt"), "w", encoding="utf-8") as f:
        for k, v in resumen.items():
            f.write(f"{k}: {v}\n")
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(f"\nListo. Comprime la carpeta '{args.out}' y compártela.")


if __name__ == "__main__":
    main()
