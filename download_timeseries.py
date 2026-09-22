"""
download_timeseries.py
======================
Descarga la serie Sentinel-2 L2A de Toluca desde Copernicus Data Space (Process API).

Por defecto: 2015-01-01 a 2026-09-01 con cobertura de nubes <= 5 % (acordado con la asesora el 15/sep/2026:
~200 imágenes en total, ~10-40 por año según el año).

Cambios respecto a la versión anterior:
  * Nubes <= 5 % (antes 15-20 %) — así lo pidió la asesora; da ~200 imágenes en vez de 271-382.
  * Búsqueda STAC año por año Y con paginación: antes "limit: 100" cortaba los resultados.
  * Se agrega la banda SCL (clasificación de escena) como banda 13 para enmascarar nubes, sombras y nieve.
  * El token OAuth se renueva solo (caduca a los ~10 min y con cientos de descargas expiraba).
  * No vuelve a descargar archivos que ya existen; reintenta si hay error temporal.
  * Guarda catalogo_stac.csv con cada imagen encontrada (fecha, id, % nubes) e imagenes_por_anio.csv
    con el conteo por año, para la tarea de "documentar cuántas imágenes hay por año y cuáles son".

Uso:
  python download_timeseries.py                       # configuración por defecto (5 %, ~200 imágenes)
  python download_timeseries.py --nubes 20            # el filtro más ancho que se probó antes (~382)
  python download_timeseries.py --solo-catalogo       # solo cuenta imágenes, no descarga
"""
import argparse
import csv
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()
USERNAME = os.getenv("COPERNICUS_USERNAME")
PASSWORD = os.getenv("COPERNICUS_PASSWORD")

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
STAC_URL = "https://catalogue.dataspace.copernicus.eu/stac/search"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

BBOX_TOLUCA = [-99.7939, 18.9839, -99.5286, 19.4525]

EVALSCRIPT = """
//VERSION=3
function setup() {
    return {
        input: [{
            bands: ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12", "SCL"],
            units: "DN"
        }],
        output: { bands: 13, sampleType: "INT16" }
    };
}
function evaluatePixel(s) {
    return [s.B01, s.B02, s.B03, s.B04, s.B05, s.B06, s.B07, s.B08, s.B8A, s.B09, s.B11, s.B12, s.SCL];
}
"""


# ----------------------------------------------------------------------------- autenticación
class Token:
    """Guarda el token y lo renueva antes de que caduque."""

    def __init__(self):
        self.valor, self.expira = None, 0

    def obtener(self):
        if self.valor is None or time.time() > self.expira - 60:
            if not USERNAME or not PASSWORD:
                raise SystemExit("Faltan COPERNICUS_USERNAME / COPERNICUS_PASSWORD en el archivo .env")
            r = requests.post(TOKEN_URL, data={"client_id": "cdse-public", "username": USERNAME,
                                                "password": PASSWORD, "grant_type": "password"}, timeout=60)
            if r.status_code != 200:
                raise SystemExit(f"Error al autenticar: {r.text}")
            d = r.json()
            self.valor, self.expira = d["access_token"], time.time() + d.get("expires_in", 600)
        return self.valor

    def invalidar(self):
        self.valor = None


# ----------------------------------------------------------------------------- catálogo
def buscar_imagenes(inicio, fin, nubes_max):
    """Consulta STAC año por año siguiendo la paginación. Devuelve lista de dicts ordenada por fecha."""
    encontrados = {}
    a0, a1 = int(inicio[:4]), int(fin[:4])
    for anio in range(a0, a1 + 1):
        ini = max(inicio, f"{anio}-01-01")
        fi = min(fin, f"{anio}-12-31")
        payload = {
            "bbox": BBOX_TOLUCA,
            "datetime": f"{ini}T00:00:00Z/{fi}T23:59:59Z",
            "collections": ["sentinel-2-l2a"],
            "query": {"eo:cloud_cover": {"lte": nubes_max}},
            "limit": 100,
        }
        url, metodo, cuerpo, n_anio = STAC_URL, "POST", payload, 0
        while url:
            r = requests.post(url, json=cuerpo, timeout=120) if metodo == "POST" else requests.get(url, timeout=120)
            r.raise_for_status()
            js = r.json()
            for f in js.get("features", []):
                encontrados[f["id"]] = {"id": f["id"], "fecha": f["properties"]["datetime"][:10],
                                        "nubes": f["properties"].get("eo:cloud_cover")}
                n_anio += 1
            url = None
            for link in js.get("links", []):
                if link.get("rel") == "next":
                    url = link["href"]
                    metodo = link.get("method", "GET").upper()
                    cuerpo = link.get("body") or cuerpo
                    break
        print(f"  {anio}: {n_anio} imágenes")
    return sorted(encontrados.values(), key=lambda x: (x["fecha"], x["id"]))


# ----------------------------------------------------------------------------- descarga
def descargar(token, fecha, carpeta, intentos=3):
    archivo = os.path.join(carpeta, f"toluca_{fecha}.tiff")
    if os.path.exists(archivo) and os.path.getsize(archivo) > 1000:
        return "existe"
    ancho = 1024
    alto = int(ancho * (BBOX_TOLUCA[3] - BBOX_TOLUCA[1]) / (BBOX_TOLUCA[2] - BBOX_TOLUCA[0]))
    payload = {
        "input": {
            "bounds": {"bbox": BBOX_TOLUCA, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
            "data": [{"type": "sentinel-2-l2a",
                      "dataFilter": {"timeRange": {"from": f"{fecha}T00:00:00Z", "to": f"{fecha}T23:59:59Z"}}}],
        },
        "output": {"width": ancho, "height": alto,
                   "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": EVALSCRIPT,
    }
    for k in range(intentos):
        try:
            r = requests.post(PROCESS_URL, json=payload, timeout=300,
                              headers={"Authorization": f"Bearer {token.obtener()}",
                                       "Content-Type": "application/json", "Accept": "image/tiff"})
        except requests.exceptions.RequestException as e:
            # timeout, conexión cortada, DNS, etc.: transitorio, se reintenta igual que un 5xx
            print(f"     [reintento] {fecha}: error de red ({type(e).__name__}), intento {k + 1}/{intentos}")
            time.sleep(10 * (k + 1))
            continue
        if r.status_code == 200:
            with open(archivo, "wb") as f:
                f.write(r.content)
            return "ok"
        if r.status_code == 401:
            token.invalidar()
        elif r.status_code in (429, 500, 502, 503, 504):
            time.sleep(10 * (k + 1))
        else:
            print(f"     [ERROR] {fecha} ({r.status_code}): {r.text[:200]}")
            return "error"
    return "error"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inicio", default="2015-01-01")
    ap.add_argument("--fin", default="2026-09-01")
    ap.add_argument("--nubes", type=float, default=5, help="porcentaje máximo de nubes (<=)")
    ap.add_argument("--carpeta", default="dataset")
    ap.add_argument("--solo-catalogo", action="store_true")
    args = ap.parse_args()

    print(f"Buscando imágenes {args.inicio} -> {args.fin} con nubes <= {args.nubes:g} %...")
    imgs = buscar_imagenes(args.inicio, args.fin, args.nubes)
    fechas = sorted({i["fecha"] for i in imgs})
    print(f"\nTotal: {len(imgs)} imágenes (productos) en {len(fechas)} fechas distintas.")
    print("  (un mismo día puede tener 2 productos si el área cruza dos teselas; se descarga 1 mosaico por día)")

    with open("catalogo_stac.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["fecha", "id", "nubes"])
        w.writeheader()
        w.writerows(imgs)
    print("Catálogo guardado en catalogo_stac.csv (cada imagen encontrada, con fecha)")

    conteo = {}
    for fecha in fechas:
        conteo[fecha[:4]] = conteo.get(fecha[:4], 0) + 1
    with open("imagenes_por_anio.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["anio", "imagenes"])
        for anio in sorted(conteo):
            w.writerow([anio, conteo[anio]])
    print("Conteo por año guardado en imagenes_por_anio.csv:")
    for anio in sorted(conteo):
        print(f"  {anio}: {conteo[anio]} imágenes")

    if args.solo_catalogo:
        return

    os.makedirs(args.carpeta, exist_ok=True)
    token = Token()
    cuenta = {"ok": 0, "existe": 0, "error": 0}
    for n, fecha in enumerate(fechas, 1):
        try:
            estado = descargar(token, fecha, args.carpeta)
        except Exception as e:
            # cualquier fallo inesperado (red, disco, etc.) no debe tumbar el resto de la descarga
            print(f"  [{n}/{len(fechas)}] {fecha}: error inesperado ({type(e).__name__}: {e})")
            estado = "error"
        cuenta[estado] += 1
        print(f"  [{n}/{len(fechas)}] {fecha}: {estado}")
    print(f"\nListo: {cuenta['ok']} descargadas, {cuenta['existe']} ya existían, {cuenta['error']} con error.")
    if cuenta["error"]:
        print("Vuelve a ejecutar el script: solo intentará las que faltan.")


if __name__ == "__main__":
    main()
