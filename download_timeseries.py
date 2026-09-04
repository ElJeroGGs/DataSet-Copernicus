import os
import requests
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
USERNAME = os.getenv("COPERNICUS_USERNAME")
PASSWORD = os.getenv("COPERNICUS_PASSWORD")

# Endpoints
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
STAC_URL = "https://catalogue.dataspace.copernicus.eu/stac/search"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

# BBox de Toluca
BBOX_TOLUCA = [-99.7939, 18.9839, -99.5286, 19.4525]

def obtener_token():
    """Obtiene el token OAuth2 para la Process API"""
    payload = {
        "client_id": "cdse-public", 
        "username": USERNAME,
        "password": PASSWORD,
        "grant_type": "password"
    }
    response = requests.post(TOKEN_URL, data=payload)
    if response.status_code == 200:
        return response.json().get("access_token")
    else:
        raise Exception(f"Error al autenticar: {response.text}")

def buscar_fechas_disponibles(fecha_inicio, fecha_fin):
    """Busca en STAC las fechas con buenas condiciones"""
    payload = {
        "bbox": BBOX_TOLUCA,
        "datetime": f"{fecha_inicio}/{fecha_fin}", 
        "collections": ["sentinel-2-l2a"],
        "query": {
            "eo:cloud_cover": {"lt": 15} # Menos de 15% de nubes
        },
        # Podemos traer hasta 100 resultados
        "limit": 100 
    }
    
    print(f"Buscando capturas en STAC entre {fecha_inicio} y {fecha_fin}...")
    response = requests.post(STAC_URL, json=payload)
    response.raise_for_status()
    
    features = response.json().get("features", [])
    fechas = []
    
    # Extraer y deduplicar las fechas (a veces hay múltiples tiles (fragmentos) el mismo día)
    for feature in features:
        # datetime viene en formato "2026-08-15T16:34:11.123Z"
        dt_str = feature["properties"]["datetime"]
        # Extraer solo la parte de YYYY-MM-DD
        fecha_corta = dt_str.split("T")[0]
        if fecha_corta not in fechas:
            fechas.append(fecha_corta)
            
    # Ordenar cronológicamente
    fechas.sort()
    print(f"Se encontraron {len(fechas)} días únicos con capturas óptimas.")
    return fechas

def descargar_tiff_por_fecha(token, fecha):
    """Descarga el TIFF multicapa para un solo día usando Process API"""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "image/tiff"
    }
    
    # Creamos un rango de 1 día entero para asegurarnos de capturar la imagen de esa fecha
    time_from = f"{fecha}T00:00:00Z"
    time_to = f"{fecha}T23:59:59Z"
    
    # Calculamos la proporción (aspect ratio) para no aplastar la imagen
    lon_width = BBOX_TOLUCA[2] - BBOX_TOLUCA[0]
    lat_height = BBOX_TOLUCA[3] - BBOX_TOLUCA[1]
    
    img_width = 1024
    img_height = int(img_width * (lat_height / lon_width))
    
    payload = {
        "input": {
            "bounds": {
                "bbox": BBOX_TOLUCA,
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/EPSG/0/4326"
                }
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": time_from,
                            "to": time_to
                        }
                    }
                }
            ]
        },
        "output": {
            "width": img_width,
            "height": img_height,
            "responses": [
                {
                    "identifier": "default",
                    "format": {
                        "type": "image/tiff"
                    }
                }
            ]
        },
        "evalscript": """
            //VERSION=3
            function setup() {
                return {
                    input: [{
                        bands: ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"],
                        units: "DN"
                    }],
                    output: { bands: 12, sampleType: "INT16" }
                };
            }
            function evaluatePixel(sample) {
                return [
                    sample.B01, sample.B02, sample.B03, sample.B04, 
                    sample.B05, sample.B06, sample.B07, sample.B08, 
                    sample.B8A, sample.B09, sample.B11, sample.B12
                ];
            }
        """
    }

    print(f"  -> Solicitando TIFF para la fecha {fecha}...")
    response = requests.post(PROCESS_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        # Creamos una carpeta para organizar las imágenes si no existe
        os.makedirs("dataset", exist_ok=True)
        filename = f"dataset/toluca_{fecha}.tiff"
        
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"     [OK] Guardado como '{filename}'")
    else:
        print(f"     [ERROR] Falló la descarga para {fecha} ({response.status_code}): {response.text}")

def main():
    # Rango de tiempo para entrenar tu modelo (puedes expandirlo a meses enteros o un año)
    FECHA_INICIO = "2026-08-01T00:00:00Z"
    FECHA_FIN = "2026-09-04T23:59:59Z"
    
    try:
        fechas_validas = buscar_fechas_disponibles(FECHA_INICIO, FECHA_FIN)
        
        if not fechas_validas:
            print("No hay fechas para descargar.")
            return
            
        token = obtener_token()
        print("Iniciando descargas...\n")
        
        for fecha in fechas_validas:
            descargar_tiff_por_fecha(token, fecha)
            
        print("\n¡Descarga de la serie de tiempo finalizada!")
        
    except Exception as e:
        print(f"Ocurrió un error general: {e}")

if __name__ == "__main__":
    main()
