import os
import requests
from dotenv import load_dotenv

load_dotenv()
USERNAME = os.getenv("COPERNICUS_USERNAME")
PASSWORD = os.getenv("COPERNICUS_PASSWORD")

# Endpoints
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

# BBox de Toluca (Longitud, Latitud)
BBOX_TOLUCA = [-99.7939, 18.9839, -99.5286, 19.4525]

def obtener_token():
    payload = {
        "client_id": "cdse-public", 
        "username": USERNAME,
        "password": PASSWORD,
        "grant_type": "password"
    }
    print("Obteniendo token...")
    response = requests.post(TOKEN_URL, data=payload)
    if response.status_code == 200:
        return response.json().get("access_token")
    else:
        raise Exception(f"Error al autenticar: {response.text}")

def descargar_imagen_procesada(token):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "image/tiff"
    }
    
    # Basado en la documentación de Process API y CRS
    # EPSG:4326 representa el sistema WGS 84 (Longitud / Latitud)
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
                            "from": "2026-08-01T00:00:00Z",
                            "to": "2026-09-04T23:59:59Z"
                        },
                        "maxCloudCoverage": 15
                    }
                }
            ]
        },
        "output": {
            "width": 1024,
            "height": 1024,
            "responses": [
                {
                    "identifier": "default",
                    "format": {
                        "type": "image/tiff"
                    }
                }
            ]
        },
        # Aquí pedimos todas las bandas multiespectrales relevantes en crudo (Digital Numbers)
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

    print("Solicitando imagen a la Process API de Sentinel Hub...")
    response = requests.post(PROCESS_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        filename = "toluca_hiperespectral.tiff"
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"¡Imagen descargada exitosamente como '{filename}'!")
    else:
        print(f"Error ({response.status_code}): {response.text}")

def main():
    try:
        token = obtener_token()
        descargar_imagen_procesada(token)
    except Exception as e:
        print(f"Ocurrió un error: {e}")

if __name__ == "__main__":
    main()
