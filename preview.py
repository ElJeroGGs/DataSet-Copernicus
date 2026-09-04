import rasterio
import matplotlib.pyplot as plt
import numpy as np

def create_preview(tiff_path, output_png):
    try:
        with rasterio.open(tiff_path) as src:
            print(f"Abriendo {tiff_path}")
            print(f"Formato: {src.meta['dtype']}, Bandas: {src.count}, Dimensiones: {src.width}x{src.height}")
            
            # En nuestro script, las bandas RGB son la 4, 3 y 2 respectivamente.
            # rasterio usa índices basados en 1, así que leemos las bandas 4, 3 y 2.
            r = src.read(4)
            g = src.read(3)
            b = src.read(2)
            
            # Normalizamos los valores (Sentinel-2 L2A típicamente tiene valores de reflectancia donde 10000 = 1.0)
            # Acotamos a un máximo para que la imagen no se vea muy oscura por culpa de las nubes muy brillantes
            max_val = 3000.0 
            
            r_norm = np.clip(r / max_val, 0, 1)
            g_norm = np.clip(g / max_val, 0, 1)
            b_norm = np.clip(b / max_val, 0, 1)
            
            # Apilamos en una imagen RGB (alto, ancho, canales)
            rgb = np.dstack((r_norm, g_norm, b_norm))
            
            # Guardamos la imagen
            plt.imsave(output_png, rgb)
            print(f"¡Vista previa guardada como {output_png}!")
            
    except Exception as e:
        print(f"Error al procesar el archivo: {e}")

import glob
import sys

if __name__ == "__main__":
    tiffs = glob.glob("dataset/*.tiff")
    if not tiffs:
        print("No se encontraron archivos TIFF en la carpeta dataset/")
        sys.exit(1)
        
    primer_tiff = tiffs[0]
    create_preview(primer_tiff, "preview.png")
