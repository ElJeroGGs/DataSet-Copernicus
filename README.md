# Vegetación en la Zona Metropolitana del Valle de Toluca con Sentinel-2

Proyecto de Reconocimiento de Patrones. A partir de 313 escenas Sentinel-2 L2A (2015–2026) se construyen **máscaras
binarias de vegetación** con cinco índices espectrales (NDVI, EVI, NDRE, CIre y NDMI) y se entrena **la misma red**
(U-Net con encoder preentrenado) con cada máscara para comparar cuál se aprende mejor a partir de una imagen RGB,
evaluando en años no vistos (2023–2026).

## Resultado principal

Prueba 2023–2026 (134 escenas), U-Net + ResNet34 (ImageNet), entrada RGB, misma configuración; ordenado por MCC.
Cifras recalculables con `analisis_resultados.py` a partir de `resultados/gpu/`.

| Máscara | Umbral | IoU | F1 | MCC | Prevalencia |
|---|---|---|---|---|---|
| NDRE | Otsu 0.312 | 0.763 | 0.866 | 0.798 | 32.6 % |
| NDVI | fijo 0.5 | 0.745 | 0.854 | 0.794 | 28.3 % |
| CIre | Otsu 1.223 | 0.694 | 0.819 | 0.774 | 20.3 % |
| NDMI | Otsu 0.052 | 0.697 | 0.821 | 0.732 | 32.1 % |
| EVI | Otsu 0.273 | 0.663 | 0.797 | 0.700 | 31.0 % |
| NDVI, línea base por píxel | fijo 0.5 | 0.725 | 0.841 | 0.776 | 28.3 % |

- NDRE y NDVI no difieren significativamente (Wilcoxon pareado por escena, p = 0.12); NDRE supera a CIre
  (p = 0.003), NDMI y EVI (p ≈ 1e-23).
- EfficientNet-B0 en lugar de ResNet34: |ΔMCC| ≤ 0.0035 en los cinco índices.
- Sin pesos de ImageNet: MCC 0.801 frente a 0.794 (NDVI). El preentreno acelera las primeras épocas, pero no
  mejora el resultado final.
- Las máscaras son etiquetas derivadas de los índices, no verdad de campo. NDVI usa umbral fijo y los demás Otsu,
  así que la comparación mezcla índice y método de umbral. Hay una sola semilla por configuración.

## Estructura

| Archivo | Qué hace |
|---|---|
| `download_timeseries.py` | Descarga la serie Sentinel-2 L2A (12 bandas + SCL) desde Copernicus Data Space |
| `analisis_preliminar.py` | Análisis exploratorio (índices, tendencias, puntuación de salud preliminar) |
| `generar_mascaras.py` | Índices, umbrales (Otsu solo con entrenamiento), máscaras 0/1/255 y RGB por escena; descarta escenas sin SCL |
| `entrenar.py` | Entrena y evalúa un modelo (U-Net con encoder de `segmentation_models_pytorch`, o línea base por píxel) |
| `correr_experimentos.sh` | Corre los 12 entrenamientos reportados |
| `analisis_resultados.py` | Tabla final con MCC, pruebas pareadas de Wilcoxon, encoders y reproducibilidad |
| `graficas_mascaras.py` | Figuras M1–M7 en `salidas/mascaras/` |
| `verificaciones/` | Pruebas de armonización radiométrica y de procedencia de los pesos preentrenados |
| `resultados/` | Métricas de cada modelo (GPU y réplica en CPU), umbrales y lista de escenas usadas |
| `entorno/overrides-rocm.txt` | Instalación de PyTorch-ROCm para GPU AMD RDNA2 |

## Cómo reproducir

### 1. Entorno (Python 3.13)

```bash
python3.13 -m venv venv
# CPU:
venv/bin/pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
venv/bin/pip install -r requirements.txt
```

Para GPU AMD (probado en RX 6600 con ROCm 7.14) ver `entorno/overrides-rocm.txt`. Con NVIDIA basta instalar la
rueda CUDA de `torch==2.15.0`; esa combinación no se probó.

### 2. Datos

```bash
cp .env.example .env        # poner usuario y contraseña de Copernicus Data Space
venv/bin/python download_timeseries.py          # 2015-01-01 a 2026-09-01, nubes ≤ 5 % → dataset/
```

Recuadro: 99.79°–99.53° O, 18.98°–19.45° N; malla de ~28 m (1024 × 1808 píxeles). La descarga usada tuvo 313
escenas. Copernicus puede reprocesar o agregar escenas, así que la lista exacta usada está en
`resultados/datos/escenas.csv` (308) y `resultados/datos/escenas_excluidas.txt` (5 sin banda SCL).

### 3. Máscaras

```bash
venv/bin/python generar_mascaras.py              # → datos/escenas/*.npz, datos/umbrales.json, datos/escenas.csv
```

Para cada píxel válido: 1 si índice ≥ umbral, 0 si índice < umbral; 255 si la SCL lo marca como nube, cirro,
sombra, nieve, sin dato, saturado o zona oscura (clases 0, 1, 2, 3, 8–11). Partición: entrenamiento 2015–2021
(146), validación 2022 (28) y prueba 2023–2026 (134). Debe producir exactamente `resultados/datos/umbrales.json`.

### 4. Entrenamientos

```bash
./correr_experimentos.sh                                         # CPU (~40 min por U-Net)
PY=venv/bin/python ROCM_GFX1030=1 ./correr_experimentos.sh       # RX 6600 (~3.7 min por U-Net)
```

Configuración (valores por defecto de `entrenar.py`): entrada RGB (B04, B03, B02) escalada linealmente de
reflectancia 0–0.4 a 0–255 y normalizada con media/desviación de ImageNet; parches aleatorios de 256 × 256 con al
menos 50 % de píxeles válidos, rotaciones de 90° y reflexión; pérdida BCE + Dice sobre píxeles válidos; Adam
(lr = 1e-4, decaimiento coseno); lote 16; 800 parches por época; máximo 10 épocas con paciencia 3 sobre el IoU de
validación; semilla 0. La evaluación de prueba se hace sobre la escena completa.

### 5. Análisis y figuras

```bash
venv/bin/python analisis_resultados.py modelos --comparar resultados/gpu   # tus resultados frente a los publicados
venv/bin/python analisis_resultados.py resultados/gpu --comparar resultados/cpu
venv/bin/python graficas_mascaras.py modelos      # M1–M7 (M4 requiere modelos/*/ejemplos_test.npz)
```

### 6. Verificaciones

```bash
venv/bin/python verificaciones/verificar_armonizacion.py   # sin salto radiométrico en el cambio de línea base de 2022
venv/bin/python verificaciones/verificar_pesos.py          # pesos de smp idénticos a torchvision / EfficientNet-PyTorch
```

## Qué tan reproducible es

- **Máscaras y umbrales:** deterministas. La muestra para Otsu usa una semilla fija por escena.
- **Entrenamiento:** con la misma semilla, la línea base por píxel da el mismo resultado en CPU y GPU. Las U-Net
  varían por el orden de las operaciones en punto flotante: IoU de prueba GPU − CPU = +0.0015 (NDVI), −0.0006 (NDRE)
  y +0.0117 (CIre). Diferencias de ese orden no deben interpretarse como reales.
- **Versiones:** la laptop (CPU) usó numpy 2.5.2 y pandas 3.0.6; la PC con GPU, numpy 2.5.4 y pandas 3.1.1.
- **No se versionan** las imágenes (12 GB), las máscaras (1.5 GB), los pesos entrenados (681 MB los 12 modelos) ni las
  predicciones guardadas; todo se regenera con los pasos anteriores.

## Referencias de los métodos usados

- Otsu, N. (1979). A Threshold Selection Method from Gray-Level Histograms. *IEEE Trans. SMC*, 9(1), 62–66. doi:10.1109/TSMC.1979.4310076
- He, K., Zhang, X., Ren, S., Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*, 770–778. doi:10.1109/CVPR.2016.90
- Deng, J., et al. (2009). ImageNet: A large-scale hierarchical image database. *CVPR 2009*, 248–255. doi:10.1109/CVPR.2009.5206848
- Tan, M., Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. arXiv:1905.11946
