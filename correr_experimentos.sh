#!/usr/bin/env bash
# Reproduce los 12 entrenamientos reportados (misma configuración por defecto de entrenar.py, semilla 0).
#
# Uso:
#   ./correr_experimentos.sh                      # usa venv/bin/python, escribe en modelos/
#   PY=~/rp_gpu/venv/bin/python ./correr_experimentos.sh
#   ROCM_GFX1030=1 ./correr_experimentos.sh       # GPU AMD RDNA2 sin soporte oficial (p. ej. RX 6600, gfx1032)
#
# Requiere haber corrido antes:  python generar_mascaras.py   (crea datos/)
# Duración medida: ~3.7 min por U-Net en una RX 6600; ~40 min por U-Net en CPU (8 núcleos).
set -u
cd "$(dirname "$0")"
PY="${PY:-venv/bin/python}"
LOG="${LOG:-modelos/cola.log}"
mkdir -p "$(dirname "$LOG")"
if [ "${ROCM_GFX1030:-0}" = "1" ]; then
  export HSA_OVERRIDE_GFX_VERSION=10.3.0   # sin esto, ROCm 7.14 falla con hipErrorInvalidImage en gfx1032
fi

run() {
  echo "### $(date '+%F %T') inicio: $*"
  "$PY" -u entrenar.py "$@" 2>&1 | grep --line-buffered -vi warning
  echo "### $(date '+%F %T') fin: $* (código ${PIPESTATUS[0]})"
}

{
  # Experimento principal: misma red (U-Net + ResNet34 ImageNet), 5 máscaras, y línea base por píxel
  run --indice NDVI
  run --indice NDVI --modelo pixel --epocas 5
  for i in NDRE CIre EVI NDMI; do run --indice "$i"; done
  # O2: ablación del preentrenamiento
  run --indice NDVI --sin_preentreno
  # O1: mismo experimento con EfficientNet-B0
  for i in NDVI NDRE CIre EVI NDMI; do run --indice "$i" --encoder efficientnet-b0; done
  echo "### COLA TERMINADA $(date '+%F %T')"
} 2>&1 | tee "$LOG"
