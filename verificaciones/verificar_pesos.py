"""
Verifica la procedencia de los pesos preentrenados que usa segmentation_models_pytorch (smp 0.6.0):

  * ResNet34  : Hugging Face smp-hub/resnet34.imagenet  ==  torchvision ResNet34_Weights.IMAGENET1K_V1
  * EfficientNet-B0 : Hugging Face smp-hub/efficientnet-b0.imagenet  ==  efficientnet-b0-355c32eb.pth
                      (release 1.0 de lukemelas/EfficientNet-PyTorch, archivo citado en la ficha de Hugging Face)

Compara tensor por tensor (torch.equal) y revisa que el SHA-256 de cada archivo original empiece por el hash que
lleva en su nombre. Descarga ~105 MB la primera vez (quedan en la caché de torch / Hugging Face).

Uso:  python verificaciones/verificar_pesos.py
Resultado observado: 182/182 tensores idénticos (ResNet34) y 360/360 (EfficientNet-B0).
"""
import hashlib
import os

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import segmentation_models_pytorch as smp

EFF_URL = "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth"


def pesos_hf(encoder):
    s = smp.encoders.encoders[encoder]["pretrained_settings"]["imagenet"]
    archivo = hf_hub_download(s["repo_id"], "model.safetensors", revision=s["revision"])
    return load_file(archivo), f'{s["repo_id"]}@{s["revision"][:8]}'


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def comparar(nombre, hf, orig, origen_hf, origen_orig):
    comunes = set(hf) & set(orig)
    solo_hf = sorted(set(hf) - set(orig))
    solo_orig = [k for k in sorted(set(orig) - set(hf)) if not k.endswith("num_batches_tracked")]
    iguales = all(torch.equal(hf[k], orig[k]) for k in comunes)
    print(f"[{nombre}] {origen_hf}  vs  {origen_orig}")
    print(f"   tensores: HF {len(hf)} | original {len(orig)} | comunes {len(comunes)} | "
          f"solo HF {solo_hf} | solo original (sin contadores BN) {solo_orig}")
    print(f"   idénticos bit a bit: {iguales}")
    return iguales and not solo_hf


def main():
    from torchvision.models import resnet34, ResNet34_Weights
    w = ResNet34_Weights.IMAGENET1K_V1
    tv = resnet34(weights=w).state_dict()
    ruta_tv = os.path.join(torch.hub.get_dir(), "checkpoints", os.path.basename(w.url))
    hash_nombre = os.path.basename(w.url).split("-")[1].split(".")[0]
    print(f"torchvision {os.path.basename(w.url)}: SHA-256 empieza por el hash del nombre: {sha256(ruta_tv).startswith(hash_nombre)}"
          f" | ImageNet top-1 reportado: {w.meta['_metrics']['ImageNet-1K']['acc@1']}")
    hf, origen = pesos_hf("resnet34")
    ok1 = comparar("ResNet34", hf, tv, origen, f"torchvision {os.path.basename(w.url)}")

    ruta = os.path.join(torch.hub.get_dir(), "checkpoints", os.path.basename(EFF_URL))
    if not os.path.exists(ruta):
        torch.hub.download_url_to_file(EFF_URL, ruta, progress=False)
    print(f"{os.path.basename(EFF_URL)}: SHA-256 empieza por el hash del nombre: {sha256(ruta).startswith('355c32eb')}")
    orig = torch.load(ruta, map_location="cpu", weights_only=True)
    hf, origen = pesos_hf("efficientnet-b0")
    ok2 = comparar("EfficientNet-B0", hf, orig, origen, os.path.basename(EFF_URL))
    print("\nTODO VERIFICADO" if ok1 and ok2 else "\nHAY DIFERENCIAS")


if __name__ == "__main__":
    main()
