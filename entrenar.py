"""
Paso 2 del modelado: entrenar una red RGB -> máscara de vegetación (una máscara/índice por corrida).

Modelos:
  * unet  : U-Net con encoder preentrenado en ImageNet (segmentation_models_pytorch), p. ej. resnet34.
  * pixel : línea base, regresión logística por píxel (convolución 1×1 sobre RGB, sin contexto espacial).

Entrada  : RGB (B04, B03, B02) normalizado con media/desv. de ImageNet. Las bandas con las que se calcula
           cada índice (B05, B07, B08, B8A, B11) NO entran a la red, para que no "aprenda la fórmula".
Etiqueta : máscara 0/1 de generar_mascaras.py; 255 = píxel inválido (nube/sombra/nieve), se ignora.
Pérdida  : BCE + Dice, sólo sobre píxeles válidos.
Partición: train 2015–2021 (parches aleatorios 256×256 + giros/reflexiones), val 2022 (parada temprana),
           test 2023–2026 (escena completa, nunca vista).

Uso:
  python entrenar.py --indice NDVI
  python entrenar.py --indice NDVI --modelo pixel
  python entrenar.py --indice NDRE --encoder resnet34 --epocas 10
  python entrenar.py --indice NDVI --sin_preentreno          (ablación del preentrenamiento)
  python entrenar.py --indice NDVI --encoder efficientnet-b0 (comparación de encoder, opcional)
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORAR = 255
MEDIA = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
DESV = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def args_():
    p = argparse.ArgumentParser()
    p.add_argument("--indice", default="NDVI")
    p.add_argument("--modelo", default="unet", choices=["unet", "pixel"])
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--epocas", type=int, default=10)
    p.add_argument("--paciencia", type=int, default=3)
    p.add_argument("--parches_epoca", type=int, default=800)
    p.add_argument("--parche", type=int, default=256)
    p.add_argument("--lote", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--datos", default="datos")
    p.add_argument("--sin_preentreno", action="store_true",
                   help="encoder con pesos aleatorios (ablación: ¿sirven los pesos de ImageNet?)")
    return p.parse_args()


def particion(nombre):
    anio = int(os.path.basename(nombre).split("_")[1][:4])
    return "train" if anio <= 2021 else ("val" if anio == 2022 else "test")


def cargar(archivos, indice):
    rgb, msk, fechas = [], [], []
    for a in archivos:
        z = np.load(a)
        rgb.append(z["rgb"])
        msk.append(z[indice])
        fechas.append(os.path.basename(a).split("_")[1].replace(".npz", ""))
    return rgb, msk, fechas


def normalizar(x_uint8):
    x = torch.as_tensor(x_uint8).float() / 255.0
    if x.dim() == 3:
        x = x.unsqueeze(0)
    return (x - MEDIA) / DESV


def lote_aleatorio(rgb, msk, n, t, rng):
    """n parches t×t de escenas aleatorias, con al menos 50 % de píxeles válidos, con aumento de datos."""
    xs, ys = [], []
    while len(xs) < n:
        i = rng.integers(len(rgb))
        _, h, w = rgb[i].shape
        r, c = rng.integers(h - t + 1), rng.integers(w - t + 1)
        y = msk[i][r:r + t, c:c + t]
        if (y != IGNORAR).mean() < 0.5:
            continue
        x = rgb[i][:, r:r + t, c:c + t]
        k = rng.integers(4)
        x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
        if rng.random() < 0.5:
            x, y = x[:, :, ::-1], y[:, ::-1]
        xs.append(np.ascontiguousarray(x))
        ys.append(np.ascontiguousarray(y))
    return normalizar(np.stack(xs)), torch.as_tensor(np.stack(ys)).long()


def perdida(logits, y):
    valido = (y != IGNORAR).float()
    yf = (y == 1).float()
    bce = (F.binary_cross_entropy_with_logits(logits, yf, reduction="none") * valido).sum() / valido.sum().clamp(min=1)
    p = torch.sigmoid(logits) * valido
    dice = 1 - (2 * (p * yf).sum() + 1) / (p.sum() + (yf * valido).sum() + 1)
    return bce + dice


def construir(a):
    if a.modelo == "pixel":
        return nn.Conv2d(3, 1, kernel_size=1)
    import segmentation_models_pytorch as smp
    return smp.Unet(a.encoder, encoder_weights=None if a.sin_preentreno else "imagenet", in_channels=3, classes=1)


@torch.no_grad()
def predecir_escena(modelo, rgb, dispositivo):
    """Predice la escena completa (rellena a múltiplo de 32 para la U-Net) y regresa probabilidad H×W."""
    _, h, w = rgb.shape
    ph, pw = (-h) % 32, (-w) % 32
    x = normalizar(rgb)
    x = F.pad(x, (0, pw, 0, ph), mode="reflect").to(dispositivo)
    return torch.sigmoid(modelo(x))[0, 0, :h, :w].cpu().numpy()


def confusion(prob, y, umbral=0.5):
    v = y != IGNORAR
    p, t = prob[v] > umbral, y[v] == 1
    return dict(tp=int((p & t).sum()), fp=int((p & ~t).sum()), fn=int((~p & t).sum()), tn=int((~p & ~t).sum()))


def metricas(c):
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return dict(IoU=tp / max(tp + fp + fn, 1), F1=2 * tp / max(2 * tp + fp + fn, 1), precision=prec,
                recall=rec, exactitud=(tp + tn) / max(tp + fp + fn + tn, 1))


def evaluar(modelo, rgb, msk, dispositivo):
    total = dict(tp=0, fp=0, fn=0, tn=0)
    por_escena = []
    for x, y in zip(rgb, msk):
        c = confusion(predecir_escena(modelo, x, dispositivo), y)
        por_escena.append(c)
        for k in total:
            total[k] += c[k]
    return metricas(total), por_escena


def main():
    a = args_()
    torch.manual_seed(a.semilla)
    rng = np.random.default_rng(a.semilla)
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"  # ROCm (AMD) también aparece como "cuda"
    nombre = f"{a.indice}_{a.modelo}" + (f"_{a.encoder}" if a.modelo == "unet" else "") + ("_sin_preentreno" if a.sin_preentreno else "")
    salida = os.path.join("modelos", nombre)
    os.makedirs(salida, exist_ok=True)
    print(f"[{nombre}] dispositivo={dispositivo}, hilos={torch.get_num_threads()}")

    archivos = sorted(glob.glob(os.path.join(a.datos, "escenas", "*.npz")))
    parte = {s: [f for f in archivos if particion(f) == s] for s in ("train", "val", "test")}
    print({s: len(v) for s, v in parte.items()})
    tr_rgb, tr_msk, _ = cargar(parte["train"], a.indice)
    va_rgb, va_msk, _ = cargar(parte["val"], a.indice)

    modelo = construir(a).to(dispositivo)
    opt = torch.optim.Adam(modelo.parameters(), lr=a.lr if a.modelo == "unet" else 1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epocas)
    pasos = a.parches_epoca // a.lote
    mejor, sin_mejora, hist = -1, 0, []

    for ep in range(1, a.epocas + 1):
        modelo.train()
        t0, acum = time.time(), 0.0
        for _ in range(pasos):
            x, y = lote_aleatorio(tr_rgb, tr_msk, a.lote, a.parche, rng)
            x, y = x.to(dispositivo), y.to(dispositivo)
            opt.zero_grad()
            l = perdida(modelo(x)[:, 0], y)
            l.backward()
            opt.step()
            acum += l.item()
        sched.step()
        modelo.eval()
        m_val, _ = evaluar(modelo, va_rgb, va_msk, dispositivo)
        fila = dict(epoca=ep, perdida_train=acum / pasos, **{f"val_{k}": v for k, v in m_val.items()},
                    segundos=time.time() - t0)
        hist.append(fila)
        print(f"ep {ep:2d}  loss {fila['perdida_train']:.4f}  val IoU {m_val['IoU']:.4f}  F1 {m_val['F1']:.4f}"
              f"  ({fila['segundos']:.0f}s)", flush=True)
        pd.DataFrame(hist).to_csv(os.path.join(salida, "historial.csv"), index=False)
        if m_val["IoU"] > mejor:
            mejor, sin_mejora = m_val["IoU"], 0
            torch.save(modelo.state_dict(), os.path.join(salida, "mejor.pt"))
        else:
            sin_mejora += 1
            if sin_mejora >= a.paciencia:
                print("parada temprana")
                break

    # ------------------------------------------------------------------ prueba 2023–2026 (nunca vista)
    del tr_rgb, tr_msk
    modelo.load_state_dict(torch.load(os.path.join(salida, "mejor.pt"), map_location=dispositivo))
    modelo.eval()
    te_rgb, te_msk, te_fechas = cargar(parte["test"], a.indice)
    m_test, por_escena = evaluar(modelo, te_rgb, te_msk, dispositivo)
    df = pd.DataFrame(por_escena)
    df.insert(0, "fecha", te_fechas)
    df["anio"] = df["fecha"].str[:4].astype(int)
    df.to_csv(os.path.join(salida, "test_por_escena.csv"), index=False)
    por_anio = {int(an): metricas(g[["tp", "fp", "fn", "tn"]].sum().to_dict()) for an, g in df.groupby("anio")}

    # guarda predicciones de algunas escenas de prueba para las figuras
    ejemplos = [i for i in np.linspace(0, len(te_rgb) - 1, 4).astype(int)]
    np.savez_compressed(os.path.join(salida, "ejemplos_test.npz"),
                        fechas=np.array([te_fechas[i] for i in ejemplos]),
                        rgb=np.stack([te_rgb[i] for i in ejemplos]),
                        y=np.stack([te_msk[i] for i in ejemplos]),
                        prob=np.stack([predecir_escena(modelo, te_rgb[i], dispositivo) for i in ejemplos]))
    res = dict(modelo=nombre, indice=a.indice, args=vars(a), mejor_val_IoU=mejor, test=m_test, test_por_anio=por_anio,
               n_escenas={s: len(v) for s, v in parte.items()})
    with open(os.path.join(salida, "metricas.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("TEST 2023–2026:", {k: round(v, 4) for k, v in m_test.items()})


if __name__ == "__main__":
    main()
