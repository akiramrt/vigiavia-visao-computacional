"""
05_benchmark_velocidade.py — Mede latência e FPS de cada modelo em GPU e CPU, com imagens
reais do conjunto de teste (aquecimento + N repetições; média e percentil 95).

"Tempo real" aqui = >= 25 FPS (vídeo padrão) para câmera veicular em GPU/edge e >= 10 FPS
em CPU para uso em celulares/notebooks (o rastreamento tolera quadros pulados).

Uso:
    python scripts/05_benchmark_velocidade.py --data datasets/vigiavia-rdd2022/data.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "False")   # nada de pip install em tempo de execução

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from ultralytics import YOLO  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]


def medir(modelo: YOLO, imagens: list[np.ndarray], device, n: int, half: bool = False) -> dict:
    for im in imagens[:15]:                                     # aquecimento
        modelo.predict(im, imgsz=640, device=device, half=half, verbose=False)
    tempos = []
    for i in range(n):
        im = imagens[i % len(imagens)]
        if device != "cpu":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        modelo.predict(im, imgsz=640, device=device, half=half, verbose=False)
        if device != "cpu":
            torch.cuda.synchronize()
        tempos.append((time.perf_counter() - t0) * 1000)
    t = np.array(tempos)
    return {"ms_media": round(float(t.mean()), 2), "ms_p95": round(float(np.percentile(t, 95)), 2),
            "fps": round(float(1000 / t.mean()), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    raiz_ds = args.data.resolve().parent
    imagens = [cv2.imread(str(p)) for p in sorted((raiz_ds / "images" / "test").glob("*.jpg"))[:60]]

    linhas = []
    for pt in sorted((RAIZ / "modelos").glob("vigiavia_E*.pt")):
        nome = pt.stem.replace("vigiavia_", "")
        m = YOLO(str(pt))
        cfgs = [("PyTorch GPU FP16", 0, True), ("PyTorch GPU FP32", 0, False), ("PyTorch CPU FP32", "cpu", False)]
        for rotulo, dev, half in cfgs:
            if dev != "cpu" and not torch.cuda.is_available():
                continue
            r = medir(m, imagens, dev, args.n if dev != "cpu" else max(60, args.n // 3), half)
            linhas.append({"modelo": nome, "execucao": rotulo, **r}); print(nome, rotulo, r)
    onnx = RAIZ / "modelos" / "vigiavia_buracos.onnx"
    if onnx.exists():
        r = medir(YOLO(str(onnx), task="detect"), imagens, "cpu", max(60, args.n // 3))
        linhas.append({"modelo": "produção (ONNX)", "execucao": "ONNX Runtime CPU", **r}); print("onnx", r)

    with open(RAIZ / "resultados" / "velocidade.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0])); w.writeheader(); w.writerows(linhas)
    info = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cpu": "AMD Ryzen 9 5900X (12 núcleos)", "imgsz": 640, "linhas": linhas}
    (RAIZ / "resultados" / "velocidade.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9, 4.6))
    rot = [f"{l['modelo']}\n{l['execucao']}" for l in linhas]
    cores = ["#2a6fdb" if "GPU" in l["execucao"] else "#f5a623" for l in linhas]
    b = ax.barh(rot, [l["fps"] for l in linhas], color=cores)
    ax.bar_label(b, fmt="%.0f FPS", fontsize=8, padding=3)
    ax.axvline(25, color="#d0021b", ls="--", lw=1); ax.text(25, -0.9, " 25 FPS (vídeo)", color="#d0021b", fontsize=8)
    ax.axvline(10, color="#7b7b7b", ls=":", lw=1)
    ax.invert_yaxis(); ax.set_xlabel("quadros por segundo (só inferência, imagem 640 px)")
    ax.set_title("Velocidade de inferência — azul: GPU RTX 3070 · laranja: CPU Ryzen 9 5900X")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(RAIZ / "resultados" / "velocidade.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
