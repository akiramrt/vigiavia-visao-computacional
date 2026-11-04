"""
01b_auditar_rotulos.py — Auditoria visual dos rótulos: recorta caixas "buraco" aleatórias de cada país e monta
uma grade por país. Foi esta checagem que revelou que uma redistribuição do RDD2022 rotulava marcas de pista,
faixas de pedestre e tampas de bueiro como "pothole" (ver relatório, seção 4).

Uso:
    python scripts/01b_auditar_rotulos.py --data datasets/vigiavia-rdd2022 --saida resultados/auditoria_rotulos.jpg
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def recortes(raiz: Path, arquivos: list[str], n: int, rng: random.Random) -> list[np.ndarray]:
    rng.shuffle(arquivos)
    saida = []
    for arq in arquivos:
        split, nome = arq.split("/", 1)
        lbl = raiz / "labels" / split / (Path(nome).stem + ".txt")
        linhas = [l for l in lbl.read_text().splitlines() if l.strip()] if lbl.exists() else []
        if not linhas:
            continue
        im = cv2.imread(str(raiz / "images" / split / nome))
        h, w = im.shape[:2]
        _, cx, cy, bw, bh = map(float, rng.choice(linhas).split())
        m = 0.15
        x1, y1 = int(max(0, (cx - bw / 2 - m * bw) * w)), int(max(0, (cy - bh / 2 - m * bh) * h))
        x2, y2 = int(min(w, (cx + bw / 2 + m * bw) * w)), int(min(h, (cy + bh / 2 + m * bh) * h))
        if x2 - x1 > 4 and y2 - y1 > 4:
            saida.append(cv2.resize(im[y1:y2, x1:x2], (150, 110)))
        if len(saida) >= n:
            break
    return saida


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="pasta do dataset (com manifest.csv)")
    ap.add_argument("--saida", type=Path, required=True)
    ap.add_argument("--por-pais", type=int, default=16)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    por_pais = defaultdict(list)
    with open(args.data / "manifest.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(r["n_buracos"]) > 0:
                por_pais[r["pais"]].append(f"{r['split']}/{r['arquivo']}")
    linhas = []
    for pais in sorted(por_pais):
        rc = recortes(args.data, por_pais[pais], args.por_pais, rng)
        rc += [np.full((110, 150, 3), 40, np.uint8)] * (args.por_pais - len(rc))
        faixa = np.full((110, 180, 3), 30, np.uint8)
        cv2.putText(faixa, pais.replace("_", " ")[:14], (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        linhas.append(np.hstack([faixa] + rc[: args.por_pais // 2]))
        linhas.append(np.hstack([np.full((110, 180, 3), 30, np.uint8)] + rc[args.por_pais // 2:]))
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.saida), np.vstack(linhas), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("Grade salva em", args.saida)


if __name__ == "__main__":
    main()
