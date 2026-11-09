"""
03b_cenas_repetidas.py — Mede quantas imagens de teste mostram o MESMO trecho de via de alguma imagem de treino
e quanto isso infla as métricas.

Por quê: o split 70/15/15 foi feito por blocos de 20 números de arquivo, supondo que números vizinhos fossem
quadros vizinhos do mesmo trajeto. No RDD2022 isso não vale (a distância dHash entre números vizinhos é igual à de
pares aleatórios), então o mesmo trecho de via pode cair no treino e no teste em fotos tiradas em momentos
diferentes. O dHash do 03_avaliar.py só pega quase-cópias; aqui usamos correspondência geométrica.

Método (por país):
  1. para cada imagem de teste, os 30 vizinhos de treino mais próximos pelo dHash (256 bits);
  2. ORB (1.500 pontos, sem capô e sem a faixa do topo) + teste da razão de Lowe + homografia RANSAC;
  3. "trecho repetido" = 20 ou mais correspondências geométricas (inliers) com deslocamento mediano >= 2 px
     (descarta marcas fixas do para-brisa). Também reportamos o limiar 40 (mais conservador);
  4. AP@0,5 (interpolação de 101 pontos), precisão, revocação e F1 no limiar de operação, com e sem essas imagens.

Uso:
    python scripts/03b_cenas_repetidas.py --data datasets/vigiavia-rdd2022/data.yaml
Saídas: resultados/cenas_repetidas.json e resultados/cenas_repetidas.jpg
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "False")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "resultados"
LADO = 640


def carregar(ds: Path, linha: dict) -> np.ndarray:
    im = cv2.imread(str(ds / "images" / linha["split"] / linha["arquivo"]), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(im, (LADO, LADO), interpolation=cv2.INTER_AREA)


def dhash(g: np.ndarray) -> np.ndarray:
    d = cv2.resize(g, (17, 16), interpolation=cv2.INTER_AREA)
    return (d[:, 1:] > d[:, :-1]).ravel()


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def rotulos(arq: Path, w: int, h: int) -> np.ndarray:
    if not arq.exists():
        return np.zeros((0, 4))
    out = []
    for linha in arq.read_text().splitlines():
        if linha.strip():
            _, cx, cy, bw, bh = (float(v) for v in linha.split())
            cx, cy, bw, bh = cx * w, cy * h, bw * w, bh * h
            out.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
    return np.array(out).reshape(-1, 4)


def metricas(regs: list[dict], limiar: float) -> dict:
    cf = np.concatenate([r["cf"] for r in regs]); tp = np.concatenate([r["tp"] for r in regs])
    n_gt = sum(r["ng"] for r in regs)
    o = np.argsort(-cf); tp, cf = tp[o], cf[o]
    t, f = np.cumsum(tp), np.cumsum(~tp)
    rc, pc = t / max(n_gt, 1), t / np.maximum(t + f, 1)
    mr = np.concatenate(([0], rc, [1])); mp = np.concatenate(([1], pc, [0]))
    mp = np.flip(np.maximum.accumulate(np.flip(mp)))
    x = np.linspace(0, 1, 101)
    ap = float(np.trapezoid(np.interp(x, mr, mp), x))
    k = cf >= limiar
    vp, fp = int(tp[k].sum()), int((~tp[k]).sum()); fn = n_gt - vp
    return {"imagens": len(regs), "buracos": n_gt, "AP50": round(ap, 3), "VP": vp, "FP": fp, "FN": fn,
            "precisao": round(vp / max(vp + fp, 1), 3), "revocacao": round(vp / max(n_gt, 1), 3),
            "f1": round(2 * vp / max(2 * vp + fp + fn, 1), 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--pesos", type=Path, default=RAIZ / "modelos" / "vigiavia_buracos.pt")
    ap.add_argument("--vizinhos", type=int, default=30)
    args = ap.parse_args()
    ds = args.data.resolve().parent
    cv2.setRNGSeed(42)
    linhas = [r for r in csv.DictReader(open(ds / "manifest.csv", encoding="utf-8")) if r["split"] in ("train", "test")]
    mascara = np.zeros((LADO, LADO), np.uint8)
    mascara[int(0.05 * LADO):int(0.78 * LADO), :] = 255           # sem a faixa do topo e sem o capô
    orb = cv2.ORB_create(1500)
    hashes, pontos, por_pais = {}, {}, {}
    for r in linhas:
        g = carregar(ds, r)
        hashes[r["arquivo"]] = dhash(g)
        kp, des = orb.detectAndCompute(g, mascara)
        pontos[r["arquivo"]] = (np.array([p.pt for p in kp], np.float32), des)
        por_pais.setdefault(r["pais"], {"train": [], "test": []})[r["split"]].append(r["arquivo"])

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    cenas = []
    for pais, d in por_pais.items():
        treino = d["train"]
        matriz = np.array([hashes[a] for a in treino])
        for t in d["test"]:
            pa, da = pontos[t]
            melhor = (0, None, 0.0)
            if da is not None and len(pa) >= 20:
                for i in np.argsort((matriz != hashes[t]).sum(1))[:args.vizinhos]:
                    pb, db = pontos[treino[i]]
                    if db is None or len(pb) < 20:
                        continue
                    bons = [m for par in bf.knnMatch(da, db, k=2) if len(par) == 2
                            for m, n in [par] if m.distance < 0.75 * n.distance]
                    if len(bons) < 15:
                        continue
                    src = pa[[m.queryIdx for m in bons]]; dst = pb[[m.trainIdx for m in bons]]
                    H, ok = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
                    if H is None:
                        continue
                    sel = ok.ravel() == 1
                    desloc = float(np.median(np.linalg.norm(src[sel] - dst[sel], axis=1)))
                    if desloc >= 2.0 and int(sel.sum()) > melhor[0]:
                        melhor = (int(sel.sum()), treino[i], desloc)
            cenas.append({"teste": t, "pais": pais, "inliers": melhor[0], "treino": melhor[1]})
        print(f"{pais}: {sum(c['inliers'] >= 20 for c in cenas if c['pais'] == pais)}/{len(d['test'])} com trecho repetido",
              flush=True)

    from ultralytics import YOLO
    op = json.loads((RAIZ / "configs" / "operacao.json").read_text(encoding="utf-8"))
    modelo = YOLO(str(args.pesos))
    por_img = {c["teste"]: c for c in cenas}
    regs = []
    for p in sorted((ds / "images" / "test").glob("*.jpg")):
        im = cv2.imread(str(p)); h, w = im.shape[:2]
        r = modelo.predict(im, conf=0.001, iou=0.7, imgsz=LADO, max_det=300, verbose=False)[0]
        pr, cf = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
        gt = rotulos(ds / "labels" / "test" / f"{p.stem}.txt", w, h)
        o = np.argsort(-cf); pr, cf = pr[o], cf[o]
        m, usado, tp = iou(pr, gt), np.zeros(len(gt), bool), np.zeros(len(pr), bool)
        for i in range(len(pr)):
            livres = np.where(~usado)[0]
            if len(livres) == 0:
                break
            j = livres[np.argmax(m[i, livres])]
            if m[i, j] >= 0.5:
                tp[i] = usado[j] = True
        regs.append({"cf": cf, "tp": tp, "ng": len(gt), "inliers": por_img[p.name]["inliers"]})

    lim = op["conf"]
    res = {"metodo": __doc__.split("Método (por país):")[1].split("Uso:")[0].strip(),
           "limiar_operacao": lim,
           "por_pais": {pais: {"imagens_teste": len(d["test"]),
                               "trecho_repetido_inliers_20": sum(c["inliers"] >= 20 for c in cenas if c["pais"] == pais),
                               "trecho_repetido_inliers_40": sum(c["inliers"] >= 40 for c in cenas if c["pais"] == pais)}
                        for pais, d in por_pais.items()},
           "metricas": {"todas": metricas(regs, lim)}}
    for corte in (20, 40):
        res["metricas"][f"com_trecho_repetido_{corte}"] = metricas([r for r in regs if r["inliers"] >= corte], lim)
        res["metricas"][f"sem_trecho_repetido_{corte}"] = metricas([r for r in regs if r["inliers"] < corte], lim)
    res["pares_exemplo"] = sorted(({"teste": c["teste"], "treino": c["treino"], "inliers": c["inliers"]}
                                   for c in cenas if c["inliers"] >= 40), key=lambda c: -c["inliers"])[:40]
    (RES / "cenas_repetidas.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    rng = random.Random(42)
    fortes = [c for c in cenas if c["inliers"] >= 40]
    amostra = sorted(rng.sample(fortes, min(8, len(fortes))), key=lambda c: c["pais"])
    celulas = []
    for c in amostra:
        a = cv2.resize(cv2.imread(str(ds / "images" / "test" / c["teste"])), (300, 300))
        b = cv2.resize(cv2.imread(str(ds / "images" / "train" / c["treino"])), (300, 300))
        for im, txt in ((a, f"teste {c['teste'][:-4]}"), (b, f"treino {c['treino'][:-4]}")):
            cv2.rectangle(im, (0, 0), (300, 22), (20, 20, 20), -1)
            cv2.putText(im, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        celulas.append(np.hstack([a, np.full((300, 4, 3), 255, np.uint8), b]))
    separador = np.full((300, 16, 3), 255, np.uint8)
    pares = [np.hstack([celulas[i], separador, celulas[i + 1]]) for i in range(0, len(celulas) - 1, 2)]
    grade = np.vstack([np.vstack([p, np.full((12, p.shape[1], 3), 255, np.uint8)]) for p in pares])
    cv2.imwrite(str(RES / "cenas_repetidas.jpg"), grade, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(json.dumps(res["por_pais"], ensure_ascii=False))
    print(json.dumps(res["metricas"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
