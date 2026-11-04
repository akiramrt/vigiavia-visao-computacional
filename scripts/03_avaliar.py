"""
03_avaliar.py — Avaliação completa, escolha do modelo de produção e análise de vieses.

Etapas:
  1. Tabela comparativa dos experimentos (resultados/E*/metricas.json).
  2. Escolha do modelo de produção (regra explícita, abaixo) e do limiar (máximo F1 na VALIDAÇÃO).
  3. No TESTE, com o modelo escolhido:
       * AP por país de origem (Ultralytics val em subconjuntos);
       * precisão/revocação/F1 no limiar por país, por iluminação (tercis de brilho) e
         revocação por tamanho do buraco (pequeno/médio/grande, critério COCO);
       * mosaicos de acertos, falsos positivos e falsos negativos (análise de erros).
  4. Calibração da severidade: tercis do índice de severidade nas caixas de TREINO.
  5. Grava configs/operacao.json e configs/severidade.json.

Regra de escolha: entre os modelos cujo mAP@0,5:0,95 de validação fica a até 1,5 p.p. do
melhor, escolhe o de MENOR latência (o produto roda em celulares e câmeras embarcadas).

Uso:
    python scripts/03_avaliar.py --data datasets/vigiavia-rdd2022/data.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from ultralytics import YOLO  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from vigiavia import severidade as sev  # noqa: E402

RES = RAIZ / "resultados"
TOLERANCIA_MAP = 0.015


# ----------------------------------------------------------------------------- utilidades
def ler_rotulos(lbl: Path, w: int, h: int) -> np.ndarray:
    if not lbl.exists():
        return np.zeros((0, 4))
    linhas = [l.split() for l in lbl.read_text().splitlines() if l.strip()]
    caixas = []
    for _, cx, cy, bw, bh in linhas:
        cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        caixas.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
    return np.array(caixas).reshape(-1, 4)


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def casar(pred: np.ndarray, conf: np.ndarray, gt: np.ndarray, limiar_iou: float = 0.5):
    """Casamento guloso por confiança. Retorna (tp_pred bool[], gt_casado bool[])."""
    ordem = np.argsort(-conf)
    m = iou(pred, gt)
    tp = np.zeros(len(pred), bool); usado = np.zeros(len(gt), bool)
    for i in ordem:
        if len(gt) == 0:
            break
        cand = np.where(~usado)[0]
        if len(cand) == 0:
            break
        j = cand[np.argmax(m[i, cand])]
        if m[i, j] >= limiar_iou:
            tp[i] = True; usado[j] = True
    return tp, usado


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precisao": round(p, 4), "revocacao": round(r, 4), "f1": round(f1, 4)}


def tam_coco(area_px: float) -> str:
    return "pequeno (<32²px)" if area_px < 32 ** 2 else "médio (32²–96²px)" if area_px < 96 ** 2 else "grande (>96²px)"


def desenhar_caso(img, gt, pred, conf, tp, usado):
    vis = img.copy()
    for (x1, y1, x2, y2), ok in zip(gt.astype(int), usado):
        cv2.rectangle(vis, (x1, y1), (x2, y2), (60, 220, 60) if ok else (255, 0, 255), 2)
    for (x1, y1, x2, y2), c, ok in zip(pred.astype(int), conf, tp):
        cor = (255, 160, 0) if ok else (0, 0, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), cor, 2)
        cv2.putText(vis, f"{c:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)
    return vis


def mosaico(imgs: list[np.ndarray], titulo: str, destino: Path, colunas: int = 4, lado: int = 360) -> None:
    if not imgs:
        return
    ims = [cv2.resize(i, (lado, lado)) for i in imgs]
    while len(ims) % colunas:
        ims.append(np.full((lado, lado, 3), 30, np.uint8))
    linhas = [np.hstack(ims[i:i + colunas]) for i in range(0, len(ims), colunas)]
    grade = np.vstack(linhas)
    faixa = np.full((44, grade.shape[1], 3), 25, np.uint8)
    cv2.putText(faixa, titulo, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(str(destino), np.vstack([faixa, grade]), [cv2.IMWRITE_JPEG_QUALITY, 90])


# ----------------------------------------------------------------------------- etapas
def comparar_experimentos() -> list[dict]:
    linhas = []
    for f in sorted(RES.glob("E*/metricas.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        v, t = m["val"], m["test"]
        linhas.append({
            "experimento": m["experimento"], "pesos_iniciais": m["pesos_iniciais"],
            "parametros_M": round(m["n_parametros"] / 1e6, 2), "treino_min": m["duracao_treino_min"],
            "val_mAP50": round(v["mAP50"], 4), "val_mAP50_95": round(v["mAP50_95"], 4),
            "val_F1": round(v.get("melhor_f1", 0), 4), "limiar": round(v.get("limiar_conf_melhor_f1", 0.25), 3),
            "teste_mAP50": round(t["mAP50"], 4), "teste_mAP50_95": round(t["mAP50_95"], 4),
            "teste_precisao": round(t["precisao"], 4), "teste_revocacao": round(t["revocacao"], 4),
            "inferencia_ms_gpu": t["velocidade_ms_por_imagem"]["inference"],
            "_metricas": m,
        })
    if not linhas:
        raise SystemExit("Nenhum resultados/E*/metricas.json — rode scripts/02_treinar.py antes.")
    campos = [k for k in linhas[0] if not k.startswith("_")]
    with open(RES / "comparacao_modelos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos); w.writeheader()
        for l in linhas:
            w.writerow({k: l[k] for k in campos})
    # gráfico
    fig, ax = plt.subplots(figsize=(8, 4.2))
    nomes = [l["experimento"] for l in linhas]; x = np.arange(len(nomes)); lw = 0.38
    b1 = ax.bar(x - lw / 2, [l["teste_mAP50"] for l in linhas], lw, label="mAP@0,5 (teste)", color="#2a6fdb")
    b2 = ax.bar(x + lw / 2, [l["teste_mAP50_95"] for l in linhas], lw, label="mAP@0,5:0,95 (teste)", color="#f5a623")
    ax.bar_label(b1, fmt="%.3f", fontsize=9); ax.bar_label(b2, fmt="%.3f", fontsize=9)
    ax.set_xticks(x, [f"{n}\n{l['parametros_M']} M parâm. · {l['inferencia_ms_gpu']:.1f} ms" for n, l in zip(nomes, linhas)])
    ax.set_ylim(0, 1); ax.set_ylabel("métrica"); ax.legend(loc="upper right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Comparação dos modelos treinados (conjunto de teste)")
    fig.tight_layout(); fig.savefig(RES / "comparacao_modelos.png", dpi=160); plt.close(fig)
    return linhas


def curvas_treino() -> None:
    """mAP de validação por época e perda de caixa (treino) de cada experimento."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    cores = ["#2a6fdb", "#f5a623", "#2f9e5b", "#d0021b"]
    for cor, f in zip(cores, sorted(RES.glob("E*/results.csv"))):
        with open(f, encoding="utf-8") as fh:
            linhas = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(fh)]
        ep = [int(float(r["epoch"])) for r in linhas]
        a1.plot(ep, [float(r["metrics/mAP50(B)"]) for r in linhas], color=cor, label=f.parent.name)
        a1.plot(ep, [float(r["metrics/mAP50-95(B)"]) for r in linhas], color=cor, ls="--", lw=1)
        a2.plot(ep, [float(r["train/box_loss"]) for r in linhas], color=cor, label=f.parent.name)
    a1.set_title("Validação: mAP@0,5 (contínua) e mAP@0,5:0,95 (tracejada)"); a1.set_xlabel("época")
    a2.set_title("Treino: perda de regressão da caixa"); a2.set_xlabel("época")
    for a in (a1, a2):
        a.spines[["top", "right"]].set_visible(False); a.legend(frameon=False)
    fig.tight_layout(); fig.savefig(RES / "curvas_treino.png", dpi=160); plt.close(fig)


def escolher(linhas: list[dict]) -> dict:
    melhor = max(l["val_mAP50_95"] for l in linhas)
    candidatos = [l for l in linhas if l["val_mAP50_95"] >= melhor - TOLERANCIA_MAP]
    return min(candidatos, key=lambda l: l["inferencia_ms_gpu"])


def calibrar_severidade(raiz_ds: Path) -> dict:
    indices = []
    for lbl in (raiz_ds / "labels" / "train").glob("*.txt"):
        img = raiz_ds / "images" / "train" / (lbl.stem + ".jpg")
        im = cv2.imread(str(img))
        if im is None:
            continue
        h, w = im.shape[:2]
        for x1, y1, x2, y2 in ler_rotulos(lbl, w, h):
            indices.append(sev.indice_severidade(x1, y1, x2, y2, w, h))
    indices = np.array(indices)
    t1, t2 = np.quantile(indices, [1 / 3, 2 / 3])
    cal = {"metodo": "tercis do índice de severidade nas caixas de treino", "n_caixas": int(len(indices)),
           "horizonte": sev.HORIZONTE, "limiar_media": round(float(t1), 5), "limiar_alta": round(float(t2), 5)}
    (RAIZ / "configs" / "severidade.json").write_text(json.dumps(cal, indent=2, ensure_ascii=False), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(np.log10(indices + 1e-6), bins=60, color="#2a6fdb", alpha=0.85)
    for t, nome in ((t1, "média"), (t2, "alta")):
        ax.axvline(np.log10(t), color="#d0021b", ls="--"); ax.text(np.log10(t), ax.get_ylim()[1] * 0.92, f" {nome}", color="#d0021b")
    ax.set_xlabel("log10(índice de severidade)"); ax.set_ylabel("nº de buracos (treino)")
    ax.set_title("Calibração da severidade: tercis no conjunto de treino")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(RES / "calibracao_severidade.png", dpi=160); plt.close(fig)
    return cal


def analisar_teste(modelo: YOLO, raiz_ds: Path, limiar: float, manifest: dict[str, str]) -> dict:
    imgs = sorted((raiz_ds / "images" / "test").glob("*.jpg"))
    brilho, regs = {}, []
    for p in imgs:
        im = cv2.imread(str(p))
        h, w = im.shape[:2]
        brilho[p.name] = float(cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[..., 2].mean())
        r = modelo.predict(im, conf=limiar, imgsz=640, verbose=False)[0]
        pred = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        conf = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)
        gt = ler_rotulos(raiz_ds / "labels" / "test" / (p.stem + ".txt"), w, h)
        tp, usado = casar(pred, conf, gt)
        regs.append({"arquivo": p.name, "pais": manifest.get(p.name, "?"), "gt": gt, "pred": pred, "conf": conf,
                     "tp": tp, "usado": usado, "caminho": p})
    t1, t2 = np.quantile(list(brilho.values()), [1 / 3, 2 / 3])
    faixa_brilho = lambda b: "escura" if b < t1 else ("média" if b < t2 else "clara")  # noqa: E731

    grupos = defaultdict(lambda: [0, 0, 0])
    tam_tot, tam_ok = defaultdict(int), defaultdict(int)
    for r in regs:
        tp, fp, fn = int(r["tp"].sum()), int((~r["tp"]).sum()), int((~r["usado"]).sum())
        for g in ("geral", f"país: {r['pais']}", f"iluminação: {faixa_brilho(brilho[r['arquivo']])}"):
            grupos[g][0] += tp; grupos[g][1] += fp; grupos[g][2] += fn
        for (x1, y1, x2, y2), ok in zip(r["gt"], r["usado"]):
            k = tam_coco((x2 - x1) * (y2 - y1)); tam_tot[k] += 1; tam_ok[k] += int(ok)
    analise = {"limiar": limiar, "limiares_brilho_V": [round(float(t1), 1), round(float(t2), 1)],
               "grupos": {g: prf(*v) for g, v in grupos.items()},
               "revocacao_por_tamanho": {k: {"buracos": tam_tot[k], "detectados": tam_ok[k],
                                             "revocacao": round(tam_ok[k] / tam_tot[k], 4)} for k in tam_tot}}

    # gráfico de vieses
    gs = [g for g in analise["grupos"] if g != "geral"]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    y = np.arange(len(gs)); hh = 0.38
    ax.barh(y - hh / 2, [analise["grupos"][g]["revocacao"] for g in gs], hh, label="revocação", color="#2a6fdb")
    ax.barh(y + hh / 2, [analise["grupos"][g]["precisao"] for g in gs], hh, label="precisão", color="#f5a623")
    ax.set_yticks(y, gs); ax.set_xlim(0, 1); ax.invert_yaxis(); ax.legend(frameon=False, loc="lower right")
    ax.axvline(analise["grupos"]["geral"]["f1"], color="#555", ls=":", lw=1)
    ax.set_title(f"Desempenho por subgrupo no teste (limiar {limiar:.2f})")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(RES / "vieses_subgrupos.png", dpi=160); plt.close(fig)

    # mosaicos: acertos, falsos positivos, falsos negativos
    rng = random.Random(7)
    acertos = [r for r in regs if r["tp"].sum() >= 1 and (~r["tp"]).sum() == 0 and r["usado"].all()]
    fps = sorted([r for r in regs if (~r["tp"]).sum() > 0], key=lambda r: -float(r["conf"][~r["tp"]].max()))
    fns = sorted([r for r in regs if (~r["usado"]).sum() > 0], key=lambda r: -int((~r["usado"]).sum()))
    sel = lambda lst, n: rng.sample(lst, min(n, len(lst)))  # noqa: E731
    leg = "verde=real detectado | azul=predicao correta | vermelho=falso positivo | magenta=buraco perdido"
    for nome, lst in (("acertos", sel(acertos, 12)), ("falsos_positivos", fps[:8]), ("falsos_negativos", fns[:8])):
        mosaico([desenhar_caso(cv2.imread(str(r["caminho"])), r["gt"], r["pred"], r["conf"], r["tp"], r["usado"])
                 for r in lst], f"{nome.replace('_', ' ')} - {leg}", RES / f"exemplos_{nome}.jpg")
    return analise


def checar_vazamento(raiz_ds: Path, manifest: dict[str, str]) -> dict:
    """Quase-duplicatas teste x treino por hash perceptual (dHash de 256 bits): distância de Hamming mínima."""
    def dhash(p: Path) -> np.ndarray:
        im = cv2.resize(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), (17, 16), interpolation=cv2.INTER_AREA)
        return np.packbits((im[:, 1:] > im[:, :-1]).ravel())
    hs = {s: defaultdict(list) for s in ("train", "test")}
    for s in hs:
        for p in (raiz_ds / "images" / s).glob("*.jpg"):
            hs[s][manifest.get(p.name, "?")].append(dhash(p))
    out = {}
    for pais, testes in hs["test"].items():
        treino = np.array(hs["train"].get(pais, []))
        if not len(treino):
            continue
        d = np.array([int(np.unpackbits(np.bitwise_xor(treino, h), axis=1).sum(1).min()) for h in testes])
        out[pais] = {"imagens_teste": len(testes), "distancia_minima_mediana": float(np.median(d)),
                     "quase_duplicatas_pct": round(float(np.mean(d <= 20)) * 100, 2)}
    return out


def ap_por_pais(modelo: YOLO, raiz_ds: Path, manifest: dict[str, str]) -> dict:
    dir_sub = RES / "_subconjuntos"; dir_sub.mkdir(exist_ok=True)
    por_pais = defaultdict(list)
    for p in sorted((raiz_ds / "images" / "test").glob("*.jpg")):
        por_pais[manifest.get(p.name, "?")].append(p.resolve().as_posix())
    out = {}
    for pais, lista in por_pais.items():
        (dir_sub / f"test_{pais}.txt").write_text("\n".join(lista) + "\n", encoding="utf-8")
        y = {"path": raiz_ds.as_posix(), "train": "images/train", "val": (dir_sub / f"test_{pais}.txt").as_posix(),
             "names": {0: "buraco"}}
        arq = dir_sub / f"data_{pais}.yaml"; arq.write_text(yaml.safe_dump(y, allow_unicode=True), encoding="utf-8")
        m = modelo.val(data=str(arq), split="val", imgsz=640, batch=16, plots=False, verbose=False,
                       project=str(RAIZ / "runs"), name=f"_sub_{pais}", exist_ok=True)
        out[pais] = {"imagens": len(lista), "mAP50": round(float(m.box.map50), 4), "mAP50_95": round(float(m.box.map), 4)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    args = ap.parse_args()
    raiz_ds = args.data.resolve().parent
    manifest = {}
    with open(raiz_ds / "manifest.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            manifest[row["arquivo"]] = row["pais"]

    import shutil
    shutil.copy2(raiz_ds / "stats.json", RES / "dataset_stats.json")
    linhas = comparar_experimentos()
    curvas_treino()
    esc = escolher(linhas)
    print("Comparação:", json.dumps([{k: v for k, v in l.items() if not k.startswith("_")} for l in linhas],
                                     indent=1, ensure_ascii=False))
    print(f"Modelo escolhido: {esc['experimento']} (limiar {esc['limiar']})")

    cal = calibrar_severidade(raiz_ds)
    print("Severidade:", cal)

    pesos = RAIZ / "modelos" / f"vigiavia_{esc['experimento']}.pt"
    modelo = YOLO(str(pesos))
    analise = analisar_teste(modelo, raiz_ds, esc["limiar"], manifest)
    analise["ap_por_pais"] = ap_por_pais(modelo, raiz_ds, manifest)
    analise["vazamento_dhash"] = checar_vazamento(raiz_ds, manifest)
    (RES / "analise_teste.json").write_text(json.dumps(analise, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Análise do teste:", json.dumps(analise, indent=1, ensure_ascii=False))

    teste = esc["_metricas"]["test"]
    op = {"experimento": esc["experimento"], "pesos": "modelos/vigiavia_buracos.pt",
          "onnx": "/modelos/vigiavia_buracos.onnx", "conf": esc["limiar"],
          "regra_escolha": f"menor latência entre modelos a até {TOLERANCIA_MAP * 100:.1f} p.p. do melhor mAP50-95 de validação",
          "metricas_teste": {"mAP50": teste["mAP50"], "mAP50_95": teste["mAP50_95"], "precisao": teste["precisao"],
                             "revocacao": teste["revocacao"],
                             "precisao_no_limiar": analise["grupos"]["geral"]["precisao"],
                             "revocacao_no_limiar": analise["grupos"]["geral"]["revocacao"],
                             "f1_no_limiar": analise["grupos"]["geral"]["f1"]}}
    import shutil
    shutil.copy2(pesos, RAIZ / "modelos" / "vigiavia_buracos.pt")
    (RAIZ / "configs" / "operacao.json").write_text(json.dumps(op, indent=2, ensure_ascii=False), encoding="utf-8")
    print("configs/operacao.json:", json.dumps(op, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
