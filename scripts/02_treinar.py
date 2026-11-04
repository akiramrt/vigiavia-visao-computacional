"""
02_treinar.py — Treino customizado (transfer learning) dos detectores de buracos do VigiaVia.

Parte de pesos pré-treinados no COCO (Ultralytics) e faz fine-tuning no dataset VigiaVia
(RDD2022, classe "buraco"). Ao final de cada experimento:
  * valida no split de validação e no de TESTE (nunca visto no treino);
  * calcula o limiar de confiança que maximiza o F1 na validação;
  * salva métricas em resultados/<experimento>/metricas.json e copia o best.pt para modelos/.

Uso:
    python scripts/02_treinar.py --data datasets/vigiavia-rdd2022/data.yaml --exp E1_yolo26n
    python scripts/02_treinar.py --data ... --exp E1_yolo26n E2_yolo26s E3_yolo11s
    python scripts/02_treinar.py --data ... --exp E1_yolo26n --epochs 1 --fraction 0.05   # teste rápido
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parents[1]


def resumo_metricas(m) -> dict:
    """Extrai as métricas relevantes de um objeto DetMetrics do Ultralytics."""
    box = m.box
    out = {
        "precisao": float(box.mp),
        "revocacao": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "velocidade_ms_por_imagem": {k: round(float(v), 3) for k, v in m.speed.items()},
    }
    try:  # limiar que maximiza F1 (média das classes; aqui só há uma)
        px = np.asarray(box.px)
        f1 = np.asarray(box.f1_curve).mean(0)
        i = int(f1.argmax())
        out["melhor_f1"] = float(f1[i])
        out["limiar_conf_melhor_f1"] = float(px[i])
        out["precisao_no_limiar"] = float(np.asarray(box.p_curve).mean(0)[i])
        out["revocacao_no_limiar"] = float(np.asarray(box.r_curve).mean(0)[i])
    except Exception as exc:  # noqa: BLE001 — métricas extras são opcionais
        out["aviso_f1"] = str(exc)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="data.yaml gerado pelo script 01")
    ap.add_argument("--config", type=Path, default=RAIZ / "configs" / "treino.yaml")
    ap.add_argument("--exp", nargs="+", default=["E1_yolo26n"], help="experimentos do treino.yaml")
    ap.add_argument("--project", type=Path, default=RAIZ / "runs")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--fraction", type=float, default=1.0, help="fração do treino (testes rápidos)")
    ap.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for exp in args.exp:
        params = dict(cfg["comum"])
        params.update(cfg["experimentos"][exp])
        if args.epochs:
            params["epochs"] = args.epochs
        if args.batch:
            params["batch"] = args.batch
        pesos = params.pop("model")
        print(f"\n===== {exp}: {pesos} | device={args.device} | epochs={params['epochs']} =====")

        modelo = YOLO(pesos)
        t0 = time.time()
        modelo.train(data=str(args.data), project=str(args.project), name=exp, exist_ok=True,
                     device=args.device, fraction=args.fraction, **params)
        duracao_min = (time.time() - t0) / 60

        dir_run = args.project / exp
        best = dir_run / "weights" / "best.pt"
        melhor = YOLO(str(best))
        res = {"experimento": exp, "pesos_iniciais": pesos, "duracao_treino_min": round(duracao_min, 1),
               "parametros": params, "device": torch.cuda.get_device_name(0) if args.device != "cpu" else "cpu",
               "n_parametros": sum(p.numel() for p in melhor.model.parameters())}
        for split in ("val", "test"):
            m = melhor.val(data=str(args.data), split=split, imgsz=params["imgsz"], batch=16,
                           device=args.device, project=str(args.project), name=f"{exp}_{split}",
                           exist_ok=True, plots=True, verbose=False)
            res[split] = resumo_metricas(m)
            print(f"[{exp}] {split}: {json.dumps(res[split], ensure_ascii=False)}")

        dir_res = RAIZ / "resultados" / exp
        dir_res.mkdir(parents=True, exist_ok=True)
        (dir_res / "metricas.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        for nome in ("results.png", "results.csv", "args.yaml", "labels.jpg"):
            if (dir_run / nome).exists():
                shutil.copy2(dir_run / nome, dir_res / nome)
        dir_teste = args.project / f"{exp}_test"
        for img in dir_teste.glob("*.png"):
            shutil.copy2(img, dir_res / f"teste_{img.name}")
        (RAIZ / "modelos").mkdir(exist_ok=True)
        shutil.copy2(best, RAIZ / "modelos" / f"vigiavia_{exp}.pt")
        print(f"[{exp}] concluído em {duracao_min:.1f} min -> modelos/vigiavia_{exp}.pt")


if __name__ == "__main__":
    main()
