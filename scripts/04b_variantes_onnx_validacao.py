"""
04b_variantes_onnx_validacao.py — Confere na VALIDAÇÃO a escolha entre as duas variantes ONNX.

O 04_exportar_onnx.py escolheu a variante (fim a fim × com NMS) medindo no conjunto de teste, uma exceção à regra
"o teste nunca decide". Este script repete a comparação na validação, com as mesmas opções de exportação, numa pasta
temporária (o modelo de produção em modelos/ não é tocado).

Uso:
    python scripts/04b_variantes_onnx_validacao.py --data datasets/vigiavia-rdd2022/data.yaml
Saída: resultados/paridade_onnx_validacao.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "False")
from ultralytics import YOLO  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--pesos", type=Path, default=RAIZ / "modelos" / "vigiavia_buracos.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    res = {}
    with tempfile.TemporaryDirectory() as tmp:
        pt = Path(tmp) / "modelo.pt"
        shutil.copy2(args.pesos, pt)

        def validar(caminho: Path, dev, lote: int) -> dict:
            m = YOLO(str(caminho), task="detect").val(data=str(args.data), split="val", imgsz=args.imgsz, batch=lote,
                                                      device=dev, plots=False, verbose=False, project=tmp,
                                                      name=f"val_{caminho.stem}", exist_ok=True)
            return {"mAP50": round(float(m.box.map50), 4), "mAP50_95": round(float(m.box.map), 4)}

        res["pytorch_gpu"] = validar(pt, 0, 16)
        for nome, nms in (("fim_a_fim", False), ("com_nms", None)):
            arq = Path(YOLO(str(pt)).export(format="onnx", imgsz=args.imgsz, simplify=True, dynamic=False,
                                            opset=17, nms=nms, device="cpu"))
            alvo = Path(tmp) / f"{nome}.onnx"
            shutil.move(str(arq), alvo)
            res[f"onnx_cpu_{nome}"] = validar(alvo, "cpu", 1)
            print(nome, res[f"onnx_cpu_{nome}"], flush=True)
    perda = res["onnx_cpu_com_nms"]["mAP50_95"] - res["onnx_cpu_fim_a_fim"]["mAP50_95"]
    res["regra"] = "fim a fim se a perda de mAP@0,5:0,95 for < 0,5 p.p.; senão, com NMS"
    res["variante_escolhida_na_validacao"] = "fim_a_fim" if perda < 0.005 else "com_nms"
    (RAIZ / "resultados" / "paridade_onnx_validacao.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
