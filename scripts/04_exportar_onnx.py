"""
04_exportar_onnx.py — Exporta o modelo de produção para ONNX (CPU, celular via
onnxruntime-web, câmeras embarcadas) e confere que a precisão se mantém após a exportação.

Uso:
    python scripts/04_exportar_onnx.py --data datasets/vigiavia-rdd2022/data.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "False")   # nada de pip install em tempo de execução
from ultralytics import YOLO  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--pesos", type=Path, default=RAIZ / "modelos" / "vigiavia_buracos.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    destino = RAIZ / "modelos" / "vigiavia_buracos.onnx"
    variantes = {}
    # Duas variantes: "fim_a_fim" (nms=False: cabeça um-para-um, saída [1,300,6], sem NMS — mais simples no
    # navegador) e "com_nms" (cabeça um-para-muitos, saída [1,5,N], NMS fora do grafo). Escolhemos pela precisão.
    for nome, nms in (("fim_a_fim", False), ("com_nms", None)):
        arq = Path(YOLO(str(args.pesos)).export(format="onnx", imgsz=args.imgsz, simplify=True, dynamic=False,
                                                opset=17, nms=nms, device="cpu"))
        alvo = RAIZ / "modelos" / f"_vigiavia_buracos_{nome}.onnx"
        shutil.move(str(arq), alvo)
        variantes[nome] = alvo

    def validar(caminho: Path, dev, lote: int) -> dict:
        m = YOLO(str(caminho), task="detect").val(data=str(args.data), split="test", imgsz=args.imgsz, batch=lote,
                                                  device=dev, plots=False, verbose=False, project=str(RAIZ / "runs"),
                                                  name=f"_paridade_{caminho.stem}", exist_ok=True)
        return {"mAP50": round(float(m.box.map50), 4), "mAP50_95": round(float(m.box.map), 4),
                "ms_inferencia": round(float(m.speed["inference"]), 2)}

    res = {"pytorch_gpu": validar(args.pesos, 0, 16)}
    print("pytorch_gpu", res["pytorch_gpu"])
    for nome, caminho in variantes.items():
        res[f"onnx_cpu_{nome}"] = validar(caminho, "cpu", 1)
        print(f"onnx_cpu_{nome}", res[f"onnx_cpu_{nome}"])
    # prefere fim a fim (mais simples) se a perda de mAP@0,5:0,95 for < 0,5 p.p.; senão, a variante com NMS
    perda = res["onnx_cpu_com_nms"]["mAP50_95"] - res["onnx_cpu_fim_a_fim"]["mAP50_95"]
    escolhida = "fim_a_fim" if perda < 0.005 else "com_nms"
    shutil.copy2(variantes[escolhida], destino)
    for caminho in variantes.values():
        caminho.unlink(missing_ok=True)
    res["variante_escolhida"] = escolhida
    res["onnx_cpu"] = res[f"onnx_cpu_{escolhida}"]
    print(f"Variante escolhida: {escolhida} -> {destino} ({destino.stat().st_size / 1e6:.1f} MB)")
    (RAIZ / "resultados" / "paridade_onnx.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
