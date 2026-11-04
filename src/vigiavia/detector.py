"""
Detector de buracos do VigiaVia (wrapper sobre o Ultralytics YOLO).

Aceita pesos PyTorch (.pt) ou ONNX (.onnx). Para cada caixa calcula a severidade (proxy)
e se o buraco está na trajetória do veículo (zona trapezoidal à frente da câmera), que é
o que dispara o alerta de realidade aumentada no modo tempo real.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from . import severidade as sev

# Trapézio da trajetória à frente do veículo, em coordenadas normalizadas (x, y).
ZONA_TRAJETORIA = ((0.12, 1.00), (0.40, 0.58), (0.60, 0.58), (0.88, 1.00))
RASTREADOR = Path(__file__).resolve().parents[2] / "configs" / "rastreador.yaml"


def sobreposicao(a: tuple, b: tuple) -> float:
    """max(IoU, interseção / menor área): alta também quando uma caixa está DENTRO da outra."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return max(inter / (aa + ab - inter), inter / max(1e-9, min(aa, ab)))


def suprimir_aninhadas(deteccoes: list, limiar: float = 0.7) -> list:
    """O NMS padrão (IoU) não remove uma caixa pequena dentro de uma grande sobre o MESMO buraco
    (IoU baixo). Mantemos a de maior confiança quando a sobreposição por contenção passa do limiar."""
    mantidas = []
    for d in sorted(deteccoes, key=lambda x: -x.confianca):
        if all(sobreposicao(d.caixa, m.caixa) < limiar for m in mantidas):
            mantidas.append(d)
    return mantidas


@dataclass
class Deteccao:
    x1: float
    y1: float
    x2: float
    y2: float
    confianca: float
    severidade: str
    indice_severidade: float
    area_relativa: float
    na_trajetoria: bool
    track_id: int | None = None
    classe: str = "buraco"

    @property
    def caixa(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    def para_dict(self) -> dict:
        d = asdict(self)
        for k in ("x1", "y1", "x2", "y2"):
            d[k] = round(d[k], 1)
        d["confianca"] = round(d["confianca"], 4)
        d["indice_severidade"] = round(d["indice_severidade"], 5)
        d["area_relativa"] = round(d["area_relativa"], 5)
        return d


def zona_em_pixels(largura: int, altura: int, zona=ZONA_TRAJETORIA) -> np.ndarray:
    return np.array([(x * largura, y * altura) for x, y in zona], dtype=np.int32)


class DetectorBuracos:
    def __init__(self, pesos: str | Path, conf: float = 0.25, iou: float = 0.5, imgsz: int = 640,
                 device: str | None = None, zona=ZONA_TRAJETORIA):
        self.pesos = Path(pesos)
        if not self.pesos.exists():
            raise FileNotFoundError(f"Pesos não encontrados: {self.pesos}")
        self.modelo = YOLO(str(self.pesos), task="detect")
        if device is None and self.pesos.suffix == ".onnx":
            device = "cpu"          # ONNX Runtime (CPU); evita a tentativa de usar CUDA/instalar onnxruntime-gpu
        self.conf, self.iou, self.imgsz, self.device = conf, iou, imgsz, device
        self.zona = zona
        self._zona_px: dict[tuple[int, int], np.ndarray] = {}

    @property
    def nome(self) -> str:
        return self.pesos.stem

    def _zona(self, w: int, h: int) -> np.ndarray:
        if (w, h) not in self._zona_px:
            self._zona_px[(w, h)] = zona_em_pixels(w, h, self.zona)
        return self._zona_px[(w, h)]

    def _converter(self, resultado, largura: int, altura: int) -> list[Deteccao]:
        caixas = resultado.boxes
        if caixas is None or len(caixas) == 0:
            return []
        xyxy = caixas.xyxy.cpu().numpy()
        confs = caixas.conf.cpu().numpy()
        ids = caixas.id.cpu().numpy().astype(int) if caixas.id is not None else [None] * len(confs)
        zona = self._zona(largura, altura)
        saida = []
        for (x1, y1, x2, y2), c, tid in zip(xyxy.astype(float), confs.astype(float), ids):
            idx = float(sev.indice_severidade(x1, y1, x2, y2, largura, altura))
            base = ((x1 + x2) / 2, y2)
            dentro = cv2.pointPolygonTest(zona, base, False) >= 0
            saida.append(Deteccao(x1, y1, x2, y2, c, sev.classificar(idx), idx,
                                  float(sev.area_relativa(x1, y1, x2, y2, largura, altura)),
                                  bool(dentro), None if tid is None else int(tid)))
        return suprimir_aninhadas(saida)

    def detectar(self, imagem: np.ndarray, conf: float | None = None) -> list[Deteccao]:
        """Detecção quadro a quadro (fotos enviadas pelo cidadão, API)."""
        h, w = imagem.shape[:2]
        r = self.modelo.predict(imagem, conf=conf or self.conf, iou=self.iou, imgsz=self.imgsz,
                                device=self.device, verbose=False)[0]
        return self._converter(r, w, h)

    def rastrear(self, quadro: np.ndarray, tracker: str | Path = RASTREADOR) -> list[Deteccao]:
        """Detecção + rastreamento multiobjeto (BoT-SORT com compensação de movimento da câmera).

        A associação em dois estágios (herdada do ByteTrack) usa também detecções de baixa confiança
        para manter trilhas existentes, por isso o limiar enviado ao rastreador é menor que o de
        confirmação de evento.
        """
        h, w = quadro.shape[:2]
        r = self.modelo.track(quadro, persist=True, tracker=str(tracker), conf=min(self.conf, 0.15),
                              iou=self.iou, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        return self._converter(r, w, h)

    def reiniciar_rastreamento(self) -> None:
        """Zera o estado do rastreador (ao trocar de vídeo)."""
        pred = getattr(self.modelo, "predictor", None)
        if pred is not None and getattr(pred, "trackers", None):
            for t in pred.trackers:
                t.reset()
