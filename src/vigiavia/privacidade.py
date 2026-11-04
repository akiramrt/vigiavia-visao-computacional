"""
Privacidade by design (LGPD, art. 6º — necessidade/minimização):

* O sistema NÃO guarda quadros inteiros: a evidência de um buraco é apenas um recorte
  da região da via ao redor da caixa detectada.
* Rostos que apareçam no recorte (ou num vídeo de saída, se o operador optar por gravá-lo)
  são borrados com o detector facial YuNet (OpenCV Zoo, licença MIT).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

YUNET_PADRAO = Path(__file__).resolve().parents[2] / "modelos" / "terceiros" / "face_detection_yunet_2023mar.onnx"


class Anonimizador:
    """Borra rostos com YuNet. Se o modelo não estiver disponível, não faz nada (e avisa)."""

    def __init__(self, modelo: Path | str = YUNET_PADRAO, limiar: float = 0.6, largura_det: int = 960,
                 intervalo: int = 1):
        self.ativo = Path(modelo).exists() and hasattr(cv2, "FaceDetectorYN")
        self._det = None
        self._tam = (0, 0)
        self.largura_det = largura_det          # detecta em resolução reduzida (mais rápido)
        self.intervalo = max(1, intervalo)      # em vídeo: detecta a cada N quadros e reaproveita as caixas
        self._n, self._cache = 0, []
        if self.ativo:
            self._det = cv2.FaceDetectorYN.create(str(modelo), "", (320, 320), limiar, 0.3, 5000)
        else:
            print(f"[privacidade] YuNet não encontrado em {modelo}; borragem de rostos desativada.")

    def _detectar(self, img: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = img.shape[:2]
        esc = min(1.0, self.largura_det / w)
        peq = cv2.resize(img, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA) if esc < 1 else img
        ph, pw = peq.shape[:2]
        if (pw, ph) != self._tam:
            self._det.setInputSize((pw, ph))
            self._tam = (pw, ph)
        _, rostos = self._det.detect(peq)
        caixas = []
        for f in (rostos if rostos is not None else []):
            x, y, fw, fh = (int(v / esc) for v in f[:4])
            # margem maior quando a caixa é reaproveitada em quadros seguintes (o rosto pode ter se movido)
            m = int((0.25 if self.intervalo == 1 else 0.45) * max(fw, fh))
            caixas.append((max(0, x - m), max(0, y - m), min(w, x + fw + m), min(h, y + fh + m)))
        return caixas

    def borrar_rostos(self, img: np.ndarray) -> tuple[np.ndarray, int]:
        if not self.ativo or img.size == 0:
            return img, 0
        if self._n % self.intervalo == 0:
            self._cache = self._detectar(img)
        self._n += 1
        if not self._cache:
            return img, 0
        saida = img.copy()
        for x1, y1, x2, y2 in self._cache:
            if x2 > x1 and y2 > y1:
                k = max(15, (max(x2 - x1, y2 - y1) // 3) | 1)
                saida[y1:y2, x1:x2] = cv2.GaussianBlur(saida[y1:y2, x1:x2], (k, k), 0)
        return saida, len(self._cache)


def recortar_evidencia(frame: np.ndarray, caixa: tuple[float, float, float, float],
                       margem: float = 0.25, lado_max: int = 480) -> np.ndarray:
    """Recorta apenas a região do buraco (+ margem) — minimização de dados."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = caixa
    mx, my = (x2 - x1) * margem, (y2 - y1) * margem
    xa, ya = int(max(0, x1 - mx)), int(max(0, y1 - my))
    xb, yb = int(min(w, x2 + mx)), int(min(h, y2 + my))
    rec = frame[ya:yb, xa:xb].copy()
    escala = lado_max / max(rec.shape[:2]) if rec.size else 1.0
    if escala < 1.0:
        rec = cv2.resize(rec, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
    return rec
