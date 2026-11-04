"""
HUD de realidade aumentada do VigiaVia: sobrepõe ao vídeo a zona de trajetória, as caixas
coloridas por severidade, o identificador de rastreamento, contadores e o alerta
"BURACO À FRENTE".

Desempenho: textos com acentuação são desenhados com Pillow (o OpenCV só tem ASCII), mas apenas
em pequenos recortes RGBA mesclados ao quadro — converter o quadro inteiro (1080p) para Pillow e de
volta custaria dezenas de milissegundos. As camadas translúcidas também só tocam a região afetada.
"""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .detector import Deteccao, zona_em_pixels

# Cores em BGR (OpenCV) e RGB (Pillow)
COR_SEV_BGR = {"baixa": (0, 215, 255), "media": (0, 140, 255), "alta": (48, 48, 235)}
COR_SEV_RGB = {k: (v[2], v[1], v[0]) for k, v in COR_SEV_BGR.items()}
ROTULO_SEV = {"baixa": "BAIXA", "media": "MÉDIA", "alta": "ALTA"}
TXT_ALERTA = "⚠  ATENÇÃO: BURACO À FRENTE"


@lru_cache(maxsize=16)
def fonte(tamanho: int, negrito: bool = False) -> ImageFont.FreeTypeFont:
    try:
        from matplotlib import font_manager
        caminho = font_manager.findfont(font_manager.FontProperties(
            family="DejaVu Sans", weight="bold" if negrito else "normal"))
        return ImageFont.truetype(caminho, tamanho)
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()


def medir(txt: str, f: ImageFont.FreeTypeFont, pad: int = 0) -> tuple[int, int]:
    l, t, r, b = f.getbbox(txt)
    return (r - l) + 2 * pad, (b - t) + 2 * pad


def escrever(img: np.ndarray, xy: tuple[int, int], txt: str, f: ImageFont.FreeTypeFont, cor_rgb,
             fundo_rgb=None, pad: int = 0) -> None:
    """Desenha `txt` com canto superior esquerdo em xy (in-place), mesclando um recorte RGBA pequeno."""
    l, t, r, b = f.getbbox(txt)
    w, h = (r - l) + 2 * pad, (b - t) + 2 * pad
    patch = Image.new("RGBA", (w, h), (*fundo_rgb, 255) if fundo_rgb else (0, 0, 0, 0))
    ImageDraw.Draw(patch).text((pad - l, pad - t), txt, font=f, fill=(*cor_rgb, 255))
    arr = np.asarray(patch)
    x, y = int(xy[0]), int(xy[1])
    hh, ww = img.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(ww, x + w), min(hh, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    sub = arr[y0 - y:y1 - y, x0 - x:x1 - x]
    alfa = sub[..., 3:4].astype(np.float32) / 255.0
    roi = img[y0:y1, x0:x1]
    roi[:] = (alfa * sub[..., 2::-1] + (1.0 - alfa) * roi).astype(np.uint8)


def _translucido(img: np.ndarray, pts_ou_ret, cor, alfa: float) -> None:
    hh, ww = img.shape[:2]
    if isinstance(pts_ou_ret, np.ndarray):
        x, y, w, h = cv2.boundingRect(pts_ou_ret)
        x0, y0, x1, y1 = max(0, x), max(0, y), min(ww, x + w), min(hh, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        roi = img[y0:y1, x0:x1]
        camada = roi.copy()                     # fora do polígono camada == roi, então a mistura não muda nada
        cv2.fillPoly(camada, [pts_ou_ret - np.array([x0, y0], dtype=pts_ou_ret.dtype)], cor)
        cv2.addWeighted(camada, alfa, roi, 1 - alfa, 0, dst=roi)
    else:
        x0, y0, x1, y1 = pts_ou_ret
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(ww, x1), min(hh, y1)
        if x1 <= x0 or y1 <= y0:
            return
        roi = img[y0:y1, x0:x1]
        camada = np.empty_like(roi)
        camada[:] = cor
        cv2.addWeighted(camada, alfa, roi, 1 - alfa, 0, dst=roi)


def desenhar(quadro: np.ndarray, deteccoes: list[Deteccao], conf_alerta: float, fps: float,
             n_unicos: int, nome_modelo: str, privacidade_ativa: bool = True,
             mostrar_zona: bool = True) -> np.ndarray:
    img = quadro.copy()
    h, w = img.shape[:2]
    u = max(1.0, w / 1280)                      # unidade de escala para espessuras/fontes
    alerta = any(d.na_trajetoria and d.confianca >= conf_alerta for d in deteccoes)
    f_lab, f_top, f_al, f_rod = fonte(int(17 * u), True), fonte(int(20 * u), True), fonte(int(34 * u), True), fonte(int(15 * u))
    barra = int(44 * u)

    # 1) Zona de trajetória (AR)
    if mostrar_zona:
        zona = zona_em_pixels(w, h)
        _translucido(img, zona, (40, 40, 220) if alerta else (60, 180, 60), 0.18)
        cv2.polylines(img, [zona], True, (40, 40, 220) if alerta else (90, 220, 90), max(1, int(2 * u)),
                      cv2.LINE_AA)

    # 2) Caixas e rótulos
    for d in deteccoes:
        x1, y1, x2, y2 = (int(v) for v in d.caixa)
        cv2.rectangle(img, (x1, y1), (x2, y2), COR_SEV_BGR[d.severidade], max(2, int(3 * u)), cv2.LINE_AA)
    for d in deteccoes:
        x1, y1 = int(d.caixa[0]), int(d.caixa[1])
        ident = f"#{d.track_id} " if d.track_id is not None else ""
        txt = f"{ident}buraco {d.confianca:.0%} · {ROTULO_SEV[d.severidade]}"
        pad = int(4 * u)
        _, lh = medir(txt, f_lab, pad)
        ty = y1 - lh - 2 if y1 - lh - 2 > barra else y1 + 2
        escrever(img, (x1, ty), txt, f_lab, (15, 15, 15), COR_SEV_RGB[d.severidade], pad)

    # 3) Barras de status
    _translucido(img, (0, 0, w, barra), (20, 20, 20), 0.65)
    _translucido(img, (0, h - int(30 * u), w, h), (20, 20, 20), 0.55)
    escrever(img, (int(14 * u), int(12 * u)), "VigiaVia · detecção de buracos em tempo real", f_top, (255, 255, 255))
    status = f"FPS {fps:4.1f}   |   buracos únicos: {n_unicos}   |   modelo: {nome_modelo}"
    sw, _ = medir(status, f_top)
    escrever(img, (w - sw - int(14 * u), int(12 * u)), status, f_top, (255, 230, 120))
    rodape = ("Privacidade: rostos borrados · apenas recortes da via e coordenadas são enviados"
              if privacidade_ativa else "Privacidade: borragem de rostos indisponível")
    escrever(img, (int(14 * u), h - int(23 * u)), rodape, f_rod, (220, 220, 220))

    # 4) Alerta de realidade aumentada
    if alerta:
        tw, th = medir(TXT_ALERTA, f_al)
        bw, bh = min(w - 20, tw + int(56 * u)), int(64 * u)
        bx, by = (w - bw) // 2, barra + int(16 * u)
        _translucido(img, (bx, by, bx + bw, by + bh), (30, 30, 210), 0.85)
        escrever(img, ((w - tw) // 2, by + (bh - th) // 2), TXT_ALERTA, f_al, (255, 255, 255))
    return img
