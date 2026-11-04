"""
Severidade (proxy) de um buraco a partir da caixa detectada.

Uma câmera monocular não mede tamanho físico sem calibração. Usamos um índice relativo:
a área da caixa (fração do quadro) corrigida pela perspectiva de uma via plana — objetos
mais próximos do horizonte aparecem menores, então a área é dividida pelo quadrado da
distância (normalizada) entre a base da caixa e a linha do horizonte.

    escala  = clamp((y2/H - HORIZONTE) / (1 - HORIZONTE), 0.15, 1)
    indice  = min(1, area_relativa / escala²)

Os limiares baixa/média/alta são os TERCIS do índice no conjunto de treino
(calculados por scripts/03_avaliar.py e salvos em configs/severidade.json). É uma
priorização relativa para a fila de zeladoria, sempre revisada por uma pessoa.
"""
from __future__ import annotations

import json
from pathlib import Path

HORIZONTE = 0.35
LIMIARES_PADRAO = (0.01, 0.03)
_ARQ_CALIBRACAO = Path(__file__).resolve().parents[2] / "configs" / "severidade.json"


def _carregar_limiares() -> tuple[float, float]:
    try:
        d = json.loads(_ARQ_CALIBRACAO.read_text(encoding="utf-8"))
        return float(d["limiar_media"]), float(d["limiar_alta"])
    except (OSError, KeyError, ValueError):
        return LIMIARES_PADRAO


LIMIARES = _carregar_limiares()


def area_relativa(x1: float, y1: float, x2: float, y2: float, largura: int, altura: int) -> float:
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) / float(largura * altura)


def indice_severidade(x1: float, y1: float, x2: float, y2: float, largura: int, altura: int,
                      horizonte: float = HORIZONTE) -> float:
    escala = (y2 / altura - horizonte) / (1.0 - horizonte)
    escala = min(1.0, max(0.15, escala))
    return min(1.0, area_relativa(x1, y1, x2, y2, largura, altura) / escala ** 2)


def classificar(indice: float, limiares: tuple[float, float] | None = None) -> str:
    lim_media, lim_alta = limiares or LIMIARES
    if indice >= lim_alta:
        return "alta"
    if indice >= lim_media:
        return "media"
    return "baixa"


PESO_SEVERIDADE = {"baixa": 1, "media": 2, "alta": 3}
