"""Configuração de operação (modelo escolhido e limiar), gerada pela etapa de avaliação."""
from __future__ import annotations

import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
ARQ_OPERACAO = RAIZ / "configs" / "operacao.json"
PESOS_PADRAO = RAIZ / "modelos" / "vigiavia_buracos.pt"
CONF_PADRAO = 0.35


def operacao() -> dict:
    """Lê configs/operacao.json (modelo escolhido + limiar que maximiza F1 na validação)."""
    try:
        return json.loads(ARQ_OPERACAO.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def pesos_padrao() -> Path:
    op = operacao()
    return RAIZ / op["pesos"] if "pesos" in op else PESOS_PADRAO


def conf_padrao() -> float:
    return float(operacao().get("conf", CONF_PADRAO))
