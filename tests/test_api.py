"""Testes de contrato da API (usam o modelo real em modelos/ e um banco temporário)."""
import base64
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

if not (RAIZ / "modelos" / "vigiavia_buracos.pt").exists():
    pytest.skip("modelo de produção ainda não treinado", allow_module_level=True)


@pytest.fixture(scope="module")
def cliente(tmp_path_factory):
    os.environ["VIGIAVIA_DB"] = str(tmp_path_factory.mktemp("db") / "teste.db")
    os.environ.setdefault("VIGIAVIA_DEVICE", "cpu")
    from fastapi.testclient import TestClient
    import api.main as m
    m.BANCO = Path(os.environ["VIGIAVIA_DB"])
    with TestClient(m.app) as c:
        yield c


def _jpg(img) -> bytes:
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_health(cliente):
    r = cliente.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_deteccao_em_imagem_sem_buraco(cliente):
    cinza = np.full((480, 640, 3), 128, np.uint8)
    r = cliente.post("/v1/deteccoes", files={"imagem": ("x.jpg", _jpg(cinza), "image/jpeg")})
    assert r.status_code == 200
    corpo = r.json()
    assert corpo["largura"] == 640 and corpo["altura"] == 480
    assert corpo["quantidade"] == len(corpo["deteccoes"]) == 0


def test_rejeita_arquivo_que_nao_e_imagem(cliente):
    r = cliente.post("/v1/deteccoes", files={"imagem": ("x.txt", b"nao sou imagem", "text/plain")})
    assert r.status_code == 422


def test_relatos_proximos_viram_confirmacao(cliente):
    evid = base64.b64encode(_jpg(np.zeros((60, 80, 3), np.uint8))).decode()
    base = {"lat": -23.561414, "lon": -46.655881, "severidade": "media", "confianca_max": 0.8,
            "evidencia_jpg_base64": evid}
    r1 = cliente.post("/v1/relatos", json=base)
    assert r1.status_code == 201 and r1.json()["duplicado"] is False
    # ~5 m de distância -> mesmo buraco
    r2 = cliente.post("/v1/relatos", json={**base, "lat": -23.561460, "severidade": "alta"})
    assert r2.json()["duplicado"] is True
    rel = r2.json()["relato"]
    assert rel["confirmacoes"] == 2 and rel["severidade"] == "alta"
    # ~1 km de distância -> outro buraco
    r3 = cliente.post("/v1/relatos", json={**base, "lat": -23.570})
    assert r3.json()["duplicado"] is False
    fila = cliente.get("/v1/relatos").json()["relatos"]
    assert fila[0]["id"] == rel["id"]          # maior prioridade primeiro


def test_validacao_humana(cliente):
    rid = cliente.get("/v1/relatos").json()["relatos"][0]["id"]
    r = cliente.patch(f"/v1/relatos/{rid}", json={"status": "confirmado", "observacao": "vistoria ok"})
    assert r.status_code == 200 and r.json()["status"] == "confirmado"
    assert cliente.get(f"/v1/relatos/{rid}/evidencia").status_code == 200
    assert cliente.patch("/v1/relatos/inexistente", json={"status": "resolvido"}).status_code == 404
