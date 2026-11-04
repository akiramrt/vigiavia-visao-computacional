"""
API do componente de Visão Computacional do VigiaVia (contrato com o MVP).

Endpoints
  GET   /health                      estado do serviço e do modelo
  POST  /v1/deteccoes                imagem (multipart) -> buracos detectados (JSON). A imagem
                                     é processada em memória e descartada (LGPD).
  POST  /v1/relatos                  cria/atualiza um relato georreferenciado. Relatos a menos de
                                     RAIO_DEDUP_M metros de um relato aberto viram CONFIRMAÇÕES do
                                     mesmo buraco (evita chamados duplicados).
  GET   /v1/relatos                  fila de zeladoria ordenada por prioridade
  PATCH /v1/relatos/{id}             validação humana (confirmado / descartado / resolvido)
  GET   /v1/relatos/{id}/evidencia   recorte da via usado como evidência
  GET   /                            interface web (cidadão + zeladoria)

Execução:  uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import base64
import math
import os
import sqlite3
import sys
import time
import uuid
from contextlib import asynccontextmanager, closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from vigiavia import __version__, config  # noqa: E402
from vigiavia.detector import DetectorBuracos  # noqa: E402
from vigiavia.severidade import PESO_SEVERIDADE  # noqa: E402

BANCO = Path(os.getenv("VIGIAVIA_DB", RAIZ / "dados" / "vigiavia.db"))
RAIO_DEDUP_M = float(os.getenv("VIGIAVIA_RAIO_DEDUP_M", "15"))
TAM_MAX_IMAGEM = 8 * 1024 * 1024
ESTADO: dict = {}


def _db() -> sqlite3.Connection:
    con = sqlite3.connect(BANCO)
    con.row_factory = sqlite3.Row
    return con


def _iniciar_banco() -> None:
    BANCO.parent.mkdir(parents=True, exist_ok=True)
    with closing(_db()) as con, con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS relatos (
                id TEXT PRIMARY KEY, criado_em TEXT, atualizado_em TEXT,
                lat REAL, lon REAL, severidade TEXT, confianca_max REAL,
                confirmacoes INTEGER, origem TEXT, status TEXT, modelo TEXT,
                evidencia BLOB, observacao TEXT)""")


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    _iniciar_banco()
    pesos = Path(os.getenv("VIGIAVIA_PESOS", config.pesos_padrao()))
    ESTADO["detector"] = DetectorBuracos(pesos, conf=config.conf_padrao(),
                                         device=os.getenv("VIGIAVIA_DEVICE") or None)
    ESTADO["detector"].detectar(np.zeros((640, 640, 3), np.uint8))  # aquece o modelo
    yield


app = FastAPI(title="VigiaVia — Visão Computacional", version=__version__, lifespan=ciclo_de_vida,
              description="Detecção de buracos em vias públicas (YOLO) integrada à fila de zeladoria.")


# ----------------------------------------------------------------------------- modelos de dados
class CaixaDetectada(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confianca: float
    severidade: Literal["baixa", "media", "alta"]
    indice_severidade: float
    area_relativa: float
    na_trajetoria: bool
    classe: str = "buraco"


class RespostaDeteccao(BaseModel):
    modelo: str
    limiar_confianca: float
    largura: int
    altura: int
    tempo_inferencia_ms: float
    quantidade: int
    deteccoes: list[CaixaDetectada]


class NovoRelato(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    severidade: Literal["baixa", "media", "alta"]
    confianca_max: float = Field(ge=0, le=1)
    origem: Literal["cidadao", "camera_veicular", "robo"] = "cidadao"
    modelo: str | None = None
    evidencia_jpg_base64: str | None = Field(default=None, description="apenas o recorte da via (≤ 200 KB)")


class Validacao(BaseModel):
    status: Literal["confirmado", "descartado", "resolvido"]
    observacao: str | None = Field(default=None, max_length=280)


# ----------------------------------------------------------------------------- utilidades
def distancia_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distância de Haversine em metros."""
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def prioridade(severidade: str, confirmacoes: int) -> float:
    """Severidade (1–3) × reforço logarítmico pelo nº de confirmações independentes."""
    return round(PESO_SEVERIDADE[severidade] * (1 + math.log2(max(1, confirmacoes))), 3)


def _relato_publico(row: sqlite3.Row) -> dict:
    d = {k: row[k] for k in row.keys() if k != "evidencia"}
    d["tem_evidencia"] = row["evidencia"] is not None
    d["prioridade"] = prioridade(row["severidade"], row["confirmacoes"])
    return d


def _agora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ----------------------------------------------------------------------------- endpoints
@app.get("/health")
def health() -> dict:
    det: DetectorBuracos = ESTADO["detector"]
    return {"status": "ok", "versao": __version__, "modelo": det.nome, "limiar_confianca": det.conf,
            "raio_deduplicacao_m": RAIO_DEDUP_M}


@app.get("/v1/modelo")
def modelo() -> dict:
    """Ficha do modelo em produção: métricas de teste, limiar, severidade e zona de trajetória."""
    from vigiavia import severidade
    from vigiavia.detector import ZONA_TRAJETORIA
    det: DetectorBuracos = ESTADO["detector"]
    op = config.operacao()
    return {"modelo": det.nome, "limiar_confianca": det.conf, "imgsz": det.imgsz,
            "onnx": op.get("onnx"), "metricas_teste": op.get("metricas_teste"),
            "severidade": {"horizonte": severidade.HORIZONTE, "limiar_media": severidade.LIMIARES[0],
                           "limiar_alta": severidade.LIMIARES[1]},
            "zona_trajetoria": ZONA_TRAJETORIA}


@app.post("/v1/deteccoes", response_model=RespostaDeteccao)
async def detectar(imagem: UploadFile = File(...),
                   conf: float | None = Query(default=None, ge=0.01, le=0.99)) -> RespostaDeteccao:
    dados = await imagem.read()
    if len(dados) > TAM_MAX_IMAGEM:
        raise HTTPException(413, "Imagem maior que 8 MB")
    img = cv2.imdecode(np.frombuffer(dados, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(422, "Arquivo não é uma imagem válida")
    det: DetectorBuracos = ESTADO["detector"]
    altura, largura = img.shape[:2]
    t0 = time.perf_counter()
    caixas = det.detectar(img, conf=conf)
    ms = (time.perf_counter() - t0) * 1000
    del img, dados                     # nada é persistido (LGPD: minimização)
    return RespostaDeteccao(modelo=det.nome, limiar_confianca=conf or det.conf, largura=largura,
                            altura=altura, tempo_inferencia_ms=round(ms, 1), quantidade=len(caixas),
                            deteccoes=[CaixaDetectada(**{k: v for k, v in c.para_dict().items()
                                                         if k != "track_id"}) for c in caixas])


@app.post("/v1/relatos", status_code=201)
def criar_relato(relato: NovoRelato) -> dict:
    evidencia = None
    if relato.evidencia_jpg_base64:
        try:
            evidencia = base64.b64decode(relato.evidencia_jpg_base64, validate=True)
        except ValueError as exc:
            raise HTTPException(422, "evidencia_jpg_base64 inválida") from exc
        if len(evidencia) > 200_000:
            raise HTTPException(413, "Evidência maior que 200 KB — envie só o recorte da via")
    agora = _agora()
    with closing(_db()) as con, con:
        abertos = con.execute("SELECT * FROM relatos WHERE status IN ('pendente', 'confirmado')").fetchall()
        proximos = sorted(((distancia_m(relato.lat, relato.lon, r["lat"], r["lon"]), r) for r in abertos),
                          key=lambda t: t[0])
        if proximos and proximos[0][0] <= RAIO_DEDUP_M:
            dist, r = proximos[0]
            sev = max(r["severidade"], relato.severidade, key=PESO_SEVERIDADE.get)
            con.execute("""UPDATE relatos SET confirmacoes = confirmacoes + 1, atualizado_em = ?,
                           severidade = ?, confianca_max = MAX(confianca_max, ?),
                           evidencia = COALESCE(evidencia, ?) WHERE id = ?""",
                        (agora, sev, relato.confianca_max, evidencia, r["id"]))
            novo = con.execute("SELECT * FROM relatos WHERE id = ?", (r["id"],)).fetchone()
            return {"mensagem": f"Buraco já registrado a {dist:.0f} m — confirmação adicionada",
                    "duplicado": True, "relato": _relato_publico(novo)}
        rid = str(uuid.uuid4())
        con.execute("""INSERT INTO relatos VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, 'pendente', ?, ?, NULL)""",
                    (rid, agora, agora, relato.lat, relato.lon, relato.severidade, relato.confianca_max,
                     relato.origem, relato.modelo, evidencia))
        novo = con.execute("SELECT * FROM relatos WHERE id = ?", (rid,)).fetchone()
    return {"mensagem": "Relato criado e enviado para validação da zeladoria", "duplicado": False,
            "relato": _relato_publico(novo)}


@app.get("/v1/relatos")
def listar_relatos(status: str | None = None, limite: int = Query(default=200, le=1000)) -> dict:
    with closing(_db()) as con:
        if status:
            linhas = con.execute("SELECT * FROM relatos WHERE status = ?", (status,)).fetchall()
        else:
            linhas = con.execute("SELECT * FROM relatos").fetchall()
    itens = sorted((_relato_publico(r) for r in linhas), key=lambda d: (-d["prioridade"], d["criado_em"]))
    return {"total": len(itens), "relatos": itens[:limite]}


@app.patch("/v1/relatos/{rid}")
def validar_relato(rid: str, validacao: Validacao) -> dict:
    """Pessoa no circuito: só a zeladoria transforma um relato em ordem de serviço."""
    with closing(_db()) as con, con:
        cur = con.execute("UPDATE relatos SET status = ?, observacao = ?, atualizado_em = ? WHERE id = ?",
                          (validacao.status, validacao.observacao, _agora(), rid))
        if cur.rowcount == 0:
            raise HTTPException(404, "Relato não encontrado")
        r = con.execute("SELECT * FROM relatos WHERE id = ?", (rid,)).fetchone()
    return _relato_publico(r)


@app.get("/v1/relatos/{rid}/evidencia")
def evidencia(rid: str) -> Response:
    with closing(_db()) as con:
        r = con.execute("SELECT evidencia FROM relatos WHERE id = ?", (rid,)).fetchone()
    if r is None or r["evidencia"] is None:
        raise HTTPException(404, "Sem evidência para este relato")
    return Response(content=r["evidencia"], media_type="image/jpeg")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(RAIZ / "api" / "static" / "index.html")


app.mount("/static", StaticFiles(directory=RAIZ / "api" / "static"), name="static")
app.mount("/modelos", StaticFiles(directory=RAIZ / "modelos"), name="modelos")
