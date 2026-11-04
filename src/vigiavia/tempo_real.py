"""
VigiaVia em tempo real: webcam, arquivo de vídeo ou stream RTSP (câmera de ônibus, viatura,
caminhão de lixo ou celular no painel).

Pipeline por quadro:  captura -> YOLO (detecção) -> ByteTrack (rastreamento) ->
severidade + zona de trajetória -> HUD de realidade aumentada -> eventos.

Cada buraco ÚNICO (trilha confirmada em >= N quadros com confiança >= limiar) vira UM evento,
mesmo que apareça em centenas de quadros — é isso que evita chamados duplicados.
O evento leva só um recorte da via (rostos borrados) + coordenadas; opcionalmente é enviado
à API do VigiaVia (POST /v1/relatos), onde uma pessoa da zeladoria valida antes da ordem de serviço.

Exemplos:
    python -m vigiavia.tempo_real --fonte 0 --mostrar                       # webcam
    python -m vigiavia.tempo_real --fonte video.mp4 --saida demo/saida.mp4  # arquivo
    python -m vigiavia.tempo_real --fonte video.mp4 --api http://localhost:8000 --lat -23.55 --lon -46.64
"""
from __future__ import annotations

import argparse
import base64
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2

from . import config, hud
from .detector import DetectorBuracos, sobreposicao
from .privacidade import Anonimizador, recortar_evidencia
from .severidade import PESO_SEVERIDADE


class RegistroTrilhas:
    """Acompanha cada trilha (buraco) e decide quando ela vira um evento confirmado.

    Duas defesas contra contagem duplicada:
      1. confirmação: a trilha precisa durar >= min_quadros e atingir a confiança de evento;
      2. deduplicação espaço-temporal: se uma trilha nova se confirma no MESMO lugar da imagem em que
         estava uma trilha já reportada há pouco (janela de `janela` quadros), ela é tratada como o
         mesmo buraco que "perdeu" o ID (movimento brusco, oclusão) e não gera evento novo.
    """

    def __init__(self, conf_evento: float, min_quadros: int, janela: int = 45, limiar_sobrep: float = 0.3):
        self.conf_evento, self.min_quadros = conf_evento, min_quadros
        self.janela, self.limiar_sobrep = janela, limiar_sobrep
        self.trilhas: dict[int, dict] = {}
        self.duplicatas = 0

    def _duplicata(self, caixa, n_quadro: int, tid: int) -> int | None:
        for outro_id, o in self.trilhas.items():
            if (outro_id != tid and o["evento"] and n_quadro - o["ultimo_quadro"] <= self.janela
                    and sobreposicao(caixa, o["ultima_caixa"]) >= self.limiar_sobrep):
                return outro_id
        return None

    def atualizar(self, deteccoes, quadro, n_quadro: int) -> list[dict]:
        novos = []
        for d in deteccoes:
            if d.track_id is None:
                continue
            t = self.trilhas.setdefault(d.track_id, {"primeiro_quadro": n_quadro, "n": 0, "conf_max": 0.0,
                                                     "sev": "baixa", "indice": 0.0, "area_max": 0.0,
                                                     "recorte": None, "emitido": False, "evento": False})
            t["n"] += 1
            t["ultimo_quadro"], t["ultima_caixa"] = n_quadro, d.caixa
            t["conf_max"] = max(t["conf_max"], d.confianca)
            if PESO_SEVERIDADE[d.severidade] > PESO_SEVERIDADE[t["sev"]]:
                t["sev"] = d.severidade
            t["indice"] = max(t["indice"], d.indice_severidade)
            if d.area_relativa > t["area_max"] and d.confianca >= self.conf_evento:
                t["area_max"] = d.area_relativa            # guarda a melhor vista do buraco
                t["recorte"] = recortar_evidencia(quadro, d.caixa)
            if not t["emitido"] and t["n"] >= self.min_quadros and t["conf_max"] >= self.conf_evento:
                t["emitido"] = True
                if self._duplicata(d.caixa, n_quadro, d.track_id) is not None:
                    self.duplicatas += 1                  # mesmo buraco, ID novo: não reporta de novo
                    t["evento"] = True                    # mas passa a valer como referência espacial
                    continue
                t["evento"] = True
                novos.append({"track_id": d.track_id, **t})
        return novos

    @property
    def confirmados(self) -> int:
        return sum(1 for t in self.trilhas.values() if t["evento"]) - self.duplicatas


def _fonte(valor: str):
    return int(valor) if valor.isdigit() else valor


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fonte", default="0", help="índice da webcam, caminho de vídeo ou URL rtsp://")
    ap.add_argument("--pesos", type=Path, default=None, help="modelo .pt ou .onnx (padrão: configs/operacao.json)")
    ap.add_argument("--conf", type=float, default=None, help="limiar para confirmar um buraco (padrão: melhor F1)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="0 (GPU) ou cpu")
    ap.add_argument("--min-quadros", type=int, default=5, help="quadros para confirmar uma trilha")
    ap.add_argument("--saida", type=Path, help="grava o vídeo anotado (mp4)")
    ap.add_argument("--eventos", type=Path, default=Path("demo/eventos.jsonl"))
    ap.add_argument("--evidencias", type=Path, default=Path("demo/evidencias"))
    ap.add_argument("--lat", type=float, help="latitude fixa (vídeos sem GPS)")
    ap.add_argument("--lon", type=float, help="longitude fixa (vídeos sem GPS)")
    ap.add_argument("--api", help="URL base da API VigiaVia para enviar os eventos")
    ap.add_argument("--mostrar", action="store_true", help="abre janela com o vídeo (tecla q encerra)")
    ap.add_argument("--max-quadros", type=int, default=0)
    ap.add_argument("--sem-privacidade", action="store_true")
    ap.add_argument("--resumo", type=Path, help="grava o resumo da execução (JSON)")
    ap.add_argument("--destaque", type=Path, help="grava o quadro (com HUD) de maior confiança somada")
    args = ap.parse_args()

    pesos = args.pesos or config.pesos_padrao()
    conf = args.conf if args.conf is not None else config.conf_padrao()
    det = DetectorBuracos(pesos, conf=conf, imgsz=args.imgsz, device=args.device)
    anon = Anonimizador(intervalo=3) if not args.sem_privacidade else None
    reg = RegistroTrilhas(conf, args.min_quadros)

    cap = cv2.VideoCapture(_fonte(args.fonte))
    if not cap.isOpened():
        raise SystemExit(f"Não foi possível abrir a fonte de vídeo: {args.fonte}")
    fps_fonte = cap.get(cv2.CAP_PROP_FPS) or 30.0
    escritor = None
    args.eventos.parent.mkdir(parents=True, exist_ok=True)
    args.evidencias.mkdir(parents=True, exist_ok=True)
    arq_eventos = args.eventos.open("a", encoding="utf-8")

    sessao_http = None
    if args.api:
        import requests
        sessao_http = requests.Session()

    n, fps_ema, t_ini = 0, 0.0, time.perf_counter()
    tempos = []
    melhor_destaque, pontos_destaque = None, 0.0
    print(f"[VigiaVia] modelo={det.nome} conf={conf:.2f} fonte={args.fonte}")
    try:
        while True:
            t0 = time.perf_counter()                       # pipeline = decodificar -> ... -> HUD
            ok, quadro = cap.read()
            if not ok:
                break
            n += 1
            deteccoes = det.rastrear(quadro)
            privado, _ = anon.borrar_rostos(quadro) if anon else (quadro, 0)
            for ev in reg.atualizar(deteccoes, privado, n):
                evento = {
                    "id": str(uuid.uuid4()),
                    "tipo": "buraco",
                    "track_id": ev["track_id"],
                    "detectado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "fonte": str(args.fonte),
                    "quadro": ev["primeiro_quadro"],
                    "tempo_video_s": round(ev["primeiro_quadro"] / fps_fonte, 2),
                    "confianca_max": round(ev["conf_max"], 4),
                    "severidade": ev["sev"],
                    "indice_severidade": round(ev["indice"], 5),
                    "quadros_observados": ev["n"],
                    "lat": args.lat, "lon": args.lon,
                    "posicao": "fixa (informada na linha de comando)" if args.lat is not None else "indisponível",
                    "modelo": det.nome,
                }
                if ev["recorte"] is not None:
                    caminho = args.evidencias / f"{evento['id']}.jpg"
                    cv2.imwrite(str(caminho), ev["recorte"], [cv2.IMWRITE_JPEG_QUALITY, 88])
                    evento["evidencia"] = caminho.as_posix()
                arq_eventos.write(json.dumps(evento, ensure_ascii=False) + "\n")
                arq_eventos.flush()
                print(f"  [evento] buraco #{evento['track_id']} sev={evento['severidade']} "
                      f"conf={evento['confianca_max']:.2f} t={evento['tempo_video_s']}s")
                if sessao_http and args.lat is not None:
                    corpo = {k: evento[k] for k in ("lat", "lon", "severidade", "confianca_max", "modelo")}
                    corpo["origem"] = "camera_veicular"
                    if ev["recorte"] is not None:
                        ok_jpg, buf = cv2.imencode(".jpg", ev["recorte"], [cv2.IMWRITE_JPEG_QUALITY, 80])
                        corpo["evidencia_jpg_base64"] = base64.b64encode(buf).decode() if ok_jpg else None
                    try:
                        r = sessao_http.post(f"{args.api.rstrip('/')}/v1/relatos", json=corpo, timeout=5)
                        print(f"    -> API {r.status_code}: {r.json().get('mensagem', '')}")
                    except Exception as exc:  # noqa: BLE001 — sem rede: o evento fica no JSONL
                        print(f"    -> API indisponível ({exc.__class__.__name__}); evento mantido em {args.eventos}")

            vis = hud.desenhar(privado, deteccoes, conf, fps_ema, reg.confirmados, det.nome,
                               privacidade_ativa=bool(anon and anon.ativo))
            dt = time.perf_counter() - t0                  # não inclui gravar o vídeo de demonstração
            tempos.append(dt)
            fps_ema = (1 / dt) if fps_ema == 0 else 0.9 * fps_ema + 0.1 * (1 / dt)
            pontos = sum(d.confianca for d in deteccoes if d.confianca >= conf)
            if args.destaque and pontos > pontos_destaque:
                melhor_destaque, pontos_destaque = vis.copy(), pontos
            if args.saida:
                if escritor is None:
                    args.saida.parent.mkdir(parents=True, exist_ok=True)
                    h, w = vis.shape[:2]
                    escritor = cv2.VideoWriter(str(args.saida), cv2.VideoWriter_fourcc(*"mp4v"), fps_fonte, (w, h))
                escritor.write(vis)
            if args.mostrar:
                cv2.imshow("VigiaVia", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.max_quadros and n >= args.max_quadros:
                break
    finally:
        cap.release()
        arq_eventos.close()
        if escritor:
            escritor.release()
        cv2.destroyAllWindows()

    total = time.perf_counter() - t_ini
    t = sorted(tempos[5:] or tempos)                     # ignora o aquecimento
    media_ms = 1000 * sum(t) / max(1, len(t))
    resumo = {"fonte": str(args.fonte), "modelo": det.nome, "limiar": conf, "quadros": n,
              "fps_fonte": round(fps_fonte, 2), "tempo_total_s": round(total, 1),
              "ms_por_quadro_pipeline": round(media_ms, 1),
              "ms_p95": round(1000 * t[int(0.95 * (len(t) - 1))], 1) if t else 0,
              "fps_pipeline": round(1000 / media_ms, 1) if media_ms else 0,
              "trilhas": len(reg.trilhas), "trilhas_duplicadas_suprimidas": reg.duplicatas,
              "buracos_confirmados": reg.confirmados}
    print("[VigiaVia] resumo:", json.dumps(resumo, ensure_ascii=False))
    if args.resumo:
        args.resumo.parent.mkdir(parents=True, exist_ok=True)
        args.resumo.write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.destaque and melhor_destaque is not None:
        args.destaque.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.destaque), melhor_destaque, [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    main()
