"""
07_gerar_video_demo.py — Monta o vídeo de demonstração (1920x1080, H.264 + AAC) com:
  * cartões de abertura, problema, arquitetura, resultados e encerramento;
  * clipes REAIS processados em tempo real pelo VigiaVia (HUD de realidade aumentada);
  * narração em português (voz sintetizada do Windows, "Microsoft Maria") e legendas embutidas.

Pré-requisitos: ffmpeg no PATH; clipes gerados por `python -m vigiavia.tempo_real ... --saida demo/clipes/<nome>.mp4`;
resultados da avaliação (scripts 03–05) e telas da interface em docs/img/telas/.
O roteiro falado também é exportado para docs/roteiro_video.md (a equipe pode regravar com a própria voz).

Uso:
    python scripts/07_gerar_video_demo.py --saida demo/VigiaVia_demo.mp4
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
import textwrap
import wave
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

RAIZ = Path(__file__).resolve().parents[1]
PARAGRAFO = chr(10) * 2   # separador de parágrafos do texto narrado
W, H, FPS = 1920, 1080, 30
FUNDO = (14, 20, 27)
DESTAQUE = (255, 176, 32)
BRANCO = (240, 244, 248)
CINZA = (160, 172, 184)


# ----------------------------------------------------------------------------- tipografia
def _fonte(tam: int, negrito: bool = False) -> ImageFont.FreeTypeFont:
    for nome in (("segoeuib.ttf" if negrito else "segoeui.ttf"), ("arialbd.ttf" if negrito else "arial.ttf")):
        p = Path("C:/Windows/Fonts") / nome
        if p.exists():
            return ImageFont.truetype(str(p), tam)
    from matplotlib import font_manager
    return ImageFont.truetype(font_manager.findfont(font_manager.FontProperties(
        family="DejaVu Sans", weight="bold" if negrito else "normal")), tam)


def quebrar(dr: ImageDraw.ImageDraw, texto: str, fonte, largura: int) -> list[str]:
    palavras, linhas, atual = texto.split(), [], ""
    for p in palavras:
        teste = (atual + " " + p).strip()
        if dr.textlength(teste, font=fonte) <= largura:
            atual = teste
        else:
            linhas.append(atual)
            atual = p
    if atual:
        linhas.append(atual)
    return linhas


# ----------------------------------------------------------------------------- cartões
def cartao_base(titulo: str | None = None, rotulo: str | None = None) -> Image.Image:
    im = Image.new("RGB", (W, H), FUNDO)
    dr = ImageDraw.Draw(im)
    dr.rectangle((0, 0, W, 10), fill=DESTAQUE)
    if rotulo:
        dr.text((96, 70), rotulo.upper(), font=_fonte(28, True), fill=DESTAQUE)
    if titulo:
        dr.text((96, 110), titulo, font=_fonte(60, True), fill=BRANCO)
    dr.text((96, H - 70), "VigiaVia · FIAP Global Solution 2026.2 · Applied Computer Vision", font=_fonte(24),
            fill=CINZA)
    return im


def colar_imagem(base: Image.Image, caminho: Path, caixa: tuple[int, int, int, int], fundo_branco=True) -> None:
    x1, y1, x2, y2 = caixa
    im = Image.open(caminho).convert("RGB")
    esc = min((x2 - x1) / im.width, (y2 - y1) / im.height)
    im = im.resize((int(im.width * esc), int(im.height * esc)), Image.LANCZOS)
    px, py = x1 + ((x2 - x1) - im.width) // 2, y1 + ((y2 - y1) - im.height) // 2
    if fundo_branco:
        ImageDraw.Draw(base).rounded_rectangle((px - 14, py - 14, px + im.width + 14, py + im.height + 14), 18,
                                               fill=(255, 255, 255))
    base.paste(im, (px, py))


def cartao_abertura(integrantes: list[str]) -> Image.Image:
    im = cartao_base()
    dr = ImageDraw.Draw(im)
    dr.text((96, 250), "VigiaVia", font=_fonte(150, True), fill=BRANCO)
    dr.text((100, 430), "Detecção de buracos em vias públicas, em tempo real,", font=_fonte(50), fill=BRANCO)
    dr.text((100, 495), "com visão computacional", font=_fonte(50), fill=BRANCO)
    dr.text((100, 600), "CitizenTech · Engenharia de Software 4º ano · Applied Computer Vision", font=_fonte(32, True),
            fill=DESTAQUE)
    for i, nome in enumerate(integrantes):
        dr.text((100, 690 + 52 * i), nome, font=_fonte(36), fill=BRANCO)
    return im


def cartao_numeros(titulo: str, rotulo: str, itens: list[tuple[str, str]], fonte_rodape: str) -> Image.Image:
    im = cartao_base(titulo, rotulo)
    dr = ImageDraw.Draw(im)
    col_w = (W - 192 - 2 * 48) // 3
    for i, (num, txt) in enumerate(itens[:3]):
        x = 96 + i * (col_w + 48)
        dr.rounded_rectangle((x, 260, x + col_w, 760), 24, fill=(25, 34, 44))
        dr.text((x + 36, 300), num, font=_fonte(96, True), fill=DESTAQUE)
        for j, linha in enumerate(quebrar(dr, txt, _fonte(36), col_w - 72)):
            dr.text((x + 36, 450 + j * 48), linha, font=_fonte(36), fill=BRANCO)
    for j, linha in enumerate(quebrar(dr, fonte_rodape, _fonte(22), W - 192)):
        dr.text((96, 800 + j * 30), linha, font=_fonte(22), fill=CINZA)
    return im


def cartao_figura(titulo: str, rotulo: str, figuras: list[Path], notas: list[str] | None = None) -> Image.Image:
    im = cartao_base(titulo, rotulo)
    topo, base_y = 220, H - 120 - (60 * len(notas or []))
    n = len(figuras)
    larg = (W - 192 - 40 * (n - 1)) // n
    for i, f in enumerate(figuras):
        x = 96 + i * (larg + 40)
        colar_imagem(im, f, (x + 14, topo + 14, x + larg - 14, base_y - 14))
    dr = ImageDraw.Draw(im)
    for j, nota in enumerate(notas or []):
        dr.text((96, base_y + 14 + j * 52), nota, font=_fonte(34), fill=BRANCO)
    return im


# ----------------------------------------------------------------------------- áudio e legendas
def sintetizar(texto: str, destino: Path, taxa: int = 1) -> float:
    """Voz pt-BR do Windows (System.Speech). Retorna a duração em segundos."""
    tmp_txt = destino.with_suffix(".txt")
    tmp_txt.write_text(texto, encoding="utf-8")
    ps = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "try { $s.SelectVoice('Microsoft Maria Desktop') } catch {};"
        f"$s.Rate = {taxa};"
        f"$s.SetOutputToWaveFile('{destino}');"
        f"$t = Get-Content -Raw -Encoding UTF8 '{tmp_txt}';"
        "$s.Speak($t); $s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    with wave.open(str(destino)) as w:
        return w.getnframes() / w.getframerate()


def blocos_legenda(texto: str, max_chars: int = 88) -> list[str]:
    return textwrap.wrap(texto, max_chars, break_long_words=False)


def desenhar_legenda(quadro_bgr: np.ndarray, linha: str) -> np.ndarray:
    pil = Image.fromarray(cv2.cvtColor(quadro_bgr, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(pil, "RGBA")
    f = _fonte(40, True)
    tw = dr.textlength(linha, font=f)
    x, y = (W - tw) / 2, H - 170
    dr.rounded_rectangle((x - 24, y - 12, x + tw + 24, y + 58), 12, fill=(0, 0, 0, 185))
    dr.text((x, y), linha, font=f, fill=(255, 255, 255))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def ajustar(quadro: np.ndarray) -> np.ndarray:
    h, w = quadro.shape[:2]
    esc = min(W / w, H / h)
    rs = cv2.resize(quadro, (int(w * esc), int(h * esc)), interpolation=cv2.INTER_AREA)
    tela = np.zeros((H, W, 3), np.uint8)
    y, x = (H - rs.shape[0]) // 2, (W - rs.shape[1]) // 2
    tela[y:y + rs.shape[0], x:x + rs.shape[1]] = rs
    return tela


# ----------------------------------------------------------------------------- segmentos
def duracao_clipes(clipes: list[Path]) -> float:
    total = 0.0
    for c in clipes:
        cap = cv2.VideoCapture(str(c))
        total += (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / (cap.get(cv2.CAP_PROP_FPS) or FPS)
        cap.release()
    return total


def alinhar_com_whisper(audio: Path, paragrafos: list[str]) -> list[float] | None:
    """Transcreve a narração (faster-whisper, CPU) e devolve o instante de início de cada parágrafo
    (ponto médio entre a última palavra do anterior e a primeira do atual). None se indisponível."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None
    import unicodedata
    bruto = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(audio), "-f", "f32le", "-ac", "1", "-ar", "16000",
                            "-"], capture_output=True, check=True).stdout
    segs, _ = WhisperModel("small", device="cpu", compute_type="int8").transcribe(
        np.frombuffer(bruto, np.float32), language="pt", word_timestamps=True)
    pal = [(w.start, w.end, w.word) for s in segs for w in s.words]
    norm = lambda s: re.sub(r"[^a-z]", "", unicodedata.normalize("NFD", s.lower()).encode("ascii", "ignore").decode())  # noqa: E731
    txt = [norm(x[2]) for x in pal]
    cortes, i0 = [0.0], 0
    for p in paragrafos[1:]:
        alvo = [norm(x) for x in p.split()[:2]]
        i = next((i for i in range(i0 + 1, len(txt) - 1) if txt[i:i + 2] == alvo), None)
        if i is None:
            return None
        cortes.append((pal[i - 1][1] + pal[i][0]) / 2)
        i0 = i
    return cortes


def dividir_narracao(audio: Path, pesos: list[float], dir_tmp: Path, paragrafos: list[str] | None = None) -> list[Path]:
    """Corta uma narração única (um parágrafo por segmento) nas pausas mais próximas das fronteiras
    esperadas, estimadas pela proporção de caracteres de cada parágrafo."""
    sonda = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                            str(audio)], capture_output=True, text=True, check=True)
    total = float(sonda.stdout.strip())
    det = subprocess.run(["ffmpeg", "-i", str(audio), "-af", "silencedetect=noise=-35dB:d=0.15", "-f", "null", "-"],
                         capture_output=True, text=True)
    ini = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", det.stderr)]
    fim = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", det.stderr)]
    pausas = [((a + b) / 2, b - a) for a, b in zip(ini, fim)]
    # Segmentação por programação dinâmica: os cortes ficam em pausas e cada parágrafo deve ter
    # velocidade de fala (caracteres/s) próxima da média da narração; pausas longas ganham bônus.
    r0 = sum(pesos) / total
    pontos = [(0.0, 0.0)] + [p for p in pausas if 0.5 < p[0] < total - 0.5] + [(total, 1.0)]
    k, n = len(pesos), len(pontos)
    inf = float("inf")
    dp = [[inf] * n for _ in range(k + 1)]
    ant = [[-1] * n for _ in range(k + 1)]
    dp[0][0] = 0.0
    for j in range(1, k + 1):
        for b in range(1, n):
            for a in range(b):
                if dp[j - 1][a] == inf:
                    continue
                dur = pontos[b][0] - pontos[a][0]
                if dur < 2.0:
                    continue
                c = dp[j - 1][a] + ((pesos[j - 1] / dur - r0) / r0) ** 2 - 0.05 * pontos[b][1]
                if c < dp[j][b]:
                    dp[j][b], ant[j][b] = c, a
    idx, b = [], n - 1
    for j in range(k, 0, -1):
        idx.append(b)
        b = ant[j][b]
    cortes = [0.0] + [pontos[i][0] for i in reversed(idx)][:-1]
    via_whisper = alinhar_com_whisper(audio, paragrafos) if paragrafos else None
    if via_whisper:
        cortes = via_whisper
    cortes.append(total)
    partes = []
    for i in range(len(pesos)):
        p = dir_tmp / f"narr_{i:02d}.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(audio), "-ss", f"{cortes[i]:.3f}",
                        "-to", f"{cortes[i + 1]:.3f}", "-ar", "48000", "-ac", "1", str(p)], check=True)
        partes.append(p)
    print("  cortes da narração (s):", [round(c, 2) for c in cortes])
    return partes


def render_segmento(seg: dict, dir_tmp: Path, idx: int) -> Path:
    wav = dir_tmp / f"s{idx:02d}.wav"
    if seg.get("wav_externo"):
        shutil.copy2(seg["wav_externo"], wav)
        with wave.open(str(wav)) as w:
            dur_audio = w.getnframes() / w.getframerate()
    else:
        dur_audio = sintetizar(seg["fala"], wav)
    dur = max(dur_audio + 0.8, seg.get("min_dur", 0))
    if seg.get("clipes"):                       # mostra o clipe inteiro (até max_dur)
        dur = min(max(dur, duracao_clipes(seg["clipes"])), seg.get("max_dur", 60))
    n_quadros = int(dur * FPS)
    linhas = blocos_legenda(seg["legenda"])
    pesos = np.cumsum([len(l) for l in linhas]) / max(1, sum(len(l) for l in linhas))
    mp4 = dir_tmp / f"s{idx:02d}_video.mp4"
    vw = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    estatico = None
    if "imagem" in seg:
        estatico = cv2.cvtColor(np.asarray(seg["imagem"]), cv2.COLOR_RGB2BGR)
    caps = [cv2.VideoCapture(str(c)) for c in seg.get("clipes", [])]
    ic, ultimo = 0, None
    for k in range(n_quadros):
        if estatico is not None:
            q = estatico
        else:
            ok, fr = caps[ic].read()
            while not ok and ic < len(caps) - 1:     # passa ao próximo clipe
                ic += 1
                ok, fr = caps[ic].read()
            if not ok:                               # acabou: congela no último quadro
                fr = ultimo
            ultimo = fr
            q = ajustar(fr)
        ts = k / FPS
        if linhas and ts < dur_audio + 0.3:
            frac = min(0.999, ts / dur_audio)
            linha = linhas[min(int(np.searchsorted(pesos, frac, side="right")), len(linhas) - 1)]
            q = desenhar_legenda(q, linha)
        vw.write(q)
    vw.release()
    for c in caps:
        c.release()

    final = dir_tmp / f"s{idx:02d}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-i", str(wav), "-af", "apad",
                    "-t", f"{dur:.2f}", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "20",
                    "-c:a", "aac", "-b:a", "160k", "-ar", "48000", str(final)], check=True)
    print(f"  segmento {idx:02d} ({dur:.1f}s): {seg['id']}")
    return final


def carregar_numeros() -> dict:
    op = json.loads((RAIZ / "configs" / "operacao.json").read_text(encoding="utf-8"))
    an = json.loads((RAIZ / "resultados" / "analise_teste.json").read_text(encoding="utf-8"))
    vel = json.loads((RAIZ / "resultados" / "velocidade.json").read_text(encoding="utf-8"))
    m = op["metricas_teste"]
    exp = op["experimento"].split("_")[0]
    linhas = [l for l in vel["linhas"] if l["modelo"].startswith(exp)]
    gpu = next((l["fps"] for l in linhas if l["execucao"] == "PyTorch GPU FP16"), None)
    cpu = next((l["fps"] for l in vel["linhas"] if l["execucao"] == "ONNX Runtime CPU"), None)
    ds = json.loads((RAIZ / "resultados" / "dataset_stats.json").read_text(encoding="utf-8"))["splits"]
    total = sum(v["imagens"] for v in ds.values())
    return {"op": op, "an": an, "map50": m["mAP50"], "map5095": m["mAP50_95"], "p": m["precisao_no_limiar"],
            "r": m["revocacao_no_limiar"], "f1": m["f1_no_limiar"], "fps_gpu": gpu, "fps_cpu": cpu,
            "n_total": f"{total:,}".replace(",", "."), "n_teste": f"{ds['test']['imagens']:,}".replace(",", "."),
            "modelo": op["experimento"].split("_", 1)[1].replace("yolo", "YOLO")}


def pct(v: float) -> str:
    return f"{v * 100:.0f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", type=Path, default=RAIZ / "demo" / "VigiaVia_demo.mp4")
    ap.add_argument("--clipes", type=Path, default=RAIZ / "demo" / "clipes")
    ap.add_argument("--narracao", type=Path, help="narração única (mp3/wav), um parágrafo por segmento, "
                                                   "no lugar da voz sintetizada do Windows")
    ap.add_argument("--texto-narracao", type=Path, default=RAIZ / "docs" / "narracao.txt",
                    help="texto narrado (parágrafos separados por linha em branco) para alinhar os cortes")
    args = ap.parse_args()
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg não encontrado no PATH")
    n = carregar_numeros()
    res, img = RAIZ / "resultados", RAIZ / "docs" / "img"
    integrantes = ["Deivison Pertel — RM 550803", "Eduardo Akira Murata — RM 98713",
                   "Wesley Souza de Oliveira — RM 97874"]
    telas = sorted((img / "telas").glob("*.png"))

    segs = [
        {"id": "abertura", "imagem": cartao_abertura(integrantes), "min_dur": 8,
         "fala": "Olá! Este é o Vigia Via, o componente de visão computacional da nossa solução cidadã para a Global "
                 "Solution da FIAP. Integrantes: Deivison Pertel, Eduardo Akira Murata e Wesley Souza de Oliveira.",
         "legenda": "Olá! Este é o VigiaVia, o componente de visão computacional da nossa CitizenTech para a Global "
                    "Solution da FIAP. Integrantes: Deivison Pertel, Eduardo Akira Murata e Wesley Souza de Oliveira."},
        {"id": "problema", "min_dur": 10, "imagem": cartao_numeros(
            "Buraco na rua é risco, custo e fila de chamados", "O problema",
            [("113 mil", "pedidos de tapa-buraco em São Paulo em 2025"),
             ("80%", "dos pontos críticos das rodovias brasileiras são buracos"),
             ("+31%", "no custo operacional do transporte com pavimento ruim")],
            "Fontes: Secretaria Municipal das Subprefeituras de SP via BNews (09/08/2026) — \"um mesmo ponto pode gerar "
            "vários chamados\"; Pesquisa CNT de Rodovias 2025 (17/12/2025)."),
         "fala": "Buracos causam quedas de motociclistas e ciclistas, danos a veículos e atrasos no transporte. Só em São "
                 "Paulo foram cento e treze mil pedidos de tapa-buraco em 2025, e a própria prefeitura reconhece que um "
                 "mesmo buraco gera vários chamados. Hoje, tudo depende de alguém parar e registrar manualmente.",
         "legenda": "Buracos causam quedas de motociclistas e ciclistas, danos a veículos e atrasos no transporte. Só em "
                    "São Paulo foram 113 mil pedidos de tapa-buraco em 2025, e a própria prefeitura reconhece que um "
                    "mesmo buraco gera vários chamados. Hoje, tudo depende de alguém parar e registrar manualmente."},
        {"id": "solucao", "min_dur": 12, "imagem": cartao_figura("Como funciona", "A solução", [img / "arquitetura.png"]),
         "fala": f"O Vigia Via detecta buracos em tempo real, com a câmera do celular ou de ônibus e caminhões de coleta. "
                 f"Treinamos um detector iôlo vinte e seis com o dataset internacional R D D dois mil e vinte e dois, "
                 f"rastreamos cada buraco para gerar um único relato, estimamos a severidade, e enviamos só um recorte "
                 f"da via e as coordenadas para a fila da zeladoria.",
         "legenda": "O VigiaVia detecta buracos em tempo real, com a câmera do celular ou de ônibus e caminhões de coleta. "
                    "Treinamos um detector YOLO26 com o dataset internacional RDD2022, rastreamos cada buraco (ByteTrack) "
                    "para gerar um único relato, estimamos a severidade e enviamos só um recorte da via e as coordenadas "
                    "para a fila da zeladoria."},
        {"id": "dados", "min_dur": 11, "imagem": cartao_figura(
            "Antes de treinar: auditoria dos rótulos", "Dados",
            [res / "auditoria_rotulos_redistribuicao_hf.jpg", res / "auditoria_rotulos_oficial.jpg"],
            ["Esquerda: cópia com rótulos trocados (faixas, bueiros)  ·  Direita: buracos reais (D40 oficial)"]),
         "fala": f"Antes de treinar, auditamos os rótulos: numa cópia do dataset, faixas apagadas e tampas de bueiro "
                 f"estavam marcadas como buraco. Usamos a versão oficial, com {n['n_total']} imagens de cinco países, e "
                 f"separamos treino e teste por trajeto, para não haver vazamento.",
         "legenda": f"Antes de treinar, auditamos os rótulos: numa cópia do dataset, faixas apagadas e tampas de bueiro "
                    f"estavam marcadas como buraco. Usamos a versão oficial ({n['n_total']} imagens de cinco países) e "
                    f"separamos treino e teste por trajeto, sem vazamento."},
        {"id": "demo_sp", "clipes": [args.clipes / "sp_bela_vista_rua_silvia.mp4"], "min_dur": 14, "max_dur": 27,
         "fala": "Este é um vídeo real gravado na Rua Sílvia, na Bela Vista, em São Paulo. O modelo nunca viu este vídeo, "
                 "nem imagens do Brasil, durante o treino. Cada caixa mostra a confiança e a severidade estimada, e quando "
                 "o buraco entra na trajetória, a interface de realidade aumentada dispara o alerta.",
         "legenda": "Vídeo real gravado na Rua Sílvia, Bela Vista, São Paulo. O modelo nunca viu este vídeo, nem imagens "
                    "do Brasil, durante o treino. Cada caixa mostra a confiança e a severidade estimada; quando o buraco "
                    "entra na trajetória, a interface de realidade aumentada dispara o alerta."},
        {"id": "demo_outros", "clipes": [args.clipes / "kumasi_pothole.mp4", args.clipes / "carros_no_buraco.mp4"],
         "min_dur": 14, "max_dur": 24,
         "fala": "Em outra cidade e com outra câmera, o rastreamento mantém o mesmo identificador para cada buraco ao longo "
                 "dos quadros, e o contador soma apenas buracos únicos. Rostos são borrados automaticamente, por "
                 "privacidade.",
         "legenda": "Em outra cidade e com outra câmera, o rastreamento mantém o mesmo identificador para cada buraco ao "
                    "longo dos quadros, e o contador soma apenas buracos únicos. Rostos são borrados automaticamente."},
        {"id": "resultados", "min_dur": 14, "imagem": cartao_figura(
            f"Resultados no conjunto de teste ({n['n_teste']} imagens nunca vistas)", "Avaliação",
            [res / "comparacao_modelos.png", res / "velocidade.png"],
            [f"Modelo em produção: {n['modelo']} · mAP@0,5 = {n['map50']:.3f} · precisão {pct(n['p'])}% e revocação "
             f"{pct(n['r'])}% no limiar escolhido"]),
         "fala": f"No conjunto de teste, com {n['n_teste']} imagens nunca vistas, o modelo escolhido atingiu "
                 f"m A P de {pct(n['map50'])} por cento com interseção de cinquenta por cento, com precisão de "
                 f"{pct(n['p'])} por cento e revocação de {pct(n['r'])} por cento no limiar escolhido. Comparamos três "
                 f"modelos e escolhemos o mais leve dentro da margem de precisão: ele roda a "
                 f"{n['fps_gpu']:.0f} quadros por segundo na placa de vídeo e {n['fps_cpu']:.0f} no processador.",
         "legenda": f"No teste ({n['n_teste']} imagens nunca vistas): mAP@0,5 de {pct(n['map50'])}%, precisão de {pct(n['p'])}% e "
                    f"revocação de {pct(n['r'])}% no limiar escolhido. Comparamos três modelos e escolhemos o mais leve "
                    f"dentro da margem de precisão: {n['fps_gpu']:.0f} FPS na GPU e {n['fps_cpu']:.0f} FPS na CPU."},
        {"id": "exemplos", "min_dur": 10, "imagem": cartao_figura(
            "Acertos e erros reais do modelo", "Análise de erros",
            [res / "exemplos_acertos.jpg", res / "exemplos_falsos_negativos.jpg"]),
         "fala": "À esquerda, acertos. À direita, buracos que o modelo perdeu: em geral pequenos, distantes ou com sombra. "
                 "Por isso a detecção é uma sugestão: uma pessoa da zeladoria valida antes de virar ordem de serviço.",
         "legenda": "À esquerda, acertos. À direita, buracos que o modelo perdeu: em geral pequenos, distantes ou com "
                    "sombra. Por isso a detecção é uma sugestão: uma pessoa da zeladoria valida antes da ordem de serviço."},
        {"id": "vieses", "min_dur": 9, "imagem": cartao_figura(
            "Onde o modelo é mais fraco (vieses medidos)", "IA responsável", [res / "vieses_subgrupos.png"]),
         "fala": "Também medimos o desempenho por país de origem das imagens e por iluminação. Saber onde o modelo erra "
                 "orienta a próxima etapa: coletar imagens brasileiras pelo próprio aplicativo e retreinar.",
         "legenda": "Medimos o desempenho por país de origem e por iluminação. Saber onde o modelo erra orienta a próxima "
                    "etapa: coletar imagens brasileiras pelo próprio aplicativo e retreinar."},
    ]
    if telas:
        segs.append({"id": "integracao", "min_dur": 12, "imagem": cartao_figura(
            "Integração com o MVP: cidadão e zeladoria", "API + interface web", telas[:2]),
            "fala": "A visão está integrada ao M V P por uma A P I. O cidadão envia uma foto ou usa a câmera ao vivo, "
                    "inclusive com o modelo rodando no próprio celular, sem internet. Relatos a menos de quinze metros "
                    "viram confirmações do mesmo buraco, e a zeladoria vê uma fila priorizada, com mapa.",
            "legenda": "A visão está integrada ao MVP por uma API. O cidadão envia uma foto ou usa a câmera ao vivo, "
                       "inclusive com o modelo rodando no próprio celular, sem internet. Relatos a menos de 15 m viram "
                       "confirmações do mesmo buraco, e a zeladoria vê uma fila priorizada, com mapa."})
    fim = cartao_base("Ruas mais seguras com visão computacional responsável", "Encerramento")
    dr = ImageDraw.Draw(fim)
    for j, t in enumerate(["ODS 11 · Cidades e comunidades sustentáveis", "ODS 9 · Indústria, inovação e infraestrutura",
                           "ODS 3 · Saúde e bem-estar (segurança viária)", "Código, modelo e relatório no GitHub do grupo"]):
        dr.text((100, 260 + j * 70), "•  " + t, font=_fonte(44), fill=BRANCO)
    for i, nome in enumerate(integrantes):
        dr.text((100, 600 + 52 * i), nome, font=_fonte(36), fill=CINZA)
    segs.append({"id": "encerramento", "imagem": fim, "min_dur": 8,
                 "fala": "Vigia Via: ruas mais seguras com visão computacional responsável, conectada aos objetivos de "
                         "desenvolvimento sustentável onze, nove e três. O código completo está no nosso GitHub. Obrigado!",
                 "legenda": "VigiaVia: ruas mais seguras com visão computacional responsável, conectada aos ODS 11, 9 e 3. "
                            "Código completo no nosso GitHub. Obrigado!"})

    # roteiro para a equipe
    md = ["# Roteiro do vídeo de demonstração (gerado por scripts/07_gerar_video_demo.py)", "",
          "A narração abaixo foi sintetizada com a voz pt-BR do Windows. A equipe pode regravá-la com a própria voz.", ""]
    for i, s in enumerate(segs, 1):
        md += [f"## {i}. {s['id']}", "", s["legenda"], ""]
    (RAIZ / "docs" / "roteiro_video.md").write_text("\n".join(md), encoding="utf-8")

    with tempfile.TemporaryDirectory() as d:
        dir_tmp = Path(d)
        if args.narracao:
            texto = (args.texto_narracao.read_text(encoding="utf-8").split(PARAGRAFO) if args.texto_narracao
                     else [s["fala"] for s in segs])
            assert len(texto) == len(segs), f"{len(texto)} parágrafos para {len(segs)} segmentos"
            for s, w in zip(segs, dividir_narracao(args.narracao, [len(x.strip()) for x in texto], dir_tmp, [x.strip() for x in texto])):
                s["wav_externo"] = w
        partes = [render_segmento(s, dir_tmp, i) for i, s in enumerate(segs)]
        lista = dir_tmp / "lista.txt"
        lista.write_text("".join(f"file '{p.as_posix()}'\n" for p in partes), encoding="utf-8")
        args.saida.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lista),
                        "-c", "copy", "-movflags", "+faststart", str(args.saida)], check=True)
    print(f"Vídeo salvo em {args.saida}")


if __name__ == "__main__":
    main()
