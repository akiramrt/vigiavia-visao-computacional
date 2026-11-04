"""
01_preparar_dataset.py — Monta o dataset VigiaVia (classe única "buraco") a partir da release OFICIAL do RDD2022.

Fonte: RDD2022 — Multi-National Road Damage Dataset (Arya et al., 2022, arXiv:2209.08538), release oficial do
CRDDC'2022 no figshare (doi:10.6084/m9.figshare.21431547), anotações Pascal VOC com o NOME da classe
(D00, D10, D20, D40...). Usamos apenas D40 = buraco (pothole).

Por que a release oficial? Uma auditoria visual mostrou que uma redistribuição em formato YOLO (Hugging Face)
rotula como "pothole" marcas de pista apagadas, faixas de pedestre e tampas de bueiro — os buracos de verdade
haviam sido descartados na conversão. Aqui a classe é lida pelo nome no XML oficial.

O zip oficial tem 13 GB; este script lê só o índice central e baixa, por requisições HTTP parciais (Range),
somente os XMLs e as imagens selecionadas (~1–2 GB).

Regras de seleção (documentadas no relatório):
  * Somente imagens de nível de rua (celular/câmera em veículo) dos países em --paises.
    China_Drone (visão aérea) fica de fora; Norway (imagens 3.650 px) também, pelo custo de download.
  * Positivas: imagens com >= 1 caixa D40. Negativas: amostra (semente fixa) de imagens SEM buraco,
    na proporção --neg-ratio das positivas — elas têm trincas, faixas e bueiros e ensinam o que NÃO é buraco.
  * Split 70/15/15 em BLOCOS de --bloco quadros consecutivos (mesmo trajeto), para reduzir vazamento de
    quadros quase idênticos entre treino e teste. (O teste oficial do RDD2022 não tem rótulos públicos.)

Uso:
    python scripts/01_preparar_dataset.py --out datasets/vigiavia-rdd2022
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import struct
import threading
import time
import xml.etree.ElementTree as ET
import zipfile
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

FIGSHARE = "https://ndownloader.figshare.com/files/38030910"   # RDD2022_released_through_CRDDC2022.zip
PAISES_PADRAO = ["Japan", "India", "China_MotorBike", "Czech", "United_States"]
USER_AGENT = "VigiaVia-dataset-prep/2.0 (FIAP GS 2026 - projeto academico)"
_local = threading.local()


# ----------------------------------------------------------------------------- HTTP com Range
class Origem:
    """Resolve o link assinado do figshare (expira em segundos) e faz GETs parciais com retentativa."""

    def __init__(self):
        self._url, self._t = None, 0.0
        self._trava = threading.Lock()

    def _sessao(self) -> requests.Session:
        if not hasattr(_local, "s"):
            _local.s = requests.Session()
            _local.s.headers["User-Agent"] = USER_AGENT
        return _local.s

    def url(self, renovar: bool = False) -> str:
        with self._trava:
            if renovar or self._url is None or time.time() - self._t > 6:
                r = self._sessao().get(FIGSHARE, allow_redirects=False, timeout=30)
                self._url, self._t = r.headers["Location"], time.time()
            return self._url

    def intervalo(self, ini: int, fim: int) -> bytes:
        espera = 2.0
        for tentativa in range(8):
            try:
                r = self._sessao().get(self.url(renovar=tentativa > 0), headers={"Range": f"bytes={ini}-{fim}"},
                                       timeout=120)
                if r.status_code == 206 and len(r.content) == fim - ini + 1:
                    return r.content
            except requests.RequestException:
                pass
            time.sleep(espera)
            espera = min(espera * 2, 30)
        raise RuntimeError(f"Falha ao baixar bytes {ini}-{fim}")

    def tamanho(self) -> int:
        r = self._sessao().get(self.url(), headers={"Range": "bytes=0-0"}, timeout=30)
        return int(r.headers["Content-Range"].split("/")[1])


class Janela:
    """Sub-intervalo [base, base+tam) do arquivo remoto. O zip oficial guarda um zip por país SEM compressão,
    então cada país é uma janela contígua que pode ser lida como um zip independente."""

    def __init__(self, origem: Origem, base: int = 0, tam: int | None = None):
        self.origem, self.base = origem, base
        self.tam = tam if tam is not None else origem.tamanho()

    def intervalo(self, ini: int, fim: int) -> bytes:
        return self.origem.intervalo(self.base + ini, self.base + fim)

    def tamanho(self) -> int:
        return self.tam


class ArquivoRemoto(io.RawIOBase):
    """Arquivo somente-leitura sobre HTTP Range, com cache de blocos — suficiente para o zipfile ler o índice."""

    def __init__(self, origem: Janela, bloco: int = 1 << 21):
        self.o, self.bloco, self.pos = origem, bloco, 0
        self.tam = origem.tamanho()
        self.cache: dict[int, bytes] = {}

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.tam + off
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.tam - self.pos)
        if n <= 0:
            return 0
        saida, p = bytearray(), self.pos
        while len(saida) < n:
            i = p // self.bloco
            if i not in self.cache:
                self.cache[i] = self.o.intervalo(i * self.bloco, min(self.tam, (i + 1) * self.bloco) - 1)
            ini = p - i * self.bloco
            pedaco = self.cache[i][ini: ini + n - len(saida)]
            saida += pedaco
            p += len(pedaco)
        b[:n] = saida
        self.pos += n
        return n


def extrair_membros(origem: Janela, membros: list[zipfile.ZipInfo], workers: int = 8,
                    folga: int = 1 << 20, max_bloco: int = 48 << 20):
    """Baixa os membros em blocos contíguos (poucos GETs grandes) e devolve (nome, bytes) já descomprimidos."""
    membros = sorted(membros, key=lambda z: z.header_offset)
    grupos, atual = [], []
    for z in membros:
        fim = z.header_offset + 30 + len(z.filename.encode()) + 512 + z.compress_size
        if atual and (z.header_offset - atual[-1][1] > folga or fim - atual[0][0].header_offset > max_bloco):
            grupos.append(atual)
            atual = []
        atual.append((z, fim))
    if atual:
        grupos.append(atual)

    def baixar_grupo(g):
        ini, fim = g[0][0].header_offset, min(max(f for _, f in g), origem_tam[0] - 1)
        buf = origem.intervalo(ini, fim)
        out = []
        for z, _ in g:
            o = z.header_offset - ini
            assinatura, *_, fn_len, ex_len = struct.unpack("<IHHHHHIIIHH", buf[o:o + 30])
            assert assinatura == 0x04034B50, z.filename
            d0 = o + 30 + fn_len + ex_len
            dados = buf[d0:d0 + z.compress_size]
            out.append((z.filename, zlib.decompress(dados, -15) if z.compress_type == 8 else bytes(dados)))
        return out

    origem_tam = [origem.tamanho()]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(baixar_grupo, g) for g in grupos]
        for i, f in enumerate(as_completed(futs), 1):
            yield from f.result()
            if i % 10 == 0 or i == len(futs):
                print(f"      blocos {i}/{len(futs)}")


# ----------------------------------------------------------------------------- anotações
def ler_xml(conteudo: bytes) -> dict:
    raiz = ET.fromstring(conteudo)
    tam = raiz.find("size")
    w, h = int(float(tam.findtext("width"))), int(float(tam.findtext("height")))
    objs = []
    for o in raiz.iter("object"):
        nome = (o.findtext("name") or "").strip()
        bb = o.find("bndbox")
        if bb is None:
            continue
        x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        objs.append((nome, x1, y1, x2, y2))
    return {"w": w, "h": h, "objs": objs}


def numero(arquivo: str) -> int:
    """Número sequencial do quadro: 'Japan_000123' ou 'Japan_000123.jpg' -> 123."""
    m = re.search(r"(\d+)(?:\.\w+)?$", arquivo)
    if not m:
        raise ValueError(f"nome sem número de quadro: {arquivo}")
    return int(m.group(1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("datasets/vigiavia-rdd2022"))
    ap.add_argument("--paises", nargs="+", default=PAISES_PADRAO)
    ap.add_argument("--neg-ratio", type=float, default=0.25)
    ap.add_argument("--bloco", type=int, default=20, help="quadros consecutivos por bloco no split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    raiz = args.out.resolve()
    raiz.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    origem = Origem()

    print("[1/5] Lendo os índices (zip oficial de 13 GB -> zip de cada país) por HTTP Range...")
    externo = Janela(origem)
    zf = zipfile.ZipFile(ArquivoRemoto(externo))
    janelas = {}
    for z in zf.infolist():
        pais = Path(z.filename).stem
        if z.filename.endswith(".zip") and pais in args.paises:
            assert z.compress_type == 0, "zip interno comprimido — leitura por intervalo impossível"
            cab = externo.intervalo(z.header_offset, z.header_offset + 29)
            fn_len, ex_len = struct.unpack("<HH", cab[26:30])
            janelas[pais] = Janela(origem, z.header_offset + 30 + fn_len + ex_len, z.compress_size)
            print(f"      {pais:16s} {z.compress_size / 1e6:8.1f} MB")
    padrao = re.compile(r"(?:.*/)?(?P<pais>[A-Za-z_]+)/train/(?P<tipo>annotations/xmls|images)/(?P<arq>[^/]+)$")
    por_pais = defaultdict(dict)
    for pais, jan in janelas.items():
        for z in zipfile.ZipFile(ArquivoRemoto(jan)).infolist():
            m = padrao.match(z.filename)
            if m and not z.is_dir():
                chave = Path(m.group("arq")).stem
                por_pais[pais].setdefault(chave, {})["xml" if m.group("tipo").startswith("ann") else "img"] = z

    print("[2/5] Baixando e lendo as anotações XML oficiais...")
    anot = {}
    for pais, jan in janelas.items():
        xmls = [d["xml"] for d in por_pais[pais].values() if "xml" in d and "img" in d]
        for nome, conteudo in extrair_membros(jan, xmls, args.workers):
            anot[Path(nome).stem] = ler_xml(conteudo)
    contagem = Counter(o[0] for a in anot.values() for o in a["objs"])
    print(f"      {len(anot)} XMLs; classes: {dict(contagem.most_common())}")

    print("[3/5] Selecionando imagens e criando splits por blocos de quadros consecutivos...")
    selecao = []
    for pais in args.paises:
        chaves = sorted((k for k in por_pais[pais] if k in anot), key=numero)
        pos = [k for k in chaves if any(o[0] == "D40" for o in anot[k]["objs"])]
        conj_pos = set(pos)
        neg_todas = [k for k in chaves if k not in conj_pos]
        neg = rng.sample(neg_todas, min(len(neg_todas), round(len(pos) * args.neg_ratio)))
        blocos = sorted({numero(k) // args.bloco for k in pos + neg})
        split_bloco = {}
        for b in blocos:
            u = rng.random()
            split_bloco[b] = "train" if u < 0.70 else ("val" if u < 0.85 else "test")
        for k in pos + neg:
            selecao.append((pais, k, split_bloco[numero(k) // args.bloco]))
        print(f"      {pais:16s} positivas={len(pos):5d} negativas={len(neg):5d} (de {len(chaves)} rotuladas)")

    print(f"[4/5] Baixando {len(selecao)} imagens (só os trechos necessários de cada zip)...")
    for pais, jan in janelas.items():
        destino = {}
        for p, k, split in selecao:
            if p == pais:
                z = por_pais[pais][k]["img"]
                destino[z.filename] = (z, raiz / "images" / split / Path(z.filename).name)
        faltando = [z for z, d in destino.values() if not d.exists()]
        print(f"      {pais}: {len(faltando)} imagens")
        for nome, conteudo in extrair_membros(jan, faltando, args.workers):
            d = destino[nome][1]
            d.parent.mkdir(parents=True, exist_ok=True)
            d.write_bytes(conteudo)

    print("[5/5] Gerando rótulos YOLO, manifest e estatísticas...")
    stats = defaultdict(Counter)
    with open(raiz / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["split", "arquivo", "pais", "largura", "altura", "n_buracos", "outras_classes"])
        for pais, k, split in selecao:
            a = anot[k]
            w, h = a["w"], a["h"]
            img = Path(por_pais[pais][k]["img"].filename).name
            linhas = []
            for nome, x1, y1, x2, y2 in a["objs"]:
                if nome != "D40":
                    continue
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                linhas.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            lbl = raiz / "labels" / split / (Path(img).stem + ".txt")
            lbl.parent.mkdir(parents=True, exist_ok=True)
            lbl.write_text("\n".join(linhas) + ("\n" if linhas else ""), encoding="utf-8")
            outras = sorted({o[0] for o in a["objs"] if o[0] != "D40"})
            wr.writerow([split, img, pais, w, h, len(linhas), " ".join(outras)])
            stats[split]["imagens"] += 1
            stats[split]["imagens_com_buraco"] += bool(linhas)
            stats[split]["caixas_buraco"] += len(linhas)
            stats[split][f"imagens_{pais}"] += 1

    (raiz / "data.yaml").write_text(
        "# Gerado por scripts/01_preparar_dataset.py — RDD2022 oficial (figshare), classe D40\n"
        f"path: {raiz.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: buraco\n", encoding="utf-8")
    meta = {"fonte": "RDD2022 oficial (CRDDC'2022) — figshare doi:10.6084/m9.figshare.21431547",
            "licenca": "CC BY 4.0 (figshare) / CC BY-SA 4.0 (repositório oficial) — usamos os termos mais restritivos",
            "classe": {"0": "buraco (D40)"}, "paises": args.paises, "neg_ratio": args.neg_ratio,
            "bloco_split": args.bloco, "seed": args.seed, "classes_no_rdd_selecionado": dict(contagem.most_common()),
            "splits": {k: dict(v) for k, v in stats.items()}}
    (raiz / "stats.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta["splits"], indent=2, ensure_ascii=False))
    print(f"Pronto. data.yaml em {raiz / 'data.yaml'}")


if __name__ == "__main__":
    main()
