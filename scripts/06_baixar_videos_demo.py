"""
06_baixar_videos_demo.py — Baixa vídeos REAIS de vias com buracos (licenças abertas) do
Wikimedia Commons para a demonstração em tempo real, registrando autor e licença.

Os vídeos NÃO fazem parte do treino: servem como teste fora da distribuição (câmera na mão,
outras cidades, outra época), inclusive um vídeo gravado em São Paulo (Bela Vista).

Uso:
    python scripts/06_baixar_videos_demo.py --out demo/videos_fonte
"""
from __future__ import annotations

import argparse
import html
import json
import re
import time
import urllib.parse
from pathlib import Path

import requests

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "VigiaVia-demo/1.0 (FIAP GS 2026 - projeto academico)"

VIDEOS = {
    "sp_bela_vista_rua_silvia": "File:Buraco na rua Sílvia, 276 (Bela Vista).webm",
    "kumasi_pothole": "File:Pothole.ogv",
    "carros_no_buraco": "File:Cars Moving into pothole.webm",
    "estrada_com_agua": "File:Rainwater-Filled Potholes on a Community Road.webm",
}


def limpar_html(txt: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", txt or "")).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("demo/videos_fonte"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    params = {"action": "query", "titles": "|".join(VIDEOS.values()), "prop": "imageinfo",
              "iiprop": "url|size|mime|extmetadata", "format": "json"}
    paginas = s.get(API, params=params, timeout=60).json()["query"]["pages"].values()
    info = {p["title"]: p["imageinfo"][0] for p in paginas if "imageinfo" in p}

    creditos = []
    for apelido, titulo in VIDEOS.items():
        ii = info.get(titulo)
        if not ii:
            print(f"[AVISO] não encontrado no Commons: {titulo}")
            continue
        em = ii.get("extmetadata", {})
        ext = Path(urllib.parse.urlparse(ii["url"]).path).suffix
        destino = args.out / f"{apelido}{ext}"
        if not destino.exists():
            print(f"Baixando {titulo} ({ii['size'] / 1e6:.1f} MB)...")
            for tentativa in range(6):          # o Commons responde 429 se baixarmos rápido demais
                r = s.get(ii["url"], timeout=300)
                if r.status_code != 429:
                    break
                espera = int(r.headers.get("Retry-After", 0)) or 15 * (tentativa + 1)
                print(f"  429 (limite de taxa) — aguardando {espera}s")
                time.sleep(espera)
            r.raise_for_status()
            destino.write_bytes(r.content)
            time.sleep(3)
        creditos.append({
            "arquivo": destino.name,
            "titulo_commons": titulo,
            "pagina": "https://commons.wikimedia.org/wiki/" + urllib.parse.quote(titulo.replace(" ", "_")),
            "autor": limpar_html(em.get("Artist", {}).get("value", "")),
            "licenca": em.get("LicenseShortName", {}).get("value", ""),
            "data": limpar_html(em.get("DateTimeOriginal", {}).get("value", "")),
            "descricao": limpar_html(em.get("ImageDescription", {}).get("value", ""))[:300],
            "resolucao": f"{ii.get('width')}x{ii.get('height')}",
        })
    (args.out / "creditos.json").write_text(json.dumps(creditos, indent=2, ensure_ascii=False), encoding="utf-8")
    for c in creditos:
        print(f"- {c['arquivo']}: {c['autor']} — {c['licenca']} — {c['pagina']}")


if __name__ == "__main__":
    main()
