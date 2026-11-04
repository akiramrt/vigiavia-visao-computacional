"""
06b_processar_videos.py — Roda o VigiaVia em tempo real sobre os vídeos reais baixados pelo
script 06 e gera: clipes anotados (demo/clipes), eventos (demo/eventos_*.jsonl), recortes de
evidência, resumos de desempenho (resultados/tempo_real/*.json) e quadros de exemplo.

Somente o vídeo de São Paulo recebe coordenada (aproximada da Rua Sílvia, Bela Vista, via
OpenStreetMap); os demais não têm posição e não são enviados à API.

Uso:
    python scripts/06b_processar_videos.py --videos demo/videos_fonte [--api http://localhost:8000]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
COORD_SP = (-23.560345, -46.650508)    # Rua Sílvia, Bela Vista (OpenStreetMap/Nominatim)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=Path, default=RAIZ / "demo" / "videos_fonte")
    ap.add_argument("--pesos", type=Path)
    ap.add_argument("--api")
    args = ap.parse_args()

    creditos = json.loads((args.videos / "creditos.json").read_text(encoding="utf-8"))
    for c in creditos:
        nome = Path(c["arquivo"]).stem
        cmd = [sys.executable, "-m", "vigiavia.tempo_real", "--fonte", str(args.videos / c["arquivo"]),
               "--saida", str(RAIZ / "demo" / "clipes" / f"{nome}.mp4"),
               "--eventos", str(RAIZ / "demo" / f"eventos_{nome}.jsonl"),
               "--evidencias", str(RAIZ / "demo" / "evidencias" / nome),
               "--resumo", str(RAIZ / "resultados" / "tempo_real" / f"{nome}.json"),
               "--destaque", str(RAIZ / "resultados" / "tempo_real" / f"{nome}.jpg")]
        if args.pesos:
            cmd += ["--pesos", str(args.pesos)]
        if nome.startswith("sp_"):
            cmd += ["--lat", str(COORD_SP[0]), "--lon", str(COORD_SP[1])]
            if args.api:
                cmd += ["--api", args.api]
        (RAIZ / "demo" / f"eventos_{nome}.jsonl").unlink(missing_ok=True)
        print(">>", nome)
        subprocess.run(cmd, check=True, cwd=RAIZ / "src")
        resumo = RAIZ / "resultados" / "tempo_real" / f"{nome}.json"
        d = json.loads(resumo.read_text(encoding="utf-8"))
        d["creditos"] = c
        resumo.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
