"""
06c_capturar_telas.py — Popula a API com relatos de demonstração e captura telas da interface web.

Relatos de demonstração: cada amostra de api/static/amostras é enviada a POST /v1/deteccoes (fluxo real);
o recorte do buraco de maior confiança vira a evidência de um relato. As COORDENADAS são simuladas em torno
da Rua Sílvia (Bela Vista, São Paulo) e duas amostras são enviadas a menos de 15 m uma da outra para
demonstrar a deduplicação. Tudo é documentado no relatório como dado simulado.

Requer a API rodando:  uvicorn api.main:app --port 8000
Uso:  python scripts/06c_capturar_telas.py --api http://localhost:8000
"""
from __future__ import annotations

import argparse
import base64
from pathlib import Path

import cv2
import requests
from playwright.sync_api import sync_playwright

RAIZ = Path(__file__).resolve().parents[1]
AMOSTRAS = RAIZ / "api" / "static" / "amostras"
TELAS = RAIZ / "docs" / "img" / "telas"
# deslocamentos (graus) em torno da Rua Sílvia; o 2º fica a ~6 m do 1º (vira confirmação)
BASE = (-23.560345, -46.650508)
DESLOC = [(0, 0), (0.00005, 0.00002), (0.0011, -0.0007), (-0.0009, 0.0012), (0.0016, 0.0015), (-0.0014, -0.0013),
          (0.0021, -0.0004), (-0.0004, 0.0024), (0.0027, 0.0009), (-0.0023, 0.0005), (0.0006, -0.0026), (-0.0018, 0.0021),
          (0.0031, -0.0018), (-0.0030, -0.0020)]


def semear(api: str) -> None:
    imgs = sorted(p for p in AMOSTRAS.glob("*.jpg"))
    for k, p in enumerate(imgs):
        r = requests.post(f"{api}/v1/deteccoes", files={"imagem": (p.name, p.read_bytes(), "image/jpeg")}, timeout=30).json()
        if not r["deteccoes"]:
            continue
        d = max(r["deteccoes"], key=lambda x: x["confianca"])
        im = cv2.imread(str(p))
        h, w = im.shape[:2]
        mx, my = 0.25 * (d["x2"] - d["x1"]), 0.25 * (d["y2"] - d["y1"])
        rec = im[int(max(0, d["y1"] - my)):int(min(h, d["y2"] + my)), int(max(0, d["x1"] - mx)):int(min(w, d["x2"] + mx))]
        esc = min(1.0, 480 / max(rec.shape[:2]))
        rec = cv2.resize(rec, None, fx=esc, fy=esc) if esc < 1 else rec
        ok, buf = cv2.imencode(".jpg", rec, [cv2.IMWRITE_JPEG_QUALITY, 80])
        dl = DESLOC[k % len(DESLOC)]
        sev = max((x["severidade"] for x in r["deteccoes"]), key={"baixa": 1, "media": 2, "alta": 3}.get)
        corpo = {"lat": BASE[0] + dl[0], "lon": BASE[1] + dl[1], "severidade": sev, "confianca_max": d["confianca"],
                 "origem": "cidadao" if k % 2 else "camera_veicular", "modelo": r["modelo"],
                 "evidencia_jpg_base64": base64.b64encode(buf.tobytes()).decode()}
        resp = requests.post(f"{api}/v1/relatos", json=corpo, timeout=30).json()
        print(f"{p.name}: {len(r['deteccoes'])} buraco(s) -> {resp['mensagem']}")
    # uma validação humana para a fila mostrar estados diferentes
    rel = requests.get(f"{api}/v1/relatos", timeout=30).json()["relatos"]
    if len(rel) >= 3:
        requests.patch(f"{api}/v1/relatos/{rel[0]['id']}", json={"status": "confirmado", "observacao": "vistoria ok"}, timeout=30)
        requests.patch(f"{api}/v1/relatos/{rel[-1]['id']}", json={"status": "descartado", "observacao": "sombra"}, timeout=30)


def capturar(api: str, amostra: str) -> None:
    TELAS.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        nav = p.chromium.launch(channel="chrome")
        pg = nav.new_page(viewport={"width": 1500, "height": 980}, device_scale_factor=1.25)
        pg.goto(f"{api}/?demo=/static/amostras/{amostra}")
        pg.wait_for_function("document.querySelector('#status').textContent.includes('detectado')", timeout=60000)
        pg.wait_for_timeout(600)
        pg.screenshot(path=str(TELAS / "01_cidadao_servidor.png"))
        pg.goto(f"{api}/?aba=zeladoria")
        pg.wait_for_selector("#fila tr", timeout=30000)
        pg.wait_for_timeout(3500)                               # tiles do mapa
        pg.screenshot(path=str(TELAS / "02_zeladoria.png"))
        pg.goto(f"{api}/?demo=/static/amostras/{amostra}&modo=aparelho")
        try:
            pg.wait_for_function("document.querySelector('#status').textContent.includes('detectado')", timeout=120000)
            pg.wait_for_timeout(600)
            pg.screenshot(path=str(TELAS / "03_cidadao_no_aparelho.png"))
        except Exception as exc:  # noqa: BLE001
            print("modo no aparelho não concluiu:", exc)
        pg.goto(f"{api}/?aba=modelo")
        pg.wait_for_timeout(1500)
        pg.screenshot(path=str(TELAS / "04_modelo.png"))
        nav.close()
    print("Telas salvas em", TELAS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--amostra", default=None, help="arquivo em api/static/amostras usado na tela do cidadão")
    ap.add_argument("--sem-semear", action="store_true")
    args = ap.parse_args()
    if not args.sem_semear:
        semear(args.api)
    amostra = args.amostra or sorted(AMOSTRAS.glob("*.jpg"))[0].name
    capturar(args.api, amostra)


if __name__ == "__main__":
    main()
