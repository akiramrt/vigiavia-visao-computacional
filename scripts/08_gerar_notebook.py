"""
08_gerar_notebook.py — Gera e EXECUTA notebooks/VigiaVia_Visao_Computacional.ipynb (com saídas salvas).

O notebook não retreina (o treino completo leva horas e é feito por scripts/02_treinar.py):
ele carrega os artefatos produzidos pelos scripts e roda demonstrações ao vivo.

Uso:
    VIGIAVIA_DATA=datasets/vigiavia-rdd2022 VIGIAVIA_VIDEOS=demo/videos_fonte python scripts/08_gerar_notebook.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "notebooks" / "VigiaVia_Visao_Computacional.ipynb"

md = nbformat.v4.new_markdown_cell
py = nbformat.v4.new_code_cell

CELULAS = [
    md("""# VigiaVia — Detecção de buracos em vias públicas com visão computacional

**FIAP · Global Solution 2026.2 · CitizenTech · Applied Computer Vision** (Profs. Arnaldo Viana e Paulo Rodrigues)

| Integrante | RM |
|---|---|
| Deivison Pertel | 550803 |
| Eduardo Akira Murata | 98713 |
| Wesley Souza de Oliveira | 97874 |

Este notebook documenta o componente de **Visão** do MVP VigiaVia: dados → treino customizado (transfer learning) →
avaliação → análise de vieses → tempo real (rastreamento + realidade aumentada) → exportação ONNX → integração via API.
Os passos pesados (download do dataset e treino) são executados pelos scripts em `scripts/`; aqui carregamos os
artefatos gerados e rodamos demonstrações ao vivo, com as saídas salvas."""),
    py("""import json, os, sys, platform
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch, ultralytics
import matplotlib.pyplot as plt
from IPython.display import Image, display, Markdown

RAIZ = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(RAIZ / "src"))
DATA = Path(os.getenv("VIGIAVIA_DATA", RAIZ / "datasets" / "vigiavia-rdd2022"))
RES = RAIZ / "resultados"
print("Python", platform.python_version(), "| PyTorch", torch.__version__, "| Ultralytics", ultralytics.__version__,
      "| OpenCV", cv2.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "não disponível")
print("Dataset:", DATA.exists() and "ok", "| imagens de teste:", len(list((DATA / "images" / "test").glob("*.jpg"))))"""),
    md("""## 1. Dados — RDD2022 oficial (classe D40, "buraco")

* **Fonte:** RDD2022 — *Multi-National Road Damage Dataset* (Arya et al., 2022), release oficial do CRDDC'2022
  (figshare, doi:10.6084/m9.figshare.21431547), anotações Pascal VOC com o **nome** da classe.
* **Auditoria de rótulos** (`scripts/01b_auditar_rotulos.py`): uma redistribuição em YOLO no Hugging Face rotulava
  como "pothole" faixas apagadas (D44), faixas de pedestre (D43) e bueiros (D50) — os buracos (D40) tinham sido
  descartados. Detectamos isso inspecionando recortes antes de treinar e trocamos para a release oficial.
* **Seleção** (`scripts/01_preparar_dataset.py`): imagens de nível de rua de Japão, Índia, China (moto), República
  Tcheca e EUA; todas as imagens com buraco + 25% de imagens **sem** buraco como negativas; split 70/15/15 em
  **blocos de 20 quadros consecutivos** (evita quadros quase idênticos em treino e teste). Só os trechos necessários
  do zip de 13 GB são baixados (HTTP Range)."""),
    py("""display(Image(filename=str(RES / "auditoria_rotulos_redistribuicao_hf.jpg"), width=900))
display(Markdown("**Acima:** rótulos 'pothole' da redistribuição (incorretos). **Abaixo:** classe D40 da release oficial."))
display(Image(filename=str(RES / "auditoria_rotulos_oficial.jpg"), width=900))"""),
    py("""stats = json.loads((RES / "dataset_stats.json").read_text(encoding="utf-8"))
man = pd.read_csv(DATA / "manifest.csv")
tab = man.assign(com_buraco=man.n_buracos > 0).groupby(["split", "pais"]).agg(
    imagens=("arquivo", "count"), com_buraco=("com_buraco", "sum"), buracos=("n_buracos", "sum"))
display(tab)
print("Totais:", {k: v["imagens"] for k, v in stats["splits"].items()}, "| caixas de buraco:",
      sum(v["caixas_buraco"] for v in stats["splits"].values()))"""),
    py("""# Tamanho relativo das caixas (fração da imagem) — buracos são, em geral, objetos pequenos
areas = []
for lbl in (DATA / "labels" / "train").glob("*.txt"):
    for l in lbl.read_text().splitlines():
        if l.strip():
            _, _, _, w, h = map(float, l.split()); areas.append(w * h)
areas = np.array(areas)
fig, ax = plt.subplots(figsize=(7, 3))
ax.hist(np.log10(areas), bins=60, color="#2a6fdb")
ax.set_xlabel("log10(área da caixa / área da imagem)"); ax.set_ylabel("caixas (treino)")
ax.set_title(f"{len(areas)} buracos no treino · mediana = {np.median(areas)*100:.2f}% da imagem")
plt.show()"""),
    py("""# Amostra do treino com as anotações (caixas verdes)
amostra = sorted((DATA / "images" / "train").glob("*.jpg"))[::350][:8]
fig, axs = plt.subplots(2, 4, figsize=(16, 8))
for ax, p in zip(axs.ravel(), amostra):
    im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB); h, w = im.shape[:2]
    for l in (DATA / "labels" / "train" / (p.stem + ".txt")).read_text().splitlines():
        _, cx, cy, bw, bh = map(float, l.split())
        cv2.rectangle(im, (int((cx-bw/2)*w), int((cy-bh/2)*h)), (int((cx+bw/2)*w), int((cy+bh/2)*h)), (0, 255, 0), 3)
    ax.imshow(im); ax.set_title(p.stem); ax.axis("off")
plt.tight_layout(); plt.show()"""),
    md("""## 2. Treinamento customizado (transfer learning)

Pesos pré-treinados no COCO → fine-tuning no dataset VigiaVia. Hiperparâmetros em `configs/treino.yaml`
(imagem 640 px, até 60 épocas com *early stopping*, AMP, cache em disco, aumentos de dados plausíveis para câmera
veicular — sem rotação nem espelhamento vertical). Comando:

```bash
python scripts/02_treinar.py --data datasets/vigiavia-rdd2022/data.yaml --exp E1_yolo26n E2_yolo11n E3_yolo26s
```"""),
    py("""print((RAIZ / "configs" / "treino.yaml").read_text(encoding="utf-8"))
display(Image(filename=str(RES / "curvas_treino.png"), width=1000))"""),
    md("## 3. Comparação dos modelos e escolha do modelo de produção"),
    py("""comp = pd.read_csv(RES / "comparacao_modelos.csv")
display(comp)
op = json.loads((RAIZ / "configs" / "operacao.json").read_text(encoding="utf-8"))
display(Markdown(f"**Modelo escolhido:** `{op['experimento']}` · limiar de confiança **{op['conf']:.3f}** "
                 f"(máximo F1 na validação) · regra: {op['regra_escolha']}"))
display(Image(filename=str(RES / "comparacao_modelos.png"), width=800))"""),
    md("## 4. Avaliação no conjunto de teste (imagens nunca vistas no treino nem na escolha do limiar)"),
    py("""met = json.loads((RES / op["experimento"] / "metricas.json").read_text(encoding="utf-8"))
display(pd.DataFrame({k: met[k] for k in ("val", "test")}).drop(index=["velocidade_ms_por_imagem"], errors="ignore"))
for nome in ("teste_BoxPR_curve.png", "teste_BoxF1_curve.png", "teste_confusion_matrix_normalized.png"):
    f = RES / op["experimento"] / nome
    if f.exists():
        display(Markdown(f"**{nome}**")); display(Image(filename=str(f), width=620))"""),
    md("""## 5. IA responsável: vieses e análise de erros

Precisão/revocação **no limiar de operação** por subgrupo (país de origem, iluminação em tercis de brilho) e revocação
por tamanho do buraco (critério COCO). Verde = buraco real detectado; azul = predição correta; vermelho = falso
positivo; magenta = buraco perdido."""),
    py("""an = json.loads((RES / "analise_teste.json").read_text(encoding="utf-8"))
display(pd.DataFrame(an["grupos"]).T)
display(pd.DataFrame(an["revocacao_por_tamanho"]).T)
display(pd.DataFrame(an["ap_por_pais"]).T)
display(Image(filename=str(RES / "vieses_subgrupos.png"), width=850))
for n in ("acertos", "falsos_positivos", "falsos_negativos"):
    display(Markdown(f"**{n.replace('_', ' ')}**")); display(Image(filename=str(RES / f"exemplos_{n}.jpg"), width=1000))"""),
    md("""## 6. Severidade (proxy) e inferência no modelo de produção

Índice = área relativa da caixa corrigida pela perspectiva de uma via plana; limiares = tercis no treino
(`configs/severidade.json`). É uma **priorização relativa**, não uma medida física."""),
    py("""from vigiavia.detector import DetectorBuracos
from vigiavia import hud, config
print(json.loads((RAIZ / "configs" / "severidade.json").read_text(encoding="utf-8")))
display(Image(filename=str(RES / "calibracao_severidade.png"), width=700))
det = DetectorBuracos(config.pesos_padrao(), conf=config.conf_padrao())
com_buraco = set(man[(man.split == "test") & (man.n_buracos > 0)].arquivo)
testes = [p for p in sorted((DATA / "images" / "test").glob("*.jpg")) if p.name in com_buraco][::80][:6]
fig, axs = plt.subplots(2, 3, figsize=(16, 10))
for ax, p in zip(axs.ravel(), testes):
    im = cv2.imread(str(p)); d = det.detectar(im)
    vis = hud.desenhar(im, d, det.conf, 0, len(d), det.nome, mostrar_zona=False)
    ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); ax.set_title(f"{p.stem}: {len(d)} buraco(s)"); ax.axis("off")
plt.tight_layout(); plt.show()"""),
    md("""## 7. Tempo real: detecção + ByteTrack + HUD de realidade aumentada

Vídeo real de São Paulo (Rua Sílvia, Bela Vista — Daniel Fraga, CC BY 3.0, Wikimedia Commons), **nunca visto no
treino**. Cada buraco recebe um ID; um evento só é emitido quando a trilha é confirmada (≥ 3 quadros e confiança ≥
limiar)."""),
    py("""import time
video = next((RAIZ / "demo" / "videos_fonte").glob("sp_*"), None) or next(Path(os.getenv("VIGIAVIA_VIDEOS", ".")).glob("sp_*"))
cap = cv2.VideoCapture(str(video)); det.reiniciar_rastreamento()
quadros, tempos, ids = [], [], set()
for i in range(240):
    ok, q = cap.read()
    if not ok: break
    t0 = time.perf_counter(); d = det.rastrear(q); tempos.append(time.perf_counter() - t0)
    ids |= {x.track_id for x in d if x.track_id is not None}
    if i in (60, 140, 220): quadros.append(hud.desenhar(q, d, det.conf, 1 / np.mean(tempos[-30:]), len(ids), det.nome))
cap.release()
print(f"{len(tempos)} quadros · {1000*np.mean(tempos[5:]):.1f} ms/quadro (detecção+rastreamento) · trilhas: {len(ids)}")
fig, axs = plt.subplots(1, 3, figsize=(18, 4.5))
for ax, q in zip(axs, quadros): ax.imshow(cv2.cvtColor(q, cv2.COLOR_BGR2RGB)); ax.axis("off")
plt.tight_layout(); plt.show()
for f in sorted((RES / "tempo_real").glob("*.json")):
    r = json.loads(f.read_text(encoding="utf-8")); r.pop("creditos", None); print(f.stem, r)"""),
    md("## 8. Velocidade e exportação ONNX (CPU, celular e navegador)"),
    py("""display(pd.read_csv(RES / "velocidade.csv"))
display(Image(filename=str(RES / "velocidade.png"), width=850))
print("Paridade PyTorch x ONNX:", json.loads((RES / "paridade_onnx.json").read_text()))
import onnxruntime as ort
s = ort.InferenceSession(str(RAIZ / "modelos" / "vigiavia_buracos.onnx"), providers=["CPUExecutionProvider"])
print("Entrada:", s.get_inputs()[0].name, s.get_inputs()[0].shape, "| Saída:", s.get_outputs()[0].name, s.get_outputs()[0].shape)"""),
    md("""## 9. Integração com o MVP (API FastAPI)

Contrato: `POST /v1/deteccoes` (imagem → caixas, nada é armazenado) e `POST /v1/relatos` (evento → fila da zeladoria,
com deduplicação geográfica de 15 m). Abaixo, a API é exercitada em processo com o `TestClient`."""),
    py("""import base64, tempfile
os.environ["VIGIAVIA_DB"] = str(Path(tempfile.mkdtemp()) / "demo.db")
from fastapi.testclient import TestClient
sys.path.insert(0, str(RAIZ)); import api.main as apimain
apimain.BANCO = Path(os.environ["VIGIAVIA_DB"])
with TestClient(apimain.app) as c:
    print(c.get("/health").json())
    img = testes[0]
    r = c.post("/v1/deteccoes", files={"imagem": (img.name, img.read_bytes(), "image/jpeg")}).json()
    print("Detecções:", r["quantidade"], "| inferência:", r["tempo_inferencia_ms"], "ms"); display(pd.DataFrame(r["deteccoes"]))
    corpo = {"lat": -23.560345, "lon": -46.650508, "severidade": "media", "confianca_max": 0.81, "origem": "cidadao"}
    print(c.post("/v1/relatos", json=corpo).json()["mensagem"])
    print(c.post("/v1/relatos", json={**corpo, "lat": -23.560390, "severidade": "alta"}).json()["mensagem"])
    display(pd.DataFrame(c.get("/v1/relatos").json()["relatos"])[["severidade", "confirmacoes", "prioridade", "status"]])"""),
    md("""## Conclusões

* O detector treinado atende ao requisito de **tempo real** em GPU e em CPU (ONNX), e a exportação ONNX preserva a
  precisão — o mesmo modelo roda no servidor, no notebook e no navegador do celular.
* O rastreamento converte centenas de quadros em **um relato por buraco**, e a API junta relatos próximos: isso ataca
  diretamente o problema dos chamados duplicados relatado pela Prefeitura de São Paulo.
* Os principais limites são **buracos pequenos/distantes** e o **viés de domínio** (dados de Japão/Índia/China).
  Próximo passo: coletar imagens brasileiras pelo próprio app (com consentimento) e retreinar, mantendo a validação
  humana antes de qualquer ordem de serviço."""),
]


def main() -> None:
    nb = nbformat.v4.new_notebook()
    nb.cells = CELULAS
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    cliente = NotebookClient(nb, timeout=1200, kernel_name="python3",
                             resources={"metadata": {"path": str(RAIZ / "notebooks")}})
    cliente.execute()
    DESTINO.parent.mkdir(exist_ok=True)
    nbformat.write(nb, DESTINO)
    print(f"Notebook executado e salvo em {DESTINO}")


if __name__ == "__main__":
    main()
