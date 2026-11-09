"""
09_gerar_relatorio.py — Gera o relatório técnico (HTML + PDF A4) a partir dos artefatos em resultados/.

Estrutura exigida pela disciplina: título, integrantes, resumo (250 palavras), introdução, objetivo (50 palavras),
explicação técnica das ferramentas, metodologia (passo a passo), exemplos de resultados e conclusão.
O script confere a contagem de palavras do resumo e do objetivo.

Uso:
    python scripts/09_gerar_relatorio.py            # links em configs/links.json
"""
from __future__ import annotations

import csv
import html
import json
import re
import shutil
import subprocess
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "resultados"
DOC = RAIZ / "docs" / "relatorio"
CHROME = [Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
          Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")]


def j(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def br(v: float, casas: int = 1) -> str:
    """Número no formato brasileiro."""
    s = f"{v:,.{casas}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def pct(v: float, casas: int = 1) -> str:
    return br(100 * v, casas) + "%"


def n_palavras(t: str) -> int:
    """Palavras separadas por espaço, como conta um editor de texto."""
    return len(re.sub(r"<[^>]+>", "", t).split())


def tabela(cab: list[str], linhas: list[list], numericas: set[int] = frozenset(), destaque: int | None = None) -> str:
    th = "".join(f'<th class="{"n" if i in numericas else ""}">{c}</th>' for i, c in enumerate(cab))
    corpo = []
    for k, l in enumerate(linhas):
        tds = "".join(f'<td class="{"n" if i in numericas else ""}">{c}</td>' for i, c in enumerate(l))
        corpo.append(f'<tr class="{"destaque" if k == destaque else ""}">{tds}</tr>')
    return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(corpo)}</tbody></table>"


def fig(src: str, legenda: str, classe: str = "") -> str:
    return f'<figure class="{classe}"><img src="{src}" alt="{html.escape(legenda)}"><figcaption>{legenda}</figcaption></figure>'


def main() -> None:
    links = j(RAIZ / "configs" / "links.json") if (RAIZ / "configs" / "links.json").exists() else {}
    repo = links.get("repositorio", "https://github.com/<usuario>/vigiavia-visao-computacional")
    video = links.get("video", "https://youtu.be/<id-do-video>")
    url_mvp = links.get("mvp", "https://<url-do-mvp>")
    rep = j(RES / "cenas_repetidas.json")
    par_val = j(RES / "paridade_onnx_validacao.json")
    ds = j(RES / "dataset_stats.json")["splits"]
    op = j(RAIZ / "configs" / "operacao.json")
    sev = j(RAIZ / "configs" / "severidade.json")
    an = j(RES / "analise_teste.json")
    vel = j(RES / "velocidade.json")
    par = j(RES / "paridade_onnx.json")
    exp = op["experimento"]
    met = j(RES / exp / "metricas.json")
    with open(RES / "comparacao_modelos.csv", encoding="utf-8") as f:
        comp = list(csv.DictReader(f))
    rt = {p.stem: j(p) for p in sorted((RES / "tempo_real").glob("*.json"))}
    testes = (RES / "testes.txt").read_text(encoding="utf-8").strip().splitlines()[-1] if (RES / "testes.txt").exists() else ""
    testes = "pytest -q tests  ->  " + testes.strip("= ")
    import importlib.util
    spec = importlib.util.spec_from_file_location("textos", DOC / "textos.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    textos = mod.TEXTOS

    nome_modelo = exp.split("_", 1)[1].replace("yolo", "YOLO")
    mt = op["metricas_teste"]
    g = an["grupos"]["geral"]
    lin_vel = vel["linhas"]
    fps_gpu = next(l["fps"] for l in lin_vel if l["modelo"] == exp and l["execucao"] == "PyTorch GPU FP16")
    fps_cpu_pt = next(l["fps"] for l in lin_vel if l["modelo"] == exp and l["execucao"] == "PyTorch CPU FP32")
    fps_onnx = next((l["fps"] for l in lin_vel if l["execucao"] == "ONNX Runtime CPU"), None)
    n_tr, n_va, n_te = ds["train"]["imagens"], ds["val"]["imagens"], ds["test"]["imagens"]
    cx_tr, cx_va, cx_te = ds["train"]["caixas_buraco"], ds["val"]["caixas_buraco"], ds["test"]["caixas_buraco"]
    rt_sp = next((v for k, v in rt.items() if k.startswith("sp_")), None)

    m_rep = rep["metricas"]
    com, sem, todas = m_rep["com_trecho_repetido_20"], m_rep["sem_trecho_repetido_20"], m_rep["todas"]
    china = rep["por_pais"]["China_MotorBike"]
    vals = dict(modelo=nome_modelo, map50=br(mt["mAP50"], 3), map5095=br(mt["mAP50_95"], 3), p=pct(g["precisao"], 0),
                r=pct(g["revocacao"], 0), f1=br(g["f1"], 2), conf=br(op["conf"], 3), fps_gpu=br(fps_gpu, 0),
                fps_onnx=br(fps_onnx or 0, 0), n_total=br(n_tr + n_va + n_te, 0), n_tr=br(n_tr, 0),
                n_va=br(n_va, 0), n_te=br(n_te, 0), cx_tr=br(cx_tr, 0),
                sev_media=br(sev["limiar_media"], 3), sev_alta=br(sev["limiar_alta"], 3),
                fps_rt=br(rt_sp["fps_pipeline"], 0) if rt_sp else "—",
                url_mvp=url_mvp, n_rep=br(com["imagens"], 0), pct_rep=pct(com["imagens"] / todas["imagens"]),
                n_sem=br(sem["imagens"], 0), ap_com=br(com["AP50"], 3), ap_sem=br(sem["AP50"], 3),
                ap_todas=br(todas["AP50"], 3), map50_sem=br(sem["AP50"], 2),
                china_rep=china["trecho_repetido_inliers_20"], china_te=china["imagens_teste"])
    resumo = textos["resumo"].format(**vals)
    objetivo = textos["objetivo"].format(**vals)
    np_res, np_obj = n_palavras(resumo), n_palavras(objetivo)
    print(f"Resumo: {np_res} palavras | Objetivo: {np_obj} palavras")
    if not (248 <= np_res <= 252) or not (49 <= np_obj <= 51):
        raise SystemExit("Ajuste os textos em docs/relatorio/textos.py (resumo 250 e objetivo 50 palavras)")

    # ------------------------------------------------------------------ tabelas
    paises = [("Japan", "Japão"), ("India", "Índia"), ("China_MotorBike", "China (moto)"), ("Czech", "Rep. Tcheca"),
              ("United_States", "EUA")]
    tab_dados = tabela(["Split", "Imagens", "Com buraco", "Caixas de buraco"] + [p[1] for p in paises],
                       [[nome, br(ds[s]["imagens"], 0), br(ds[s]["imagens_com_buraco"], 0), br(ds[s]["caixas_buraco"], 0)]
                        + [br(ds[s].get(f"imagens_{p[0]}", 0), 0) for p in paises]
                        for s, nome in (("train", "Treino"), ("val", "Validação"), ("test", "Teste"))],
                       numericas=set(range(1, 9)))
    ordem = sorted(comp, key=lambda c: c["experimento"])
    idx_esc = next(i for i, c in enumerate(ordem) if c["experimento"] == exp)
    tab_comp = tabela(
        ["Experimento", "Parâm. (M)", "Treino (min)", "Val mAP50-95", "Teste mAP50", "Teste mAP50-95",
         "Teste P", "Teste R", "Inferência GPU (ms)", "Limiar F1"],
        [[c["experimento"], br(float(c["parametros_M"]), 2), br(float(c["treino_min"]), 0), br(float(c["val_mAP50_95"]), 3),
          br(float(c["teste_mAP50"]), 3), br(float(c["teste_mAP50_95"]), 3), br(float(c["teste_precisao"]), 3),
          br(float(c["teste_revocacao"]), 3), br(float(c["inferencia_ms_gpu"]), 2), br(float(c["limiar"]), 3)]
         for c in ordem], numericas=set(range(1, 10)), destaque=idx_esc)
    rot_rep = [("todas", "Teste inteiro"), ("com_trecho_repetido_20", "Com trecho repetido (≥ 20 correspondências)"),
               ("sem_trecho_repetido_20", "Sem trecho repetido (< 20)"),
               ("com_trecho_repetido_40", "Com trecho repetido (≥ 40, mais rigoroso)"),
               ("sem_trecho_repetido_40", "Sem trecho repetido (< 40)")]
    tab_rep = tabela(["Imagens de teste", "Imagens", "Buracos", "AP@0,5", "Precisão", "Revocação", "F1"],
                     [[rotulo, br(m_rep[k]["imagens"], 0), br(m_rep[k]["buracos"], 0), br(m_rep[k]["AP50"], 3),
                       pct(m_rep[k]["precisao"]), pct(m_rep[k]["revocacao"]), br(m_rep[k]["f1"], 3)]
                      for k, rotulo in rot_rep], numericas={1, 2, 3, 4, 5, 6}, destaque=2)
    tab_rep_pais = tabela(["País (teste)", "Imagens", "Com trecho repetido (≥ 20)", "Com trecho repetido (≥ 40)"],
                          [[p.replace("_", " "), v["imagens_teste"], v["trecho_repetido_inliers_20"],
                            v["trecho_repetido_inliers_40"]] for p, v in rep["por_pais"].items()], numericas={1, 2, 3})
    grupos = [(k, v) for k, v in an["grupos"].items()]
    tab_vies = tabela(["Subgrupo (teste)", "VP", "FP", "FN", "Precisão", "Revocação", "F1"],
                      [[k, v["tp"], v["fp"], v["fn"], pct(v["precisao"]), pct(v["revocacao"]), br(v["f1"], 3)]
                       for k, v in grupos], numericas={1, 2, 3, 4, 5, 6}, destaque=0)
    tab_tam = tabela(["Tamanho do buraco (COCO)", "Buracos no teste", "Detectados", "Revocação"],
                     [[k, v["buracos"], v["detectados"], pct(v["revocacao"])]
                      for k, v in sorted(an["revocacao_por_tamanho"].items(), key=lambda kv: ("peq" not in kv[0], "méd" not in kv[0]))],
                     numericas={1, 2, 3})
    tab_pais = tabela(["País de origem (teste)", "Imagens", "mAP@0,5", "mAP@0,5:0,95"],
                      [[k.replace("_", " "), v["imagens"], br(v["mAP50"], 3), br(v["mAP50_95"], 3)]
                       for k, v in an["ap_por_pais"].items()], numericas={1, 2, 3})
    tab_vel = tabela(["Modelo", "Execução", "ms/imagem (média)", "ms (p95)", "FPS"],
                     [[l["modelo"], l["execucao"], br(l["ms_media"], 1), br(l["ms_p95"], 1), br(l["fps"], 0)] for l in lin_vel],
                     numericas={2, 3, 4})
    tab_rt = tabela(["Vídeo real (fora do treino)", "Resolução", "Quadros", "Pipeline (ms/quadro)", "FPS do pipeline",
                     "Trilhas", "Buracos confirmados"],
                    [[v["creditos"]["titulo_commons"].replace("File:", ""), v["creditos"]["resolucao"], v["quadros"],
                      br(v["ms_por_quadro_pipeline"], 1), br(v["fps_pipeline"], 0), v["trilhas"], v["buracos_confirmados"]]
                     for v in rt.values()], numericas={2, 3, 4, 5, 6})
    tab_creditos = "".join(
        f'<li>{html.escape(v["creditos"]["titulo_commons"].replace("File:", ""))} — {html.escape(v["creditos"]["autor"])}, '
        f'{v["creditos"]["licenca"]}, Wikimedia Commons: <a href="{v["creditos"]["pagina"]}">{v["creditos"]["pagina"]}</a></li>'
        for v in rt.values())
    hp = j(RES / exp / "metricas.json")["parametros"]
    tab_hp = tabela(["Hiperparâmetro", "Valor", "Justificativa"], [
        ["Resolução de entrada", f'{hp["imgsz"]} px', "as imagens selecionadas têm de 512 a 1.080 px (a maioria 600 e 720 px); "
         "640 px preserva buracos pequenos e cabe na VRAM"],
        ["Épocas / early stopping", f'{hp["epochs"]} / paciência {hp["patience"]}', "transfer learning converge rápido"],
        ["Lote (batch)", hp["batch"], "cabe em 8 GB de VRAM com precisão mista (AMP)"],
        ["Otimizador", textos.get("otimizador", "auto (Ultralytics)"), "escolha automática pelo nº de iterações"],
        ["Taxa de aprendizado", "decaimento cosseno", "convergência estável no fim do treino"],
        ["Aumentos de dados", "mosaic, HSV, escala ±50%, translação 10%, espelho horizontal",
         "variação de luz, distância e posição da câmera veicular"],
        ["Aumentos evitados", "rotação, espelho vertical", "não ocorrem com câmera veicular (céu embaixo)"],
        ["close_mosaic", "10 últimas épocas", "aproxima o treino da distribuição real no final"],
        ["Semente", hp["seed"], "reprodutibilidade (deterministic=True)"],
        ["Cache", "disco (.npy)", "o gargalo medido era a CPU (GPU a ~40%); com cache a GPU passou de 85%"],
    ])

    figura_rt = "".join(
        f'<div><img src="../../resultados/tempo_real/{k}.jpg" alt="Quadro do vídeo {k}"></div>'
        for k in rt if (RES / "tempo_real" / f"{k}.jpg").exists())
    telas = sorted((RAIZ / "docs" / "img" / "telas").glob("*.png"))
    figura_telas = "".join(f'<div><img src="../img/telas/{t.name}" alt="Tela {t.stem}"></div>' for t in telas[:2])
    ods = tabela(["ODS e meta", "Como o VigiaVia contribui", "Indicador do produto"], textos["ods"])

    T = {k: v.format(**vals) if isinstance(v, str) else v for k, v in textos.items()}

    corpo = f"""
<section class="capa">
  <div>
    <div class="topo">FIAP · Global Solution 2026.2 · CitizenTech · Engenharia de Software · 4º ano (4ES)</div>
    <p class="disc">Applied Computer Vision — <em>Enxergando o mundo real</em><br>Professores: Arnaldo Viana e Paulo Rodrigues</p>
  </div>
  <div>
    <h1>VigiaVia</h1>
    <p class="sub">Detecção de buracos em vias públicas em tempo real com YOLO26,<br>rastreamento multiobjeto e realidade aumentada</p>
  </div>
  <div>
    <p><strong>Integrantes</strong></p>
    <table>
      <tr><td>Deivison Pertel</td><td>RM 550803</td></tr>
      <tr><td>Eduardo Akira Murata</td><td>RM 98713</td></tr>
      <tr><td>Wesley Souza de Oliveira</td><td>RM 97874</td></tr>
    </table>
  </div>
  <div class="links">
    <p><strong>MVP publicado (URL pública):</strong> <a href="{url_mvp}">{url_mvp}</a></p>
    <p><strong>Repositório (código e documentação):</strong> <a href="{repo}">{repo}</a></p>
    <p><strong>Vídeo de demonstração:</strong> <a href="{video}">{video}</a></p>
  </div>
  <div class="rodape">São Paulo, novembro de 2026 · versão revisada em 09/11/2026 (correções listadas ao fim da Conclusão)</div>
</section>

<h2 class="quebra">Ficha técnica do MVP</h2>
{tabela(["Item", "VigiaVia"], [
    ["Nome do produto", "VigiaVia — zeladoria inteligente de vias públicas"],
    ["Cidadão e território", T["ficha_cidadao"]],
    ["Problema, em uma frase", T["ficha_problema"]],
    ["Proposta de valor", T["ficha_valor"]],
    ["Arquitetura: componentes", T["ficha_arquitetura"]],
    ["Fontes de dados", T["ficha_dados"]],
    ["Modelos de IA utilizados", f"{nome_modelo} (produção) + YOLO26n/YOLO11n/YOLO26s comparados (Ultralytics, fine-tuning); BoT-SORT (rastreamento, com a associação do ByteTrack); YuNet (borragem de rostos)"],
    ["URL do MVP e repositório", f'MVP no ar: <a href="{url_mvp}">{url_mvp}</a> (publicado pelo componente Produção, <a href="https://github.com/akiramrt/vigiavia-mvp">github.com/akiramrt/vigiavia-mvp</a>) · Repositório deste componente: <a href="{repo}">{repo}</a>'],
    ["Indicadores de impacto", T["ficha_indicadores"]],
    ["ODS conectados", T["ficha_ods"]],
])}

<h2>Resumo</h2>
<p>{resumo}</p>
<p class="contagem">{np_res} palavras</p>
<p><strong>Palavras-chave:</strong> visão computacional; detecção de objetos; YOLO26; rastreamento multiobjeto; zeladoria urbana; CitizenTech.</p>

<h2>1. Introdução</h2>
{T["introducao"]}
{ods}
<p class="pequeno">Tabela 1 — Metas dos ODS a que o VigiaVia se conecta, como contribui e o indicador do produto que acompanha cada uma.</p>

<h2>2. Objetivo</h2>
<div class="caixa"><p>{objetivo}</p></div>
<p class="contagem">{np_obj} palavras</p>

<h2>3. Explicação técnica das ferramentas</h2>
{T["ferramentas"]}
{tabela(["Ferramenta", "Versão", "Papel no projeto", "Licença"], T["tabela_ferramentas"])}
<p class="pequeno">Tabela 2 — Ferramentas, versões, papel no projeto e licenças.</p>

<h2 class="quebra">4. Metodologia (passo a passo)</h2>
{T["metodologia_1"]}
{tab_dados}
<p class="pequeno">Tabela 3 — Dataset VigiaVia derivado da release oficial do RDD2022 (classe D40), dividido 70/15/15 por blocos de 20 números de arquivo consecutivos; essa divisão não separou trajetos (seção 5.3).</p>
{T["metodologia_2"]}
{tab_hp}
<p class="pequeno">Tabela 4 — Configuração de treino (<code>configs/treino.yaml</code>), comum aos três experimentos.</p>
{T["metodologia_3"]}

<h2 class="quebra">5. Exemplos de resultados</h2>
<div class="kpis">
  <div class="kpi"><b>{vals["map50"]}</b><span>mAP@0,5 no teste ({nome_modelo})</span></div>
  <div class="kpi"><b>{vals["p"]} / {vals["r"]}</b><span>precisão / revocação no limiar {vals["conf"]}</span></div>
  <div class="kpi"><b>{vals["fps_gpu"]} FPS</b><span>inferência em GPU (RTX 3070, FP16)</span></div>
  <div class="kpi"><b>{vals["fps_onnx"]} FPS</b><span>ONNX Runtime em CPU (Ryzen 9 5900X)</span></div>
</div>
<h3>5.1 Treinamento e comparação dos modelos</h3>
{fig("../../resultados/curvas_treino.png", "Figura 1 — Curvas de treino: mAP de validação por época e perda de caixa.")}
{tab_comp}
<p class="pequeno">Tabela 5 — Comparação dos experimentos (linha destacada: modelo escolhido para produção). "Val" = validação; as demais métricas são do conjunto de teste; inferência medida pelo Ultralytics no teste (lote 16); limiar que maximiza o F1 na validação.</p>
{T["res_comparacao"]}
{fig("../../resultados/comparacao_modelos.png", "Figura 2 — mAP no teste × tamanho e latência de cada modelo.")}

<h3>5.2 Desempenho do modelo escolhido no conjunto de teste</h3>
{T["res_teste"]}
<figure class="dupla"><div><img src="../../resultados/{exp}/teste_BoxPR_curve.png" alt="Curva precisão-revocação"></div>
<div><img src="../../resultados/{exp}/teste_BoxF1_curve.png" alt="Curva F1 por limiar"></div>
<figcaption>Figura 3 — Curva precisão × revocação (esq.) e F1 por limiar de confiança (dir.) no teste.</figcaption></figure>

<h3>5.3 Trechos de via repetidos entre treino e teste</h3>
{T["res_repetidos"]}
{fig("../../resultados/cenas_repetidas.jpg", "Figura 4 — Exemplos de imagens de teste (esq. de cada par) que mostram o mesmo trecho de via de uma imagem de treino (dir.), encontrados por correspondência geométrica (scripts/03b_cenas_repetidas.py).", "larga")}
{tab_rep}
<p class="pequeno">Tabela 6 — YOLO26n no teste com e sem as imagens que repetem um trecho de via do treino. AP@0,5 com interpolação de 101 pontos (por isso 0,583 no teste inteiro, contra 0,585 do Ultralytics); precisão, revocação e F1 no limiar {vals["conf"]}. Linha destacada: a estimativa para ruas novas.</p>
{tab_rep_pais}
<p class="pequeno">Tabela 7 — Imagens de teste com trecho de via repetido do treino, por país (<code>resultados/cenas_repetidas.json</code>).</p>

<h3>5.4 Exemplos de acertos e de erros</h3>
{fig("../../resultados/exemplos_acertos.jpg", "Figura 5 — Acertos no teste (verde: buraco anotado detectado; azul: predição correta).", "larga")}
{fig("../../resultados/exemplos_falsos_positivos.jpg", "Figura 6 — Falsos positivos (vermelho) de maior confiança.", "larga")}
{fig("../../resultados/exemplos_falsos_negativos.jpg", "Figura 7 — Falsos negativos (magenta: buraco anotado não detectado).", "larga")}
{T["res_erros"]}

<h3>5.5 IA responsável: desempenho por subgrupo</h3>
{tab_vies}
<p class="pequeno">Tabela 8 — Acertos (VP), falsos positivos (FP), buracos perdidos (FN), precisão, revocação e F1 no limiar {vals["conf"]}, por país de origem e por iluminação (tercis do brilho médio).</p>
{tab_tam}
<p class="pequeno">Tabela 9 — Revocação por tamanho do buraco (critério COCO), no limiar {vals["conf"]}.</p>
{tab_pais}
<p class="pequeno">Tabela 10 — AP por país de origem das imagens de teste (a China tem trechos repetidos do treino: ver seção 5.3).</p>
{fig("../../resultados/vieses_subgrupos.png", "Figura 8 — Precisão e revocação no limiar de operação por país e iluminação (tercis de brilho).")}
{T["res_vieses"]}

<h3>5.6 Tempo real, velocidade e portabilidade</h3>
{tab_vel}
<p class="pequeno">Tabela 11 — Latência de inferência (imagem 640 px, sem pré/pós-processamento de vídeo). ONNX: modelo de produção. Paridade PyTorch × ONNX no teste: mAP@0,5 {br(par["pytorch_gpu"]["mAP50"], 3)} × {br(par["onnx_cpu"]["mAP50"], 3)}; na validação: {br(par_val["pytorch_gpu"]["mAP50"], 3)} × {br(par_val["onnx_cpu_com_nms"]["mAP50"], 3)} (variante fim a fim: {br(par_val["onnx_cpu_fim_a_fim"]["mAP50"], 3)}).</p>
{tab_rt}
<p class="pequeno">Tabela 12 — Pipeline completo em vídeos reais (decodificação + detecção + BoT-SORT + borragem de rostos + HUD) na RTX 3070. "Buracos confirmados" são eventos do rastreador, e não buracos verificados: parte deles são trincas e remendos.</p>
<figure class="grade2">{figura_rt}<figcaption>Figura 9 — Quadros do HUD de realidade aumentada nos vídeos reais (fora do treino); algumas caixas marcam trincas, não buracos.</figcaption></figure>
{T["res_tempo_real"]}

<h3>5.7 Integração com o MVP</h3>
{T["res_integracao"]}
{f'<figure class="pilha">{figura_telas}<figcaption>Figura 10 — Interface web publicada: cidadão (em cima: detecção numa imagem de exemplo e mapa para marcar o local do relato) e zeladoria (embaixo: mapa e fila priorizada com relatos de demonstração).</figcaption></figure>' if figura_telas else ""}
<pre>{html.escape(testes)}</pre>

<h2 class="quebra">6. Conclusão</h2>
{T["conclusao"]}

<h2>Referências</h2>
<ol class="pequeno">{T["referencias"]}{tab_creditos}</ol>

<h2>Apêndice — Como reproduzir</h2>
<pre>{html.escape(T["reproduzir"])}</pre>
"""
    doc = f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<title>VigiaVia — Relatório Applied Computer Vision</title><link rel="stylesheet" href="relatorio.css"></head>
<body>{corpo}</body></html>"""
    DOC.mkdir(parents=True, exist_ok=True)
    arq_html = DOC / "relatorio.html"
    arq_html.write_text(doc, encoding="utf-8")
    chrome = next((c for c in CHROME if c.exists()), None) or shutil.which("chrome")
    pdf = DOC / "VigiaVia_Relatorio_Applied_Computer_Vision.pdf"
    subprocess.run([str(chrome), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--user-data-dir={DOC / '.chrome'}", f"--print-to-pdf={pdf}", arq_html.resolve().as_uri()],
                   check=True, capture_output=True)
    shutil.rmtree(DOC / ".chrome", ignore_errors=True)
    print(f"PDF gerado: {pdf} ({pdf.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
