"""Textos do relatório (HTML). Placeholders entre chaves são preenchidos por scripts/09_gerar_relatorio.py.

Revisão de 09/11/2026: trechos de via repetidos entre treino e teste (seção 5.3), privacidade descrita por modo de uso,
rastreador BoT-SORT, URL do MVP, dados simulados declarados, ODS explicados e números conferidos com os artefatos.
"""

TEXTOS = {
    # ------------------------------------------------------------------ resumo (250 palavras, contadas por espaços)
    "resumo": (
        "Buracos em vias públicas provocam quedas de motociclistas e ciclistas, danos a veículos e aumento do custo do "
        "transporte. Em São Paulo, foram 113,1 mil pedidos de tapa-buraco em 2025, e a prefeitura reconhece que um mesmo "
        "ponto gera vários chamados. Este trabalho apresenta o VigiaVia, componente de visão computacional de uma "
        "CitizenTech de zeladoria urbana, que detecta buracos em tempo real com câmeras de celulares e de veículos de "
        "serviço e converte cada detecção em um relato georreferenciado e priorizado. Realizamos o treinamento "
        "customizado, por transferência de aprendizado, de detectores YOLO com {n_total} imagens de rua de cinco países "
        "do dataset público RDD2022; uma auditoria visual revelou rótulos trocados numa redistribuição, e adotamos as "
        "anotações oficiais. Três arquiteturas foram comparadas, e o modelo de produção, o {modelo}, foi escolhido por "
        "uma regra explícita entre precisão e latência. No teste, ele alcançou mAP@0,5 de {map50}, com precisão de {p} "
        "e revocação de {r} no limiar que maximiza o F1, processando {fps_gpu} quadros por segundo em GPU e {fps_onnx} "
        "em CPU. Uma verificação geométrica mostrou que {n_rep} imagens de teste repetem trechos de via do treino; sem "
        "elas, o mAP@0,5 cai para cerca de {map50_sem}, a estimativa mais honesta para ruas novas. Em vídeo, o "
        "rastreador BoT-SORT agrupa as detecções de cada buraco em eventos, e uma interface de realidade aumentada "
        "alerta o condutor. A fila da zeladoria recebe só o recorte da via e as coordenadas, e uma pessoa valida cada "
        "relato. A análise de vieses orienta os próximos passos."
    ),
    # ------------------------------------------------------------------ objetivo (50 palavras)
    "objetivo": (
        "Treinar e validar um detector de buracos, baseado em YOLO, que opere em tempo real em celulares e câmeras "
        "veiculares, e integrá-lo ao MVP VigiaVia para transformar imagens de vias públicas em relatos "
        "georreferenciados, deduplicados e priorizados, ajudando a zeladoria a reparar buracos mais rápido, com "
        "privacidade e supervisão humana."
    ),
    # ------------------------------------------------------------------ ficha técnica
    "ficha_cidadao": "Quem circula pelas vias — motociclistas (inclusive entregadores), ciclistas, pedestres, motoristas e "
                     "usuários de ônibus — e os agentes de zeladoria das subprefeituras. Território piloto: centro de São "
                     "Paulo (Bela Vista); replicável em cidades médias e em estradas vicinais.",
    "ficha_problema": "Buracos nas vias causam acidentes e custos, e sua identificação depende de chamados manuais, lentos, "
                      "imprecisos e duplicados.",
    "ficha_valor": "Detectar buracos automaticamente, em tempo real, com as câmeras que já circulam pela cidade (celulares, "
                   "ônibus, coleta de lixo), gerando um relato georreferenciado e priorizado, com relatos do mesmo buraco "
                   "agrupados — com privacidade e validação humana.",
    "ficha_arquitetura": "Visão (este trabalho: YOLO + rastreamento BoT-SORT + HUD de realidade aumentada + API); Dados "
                         "(lakehouse Bronze/Prata/Ouro); Inteligência (assistente GenAI); Automação (RPA de chamados); "
                         "Produção (container, CI/CD, monitoramento); Físico (robô do Maker Lab); Negócio (modelo B2G e pitch).",
    "ficha_dados": "RDD2022, release oficial do CRDDC'2022 (figshare; CC BY 4.0 / CC BY-SA 4.0) para treino, validação "
                   "e teste; vídeos reais do Wikimedia Commons (CC BY / CC BY-SA) para teste fora da distribuição; "
                   "relatos enviados pelo próprio app. Os relatos das telas (Figura 10) são de demonstração: imagens reais "
                   "analisadas pela API, com coordenadas simuladas em torno da Rua Sílvia (Bela Vista).",
    "ficha_indicadores": "Tempo entre detecção e reparo; % de chamados duplicados evitados; buracos detectados por km "
                         "rodado; precisão das detecções confirmadas pela zeladoria; quedas e sinistros em vias monitoradas.",
    "ficha_ods": "ODS 11 (metas 11.2 e 11.3), ODS 9 (meta 9.1) e ODS 3 (meta 3.6); a contribuição para cada meta está na "
                 "Tabela 1.",
    # ------------------------------------------------------------------ introdução
    "introducao": """
<p>As vias públicas são a infraestrutura mais usada do cotidiano urbano: por elas circulam ônibus, entregadores,
ciclistas, pedestres e serviços de emergência. Quando o pavimento se degrada, o buraco deixa de ser um incômodo e passa a
ser risco — sobretudo para motociclistas e ciclistas, para quem uma roda presa significa queda — e custo: segundo a
Pesquisa CNT de Rodovias 2025, a má qualidade do pavimento eleva em média 31,2% o custo operacional do transporte
rodoviário, e 80% dos 2.146 pontos críticos identificados nas rodovias avaliadas são buracos [3].</p>
<p>Nas cidades, o problema aparece como fila de chamados. Em São Paulo, a Secretaria Municipal das Subprefeituras
registrou 177,3 mil pedidos de tapa-buraco em 2023 e 113,1 mil em 2025; no primeiro semestre de 2026 foram quase 52 mil,
uma média de 287 por dia. A própria prefeitura ressalta que "um mesmo ponto pode gerar vários chamados" [4]. Hoje o
fluxo depende de um cidadão que pare, descreva o endereço e registre a solicitação; a informação chega sem padrão de
imagem, com localização aproximada e muitas vezes duplicada, e a equipe de vistoria precisa ir a campo para confirmar e
dimensionar o problema antes de programar o reparo.</p>
<p>Ao mesmo tempo, a cidade já está cheia de câmeras em movimento: celulares, câmeras de ônibus, de veículos de coleta de
lixo e de frotas de entrega percorrem diariamente as mesmas ruas que precisam de manutenção. A visão computacional
permite transformar esse fluxo de imagens em dados estruturados — onde está cada buraco, qual sua gravidade relativa e
quantas vezes foi observado — sem depender de alguém parar para registrar.</p>
<p>Este documento apresenta o componente <strong>Visão</strong> do VigiaVia, a CitizenTech desenvolvida pelo grupo para a
Global Solution 2026.2. O componente realiza o treinamento customizado de um detector de objetos da família YOLO para
reconhecer buracos em imagens de rua, executa a detecção em tempo real com rastreamento multiobjeto e uma interface de
realidade aumentada, e entrega ao restante do MVP — API de relatos, fila da zeladoria, pipeline de dados, automação e
robô do Maker Lab — eventos georreferenciados e com privacidade preservada. O trabalho segue os princípios do desafio:
dados reais e abertos, IA responsável com pessoa no circuito, LGPD desde o projeto, resiliência à baixa conectividade e
reprodutibilidade.</p>
<h3>1.1 Conexão com os Objetivos de Desenvolvimento Sustentável</h3>
<p>O VigiaVia se conecta a quatro metas da Agenda 2030 [10]. A Tabela 1 mostra como o componente contribui para cada
uma e qual indicador do produto acompanha essa contribuição.</p>
""",
    "ods": [
        ["ODS 11 — meta 11.2: transporte seguro e acessível para todos",
         "Vias sem buracos reduzem quedas de motociclistas e ciclistas e mantêm ônibus e pedestres circulando com segurança; "
         "o alerta de realidade aumentada avisa o condutor antes do buraco.",
         "Tempo entre detecção e reparo; buracos abertos por km"],
        ["ODS 11 — meta 11.3: planejamento e gestão participativos",
         "O cidadão relata com a câmera do celular, sem cadastro, e a fila pública prioriza o que a população confirma.",
         "Relatos e confirmações por bairro; % de relatos validados"],
        ["ODS 9 — meta 9.1: infraestrutura de qualidade, confiável e resiliente",
         "A detecção precoce leva ao reparo antes de o buraco crescer e de a base do pavimento se degradar.",
         "Buracos reparados por mês; reincidência no mesmo ponto"],
        ["ODS 3 — meta 3.6: reduzir mortes e lesões no trânsito",
         "Buraco é causa direta de queda de moto e bicicleta; reparar mais rápido e avisar o condutor ataca essa causa.",
         "Sinistros com motos e bicicletas nas vias monitoradas"],
    ],
    # ------------------------------------------------------------------ ferramentas
    "ferramentas": """
<h3>3.1 Detecção de objetos e métricas</h3>
<p>Detecção de objetos combina localização (uma caixa delimitadora) e classificação — aqui, uma única classe:
<em>buraco</em>. Uma predição é correta quando a interseção sobre a união (IoU) entre a caixa prevista e a anotada é de
pelo menos 0,5. A partir daí medimos <strong>precisão</strong> (fração das detecções que são buracos de verdade),
<strong>revocação</strong> (fração dos buracos existentes que foram encontrados) e <strong>F1</strong> (média harmônica
das duas). A <em>average precision</em> (AP) é a área sob a curva precisão × revocação; reportamos o mAP@0,5 (IoU 0,5) e
o mAP@0,5:0,95 (média para IoU de 0,5 a 0,95, padrão COCO), mais exigente quanto ao ajuste da caixa.</p>
<h3>3.2 YOLO e a arquitetura YOLO26</h3>
<p>Os detectores YOLO (<em>You Only Look Once</em>) são de estágio único: uma rede convolucional com <em>backbone</em>
(extração de características), <em>neck</em> (fusão multiescala, que ajuda a detectar objetos pequenos e grandes) e
<em>head</em> (caixas e classes) processa a imagem inteira em uma única passada, o que viabiliza tempo real. O YOLO26
(Ultralytics, 2026) é a geração mais recente e traz quatro mudanças relevantes para este problema [2]: (i) cabeça
um-para-um, que permite inferência fim a fim sem NMS (supressão de não-máximos); (ii) remoção da <em>Distribution Focal
Loss</em>, o que simplifica a exportação para dispositivos de borda; (iii) <em>ProgLoss</em> e <em>STAL</em> (atribuição
de rótulos sensível a alvos pequenos), que favorecem objetos pequenos — exatamente o caso de buracos vistos à distância; e
(iv) o otimizador MuSGD — no nosso treino, porém, o Ultralytics escolheu automaticamente o AdamW, adequado ao número de
iterações (Tabela 4). No COCO, o YOLO26n tem 2,4 milhões de parâmetros e 40,9 de mAP@0,5:0,95; o YOLO26s, 9,5 milhões e
48,6. Como linha de base usamos o YOLO11n [8], da geração anterior (blocos C3k2 e atenção C2PSA, com NMS).</p>
<h3>3.3 Transferência de aprendizado</h3>
<p>Em vez de treinar do zero, partimos de pesos pré-treinados no COCO [6] (118 mil imagens, 80 classes), que já
codificam bordas, texturas e formas. A cabeça é substituída por uma de uma classe e toda a rede é ajustada
(<em>fine-tuning</em>) com o dataset de buracos. Isso reduz drasticamente a quantidade de dados e de GPU necessária: com
menos de 3 mil imagens de treino, cada modelo convergiu em 33 a 43 minutos numa GPU de mesa.</p>
<h3>3.4 Ultralytics, PyTorch e CUDA</h3>
<p>O treino usa a biblioteca Ultralytics 8.4 sobre PyTorch 2.11 com CUDA 12.8, numa GPU NVIDIA RTX 3070 (8 GB), com
precisão mista automática (AMP/FP16), aumentos de dados em tempo de treino (mosaico de quatro imagens, variação de
matiz/saturação/brilho, escala, translação e espelhamento horizontal), <em>early stopping</em>, decaimento cosseno da taxa
de aprendizado e cache das imagens decodificadas em disco.</p>
<h3>3.5 Dataset RDD2022</h3>
<p>O RDD2022 [1] reúne 47.420 imagens de vias de seis países, capturadas por smartphones e câmeras montadas em veículos,
anotadas com quatro tipos de dano avaliados — trincas longitudinais (D00), transversais (D10), couro-de-jacaré (D20) e
buracos (D40) — além de classes auxiliares, como faixa de pedestre apagada (D43), faixa de pista apagada (D44) e tampa
de bueiro (D50). É o dataset do <em>Crowdsensing-based Road Damage Detection Challenge</em> (CRDDC'2022) e corresponde ao
cenário de uso do VigiaVia: câmera de celular em um veículo. Usamos a release oficial (figshare), cujas anotações Pascal
VOC trazem o <strong>nome</strong> de cada classe.</p>
<h3>3.6 Rastreamento multiobjeto com BoT-SORT</h3>
<p>Em vídeo, o mesmo buraco aparece em dezenas ou centenas de quadros. O rastreador associa as detecções de quadros
consecutivos: um filtro de Kalman prevê o movimento de cada trilha e o algoritmo húngaro casa previsões e detecções por
IoU, em duas etapas — a segunda aproveita detecções de baixa confiança, ideia do ByteTrack [5] —, o que mantém o
identificador mesmo com borrão de movimento ou oclusão parcial. Usamos o BoT-SORT [11], que acrescenta a
<strong>compensação global de movimento da câmera</strong>: com a câmera na mão ou num veículo que balança, ela evita que o
mesmo buraco ganhe identificadores novos (seção 5.6). No VigiaVia, cada trilha confirmada (≥ 5 quadros e confiança ≥
limiar) gera um evento, e uma deduplicação espaço-temporal junta trilhas que se confirmam no mesmo lugar, o que reduz a
duplicação já na origem.</p>
<h3>3.7 Realidade aumentada (HUD) e severidade</h3>
<p>O modo tempo real sobrepõe ao vídeo, alinhados à cena: a zona de trajetória à frente do veículo (trapézio), as caixas
coloridas pela severidade, o identificador de cada buraco, contadores e o alerta "BURACO À FRENTE" quando um buraco entra
na trajetória — na interface web, o alerta também é falado, o que ajuda condutores e pessoas com baixa visão. A
severidade é um índice relativo: a área da caixa corrigida pela perspectiva de uma via plana (objetos próximos do
horizonte aparecem menores, então a área é dividida pelo quadrado da distância normalizada até o horizonte). Os limiares
baixa/média/alta são os tercis do índice nas {cx_tr} caixas de treino.</p>
<h3>3.8 Privacidade (LGPD) e YuNet</h3>
<p>Para atender à LGPD (finalidade, necessidade e minimização) [9], o sistema não guarda quadros inteiros: o relato leva
só um recorte da via ao redor do buraco e as coordenadas. No modo de tempo real em Python (vídeo e câmera veicular),
rostos que apareçam no vídeo de saída e nos recortes são borrados com o YuNet [7], detector facial de cerca de 230 KB do
OpenCV Zoo (licença MIT); placas de veículos ainda não são anonimizadas. Na interface web, o modo "no servidor" envia
cada quadro à API, que o analisa em memória e o descarta sem gravar, e o modo "no aparelho" não envia imagem nenhuma. O
recorte que a interface web anexa ao relato ainda não passa pelo YuNet: como mostra só o asfalto ao redor do buraco, o
risco é baixo, e a borragem no navegador é um próximo passo.</p>
<h3>3.9 ONNX, ONNX Runtime e navegador</h3>
<p>O modelo de produção é exportado para ONNX, formato aberto executado pelo ONNX Runtime em CPU (servidores sem GPU,
notebooks, câmeras embarcadas) e pelo onnxruntime-web no navegador do celular (WebGPU ou WebAssembly). No modo "no
aparelho" da interface web, a imagem nem sai do telefone — o que reforça a privacidade e permite operar com
conectividade ruim depois que o modelo é baixado.</p>
<h3>3.10 Integração: FastAPI, SQLite, Leaflet e Docker</h3>
<p>A API em FastAPI expõe o contrato do componente: <code>POST /v1/deteccoes</code> (imagem → caixas, processada em
memória e descartada) e <code>POST /v1/relatos</code> (evento → fila), com deduplicação geográfica pela distância de
Haversine — relatos a menos de 15 m de um relato aberto viram confirmações do mesmo buraco — e prioridade =
severidade × (1 + log₂ confirmações). A fila fica em SQLite, o mapa usa Leaflet e OpenStreetMap, e um Dockerfile com
usuário não root empacota a API com o modelo ONNX para o componente de Produção, que a publica em <a
href="{url_mvp}">{url_mvp}</a>.</p>
""",
    "tabela_ferramentas": [
        ["Ultralytics (YOLO26n/s, YOLO11n, BoT-SORT)", "8.4.202", "treino, validação, rastreamento e exportação", "AGPL-3.0"],
        ["PyTorch + CUDA", "2.11.0 / 12.8", "treino e inferência na GPU (AMP)", "BSD-3"],
        ["RDD2022", "2022", "imagens e anotações de treino, validação e teste", "CC BY-SA 4.0"],
        ["OpenCV + YuNet", "5.0 / 2023mar", "vídeo, ORB, desenho e borragem de rostos", "Apache-2.0 / MIT"],
        ["ONNX / ONNX Runtime", "1.23 / 1.31", "modelo portável e inferência em CPU", "Apache-2.0 / MIT"],
        ["onnxruntime-web", "1.20.1", "inferência no navegador (WebGPU/WASM)", "MIT"],
        ["FastAPI + Uvicorn + SQLite", "0.162 / 0.54", "API de detecção e fila de relatos", "MIT / BSD-3 / domínio público"],
        ["Leaflet + OpenStreetMap", "1.9.4", "mapas do relato e da zeladoria", "BSD-2 / ODbL"],
        ["Pillow, Matplotlib, pandas, pytest", "13.0 / 3.11 / 3.1 / 9.2", "HUD, gráficos, análises e testes", "licenças livres"],
    ],
    # ------------------------------------------------------------------ metodologia
    "metodologia_1": """
<p>O trabalho seguiu o percurso proposto pelo desafio — descobrir, arquitetar, dados, embarcar IA e colocar no ar — com
todos os passos automatizados por scripts versionados no repositório (Apêndice).</p>
<h3>Passo 1 — Problema, cidadão e requisitos</h3>
<p>Definimos o cidadão (quem circula pelas vias e o agente de zeladoria), o território (piloto no centro de São Paulo) e
requisitos mensuráveis: detectar buracos em imagens de celular e de câmera veicular; inferência em <strong>tempo
real</strong> (≥ 25 quadros por segundo em GPU e ≥ 10 em CPU); gerar um relato por buraco, sem duplicação; não
armazenar imagens pessoais; e manter a decisão final com uma pessoa.</p>
<h3>Passo 2 — Fonte dos dados e auditoria dos rótulos (<code>scripts/01b_auditar_rotulos.py</code>)</h3>
<p>Começamos por uma redistribuição do RDD2022 em formato YOLO publicada no Hugging Face, mais prática de baixar. Antes
de confiar no treino, auditamos visualmente recortes aleatórios das caixas "pothole" de cada país (Figura A): eram
faixas de pista apagadas, faixas de pedestre e tampas de bueiro — não buracos. Cruzando com as anotações oficiais,
confirmamos que a conversão havia descartado justamente a classe D40 (≈6,5 mil caixas) e rotulado as classes auxiliares
D43/D44/D50 como buraco. O treino em andamento foi interrompido e o pipeline passou a usar a release oficial, lendo a
classe pelo nome. A lição — <strong>auditar os rótulos antes de treinar</strong> — virou um passo fixo do método.</p>
<figure class="dupla"><div><img src="../../resultados/auditoria_rotulos_redistribuicao_hf.jpg" alt="Rótulos incorretos"></div>
<div><img src="../../resultados/auditoria_rotulos_oficial.jpg" alt="Rótulos oficiais D40"></div>
<figcaption>Figura A — Auditoria de rótulos: recortes rotulados como "buraco" na redistribuição (esq.: faixas, bueiros)
e na release oficial, classe D40 (dir.: buracos de fato).</figcaption></figure>
<h3>Passo 3 — Seleção, preparação e divisão dos dados (<code>scripts/01_preparar_dataset.py</code>)</h3>
<p>O zip oficial tem 13 GB, com um zip por país armazenado sem compressão. O script lê apenas os índices e baixa, por
requisições HTTP parciais, só os XMLs e as imagens selecionadas. Seleção reprodutível (semente 42): (a) imagens de nível
de rua de Japão, Índia, China (câmera em motocicleta), República Tcheca e Estados Unidos — a visão aérea de drone foi
descartada por não corresponder ao uso, e a Noruega, por ter imagens de 3.650 px (10,6 GB); (b) todas as imagens com
pelo menos um buraco (D40); (c) uma amostra de imagens <em>sem</em> buraco equivalente a 25% das positivas, com trincas,
faixas e bueiros, que ensinam o modelo a não confundi-los com buracos. Como o teste oficial do RDD2022 não tem rótulos
públicos, dividimos as imagens rotuladas em 70/15/15 por <strong>blocos de 20 números de arquivo consecutivos</strong>,
supondo que números vizinhos fossem quadros vizinhos de um mesmo trajeto. A revisão desta entrega mostrou que essa
suposição não vale: no RDD2022, imagens de números vizinhos não são mais parecidas entre si do que pares aleatórios, e o
mesmo trecho de via aparece em fotos distantes na numeração. Por isso o teste contém trechos de via repetidos do treino;
o Passo 7 mede o efeito, e a seção 5.3 corrige a estimativa de desempenho.</p>
""",
    "metodologia_2": """
<h3>Passo 4 — Configuração do treino (<code>configs/treino.yaml</code>)</h3>
<p>Os hiperparâmetros foram escolhidos a partir das características do problema (Tabela 4). Destaca-se a decisão sobre os
aumentos de dados: rotação e espelhamento vertical foram desativados porque não ocorrem com câmera veicular, enquanto
variações de brilho e de escala reproduzem sol, sombra e distância.</p>
""",
    "metodologia_3": """
<h3>Passo 5 — Treino de três modelos (<code>scripts/02_treinar.py</code>)</h3>
<p>Treinamos o YOLO26n (nano, para celular e borda), o YOLO11n (mesmo porte, geração anterior — comparação justa de
arquitetura) e o YOLO26s (cerca de 4× mais parâmetros — para medir o ganho de capacidade). No primeiro teste a GPU ficou
em ~40% de uso: o gargalo era a decodificação das imagens pela CPU. Com o cache em disco das imagens decodificadas, a
utilização passou de 85%.</p>
<h3>Passo 6 — Avaliação, limiar e escolha do modelo (<code>scripts/03_avaliar.py</code>)</h3>
<p>Cada modelo foi avaliado na validação e no teste. O limiar de confiança de operação é o que maximiza o F1 na
<em>validação</em>, e a escolha do modelo de produção segue uma regra definida antes de olhar o teste: entre os modelos a
até 1,5 ponto percentual do melhor mAP@0,5:0,95 de validação, escolhe-se o de menor latência, porque o produto roda em
celulares e câmeras embarcadas. Houve uma exceção à regra de não decidir pelo teste: a escolha entre as duas variantes
ONNX (Passo 9) foi feita no teste. Repetimos essa comparação na validação
(<code>scripts/04b_variantes_onnx_validacao.py</code>) e a decisão se manteve.</p>
<h3>Passo 7 — Vieses, análise de erros e trechos repetidos (<code>scripts/03_avaliar.py</code> e <code>03b</code>)</h3>
<p>No teste, casamos predições e anotações (IoU ≥ 0,5) para medir precisão, revocação e F1 no limiar de operação por
país de origem e por iluminação (tercis do brilho médio da imagem), além da revocação por tamanho do buraco (pequeno,
médio e grande, critério COCO) e do AP por país. Também geramos mosaicos de acertos, falsos positivos e falsos negativos.
Por fim, medimos quantas imagens de teste mostram o mesmo trecho de via de alguma imagem de treino: para cada imagem de
teste, os 30 vizinhos de treino pelo hash perceptual (dHash) passam por correspondência de pontos ORB e homografia
RANSAC, e 20 ou mais correspondências geométricas caracterizam um trecho repetido. As métricas foram recalculadas com e
sem essas imagens.</p>
<h3>Passo 8 — Calibração da severidade</h3>
<p>Calculamos o índice de severidade das {cx_tr} caixas de treino e adotamos os tercis como limiares
(média ≥ {sev_media}; alta ≥ {sev_alta}).</p>
<h3>Passo 9 — Exportação e desempenho (<code>scripts/04</code>, <code>04b</code> e <code>05</code>)</h3>
<p>Exportamos o modelo para ONNX (opset 17, simplificado) em duas variantes — fim a fim, sem NMS, e com NMS —,
conferimos a precisão de cada uma e medimos a latência em GPU (FP16 e FP32) e em CPU, com 15 inferências de aquecimento
e média e percentil 95 de até 200 repetições sobre imagens reais.</p>
<h3>Passo 10 — Tempo real em vídeos reais (<code>src/vigiavia/tempo_real.py</code>, <code>scripts/06</code> e <code>06b</code>)</h3>
<p>Testamos o pipeline completo — leitura do vídeo, detecção, rastreamento BoT-SORT, severidade, HUD e borragem de
rostos — em quatro vídeos do Wikimedia Commons com licença aberta [12–15], <strong>fora da distribuição de
treino</strong> (câmera na mão, outra época e outros países), incluindo um vídeo gravado na Rua Sílvia, Bela Vista, São
Paulo.</p>
<h3>Passo 11 — Integração com o MVP e testes automatizados (<code>api/</code>, <code>tests/</code>)</h3>
<p>Implementamos a API, a interface web (cidadão e zeladoria) e testes automatizados (pytest) do contrato: detecção,
rejeição de arquivos inválidos, deduplicação geográfica, ordenação por prioridade e validação humana.</p>
<h3>Passo 12 — Documentação e demonstração (<code>scripts/07</code> a <code>09</code>)</h3>
<p>O notebook executado, este relatório e o vídeo de demonstração são gerados por scripts a partir dos mesmos artefatos,
o que mantém números, figuras e código consistentes.</p>
""",
    # ------------------------------------------------------------------ resultados (interpretação)
    "res_comparacao": """<p>Os três modelos convergiram em 33 a 43 minutos (Figura 1). Na validação, o YOLO26s obteve o maior
mAP@0,5:0,95 (0,264), seguido do YOLO26n (0,255) e do YOLO11n (0,244). No mesmo porte (≈2,5 milhões de parâmetros), a
nova geração superou a anterior no teste em 1,0 p.p. de mAP@0,5:0,95 e 2,4 p.p. de mAP@0,5. O ganho veio sobretudo da
precisão (67,0% contra 63,4% no teste, com revocação parecida: 52,6% e 52,1%); não medimos um ganho específico em
buracos pequenos. Já o YOLO26s, com quase 4× mais parâmetros, ganhou só 0,9 p.p. na validação. Pela regra definida antes
de olhar o teste, o <strong>YOLO26n</strong> foi escolhido: está dentro da margem de 1,5 p.p. e é o mais leve, ou seja,
o que roda melhor em celulares e câmeras embarcadas. O teste confirmou a escolha: 0,585 contra 0,589 de mAP@0,5 do
YOLO26s.</p>""",
    "res_teste": """<p>No conjunto de teste, o YOLO26n alcançou mAP@0,5 de 0,585 e mAP@0,5:0,95 de 0,259. No limiar de operação de
0,305, escolhido na validação, foram 459 acertos, 193 falsos positivos e 451 buracos perdidos: <strong>precisão de 70%
e revocação de 50%</strong> (F1 de 0,59). Na prática, de cada 10 alertas, 7 são buracos reais. Uma revocação de 50% por
imagem é menos limitante do que parece em vídeo: a câmera veicular vê o mesmo buraco em vários quadros, e basta uma
detecção confirmada pelo rastreador para gerar o relato. Esses números estão em linha com a literatura do RDD2022, em
que buracos (D40) estão entre as classes mais difíceis por serem pequenos, de bordas difusas e de rotulagem ambígua. A
seção seguinte mostra que eles são um pouco otimistas.</p>""",
    "res_repetidos": """<p>Na revisão da divisão dos dados encontramos um vazamento que a verificação por dHash não detectava:
<strong>{n_rep} das {n_te} imagens de teste ({pct_rep}) mostram o mesmo trecho de via de alguma imagem de treino</strong>,
fotografado em outro momento (Figura 4). A causa é a suposição do Passo 3: os números de arquivo do RDD2022 não seguem a
ordem do trajeto. O efeito aparece nas métricas (Tabela 6): com o mesmo cálculo, o AP@0,5 é {ap_com} nas imagens com
trecho repetido e {ap_sem} nas {n_sem} sem trecho repetido ({ap_todas} no teste inteiro). Por isso, a estimativa mais
honesta para ruas que o modelo nunca viu é um <strong>mAP@0,5 de cerca de {map50_sem}</strong>, e não {map50}.
A diferença é de cerca de 3 pontos e não muda a escolha do modelo, feita na validação com a mesma divisão para os três,
mas muda a leitura por país: na China (moto), {china_rep} das {china_te} imagens de teste repetem trechos do treino, e o
F1 de 0,95 reflete memorização do lugar, não generalização. A correção definitiva é refazer a divisão agrupando as
imagens por local (por similaridade de cena ou por coordenadas, quando houver) e retreinar os três modelos; não houve
tempo de fazê-la nesta entrega.</p>""",
    "res_erros": """<p>A inspeção dos mosaicos mostra três padrões. (1) Parte dos <strong>falsos positivos são buracos reais não
anotados</strong> ou detecções corretas com a caixa deslocada (IoU &lt; 0,5): a métrica penaliza o modelo por
incompletude do rótulo, o que reforça a validação humana antes da ordem de serviço. (2) Remendos escuros, trincas largas
e manchas de umidade às vezes são confundidos com buracos. (3) Os falsos negativos concentram-se em buracos rasos, com
pouco contraste ou parcialmente encobertos por sombras. Tampas de bueiro ainda aparecem entre os falsos positivos,
embora com pouca frequência: a auditoria de rótulos evitou que o modelo aprendesse a chamá-las de buraco.</p>""",
    "res_vieses": """<p>O desempenho varia por origem das imagens: o F1 vai de 0,52 (Índia, estradas sem marcação e com poeira) a 0,65
(Japão). O 0,95 da China (moto) não deve ser lido como desempenho real: são só 22 imagens, e {china_rep} delas repetem
trechos de via do treino (seção 5.3). A verificação por hash perceptual (dHash) não encontrou quase-cópias entre teste e
treino, mas ela não detecta o mesmo lugar fotografado em outro momento — por isso foi complementada pela verificação
geométrica. Imagens escuras têm revocação menor (0,47 contra 0,52 nas claras), e a revocação é parecida entre buracos
pequenos (0,51), médios (0,49) e grandes (0,53). O ponto de atenção para o Brasil é que nenhum país do treino é
brasileiro: o próximo passo é coletar imagens brasileiras pelo próprio aplicativo, com consentimento, e retreinar,
medindo de novo os vieses.</p>""",
    "res_tempo_real": """<p>Só a inferência do YOLO26n roda a 62 FPS na GPU e a 39 FPS em CPU com ONNX Runtime, acima dos requisitos
de 25 e 10 FPS. Na GPU, as médias dos três modelos (13 a 16 ms por imagem) diferem menos que a variação dentro de cada
medição (o percentil 95 chega a 4 ms acima da média); por isso a regra de escolha usou a latência do Ultralytics em lote
(Tabela 5), e não a Tabela 10. O pipeline completo (decodificação, detecção, rastreamento com compensação de movimento,
borragem de rostos e HUD) roda a 14–19 FPS na RTX 3070, abaixo dos 25 quadros por segundo de um vídeo: em tempo real,
parte dos quadros é pulada. A 40 km/h, 14 FPS equivalem a um quadro a cada ~0,8 m de via, suficiente para cada buraco
aparecer em vários quadros. No vídeo de São Paulo (Figura 9, à direita), gravado com a câmera na mão a poucos
centímetros do asfalto, o modelo — que nunca viu imagens do Brasil — detectou o buraco e o alerta de realidade aumentada
disparou. A câmera tremida fragmentava trilhas: trocamos o ByteTrack pelo <strong>BoT-SORT com compensação global de
movimento</strong> (fluxo óptico esparso), suprimimos caixas aninhadas e criamos uma deduplicação espaço-temporal de
eventos, que eliminou 4 trilhas duplicadas e reduziu a contagem de 21 para 7 eventos neste vídeo. Esses 7 eventos não
são 7 buracos: o vídeo mostra um buraco principal, e parte dos eventos são trincas e remendos marcados como buraco —
falsos positivos que a validação humana da zeladoria precisa descartar. O ONNX com NMS manteve a precisão do PyTorch
(mAP@0,5 de 0,582 contra 0,585); a variante fim a fim, sem NMS, perdeu 4,3 p.p. no teste e 3,6 p.p. na validação e foi
descartada.</p>""",
    "res_integracao": """
<p>A Figura 10 mostra a interface web servida pela API — a mesma que o componente Produção publica em <a
href="{url_mvp}">{url_mvp}</a>. No fluxo do cidadão, uma foto, um vídeo (que pode ser pausado no buraco) ou a câmera ao
vivo é analisada — no servidor ou no próprio aparelho, via ONNX — e o relato leva apenas o recorte da via, a
severidade, a confiança e o local marcado no mapa. No fluxo da zeladoria, relatos repetidos aparecem como confirmações do
mesmo buraco e a fila é ordenada por prioridade; o servidor público valida, descarta ou marca como resolvido, e nenhuma
ordem de serviço nasce sem decisão humana. Os relatos da figura são de demonstração: as imagens são amostras reais
analisadas pela API, mas as coordenadas são simuladas em torno da Rua Sílvia (<code>scripts/06c_capturar_telas.py</code>).
Resultado dos testes automatizados do contrato da API:</p>
""",
    "conclusao": """<p>O objetivo foi atingido em boa parte. Realizamos o treinamento customizado de três arquiteturas YOLO para detectar
buracos com dados reais e abertos e escolhemos o YOLO26n por uma regra explícita. Ele detecta buracos em tempo real em
GPU e em CPU (medido) e roda no navegador via onnxruntime-web; a taxa num celular de entrada ainda não foi medida. O
componente está integrado ao MVP por uma API que converte detecções em relatos georreferenciados e priorizados e agrupa
os relatos do mesmo buraco. Isso ataca diretamente a fila de chamados duplicados relatada pela Prefeitura de São Paulo,
e a interface de realidade aumentada alerta o condutor no momento em que o buraco entra na trajetória.</p>
<p>Três resultados de engenharia merecem destaque. (1) A <strong>auditoria dos rótulos</strong> evitou treinar um modelo
que detectaria faixas e bueiros como buracos. (2) As decisões foram tomadas por medição: escolha do modelo, do limiar,
da variante ONNX e do rastreador. (3) A revisão da divisão dos dados encontrou trechos de via repetidos entre treino e
teste e corrigiu a estimativa de desempenho para cerca de {map50_sem} de mAP@0,5 em ruas novas: preferimos o número
honesto ao número alto.</p>
<p><strong>Limitações:</strong> revocação de 50% por imagem; trechos repetidos entre treino e teste (a divisão precisa ser
refeita por local); nenhum dado brasileiro no treino; severidade como índice relativo, não medida física; pipeline
completo abaixo de 25 FPS; falsos positivos em trincas e remendos, como no vídeo da Rua Sílvia; recorte da interface web
sem borragem de rostos e placas; e coordenadas fixas nos vídeos de teste (sem GPS). <strong>Próximos passos:</strong>
refazer a divisão por local e retreinar; coletar e anotar imagens brasileiras pelo próprio app (aprendizado ativo com
as validações da zeladoria); treinar em resolução maior para buracos distantes; medir o desempenho num celular de
entrada; integrar o SP156; e medir em campo os indicadores de impacto: tempo entre detecção e reparo, chamados duplicados
evitados e precisão das detecções confirmadas. Com isso, o VigiaVia contribui para os ODS 11 (11.2 e 11.3), 9 (9.1) e
3 (3.6), como detalhado na Tabela 1.</p>
<div class="caixa"><p><strong>Revisão de 09/11/2026.</strong> Esta versão substitui a de 04/11 e corrige: a afirmação de
que a divisão por blocos evitava vazamento (seções 4 e 5.3, com nova medição); a descrição da privacidade por modo de
uso (seção 3.8); o rastreador (BoT-SORT, seção 3.6); a URL do MVP na ficha; a declaração dos dados simulados das telas;
a explicação dos ODS (Tabela 1); e números que divergiam dos artefatos (comparação entre modelos, tempos de treino,
resolução das imagens, limiares e legendas). As telas foram atualizadas para a versão publicada da interface.</p></div>""",
    # ------------------------------------------------------------------ referências
    "referencias": """
<li>ARYA, D.; MAEDA, H.; GHOSH, S. K.; TOSHNIWAL, D.; SEKIMOTO, Y. <em>RDD2022: A multi-national image dataset for
automatic Road Damage Detection</em>. arXiv:2209.08538, 2022. Dados: CC BY-SA 4.0 (redistribuição YOLO:
huggingface.co/datasets/dronefreak/RDD2022).</li>
<li>JOCHER, G.; QIU, J.; LIU, M.; LYU, S.; AKYON, F. C.; KALFAOGLU, M. E. <em>Ultralytics YOLO26: Unified Real-Time
End-to-End Vision Models</em>. arXiv:2606.03748, 2026. https://docs.ultralytics.com/models/yolo26/</li>
<li>CONFEDERAÇÃO NACIONAL DO TRANSPORTE. <em>Pesquisa CNT de Rodovias 2025</em>. Divulgada em 17/12/2025, conforme
Agência iNFRA: "Rodovias do país voltam a apresentar melhora, aponta pesquisa da CNT".</li>
<li>BNEWS SÃO PAULO. <em>Mesmo após recapeamento, SP registra quase 52 mil pedidos de tapa-buraco</em>. 09/08/2026
(dados da Secretaria Municipal das Subprefeituras).</li>
<li>ZHANG, Y. et al. <em>ByteTrack: Multi-Object Tracking by Associating Every Detection Box</em>. ECCV, 2022.</li>
<li>LIN, T.-Y. et al. <em>Microsoft COCO: Common Objects in Context</em>. ECCV, 2014.</li>
<li>WU, W.; PENG, H.; YU, S. <em>YuNet: A Tiny Millisecond-level Face Detector</em>. Machine Intelligence Research,
v. 20, n. 5, p. 656–665, 2023.</li>
<li>JOCHER, G.; QIU, J. <em>Ultralytics YOLO11</em>, 2024. https://github.com/ultralytics/ultralytics</li>
<li>BRASIL. Lei nº 13.709, de 14 de agosto de 2018 — Lei Geral de Proteção de Dados Pessoais (LGPD).</li>
<li>ONU BRASIL. <em>Objetivos de Desenvolvimento Sustentável</em> (ODS 3, 9 e 11). https://brasil.un.org/pt-br/sdgs</li>
<li>AHARON, N.; ORFAIG, R.; BOBROVSKY, B.-Z. <em>BoT-SORT: Robust Associations Multi-Pedestrian Tracking</em>.
arXiv:2206.14651, 2022.</li>
""",
    # ------------------------------------------------------------------ apêndice
    "otimizador": "AdamW, lr 0,002 (auto)",
    "reproduzir": """# 1) Ambiente (Python 3.12+; GPU NVIDIA opcional)
python -m venv .venv && .venv\\Scripts\\activate          # Linux/macOS: source .venv/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 2) Dados: RDD2022 -> dataset VigiaVia (classe "buraco")
python scripts/01_preparar_dataset.py --out datasets/vigiavia-rdd2022

# 3) Treino dos três modelos (33 a 43 min cada numa RTX 3070)
python scripts/02_treinar.py --data datasets/vigiavia-rdd2022/data.yaml --exp E1_yolo26n E2_yolo11n E3_yolo26s

# 4) Avaliação, trechos repetidos, escolha do modelo, vieses, severidade, ONNX e velocidade
python scripts/03_avaliar.py --data datasets/vigiavia-rdd2022/data.yaml
python scripts/03b_cenas_repetidas.py --data datasets/vigiavia-rdd2022/data.yaml
python scripts/04_exportar_onnx.py --data datasets/vigiavia-rdd2022/data.yaml
python scripts/04b_variantes_onnx_validacao.py --data datasets/vigiavia-rdd2022/data.yaml
python scripts/05_benchmark_velocidade.py --data datasets/vigiavia-rdd2022/data.yaml

# 5) Tempo real (webcam) e vídeos reais
cd src && python -m vigiavia.tempo_real --fonte 0 --mostrar && cd ..
python scripts/06_baixar_videos_demo.py && python scripts/06b_processar_videos.py

# 6) API + interface web (http://localhost:8000) e testes
uvicorn api.main:app --port 8000
pytest -q
""",
}
