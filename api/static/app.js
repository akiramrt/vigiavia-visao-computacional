/* VigiaVia — interface web do componente de visão computacional.
 * Modo "servidor": cada quadro, reduzido ao tamanho de entrada do modelo (640 px), vai para POST /v1/deteccoes
 * (ONNX Runtime no servidor). Até 2 quadros ficam em análise ao mesmo tempo: o gargalo medido é a rede
 * (~250 ms de ida e volta para ~65 ms de inferência), não o modelo.
 * Modo "aparelho": o modelo ONNX roda no navegador (onnxruntime-web) e a imagem nunca sai do celular.
 * Vídeo e câmera aparecem no próprio <video> (fluido, com controles); as caixas vão numa camada transparente por cima.
 */
"use strict";
const $ = (s) => document.querySelector(s);
const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.webgpu.min.js";
const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
// Os serviços do OpenStreetMap (tiles e busca de endereço) exigem Referer, e a página usa Referrer-Policy:
// no-referrer. Só esses pedidos enviam a origem, nunca o caminho: https://operations.osmfoundation.org/policies/
const REF_OSM = "strict-origin-when-cross-origin";
const OSM_TILES = "https://tile.openstreetmap.org/{z}/{x}/{y}.png";
const OSM_OPCOES = { maxZoom: 19, attribution: "© OpenStreetMap", referrerPolicy: REF_OSM };
const ROTULO = { baixa: "BAIXA", media: "MÉDIA", alta: "ALTA" };
const COR = { baixa: "#ffd400", media: "#ff8c00", alta: "#ff3b30" };
const PESO_SEV = { baixa: 1, media: 2, alta: 3 };
const fmtPrio = (v) => Number(v).toLocaleString("pt-BR", { maximumFractionDigits: 1 });   // 7.755 -> "7,8"
const fmtTempo = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const EXT_VIDEO = /\.(mp4|m4v|mov|webm|ogv|mkv|avi|3gp)$/i;

const st = {
  modelo: null, sessao: null, fonte: null, rodando: false, stream: null, ultimaVoz: 0,
  geracao: 0,          // muda a cada troca de fonte; resultados atrasados da fonte anterior são descartados
  seq: 0, exibido: 0,  // ordem dos quadros enviados e do último exibido (resposta fora de ordem é ignorada)
  emVoo: 0, tempoAnalisado: -1, pausaAte: 0, taxa: 0, tempos: [], rtt: 0, ultimoEnvio: 0,
  atual: null,         // quadro exibido agora, se tem buraco: { q, d, severidade, confianca }
  candidato: null,     // quadro com a detecção mais confiante da fonte atual
  enviando: false,
  mapa: null, camadaMapa: null, mapaLocal: null, pino: null,
};
const tela = $("#tela"), ctx = tela.getContext("2d"), video = $("#video"), palco = $("#palco");

/* ------------------------------------------------------------------ abas (teclado acessível) */
const abas = [...document.querySelectorAll('[role="tab"]')];
function ativarAba(aba) {
  abas.forEach((a) => {
    const sel = a === aba;
    a.setAttribute("aria-selected", sel);
    a.tabIndex = sel ? 0 : -1;
    $("#" + a.getAttribute("aria-controls")).hidden = !sel;
  });
  if (aba.id === "tab-cidadao" && st.mapaLocal) setTimeout(() => st.mapaLocal.invalidateSize(), 50);
  if (aba.id === "tab-zeladoria") carregarFila();
  if (aba.id === "tab-modelo") mostrarModelo();
}
abas.forEach((a, i) => {
  a.addEventListener("click", () => ativarAba(a));
  a.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      const n = abas[(i + (e.key === "ArrowRight" ? 1 : abas.length - 1)) % abas.length];
      n.focus(); ativarAba(n);
    }
  });
});

/* ------------------------------------------------------------------ utilidades */
const status = (t) => ($("#status").textContent = t);
const confAtual = () => parseFloat($("#conf").value);
$("#conf").addEventListener("input", () => ($("#conf-val").textContent = confAtual().toFixed(2).replace(".", ",")));

function severidade(x1, y1, x2, y2, W, H) {
  const s = st.modelo?.severidade || { horizonte: 0.35, limiar_media: 0.01, limiar_alta: 0.03 };
  let esc = (y2 / H - s.horizonte) / (1 - s.horizonte);
  esc = Math.min(1, Math.max(0.15, esc));
  const area = (Math.max(0, x2 - x1) * Math.max(0, y2 - y1)) / (W * H);
  const indice = Math.min(1, area / (esc * esc));
  return { indice, area, nivel: indice >= s.limiar_alta ? "alta" : indice >= s.limiar_media ? "media" : "baixa" };
}

function zonaPx(W, H) {
  const z = st.modelo?.zona_trajetoria || [[0.12, 1], [0.4, 0.58], [0.6, 0.58], [0.88, 1]];
  return z.map(([x, y]) => [x * W, y * H]);
}

function dentro(pt, poli) {  // ray casting
  let d = false;
  for (let i = 0, j = poli.length - 1; i < poli.length; j = i++) {
    const [xi, yi] = poli[i], [xj, yj] = poli[j];
    if ((yi > pt[1]) !== (yj > pt[1]) && pt[0] < ((xj - xi) * (pt[1] - yi)) / (yj - yi) + xi) d = !d;
  }
  return d;
}

function capturar(fonte, w, h) {   // cada quadro analisado tem o próprio canvas: é dele que sai o recorte do relato
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  c.getContext("2d").drawImage(fonte, 0, 0, w, h);
  return c;
}

/* ------------------------------------------------------------------ detecção no servidor */
async function detectarServidor(q) {
  const lado = st.modelo?.imgsz || 640;     // o modelo trabalha em 640 px: mandar mais que isso é só tempo de rede
  const esc = Math.min(1, lado / Math.max(q.width, q.height));
  const fonte = esc < 1 ? capturar(q, Math.round(q.width * esc), Math.round(q.height * esc)) : q;
  const blob = await new Promise((r) => fonte.toBlob(r, "image/jpeg", 0.85));
  const fd = new FormData(); fd.append("imagem", blob, "quadro.jpg");
  const resp = await fetch(`/v1/deteccoes?conf=${confAtual()}`, { method: "POST", body: fd });
  if (!resp.ok) throw new Error(`API ${resp.status}`);
  const j = await resp.json();
  return j.deteccoes.map((d) => ({ ...d, x1: d.x1 / esc, y1: d.y1 / esc, x2: d.x2 / esc, y2: d.y2 / esc }));
}

/* ------------------------------------------------------------------ detecção no aparelho (ONNX) */
function carregarScript(src) {
  return new Promise((ok, falha) => { const s = document.createElement("script"); s.src = src; s.onload = ok; s.onerror = falha; document.head.appendChild(s); });
}
async function prepararOnnx() {
  if (st.sessao) return;
  if (!st.modelo?.onnx) throw new Error("Modelo ONNX não publicado no servidor");
  status("Baixando o modelo para o aparelho (uma vez só)…");
  if (!window.ort) await carregarScript(ORT_URL);
  ort.env.wasm.wasmPaths = ORT_WASM;
  const provs = navigator.gpu ? ["webgpu", "wasm"] : ["wasm"];
  st.sessao = await ort.InferenceSession.create(st.modelo.onnx, { executionProviders: provs });
  status(`Modelo carregado no aparelho (${provs[0]}). A imagem não sai do seu celular.`);
}
function nms(caixas, iou = 0.5) {
  caixas.sort((a, b) => b.s - a.s);
  const out = [];
  const area = (b) => (b.x2 - b.x1) * (b.y2 - b.y1);
  for (const c of caixas) {
    if (out.every((o) => {
      const iw = Math.max(0, Math.min(c.x2, o.x2) - Math.max(c.x1, o.x1));
      const ih = Math.max(0, Math.min(c.y2, o.y2) - Math.max(c.y1, o.y1));
      const inter = iw * ih; return inter / (area(c) + area(o) - inter) < iou;
    })) out.push(c);
  }
  return out;
}
async function detectarAparelho(q) {
  await prepararOnnx();
  const S = st.modelo.imgsz || 640, W = q.width, H = q.height;
  const r = Math.min(S / W, S / H), nw = Math.round(W * r), nh = Math.round(H * r);
  const px = Math.floor((S - nw) / 2), py = Math.floor((S - nh) / 2);
  const lb = document.createElement("canvas"); lb.width = S; lb.height = S;
  const lctx = lb.getContext("2d");
  lctx.fillStyle = "rgb(114,114,114)"; lctx.fillRect(0, 0, S, S);
  lctx.drawImage(q, px, py, nw, nh);
  const { data } = lctx.getImageData(0, 0, S, S);
  const t = new Float32Array(3 * S * S);
  for (let i = 0, p = 0; i < S * S; i++, p += 4) {
    t[i] = data[p] / 255; t[i + S * S] = data[p + 1] / 255; t[i + 2 * S * S] = data[p + 2] / 255;
  }
  const entrada = new ort.Tensor("float32", t, [1, 3, S, S]);
  const saidas = await st.sessao.run({ [st.sessao.inputNames[0]]: entrada });
  const o = saidas[st.sessao.outputNames[0]], d = o.data, dims = o.dims;
  let cx = [];
  if (dims[2] === 6) {                       // YOLO26 fim-a-fim (sem NMS): [1, 300, 6]
    for (let i = 0; i < dims[1]; i++) {
      const k = i * 6; if (d[k + 4] >= confAtual()) cx.push({ x1: d[k], y1: d[k + 1], x2: d[k + 2], y2: d[k + 3], s: d[k + 4] });
    }
  } else {                                   // formato clássico [1, 5, N] -> requer NMS
    const n = dims[2];
    for (let i = 0; i < n; i++) {
      const s = d[4 * n + i]; if (s < confAtual()) continue;
      const x = d[i], y = d[n + i], w = d[2 * n + i], h = d[3 * n + i];
      cx.push({ x1: x - w / 2, y1: y - h / 2, x2: x + w / 2, y2: y + h / 2, s });
    }
    cx = nms(cx);
  }
  const zona = zonaPx(W, H);
  return cx.map((c) => {
    const x1 = (c.x1 - px) / r, y1 = (c.y1 - py) / r, x2 = (c.x2 - px) / r, y2 = (c.y2 - py) / r;
    const sv = severidade(x1, y1, x2, y2, W, H);
    return { x1, y1, x2, y2, confianca: c.s, severidade: sv.nivel, indice_severidade: sv.indice,
             area_relativa: sv.area, na_trajetoria: dentro([(x1 + x2) / 2, y2], zona) };
  });
}

/* ------------------------------------------------------------------ desenho (AR) */
function ajustarTela(w, h) {
  if (tela.width !== w || tela.height !== h) { tela.width = w; tela.height = h; }
  $("#vazio").hidden = true;
}
function limparTela() {
  ctx.clearRect(0, 0, tela.width, tela.height);
  $("#alerta").hidden = true; $("#lista-det").innerHTML = "";
}
function desenhar(dets, q) {
  const W = tela.width, H = tela.height, u = Math.max(1, W / 1280);
  ctx.clearRect(0, 0, W, H);
  if (st.fonte === "foto") ctx.drawImage(q, 0, 0);   // no vídeo e na câmera, a imagem é o próprio <video>, por baixo
  const alerta = dets.some((d) => d.na_trajetoria && d.confianca >= confAtual());
  const z = zonaPx(W, H);
  ctx.beginPath(); z.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.closePath();
  ctx.fillStyle = alerta ? "rgba(255,59,48,.18)" : "rgba(60,207,122,.14)"; ctx.fill();
  ctx.lineWidth = 2 * u; ctx.strokeStyle = alerta ? "#ff3b30" : "#3ccf7a"; ctx.stroke();
  ctx.font = `bold ${Math.round(17 * u)}px system-ui, sans-serif`;
  for (const d of dets) {
    ctx.lineWidth = 3 * u; ctx.strokeStyle = COR[d.severidade];
    ctx.strokeRect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);
    const txt = `buraco ${(d.confianca * 100).toFixed(0)}% · ${ROTULO[d.severidade]}`;
    const tw = ctx.measureText(txt).width + 10 * u, th = 24 * u;
    const ty = d.y1 - th > 0 ? d.y1 - th : d.y1;
    ctx.fillStyle = COR[d.severidade]; ctx.fillRect(d.x1, ty, tw, th);
    ctx.fillStyle = "#111"; ctx.fillText(txt, d.x1 + 5 * u, ty + 18 * u);
  }
  $("#alerta").hidden = !alerta;
  if (alerta && $("#voz").checked && "speechSynthesis" in window && Date.now() - st.ultimaVoz > 5000) {
    const f = new SpeechSynthesisUtterance("Atenção, buraco à frente"); f.lang = "pt-BR";
    speechSynthesis.speak(f); st.ultimaVoz = Date.now();
  }
  const lista = $("#lista-det");
  lista.innerHTML = dets.length ? "" : "<li>Nenhum buraco detectado neste quadro.</li>";
  dets.slice(0, 6).forEach((d, i) => {
    const li = document.createElement("li");
    li.innerHTML = `<span>Buraco ${i + 1} · confiança ${(d.confianca * 100).toFixed(0)}%${d.na_trajetoria ? " · à frente" : ""}</span>
      <span class="chip ${d.severidade}">${ROTULO[d.severidade]}</span>`;
    lista.appendChild(li);
  });
}

/* ------------------------------------------------------------------ análise dos quadros */
const maxEmVoo = () => ($("#modo").value === "servidor" ? 2 : 1);
function resumo(dets, q) {
  const d = dets.reduce((a, b) => (b.confianca > a.confianca ? b : a));
  const sev = dets.reduce((m, x) => (PESO_SEV[x.severidade] > PESO_SEV[m] ? x.severidade : m), "baixa");
  return { q, d, severidade: sev, confianca: d.confianca };
}
async function analisar(q) {
  const g = st.geracao, seq = ++st.seq, t0 = performance.now();
  st.emVoo++;
  try {
    const dets = $("#modo").value === "aparelho" ? await detectarAparelho(q) : await detectarServidor(q);
    const ida = performance.now() - t0;
    st.rtt = st.rtt ? 0.8 * st.rtt + 0.2 * ida : ida;
    if (g !== st.geracao || seq < st.exibido) return null;   // fonte trocada ou resposta fora de ordem
    st.exibido = seq;
    desenhar(dets, q);
    st.atual = dets.length ? resumo(dets, q) : null;
    if (st.atual && (!st.candidato || st.atual.confianca > st.candidato.confianca)) st.candidato = st.atual;
    atualizarBotao();
    // Taxa = análises nas últimas 12 respostas ÷ tempo entre elas (com 2 em análise, as respostas chegam em pares;
    // só aparece depois de 6 respostas, para não mostrar um número absurdo logo no início).
    const agora = performance.now();
    st.tempos.push(agora);
    if (st.tempos.length > 12) st.tempos.shift();
    st.taxa = st.tempos.length >= 6 ? ((st.tempos.length - 1) * 1000) / (agora - st.tempos[0]) : 0;
    return dets;
  } finally {
    if (g === st.geracao) st.emVoo--;
  }
}
function mostrarStatus(d) {
  const n = `${d.length} buraco(s)`;
  if (st.fonte === "video" && video.ended) {
    status(st.candidato ? "Vídeo terminado. Buraco detectado: já dá para enviar o relato (ou volte a barra até ele)."
                        : "Vídeo terminado. Nenhum buraco detectado.");
  } else if (st.fonte === "video" && video.paused) {
    status(`Pausado em ${fmtTempo(video.currentTime)} · ${n} neste quadro${d.length ? " · o relato vai com este quadro" : ""}`);
  } else {
    status(`${n} · ${st.taxa ? st.taxa.toFixed(1).replace(".", ",") : "…"} análises/s · ${$("#modo").value}`);
  }
}
// O vídeo toca sozinho; o laço só decide quando mandar um quadro: sempre que houver vaga e o quadro for novo
// (tocando, quase sempre; pausado, uma vez ao pausar e a cada vez que a barra é arrastada). Com 2 em análise, os
// envios ficam espaçados em meia ida e volta: as respostas chegam distribuídas, e não em pares quase iguais.
function laco() {
  const g = st.geracao;
  const passo = () => {
    if (!st.rodando || g !== st.geracao) return;
    if (video.readyState >= 2 && video.videoWidth) {
      ajustarTela(video.videoWidth, video.videoHeight);
      const agora = performance.now(), vagas = maxEmVoo();
      const quadroNovo = video.currentTime !== st.tempoAnalisado;
      const espacado = vagas === 1 || agora - st.ultimoEnvio >= (st.rtt || 0) / vagas;
      if (quadroNovo && espacado && st.emVoo < vagas && agora >= st.pausaAte) {
        st.tempoAnalisado = video.currentTime; st.ultimoEnvio = agora;
        analisar(capturar(video, video.videoWidth, video.videoHeight))
          .then((d) => { if (d) mostrarStatus(d); })
          .catch((e) => { st.pausaAte = performance.now() + 1000; status("Erro na detecção: " + e.message); });
      }
    }
    requestAnimationFrame(passo);
  };
  requestAnimationFrame(passo);
}
function iniciarAnalise() {
  if (st.rodando) return;
  st.rodando = true; st.tempoAnalisado = -1; st.tempos = []; st.taxa = 0;
  $("#btn-parar").hidden = false;
  laco();
}
function parar() {
  st.rodando = false; st.geracao++; st.emVoo = 0;
  st.stream?.getTracks().forEach((t) => t.stop()); st.stream = null;
  video.pause(); $("#btn-parar").hidden = true; $("#alerta").hidden = true;
}
function novaFonte(fonte) {
  parar(); limparCandidato(); limparTela();
  if (video.srcObject || video.getAttribute("src")) { video.srcObject = null; video.removeAttribute("src"); video.load(); }
  st.fonte = fonte;
  palco.classList.toggle("sobre-video", fonte !== "foto");
  video.hidden = fonte === "foto";
  video.controls = fonte === "video";
}

/* ------------------------------------------------------------------ câmera / vídeo / foto */
$("#btn-parar").addEventListener("click", () => {
  parar();
  if (st.fonte === "camera") { limparTela(); status("Câmera desligada."); }   // sem imagem por baixo, as caixas somem
});
video.addEventListener("play", () => { st.tempos = []; st.taxa = 0; if (st.fonte === "video") iniciarAnalise(); });
video.addEventListener("pause", () => { st.tempos = []; st.taxa = 0; });
$("#btn-camera").addEventListener("click", async () => {
  try {
    novaFonte("camera");
    st.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: { ideal: 1280 } }, audio: false });
    video.srcObject = st.stream; await video.play(); iniciarAnalise();
  } catch (e) { status("Não foi possível abrir a câmera: " + e.message); }
});
$("#arquivo").addEventListener("change", async (ev) => {
  const f = ev.target.files[0];
  ev.target.value = "";                      // escolher o mesmo arquivo de novo também dispara "change"
  if (!f) return;
  const ehVideo = f.type.startsWith("video/") || (!f.type && EXT_VIDEO.test(f.name));
  novaFonte(ehVideo ? "video" : "foto");
  if (st.urlArquivo) URL.revokeObjectURL(st.urlArquivo);
  const url = (st.urlArquivo = URL.createObjectURL(f));
  if (ehVideo) {
    status("Abrindo o vídeo…");
    try {
      video.src = url; await video.play(); iniciarAnalise();
    } catch (e) {
      status(`Não foi possível abrir este vídeo (${e.message}). Tente um MP4 (H.264) ou WebM.`);
    }
  } else {
    const img = new Image();
    img.onload = () => analisarFoto(img).catch((e) => status("Erro: " + e.message));
    img.onerror = () => status("Não foi possível abrir este arquivo. Envie uma foto (JPG ou PNG) ou um vídeo (MP4 ou WebM).");
    img.src = url;
  }
});
async function analisarFoto(img, exemplo = "") {
  const q = capturar(img, img.naturalWidth, img.naturalHeight);
  ajustarTela(q.width, q.height); ctx.drawImage(q, 0, 0);
  status("Analisando…");
  const d = await analisar(q);
  if (d) status(`${d.length} buraco(s) detectado(s)${exemplo ? " · exemplo " + exemplo : "."}`);
}

/* ------------------------------------------------------------------ local do buraco (mapa) */
const msgLocal = (t, ok = true) => { const el = $("#local-msg"); el.className = "ajuda" + (ok ? "" : " erro-txt"); el.textContent = t; };
function iniciarMapaLocal() {
  if (!window.L || st.mapaLocal) return;
  const pos = [parseFloat($("#lat").value), parseFloat($("#lon").value)];
  st.mapaLocal = L.map("mapa-local", { scrollWheelZoom: false }).setView(pos, 16);
  L.tileLayer(OSM_TILES, OSM_OPCOES).addTo(st.mapaLocal);
  st.pino = L.marker(pos, { draggable: true, autoPan: true, title: "Local do buraco (arraste)", alt: "Local do buraco",
    icon: L.divIcon({ className: "pino", iconSize: [22, 22], iconAnchor: [11, 11] }) }).addTo(st.mapaLocal);
  st.pino.on("dragend", () => definirLocal(st.pino.getLatLng(), false));
  st.mapaLocal.on("click", (e) => definirLocal(e.latlng, false));
}
function definirLocal({ lat, lng }, centralizar = true) {
  $("#lat").value = lat.toFixed(6); $("#lon").value = lng.toFixed(6);
  if (!st.pino) return;
  st.pino.setLatLng([lat, lng]);
  if (centralizar) st.mapaLocal.setView([lat, lng], Math.max(st.mapaLocal.getZoom(), 17));
}
["#lat", "#lon"].forEach((s) => $(s).addEventListener("change", () => {
  const lat = parseFloat($("#lat").value), lng = parseFloat($("#lon").value);
  if (Number.isFinite(lat) && Number.isFinite(lng)) definirLocal({ lat, lng });
}));
// Busca sob demanda (um pedido por clique, sem autocompletar), como pede a política de uso do Nominatim.
$("#form-busca").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("#busca-endereco").value.trim();
  if (!q) return;
  const btn = $("#btn-buscar"); btn.disabled = true; msgLocal("Buscando o endereço…");
  try {
    const p = new URLSearchParams({ q, format: "jsonv2", limit: "1", countrycodes: "br", "accept-language": "pt-BR" });
    if (st.mapaLocal) {
      const b = st.mapaLocal.getBounds();
      p.set("viewbox", [b.getWest(), b.getNorth(), b.getEast(), b.getSouth()].map((v) => v.toFixed(5)).join(","));
    }
    const r = await fetch(`https://nominatim.openstreetmap.org/search?${p}`, { referrerPolicy: REF_OSM });
    if (!r.ok) throw new Error(`busca ${r.status}`);
    const [achado] = await r.json();
    if (!achado) { msgLocal("Endereço não encontrado. Tente incluir o bairro ou a cidade.", false); return; }
    definirLocal({ lat: parseFloat(achado.lat), lng: parseFloat(achado.lon) });
    msgLocal("Local: " + achado.display_name);
  } catch (err) {
    msgLocal("Não foi possível buscar o endereço: " + err.message, false);
  } finally { btn.disabled = false; }
});
$("#btn-gps").addEventListener("click", () => {
  if (!navigator.geolocation) return msgLocal("Geolocalização indisponível neste navegador.", false);
  msgLocal("Obtendo a sua localização…");
  navigator.geolocation.getCurrentPosition(
    (p) => { definirLocal({ lat: p.coords.latitude, lng: p.coords.longitude }); msgLocal(`Localização obtida (±${Math.round(p.coords.accuracy)} m).`); },
    (e) => msgLocal("Localização negada: marque o local no mapa ou busque o endereço. " + e.message, false),
    { enableHighAccuracy: true, timeout: 10000 });
});

/* ------------------------------------------------------------------ relato */
function recorteEvidencia({ q, d }) {
  const m = 0.25, w = d.x2 - d.x1, h = d.y2 - d.y1;
  const x = Math.max(0, d.x1 - w * m), y = Math.max(0, d.y1 - h * m);
  const x2 = Math.min(q.width, d.x2 + w * m), y2 = Math.min(q.height, d.y2 + h * m);
  const esc = Math.min(1, 480 / Math.max(x2 - x, y2 - y));
  const c = document.createElement("canvas"); c.width = Math.round((x2 - x) * esc); c.height = Math.round((y2 - y) * esc);
  c.getContext("2d").drawImage(q, x, y, x2 - x, y2 - y, 0, 0, c.width, c.height);
  return c.toDataURL("image/jpeg", 0.8).split(",")[1];
}
function atualizarBotao() { $("#btn-relatar").disabled = !st.candidato || st.enviando; }
function limparCandidato() {
  st.candidato = null; st.atual = null; atualizarBotao(); $("#resposta").textContent = "";
}
// Vídeo pausado num quadro com buraco: vai esse quadro (o que a pessoa está vendo). Senão, o mais confiante.
function quadroDoRelato() {
  const pausado = st.fonte === "video" && (video.paused || video.ended);
  return pausado && st.atual ? st.atual : st.candidato;
}
$("#btn-relatar").addEventListener("click", async () => {
  const c = quadroDoRelato();
  if (!c) return;
  const el = $("#resposta");
  const lat = parseFloat($("#lat").value), lon = parseFloat($("#lon").value);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
    el.className = "resposta erro"; el.textContent = "Marque no mapa onde está o buraco."; return;
  }
  const corpo = { lat, lon, severidade: c.severidade, confianca_max: c.confianca, origem: "cidadao",
    modelo: st.modelo?.modelo, evidencia_jpg_base64: recorteEvidencia(c) };
  st.enviando = true; atualizarBotao(); el.className = "resposta"; el.textContent = "Enviando…";
  try {
    const r = await fetch("/v1/relatos", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(corpo) });
    const j = await r.json();
    el.className = "resposta " + (r.ok ? "ok" : "erro");
    el.textContent = r.ok ? `${j.mensagem} (prioridade ${fmtPrio(j.relato.prioridade)}).` : `Erro: ${JSON.stringify(j.detail)}`;
    if (r.ok) { st.candidato = null; st.atual = null; }   // o mesmo buraco não vai duas vezes sem querer
  } catch (e) {
    el.className = "resposta erro"; el.textContent = "Não foi possível enviar o relato: " + e.message;
  } finally {
    st.enviando = false; atualizarBotao();
  }
});

/* ------------------------------------------------------------------ zeladoria */
const ACOES = [["confirmado", "Validar"], ["descartado", "Descartar"], ["resolvido", "Resolvido"]];
const msgFila = (t, ok = true) => { const el = $("#fila-msg"); el.className = "resposta " + (ok ? "ok" : "erro"); el.textContent = t; };
async function carregarFila() {
  let rel;
  try {
    const r = await fetch("/v1/relatos");
    if (!r.ok) throw new Error(`API ${r.status}`);
    rel = (await r.json()).relatos;
  } catch (e) { msgFila("Não foi possível carregar a fila: " + e.message, false); return; }
  if ($("#fila-msg").classList.contains("erro")) $("#fila-msg").textContent = "";
  const c = (f) => rel.filter(f).length;
  $("#kpis").innerHTML = [
    ["Relatos abertos", c((r) => r.status === "pendente" || r.status === "confirmado")],
    ["Aguardando validação", c((r) => r.status === "pendente")],
    ["Severidade alta", c((r) => r.severidade === "alta" && r.status !== "resolvido")],
    ["Confirmações recebidas", rel.reduce((s, r) => s + r.confirmacoes, 0)],
    ["Resolvidos", c((r) => r.status === "resolvido")],
  ].map(([t, v]) => `<div class="kpi"><b>${v}</b><span>${t}</span></div>`).join("");
  const tb = $("#fila"); tb.innerHTML = "";
  for (const r of rel) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td><b>${fmtPrio(r.prioridade)}</b></td>
      <td>${r.tem_evidencia ? `<img src="/v1/relatos/${r.id}/evidencia" alt="Recorte da via com o buraco">` : "—"}</td>
      <td><span class="chip ${r.severidade}">${ROTULO[r.severidade]}</span></td>
      <td>${r.confirmacoes}</td><td>${(r.confianca_max * 100).toFixed(0)}%</td><td>${r.status}</td>
      <td class="acoes-fila">${ACOES.map(([s, rotulo]) => `<button class="mini" data-id="${r.id}" data-s="${s}"${r.status === s ? " disabled" : ""}>${rotulo}</button>`).join("")}</td>`;
    tb.appendChild(tr);
  }
  tb.querySelectorAll("button").forEach((b) => b.addEventListener("click", async () => {
    b.disabled = true;
    try {
      const r = await fetch(`/v1/relatos/${b.dataset.id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ status: b.dataset.s }) });
      if (!r.ok) throw new Error(`API ${r.status}`);
      msgFila(`Relato atualizado: ${b.dataset.s}${b.dataset.s === "confirmado" ? " (vira ordem de serviço)" : ""}.`);
    } catch (e) { msgFila("Não foi possível atualizar o relato: " + e.message, false); }
    carregarFila();
  }));
  if (window.L) {
    if (!st.mapa) {
      st.mapa = L.map("mapa").setView([-23.5603, -46.6505], 15);
      L.tileLayer(OSM_TILES, OSM_OPCOES).addTo(st.mapa);
      st.camadaMapa = L.layerGroup().addTo(st.mapa);
    }
    st.camadaMapa.clearLayers();
    rel.forEach((r) => L.circleMarker([r.lat, r.lon], { radius: 6 + 2 * r.confirmacoes, color: COR[r.severidade], fillOpacity: 0.7 })
      .bindPopup(`<b>Buraco · ${ROTULO[r.severidade]}</b><br>${r.confirmacoes} confirmação(ões)<br>Status: ${r.status}`).addTo(st.camadaMapa));
    if (rel.length) st.mapa.fitBounds(rel.map((r) => [r.lat, r.lon]), { maxZoom: 16, padding: [30, 30] });
    setTimeout(() => st.mapa.invalidateSize(), 50);
  }
}
$("#btn-atualizar").addEventListener("click", carregarFila);

/* ------------------------------------------------------------------ ficha do modelo */
async function mostrarModelo() {
  const m = st.modelo; if (!m) return;
  const t = m.metricas_teste || {};
  const fmt = (v) => (v == null ? "—" : (v * 100).toFixed(1) + "%");
  $("#ficha-modelo").innerHTML = `
    <p><b>Modelo:</b> ${m.modelo} · <b>entrada:</b> ${m.imgsz}px · <b>limiar:</b> ${m.limiar_confianca.toFixed(2)}
       (máximo F1 na validação) · <b>ONNX:</b> ${m.onnx ? `<a href="${m.onnx}">${m.onnx}</a>` : "—"}</p>
    <div class="metricas">
      <div class="kpi"><b>${fmt(t.mAP50)}</b><span>mAP@0,5 (teste)</span></div>
      <div class="kpi"><b>${fmt(t.mAP50_95)}</b><span>mAP@0,5:0,95 (teste)</span></div>
      <div class="kpi"><b>${fmt(t.precisao_no_limiar)}</b><span>Precisão no limiar</span></div>
      <div class="kpi"><b>${fmt(t.revocacao_no_limiar)}</b><span>Revocação no limiar</span></div>
    </div>
    <p class="ajuda">Treinado na release oficial do RDD2022 (Japão, Índia, China, República Tcheca e EUA), classe D40
      "buraco". Severidade é um índice relativo (área corrigida por perspectiva), não uma medida física. Decisões
      finais são humanas.</p>`;
}

/* ------------------------------------------------------------------ início */
async function analisarImagemUrl(url) {
  novaFonte("foto");
  const img = new Image();
  await new Promise((ok, falha) => { img.onload = ok; img.onerror = falha; img.src = url; });
  await analisarFoto(img, url.split("/").pop());
}

iniciarMapaLocal();
(async () => {
  try {
    st.modelo = await (await fetch("/v1/modelo")).json();
    $("#conf").value = st.modelo.limiar_confianca.toFixed(2);
    $("#conf").dispatchEvent(new Event("input"));
    status(`Modelo ${st.modelo.modelo} pronto.`);
  } catch { status("API indisponível."); return; }
  // Modo demonstração: /?demo=/static/amostras/x.jpg&modo=aparelho  ·  /?aba=zeladoria
  const q = new URLSearchParams(location.search);
  if (q.get("modo")) $("#modo").value = q.get("modo");
  if (q.get("aba")) { const a = document.getElementById("tab-" + q.get("aba")); if (a) ativarAba(a); }
  if (q.get("demo")) {
    try { await analisarImagemUrl(q.get("demo")); } catch (e) { status("Falha no exemplo: " + e.message); }
  }
})();
