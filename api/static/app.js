/* VigiaVia — interface web do componente de visão computacional.
 * Modo "servidor": cada quadro vai para POST /v1/deteccoes (GPU).
 * Modo "aparelho": o modelo ONNX roda no navegador (onnxruntime-web) e a imagem nunca sai do celular.
 */
"use strict";
const $ = (s) => document.querySelector(s);
const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.webgpu.min.js";
const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
const ROTULO = { baixa: "BAIXA", media: "MÉDIA", alta: "ALTA" };
const COR = { baixa: "#ffd400", media: "#ff8c00", alta: "#ff3b30" };

const st = {
  modelo: null, rodando: false, stream: null, deteccoes: [], ort: null, sessao: null,
  ultimaVoz: 0, fpsEma: 0, mapa: null, camadaMapa: null,
};
const quadro = document.createElement("canvas");        // quadro limpo (detecção e recorte)
const tela = $("#tela"), ctx = tela.getContext("2d");
const video = $("#video");

/* ------------------------------------------------------------------ abas (teclado acessível) */
const abas = [...document.querySelectorAll('[role="tab"]')];
function ativarAba(aba) {
  abas.forEach((a) => {
    const sel = a === aba;
    a.setAttribute("aria-selected", sel);
    a.tabIndex = sel ? 0 : -1;
    $("#" + a.getAttribute("aria-controls")).hidden = !sel;
  });
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

/* ------------------------------------------------------------------ detecção no servidor */
async function detectarServidor() {
  const esc = Math.min(1, 1280 / quadro.width);
  let fonte = quadro;
  if (esc < 1) {
    fonte = document.createElement("canvas");
    fonte.width = Math.round(quadro.width * esc); fonte.height = Math.round(quadro.height * esc);
    fonte.getContext("2d").drawImage(quadro, 0, 0, fonte.width, fonte.height);
  }
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
async function detectarAparelho() {
  await prepararOnnx();
  const S = st.modelo.imgsz || 640, W = quadro.width, H = quadro.height;
  const r = Math.min(S / W, S / H), nw = Math.round(W * r), nh = Math.round(H * r);
  const px = Math.floor((S - nw) / 2), py = Math.floor((S - nh) / 2);
  const lb = document.createElement("canvas"); lb.width = S; lb.height = S;
  const lctx = lb.getContext("2d");
  lctx.fillStyle = "rgb(114,114,114)"; lctx.fillRect(0, 0, S, S);
  lctx.drawImage(quadro, px, py, nw, nh);
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
function desenhar(dets) {
  const W = tela.width, H = tela.height, u = Math.max(1, W / 1280);
  ctx.drawImage(quadro, 0, 0);
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
  $("#btn-relatar").disabled = dets.length === 0;
}

function prepararTela(w, h) {
  quadro.width = tela.width = w; quadro.height = tela.height = h;
  $("#vazio").hidden = true;
}

async function detectarQuadro() {
  const t0 = performance.now();
  const dets = $("#modo").value === "aparelho" ? await detectarAparelho() : await detectarServidor();
  st.deteccoes = dets;
  desenhar(dets);
  const fps = 1000 / (performance.now() - t0);
  st.fpsEma = st.fpsEma ? 0.85 * st.fpsEma + 0.15 * fps : fps;
  return dets;
}

/* ------------------------------------------------------------------ câmera / vídeo / foto */
async function laco() {
  while (st.rodando) {
    if (video.readyState >= 2 && !video.ended) {
      if (quadro.width !== video.videoWidth) prepararTela(video.videoWidth, video.videoHeight);
      quadro.getContext("2d").drawImage(video, 0, 0);
      try { const d = await detectarQuadro(); status(`${d.length} buraco(s) · ${st.fpsEma.toFixed(1)} FPS · ${$("#modo").value}`); }
      catch (e) { status("Erro na detecção: " + e.message); await new Promise((r) => setTimeout(r, 800)); }
    } else if (video.ended) { parar(); break; }
    await new Promise((r) => requestAnimationFrame(r));
  }
}
function parar() {
  st.rodando = false;
  st.stream?.getTracks().forEach((t) => t.stop()); st.stream = null;
  video.pause(); $("#btn-parar").hidden = true; $("#alerta").hidden = true;
}
$("#btn-parar").addEventListener("click", parar);
$("#btn-camera").addEventListener("click", async () => {
  try {
    parar();
    st.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: { ideal: 1280 } }, audio: false });
    video.srcObject = st.stream; video.hidden = false; await video.play();
    st.rodando = true; $("#btn-parar").hidden = false; laco();
  } catch (e) { status("Não foi possível abrir a câmera: " + e.message); }
});
$("#arquivo").addEventListener("change", async (ev) => {
  const f = ev.target.files[0]; if (!f) return;
  parar();
  const url = URL.createObjectURL(f);
  if (f.type.startsWith("video/")) {
    video.srcObject = null; video.src = url; video.hidden = false; await video.play();
    st.rodando = true; $("#btn-parar").hidden = false; laco();
  } else {
    const img = new Image();
    img.onload = async () => {
      prepararTela(img.naturalWidth, img.naturalHeight);
      quadro.getContext("2d").drawImage(img, 0, 0);
      status("Analisando…");
      try { const d = await detectarQuadro(); status(`${d.length} buraco(s) detectado(s).`); }
      catch (e) { status("Erro: " + e.message); }
    };
    img.src = url;
  }
});

/* ------------------------------------------------------------------ relato */
$("#btn-gps").addEventListener("click", () => {
  if (!navigator.geolocation) return status("Geolocalização indisponível neste navegador.");
  navigator.geolocation.getCurrentPosition(
    (p) => { $("#lat").value = p.coords.latitude.toFixed(6); $("#lon").value = p.coords.longitude.toFixed(6); status(`Localização obtida (±${Math.round(p.coords.accuracy)} m).`); },
    (e) => status("Localização negada — informe as coordenadas manualmente. " + e.message),
    { enableHighAccuracy: true, timeout: 10000 });
});
function recorteEvidencia(d) {
  const m = 0.25, w = d.x2 - d.x1, h = d.y2 - d.y1;
  const x = Math.max(0, d.x1 - w * m), y = Math.max(0, d.y1 - h * m);
  const x2 = Math.min(quadro.width, d.x2 + w * m), y2 = Math.min(quadro.height, d.y2 + h * m);
  const esc = Math.min(1, 480 / Math.max(x2 - x, y2 - y));
  const c = document.createElement("canvas"); c.width = Math.round((x2 - x) * esc); c.height = Math.round((y2 - y) * esc);
  c.getContext("2d").drawImage(quadro, x, y, x2 - x, y2 - y, 0, 0, c.width, c.height);
  return c.toDataURL("image/jpeg", 0.8).split(",")[1];
}
$("#btn-relatar").addEventListener("click", async () => {
  const d = [...st.deteccoes].sort((a, b) => b.confianca - a.confianca)[0];
  if (!d) return;
  const pesoSev = { baixa: 1, media: 2, alta: 3 };
  const sevMax = st.deteccoes.reduce((m, x) => (pesoSev[x.severidade] > pesoSev[m] ? x.severidade : m), "baixa");
  const corpo = { lat: parseFloat($("#lat").value), lon: parseFloat($("#lon").value), severidade: sevMax,
    confianca_max: d.confianca, origem: "cidadao", modelo: st.modelo?.modelo, evidencia_jpg_base64: recorteEvidencia(d) };
  const r = await fetch("/v1/relatos", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(corpo) });
  const j = await r.json();
  const el = $("#resposta");
  el.className = "resposta " + (r.ok ? "ok" : "erro");
  el.textContent = r.ok ? `${j.mensagem} (prioridade ${j.relato.prioridade}).` : `Erro: ${JSON.stringify(j.detail)}`;
});

/* ------------------------------------------------------------------ zeladoria */
async function carregarFila() {
  const j = await (await fetch("/v1/relatos")).json();
  const rel = j.relatos;
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
    tr.innerHTML = `<td><b>${r.prioridade}</b></td>
      <td>${r.tem_evidencia ? `<img src="/v1/relatos/${r.id}/evidencia" alt="Recorte da via com o buraco">` : "—"}</td>
      <td><span class="chip ${r.severidade}">${ROTULO[r.severidade]}</span></td>
      <td>${r.confirmacoes}</td><td>${(r.confianca_max * 100).toFixed(0)}%</td><td>${r.status}</td>
      <td>${["confirmado", "descartado", "resolvido"].map((s) => `<button class="mini" data-id="${r.id}" data-s="${s}">${s}</button>`).join("")}</td>`;
    tb.appendChild(tr);
  }
  tb.querySelectorAll("button").forEach((b) => b.addEventListener("click", async () => {
    await fetch(`/v1/relatos/${b.dataset.id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ status: b.dataset.s }) });
    carregarFila();
  }));
  if (window.L) {
    if (!st.mapa) {
      st.mapa = L.map("mapa").setView([-23.5603, -46.6505], 15);
      L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19, attribution: "© OpenStreetMap" }).addTo(st.mapa);
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
  const img = new Image();
  await new Promise((ok, falha) => { img.onload = ok; img.onerror = falha; img.src = url; });
  prepararTela(img.naturalWidth, img.naturalHeight);
  quadro.getContext("2d").drawImage(img, 0, 0);
  const d = await detectarQuadro();
  status(`${d.length} buraco(s) detectado(s) · exemplo ${url.split("/").pop()}`);
}

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
