/* Parikshan AI front end: plain JavaScript, no build step.
   Talks to FastAPI: POST /api/chat/stream (Server-Sent Events), /api/threads, /api/reports, /api/compare. */

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const state = { threadId: null, health: null, busy: false, sources: {} };

// ---------------------------------------------------------------- icons (inline SVG, stroke = currentColor)
const P = 'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"';
const ICONS = {
  plus: `<svg viewBox="0 0 24 24" ${P}><path d="M12 5v14M5 12h14"/></svg>`,
  play: `<svg viewBox="0 0 24 24" ${P}><path d="M7 5l12 7-12 7z"/></svg>`,
  sun: `<svg viewBox="0 0 24 24" ${P}><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>`,
  moon: `<svg viewBox="0 0 24 24" ${P}><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>`,
  trials: `<svg viewBox="0 0 24 24" ${P}><rect x="5" y="4" width="14" height="17" rx="2"/><path d="M9 4h6v3H9zM9 12h6M9 16h4"/></svg>`,
  papers: `<svg viewBox="0 0 24 24" ${P}><path d="M4 5a2 2 0 0 1 2-2h9l5 5v11a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2z"/><path d="M14 3v5h5M8 13h8M8 17h6"/></svg>`,
  safety: `<svg viewBox="0 0 24 24" ${P}><path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/><path d="M12 8v5M12 16h.01"/></svg>`,
  extract: `<svg viewBox="0 0 24 24" ${P}><path d="M4 7h16M4 12h10M4 17h7"/><circle cx="18" cy="16" r="3"/><path d="M20.2 18.2L22 20"/></svg>`,
  report: `<svg viewBox="0 0 24 24" ${P}><path d="M6 3h9l4 4v14H6z"/><path d="M14 3v5h5M9 13h7M9 17h7M9 9h2"/></svg>`,
  plan: `<svg viewBox="0 0 24 24" ${P}><path d="M9 18h6M10 21h4M12 3a6 6 0 0 0-4 10.5c.7.7 1 1.5 1 2.5h6c0-1 .3-1.8 1-2.5A6 6 0 0 0 12 3z"/></svg>`,
  answer: `<svg viewBox="0 0 24 24" ${P}><path d="M5 12l5 5L20 7"/></svg>`,
  think: `<svg viewBox="0 0 24 24" ${P}><path d="M12 3a6 6 0 0 0-4 10.5c.7.7 1 1.5 1 2.5h6c0-1 .3-1.8 1-2.5A6 6 0 0 0 12 3zM10 21h4"/></svg>`,
  tool: `<svg viewBox="0 0 24 24" ${P}><path d="M14.7 6.3a4 4 0 0 0-5.4 5.2L3 17.8V21h3.2l6.3-6.3a4 4 0 0 0 5.2-5.4l-2.6 2.6-2.4-.6-.6-2.4z"/></svg>`,
  check: `<svg viewBox="0 0 24 24" ${P}><path d="M5 12l5 5L20 7"/></svg>`,
  x: `<svg viewBox="0 0 24 24" ${P}><path d="M6 6l12 12M18 6L6 18"/></svg>`,
  close: `<svg viewBox="0 0 24 24" ${P}><path d="M6 6l12 12M18 6L6 18"/></svg>`,
  trash: `<svg viewBox="0 0 24 24" ${P}><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/></svg>`,
  download: `<svg viewBox="0 0 24 24" ${P}><path d="M12 4v11M7 10l5 5 5-5M5 20h14"/></svg>`,
  open: `<svg viewBox="0 0 24 24" ${P}><path d="M4 4h7M4 4v16h16v-7M14 4h6v6M20 4l-9 9"/></svg>`,
  chevron: `<svg viewBox="0 0 24 24" ${P}><path d="M6 9l6 6 6-6"/></svg>`,
};
const icon = name => `<span class="ico">${ICONS[name] || ""}</span>`;
function hydrateIcons(root = document) {
  $$(".ico[data-icon]", root).forEach(el => { el.innerHTML = ICONS[el.dataset.icon] || ""; el.removeAttribute("data-icon"); });
}

// ---------------------------------------------------------------- theme
const systemDark = () => matchMedia("(prefers-color-scheme: dark)").matches;
const currentTheme = () => document.documentElement.dataset.theme || (systemDark() ? "dark" : "light");
function paintThemeButton() {
  const dark = currentTheme() === "dark";
  const b = $("#themeBtn");
  b.innerHTML = icon(dark ? "sun" : "moon");
  b.setAttribute("aria-label", dark ? "Switch to light mode" : "Switch to dark mode");
  b.title = dark ? "Light mode" : "Dark mode";
}
$("#themeBtn").addEventListener("click", () => {
  const next = currentTheme() === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("parikshan-theme", next); } catch (e) { /* storage blocked */ }
  paintThemeButton();
});
matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", paintThemeButton);

// ---------------------------------------------------------------- helpers
const fmt = n => Number(n || 0).toLocaleString("en-IN");
const usd = n => "$" + Number(n || 0).toFixed(n >= 0.01 ? 4 : 6);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}

const STAGES = [
  ["plan", "Plan", "plan"], ["trials", "Trials", "trials"], ["papers", "Literature", "papers"],
  ["safety", "FDA safety", "safety"], ["extract", "Extract", "extract"], ["report", "Report", "report"], ["answer", "Answer", "answer"],
];
const TOOL_STAGE = {
  search_clinical_trials: "trials", get_trial_details: "trials", search_pubmed: "papers",
  get_drug_safety_data: "safety", extract_entities: "extract", write_report: "report",
};

// ---------------------------------------------------------------- citations -> numbered superscripts
const CITE_GROUP = /\[((?:\s*(?:NCT\d{8}|PMID:\s*\d+|FAERS:[\w \-]+|LABEL:[\w \-]+)\s*[,;]?)+)\]/gi;
const CITE_ID = /NCT\d{8}|PMID:\s*\d+|FAERS:[\w \-]+|LABEL:[\w \-]+/gi;

function normId(raw) {
  const r = raw.trim();
  if (/^(NCT|PMID)/i.test(r)) return r.replace(/\s+/g, "").toUpperCase();
  const [p, ...rest] = r.split(":");
  return `${p.toUpperCase()}:${rest.join(":").trim().toLowerCase()}`;
}
function citeUrl(id) {
  if (/^NCT\d{8}$/.test(id)) return `https://clinicaltrials.gov/study/${id}`;
  if (/^PMID:\d+$/.test(id)) return `https://pubmed.ncbi.nlm.nih.gov/${id.split(":")[1]}/`;
  if (/^FAERS:/.test(id)) return "https://open.fda.gov/data/faers/";
  if (/^LABEL:/.test(id)) return `https://dailymed.nlm.nih.gov/dailymed/search.cfm?query=${encodeURIComponent(id.split(":")[1])}`;
  return "#";
}
const kindOf = id => /^NCT/.test(id) ? "ClinicalTrials.gov" : /^PMID/.test(id) ? "PubMed" : /^FAERS/.test(id) ? "FDA FAERS" : "FDA label";

/* Renders Markdown, replacing citation brackets with numbered superscript links.
   Returns {html, refs}. `meta` maps id -> {title, url}. */
let refSeq = 0;
function renderCited(md, { meta = {}, unverified = [] } = {}) {
  const bad = new Set(unverified.map(x => x.toLowerCase()));
  const scope = "r" + (++refSeq);
  const order = [];
  const num = id => { if (!order.includes(id)) order.push(id); return order.indexOf(id) + 1; };
  md = (md || "").replace(/\n-{3,}\s*\n+## Sources[\s\S]*$/, "");  // older reports stored a sources block
  let html = DOMPurify.sanitize(marked.parse(md));
  html = html.replace(CITE_GROUP, (_, inner) => {
    const links = (inner.match(CITE_ID) || []).map(raw => {
      const id = normId(raw);
      const n = num(id);
      const m = meta[id] || state.sources[id] || {};
      const isBad = bad.has(id.toLowerCase());
      const cls = isBad ? "bad" : /^(FAERS|LABEL)/.test(id) ? "safety" : "";
      const tip = isBad ? `${id}: not found in fetched evidence` : `${id}${m.title ? ": " + m.title : ""}`;
      return `<a class="${cls}" href="#${scope}-${n}" title="${esc(tip)}">${n}${isBad ? "?" : ""}</a>`;
    });
    return `<sup class="cite">[${links.join(",")}]</sup>`;
  }).replace(/\s+<sup class="cite">/g, '<sup class="cite">');
  const refs = order.map((id, i) => {
    const m = meta[id] || state.sources[id] || {};
    return { n: i + 1, id, title: m.title || "", url: m.url || citeUrl(id), bad: bad.has(id.toLowerCase()), scope };
  });
  return { html, refs };
}
function refsHtml(refs, heading = "Sources") {
  if (!refs.length) return "";
  return `<h4>${heading}</h4><ol>` + refs.map(r => `<li id="${r.scope}-${r.n}" class="${r.bad ? "bad" : ""}">
      <span class="n">[${r.n}]</span>
      <span><span class="t">${r.bad ? `${esc(r.id)}: not found in fetched evidence` : esc(r.title || r.id)}</span>
      ${r.bad ? "" : `<br><span class="meta">${kindOf(r.id)}, ${esc(r.id)}. </span><a href="${esc(r.url)}" target="_blank" rel="noopener">Open source</a>`}</span>
    </li>`).join("") + "</ol>";
}

// remember titles of everything the tools fetched, so citations in answers can show them
function learnSources(raw) {
  try {
    const d = JSON.parse(raw);
    (d.articles || []).forEach(a => state.sources[normId(a.cite_as)] = { title: a.title, url: citeUrl(normId(a.cite_as)) });
    (d.trials || []).forEach(t => state.sources[normId(t.cite_as)] = { title: t.title, url: citeUrl(normId(t.cite_as)) });
    if (d.cite_as && d.title) state.sources[normId(d.cite_as)] = { title: d.title, url: citeUrl(normId(d.cite_as)) };
    if (d.faers?.cite_as) state.sources[normId(d.faers.cite_as)] = { title: `FDA FAERS reports for ${d.faers.drug}`, url: citeUrl(normId(d.faers.cite_as)) };
    if (d.label?.cite_as) state.sources[normId(d.label.cite_as)] = { title: "FDA drug label", url: citeUrl(normId(d.label.cite_as)) };
    return d;
  } catch { return null; }
}

// ---------------------------------------------------------------- tabs
$$(".tab").forEach(t => t.addEventListener("click", () => {
  $$(".tab").forEach(x => { x.classList.toggle("active", x === t); x.setAttribute("aria-selected", x === t); });
  $$("main.view").forEach(v => v.hidden = v.id !== "view-" + t.dataset.view);
}));

// ---------------------------------------------------------------- health + threads
async function loadHealth() {
  try {
    const h = state.health = await api("/api/health");
    const p = h.price_per_1m_tokens_usd;
    $("#modelText").textContent = `${h.model}  ·  ${h.tier} tier  ·  $${p.input} in / $${p.output} out per 1M`;
    $("#statusDot").className = "status-dot " + (h.status === "ok" ? "ok" : "bad");
    $("#mTier").textContent = h.tier === "free"
      ? "Free tier: you are billed $0. This is what the run would cost on the paid tier."
      : "Paid tier: this is the billed cost.";
    if (h.status !== "ok") { const b = $("#keyBanner"); b.hidden = false; b.textContent = `Agent is offline: ${h.error}`; }
  } catch {
    $("#modelText").textContent = "server offline";
    $("#statusDot").className = "status-dot bad";
  }
}

async function loadThreads() {
  const list = $("#threadList");
  const threads = await api("/api/threads").catch(() => []);
  list.innerHTML = threads.length ? "" : `<li class="muted small" style="padding:6px 10px">No threads yet</li>`;
  for (const t of threads) {
    const li = document.createElement("li");
    li.classList.toggle("active", t.thread_id === state.threadId);
    li.innerHTML = `<button class="open" title="${esc(t.title)}"><span>${esc(t.title)}</span><small>${t.runs} run${t.runs === 1 ? "" : "s"}, ${usd(t.cost_usd)}</small></button>
      <button class="del" aria-label="Delete thread">${icon("trash")}</button>`;
    li.querySelector(".open").onclick = () => openThread(t.thread_id);
    li.querySelector(".del").onclick = async () => {
      if (!confirm("Delete this thread and its memory?")) return;
      await api(`/api/threads/${t.thread_id}`, { method: "DELETE" });
      if (state.threadId === t.thread_id) newThread();
      loadThreads();
    };
    list.appendChild(li);
  }
  const s = await api("/api/stats").catch(() => null);
  $("#lifetimeStats").innerHTML = s && s.runs ? `<dl>
      <dt>All runs</dt><dd>${fmt(s.runs)}</dd>
      <dt>Tokens in</dt><dd>${fmt(s.input_tokens)}</dd>
      <dt>Tokens out</dt><dd>${fmt(s.output_tokens)}</dd>
      <dt>Paid-tier cost</dt><dd>${usd(s.cost_usd)}</dd>
      <dt>Completion rate</dt><dd>${Math.round(s.completion_rate * 100)}%</dd></dl>` : "";
}

function newThread() {
  state.threadId = null;
  $("#feed").innerHTML = "";
  $("#feed").appendChild(emptyState);
  resetMeter(); resetReport(); loadThreads();
}

async function openThread(id) {
  state.threadId = id;
  const data = await api(`/api/threads/${id}`);
  (data.sources || []).forEach(s => state.sources[normId(s.source_id)] = { title: s.title, url: s.url });
  const feed = $("#feed");
  feed.innerHTML = "";
  resetReport();
  for (const r of [...data.runs].reverse()) {
    const ui = makeRun(r.question);
    ui.el.querySelector(".trace-title").textContent = "Agent run";
    ui.meta.textContent = `${r.steps} steps, ${r.tools_used.length} tool calls, ${fmt(r.input_tokens)} in / ${fmt(r.output_tokens)} out, ${usd(r.cost_usd)}`;
    r.tools_used.forEach(t => {
      ui.trace.insertAdjacentHTML("beforeend", `<li class="step act done-ok"><span class="node">${icon("check")}</span><span class="kind">Act</span><span class="tool">${esc(t)}</span></li>`);
      ui.setStage(TOOL_STAGE[t], "done");
    });
    ui.setStage("plan", "done");
    ui.finish(r.status === "completed");
    ui.el.querySelector(".trace-wrap").classList.add("collapsed");
    ui.showAnswer(r.answer || r.error || "", r.status !== "completed" && !r.answer);
    feed.appendChild(ui.el);
    if (r.report_id) showReport(r.report_id);
  }
  feed.scrollTop = feed.scrollHeight;
  const last = data.runs[0];
  if (last && state.health) {
    resetMeter();
    liveCalls.length = 0;
    updateMeter({ llm_calls: last.llm_calls, input_tokens: last.input_tokens, output_tokens: last.output_tokens,
      equivalent_cost_usd: last.cost_usd, equivalent_cost_inr: last.cost_usd * (state.health.usd_to_inr || 84),
      price_per_1m: state.health.price_per_1m_tokens_usd, calls: [] }, last.tools_used.length);
    $("#runState").textContent = "last run";
  }
  loadThreads();
}

// ---------------------------------------------------------------- meter
function resetMeter() {
  ["#mCalls", "#mTools", "#mIn", "#mOut"].forEach(s => $(s).textContent = "0");
  $("#mCost").textContent = "$0.0000"; $("#mInr").textContent = "≈ ₹0.00";
  $("#mInCost").textContent = "$0"; $("#mOutCost").textContent = "$0";
  $("#splitIn").style.width = $("#splitOut").style.width = "0";
  $("#callBars").innerHTML = ""; $("#callsBox").hidden = true;
  $("#runState").textContent = "idle";
}

const liveCalls = [];
function updateMeter(u, toolCount) {
  $("#mCalls").textContent = fmt(u.llm_calls);
  $("#mIn").textContent = fmt(u.input_tokens);
  $("#mOut").textContent = fmt(u.output_tokens);
  if (toolCount !== undefined) $("#mTools").textContent = toolCount;
  const pin = u.price_per_1m.input, pout = u.price_per_1m.output;
  const calls = u.calls || liveCalls;
  if (!u.calls && u.last_call && liveCalls.length < u.llm_calls) liveCalls.push(u.last_call);
  // split the total cost into its input and output parts (scaled, since a backup model may have other prices)
  const rawIn = u.input_tokens / 1e6 * pin, rawOut = u.output_tokens / 1e6 * pout;
  const scale = (rawIn + rawOut) ? u.equivalent_cost_usd / (rawIn + rawOut) : 0;
  const inCost = rawIn * scale, outCost = rawOut * scale;
  const total = (inCost + outCost) || 1;
  $("#splitIn").style.width = (inCost / total * 100) + "%";
  $("#splitOut").style.width = (outCost / total * 100) + "%";
  $("#mInCost").textContent = usd(inCost); $("#mOutCost").textContent = usd(outCost);
  $("#mCost").textContent = usd(u.equivalent_cost_usd);
  $("#mInr").textContent = `≈ ₹${Number(u.equivalent_cost_inr).toFixed(2)}`;
  drawCallBars(calls);
}

function drawCallBars(calls) {
  if (!calls.length) return;
  $("#callsBox").hidden = false;
  const max = Math.max(...calls.map(c => c.input_tokens + c.output_tokens), 1);
  $("#callBars").innerHTML = calls.map(c => {
    const tot = c.input_tokens + c.output_tokens;
    return `<li title="${esc(c.purpose)}: ${fmt(c.input_tokens)} in, ${fmt(c.output_tokens)} out, ${fmt(c.latency_ms)} ms, ${esc(c.model || "")}">
      <span class="lbl">${esc(c.purpose.replace(/_/g, " "))}</span>
      <span class="bar" style="width:${Math.max(tot / max * 100, 4)}%"><i class="in" style="width:${c.input_tokens / (tot || 1) * 100}%"></i><i class="out" style="width:${c.output_tokens / (tot || 1) * 100}%"></i></span>
      <span class="num">${fmt(tot)}</span></li>`;
  }).join("");
}

// ---------------------------------------------------------------- report card + reader
function resetReport() {
  $("#reportCard").className = "report-empty";
  $("#reportCard").textContent = "When the agent writes a report, it shows up here with verified citations and a PDF download.";
}

function reportMeta(rep) {
  const meta = {};
  (rep.citations || []).forEach(c => meta[normId(c.id)] = { title: c.title, url: c.url });
  return meta;
}

function renderReport(rep) {
  const id = rep.report_id || rep.id;
  const verified = rep.citations?.length || 0;
  const bad = rep.unverified_citations || [];
  const md = (rep.markdown || "").replace(/^#\s+.+\n?/, "");
  const summary = (md.match(/## Executive summary\s*\n([\s\S]*?)(\n## |$)/) || [, md])[1]
    .replace(/\[[^\]]*(NCT|PMID|FAERS|LABEL)[^\]]*\]/g, "").replace(/^\s*[-*+]\s+/gm, "").replace(/[*#>|]/g, "")
    .replace(/\s+([.,;])/g, "$1").replace(/\s+/g, " ").trim();
  const card = $("#reportCard");
  card.className = "report-card";
  card.innerHTML = `<h4>${esc(rep.title)}</h4>
    <p class="verified">${icon("check")} ${verified} citation${verified === 1 ? "" : "s"} verified against fetched evidence</p>
    ${bad.length ? `<p class="warn">${bad.length} citation(s) not found in fetched evidence: ${bad.map(esc).join(", ")}</p>` : ""}
    <p class="preview">${esc(summary.slice(0, 320))}</p>
    <div class="actions">
      <button class="btn primary small" id="openReport">${icon("open")} Open report</button>
      <a class="btn small" href="/api/reports/${id}/pdf" download>${icon("download")} PDF</a>
    </div>`;
  $("#openReport").onclick = () => openReader(rep);
}

function openReader(rep) {
  const id = rep.report_id || rep.id;
  const bad = rep.unverified_citations || [];
  const { html, refs } = renderCited((rep.markdown || "").replace(/^#\s+.+\n?/, ""), { meta: reportMeta(rep), unverified: bad });
  $("#readerTitle").textContent = rep.title;
  $("#readerMeta").textContent = `${rep.citations?.length || 0} verified sources${bad.length ? `, ${bad.length} to check` : ""}`;
  $("#readerPdf").href = `/api/reports/${id}/pdf`;
  $("#readerBody").innerHTML =
    (bad.length ? `<p class="warn">Citations marked with ? were not in the evidence the agent fetched: ${bad.map(esc).join(", ")}</p>` : "") +
    `<div class="md">${html}</div><div class="refs">${refsHtml(refs, "References")}</div>`;
  $("#reader").showModal();
  $("#readerBody").scrollTop = 0;
}
$("#readerClose").onclick = () => $("#reader").close();
$("#reader").addEventListener("click", e => { if (e.target.id === "reader") $("#reader").close(); });

async function showReport(id) {
  try { renderReport(await api(`/api/reports/${id}`)); } catch { /* report deleted */ }
}

// ---------------------------------------------------------------- one run (UI object)
const tpl = $("#runTpl");
const emptyState = $("#emptyState");

function makeRun(question) {
  const el = tpl.content.firstElementChild.cloneNode(true);
  hydrateIcons(el);
  el.querySelector(".q").textContent = question;
  const stages = el.querySelector(".stages");
  stages.innerHTML = STAGES.map(([k, label, ic]) => `<li class="stage" data-stage="${k}">${icon(ic)}${label}</li>`).join("");
  const wrap = el.querySelector(".trace-wrap");
  const toggle = el.querySelector(".trace-toggle");
  toggle.onclick = () => { wrap.classList.toggle("collapsed"); toggle.setAttribute("aria-expanded", !wrap.classList.contains("collapsed")); };
  const answer = el.querySelector(".answer");
  const ui = {
    el, trace: el.querySelector(".trace"), meta: el.querySelector(".trace-meta"),
    setStage(key, status) {
      const s = stages.querySelector(`[data-stage="${key}"]`);
      if (!s || (s.classList.contains("done") && status !== "done")) return;
      s.classList.remove("active", "failed", "skipped");
      s.classList.add(status);
    },
    finish(ok) {
      stages.querySelectorAll(".stage").forEach(s => { if (!s.classList.contains("done") && !s.classList.contains("failed")) s.classList.add("skipped"); s.classList.remove("active"); });
      if (ok) { const a = stages.querySelector('[data-stage="answer"]'); a.classList.remove("skipped"); a.classList.add("done"); }
    },
    showAnswer(text, isError = false) {
      answer.hidden = false;
      answer.classList.toggle("error", isError);
      if (isError) { answer.querySelector(".answer-body").textContent = text; return; }
      const { html, refs } = renderCited(text);
      answer.querySelector(".answer-body").innerHTML = html;
      answer.querySelector(".refs").innerHTML = refsHtml(refs);
      const kinds = { trials: 0, papers: 0, fda: 0 };
      refs.forEach(r => /^NCT/.test(r.id) ? kinds.trials++ : /^PMID/.test(r.id) ? kinds.papers++ : kinds.fda++);
      answer.querySelector(".source-count").innerHTML =
        (kinds.trials ? `<span class="count-chip">${kinds.trials} trial${kinds.trials > 1 ? "s" : ""}</span>` : "") +
        (kinds.papers ? `<span class="count-chip">${kinds.papers} paper${kinds.papers > 1 ? "s" : ""}</span>` : "") +
        (kinds.fda ? `<span class="count-chip safety">FDA data</span>` : "");
    },
  };
  return ui;
}

function argsText(args) {
  return Object.entries(args || {}).filter(([, v]) => v !== null && v !== "")
    .map(([k, v]) => `${k}: ${Array.isArray(v) ? v.join(", ") : v}`).join("   ");
}

function previewText(d, raw) {
  if (!d) return raw.slice(0, 160);
  if (d.error) return d.error;
  const pl = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;
  if (d.articles) return `${pl(d.articles.length, "paper")} (of ${fmt(d.total_found)} found): ` + d.articles.map(a => a.cite_as).join(", ");
  if (d.trials) return `${pl(d.trials.length, "trial")} (of ${fmt(d.total_found)} registered): ` + d.trials.map(t => `${t.cite_as} ${t.status.toLowerCase().replace(/_/g, " ")}`).join(", ");
  if (d.faers) return `FAERS: ${fmt(d.faers.total_reports)} reports, ${fmt(d.faers.serious_reports)} serious. Top: ` + (d.faers.top_reactions || []).slice(0, 4).map(r => r.reaction).join(", ");
  if (d.sources) return `Extracted entities from ${pl(d.sources.length, "source")}`;
  if (d.report_id) return `Report written: ${d.verified_citations} verified citations from ${d.evidence_chunks_used} evidence chunks`;
  if (d.cite_as) return `${d.cite_as}: ${d.title || ""}`;
  return raw.slice(0, 160);
}

async function runAgent(question) {
  if (state.busy) return;
  state.busy = true;
  $("#sendBtn").disabled = true;
  emptyState.remove();
  resetMeter();
  liveCalls.length = 0;
  $("#runState").textContent = "running…";
  const ui = makeRun(question);
  $("#feed").appendChild(ui.el);
  const steps = {};
  let toolCount = 0;
  const t0 = performance.now();
  const timer = setInterval(() => ui.meta.textContent = `${((performance.now() - t0) / 1000).toFixed(1)}s`, 200);
  const feed = $("#feed");
  const scroll = () => feed.scrollTop = feed.scrollHeight;
  ui.setStage("plan", "active");
  scroll();

  const handle = (type, ev) => {
    if (type === "run_started") { state.threadId = ev.thread_id; loadThreads(); }
    else if (type === "thought") {
      ui.setStage("plan", "done");
      ui.trace.insertAdjacentHTML("beforeend", `<li class="step think"><span class="node">${icon("think")}</span><span class="kind">Think</span><span class="body">${esc(ev.text)}</span></li>`);
    } else if (type === "tool_call") {
      ui.setStage("plan", "done");
      ui.setStage(TOOL_STAGE[ev.name], "active");
      toolCount++;
      const li = document.createElement("li");
      li.className = "step act pending";
      li.innerHTML = `<span class="node">${icon("tool")}</span><span class="kind">Act</span><span class="tool">${esc(ev.label)}</span> <span class="muted small">${esc(ev.name)}</span><span class="args">${esc(argsText(ev.args))}</span>`;
      ui.trace.appendChild(li);
      steps[ev.id] = { li, name: ev.name };
    } else if (type === "tool_result") {
      const s = steps[ev.id];
      const d = learnSources(ev.preview);
      ui.setStage(TOOL_STAGE[ev.name], ev.ok ? "done" : "failed");
      if (s) {
        s.li.classList.remove("pending");
        s.li.classList.add(ev.ok ? "done-ok" : "done-err");
        s.li.querySelector(".node").innerHTML = icon(ev.ok ? "check" : "x");
        s.li.insertAdjacentHTML("beforeend",
          `<details class="obs ${ev.ok ? "" : "err"}"><summary><b>Observe</b> ${esc(previewText(d, ev.preview))} <span class="muted">(${fmt(ev.ms)} ms, ${fmt(ev.chars)} chars back to the LLM)</span></summary><pre>${esc(ev.preview)}</pre></details>`);
      }
    } else if (type === "usage") updateMeter(ev, toolCount);
    else if (type === "report") { renderReport(ev); }
    else if (type === "final") {
      ui.trace.insertAdjacentHTML("beforeend", `<li class="step done"><span class="node">${icon("check")}</span><span class="kind">Answer</span><span class="muted small">No more tool calls, so the loop ends</span></li>`);
      ui.showAnswer(ev.answer);
    } else if (type === "error") {
      const m = ev.message;
      ui.showAnswer(/429|quota|rate/i.test(m) ? "Gemini rate limit reached on both models (free tier). Wait a minute and run again."
        : /503|UNAVAILABLE|high demand/i.test(m) ? "Gemini is overloaded right now on both models. Try again in a few minutes."
        : `The run stopped: ${m}`, true);
    } else if (type === "run_finished") {
      updateMeter(ev.usage, toolCount);
      ui.finish(ev.status === "completed");
      ui.el.querySelector(".trace-title").textContent = ev.status === "completed" ? "Agent run" : `Agent run (${ev.status.replace("_", " ")})`;
      clearInterval(timer);
      ui.meta.textContent = `${ev.steps} steps, ${toolCount} tool calls, ${ev.seconds}s`;
      $("#runState").textContent = ev.status === "completed" ? `done in ${ev.seconds}s` : ev.status.replace("_", " ");
    }
    scroll();
  };

  try {
    const res = await fetch("/api/chat/stream", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: question, thread_id: state.threadId }),
    });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
    await readSSE(res, handle);
  } catch (e) {
    ui.showAnswer(`Could not run the agent: ${e.message}`, true);
    ui.finish(false);
    $("#runState").textContent = "error";
  } finally {
    clearInterval(timer);
    state.busy = false;
    $("#sendBtn").disabled = false;
    loadThreads();
  }
}

// Server-Sent Events over fetch: blocks separated by a blank line, each with "event:" and "data:" lines.
async function readSSE(res, onEvent) {
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, i); buf = buf.slice(i + 2);
      let type = "message", data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) type = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (data) onEvent(type, JSON.parse(data));
    }
  }
}

// ---------------------------------------------------------------- compare tab
function statsHtml(u, ms, tools) {
  const cell = (k, v) => `<div><span>${k}</span><b>${v}</b></div>`;
  return cell("Time", `${(ms / 1000).toFixed(1)}s`) + cell("LLM calls", u.llm_calls) +
    (tools !== undefined ? cell("Tool calls", tools) : "") +
    cell("Tokens in", fmt(u.input_tokens)) + cell("Tokens out", fmt(u.output_tokens)) + cell("Cost", usd(u.equivalent_cost_usd));
}

$("#compareForm").addEventListener("submit", async e => {
  e.preventDefault();
  const q = $("#compareQ").value.trim();
  const btn = $("#compareBtn");
  btn.disabled = true;
  $("#llmAnswer").innerHTML = $("#agentAnswer").innerHTML = `<p class="muted">Running…</p>`;
  $("#llmStats").innerHTML = $("#agentStats").innerHTML = $("#agentMiniTrace").innerHTML = "";
  const llm = api("/api/compare", { method: "POST", body: JSON.stringify({ message: q }) }).then(r => {
    $("#llmStats").innerHTML = statsHtml(r.usage, r.latency_ms);
    $("#llmAnswer").innerHTML = DOMPurify.sanitize(marked.parse(r.answer || "")) +
      `<p class="warn">No sources were fetched. Specific details (trial IDs, numbers) come from training data and may be outdated or wrong.</p>`;
  }).catch(err => $("#llmAnswer").innerHTML = `<p class="warn">${esc(err.message)}</p>`);
  const t0 = performance.now();
  const agent = api("/api/chat", { method: "POST", body: JSON.stringify({ message: q }) }).then(r => {
    r.trace.filter(t => t.type === "tool_result").forEach(t => learnSources(t.preview));
    $("#agentStats").innerHTML = statsHtml(r.usage, performance.now() - t0, r.tools_used.length);
    $("#agentMiniTrace").innerHTML = r.trace.filter(t => t.type !== "tool_result")
      .map(t => t.type === "thought" ? `<li><i>${esc(t.text)}</i></li>` : `<li>Called <b>${esc(t.name)}</b> ${esc(argsText(t.args))}</li>`).join("");
    const { html, refs } = renderCited(r.answer || r.error || "No answer");
    $("#agentAnswer").innerHTML = html + `<div class="refs">${refsHtml(refs)}</div>`;
  }).catch(err => $("#agentAnswer").innerHTML = `<p class="warn">${esc(err.message)}</p>`);
  await Promise.all([llm, agent]);
  btn.disabled = false;
});

// ---------------------------------------------------------------- boot
const SAMPLES = [
  ["Full report", "", "Review the efficacy and safety of semaglutide for obesity and write a report"],
  ["Comparison", "", "Compare tirzepatide vs semaglutide: phase 3 weight-loss outcomes and GI adverse events"],
  ["Quick lookup", "quick", "Which phase 3 trials of pembrolizumab in non-small cell lung cancer are recruiting?"],
  ["Drug safety", "safety", "What are the most reported adverse events for isotretinoin, and what does its boxed warning say?"],
];
$("#samples").innerHTML = SAMPLES.map(([tag, cls, q]) =>
  `<button class="sample" data-q="${esc(q)}"><span class="tag ${cls}">${tag}</span><span class="q-text">${esc(q)}</span></button>`).join("");
$$(".sample").forEach(b => b.onclick = () => { $("#question").value = b.dataset.q; $("#question").focus(); });
$("#askForm").addEventListener("submit", e => { e.preventDefault(); const q = $("#question").value.trim(); if (q) { $("#question").value = ""; runAgent(q); } });
$("#question").addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#askForm").requestSubmit(); } });
$("#newChat").onclick = newThread;
hydrateIcons();
paintThemeButton();
loadHealth();
loadThreads();
