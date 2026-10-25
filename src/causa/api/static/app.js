// causa dashboard: vanilla JS, no build step. Every number shown comes from the API.
"use strict";

const $ = (sel) => document.querySelector(sel);
const KEY_STORE = "causa-api-key";

function esc(value) {
  return String(value).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function fmt(x, digits = 2) {
  if (x === null || x === undefined || Number.isNaN(x)) return "n/a";
  return Number(x).toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

function pct(x, digits = 1) {
  return x === null || x === undefined ? "n/a" : `${fmt(100 * x, digits)}%`;
}

function readKey() {
  try { return sessionStorage.getItem(KEY_STORE) || ""; } catch { return ""; }
}

function saveKey(value) {
  try { sessionStorage.setItem(KEY_STORE, value); } catch { /* storage unavailable: key lives in the input only */ }
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  const key = $("#api-key").value.trim();
  if (key) headers.set("X-API-Key", key);
  const response = await fetch(path, { ...options, headers });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail || body);
    throw new Error(`${response.status}: ${detail}`);
  }
  return body;
}

// Minimal SVG line chart with an optional band and reference lines.
function chart(container, spec) {
  const W = 640, H = 260, L = 62, R = 14, T = 12, B = 38;
  const xs = spec.xs;
  const ys = [...(spec.lines || []).flatMap((l) => l.ys), ...(spec.band ? [...spec.band.lo, ...spec.band.hi] : []),
    ...(spec.hlines || []).map((h) => h.y)].filter(Number.isFinite);
  if (!xs.length || !ys.length) { container.innerHTML = '<p class="muted">No data.</p>'; return; }
  let [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  if (y0 === y1) { y0 -= 1; y1 += 1; }
  const pad = 0.06 * (y1 - y0); y0 -= pad; y1 += pad;
  const [x0, x1] = [Math.min(...xs), Math.max(...xs) === Math.min(...xs) ? Math.min(...xs) + 1 : Math.max(...xs)];
  const sx = (x) => L + ((x - x0) / (x1 - x0)) * (W - L - R);
  const sy = (y) => T + (1 - (y - y0) / (y1 - y0)) * (H - T - B);
  const path = (px, py) => px.map((x, i) => `${i ? "L" : "M"}${sx(x).toFixed(1)},${sy(py[i]).toFixed(1)}`).join("");
  let svg = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">`;
  for (let i = 0; i <= 4; i++) {
    const v = y0 + (i / 4) * (y1 - y0);
    svg += `<line x1="${L}" x2="${W - R}" y1="${sy(v)}" y2="${sy(v)}" stroke="var(--line)" stroke-width="1"/>`;
    svg += `<text x="${L - 6}" y="${sy(v) + 4}" text-anchor="end">${esc(fmt(v, Math.abs(v) < 10 ? 2 : 0))}</text>`;
  }
  const ticks = xs.length <= 16 ? xs : xs.filter((_, i) => i % Math.ceil(xs.length / 12) === 0);
  for (const x of ticks) svg += `<text x="${sx(x)}" y="${H - B + 16}" text-anchor="middle">${esc(spec.xFormat ? spec.xFormat(x) : x)}</text>`;
  svg += `<text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">${esc(spec.xLabel || "")}</text>`;
  if (spec.band) {
    const top = xs.map((x, i) => `${i ? "L" : "M"}${sx(x).toFixed(1)},${sy(spec.band.hi[i]).toFixed(1)}`).join("");
    const bottom = [...xs].reverse().map((x, i) => `L${sx(x).toFixed(1)},${sy(spec.band.lo[xs.length - 1 - i]).toFixed(1)}`).join("");
    svg += `<path d="${top}${bottom}Z" fill="var(--band)" stroke="none"/>`;
  }
  for (const h of spec.hlines || []) {
    svg += `<line x1="${L}" x2="${W - R}" y1="${sy(h.y)}" y2="${sy(h.y)}" stroke="${h.color || "var(--muted)"}" stroke-dasharray="${h.dash || "4 4"}"/>`;
    if (h.label) svg += `<text x="${W - R - 4}" y="${sy(h.y) - 4}" text-anchor="end">${esc(h.label)}</text>`;
  }
  for (const l of spec.lines || []) {
    svg += `<path d="${path(xs, l.ys)}" fill="none" stroke="${l.color || "var(--accent)"}" stroke-width="2"/>`;
  }
  container.innerHTML = `${svg}</svg>`;
}

const VERDICTS = {
  ship: "SHIP", do_not_ship: "DO NOT SHIP", keep_running: "KEEP RUNNING", stop_no_effect: "STOP: NO EFFECT", invalid: "INVALID",
};

function renderReport(record) {
  const r = record.report;
  const p = r.primary, pw = r.power;
  const box = $("#decision");
  box.classList.remove("hidden");
  box.innerHTML = `<div class="verdict v-${esc(r.decision)}">${esc(VERDICTS[r.decision] || r.decision)}</div>
    <div class="muted">${esc(record.name)} (report ${esc(record.id)})</div>
    <ul>${r.reasons.map((x) => `<li>${esc(x)}</li>`).join("")}${r.warnings.map((x) => `<li class="muted">${esc(x)}</li>`).join("")}</ul>`;
  const rows = [
    ["Effect (" + p.method + ")", fmt(p.effect)],
    ["Relative effect", pct(p.relative_effect, 2)],
    ["Always-valid 95% interval", `${fmt(p.always_valid_ci[0])} to ${fmt(p.always_valid_ci[1])}`],
    ["Always-valid p-value", fmt(p.always_valid_p_value, 4)],
    ["Fixed-horizon interval (one look only)", `${fmt(p.fixed_ci[0])} to ${fmt(p.fixed_ci[1])}`],
    ["Unadjusted effect", `${fmt(r.unadjusted.effect)} (SE ${fmt(r.unadjusted.se)})`],
    ["Variance removed by CUPED", pct(pw.variance_reduction)],
    ["Power at the MDE of " + fmt(pw.mde), pct(pw.power_at_mde)],
    ["Users per arm for target power", pw.required_n_per_arm.toLocaleString()],
  ];
  $("#effect-table").innerHTML = rows.map(([k, v]) => `<tr><td>${esc(k)}</td><td class="num">${esc(v)}</td></tr>`).join("");
  const checks = [
    ["Users (control / treatment)", `${pw.n_control.toLocaleString()} / ${pw.n_treatment.toLocaleString()}`],
    ["SRM chi-square p-value", fmt(r.srm.p_value, 4)],
    ["SRM sequential p-value", fmt(r.srm_sequential_p_value, 4)],
    ...r.guardrails.map((g) => [`Guardrail ${g.metric} (margin ${fmt(g.max_harm, 3)})`,
      `<span class="status-${esc(g.status)}">${esc(g.status)}</span> harm ${esc(fmt(g.harm, 4))}`]),
  ];
  $("#checks-table").innerHTML = checks.map(([k, v]) => `<tr><td>${esc(k)}</td><td class="num">${v.includes("<span") ? v : esc(v)}</td></tr>`).join("");
  const looks = r.looks;
  chart($("#cs-chart"), {
    xs: looks.map((l) => l.day),
    lines: [{ ys: looks.map((l) => l.effect) }],
    band: { lo: looks.map((l) => l.ci_low), hi: looks.map((l) => l.ci_high) },
    hlines: [{ y: 0, label: "no effect" }, { y: pw.mde, label: "MDE", dash: "2 4" }, { y: -pw.mde, dash: "2 4" }],
    xLabel: "day of the experiment",
  });
}

async function runSimulation(event) {
  event.preventDefault();
  const form = new FormData(event.target);
  const button = event.target.querySelector("button");
  button.disabled = true;
  try {
    const body = {
      n_users: Number(form.get("n_users")),
      lift: Number(form.get("lift")) / 100,
      srm_drop: Number(form.get("srm_drop")) / 100,
      support_harm: Number(form.get("support_harm")) / 100,
      latency_shift_ms: Number(form.get("latency_shift_ms")),
      seed: Number(form.get("seed")),
    };
    renderReport(await api("/v1/experiments/simulate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }));
    loadHistory();
  } catch (err) {
    showError("#decision", err);
  } finally {
    button.disabled = false;
  }
}

async function uploadCsv(event) {
  const file = event.target.files[0];
  if (!file) return;
  const data = new FormData();
  data.append("file", file);
  data.append("spec", JSON.stringify({ name: file.name, guardrails: [] }));
  try {
    renderReport(await api("/v1/experiments/csv", { method: "POST", body: data }));
    loadHistory();
  } catch (err) {
    showError("#decision", err);
  }
}

function showError(sel, err) {
  const el = $(sel);
  el.classList.remove("hidden");
  el.innerHTML = `<p class="error">${esc(err.message)}</p>`;
}

async function loadModel() {
  try {
    const m = await api("/v1/targeting/model");
    const best = m.learner;
    const qinis = Object.entries(m.validation).map(([k, v]) => [k, v.qini]);
    const top = Math.max(...qinis.map(([, q]) => q), 1e-9);
    $("#qini-bars").innerHTML = qinis.map(([k, q]) => `<div class="bar-row"><span>${esc(k)}</span>
      <div class="bar${k === best ? " best" : ""}" style="width:${Math.max(0, (100 * q) / top).toFixed(1)}%"></div>
      <span class="num">${esc(fmt(q, 2))}</span></div>`).join("") + `<p class="muted">Serving: ${esc(best)}, trained on ${m.n_train.toLocaleString()} customers.</p>`;
    chart($("#budget-chart"), {
      xs: m.budget_curve.budgets,
      lines: [{ ys: m.budget_curve.ipw_value_per_customer }],
      hlines: [{ y: 0 }],
      xLabel: "share of customers who get a coupon",
      xFormat: (x) => `${Math.round(100 * x)}%`,
    });
  } catch (err) {
    showError("#qini-bars", err);
  }
}

async function runTargeting(event) {
  event.preventDefault();
  const form = new FormData(event.target);
  try {
    const customers = await api("/v1/customers/sample?n=500&seed=11");
    const body = { customers, budget: Number(form.get("budget")) / 100, cost_per_contact: Number(form.get("cost")) };
    const res = await api("/v1/targeting", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    $("#target-summary").textContent = `${res.model}: coupons for ${res.n_treated} of ${res.n_customers} customers, expected incremental revenue ${fmt(res.expected_incremental_revenue, 0)} INR (${fmt(res.expected_incremental_revenue_per_customer)} per customer scored).`;
    const top = [...res.results].sort((a, b) => a.rank - b.rank).slice(0, 10);
    $("#target-table").innerHTML = `<tr><th>rank</th><th>customer</th><th class="num">predicted uplift (INR)</th><th>coupon</th></tr>` +
      top.map((r) => `<tr><td>${r.rank}</td><td>${esc(r.customer_id ?? "")}</td><td class="num">${esc(fmt(r.uplift))}</td><td>${r.treat ? "yes" : "no"}</td></tr>`).join("");
  } catch (err) {
    showError("#target-summary", err);
  }
}

async function loadBenchmark() {
  try {
    const b = await api("/v1/benchmark");
    const ate = b.ate.scenarios.moderate_overlap;
    const s = b.sequential;
    const rows = Object.entries(ate.estimators).map(([k, e]) => `<tr><td>${esc(k)}</td><td class="num">${fmt(e.bias)}</td><td class="num">${fmt(e.rmse)}</td><td class="num">${fmt(e.coverage, 3)}</td></tr>`).join("");
    const learners = ["s_learner", "t_learner", "x_learner", "dr_learner"].map((k) => {
      const r = b.cate.learners[k];
      return `<tr><td>${k}</td><td class="num">${fmt(r.pehe_rel, 3)}</td><td class="num">${fmt(r.qini_rel, 3)}</td><td class="num">${fmt(r.policy_value_rel, 3)}</td></tr>`;
    }).join("");
    $("#bench-body").innerHTML = `<div class="two-col"><div><h3>ATE, ${ate.replications} replications (true ATE ${fmt(ate.true_ate)})</h3>
      <table class="data"><tr><th>estimator</th><th class="num">bias</th><th class="num">RMSE</th><th class="num">coverage</th></tr>${rows}</table></div>
      <div><h3>CATE learners (${b.cate.seeds} seeds)</h3><table class="data"><tr><th>learner</th><th class="num">PEHE / SD</th><th class="num">Qini / oracle</th><th class="num">value / oracle</th></tr>${learners}</table>
      <h3>Experiments</h3><table class="kv">
      <tr><td>A/A false positives, peeking at ${s.looks} looks</td><td class="num">${pct(s.aa.reject.naive.rate)}</td></tr>
      <tr><td>A/A false positives, mixture SPRT</td><td class="num">${pct(s.aa.reject.msprt.rate)}</td></tr>
      <tr><td>CUPED variance reduction</td><td class="num">${pct(b.cuped.variance_reduction)}</td></tr></table></div></div>
      <p class="muted">Seed ${b.seed}, commit ${esc(b.environment.git_commit)}, ${b.timings_seconds.total} s.</p>`;
  } catch (err) {
    $("#bench-body").innerHTML = `<p class="muted">${esc(err.message)}</p>`;
  }
}

async function loadHistory() {
  try {
    const items = await api("/v1/experiments?limit=10");
    $("#history-table").innerHTML = `<tr><th>created</th><th>name</th><th>decision</th><th>id</th></tr>` +
      items.map((i) => `<tr data-id="${esc(i.id)}"><td>${esc(i.created_at)}</td><td>${esc(i.name)}</td><td class="v-${esc(i.decision)}">${esc(VERDICTS[i.decision] || i.decision)}</td><td><code>${esc(i.id)}</code></td></tr>`).join("");
  } catch (err) {
    $("#history-table").innerHTML = `<tr><td class="muted">${esc(err.message)}</td></tr>`;
  }
}

async function openRecord(event) {
  const row = event.target.closest("tr[data-id]");
  if (!row) return;
  try { renderReport(await api(`/v1/experiments/${encodeURIComponent(row.dataset.id)}`)); } catch (err) { showError("#decision", err); }
}

async function loadHealth() {
  try {
    const h = await api("/health");
    $("#health").textContent = `ok, model ${h.model_learner || "not loaded"}, ${h.experiments_stored} reports`;
  } catch (err) {
    $("#health").textContent = "unreachable";
  }
}

function init() {
  $("#api-key").value = readKey();
  $("#api-key").addEventListener("change", (e) => { saveKey(e.target.value.trim()); refresh(); });
  $("#sim-form").addEventListener("submit", runSimulation);
  $("#csv-file").addEventListener("change", uploadCsv);
  $("#target-form").addEventListener("submit", runTargeting);
  $("#target-form").budget.addEventListener("input", (e) => { $("#budget-out").textContent = `${e.target.value}%`; });
  $("#history-table").addEventListener("click", openRecord);
  refresh();
}

function refresh() {
  loadHealth();
  loadModel();
  loadBenchmark();
  loadHistory();
}

init();
