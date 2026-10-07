const API = "/api";

const state = {
  page: 1,
  pageSize: 25,
  status: "",
  minRisk: 0,
  sort: "risk_desc",
  totalPages: 1,
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function money(amount, currency) {
  try {
    return new Intl.NumberFormat("en-US", { style: "currency", currency }).format(amount);
  } catch {
    return `${amount.toFixed(2)} ${currency}`;
  }
}

function riskTier(score) {
  if (score >= 0.5) return "critical";
  if (score >= 0.15) return "caution";
  return "safe";
}

// Avoid rendering a near-1 score as "1.000".
function fmtRisk(score) {
  return score >= 0.9995 ? ">0.999" : score.toFixed(3);
}

function confTier(conf) {
  if (conf >= 0.85) return "high";
  if (conf >= 0.6) return "mid";
  return "low";
}

// ---------------- Queue view ----------------

async function loadStats() {
  const res = await fetch(`${API}/stats`);
  const data = await res.json();
  $("#stat-total").textContent = data.total_documents.toLocaleString();
  $("#stat-pending").textContent = data.pending_high_risk.toLocaleString();
  $("#stat-approved").textContent = data.approved.toLocaleString();
  $("#stat-rejected").textContent = data.rejected.toLocaleString();
}

async function loadQueue() {
  const body = $("#docket-body");
  body.innerHTML = `<tr><td colspan="6" class="empty-row">Loading…</td></tr>`;

  const params = new URLSearchParams({
    page: state.page,
    page_size: state.pageSize,
    min_risk: state.minRisk,
    sort: state.sort,
  });
  if (state.status) params.set("status", state.status);

  const res = await fetch(`${API}/transactions?${params}`);
  const data = await res.json();
  state.totalPages = Math.max(1, data.total_pages);

  if (data.items.length === 0) {
    body.innerHTML = `<tr><td colspan="6" class="empty-row">No documents match these filters.</td></tr>`;
  } else {
    body.innerHTML = data.items.map(rowHtml).join("");
    $$(".docket tbody tr").forEach((tr) => {
      tr.addEventListener("click", () => openDetail(tr.dataset.id));
    });
  }

  $("#pager-label").textContent = `Page ${state.page} of ${state.totalPages} · ${data.total.toLocaleString()} documents`;
  $("#pager-prev").disabled = state.page <= 1;
  $("#pager-next").disabled = state.page >= state.totalPages;
}

function rowHtml(item) {
  const tier = riskTier(item.risk_score);
  const statusHtml = item.status === "pending"
    ? `<span class="status-chip">Pending</span>`
    : `<span class="status-chip ${item.status}">${item.status[0].toUpperCase()}${item.status.slice(1)}</span>`;
  return `
    <tr data-id="${item.document_id}">
      <td class="col-risk"><span class="risk-pill risk-${tier}">${fmtRisk(item.risk_score)}</span></td>
      <td>${item.vendor_name}</td>
      <td class="col-num">${money(item.amount, item.currency)}</td>
      <td>${item.merchant_category.replace("_", " ")}</td>
      <td class="detail-id">${item.document_id}</td>
      <td class="col-status">${statusHtml}</td>
    </tr>`;
}

function wireQueueControls() {
  $$(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      $$(".tab").forEach((t) => t.classList.remove("is-active"));
      tab.classList.add("is-active");
      state.status = tab.dataset.status;
      state.page = 1;
      loadQueue();
    });
  });

  const riskInput = $("#risk-filter");
  riskInput.addEventListener("input", () => {
    $("#risk-filter-value").textContent = parseFloat(riskInput.value).toFixed(2);
  });
  riskInput.addEventListener("change", () => {
    state.minRisk = parseFloat(riskInput.value);
    state.page = 1;
    loadQueue();
  });

  $("#sort-select").addEventListener("change", (e) => {
    state.sort = e.target.value;
    state.page = 1;
    loadQueue();
  });

  $("#pager-prev").addEventListener("click", () => {
    if (state.page > 1) { state.page -= 1; loadQueue(); }
  });
  $("#pager-next").addEventListener("click", () => {
    if (state.page < state.totalPages) { state.page += 1; loadQueue(); }
  });
}

// ---------------- Detail drawer ----------------

const FIELD_LABELS = {
  vendor_name: "Vendor", account_number: "Account", transaction_date: "Date",
  amount: "Amount", currency: "Currency", merchant_category: "Category",
  payment_method: "Payment method", billing_country: "Billing country",
  shipping_country: "Shipping country",
};

async function openDetail(documentId) {
  const res = await fetch(`${API}/transactions/${documentId}`);
  if (!res.ok) return;
  const doc = await res.json();
  renderDetail(doc);
  $("#drawer-backdrop").classList.add("is-open");
}

function fieldDisplayValue(name, value) {
  if (name === "amount") return typeof value === "number" ? value.toFixed(2) : value;
  return String(value).replace(/_/g, " ");
}

function renderDetail(doc) {
  const tier = riskTier(doc.risk_score);

  const boxesHtml = Object.entries(doc.bounding_boxes).map(([name, box]) => {
    const field = doc.fields[name];
    const ct = confTier(field.confidence);
    const style = `left:${box.x * 100}%; top:${box.y * 100}%; width:${box.w * 100}%; height:${box.h * 100}%;`;
    return `
      <div class="doc-field-box conf-${ct}" style="${style}" title="${FIELD_LABELS[name]}">
        <span class="fv">${fieldDisplayValue(name, field.value)}</span>
        <span class="fc">${(field.confidence * 100).toFixed(0)}%</span>
      </div>`;
  }).join("");

  const maxAbs = Math.max(...doc.top_contributions.map((c) => Math.abs(c.contribution)), 0.001);
  const contribHtml = doc.top_contributions.map((c) => {
    const pct = (Math.abs(c.contribution) / maxAbs) * 100;
    const sign = c.contribution >= 0 ? "pos" : "neg";
    return `
      <div class="contrib-row">
        <div class="contrib-name">${c.feature.replace(/_/g, " ")}</div>
        <div class="contrib-bar-track"><div class="contrib-bar ${sign}" style="width:${pct}%"></div></div>
      </div>`;
  }).join("");

  const statusBlock = doc.status === "pending"
    ? `
      <div class="review-actions">
        <button class="btn approve" id="btn-approve">Approve</button>
        <button class="btn reject" id="btn-reject">Reject</button>
      </div>
      <textarea class="review-note" id="review-note" placeholder="Analyst note (optional)"></textarea>`
    : `<div class="review-status ${doc.status}">${doc.status[0].toUpperCase()}${doc.status.slice(1)}${doc.analyst_note ? ` — "${doc.analyst_note}"` : ""}</div>`;

  $("#drawer-content").innerHTML = `
    <div class="detail-header">
      <div class="detail-id">${doc.document_id} · ${doc.template.replace("_", " ")}</div>
      <h2 class="detail-vendor">${doc.fields.vendor_name.value}</h2>
    </div>
    <div class="detail-grid">
      <div>
        <p class="section-label">Extracted document, fields grounded by confidence</p>
        <div class="doc-mock">${boxesHtml}</div>
        <div class="doc-legend">
          <span class="lg-high">High confidence</span>
          <span class="lg-mid">Medium</span>
          <span class="lg-low">Low confidence</span>
        </div>
      </div>
      <div class="side-panel">
        <div class="risk-block">
          <div class="rv">${fmtRisk(doc.risk_score)}</div>
          <div class="rl">model risk score · ${tier}</div>
        </div>
        <div>
          <p class="section-label">Top contributing signals</p>
          <div class="contrib-list">${contribHtml}</div>
        </div>
        <div>
          <p class="section-label">Analyst decision</p>
          ${statusBlock}
        </div>
      </div>
    </div>
  `;

  if (doc.status === "pending") {
    $("#btn-approve").addEventListener("click", () => submitReview(doc.document_id, "approved"));
    $("#btn-reject").addEventListener("click", () => submitReview(doc.document_id, "rejected"));
  }
}

async function submitReview(documentId, decision) {
  const note = $("#review-note") ? $("#review-note").value : "";
  await fetch(`${API}/transactions/${documentId}/review`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ decision, analyst_note: note || null }),
  });
  await openDetail(documentId);
  loadQueue();
  loadStats();
}

$("#drawer-close").addEventListener("click", () => $("#drawer-backdrop").classList.remove("is-open"));
$("#drawer-backdrop").addEventListener("click", (e) => {
  if (e.target.id === "drawer-backdrop") $("#drawer-backdrop").classList.remove("is-open");
});

// ---------------- Model view ----------------

async function loadModelView() {
  const res = await fetch(`${API}/metrics`);
  const m = await res.json();

  const cells = [
    ["Accuracy", (m.accuracy * 100).toFixed(1) + "%"],
    ["Precision", (m.precision * 100).toFixed(1) + "%"],
    ["Recall", (m.recall * 100).toFixed(1) + "%"],
    ["F1 (fraud class)", (m.f1 * 100).toFixed(1) + "%"],
    ["ROC-AUC", m.roc_auc.toFixed(4)],
  ];
  $("#metric-grid").innerHTML = cells.map(([k, v]) => `
    <div class="metric-cell"><span class="v">${v}</span><span class="k">${k}</span></div>
  `).join("");

  const [[tn, fp], [fn, tp]] = m.confusion_matrix;
  $("#confusion-wrap").innerHTML = `
    <h2>Confusion matrix — test split (n=${m.n_test.toLocaleString()})</h2>
    <table>
      <tr><th></th><th>Predicted legit</th><th>Predicted fraud</th></tr>
      <tr><th>Actually legit</th><td>${tn.toLocaleString()}</td><td>${fp.toLocaleString()}</td></tr>
      <tr><th>Actually fraud</th><td>${fn.toLocaleString()}</td><td>${tp.toLocaleString()}</td></tr>
    </table>
  `;
}

// ---------------- Nav ----------------

function wireNav() {
  $$(".rail-link").forEach((link) => {
    link.addEventListener("click", () => {
      $$(".rail-link").forEach((l) => l.classList.remove("is-active"));
      link.classList.add("is-active");
      const view = link.dataset.view;
      $$(".view").forEach((v) => v.hidden = v.id !== `view-${view}`);
      if (view === "model") loadModelView();
      if (view === "playground") initPlayground();
      if (view === "live") startLiveFeed();
      else stopLiveFeed();
    });
  });
}

// ---------------- Scoring playground ----------------

const PG_PRESETS = {
  benign: { amount: 60, merchant_category: "grocery", payment_method: "credit_card",
    currency: "USD", billing_country: "US", shipping_country: "US",
    account_age_days: 800, documents_last_24h: 1, hour_of_day: 14 },
  fraud: { amount: 4200, merchant_category: "crypto_exchange", payment_method: "wire",
    currency: "USD", billing_country: "US", shipping_country: "RU",
    account_age_days: 3, documents_last_24h: 9, hour_of_day: 3 },
};

let pgReady = false;
let pgDebounce = null;

async function initPlayground() {
  if (pgReady) return;
  const opts = await (await fetch(`${API}/score/options`)).json();
  fillSelect("pg-category", opts.categories);
  fillSelect("pg-payment", opts.payment_methods);
  fillSelect("pg-currency", opts.currencies);
  fillSelect("pg-billing", opts.countries);
  fillSelect("pg-shipping", opts.countries);

  [["pg-amount", "pg-amount-out"], ["pg-age", "pg-age-out"],
   ["pg-velocity", "pg-velocity-out"], ["pg-hour", "pg-hour-out"]].forEach(([inp, out]) => {
    $(`#${inp}`).addEventListener("input", () => {
      $(`#${out}`).textContent = $(`#${inp}`).value;
      scheduleScore();
    });
  });
  ["pg-category", "pg-payment", "pg-currency", "pg-billing", "pg-shipping"].forEach((id) => {
    $(`#${id}`).addEventListener("change", scheduleScore);
  });
  $("#pg-preset-benign").addEventListener("click", () => applyPreset("benign"));
  $("#pg-preset-fraud").addEventListener("click", () => applyPreset("fraud"));

  pgReady = true;
  applyPreset("benign");
}

function fillSelect(id, values) {
  $(`#${id}`).innerHTML = values
    .map((v) => `<option value="${v}">${v.replace(/_/g, " ")}</option>`).join("");
}

function applyPreset(which) {
  const p = PG_PRESETS[which];
  $("#pg-amount").value = p.amount; $("#pg-amount-out").textContent = p.amount;
  $("#pg-age").value = p.account_age_days; $("#pg-age-out").textContent = p.account_age_days;
  $("#pg-velocity").value = p.documents_last_24h; $("#pg-velocity-out").textContent = p.documents_last_24h;
  $("#pg-hour").value = p.hour_of_day; $("#pg-hour-out").textContent = p.hour_of_day;
  $("#pg-category").value = p.merchant_category;
  $("#pg-payment").value = p.payment_method;
  $("#pg-currency").value = p.currency;
  $("#pg-billing").value = p.billing_country;
  $("#pg-shipping").value = p.shipping_country;
  runScore();
}

function scheduleScore() {
  clearTimeout(pgDebounce);
  pgDebounce = setTimeout(runScore, 150);
}

async function runScore() {
  const payload = {
    amount: parseFloat($("#pg-amount").value),
    currency: $("#pg-currency").value,
    merchant_category: $("#pg-category").value,
    payment_method: $("#pg-payment").value,
    billing_country: $("#pg-billing").value,
    shipping_country: $("#pg-shipping").value,
    account_age_days: parseInt($("#pg-age").value),
    documents_last_24h: parseInt($("#pg-velocity").value),
    hour_of_day: parseInt($("#pg-hour").value),
  };
  const res = await fetch(`${API}/score`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) return;
  renderScore(await res.json());
}

function renderScore(r) {
  $("#pg-score").textContent = fmtRisk(r.risk_score);
  const isFraud = r.is_fraud_pred === 1;
  const verdict = $("#pg-verdict");
  verdict.textContent = isFraud ? "flagged as fraud" : "cleared as legit";
  verdict.className = "pg-verdict " + (isFraud ? "fraud" : "legit");

  const bar = $("#pg-bar");
  bar.style.width = `${Math.min(r.risk_score * 100, 100)}%`;
  bar.style.background = isFraud
    ? "var(--signal-critical)"
    : (r.risk_score >= 0.15 ? "var(--signal-caution)" : "var(--signal-safe)");
  $("#pg-threshold").textContent = `decision threshold: ${r.threshold.toFixed(2)}`;

  const maxAbs = Math.max(...r.top_contributions.map((c) => Math.abs(c.contribution)), 0.001);
  $("#pg-contribs").innerHTML = r.top_contributions.map((c) => {
    const pct = (Math.abs(c.contribution) / maxAbs) * 100;
    const sign = c.contribution >= 0 ? "pos" : "neg";
    return `
      <div class="contrib-row">
        <div class="contrib-name">${c.feature.replace(/_/g, " ")}</div>
        <div class="contrib-bar-track"><div class="contrib-bar ${sign}" style="width:${pct}%"></div></div>
      </div>`;
  }).join("");
}

// ---------------- Live feed ----------------

let liveTimer = null;

async function startLiveFeed() {
  const status = $("#live-status");
  const table = $("#live-table");
  const note = $("#live-note");
  async function poll() {
    try {
      const res = await fetch(`${API}/stream/recent`);
      if (!res.ok) throw new Error("no stream");
      const data = await res.json();
      if (data.demo) {
        note.hidden = false;
        note.textContent = "Preview — sample transactions scored by the model. "
          + "The live Redpanda/Faust feed runs with the full stack.";
      } else {
        note.hidden = true;
      }
      if (data.events.length === 0) {
        status.hidden = false; table.hidden = true;
        status.textContent = "Waiting for transactions…";
        return;
      }
      status.hidden = true; table.hidden = false;
      $("#live-body").innerHTML = data.events.map((e) => {
        const tier = riskTier(e.risk_score);
        const pred = e.is_fraud_pred
          ? `<span class="status-chip rejected">Fraud</span>`
          : `<span class="status-chip approved">Legit</span>`;
        return `<tr>
          <td class="col-risk"><span class="risk-pill risk-${tier}">${fmtRisk(e.risk_score)}</span></td>
          <td>${e.vendor_name || "—"}</td>
          <td class="col-num">${e.amount != null ? e.amount.toFixed(2) : "—"} ${e.currency || ""}</td>
          <td>${(e.merchant_category || "").replace(/_/g, " ")}</td>
          <td class="detail-id">${e.document_id}</td>
          <td class="col-status">${pred}</td>
        </tr>`;
      }).join("");
    } catch {
      note.hidden = true;
      status.hidden = false; table.hidden = true;
      status.textContent = "Feed unavailable.";
    }
  }
  await poll();
  liveTimer = setInterval(poll, 2000);
}

function stopLiveFeed() {
  if (liveTimer) { clearInterval(liveTimer); liveTimer = null; }
}

wireQueueControls();
wireNav();
loadStats();
loadQueue();
