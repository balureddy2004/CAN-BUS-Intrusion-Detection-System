// src/api.js — REST API helpers for frontend

// ✅ Use same-origin so Vite's proxy (5173 → 8000) handles it in dev.
//    You can still override with VITE_API_URL if you deploy the backend elsewhere.
const API_URL = (import.meta.env.VITE_API_URL ?? "").trim();

// join helper: ensures leading slash, no double slashes
const api = (p) => {
  const path = p.startsWith("/") ? p : `/${p}`;
  // If API_URL is empty (same-origin), just return the path
  if (!API_URL) return path;
  // Otherwise prefix with explicit base (useful in prod behind a different domain)
  return `${API_URL.replace(/\/+$/, "")}${path}`;
};

// Small fetch wrapper
async function safeFetch(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status}${text ? `: ${text}` : ""}`);
  }
  // try json, fallback to text
  const ct = res.headers.get("content-type") || "";
  return ct.includes("application/json") ? res.json() : res.text();
}

/* -------------------------
 * Metrics (optional / legacy)
 * ------------------------- */
// NOTE: Your current backend doesn't expose /api/metrics.
// Keep this here only if you add that route later.
export async function getMetrics() {
  return safeFetch(api("/api/metrics"));
}

/* -------------------------
 * Training
 * ------------------------- */
export async function startTraining(dataset = "ocs-lab", extra = {}) {
  // Backend expects POST /train/start with JSON body
  // { model, epochs, batchSize, learningRate, optimizer, trainSplit, seed, augment, dataset }
  const body = JSON.stringify({ dataset, ...extra });
  return safeFetch(api("/train/start"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
  });
}

export async function getTrainingStatus() {
  return safeFetch(api("/train/status"));
}

/* -------------------------
 * Model management (legacy)
 * ------------------------- */
// NOTE: These endpoints are not present in your current server.py.
// Keep them only if you implement matching routes.
export async function getModels() {
  return safeFetch(api("/api/models"));
}

export async function loadModel(file) {
  return safeFetch(api("/api/models/load"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file }),
  });
}

/* -------------------------
 * Settings (legacy)
 * ------------------------- */
// NOTE: Not in current server.py; retained for compatibility.
export async function updateSettings({ emailOn, thresh }) {
  return safeFetch(api("/api/settings"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ emailOn, thresh }),
  });
}

/* -------------------------
 * IDS Analytics (Time-Series)
 * ------------------------- */
export async function getAlertSeries(since = "5m", window = "1s", types) {
  const q = new URLSearchParams({ since, window });
  if (types?.length) q.set("types", types.join(","));
  return safeFetch(api(`/api/alerts/timeseries?${q.toString()}`));
}

export async function getPacketSeries(since = "5m", window = "1s", canId) {
  const q = new URLSearchParams({ since, window });
  if (canId) q.set("can_id", canId);
  return safeFetch(api(`/api/packets/timeseries?${q.toString()}`));
}
