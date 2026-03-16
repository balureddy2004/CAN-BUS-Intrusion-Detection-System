// src/hooks/useAlerts.ts
import { useEffect, useRef, useState, useCallback } from "react";

export type AlertItem = {
  ts?: number; // seconds or ms (we normalize to ms)
  ts_ms?: number;
  timestamp?: number;
  type?: string;
  attack_type?: string;
  severity?: string;
  score?: number;
  details?: Record<string, any>;
};

export default function useAlerts(opts?: {
  initialWindow?: string; // like "30s" or "1m"
  maxItems?: number;
}) {
  const initialWindow = opts?.initialWindow ?? "60s";
  const MAX = opts?.maxItems ?? 500;

  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [connected, setConnected] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectRef = useRef<number | null>(null);

  const _normalize = (a: AlertItem) => {
    const out = { ...a } as any;
    if (!out.ts_ms) {
      if (out.ts) {
        out.ts_ms = Number(out.ts) >= 1e12 ? Number(out.ts) : Math.round(Number(out.ts) * 1000);
      } else if (out.timestamp) {
        out.ts_ms = Number(out.timestamp) >= 1e12 ? Number(out.timestamp) : Math.round(Number(out.timestamp) * 1000);
      } else {
        out.ts_ms = Math.round(Date.now());
      }
    }
    out.attack_type = out.attack_type || out.type || "unknown";
    out.severity = (out.severity || "medium").toLowerCase();
    return out as AlertItem;
  };

  // fetch initial list from REST
  const fetchAlerts = useCallback(async (query?: { since?: string; type?: string; severity?: string; limit?: number }) => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams();
      if (query?.since) params.set("since", query.since);
      if (query?.type) params.set("type", query.type);
      if (query?.severity) params.set("severity", query.severity);
      if (query?.limit) params.set("limit", String(query.limit));
      const r = await fetch(`/api/alerts?${params.toString()}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      const normalized = (Array.isArray(data) ? data : []).map(_normalize).sort((a,b) => (b.ts_ms||0)-(a.ts_ms||0));
      setAlerts((prev) => {
        const merged = [...normalized];
        // keep up to MAX (initial load should replace)
        return merged.slice(0, MAX);
      });
    } catch (e: any) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [MAX]);

  // WebSocket + reconnect logic
  useEffect(() => {
    let mounted = true;
    let attempts = 0;
    const connect = () => {
      if (!mounted) return;
      try {
        // use relative path so Vite proxy handles ws -> backend
        const url = (location.protocol === "https:" ? "wss" : "ws") + "://" + location.host + "/ws/alerts";
        const ws = new WebSocket(url);
        wsRef.current = ws;

        ws.onopen = () => {
          if (!mounted) return;
          attempts = 0;
          setConnected(true);
          console.debug("[useAlerts] ws open", url);
        };

        ws.onmessage = (ev) => {
          if (!mounted) return;
          try {
            const raw = typeof ev.data === "string" ? JSON.parse(ev.data) : ev.data;
            const typ = raw?.type || raw?.t || null;
            // handle single alert or replay_alerts
            if (Array.isArray(raw.items) && raw.type?.startsWith("replay")) {
              const items = raw.items.map(_normalize);
              setAlerts((prev) => {
                const merged = [...items, ...prev].sort((a,b) => (b.ts_ms||0)-(a.ts_ms||0));
                return merged.slice(0, MAX);
              });
            } else if (raw.type === "replay_alerts" && Array.isArray(raw.items)) {
              const items = raw.items.map(_normalize);
              setAlerts((prev) => {
                const merged = [...items, ...prev].sort((a,b) => (b.ts_ms||0)-(a.ts_ms||0));
                return merged.slice(0, MAX);
              });
            } else if (raw.attack_type || raw.type) {
              const item = _normalize(raw);
              setAlerts((prev) => {
                const merged = [item, ...prev];
                return merged.slice(0, MAX);
              });
            } else {
              // ignore other messages
            }
          } catch (e) {
            console.warn("[useAlerts] ws parse fail", e);
          }
        };

        ws.onclose = () => {
          if (!mounted) return;
          setConnected(false);
          wsRef.current = null;
          // reconnect with backoff
          attempts = Math.min(12, attempts + 1);
          const delay = Math.min(30000, Math.pow(1.8, attempts) * 1000 + Math.floor(Math.random() * 500));
          reconnectRef.current = window.setTimeout(connect, delay);
        };

        ws.onerror = (ev) => {
          console.debug("[useAlerts] ws error", ev);
        };
      } catch (e) {
        attempts = Math.min(12, attempts + 1);
        const delay = Math.min(30000, Math.pow(1.8, attempts) * 1000 + Math.floor(Math.random() * 500));
        reconnectRef.current = window.setTimeout(connect, delay);
      }
    };

    // startup: load recent alerts (window) + connect ws
    fetchAlerts({ since: initialWindow, limit: MAX });
    connect();

    return () => {
      mounted = false;
      if (reconnectRef.current) {
        window.clearTimeout(reconnectRef.current);
        reconnectRef.current = null;
      }
      try { wsRef.current?.close(); } catch {}
      wsRef.current = null;
    };
  }, [fetchAlerts, initialWindow, MAX]);

  const clearAlerts = useCallback(async () => {
    try {
      const r = await fetch("/api/alerts/clear", { method: "POST" });
      if (r.ok) setAlerts([]);
    } catch (e) {
      console.warn("clearAlerts failed", e);
    }
  }, []);

  const refresh = useCallback((params?: { since?: string; type?: string; severity?: string }) => {
    fetchAlerts({ since: params?.since, type: params?.type, severity: params?.severity, limit: MAX });
  }, [fetchAlerts, MAX]);

  return {
    alerts,
    connected,
    loading,
    error,
    clearAlerts,
    refresh,
  };
}
