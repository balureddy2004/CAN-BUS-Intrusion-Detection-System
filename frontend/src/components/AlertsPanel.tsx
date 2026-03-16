// frontend/src/components/AlertsPanel.tsx
import React, { useMemo } from "react";
import { format } from "date-fns/format";
import { formatDistanceToNow } from "date-fns/formatDistanceToNow";
import useAlerts from "@/hooks/useAlerts";

/**
 * Local AlertItem type — matches the backend minimal shape we expect.
 * Adjust if your backend sends different property names.
 */
export interface AlertItem {
  id: number;
  /** epoch ms */
  timestamp: number;
  attack_type?: string;
  severity?: string;
  message?: string;
  meta?: any;
}

/**
 * Small helper: normalize a timestamp which may be:
 * - epoch milliseconds (preferred)
 * - epoch seconds (convert to ms)
 * - undefined -> Date.now()
 */
function tsToMs(v?: number): number {
  if (!v && v !== 0) return Date.now();
  // if it's obviously seconds (less than year 3000 in ms), convert
  if (v < 1_000_000_000_000) {
    return Math.floor(v * 1000);
  }
  return Math.floor(v);
}

const badgeStyle: React.CSSProperties = {
  display: "inline-block",
  padding: "4px 8px",
  borderRadius: 999,
  fontSize: 12,
  fontWeight: 700,
  color: "#fff",
};

function sevColor(s?: string) {
  if (!s) return "#6b7280";
  const k = s.toLowerCase();
  if (k === "low") return "#16a34a";
  if (k === "medium") return "#f59e0b";
  if (k === "high") return "#dc2626";
  return "#6b7280";
}

/**
 * AlertsPanel component
 *
 * This is intentionally defensive:
 * - uses default import `useAlerts` (matches your repo where only useAlerts.ts exists)
 * - tolerates missing fields
 * - provides controls (refresh/clear/start/stop) if the hook exposes them
 */
export default function AlertsPanel(): React.ReactElement {
  // your hook may expose different shape; we try to use common names
  // If types don't match exactly, the code will still run because we check existence
  const alertsHook = useAlerts?.() as
    | {
        alerts?: AlertItem[];
        connected?: boolean;
        loading?: boolean;
        error?: string | null;
        // optional control functions
        refresh?: (params?: any) => Promise<void> | void;
        clearAlerts?: () => Promise<void> | void;
        start?: () => Promise<void> | void;
        stop?: () => Promise<void> | void;
      }
    | undefined;

  const alerts = alertsHook?.alerts ?? [];
  const connected = !!alertsHook?.connected;
  const loading = !!alertsHook?.loading;
  const error = alertsHook?.error ?? null;

  const onRefresh = async () => {
    try {
      if (alertsHook?.refresh) await alertsHook.refresh();
    } catch (e) {
      // swallow — hook likely logs
      // optionally show a notification here
    }
  };

  const onClear = async () => {
    try {
      if (alertsHook?.clearAlerts) await alertsHook.clearAlerts();
    } catch (e) {}
  };

  const onStart = async () => {
    try {
      if (alertsHook?.start) await alertsHook.start();
    } catch (e) {}
  };

  const onStop = async () => {
    try {
      if (alertsHook?.stop) await alertsHook.stop();
    } catch (e) {}
  };

  const sorted = useMemo(() => {
    // defensive: copy and sort descending by timestamp
    return (Array.isArray(alerts) ? alerts.slice() : [])
      .map((a) => ({ ...a }))
      .sort((a, b) => tsToMs(b.timestamp) - tsToMs(a.timestamp));
  }, [alerts]);

  return (
    <div style={{ padding: 12, maxWidth: 860 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
        <h3 style={{ margin: 0 }}>Alerts</h3>
        <div style={{ marginLeft: "auto", display: "flex", gap: 8, alignItems: "center" }}>
          <div style={{ fontSize: 13, color: connected ? "#10b981" : "#ef4444" }}>
            {connected ? "WS: connected" : "WS: disconnected"}
          </div>
          <button onClick={onRefresh} style={{ padding: "6px 10px" }}>
            Refresh
          </button>
          {alertsHook?.start && (
            <button onClick={onStart} style={{ padding: "6px 10px" }}>
              Start
            </button>
          )}
          {alertsHook?.stop && (
            <button onClick={onStop} style={{ padding: "6px 10px" }}>
              Stop
            </button>
          )}
          {alertsHook?.clearAlerts && (
            <button onClick={onClear} style={{ padding: "6px 10px" }}>
              Clear
            </button>
          )}
        </div>
      </div>

      <div style={{ color: "#6b7280", marginBottom: 10 }}>
        {loading ? "Loading…" : error ? `Error: ${error}` : `${sorted.length} alert(s)`}
      </div>

      <div style={{ display: "grid", gap: 8 }}>
        {sorted.length === 0 && (
          <div style={{ padding: 20, border: "1px dashed #e5e7eb", borderRadius: 8, color: "#6b7280" }}>
            No alerts yet.
          </div>
        )}

        {sorted.map((a) => {
          // map backend props to friendly values (defensive)
          const id = a.id ?? (a.meta && a.meta.id) ?? Math.floor(Math.random() * 1_000_000);
          const ts = tsToMs(a.timestamp ?? (a as any).ts);
          let timeLabel = "";
          try {
            timeLabel = format(new Date(ts), "yyyy-MM-dd HH:mm:ss");
          } catch {
            timeLabel = String(ts);
          }
          let rel = "";
          try {
            rel = formatDistanceToNow(new Date(ts), { addSuffix: true });
          } catch {
            rel = "";
          }
          const sev = a.severity ?? (a.meta && a.meta.severity) ?? "unknown";
          const attackType = (a as any).attack_type ?? (a as any).type ?? "-";
          const message = a.message ?? a.meta?.message ?? a.meta?.details ?? attackType;

          return (
            <div
              key={id}
              style={{
                display: "flex",
                gap: 12,
                padding: 12,
                borderRadius: 10,
                background: "#0f172a",
                color: "#fff",
                alignItems: "center",
              }}
            >
              <div style={{ width: 12, height: 12, borderRadius: 999, background: sevColor(sev) }} />
              <div style={{ flex: 1 }}>
                <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                  <div style={{ fontWeight: 700, fontSize: 14 }}>{attackType ?? "alert"}</div>
                  <div style={{ opacity: 0.8, fontSize: 12 }}>{message}</div>
                  <div style={{ marginLeft: "auto", fontSize: 12, color: "#9CA3AF" }}>
                    {timeLabel} {rel ? `· ${rel}` : ""}
                  </div>
                </div>
                <div style={{ marginTop: 6, display: "flex", gap: 8, alignItems: "center" }}>
                  <div style={{ ...badgeStyle, background: sevColor(sev) }}>{(sev || "-").toUpperCase()}</div>
                  {a.meta && (
                    <div style={{ fontSize: 12, color: "#9CA3AF" }}>
                      {typeof a.meta === "object" ? JSON.stringify(a.meta) : String(a.meta)}
                    </div>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
