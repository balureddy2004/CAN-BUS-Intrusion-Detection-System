// src/components/AlertModal.tsx
import React, { useEffect, useState } from "react";

type Props = {
  open: boolean;
  onClose: () => void;
  alert: any | null;
};

export default function AlertModal({ open, onClose, alert }: Props) {
  const [packet, setPacket] = useState<any | null>(null);
  const [loading, setLoading] = useState(false);
  useEffect(() => {
    if (!open) return;
    setPacket(null);
    setLoading(false);
    if (!alert) return;
    // try fetching packet details if details.src or id exists
    const src = (alert?.details?.src) || alert?.id || alert?.can_id || null;
    if (!src) return;
    let mounted = true;
    (async () => {
      setLoading(true);
      try {
        // try /api/packets with since a bit before alert time to locate it
        const since = alert.ts_ms ? (alert.ts_ms - 30000) : undefined;
        const qs = new URLSearchParams();
        if (since) qs.set("since", String(Math.floor(since)));
        qs.set("limit", "200");
        const r = await fetch(`/api/packets?${qs.toString()}`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const arr = await r.json();
        // try to find packet by can_id or matching src
        const found = (arr || []).find((p: any) => {
          if (!p) return false;
          if (p.can_id && (String(p.can_id) === String(src) || ("0x"+p.can_id.toString(16)) === String(src))) return true;
          if (p.meta && p.meta.id_hex && p.meta.id_hex === String(alert.details?.src)) return true;
          return false;
        });
        if (mounted) setPacket(found || null);
      } catch (e) {
        // ignore
        if ((window as any).__DEV__) console.debug("alert modal fetch fail", e);
      } finally {
        if (mounted) setLoading(false);
      }
    })();
    return () => { mounted = false; };
  }, [open, alert]);

  if (!open || !alert) return null;

  const ts = alert.ts_ms ? new Date(alert.ts_ms) : new Date();
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/50" onClick={onClose} />
      <div className="relative max-w-3xl w-full bg-white dark:bg-slate-900 rounded-xl shadow-2xl overflow-auto">
        <div className="p-4 border-b flex items-start justify-between">
          <div>
            <h3 className="text-lg font-semibold">{(alert.attack_type || alert.type || "Alert").toString()}</h3>
            <div className="text-xs text-muted-foreground">{ts.toLocaleString()}</div>
          </div>
          <div className="flex items-center gap-2">
            <div className="text-sm px-3 py-1 rounded-full bg-yellow-500 text-white font-semibold">
              {alert.severity ?? "medium"}
            </div>
            <button onClick={onClose} aria-label="Close" className="text-sm px-3 py-1">Close</button>
          </div>
        </div>

        <div className="p-4 grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h4 className="text-sm font-medium mb-2">Summary</h4>
            <div className="text-sm whitespace-pre-wrap font-mono bg-slate-50 p-3 rounded">
              {JSON.stringify(alert, null, 2)}
            </div>
          </div>

          <div>
            <h4 className="text-sm font-medium mb-2">Packet (if available)</h4>
            {loading ? (
              <div className="text-sm">Loading packet...</div>
            ) : packet ? (
              <div className="text-sm bg-slate-50 p-3 rounded font-mono">
                <div>CAN ID: {packet.can_id} ({packet.id ?? "—"})</div>
                <div>DLC: {packet.dlc}</div>
                <div>Bytes: {packet.data ? JSON.stringify(packet.data) : packet.data_hex}</div>
                <div>Signal: {packet.signal}</div>
              </div>
            ) : (
              <div className="text-sm text-muted-foreground">No matching packet found nearby.</div>
            )}
          </div>
        </div>

        <div className="p-3 border-t flex justify-end gap-2">
          <button onClick={onClose} className="px-4 py-2 rounded-md bg-slate-200">Close</button>
        </div>
      </div>
    </div>
  );
}
