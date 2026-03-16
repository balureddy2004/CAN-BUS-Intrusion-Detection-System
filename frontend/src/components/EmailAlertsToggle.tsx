// src/components/EmailAlertsToggle.tsx
import React, { useEffect, useMemo, useState } from "react";

type StatusResp = {
  enabled?: boolean;
  snoozed_for_s?: number;
  snoozed_until?: string | null;
  ok?: boolean;
  error?: string;
};

type ToggleResp = {
  ok?: boolean;
  enabled?: boolean;
  error?: string;
};

type SnoozeResp = {
  ok?: boolean;
  minutes?: number;
  error?: string;
};

type TestResp = {
  ok?: boolean;
  error?: string;
};

export default function EmailAlertsToggle({ apiBase = "" }: { apiBase?: string }) {
  const [enabled, setEnabled] = useState<boolean>(true);
  const [loading, setLoading] = useState<boolean>(false);
  const [snoozeMins, setSnoozeMins] = useState<string>("15");
  const [snoozedFor, setSnoozedFor] = useState<number>(0);
  const [error, setError] = useState<string>("");
  const [okMsg, setOkMsg] = useState<string>("");

  const api = useMemo(() => (path: string) => `${apiBase}${path}`, [apiBase]);

  async function fetchStatus(signal?: AbortSignal) {
    setError("");
    try {
      const res = await fetch(api("/api/alerts/email/status"), { signal });
      const j = (await res.json()) as StatusResp;
      if (!res.ok || j.ok === false) throw new Error(j.error || `status ${res.status}`);
      setEnabled(Boolean(j.enabled));
      setSnoozedFor(j.snoozed_for_s || 0);
    } catch (e: any) {
      if (e?.name === "AbortError") return; // unmount
      setError(`Failed to load status: ${e?.message || e}`);
    }
  }

  useEffect(() => {
    const ac = new AbortController();
    fetchStatus(ac.signal);
    const id = setInterval(() => fetchStatus(ac.signal), 5000);
    return () => {
      ac.abort();
      clearInterval(id);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function setEnabledRemote(next: boolean) {
    setLoading(true);
    setError("");
    setOkMsg("");
    try {
      const res = await fetch(api("/api/alerts/email/enabled"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: next }),
      });
      const j = (await res.json()) as ToggleResp;
      if (!res.ok || j.ok === false) throw new Error(j.error || `status ${res.status}`);
      setEnabled(Boolean(j.enabled ?? next));
      setOkMsg((j.enabled ?? next) ? "Email alerts enabled" : "Email alerts disabled");
    } catch (e: any) {
      setError(`Failed to update: ${e?.message || e}`);
    } finally {
      setLoading(false);
    }
  }

  async function snooze() {
    const minutes = Math.max(1, Math.floor(Number(snoozeMins) || 0));
    setLoading(true);
    setError("");
    setOkMsg("");
    try {
      const res = await fetch(api("/api/alerts/email/snooze"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ minutes }),
      });
      const j = (await res.json()) as SnoozeResp;
      if (!res.ok || j.ok === false) throw new Error(j.error || `status ${res.status}`);
      setOkMsg(`Snoozed for ${j.minutes ?? minutes} minute(s)`);
      await fetchStatus(); // refresh snoozed_for_s
    } catch (e: any) {
      setError(`Failed to snooze: ${e?.message || e}`);
    } finally {
      setLoading(false);
    }
  }

  async function sendTest(force = false) {
    setLoading(true);
    setError("");
    setOkMsg("");
    try {
      const res = await fetch(api(`/api/alerts/test_email${force ? "?force=1" : ""}`), { method: "POST" });
      const j = (await res.json()) as TestResp;
      if (!res.ok || j.ok === false) throw new Error(j.error || `status ${res.status}`);
      setOkMsg(force ? "Test sent (forced)" : "Test sent");
    } catch (e: any) {
      setError(`Failed to send test: ${e?.message || e}`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-xl w-full rounded-2xl shadow p-4 bg-white border border-slate-200">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-lg font-semibold">Email Alerts</div>
          <div className="text-slate-500 text-sm">Control backend email notifications</div>
        </div>
        <label className="inline-flex items-center cursor-pointer" aria-label="Toggle email alerts">
          <input
            type="checkbox"
            className="sr-only peer"
            checked={enabled}
            onChange={(e) => setEnabledRemote(e.target.checked)}
            disabled={loading}
          />
          <div className="w-12 h-7 bg-slate-200 peer-focus:outline-none rounded-full peer transition-all peer-checked:bg-emerald-500 relative">
            <div className="absolute top-1 left-1 w-5 h-5 bg-white rounded-full transition-all peer-checked:translate-x-5" />
          </div>
        </label>
      </div>

      <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-3">
        <div className="col-span-2">
          <label htmlFor="snooze-mins" className="text-sm text-slate-600">
            Snooze (minutes)
          </label>
          <div className="flex items-center gap-2 mt-1">
            <input
              id="snooze-mins"
              type="number"
              min={1}
              value={snoozeMins}
              onChange={(e) => setSnoozeMins(e.target.value)}
              className="w-24 border rounded-lg px-3 py-2 text-sm"
              inputMode="numeric"
            />
            <button
              onClick={snooze}
              disabled={loading}
              className="px-3 py-2 rounded-lg border bg-slate-50 hover:bg-slate-100 text-sm disabled:opacity-60"
            >
              Snooze
            </button>
          </div>
          {snoozedFor > 0 && (
            <div className="text-xs text-amber-600 mt-1">Snoozed for another {snoozedFor}s</div>
          )}
        </div>
        <div className="flex flex-col gap-2 mt-5 md:mt-0">
          <button
            onClick={() => sendTest(false)}
            disabled={loading}
            className="px-3 py-2 rounded-lg border text-sm hover:bg-slate-50 disabled:opacity-60"
          >
            Send Test
          </button>
          <button
            onClick={() => sendTest(true)}
            disabled={loading}
            className="px-3 py-2 rounded-lg border text-sm hover:bg-slate-50 disabled:opacity-60"
          >
            Force Test
          </button>
        </div>
      </div>

      {(error || okMsg) && (
        <div className="mt-3 text-sm">
          {error && <div className="text-red-600">{error}</div>}
          {okMsg && <div className="text-emerald-600">{okMsg}</div>}
        </div>
      )}
    </div>
  );
}
