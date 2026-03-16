// src/components/ModelSelector.tsx
import React, { useEffect, useState } from "react";

export default function ModelSelector({
  value,
  onChange,
  includeEmpty = false,
}: {
  value?: string;
  onChange?: (v: string) => void;
  includeEmpty?: boolean;
}) {
  const [opts, setOpts] = useState<{ id: string; label: string }[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    async function load() {
      setLoading(true);
      setErr(null);
      try {
        // try the more explicit list endpoint first
        const endpoints = ["/api/models/list", "/api/models", "/api/models?list=1"];
        let parsed: any = null;
        for (const ep of endpoints) {
          try {
            const res = await fetch(ep, { cache: "no-store" });
            if (!res.ok) {
              // keep trying other endpoints
              continue;
            }
            const j = await res.json();
            parsed = j;
            break;
          } catch (e) {
            // try next
            console.debug("[ModelSelector] fetch failed for", ep, e);
            continue;
          }
        }
        if (!parsed) {
          throw new Error("no response from /api/models endpoints");
        }

        // normalize possible response shapes:
        // - an array of {id,label}
        // - {"models":[{...}], "ok":true}
        // - an array of model objects (older APIs)
        // - {"models":[{"dir","name","path"...}]} (list of saved model files)
        let items: { id: string; label: string }[] = [];

        if (Array.isArray(parsed)) {
          // array could be [{id,label}, ...] or [{dir,name,...}, ...]
          items = parsed
            .map((it: any) => {
              if (!it) return null;
              if (typeof it === "string") return { id: it, label: it };
              if (it.id && it.label) return { id: String(it.id), label: String(it.label) };
              if (it.id && !it.label) return { id: String(it.id), label: String(it.id) };
              if (it.name) {
                // if it's a model file entry, use name without extension as label
                const name = String(it.name);
                const id = name.replace(/\.[^/.]+$/, "");
                return { id, label: id };
              }
              return null;
            })
            .filter(Boolean) as { id: string; label: string }[];
        } else if (parsed?.models && Array.isArray(parsed.models)) {
          items = parsed.models
            .map((m: any) => {
              if (!m) return null;
              if (m.id && m.label) return { id: String(m.id), label: String(m.label) };
              if (m.name) {
                const name = String(m.name);
                const id = name.replace(/\.[^/.]+$/, "");
                return { id, label: id };
              }
              // fallback to JSON string
              return { id: String(m.id ?? m.label ?? JSON.stringify(m)), label: String(m.label ?? m.id ?? JSON.stringify(m)) };
            })
            .filter(Boolean) as { id: string; label: string }[];
        } else if (parsed?.ok && parsed?.models && Array.isArray(parsed.models)) {
          // defensive
          items = parsed.models.map((m: any) => ({ id: String(m.id ?? m.name ?? m), label: String(m.label ?? m.id ?? m.name ?? m) }));
        } else {
          // try to infer keys
          const keys = Object.keys(parsed || {});
          if (keys.length && keys.every(k => typeof (parsed as any)[k] === "object")) {
            // convert object map -> list
            items = keys.map(k => ({ id: k, label: (parsed as any)[k]?.label ?? k }));
          }
        }

        // dedupe and normalize ids
        const seen = new Set<string>();
        items = items
          .map((it) => ({ id: String(it.id), label: String(it.label) }))
          .filter(it => {
            if (seen.has(it.id)) return false;
            seen.add(it.id);
            return true;
          });

        if (mounted) {
          setOpts(items);
        }
      } catch (e: any) {
        console.error("[ModelSelector] load error:", e);
        if (mounted) setErr(String(e?.message ?? e));
      } finally {
        if (mounted) setLoading(false);
      }
    }
    load();
    return () => {
      mounted = false;
    };
  }, []);

  const handleChange = (v: string) => {
    if (onChange) onChange(v);
  };

  // UI: show helpful status so user sees why nothing appeared
  if (loading) {
    return (
      <div className="text-xs text-muted-foreground">Loading models…</div>
    );
  }

  if (err) {
    return (
      <div className="text-xs text-red-500">
        Failed to load models — check backend. <button className="ml-2 underline" onClick={() => window.location.reload()}>Reload</button>
      </div>
    );
  }

  if (!opts.length) {
    return (
      <div className="text-xs text-muted-foreground">
        No models found. Ensure backend `/api/models/list` or `/api/models` is available.
      </div>
    );
  }

  return (
    <select
      className="w-full p-2 rounded border bg-transparent text-sm"
      value={value ?? (includeEmpty ? "" : (opts[0]?.id ?? ""))}
      onChange={(e) => handleChange(e.target.value)}
    >
      {includeEmpty && <option value="">— select model —</option>}
      {opts.map((o) => (
        <option key={o.id} value={o.id}>
          {o.label}
        </option>
      ))}
    </select>
  );
}
