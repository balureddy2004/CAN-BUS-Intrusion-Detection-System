// src/hooks/useAvailableModels.ts
import { useEffect, useState, useCallback } from "react";

export type ModelInfo = {
  id: string;
  display: string;
  available?: boolean; // maps to enabled
  requires_gpu?: boolean;
  notes?: string;
  weights_path?: string;
  size_bytes?: number;
  last_scanned_at?: string;
};

export function useAvailableModels(pollIntervalMs?: number) {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchModels = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/models");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const list = Array.isArray(data) ? data : data.models ?? [];
      setModels(list);
    } catch (err: any) {
      setError(err?.message ?? String(err));
      setModels([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchModels();
    if (pollIntervalMs && pollIntervalMs > 0) {
      const t = setInterval(fetchModels, pollIntervalMs);
      return () => clearInterval(t);
    }
  }, [fetchModels, pollIntervalMs]);

  return { models, loading, error, refresh: fetchModels };
}
