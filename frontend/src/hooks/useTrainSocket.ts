// src/hooks/useTrainSocket.ts
import { useEffect, useRef, useState, useCallback } from "react";

/**
 * useTrainSocket
 * - url: full ws URL (ws:// or wss://) or relative path ("/ws/train")
 * - enabled: boolean (default true) — control whether socket should be kept open
 *
 * Returns:
 *  { connected, lastMessage, send, readyState }
 *
 * Notes:
 * - reconnects with exponential backoff + jitter
 * - ensures only one WS instance per hook (per-mount)
 */
export function useTrainSocket(url: string, enabled = true) {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const attemptsRef = useRef(0);
  const forcedCloseRef = useRef(false);

  const [connected, setConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<any>(null);
  const [readyState, setReadyState] = useState<number>(WebSocket.CLOSED);

  const clearReconnectTimer = () => {
    if (reconnectTimerRef.current) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  };

  const cleanupSocket = () => {
    try {
      if (wsRef.current) {
        forcedCloseRef.current = true;
        wsRef.current.close();
      }
    } catch (e) {
      // ignore
    }
    wsRef.current = null;
    setConnected(false);
    setReadyState(WebSocket.CLOSED);
    clearReconnectTimer();
  };

  // send helper (safe)
  const send = useCallback((data: any) => {
    try {
      const s = wsRef.current;
      if (!s || s.readyState !== WebSocket.OPEN) {
        console.warn("[useTrainSocket] send: socket not open");
        return false;
      }
      if (typeof data === "string") s.send(data);
      else s.send(JSON.stringify(data));
      return true;
    } catch (e) {
      console.warn("[useTrainSocket] send failed", e);
      return false;
    }
  }, []);

  // create or reconnect
  const connect = useCallback(
    (connectUrl: string) => {
      // Prevent double-connect if already open
      if (!enabled) return;
      if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
        // already connecting/open
        return;
      }

      // Reset forced-close marker when trying to reconnect by logic
      forcedCloseRef.current = false;

      // build resolvedUrl: prefer dev backend ws when in DEV environment
      let resolvedUrl = connectUrl;

      // If a relative path was given, build absolute URL according to environment
      if (!/^wss?:\/\//i.test(connectUrl)) {
        if (import.meta.env.DEV) {
          // In dev, prefer the backend WS directly (bypass relying on window.location)
          // This matches the Vite proxy behavior but makes the client robust if proxy not used.
          const DEV_BACKEND_WS = "ws://127.0.0.1:5000";
          // ensure connectUrl begins with a slash
          const path = connectUrl.startsWith("/") ? connectUrl : `/${connectUrl}`;
          resolvedUrl = `${DEV_BACKEND_WS}${path}`;
        } else {
          // production: use same-origin behaviour
          const scheme = window.location.protocol === "https:" ? "wss" : "ws";
          resolvedUrl = `${scheme}://${window.location.host}${connectUrl}`;
        }
      }

      try {
        const socket = new WebSocket(resolvedUrl);
        wsRef.current = socket;
        setReadyState(socket.readyState);
        setLastMessage(null);

        socket.onopen = () => {
          attemptsRef.current = 0;
          clearReconnectTimer();
          setConnected(true);
          setReadyState(socket.readyState);
          console.debug("[useTrainSocket] ws open", resolvedUrl);
        };

        socket.onmessage = (ev) => {
          try {
            const raw = ev.data;
            let parsed: any;
            if (typeof raw === "string") {
              try {
                parsed = JSON.parse(raw);
              } catch {
                parsed = raw;
              }
            } else {
              parsed = raw;
            }
            setLastMessage(parsed);
          } catch (e) {
            console.warn("[useTrainSocket] onmessage parse failed", e);
          }
        };

        socket.onerror = (ev) => {
          console.debug("[useTrainSocket] websocket error", ev);
        };

        socket.onclose = (ev) => {
          setConnected(false);
          setReadyState(socket.readyState);
          wsRef.current = null;
          console.debug("[useTrainSocket] ws closed", ev.code, ev.reason);

          // If we were forced closed via cleanupSocket(), do not attempt reconnect
          if (forcedCloseRef.current || !enabled) {
            return;
          }

          // Reconnect with exponential backoff + jitter
          attemptsRef.current = Math.min(12, attemptsRef.current + 1);
          const base = Math.min(30_000, 1000 * Math.pow(1.8, attemptsRef.current)); // cap ~30s
          const jitter = Math.floor(Math.random() * 1000);
          const delay = Math.max(500, Math.min(base + jitter, 30_000));

          console.debug(`[useTrainSocket] scheduling reconnect in ${delay}ms (attempt ${attemptsRef.current})`);
          clearReconnectTimer();
          reconnectTimerRef.current = window.setTimeout(() => {
            reconnectTimerRef.current = null;
            connect(connectUrl);
          }, delay);
        };
      } catch (e) {
        console.warn("[useTrainSocket] connect failed", e);
        // schedule reconnect
        attemptsRef.current = Math.min(12, attemptsRef.current + 1);
        const delay = Math.min(30000, 1000 * Math.pow(1.8, attemptsRef.current) + Math.floor(Math.random() * 500));
        clearReconnectTimer();
        reconnectTimerRef.current = window.setTimeout(() => {
          reconnectTimerRef.current = null;
          connect(connectUrl);
        }, delay);
      }
    },
    [enabled]
  );

  // main effect: create socket when enabled/url changes
  useEffect(() => {
    if (!url || !enabled) {
      cleanupSocket();
      return;
    }

    connect(url);

    return () => {
      // on unmount, prevent reconnect and close
      forcedCloseRef.current = true;
      clearReconnectTimer();
      try {
        if (wsRef.current) wsRef.current.close();
      } catch {}
      wsRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url, enabled]); // connect is stable via useCallback; ok to include url,enabled

  return {
    connected,
    lastMessage,
    send,
    readyState,
  };
}

export default useTrainSocket;
