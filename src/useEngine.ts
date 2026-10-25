import { useCallback, useEffect, useRef, useState } from "react";
import { cleanGraph, type Analysis, type ModelGraph } from "./model";

export function useEngine(
  graph: ModelGraph,
  onNotice: (message: string, error?: boolean) => void,
) {
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [status, setStatus] = useState<"connecting" | "connected" | "offline">(
    "connecting",
  );
  const [busy, setBusy] = useState(false);
  const [checking, setChecking] = useState(true);
  const [compilation, setCompilation] = useState<{
    parameters: number;
    outputs: Record<string, number[]>;
  } | null>(null);
  const socket = useRef<WebSocket | null>(null);
  const currentGraph = useRef(graph);
  const notice = useRef(onNotice);
  const revision = useRef(0);
  const pending = useRef(
    new Map<
      string,
      { action: "export" | "compile"; fingerprint: string; name: string }
    >(),
  );
  const fingerprint = (value: ModelGraph) =>
    JSON.stringify({
      mode: value.mode,
      declaredMetadata: value.mode === "schematic" ? { validation: value.metadata?.validation, parameter_counts: value.metadata?.parameter_counts } : undefined,
      nodes: value.nodes.map((n) => ({ id: n.id, data: n.data })),
      edges: value.edges.map((e) => ({ source: e.source, target: e.target })),
    });
  const fingerprintRef = useRef(fingerprint(graph));
  fingerprintRef.current = fingerprint(graph);
  currentGraph.current = graph;
  notice.current = onNotice;

  const analyze = useCallback(() => {
    if (socket.current?.readyState === WebSocket.OPEN)
      socket.current.send(
        JSON.stringify({
          type: "graph.changed",
          graph: cleanGraph(currentGraph.current),
          request_id: `analyze-${++revision.current}`,
        }),
      );
  }, []);

  useEffect(() => {
    let disposed = false;
    let reconnect: ReturnType<typeof setTimeout>;
    let attempts = 0;
    const connect = () => {
      const ws = new WebSocket(
        `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`,
      );
      socket.current = ws;
      ws.onopen = () => {
        if (disposed) return;
        attempts = 0;
        setStatus("connected");
        analyze();
      };
      ws.onmessage = (message) => {
        if (disposed) return;
        let event;
        try {
          event = JSON.parse(message.data);
        } catch {
          return;
        }
        if (
          event.type === "graph.analyzed" &&
          event.request_id === `analyze-${revision.current}`
        ) {
          setAnalysis(event.analysis);
          setChecking(false);
        }
        const operation = pending.current.get(event.request_id);
        if (event.type === "model.exported" && operation) {
          download(
            event.source,
            `${fileName(operation.name)}.py`,
            "text/x-python",
          );
          notice.current(
            "PyTorch model exported. Defaults to the allocation-free meta device.",
          );
        }
        if (event.type === "model.compiled" && operation) {
          if (operation.fingerprint === fingerprintRef.current) {
            setCompilation(event.result);
            notice.current(
              `PyTorch verified · ${event.result.parameters.toLocaleString()} parameters · no weight allocation`,
            );
          } else
            notice.current(
              "The previous graph passed PyTorch verification. Verify again to check your latest edits.",
            );
        }
        if (event.type === "error") notice.current(event.message, true);
        if (pending.current.delete(event.request_id))
          setBusy(pending.current.size > 0);
      };
      ws.onclose = () => {
        if (disposed) return;
        setStatus("offline");
        setBusy(false);
        setChecking(true);
        pending.current.clear();
        reconnect = setTimeout(
          connect,
          Math.min(1000 * 2 ** attempts++, 10000),
        );
      };
      ws.onerror = () => ws.close();
    };
    connect();
    return () => {
      disposed = true;
      clearTimeout(reconnect);
      socket.current?.close();
    };
  }, [analyze]);

  // Position/selection updates stay in the editor; only architectural edits hit the engine.
  const semanticGraph = fingerprintRef.current;
  useEffect(() => {
    setCompilation(null);
    setChecking(true);
    revision.current += 1;
    const timeout = setTimeout(analyze, 110);
    return () => clearTimeout(timeout);
  }, [semanticGraph, analyze]);

  const request = useCallback((action: "export" | "compile") => {
    if (socket.current?.readyState !== WebSocket.OPEN) {
      notice.current(
        "The Python engine is reconnecting. Your graph is saved locally.",
        true,
      );
      return;
    }
    const id = crypto.randomUUID();
    pending.current.set(id, {
      action,
      fingerprint: fingerprintRef.current,
      name: currentGraph.current.name,
    });
    setBusy(true);
    socket.current.send(
      JSON.stringify({
        type: `model.${action}`,
        graph: cleanGraph(currentGraph.current),
        request_id: id,
      }),
    );
  }, []);
  return { analysis, status, busy, checking, compilation, request };
}

export const fileName = (name: string) =>
  name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_|_$/g, "") || "architecture";
export function download(
  content: string,
  name: string,
  type = "application/json",
) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 500);
}
