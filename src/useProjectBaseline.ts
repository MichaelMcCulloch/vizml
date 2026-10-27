import { useCallback, useEffect, useRef, useState } from "react";
import { cleanGraph, type ModelGraph } from "./model";
import { openWorkspace } from "./workspace";

type LoadProject = (id: string) => Promise<{ graph: ModelGraph }>;
const loadProject: LoadProject = id => openWorkspace("projects", id);

/** Compare the browser draft with the actual saved project, including after reload. */
export function useProjectBaseline(projectId: string | null, load: LoadProject = loadProject) {
  const [baseline, setBaseline] = useState<string | null>(null);
  const [status, setStatus] = useState<"checking" | "ready" | "unavailable">("checking");
  const revision = useRef(0);
  const clear = useCallback(() => { revision.current++; setBaseline(null); setStatus("checking"); }, []);
  const saved = useCallback((graph: ModelGraph) => {
    revision.current++;
    setBaseline(JSON.stringify(cleanGraph(graph)));
    setStatus("ready");
  }, []);
  useEffect(() => {
    if (!projectId) return;
    const requestRevision = revision.current;
    let cancelled = false;
    load(projectId).then(record => {
      if (cancelled || revision.current !== requestRevision) return;
      setBaseline(JSON.stringify(cleanGraph(record.graph)));
      setStatus("ready");
    }).catch(() => {
      if (cancelled || revision.current !== requestRevision) return;
      setBaseline(null);
      setStatus("unavailable");
    });
    return () => { cancelled = true; };
  }, [projectId, load]);
  return { baseline, status, clear, saved };
}
