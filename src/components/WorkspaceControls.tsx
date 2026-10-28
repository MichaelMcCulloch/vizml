import { useEffect, useState } from "react";
import { ChevronDown, FolderOpen, Save, X } from "lucide-react";
import type { ModelGraph } from "../model";
import { deleteWorkspace, listWorkspace, openWorkspace, saveWorkspace, type WorkspaceItem } from "../workspace";

export default function WorkspaceControls({ graph, projectId, onProjectId, onSaved, onOpen, onNew, getWorkspaceVersion, onNotice }: {
  graph: ModelGraph; projectId: string | null; onProjectId: (id: string | null) => void;
  onSaved: (id: string, graph: ModelGraph, previousName: string, workspaceVersion: number) => boolean;
  onOpen: (graph: ModelGraph, id: string, workspaceVersion: number, originalGraph: ModelGraph) => boolean; onNew: () => void; getWorkspaceVersion: () => number;
  onNotice: (message: string, error?: boolean) => void;
}) {
  const [open, setOpen] = useState(false);
  const [projects, setProjects] = useState<WorkspaceItem[]>([]);
  const [busy, setBusy] = useState(false);
  const [templateName, setTemplateName] = useState<string | null>(null);
  const [projectName, setProjectName] = useState<string | null>(null);
  const [saveCopy, setSaveCopy] = useState(false);
  const refresh = async () => { try { setProjects(await listWorkspace("projects")); } catch (e) { onNotice((e as Error).message, true); } };
  useEffect(() => { if (open) void refresh(); }, [open]);
  const saveProject = async (copy = false, name = graph.name) => {
    if (!name.trim()) return;
    const workspaceVersion = getWorkspaceVersion();
    setBusy(true);
    try {
      const namedGraph = { ...graph, name: name.trim() };
      const result = await saveWorkspace("projects", name.trim(), namedGraph, copy ? undefined : projectId ?? undefined);
      const attached = onSaved(result.id, namedGraph, graph.name, workspaceVersion);
      await refresh(); setProjectName(null); setOpen(false); onNotice(attached ? `Project “${result.name}” saved to the workspace.` : `Project “${result.name}” saved. Your current workspace was left unchanged.`);
    } catch (e) { onNotice((e as Error).message, true); }
    finally { setBusy(false); }
  };
  const openProject = async (item: WorkspaceItem) => {
    const workspaceVersion = getWorkspaceVersion();
    setBusy(true);
    try { const result = await openWorkspace("projects", item.id); const opened = onOpen(result.graph, result.id, workspaceVersion, graph); setOpen(false); onNotice(opened ? `Opened project “${result.name}”.` : `Project “${result.name}” was not opened because the workspace changed.`); }
    catch (e) { onNotice((e as Error).message, true); }
    finally { setBusy(false); }
  };
  const saveTemplate = async () => {
    if (!templateName?.trim()) return;
    setBusy(true);
    try {
      const metadata = { ...graph.metadata }; delete metadata.weights;
      await saveWorkspace("templates", templateName.trim(), { ...graph, metadata }); onNotice(`Template “${templateName.trim()}” saved.`); setTemplateName(null); setOpen(false); }
    catch (e) { onNotice((e as Error).message, true); }
    finally { setBusy(false); }
  };
  useEffect(() => {
    const save = () => {
      if (busy) return;
      if (projectId) void saveProject();
      else { setSaveCopy(false); setProjectName(graph.name); setOpen(false); }
    };
    window.addEventListener("vizml:save-project", save);
    return () => window.removeEventListener("vizml:save-project", save);
  }, [graph, projectId, busy]);
  return <>
    <div className="workspace-menu-wrap">
      <button className={`quiet-button example-trigger ${open ? "active" : ""}`} aria-label="Projects" aria-expanded={open} onClick={() => setOpen(!open)}><FolderOpen size={14} /> Projects <ChevronDown size={12} /></button>
      {open && <><button className="menu-dismiss" aria-label="Close projects menu" onClick={() => setOpen(false)} /><div className="workspace-menu">
        <button disabled={busy} onClick={() => { onNew(); setOpen(false); }}>New project<small>Start with an empty graph</small></button>
        <button disabled={busy || !graph.name.trim()} onClick={() => { if (projectId) void saveProject(); else { setSaveCopy(false); setProjectName(graph.name); setOpen(false); } }}><Save size={13} /> Save project<small>{projectId ? "Update this saved project" : "Create a saved project"}</small></button>
        {projectId && <button disabled={busy || !graph.name.trim()} onClick={() => { setSaveCopy(true); setProjectName(graph.name); setOpen(false); }}>Save as new project<small>Keep the existing project too</small></button>}
        <button disabled={busy} onClick={() => { setTemplateName(graph.name); setOpen(false); }}>Save as template<small>Reuse this graph as a starting point</small></button>
        <div className="workspace-heading">SAVED PROJECTS</div>
        {!projects.length && <p className="workspace-empty">No saved projects yet.</p>}
        {projects.map(item => <div className="workspace-item" key={item.id}>
          <button disabled={busy} onClick={() => void openProject(item)}><strong>{item.name}</strong><small>{new Date(item.updated_at).toLocaleString()}</small></button>
          <button disabled={busy} className="icon-button" aria-label={`Delete project ${item.name}`} onClick={async () => {
            if (!window.confirm(`Delete saved project “${item.name}”?`)) return;
            try { await deleteWorkspace("projects", item.id); if (projectId === item.id) onProjectId(null); await refresh(); onNotice("Project deleted."); }
            catch (e) { onNotice((e as Error).message, true); }
          }}><X size={13}/></button>
        </div>)}
      </div></>}
    </div>
    {projectName !== null && <div className="modal-backdrop" onClick={() => setProjectName(null)}><section className="workspace-dialog" role="dialog" aria-label="Save project" aria-modal="true" onClick={e => e.stopPropagation()}>
      <h2>{saveCopy ? "Save as new project" : "Save project"}</h2><p>Save this graph in the workspace.</p>
      <label className="field-label" htmlFor="project-name">Project name</label>
      <input id="project-name" autoFocus className="text-field" value={projectName} maxLength={100} onChange={e => setProjectName(e.target.value)} onKeyDown={e => { if (e.key === "Enter") void saveProject(saveCopy, projectName); if (e.key === "Escape") setProjectName(null); }} />
      <div className="workspace-dialog-actions"><button className="quiet-button" onClick={() => setProjectName(null)}>Cancel</button><button className="export-button" disabled={busy || !projectName.trim()} onClick={() => void saveProject(saveCopy, projectName)}>Save project</button></div>
    </section></div>}
    {templateName !== null && <div className="modal-backdrop" onClick={() => setTemplateName(null)}><section className="workspace-dialog" role="dialog" aria-label="Save template" aria-modal="true" onClick={e => e.stopPropagation()}>
      <h2>Save as template</h2><p>Save a reusable copy of the current graph.</p>
      <label className="field-label" htmlFor="template-name">Template name</label>
      <input id="template-name" autoFocus className="text-field" value={templateName} maxLength={100} onChange={e => setTemplateName(e.target.value)} onKeyDown={e => { if (e.key === "Enter") void saveTemplate(); if (e.key === "Escape") setTemplateName(null); }} />
      <div className="workspace-dialog-actions"><button className="quiet-button" onClick={() => setTemplateName(null)}>Cancel</button><button className="export-button" disabled={busy || !templateName.trim()} onClick={() => void saveTemplate()}>Save template</button></div>
    </section></div>}
  </>;
}
