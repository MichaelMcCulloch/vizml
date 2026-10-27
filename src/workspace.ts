import { cleanGraph, parseGraph, type ModelGraph } from "./model";

export type WorkspaceKind = "projects" | "templates";
export type WorkspaceItem = { id: string; name: string; updated_at: string };
export type WorkspaceRecord = WorkspaceItem & { graph: ModelGraph };

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  const result = await response.json();
  if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "Workspace request failed.");
  return result as T;
}
export async function listWorkspace(kind: WorkspaceKind): Promise<WorkspaceItem[]> {
  return (await request<{ items: WorkspaceItem[] }>(`/api/workspace/${kind}`)).items;
}
export async function openWorkspace(kind: WorkspaceKind, id: string): Promise<WorkspaceRecord> {
  const result = await request<WorkspaceRecord>(`/api/workspace/${kind}/${encodeURIComponent(id)}`);
  return { ...result, graph: parseGraph(JSON.stringify(result.graph)) };
}
export async function saveWorkspace(kind: WorkspaceKind, name: string, graph: ModelGraph, id?: string): Promise<WorkspaceRecord> {
  const path = `/api/workspace/${kind}${id ? `/${encodeURIComponent(id)}` : ""}`;
  return request(path, { method: id ? "PUT" : "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name, graph: cleanGraph(graph) }) });
}
export async function deleteWorkspace(kind: WorkspaceKind, id: string): Promise<void> {
  await request(`/api/workspace/${kind}/${encodeURIComponent(id)}`, { method: "DELETE" });
}
