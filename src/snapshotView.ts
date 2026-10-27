import type { ModelGraph } from "./model";

export function snapshotScopes(graph: ModelGraph) {
  const scopes = [{ id: "all", label: "Entire model", ids: graph.nodes.map(n => n.id) }];
  const declared = graph.metadata?.view_scopes;
  if (Array.isArray(declared)) {
    const known = new Set(graph.nodes.map(n => n.id));
    for (const item of declared) {
      if (!item || typeof item !== "object" || typeof item.id !== "string" || typeof item.label !== "string" || !Array.isArray(item.ids)) continue;
      const ids = item.ids.filter((id: unknown): id is string => typeof id === "string" && known.has(id));
      if (ids.length && !scopes.some(s => s.id === item.id)) scopes.push({ id: item.id, label: item.label, ids });
    }
    if (scopes.length > 1) return scopes;
  }
  for (const branch of ["actor", "value"]) {
    const ids = graph.nodes.filter(n => n.id === "board" || n.id === "context" ||
      (n.id.startsWith(`${branch}_`) && /^(stem|context|context_norm|condition|conditioned|block\d+_.*|final_norm|mean|max|global_join|state|state_norm|mlp1|mlp2)$/.test(n.id.slice(branch.length + 1)))).map(n => n.id);
    if (ids.length > 5) scopes.push({ id: `${branch}_cnn`, label: `${branch === "actor" ? "Actor" : "Value"} CNN`, ids });
    const blocks = graph.nodes.filter(n => new RegExp(`^${branch}_block\\d+_add$`).test(n.id));
    for (const block of blocks) {
      const number = Number(block.id.match(/block(\d+)/)![1]);
      const prefix = `${branch}_block${number}_`;
      const previous = number === 1 ? `${branch}_conditioned` : `${branch}_block${number - 1}_add`;
      const conv = graph.nodes.find(n => n.id === `${prefix}conv1`);
      scopes.push({ id: `${branch}_block${number}`, label: `${branch === "actor" ? "Actor" : "Value"} · block ${number} · d${conv?.data.params.dilation ?? 1}`, ids: graph.nodes.filter(n => n.id === previous || n.id.startsWith(prefix)).map(n => n.id) });
    }
  }
  if (graph.nodes.some(n => n.id === "policy")) scopes.push({ id: "policy", label: "Legal-action policy", ids: graph.nodes.filter(n => /^(legal|payload_embedding|family_embedding|action_|policy|family_logits|actor_(mlp2|entity|ship|entities))/.test(n.id)).map(n => n.id) });
  if (graph.nodes.some(n => n.id === "value_outcomes")) scopes.push({ id: "outcomes", label: "Outcome heads", ids: graph.nodes.filter(n => /^(actor|value)_(critic|public|board_score|public_score|board_draw|public_draw|signed_score|draw_score|outcome|mean_value|mlp2)/.test(n.id)).map(n => n.id) });
  return scopes;
}

export function preferredSnapshotScope(graph: ModelGraph) {
  // New nodes are not part of the imported snapshot's predefined scopes.
  if (graph.metadata?.source_edited === true) return "all";
  const scopes = snapshotScopes(graph);
  const declared = graph.metadata?.default_scope;
  return typeof declared === "string" && scopes.some(s => s.id === declared) ? declared : scopes.some(s => s.id === "actor_cnn") ? "actor_cnn" : "all";
}
