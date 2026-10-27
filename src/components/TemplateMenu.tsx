import { useEffect, useState } from "react";
import { ArrowUpRight, GitBranch, Plus, Search, X } from "lucide-react";
import { templateInfo, type ModelGraph } from "../model";
import { deleteWorkspace, listWorkspace, openWorkspace, saveWorkspace, type WorkspaceItem } from "../workspace";

export default function TemplateMenu({
  onLoad, onLoadSaved, onNotice, graph,
}: {
  graph: ModelGraph;
  onLoad: (key: string) => void;
  onLoadSaved: (graph: ModelGraph) => void;
  onNotice: (message: string, error?: boolean) => void;
}) {
  const [query, setQuery] = useState("");
  const [savedTemplates, setSavedTemplates] = useState<WorkspaceItem[]>([]);
  const [workspaceError, setWorkspaceError] = useState("");
  const [newName, setNewName] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const refresh = () => listWorkspace("templates").then(items => { setSavedTemplates(items); setWorkspaceError(""); }).catch(error => setWorkspaceError(error.message));
  useEffect(() => { void refresh(); }, []);
  const savedMatches = savedTemplates.filter(item => item.name.toLowerCase().includes(query.toLowerCase()));
  const starters = {
    multimodal: {
      title: "Vision + features",
      subtitle: "CNN meets structured data",
    },
    residual: { title: "Residual vision", subtitle: "A skip connection in 3D" },
    mlp: {
      title: "Perceptron regressor",
      subtitle: "A fully connected network",
    },
    dilated: {
      title: "Dilated feature extractor",
      subtitle: "A wider receptive field",
    },
    blank: {
      title: "Empty architecture",
      subtitle: "Make something of your own",
    },
  };
  const groups: [string, string[]][] = [
    ["Recurrent", ["rnn", "gru", "lstm", "conv_rnn", "conv_gru", "conv_lstm"]],
    ["Catan · PPG", ["catan_convnext_small", "catan_original"]],
    [
      "Transformers",
      [
        "transformer_encoder",
        "transformer_decoder",
        "transformer_seq2seq",
        "vit_tiny",
      ],
    ],
    [
      "DINOv3 · ViT",
      ["dinov3_vit_small", "dinov3_vit_base", "dinov3_vit_large"],
    ],
    ["Reference models", ["deepseek_v41_flash_1017"]],
    [
      "DINOv3 · ConvNeXt",
      [
        "dinov3_convnext_tiny",
        "dinov3_convnext_small",
        "dinov3_convnext_base",
        "dinov3_convnext_large",
      ],
    ],
    ["Starter graphs", Object.keys(starters)],
  ];
  const items: Record<string, { title: string; subtitle: string }> = {
    ...templateInfo,
    ...starters,
  };
  let found = false;
  return (
    <div className="example-menu template-menu">
      <div className="template-search">
        <Search size={13} />
        <input
          autoFocus
          aria-label="Find a template"
          placeholder="Find a template…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </div>
      <div className="template-save-current">
        {newName === null ? <button className="quiet-button" onClick={() => setNewName(graph.name)}><Plus size={13} /> Save current as template</button> : <>
          <label className="field-label" htmlFor="inline-template-name">Template name</label>
          <input id="inline-template-name" className="text-field" value={newName} maxLength={100} onChange={e => setNewName(e.target.value)} />
          <div className="schematic-actions"><button className="quiet-button" onClick={() => setNewName(null)}>Cancel</button><button className="quiet-button" disabled={saving || !newName.trim()} onClick={async () => {
            setSaving(true);
            try {
              const metadata = { ...graph.metadata }; delete metadata.weights;
              await saveWorkspace("templates", newName.trim(), { ...graph, metadata });
              await refresh(); setNewName(null); onNotice("Template saved.");
            } catch (e) { onNotice((e as Error).message, true); } finally { setSaving(false); }
          }}>Save template</button></div>
        </>}
      </div>
      {(savedMatches.length > 0 || workspaceError) && <section>
        <div className="menu-eyebrow">Saved templates</div>
        {workspaceError && <p className="workspace-empty">Saved templates unavailable: {workspaceError}</p>}
        {savedMatches.map(item => <div className="saved-template-item" key={item.id}>
          <button onClick={async () => { try { const result = await openWorkspace("templates", item.id); onLoadSaved({ ...result.graph, name: result.name }); } catch (e) { onNotice((e as Error).message, true); } }}><span className="example-icon"><GitBranch size={15}/></span><span><strong>{item.name}</strong><small>Saved template</small></span><ArrowUpRight size={12}/></button>
          <button className="icon-button" aria-label={`Delete template ${item.name}`} onClick={async () => {
            if (!window.confirm(`Delete saved template “${item.name}”?`)) return;
            try { await deleteWorkspace("templates", item.id); await refresh(); onNotice("Template deleted."); } catch (e) { onNotice((e as Error).message, true); }
          }}><X size={13}/></button>
        </div>)}
      </section>}
      {groups.map(([label, keys]) => {
        const matches = keys.filter((key) =>
          `${items[key].title} ${items[key].subtitle} ${label}`
            .toLowerCase()
            .includes(query.toLowerCase()),
        );
        if (!matches.length) return null;
        found = true;
        return (
          <section key={label}>
            <div className="menu-eyebrow">{label}</div>
            {matches.map((key) => (
              <button key={key} onClick={() => onLoad(key)}>
                <span className="example-icon">
                  {key === "blank" ? (
                    <Plus size={15} />
                  ) : (
                    <GitBranch size={15} />
                  )}
                </span>
                <span>
                  <strong>{items[key].title}</strong>
                  <small>{items[key].subtitle}</small>
                </span>
                <ArrowUpRight size={12} />
              </button>
            ))}
          </section>
        );
      })}
      {!found && !savedMatches.length && <p className="no-results">No matching templates.</p>}
    </div>
  );
}
