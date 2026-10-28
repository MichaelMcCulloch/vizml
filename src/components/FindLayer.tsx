import { useState } from "react";
import { Search, X } from "lucide-react";
import { catalog, type ModelGraph } from "../model";

export default function FindLayer({ graph, onFind }: { graph: ModelGraph; onFind: (id: string) => void }) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const term = query.trim().toLowerCase();
  const matches = graph.nodes.filter(node => `${node.data.label} ${node.id} ${catalog[node.data.kind].title}`.toLowerCase().includes(term));
  return <div className="find-layer">
    <button className="icon-button" title="Find a layer" aria-label="Find a layer" aria-expanded={open}
      onClick={() => setOpen(value => !value)}><Search size={14} /></button>
    {open && <div className="find-layer-menu">
      <div className="find-layer-search"><Search size={14} /><input autoFocus aria-label="Search layers" placeholder="Search layers…" value={query}
        onChange={event => setQuery(event.target.value)} onKeyDown={event => {
          if (event.key === "Escape") { event.stopPropagation(); setOpen(false); }
          if (event.key === "Enter" && matches[0]) { onFind(matches[0].id); setOpen(false); }
        }} /><button className="icon-button" aria-label="Close layer search" onClick={() => setOpen(false)}><X size={13} /></button></div>
      <div className="find-layer-results">{matches.length ? matches.map(node => <button key={node.id} onClick={() => { onFind(node.id); setOpen(false); }}>
        <strong>{node.data.label}</strong><small>{catalog[node.data.kind].title} · {node.id}</small>
      </button>) : <p>No matching layers.</p>}</div>
    </div>}
  </div>;
}
