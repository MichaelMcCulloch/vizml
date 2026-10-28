import { useEffect, useId, useRef, useState } from "react";
import { Search, X } from "lucide-react";
import { catalog, type ModelGraph } from "../model";

export default function FindLayer({ graph, onFind }: { graph: ModelGraph; onFind: (id: string) => void }) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const root = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const listId = useId();
  const term = query.trim().toLowerCase();
  const matches = graph.nodes.filter(node => `${node.data.label} ${node.id} ${catalog[node.data.kind].title}`.toLowerCase().includes(term));
  const index = Math.min(active, matches.length - 1);
  useEffect(() => {
    if (!open) return;
    const dismiss = (event: PointerEvent) => { if (!root.current?.contains(event.target as Node)) setOpen(false); };
    document.addEventListener("pointerdown", dismiss);
    return () => document.removeEventListener("pointerdown", dismiss);
  }, [open]);
  useEffect(() => {
    if (open && index >= 0) document.getElementById(`${listId}-${index}`)?.scrollIntoView?.({ block: "nearest" });
  }, [open, index, listId]);
  return <div className="find-layer" ref={root}>
    <button ref={trigger} className="icon-button" title="Find a layer" aria-label="Find a layer" aria-expanded={open}
      onClick={() => setOpen(value => !value)}><Search size={14} /></button>
    {open && <div className="find-layer-menu">
      <div className="find-layer-search"><Search size={14} /><input autoFocus role="combobox" aria-label="Search layers" aria-expanded="true" aria-controls={listId} aria-activedescendant={index >= 0 ? `${listId}-${index}` : undefined} placeholder="Search layers…" value={query}
        onChange={event => { setQuery(event.target.value); setActive(0); }} onKeyDown={event => {
          if (event.key === "Escape") { event.stopPropagation(); setOpen(false); trigger.current?.focus(); }
          if (event.key === "ArrowDown" || event.key === "ArrowUp") { event.preventDefault(); if (matches.length) setActive((index + (event.key === "ArrowDown" ? 1 : -1) + matches.length) % matches.length); }
          if (event.key === "Enter" && matches[index]) { event.preventDefault(); onFind(matches[index].id); setOpen(false); trigger.current?.focus(); }
        }} /><button className="icon-button" aria-label="Close layer search" onClick={() => setOpen(false)}><X size={13} /></button></div>
      <div className="find-layer-results" id={listId} role="listbox" aria-label="Matching layers">{matches.length ? matches.map((node, i) => <button key={node.id} id={`${listId}-${i}`} role="option" aria-selected={i === index} onMouseEnter={() => setActive(i)} onClick={() => { onFind(node.id); setOpen(false); }}>
        <strong>{node.data.label}</strong><small>{catalog[node.data.kind].title} · {node.id}</small>
      </button>) : <p>No matching layers.</p>}</div>
    </div>}
  </div>;
}
