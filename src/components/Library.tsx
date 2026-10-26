import { Search, Plus, ArrowUpRight } from "lucide-react";
import { useRef, useState, useEffect } from "react";
import { catalog, type Kind } from "../model";

export default function Library({ onAdd }: { onAdd: (kind: Kind) => void }) {
  const [search, setSearch] = useState("");
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => {
    const listener = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (
        event.key === "/" &&
        !["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)
      ) {
        event.preventDefault();
        input.current?.focus();
      }
    };
    window.addEventListener("keydown", listener);
    return () => window.removeEventListener("keydown", listener);
  }, []);
  const filtered = Object.entries(catalog).filter(([, item]) =>
    item.category !== "Snapshot" && `${item.title} ${item.subtitle} ${item.category}`
      .toLowerCase()
      .includes(search.toLowerCase()),
  );
  return (
    <aside className="library">
      <div className="library-heading">
        <span>Building blocks</span>
        <span className="count-tag">{Object.values(catalog).filter((item) => item.category !== "Snapshot").length}</span>
      </div>
      <div className="search-box">
        <Search size={14} />
        <input
          ref={input}
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Find a block…"
          aria-label="Find a building block"
        />
        <kbd>/</kbd>
      </div>
      <div className="library-scroll">
        {["Inputs", "Recurrent", "Attention", "Layers", "Composition", "Reference", "Outputs"].map((category) => {
          const items = filtered.filter(
            ([, item]) => item.category === category,
          );
          return items.length ? (
            <section className="block-group" key={category}>
              <h3>{category}</h3>
              {items.map(([kind, item]) => (
                <button
                  key={kind}
                  className="block-button"
                  onClick={() => onAdd(kind as Kind)}
                  draggable
                  onDragStart={(e) => {
                    e.dataTransfer.setData("application/vizml", kind);
                    e.dataTransfer.effectAllowed = "copy";
                  }}
                  title={`Add ${item.title} · drag onto the graph`}
                >
                  <span
                    className="block-icon"
                    style={{
                      color: item.color,
                      background: `${item.color}0c`,
                      borderColor: `${item.color}20`,
                    }}
                  >
                    {item.symbol}
                  </span>
                  <span className="block-copy">
                    <strong>{item.title}</strong>
                    <small>{item.subtitle}</small>
                  </span>
                  <Plus size={13} className="block-plus" />
                </button>
              ))}
            </section>
          ) : null;
        })}
        {!filtered.length && (
          <div className="no-results">No matching blocks.</div>
        )}
      </div>
      <div className="library-footer">
        <div>
          <span className="small-spark">✳</span> ROOM TO GROW{" "}
          <ArrowUpRight size={12} />
        </div>
        <p>
          Safetensors weight import
          <br />
          are on the roadmap.
        </p>
      </div>
    </aside>
  );
}
