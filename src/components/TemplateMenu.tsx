import { useState } from "react";
import { ArrowUpRight, GitBranch, Plus, Search } from "lucide-react";
import { templateInfo } from "../model";

export default function TemplateMenu({
  onLoad,
}: {
  onLoad: (key: string) => void;
}) {
  const [query, setQuery] = useState("");
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
      {!found && <p className="no-results">No matching templates.</p>}
    </div>
  );
}
