import {
  Check,
  CheckCircle2,
  CircleAlert,
  HelpCircle,
  Loader2,
  Terminal,
} from "lucide-react";
import {
  bytesText,
  formatNumber,
  type Analysis,
  type ModelGraph,
} from "../model";

type Props = {
  graph: ModelGraph;
  analysis: Analysis | null;
  engineStatus: "connecting" | "connected" | "offline";
  checking: boolean;
  busy: boolean;
  verified: boolean;
  canBuild: boolean;
  saved: boolean;
  onIssues: () => void;
  onVerify: () => void;
  onHelp: () => void;
};

export default function StatusBar({
  graph,
  analysis,
  engineStatus,
  checking,
  busy,
  verified,
  canBuild,
  saved,
  onIssues,
  onVerify,
  onHelp,
}: Props) {
  const proposal = !!graph.metadata?.proposal;
  const residuals = graph.nodes
    .filter((n) =>
      ["convnext_stage", "transformer_block", "deepseek_stage"].includes(
        n.data.kind,
      ),
    )
    .reduce((count, n) => count + Number(n.data.params.blocks), 0);
  const health =
    checking || !analysis
      ? "Checking"
      : analysis.valid
        ? proposal ? "Proposal" : graph.mode === "schematic" ? "Snapshot" : "Valid"
        : "Check graph";
  const count = analysis?.errors.length || analysis?.warnings.length || 0;
  const issueLabel = analysis?.errors.length
    ? `${count} ${count === 1 ? "issue" : "issues"}`
    : `${count} ${count === 1 ? "note" : "notes"}`;
  return (
    <footer className="app-footer status-bar">
      <div className="status-metrics">
        <div
          className="status-metric"
          title={`${graph.mode === "schematic" ? "Declared parameters" : "Parameters"}: ${analysis?.parameters.toLocaleString() ?? "pending"} · fp32 weight storage`}
        >
          <span className="status-label">Params</span>
          <span className="status-label-short">P</span>
          <strong>{analysis ? formatNumber(analysis.parameters) : "—"}</strong>
          <span className="status-detail">
            {analysis ? `${bytesText(analysis.parameters * 4)} fp32` : ""}
          </span>
        </div>
        <div
          className="status-metric"
          title={`${graph.nodes.length} graph nodes · ${graph.edges.length} connections${residuals ? ` · ${residuals} internal residual blocks` : ""}`}
        >
          <span className="status-label">Nodes</span>
          <span className="status-label-short">N</span>
          <strong>{graph.nodes.length}</strong>
          <span className="status-detail">
            {residuals
              ? `${residuals} residual blocks`
              : `${graph.edges.length} connections`}
          </span>
        </div>
        <div
          className="status-metric"
          title={
            graph.mode === "schematic"
              ? "Sum of declared output tensor sizes · batch 1 · fp32 · source example used for symbolic dimensions · excludes hidden operations and optimizer state"
              : analysis?.executable === false
              ? "Visible output tensor sizes only · batch 1 · fp32 · specialized reference internals excluded"
              : "Activation output estimate · batch 1 · fp32 · includes grouped stage operations"
          }
        >
          <span className="status-label">Activations</span>
          <span className="status-label-short">Act</span>
          <strong>
            {analysis ? bytesText(analysis.activation_bytes) : "—"}
          </strong>
        </div>
        <button
          className={`status-health ${analysis?.valid ? "healthy" : "unhealthy"}`}
          onClick={onIssues}
          aria-label={`Graph health: ${health}${count ? `, ${issueLabel}` : ""}`}
          title="Inspect graph validation and notes"
        >
          {checking || !analysis ? (
            <Loader2 size={11} className="spin" />
          ) : analysis.valid ? (
            <CheckCircle2 size={11} />
          ) : (
            <CircleAlert size={11} />
          )}
          <strong>{health}</strong>
          {!!count && <span>{issueLabel}</span>}
        </button>
      </div>
      <div className="status-actions">
        <span
          className={`status-engine ${engineStatus === "connected" ? "connected" : "offline"}`}
          title={
            engineStatus === "connected"
              ? "Python engine connected · WebSocket"
              : "Reconnecting to the Python engine"
          }
        >
          <i />
          WS
        </span>
        <span
          className="status-saved"
          title={saved ? "Saved locally in this browser" : "Saving graph…"}
          aria-label={saved ? "Saved locally" : "Saving graph"}
        >
          {saved ? <Check size={11} /> : <i />}
        </span>
        <button
          disabled={!canBuild || busy || analysis?.executable === false}
          className={`verify-button ${verified ? "verified" : ""}`}
          onClick={onVerify}
          title={
            analysis?.executable === false
              ? proposal ? "Untrained proposal. The standalone CNN encoder is implemented; full game-specific forward and PPG training require application integration." : graph.mode === "schematic" ? "Supplied schematic: connections and operations are descriptive claims. Tensor ownership can be checked against a checkpoint." : "Reference architecture: specialized DeepSeek kernels are not implemented."
              : "Build and verify the model in PyTorch without allocating weights"
          }
        >
          {busy ? (
            <Loader2 className="spin" size={12} />
          ) : verified ? (
            <CheckCircle2 size={12} />
          ) : (
            <Terminal size={12} />
          )}
          <span className="status-verify-full">
            {analysis?.executable === false
              ? proposal ? "Untrained proposal" : graph.mode === "schematic" ? "Descriptive schematic" : "Reference architecture"
              : verified
                ? "PyTorch verified"
                : "Verify in PyTorch"}
          </span>
          <span className="status-verify-short">
            {analysis?.executable === false
              ? proposal ? "Untrained" : graph.mode === "schematic" ? "Schematic" : "Reference"
              : verified
                ? "Verified"
                : "Verify"}
          </span>
          {analysis?.executable !== false && (
            <span className="status-meta">meta</span>
          )}
        </button>
        <button
          className="status-help"
          onClick={onHelp}
          aria-label="Quick guide"
          title="Quick guide · ?"
        >
          <HelpCircle size={12} />
          <span>Quick guide</span>
        </button>
      </div>
    </footer>
  );
}
