import { catalog, isRecurrent, shapeText, type LayerNode, type NodeAnalysis } from "../model";

export default function RecurrentInspector({ node, info, onChange }: {
  node: LayerNode;
  info?: NodeAnalysis;
  onChange: (key: string, value: string | number) => void;
}) {
  const { kind, params } = node.data;
  if (!isRecurrent(kind)) return null;
  const p = { ...catalog[kind].defaults, ...params };
  const spatial = kind.startsWith("conv_");
  const lstm = kind.endsWith("lstm");
  const number = (key: string, label: string, max: number, min = 1, step = 1) => (
    <label className="parameter-field" key={key}>
      <span>{label}</span>
      <input aria-label={label} type="number" min={min} max={max} step={step} value={p[key]}
        onChange={e => { if (Number.isFinite(e.target.valueAsNumber)) onChange(key, e.target.valueAsNumber); }} />
    </label>
  );
  const select = (key: string, label: string, choices: [string | number, string][]) => (
    <label className="sequence-option" key={key}>
      <span>{label}</span>
      <select className="text-field" aria-label={label} value={p[key]}
        onChange={e => onChange(key, typeof choices[0][0] === "number" ? Number(e.target.value) : e.target.value)}>
        {choices.map(([value, text]) => <option key={value} value={value}>{text}</option>)}
      </select>
    </label>
  );
  return <>
    <div className="inspector-section">
      <div className="section-title">RECURRENT MEMORY</div>
      <div className="parameter-grid">
        {number("hidden_size", spatial ? "Hidden channels" : "Hidden features", 65535)}
        {number("layers", "Stacked layers", 16)}
        {spatial && number("kernel", "Spatial kernel · odd", 31, 1, 2)}
        {spatial && number("dilation", "Dilation", 16)}
      </div>
    </div>
    <div className="inspector-section sequence-options">
      {select("return_sequences", "Output", [[1, "Every time step"], [0, "Final hidden state"]])}
      {!spatial && select("bidirectional", "Direction", [[0, "Forward · causal"], [1, "Both directions · full context"]])}
      {(kind === "rnn" || kind === "conv_rnn") && select("activation", "State activation", [["tanh", "tanh"], ["relu", "ReLU"]])}
      {select("bias", "Bias", [[1, "Learned bias"], [0, "No bias"]])}
      {number("dropout", "Dropout between layers", 0.99, 0, 0.01)}
      <p className="context-note">Dropout applies between stacked layers; use 0 with a single layer.</p>
    </div>
    <div className="convnext-guide">
      <div className="section-title">{lstm ? "HIDDEN + CELL STATE" : "HIDDEN STATE"}</div>
      <p>{spatial ? "A faint spatial lattice and translucent fill sit inside the woven hoop. Height and width are preserved."
        : "The woven hoop represents recurrence; faint stacked hoops suggest layer depth."} Small stitches suggest gates, and LSTM has an inner thread for its separate cell memory.</p>
      {(kind === "rnn" || kind === "conv_rnn") && <p>Input and previous hidden state meet at a learned weighted sum, then pass through tanh or ReLU. The result returns along the feedback path.</p>}
      {(kind === "gru" || kind === "conv_gru") && <>
        <p><strong>r · Reset</strong> controls what the candidate reads from the previous state. <strong>z · Update</strong> blends retained memory with the new candidate.</p>
        <code>h = z ⊙ h_prev + (1−z) ⊙ candidate</code>
        <p>{spatial ? "Reset acts before the hidden convolution." : "Native PyTorch GRU applies reset after the hidden projection."}</p>
      </>}
      {lstm && <>
        <p>The separate <strong>cell memory c</strong> retains information through time. The forget gate retains old memory, the input gate writes a candidate, and the output gate exposes tanh(c) as hidden state h.</p>
        <code>c = f ⊙ c_prev + i ⊙ candidate</code><br />
        <code>h = o ⊙ tanh(c)</code>
      </>}
      <p>The weave is an architectural illustration. Motion suggests flow, not measured activations or gate values.</p>
      {info?.recurrent && <p>State per batch item: <strong>{shapeText(info.recurrent.state_shape)}</strong>{lstm ? " each for h and c" : " for h"}.</p>}
      <p>State starts at zero for each model call. Recurrence happens inside the layer; connect the graph forward as usual.</p>
      {spatial && <p>{kind === "conv_gru" ? "Reset-before ConvGRU." : kind === "conv_lstm" ? "ConvLSTM without peepholes." : "Convolutional hidden-state transition."} Global average pooling converts each spatial step into a feature vector.</p>}
    </div>
  </>;
}
