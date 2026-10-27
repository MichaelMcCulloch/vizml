import { expect, test } from "bun:test";
import { renderToStaticMarkup } from "react-dom/server";
import Library from "../src/components/Library";
import SchematicInspector from "../src/components/SchematicInspector";
import WorkspaceControls from "../src/components/WorkspaceControls";
import TemplateMenu from "../src/components/TemplateMenu";
import { parseGraph } from "../src/model";

const graph = parseGraph(JSON.stringify({ version: 1, name: "Editable schematic", mode: "schematic", nodes: [
  { id: "a", position: { x: 0, y: 0 }, data: { kind: "gather", label: "Gather", params: { input_channels: 32, bias: true, nested: [1, 2] }, output_shape: ["B", 4], parameter_count: 0 } },
], edges: [] }));
const noop = () => {};

test("schematic library includes both builder and schematic-specific blocks", () => {
  const markup = renderToStaticMarkup(<Library schematic onAdd={noop} />);
  expect(markup).toContain("Add / residual");
  expect(markup).toContain("Gather");
  expect(markup).toContain("Perceptron");
});
test("schematic inspector exposes editable declaration and deletion", () => {
  const markup = renderToStaticMarkup(<SchematicInspector graph={graph} node={graph.nodes[0]} analysis={null}
    onSelect={noop} onUpdate={noop} onDelete={noop} onDuplicate={noop} onDisconnect={noop} />);
  expect(markup).toContain("Declared output shape");
  expect(markup).toContain("Declared parameter count");
  expect(markup).toContain("Operation settings");
  expect(markup).toContain("input channels");
  expect(markup).toContain('type="checkbox"');
  expect(markup).toContain("Advanced settings");
  expect(markup).toContain("Separate dimensions with × or commas");
  expect(markup).toContain("Delete block");
});
test("project control is available from the top bar", () => {
  const markup = renderToStaticMarkup(<WorkspaceControls graph={graph} projectId={null} onProjectId={noop}
    onSaved={noop} onOpen={noop} onNew={noop} onRename={noop} onNotice={noop} />);
  expect(markup).toContain('aria-label="Projects"');
});

test("the template menu offers saving the current graph", () => {
  const markup = renderToStaticMarkup(<TemplateMenu graph={graph} onLoad={noop} onLoadSaved={noop} onNotice={noop} />);
  expect(markup).toContain("Save current as template");
});
