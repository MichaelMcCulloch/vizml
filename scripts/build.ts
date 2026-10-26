import { rm } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(import.meta.dir, "..");
const outdir = resolve(root, "dist");
await rm(outdir, { recursive: true, force: true });
const result = await Bun.build({ entrypoints: [resolve(root, "index.html")], outdir, target: "browser",
  minify: true, splitting: true, sourcemap: "none", define: { "process.env.NODE_ENV": '"production"' } });
if (!result.success) { console.error(result.logs); process.exit(1); }
await Bun.write(resolve(outdir, "favicon.svg"), Bun.file(resolve(root, "public/favicon.svg")));
for (const output of result.outputs) console.log(`${output.path.replace(root + "/", "")} · ${(output.size / 1024).toFixed(1)} KiB`);
