import { resolve } from "node:path";
import { startServer } from "./server";

const production = process.argv.includes("--production");
const port = Number(process.env.PORT ?? 5173);
const enginePort = Number(process.env.VIZML_ENGINE_PORT ?? 8000);
const server = startServer({ production, port, engineURL: `http://127.0.0.1:${enginePort}` });
const engine = Bun.spawn([
  "uv", "run", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", String(enginePort),
  ...(production ? [] : ["--reload"]),
], { cwd: resolve(import.meta.dir, ".."), stdout: "inherit", stderr: "inherit",
  env: { ...process.env, VIZML_APP_URL: String(server.url) } });
let stopping = false;
async function stop(code = 0) {
  if (stopping) return;
  stopping = true;
  server.stop(true);
  engine.kill("SIGTERM");
  const timeout = setTimeout(() => { engine.kill("SIGKILL"); }, 5000);
  await engine.exited;
  clearTimeout(timeout);
  process.exit(code);
}
process.once("SIGINT", () => void stop());
process.once("SIGTERM", () => void stop());
engine.exited.then(code => { if (!stopping) void stop(code || 1); });
console.log(`vizml · Bun ${Bun.version} · ${production ? "production" : "development"} · ${server.url}`);
