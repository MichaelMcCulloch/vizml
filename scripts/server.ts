import { resolve, sep } from "node:path";
import type { ServerWebSocket } from "bun";
import homepage from "../index.html";

const root = resolve(import.meta.dir, "..");
const UPLOAD_LIMIT = 2 * 1024 ** 3;
type SocketData = { upstream: WebSocket; pending: (string | ArrayBuffer)[]; client?: ServerWebSocket<SocketData> };
export type ServerOptions = { port?: number; hostname?: string; engineURL?: string; production?: boolean; dist?: string };

export function startServer(options: ServerOptions = {}) {
  const engine = new URL(options.engineURL ?? process.env.VIZML_ENGINE_URL ?? "http://127.0.0.1:8000");
  const dist = resolve(options.dist ?? resolve(root, "dist"));
  const production = options.production ?? false;
  return Bun.serve({
    hostname: options.hostname ?? process.env.HOST ?? "127.0.0.1",
    port: options.port ?? Number(process.env.PORT ?? 5173),
    development: production ? false : { hmr: true, console: false },
    maxRequestBodySize: UPLOAD_LIMIT,
    routes: production ? undefined : { "/": homepage },
    async fetch(request, server) {
      const url = new URL(request.url);
      if (url.pathname === "/ws") {
        const upstreamURL = new URL("/ws", engine);
        upstreamURL.protocol = engine.protocol === "https:" ? "wss:" : "ws:";
        const upstream = new WebSocket(upstreamURL);
        upstream.binaryType = "arraybuffer";
        const data: SocketData = { upstream, pending: [] };
        upstream.onmessage = event => {
          if (data.client) data.client.send(event.data);
          else data.pending.push(event.data);
        };
        upstream.onclose = event => data.client?.close(event.code === 1000 ? 1000 : 1011, "Engine connection closed");
        upstream.onerror = () => data.client?.close(1011, "Engine unavailable");
        try {
          await new Promise<void>((resolve, reject) => {
            const timeout = setTimeout(() => reject(Error("Engine connection timed out")), 5000);
            upstream.addEventListener("open", () => { clearTimeout(timeout); resolve(); }, { once: true });
            upstream.addEventListener("error", () => { clearTimeout(timeout); reject(Error("Engine unavailable")); }, { once: true });
          });
          if (upstream.readyState !== WebSocket.OPEN) throw Error("Engine disconnected");
          if (server.upgrade(request, { data })) return;
          upstream.close();
          return new Response("WebSocket upgrade required", { status: 426 });
        } catch {
          upstream.close();
          return new Response("Python engine unavailable", { status: 503 });
        }
      }
      if (url.pathname.startsWith("/api/")) {
        const target = new URL(url.pathname + url.search, engine);
        const headers = new Headers(request.headers);
        for (const name of ["host", "connection", "upgrade", "proxy-connection"]) headers.delete(name);
        try {
          return await fetch(target, { method: request.method, headers,
            body: ["GET", "HEAD"].includes(request.method) ? undefined : request.body,
            redirect: "manual", signal: request.signal });
        } catch {
          return Response.json({ detail: "Python engine unavailable. Start the app with bun run dev or bun run start." }, { status: 503 });
        }
      }
      if (!["GET", "HEAD"].includes(request.method)) return new Response("Method not allowed", { status: 405 });
      if (!production && url.pathname === "/favicon.svg") return new Response(Bun.file(resolve(root, "public/favicon.svg")));
      if (production) {
        let pathname: string;
        try { pathname = decodeURIComponent(url.pathname); } catch { return new Response("Not found", { status: 404 }); }
        const filePath = resolve(dist, `.${pathname === "/" ? "/index.html" : pathname}`);
        if (filePath.startsWith(dist + sep) && !pathname.includes("\0")) {
          const file = Bun.file(filePath);
          if (await file.exists()) return new Response(request.method === "HEAD" ? null : file, { headers: {
            "Content-Type": file.type,
            "Cache-Control": filePath.endsWith(".html") ? "no-cache" : "public, max-age=31536000, immutable",
          } });
        }
      }
      return new Response("Not found", { status: 404 });
    },
    websocket: {
      data: {} as SocketData,
      maxPayloadLength: 2_000_000,
      backpressureLimit: 4 * 1024 ** 2,
      closeOnBackpressureLimit: true,
      open(socket) {
        socket.data.client = socket;
        for (const message of socket.data.pending) socket.send(message);
        socket.data.pending.length = 0;
        if (socket.data.upstream.readyState !== WebSocket.OPEN) socket.close(1011, "Engine disconnected");
      },
      message(socket, message) {
        if (socket.data.upstream.readyState === WebSocket.OPEN) socket.data.upstream.send(message);
        else socket.close(1011, "Engine disconnected");
      },
      close(socket) { socket.data.client = undefined; socket.data.upstream.close(); },
    },
  });
}

if (import.meta.main) {
  const server = startServer({ production: process.argv.includes("--production") });
  console.log(`vizml · Bun ${Bun.version} · ${server.url}`);
}
