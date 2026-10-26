import { afterEach, expect, test } from "bun:test";
import { startServer } from "./server";

const cleanup: (() => void)[] = [];
afterEach(() => { while (cleanup.length) cleanup.pop()!(); });
function fixture(production=true) {
  const engine = Bun.serve({ port: 0, hostname: "127.0.0.1", fetch: async (request, server) => {
    const url=new URL(request.url);
    if (url.pathname === "/ws") { if (server.upgrade(request)) return; return new Response("Upgrade required",{status:426}); }
    if (url.pathname === "/api/upload") return new Response(await request.arrayBuffer(),{status:201,headers:{"Content-Type":"application/octet-stream","Content-Disposition":"attachment; filename=checkpoint.pt"}});
    if (url.pathname === "/api/health") return Response.json({status:"ok",query:url.searchParams.get("check")});
    return new Response("Engine not found",{status:404});
  }, websocket: { open(socket) { socket.send(JSON.stringify({type:"connected"})); }, message(socket,message) {socket.send(message);} } });
  const app=startServer({port:0,production,engineURL:String(engine.url)});
  cleanup.push(() => engine.stop(true),() => app.stop(true));
  return {app,engine};
}
test("Bun serves the built app and its hashed assets",async () => {
  const {app}=fixture();
  const response=await fetch(app.url);
  expect(response.status).toBe(200);
  expect(response.headers.get("content-type")).toContain("text/html");
  const html=await response.text();
  const source=html.match(/<script[^>]+src=["']([^"']+)/)?.[1];
  expect(source).toBeDefined();
  const asset=await fetch(new URL(source!,app.url));
  expect(asset.status).toBe(200);
  expect(asset.headers.get("content-type")).toMatch(/javascript/);
  expect((await fetch(new URL("/.env",app.url))).status).toBe(404);
  expect((await fetch(new URL("/%2e%2e%2fpackage.json",app.url))).status).toBe(404);
});
test("Bun streams API bodies and preserves download metadata",async () => {
  const {app}=fixture();
  const health=await fetch(new URL("/api/health?check=proxy",app.url));
  expect(await health.json()).toEqual({status:"ok",query:"proxy"});
  const bytes=new Uint8Array(512*1024); crypto.getRandomValues(bytes.subarray(0,65536));
  const response=await fetch(new URL("/api/upload",app.url),{method:"POST",body:bytes,headers:{"Content-Type":"application/octet-stream"}});
  expect(response.status).toBe(201);
  expect(response.headers.get("content-disposition")).toBe("attachment; filename=checkpoint.pt");
  expect(new Uint8Array(await response.arrayBuffer())).toEqual(bytes);
});
test("Bun proxies WebSocket greeting and ordered graph events",async () => {
  const {app}=fixture();
  await new Promise<void>((resolve,reject) => {
    const socket=new WebSocket(new URL("/ws",app.url).href.replace("http:","ws:"));
    const messages:string[]=[];
    const timer=setTimeout(() => {socket.close();reject(Error("WebSocket test timed out"));},4000);
    socket.onopen=() => {socket.send('{"type":"ping","id":1}');socket.send('{"type":"graph.changed","id":2}');};
    socket.onerror=() => {clearTimeout(timer);reject(Error("WebSocket error"));};
    socket.onmessage=event => {
      messages.push(String(event.data));
      if (messages.length !== 3) return;
      try {expect(messages.map(m=>JSON.parse(m).type)).toEqual(["connected","ping","graph.changed"]);socket.close();clearTimeout(timer);resolve();}
      catch(error){clearTimeout(timer);reject(error);}
    };
  });
});
test("Bun returns a useful failure when the Python engine is unavailable",async () => {
  const {app,engine}=fixture();engine.stop(true);
  const response=await fetch(new URL("/api/health",app.url));
  expect(response.status).toBe(503);
  expect((await response.json()).detail).toContain("Python engine unavailable");
});
test("Bun native development HTML bundles React and CSS",async () => {
  const {app}=fixture(false);
  const response=await fetch(app.url);
  expect(response.status).toBe(200);
  const html=await response.text();
  expect(html).toContain('id="root"');
  expect(html).not.toContain("/@vite/client");
  expect(html).toMatch(/\.css/);
});
