#!/usr/bin/env node

import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { dirname, extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";

import { chromium } from "playwright-core";
import { WebSocketServer } from "ws";

const projectDirectory = dirname(dirname(fileURLToPath(import.meta.url)));
const distributionDirectory = join(projectDirectory, "dist");
const executableCandidates = [
  process.env.KIWI_CHROMIUM_EXECUTABLE,
  chromium.executablePath(),
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/Applications/Chromium.app/Contents/MacOS/Chromium",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
  "/usr/bin/chromium-browser",
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
].filter(Boolean);

const contentTypes = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".map": "application/json; charset=utf-8",
};

function executablePath() {
  const resolved = executableCandidates.find((candidate) => existsSync(candidate));
  if (!resolved) {
    throw new Error(
      "No Chromium executable found. Set KIWI_CHROMIUM_EXECUTABLE to Chrome or Chromium.",
    );
  }
  return resolved;
}

function staticPath(url) {
  const requested = decodeURIComponent(new URL(url, "http://localhost").pathname);
  const relative = normalize(requested === "/" ? "visual-runner.html" : requested.slice(1));
  const candidate = join(distributionDirectory, relative);
  if (!candidate.startsWith(`${distributionDirectory}/`) && candidate !== distributionDirectory) {
    return undefined;
  }
  return candidate;
}

if (!existsSync(join(distributionDirectory, "visual-runner.html"))) {
  throw new Error("Headless visual assets are missing. Run `npm run build` in simulator-web first.");
}

const server = createServer((request, response) => {
  const path = staticPath(request.url ?? "/");
  if (!path || !existsSync(path) || !statSync(path).isFile()) {
    response.writeHead(404).end("not found");
    return;
  }
  response.writeHead(200, {
    "content-type": contentTypes[extname(path)] ?? "application/octet-stream",
    "cache-control": "no-store",
  });
  createReadStream(path).pipe(response);
});

const webSockets = new WebSocketServer({ noServer: true, maxPayload: 128 * 1024 * 1024 });
server.on("upgrade", (request, socket, head) => {
  if (new URL(request.url ?? "/", "http://localhost").pathname !== "/worker") {
    socket.destroy();
    return;
  }
  webSockets.handleUpgrade(request, socket, head, (connection) => {
    webSockets.emit("connection", connection, request);
  });
});

await new Promise((resolve, reject) => {
  server.once("error", reject);
  server.listen(0, "127.0.0.1", resolve);
});
const address = server.address();
if (!address || typeof address === "string") throw new Error("Unable to resolve supervisor port");

const browser = await chromium.launch({
  executablePath: executablePath(),
  headless: true,
  args: [
    "--enable-webgl",
    "--ignore-gpu-blocklist",
    "--use-angle=swiftshader",
    "--disable-background-timer-throttling",
  ],
});
const page = await browser.newPage({ viewport: { width: 960, height: 640 } });
page.on("console", (message) => process.stderr.write(`[visual-runner] ${message.text()}\n`));
page.on("pageerror", (error) => process.stderr.write(`[visual-runner] ${error.stack ?? error.message}\n`));

const connectionPromise = new Promise((resolve, reject) => {
  const timeout = setTimeout(() => reject(new Error("Timed out waiting for visual worker")), 30_000);
  webSockets.once("connection", (connection) => {
    clearTimeout(timeout);
    resolve(connection);
  });
});
const rendererStack = `chromium-${browser.version()}-webgl`;
const workerUrl = new URL(`http://127.0.0.1:${address.port}/visual-runner.html`);
workerUrl.searchParams.set("ws", `ws://127.0.0.1:${address.port}/worker`);
workerUrl.searchParams.set("renderer_stack", rendererStack);
await page.goto(workerUrl.href, { waitUntil: "load" });
const worker = await connectionPromise;

let pendingResponse;
worker.on("message", (data, isBinary) => {
  if (!isBinary || !pendingResponse) return;
  const resolve = pendingResponse;
  pendingResponse = undefined;
  resolve(Buffer.from(data));
});

function exchange(frame) {
  if (pendingResponse) return Promise.reject(new Error("Only one in-flight request is supported"));
  return new Promise((resolve, reject) => {
    pendingResponse = resolve;
    worker.send(frame, { binary: true }, (error) => {
      if (!error) return;
      pendingResponse = undefined;
      reject(error);
    });
  });
}

let input = Buffer.alloc(0);
for await (const chunk of process.stdin) {
  input = Buffer.concat([input, chunk]);
  while (input.length >= 4) {
    const headerLength = input.readUInt32LE(0);
    if (input.length < 4 + headerLength) break;
    const header = JSON.parse(input.subarray(4, 4 + headerLength).toString("utf8"));
    const binaryLength = Number(header.binary_length ?? 0);
    const frameLength = 4 + headerLength + binaryLength;
    if (!Number.isSafeInteger(binaryLength) || binaryLength < 0) {
      throw new Error("Invalid binary_length in client frame");
    }
    if (input.length < frameLength) break;
    const frame = input.subarray(0, frameLength);
    input = input.subarray(frameLength);
    const response = await exchange(frame);
    if (!process.stdout.write(response)) {
      await new Promise((resolve) => process.stdout.once("drain", resolve));
    }
  }
}

worker.close();
await browser.close();
webSockets.close();
await new Promise((resolve) => server.close(resolve));
