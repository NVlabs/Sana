import { defineConfig } from "vite";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";

const here = path.dirname(fileURLToPath(import.meta.url));
const sdkDirectory = path.dirname(
  createRequire(import.meta.url).resolve("@reactor-team/js-sdk"),
);
export default defineConfig({
  optimizeDeps: {
    exclude: ["@reactor-team/js-sdk"],
    include: [
      "react",
      "react/jsx-runtime",
      "react/jsx-dev-runtime",
      "react-dom/client",
      "awaitqueue",
      "hls.js",
      "mp4box",
    ],
  },
  server: {
    host: "127.0.0.1",
    strictPort: true,
    fs: { allow: [here, sdkDirectory] },
  },
  build: { target: "es2022" },
  plugins: [
    {
      name: "reactor-sdk-wasm-assets",
      generateBundle() {
        for (const name of ["reactor_wasm.js", "reactor_wasm_bg.wasm"]) {
          this.emitFile({
            type: "asset",
            fileName: `assets/wasm/${name}`,
            source: readFileSync(path.join(sdkDirectory, "wasm", name)),
          });
        }
      },
    },
  ],
});
