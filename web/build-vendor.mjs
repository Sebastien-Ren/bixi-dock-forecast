// One-time build of web/vendor. Run with: npm install && npm run vendor
// MapLibre ships ready-to-use module files, so they are copied as they are.
// deck.gl ships many small modules, so the two pieces the map needs are bundled into one file.
import { build } from "esbuild";
import { copyFileSync, mkdirSync, rmSync, writeFileSync } from "node:fs";

mkdirSync("vendor", { recursive: true });
for (const f of ["maplibre-gl.mjs", "maplibre-gl-worker.mjs", "maplibre-gl.css"]) {
  copyFileSync(`node_modules/maplibre-gl/dist/${f}`, `vendor/${f}`);
}
writeFileSync("vendor/_deck-entry.mjs",
  `export { MapLibreOverlay } from "@deck.gl/maplibre";\nexport { ScatterplotLayer } from "@deck.gl/layers";\n`);
await build({
  entryPoints: ["vendor/_deck-entry.mjs"],
  outfile: "vendor/deck.mjs",
  bundle: true, format: "esm", minify: true, target: "es2022",
  external: ["maplibre-gl"],
  logLevel: "info",
});
rmSync("vendor/_deck-entry.mjs");
