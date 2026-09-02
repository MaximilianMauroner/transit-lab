import { resolve } from "node:path";

export function buildExplorerClient(outdir = resolve(import.meta.dir, "../public/dist")) {
  return Bun.build({
    entrypoints: [resolve(import.meta.dir, "app.ts")],
    outdir,
    naming: "app.[ext]",
    minify: false,
    sourcemap: "external"
  });
}

if (import.meta.main) {
  const result = await buildExplorerClient();
  if (!result.success) {
    for (const log of result.logs) console.error(log);
    process.exit(1);
  }
  console.log(`Built ${result.outputs.length} Explorer client bundle(s).`);
}
