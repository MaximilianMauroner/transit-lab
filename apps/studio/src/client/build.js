import { resolve } from "node:path";

const result = await Bun.build({
  entrypoints: [resolve(import.meta.dir, "app.js")],
  outdir: resolve(import.meta.dir, "../../public/dist"),
  naming: "app.[ext]",
  minify: false,
  sourcemap: "external"
});
if (!result.success) {
  for (const log of result.logs) console.error(log);
  process.exit(1);
}
console.log(`Built ${result.outputs.length} Studio client bundle(s).`);
