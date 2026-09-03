import { mkdir } from "node:fs/promises";
import { resolve } from "node:path";

async function buildStyles(outdir: string) {
  const input = resolve(import.meta.dir, "styles.css");
  const output = resolve(outdir, "styles.css");
  const tailwind = resolve(import.meta.dir, "../../../../node_modules/.bin/tailwindcss");
  const process = Bun.spawn([
    tailwind,
    "-i",
    input,
    "-o",
    output,
    "--minify"
  ], { stdout: "pipe", stderr: "pipe" });
  const [stdout, stderr] = await Promise.all([
    new Response(process.stdout).text(),
    new Response(process.stderr).text()
  ]);
  const exitCode = await process.exited;
  if (exitCode !== 0) {
    throw new Error(`Tailwind build failed (${exitCode}): ${stderr || stdout}`);
  }
}

export async function buildStudioClient(outdir = resolve(import.meta.dir, "../../public/dist")) {
  await mkdir(outdir, { recursive: true });
  await buildStyles(outdir);
  return Bun.build({
    entrypoints: [resolve(import.meta.dir, "app.tsx")],
    outdir,
    naming: "app.[ext]",
    minify: false,
    sourcemap: "external"
  });
}

if (import.meta.main) {
  const result = await buildStudioClient();
  if (!result.success) {
    for (const log of result.logs) console.error(log);
    process.exit(1);
  }
  console.log(`Built ${result.outputs.length} Studio client bundle(s).`);
}
