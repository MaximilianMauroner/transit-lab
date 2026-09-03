import { createHash, randomUUID } from "node:crypto";
import { link, mkdir, open, readFile, stat, unlink } from "node:fs/promises";
import { basename, dirname, relative, resolve, sep } from "node:path";
import {
  ARTIFACT_MANIFEST_SCHEMA_VERSION,
  assertSafeId,
  stableStringify,
  validateArtifactManifest
} from "../../contracts/src/index.ts";

function inside(root, candidate) {
  const base = resolve(root);
  const path = resolve(candidate);
  return path === base || path.startsWith(`${base}${sep}`);
}

export function repositoryRelative(root, path) {
  const absolute = resolve(path);
  if (!inside(root, absolute)) throw new Error("artifact file must be inside the repository root");
  return relative(root, absolute).split(sep).join("/");
}

export async function sha256File(path) {
  const bytes = await readFile(path);
  return createHash("sha256").update(bytes).digest("hex");
}

export async function describeArtifactFile(root, path) {
  const absolute = resolve(path);
  const info = await stat(absolute);
  return {
    path: repositoryRelative(root, absolute),
    sizeBytes: info.size,
    sha256: await sha256File(absolute)
  };
}

export async function createArtifactManifest({
  root,
  artifactId,
  kind,
  fingerprint,
  files,
  producingRunId = null,
  gitCommit = process.env.TRANSIT_LAB_GIT_COMMIT || "working-tree",
  configuration = {},
  inputs = [],
  metadata = {},
  createdAt = new Date().toISOString()
}) {
  assertSafeId(artifactId, "artifactId");
  const manifest = {
    schemaVersion: ARTIFACT_MANIFEST_SCHEMA_VERSION,
    artifactId,
    kind,
    fingerprint,
    sha256: files.length === 1 ? files[0].sha256 : await sha256FileFromDescriptors(files),
    createdAt,
    producingRunId,
    inputs,
    gitCommit,
    configuration,
    files,
    metadata
  };
  validateArtifactManifest(manifest);
  return manifest;
}

async function sha256FileFromDescriptors(files) {
  const digest = createHash("sha256");
  for (const file of files) digest.update(`${file.path}\0${file.sha256}\0${file.sizeBytes ?? 0}\n`);
  return digest.digest("hex");
}

export async function writeArtifactManifest(path, manifest) {
  validateArtifactManifest(manifest);
  const target = resolve(path);
  await mkdir(dirname(target), { recursive: true });
  try {
    const existing = JSON.parse(await readFile(target, "utf8"));
    validateArtifactManifest(existing);
    if (stableStringify(existing) !== stableStringify(manifest)) {
      throw new Error(`refusing to overwrite immutable artifact manifest ${target}`);
    }
    return existing;
  } catch (error) {
    if (error instanceof Error && error.message.startsWith("refusing to overwrite")) throw error;
  }

  const encoded = `${JSON.stringify(manifest, null, 2)}\n`;
  const temporary = resolve(
    dirname(target),
    `.${basename(target)}.tmp-${process.pid}-${randomUUID()}`
  );
  let installed = false;
  try {
    const file = await open(temporary, "wx");
    try {
      await file.writeFile(encoded);
      await file.sync();
    } finally {
      await file.close();
    }

    // A hard link creates the final name without ever replacing a competing
    // immutable manifest. The temporary file is on the same filesystem, so
    // this is an atomic publish after the payload has been synced.
    try {
      await link(temporary, target);
      installed = true;
    } catch (error) {
      if (error?.code !== "EEXIST") throw error;
    }
  } finally {
    try { await unlink(temporary); } catch (error) {
      if (error?.code !== "ENOENT") throw error;
    }
  }

  if (!installed) {
    const existing = JSON.parse(await readFile(target, "utf8"));
    validateArtifactManifest(existing);
    if (stableStringify(existing) !== stableStringify(manifest)) {
      throw new Error(`refusing to overwrite immutable artifact manifest ${target}`);
    }
    return existing;
  }

  // Directory fsync is available on Unix. The file itself is synced on every
  // platform supported by Node; Windows/NTFS still gets the atomic publish.
  if (process.platform !== "win32") {
    const directory = await open(dirname(target), "r");
    try { await directory.sync(); } finally { await directory.close(); }
  }
  return manifest;
}
