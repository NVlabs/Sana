import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { copyFile, link, mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const demo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const source = path.resolve(
  process.argv[2] ||
    process.env.STREAMRENDER_SOURCE ||
    path.join(demo, "../.."),
);
const root = path.join(source, "models/streamrender_h3");
const output = path.join(demo, "public");
const inventory = JSON.parse(
  await readFile(path.join(root, "examples/media_inventory.json"), "utf8"),
);
const entries = [];
for (const name of ["requests_A.jsonl", "requests_B.jsonl"]) {
  for (const line of (await readFile(path.join(root, "examples", name), "utf8"))
    .split("\n")
    .filter(Boolean)) {
    const row = JSON.parse(line);
    entries.push({
      name: row.name,
      video: `/${row.reference_video_path}`,
      image: `/${row.picture}`,
      prompt: row.prompt,
      max_frames: row.num_frames,
    });
  }
}
if (entries.length !== 60 || inventory.files.length !== 90)
  throw Error("Unexpected official example inventory.");
for (const item of inventory.files) {
  if (
    !/^examples\/media\/(videos|pictures)\/[a-zA-Z0-9_-]+\.(mp4|png)$/.test(
      item.path,
    )
  )
    throw Error("Unexpected media path.");
  const origin = path.join(root, item.path),
    target = path.join(output, item.path);
  const hash = createHash("sha256");
  let bytes = 0;
  for await (const chunk of createReadStream(origin)) {
    hash.update(chunk);
    bytes += chunk.length;
  }
  if (bytes !== item.bytes || hash.digest("hex") !== item.sha256)
    throw Error(`Official media hash mismatch: ${item.path}`);
  await mkdir(path.dirname(target), { recursive: true });
  try {
    await link(origin, target);
  } catch (error) {
    if (error.code !== "EEXIST") {
      if (!["EXDEV", "EPERM", "EACCES"].includes(error.code)) throw error;
      await copyFile(origin, target);
    }
  }
}
await writeFile(
  path.join(output, "examples/index.json"),
  JSON.stringify(entries),
);
console.log(
  "Prepared 60 official pairs: 30 videos and 60 unmodified reference pictures.",
);
