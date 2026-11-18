export function validateSelection({ video, image, prompt, seed, limit }) {
  if (!video || !image || !prompt.trim())
    throw Error("Select a semantic video, reference image and prompt.");
  if (prompt.length > 32768) throw Error("Prompt exceeds 32768 characters.");
  if (!Number.isInteger(seed) || seed < 0 || seed > 2147483647)
    throw Error("Seed must be an integer from 0 to 2147483647.");
  if (!Number.isInteger(limit) || limit < 0 || limit > 14400)
    throw Error("Source frame limit must be an integer from 0 to 14400.");
}

export async function sourceFile(source, kind) {
  let file = source;
  if (typeof source === "string") {
    if (!source.startsWith("/examples/media/"))
      throw Error("Unsupported example asset path.");
    const response = await fetch(source);
    if (!response.ok)
      throw Error(`Could not load ${kind}. Reinstall the official examples.`);
    file = new File([await response.blob()], source.split("/").at(-1));
  }
  const limit = kind === "video" ? 256 : 16;
  if (!file?.size || file.size > limit * 1048576)
    throw Error(`${kind} must contain data and be at most ${limit} MiB.`);
  return file;
}
