import fs from "node:fs/promises";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const source = new URL("./input/ncstate-template.potx", import.meta.url);
const outputDir = new URL("./template-inspection/", import.meta.url);
await fs.mkdir(outputDir, { recursive: true });

const presentation = await PresentationFile.importPptx(
  await FileBlob.load(source.pathname.replace(/^\/(.:)/, "$1")),
);

const content = await presentation.inspect({
  select: "$.slides[*]",
  include: [
    "id", "index", "layout", "title",
    "elements.id", "elements.kind", "elements.name", "elements.text",
    "elements.position", "elements.style",
  ],
});
await fs.writeFile(new URL("content.ndjson", outputDir), content.ndjson, "utf8");

const layouts = [];
for (const [layoutIndex, layout] of presentation.layouts.items.entries()) {
  layouts.push({
    id: layout.id,
    name: layout.name,
    placeholders: layout.placeholders.summary(),
  });
  const layoutPng = await layout.export({ format: "png", scale: 0.75 });
  await fs.writeFile(
    new URL(`layout-${String(layoutIndex + 1).padStart(2, "0")}-${(layout.name || "master").replace(/[^a-z0-9]+/gi, "-").toLowerCase()}.png`, outputDir),
    new Uint8Array(await layoutPng.arrayBuffer()),
  );
}
await fs.writeFile(
  new URL("layouts.json", outputDir),
  JSON.stringify(layouts, null, 2),
  "utf8",
);

for (const [index, slide] of presentation.slides.items.entries()) {
  const png = await presentation.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(
    new URL(`slide-${String(index + 1).padStart(2, "0")}.png`, outputDir),
    new Uint8Array(await png.arrayBuffer()),
  );
  const layout = await slide.export({ format: "layout", detail: "full" });
  await fs.writeFile(
    new URL(`slide-${String(index + 1).padStart(2, "0")}.layout.json`, outputDir),
    await layout.text(),
    "utf8",
  );
}

const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
await fs.writeFile(
  new URL("montage.webp", outputDir),
  new Uint8Array(await montage.arrayBuffer()),
);

console.log(JSON.stringify({
  slides: presentation.slides.items.length,
  masters: presentation.masters.items.map((m) => ({ id: m.id, name: m.name })),
  layouts,
}, null, 2));
