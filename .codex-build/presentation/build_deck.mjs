import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "D:/NCSU/SEM3/PS/PROJECT/Resource_Aware_planned_Diffusion";
const buildDir = path.join(workspaceDir, ".codex-build/presentation");
const inputPath = path.join(buildDir, "input/ncstate-template.potx");
const referencePptxPath = path.join(buildDir, "input/ncstate-template-reference.pptx");
const brandHeaderPath = path.join(buildDir, "input/template-assets/ppt/media/image1.jpg");
const previewDir = path.join(buildDir, "preview");
const stagingDir = path.join(buildDir, "finalizer");
const outputDir = path.join(workspaceDir, "output/presentation");
const finalPath = path.join(outputDir, "resource_aware_planned_diffusion_4min_ncstate_v6.pptx");

const skillDir = "C:/Users/khush/.codex/plugins/cache/openai-primary-runtime/presentations/26.1112.11217/skills/presentations";
const pythonExecutable = "C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1112-737-0_26.1112.737.0_x64__3k8sg7r9htsxt/dependencies/python/python.exe";

await fs.mkdir(previewDir, { recursive: true });
await fs.mkdir(stagingDir, { recursive: true });
await fs.mkdir(outputDir, { recursive: true });
await fs.copyFile(inputPath, referencePptxPath);
const brandHeaderFile = await fs.readFile(brandHeaderPath);
const brandHeaderBytes = brandHeaderFile.buffer.slice(
  brandHeaderFile.byteOffset,
  brandHeaderFile.byteOffset + brandHeaderFile.byteLength,
);

const presentation = await PresentationFile.importPptx(await FileBlob.load(inputPath));

const RED = "#CC0000";
const DARK_RED = "#990000";
const BLACK = "#111111";
const CHARCOAL = "#333333";
const MID = "#666666";
const LIGHT = "#F2F2F2";
const PALE_RED = "#F9E6E6";
const WHITE = "#FFFFFF";
const FONT = "Arial";

function addText(slide, text, position, options = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: options.fill ?? "none",
    line: options.line ?? { style: "solid", fill: "none", width: 0 },
    ...(options.name ? { name: options.name } : {}),
  });
  shape.text = text;
  shape.text.style = {
    typeface: FONT,
    fontSize: options.fontSize ?? 24,
    color: options.color ?? BLACK,
    bold: options.bold ?? false,
    alignment: options.alignment ?? "left",
    verticalAlignment: options.verticalAlignment ?? "middle",
    autoFit: "shrinkText",
  };
  return shape;
}

function addBox(slide, position, options = {}) {
  return slide.shapes.add({
    geometry: options.geometry ?? "roundRect",
    position,
    fill: options.fill ?? WHITE,
    line: options.line ?? { style: "solid", fill: "#C8C8C8", width: 1 },
    ...(options.name ? { name: options.name } : {}),
  });
}

function addDivider(slide, left, top, width, color = "#D5D5D5", height = 2) {
  return slide.shapes.add({
    geometry: "rect",
    position: { left, top, width, height },
    fill: color,
    line: { style: "solid", fill: color, width: 0 },
  });
}

function setTitle(slide, text) {
  const title = slide.placeholders.getItem("title");
  title.text = text;
  title.text.style = {
    typeface: FONT,
    fontSize: 38,
    bold: true,
    color: BLACK,
    alignment: "left",
    verticalAlignment: "middle",
    autoFit: "shrinkText",
  };
  return title;
}

function repairBrandHeader(slide) {
  slide.images.add({
    blob: brandHeaderBytes,
    contentType: "image/jpeg",
    alt: "NC State University brand header",
    fit: "cover",
    position: { left: 0, top: 0, width: 960, height: 48 },
  });
}

function addTitleOnlySlide(titleText) {
  const slide = presentation.slides.add({ layout: "Title Only" });
  slide.placeholders.addFromLayout();
  setTitle(slide, titleText);
  return slide;
}

// Slide 1: title
const slide1 = presentation.slides.getItem(0);
repairBrandHeader(slide1);
slide1.placeholders.getItem("title").text = "Resource-Aware Planned Diffusion";
slide1.placeholders.getItem("title").text.style = {
  typeface: FONT,
  fontSize: 48,
  bold: true,
  color: BLACK,
  alignment: "center",
  verticalAlignment: "middle",
  autoFit: "shrinkText",
};
slide1.placeholders.getItem("subtitle").text =
  "Extending Planned Diffusion from one GPU to resource-aware multi-GPU execution\nCSC 548 · Parallel Systems · Fall 2026";
slide1.placeholders.getItem("subtitle").text.style = {
  typeface: FONT,
  fontSize: 27,
  color: MID,
  alignment: "center",
  verticalAlignment: "top",
  autoFit: "shrinkText",
};
slide1.speakerNotes.text = [
  "Timing: 0:00–0:10",
  "Planned Diffusion exposes independent semantic chunks inside one response. We extend that idea from single-GPU generation to resource-aware execution across replicated GPUs.",
  "Source: PLANNED_DIFFUSION_PROPOSAL.pdf; README.md",
].join("\n\n");

// Slide 2: original paper
const slide2 = addTitleOnlySlide("Planned Diffusion: plan first, denoise chunks in parallel");
repairBrandHeader(slide2);
addText(slide2, "The problem", { left: 55, top: 155, width: 330, height: 31 }, { fontSize: 23, bold: true, color: DARK_RED });
addBox(slide2, { left: 55, top: 194, width: 350, height: 108 }, { fill: LIGHT, line: { style: "solid", fill: "#D0D0D0", width: 1 } });
addText(slide2, "Autoregressive models", { left: 74, top: 207, width: 310, height: 30 }, { fontSize: 22, bold: true, color: CHARCOAL });
addText(slide2, "Generate token by token; quality is strong, but decoding is sequential.", { left: 74, top: 241, width: 310, height: 48 }, { fontSize: 19, color: MID, verticalAlignment: "top" });
addBox(slide2, { left: 55, top: 316, width: 350, height: 108 }, { fill: LIGHT, line: { style: "solid", fill: "#D0D0D0", width: 1 } });
addText(slide2, "Diffusion language models", { left: 74, top: 329, width: 310, height: 30 }, { fontSize: 22, bold: true, color: CHARCOAL });
addText(slide2, "Can update tokens in parallel, but a fixed denoising order creates a quality–latency tradeoff.", { left: 74, top: 363, width: 310, height: 49 }, { fontSize: 19, color: MID, verticalAlignment: "top" });

addText(slide2, "The paper's answer", { left: 448, top: 155, width: 440, height: 31 }, { fontSize: 23, bold: true, color: DARK_RED });
addBox(slide2, { left: 448, top: 194, width: 440, height: 74 }, { fill: PALE_RED, line: { style: "solid", fill: "#E3A1A1", width: 1 } });
addText(slide2, "1  Autoregressively create a semantic plan", { left: 468, top: 205, width: 400, height: 28 }, { fontSize: 21, bold: true, color: CHARCOAL });
addText(slide2, "Introduction  |  Methods  |  Results  |  Conclusion", { left: 468, top: 235, width: 400, height: 24 }, { fontSize: 17, color: MID, alignment: "center" });
addBox(slide2, { left: 448, top: 282, width: 440, height: 142 }, { fill: WHITE, line: { style: "solid", fill: "#C9C9C9", width: 1 } });
addText(slide2, "2  Denoise every planned chunk in parallel", { left: 468, top: 294, width: 400, height: 29 }, { fontSize: 21, bold: true, color: CHARCOAL });
addText(slide2, "One combined diffusion stage on one GPU", { left: 468, top: 326, width: 400, height: 24 }, { fontSize: 17, color: MID, alignment: "center" });
const paperChunkLabels = ["Intro", "Methods", "Results", "Conclusion"];
const paperChunkWidths = [65, 125, 97, 95];
let paperChunkX = 475;
for (let i = 0; i < paperChunkLabels.length; i += 1) {
  addBox(slide2, { left: paperChunkX, top: 365, width: paperChunkWidths[i], height: 39 }, { geometry: "rect", fill: i === 1 ? RED : "#E17676", line: { style: "solid", fill: WHITE, width: 1 } });
  addText(slide2, paperChunkLabels[i], { left: paperChunkX, top: 365, width: paperChunkWidths[i], height: 39 }, { fontSize: 15, bold: true, color: WHITE, alignment: "center" });
  paperChunkX += paperChunkWidths[i];
}
addText(slide2, "GPU 0", { left: 812, top: 329, width: 58, height: 25 }, { fontSize: 17, bold: true, color: RED, alignment: "right" });
addBox(slide2, { left: 55, top: 448, width: 833, height: 50 }, { fill: CHARCOAL, line: { style: "solid", fill: CHARCOAL, width: 0 } });
addText(slide2, "Reported on AlpacaEval: 1.27×–1.81× faster than autoregressive decoding, with a 0.87%–5.4% win-rate drop", { left: 72, top: 448, width: 799, height: 50 }, { fontSize: 19, bold: true, color: WHITE, alignment: "center" });
slide2.speakerNotes.text = [
  "Timing: 0:10–0:45",
  "The original Planned Diffusion paper starts from a tension. Autoregressive models preserve quality but decode sequentially. Diffusion language models can update tokens in parallel, but choosing a fixed denoising order hurts either quality or latency. Their solution is a two-phase generation process. First, the model autoregressively writes a semantic plan that partitions the response into independent chunks. Second, it denoises all chunks together as one combined sparse sequence on a single GPU. On AlpacaEval, the paper reports a 1.27 to 1.81 times speedup over autoregressive decoding, with a small win-rate reduction.",
  "Source: https://arxiv.org/abs/2510.18087; README.md",
].join("\n\n");

// Slide 3: our proposal
const slide3 = addTitleOnlySlide("Our systems question: can those chunks use different GPUs?");
repairBrandHeader(slide3);
addText(slide3, "We keep the model's semantic plan and token positions.", { left: 58, top: 157, width: 844, height: 38 }, { fontSize: 27, bold: true, color: CHARCOAL, alignment: "center" });
addText(slide3, "We change where each chunk executes.", { left: 58, top: 194, width: 844, height: 39 }, { fontSize: 29, bold: true, color: RED, alignment: "center" });

addBox(slide3, { left: 58, top: 252, width: 390, height: 176 }, { fill: LIGHT, line: { style: "solid", fill: "#C9C9C9", width: 1 } });
addText(slide3, "Original paper", { left: 78, top: 266, width: 350, height: 34 }, { fontSize: 24, bold: true, color: CHARCOAL, alignment: "center" });
addText(slide3, "One combined sparse sequence", { left: 78, top: 306, width: 350, height: 29 }, { fontSize: 20, color: MID, alignment: "center" });
addBox(slide3, { left: 122, top: 354, width: 262, height: 47 }, { geometry: "rect", fill: CHARCOAL, line: { style: "solid", fill: CHARCOAL, width: 0 } });
addText(slide3, "All chunks  ·  GPU 0", { left: 122, top: 354, width: 262, height: 47 }, { fontSize: 21, bold: true, color: WHITE, alignment: "center" });

addBox(slide3, { left: 512, top: 252, width: 390, height: 176 }, { fill: PALE_RED, line: { style: "solid", fill: RED, width: 2 } });
addText(slide3, "Our extension", { left: 532, top: 266, width: 350, height: 34 }, { fontSize: 24, bold: true, color: DARK_RED, alignment: "center" });
addText(slide3, "Rank 0 schedules compact chunk subsets", { left: 532, top: 306, width: 350, height: 29 }, { fontSize: 20, color: MID, alignment: "center" });
for (let i = 0; i < 4; i += 1) {
  const left = 534 + i * 88;
  addBox(slide3, { left, top: 354, width: 78, height: 47 }, { geometry: "rect", fill: i === 0 ? CHARCOAL : RED, line: { style: "solid", fill: WHITE, width: 1 } });
  addText(slide3, `GPU ${i}`, { left, top: 354, width: 78, height: 47 }, { fontSize: 18, bold: true, color: WHITE, alignment: "center" });
}
addText(slide3, "Inputs: predicted chunk lengths · stage size · available GPUs", { left: 58, top: 451, width: 500, height: 30 }, { fontSize: 19, bold: true, color: CHARCOAL });
addText(slide3, "Choices: 1 / 2 / 4 GPUs · round robin or LPT", { left: 478, top: 451, width: 424, height: 30 }, { fontSize: 19, bold: true, color: CHARCOAL, alignment: "right" });
addText(slide3, "Goal: lower latency without changing the planned response", { left: 58, top: 487, width: 844, height: 28 }, { fontSize: 21, color: RED, bold: true, alignment: "center" });
// Reinforce the wordmark at the topmost z-order; some renderers partially mask the template image on this slide.
addBox(slide3, { left: 0, top: 0, width: 280, height: 48 }, { geometry: "rect", fill: RED, line: { style: "solid", fill: RED, width: 0 } });
addText(slide3, "NC STATE UNIVERSITY", { left: 10, top: 0, width: 260, height: 48 }, { fontSize: 17, bold: true, color: WHITE, verticalAlignment: "middle" });
slide3.speakerNotes.text = [
  "Timing: 0:45–1:15",
  "Our proposal is a systems extension, not a new generation objective. We keep the paper's semantic plan, sparse attention structure, and global token positions. We change the execution placement. Rank 0 uses predicted chunk lengths, stage size, and the number of available GPUs to choose one, two, or four GPUs and either round robin or longest-processing-time-first scheduling. Each worker holds the same Dream 7B model. The goal is lower latency without changing the planned response.",
  "Source: semantic_parallel/; dream/generation_utils.py; README.md",
].join("\n\n");

// Slide 4: worked example
const slide4 = addTitleOnlySlide("Worked example: one plan, two executions");
repairBrandHeader(slide4);
addText(slide4, "Generated plan and predicted chunk lengths", { left: 55, top: 151, width: 850, height: 30 }, { fontSize: 21, bold: true, color: CHARCOAL });
const exampleChunks = [
  { label: "Intro  20", width: 92, fill: "#E9A3A3" },
  { label: "Methods  80", width: 280, fill: RED },
  { label: "Results  50", width: 185, fill: DARK_RED },
  { label: "Conclusion  30", width: 125, fill: "#E17676" },
];
let exampleX = 164;
for (const chunk of exampleChunks) {
  addBox(slide4, { left: exampleX, top: 188, width: chunk.width, height: 39 }, { geometry: "rect", fill: chunk.fill, line: { style: "solid", fill: WHITE, width: 1 } });
  addText(slide4, chunk.label, { left: exampleX, top: 188, width: chunk.width, height: 39 }, { fontSize: 17, bold: true, color: WHITE, alignment: "center" });
  exampleX += chunk.width;
}

addBox(slide4, { left: 55, top: 254, width: 405, height: 205 }, { fill: LIGHT, line: { style: "solid", fill: "#C9C9C9", width: 1 } });
addText(slide4, "Original: one GPU", { left: 75, top: 266, width: 365, height: 35 }, { fontSize: 24, bold: true, color: CHARCOAL, alignment: "center" });
addText(slide4, "One combined sparse sequence", { left: 75, top: 307, width: 365, height: 27 }, { fontSize: 19, color: MID, alignment: "center" });
addText(slide4, "shared prefix + all four chunks", { left: 75, top: 335, width: 365, height: 26 }, { fontSize: 18, color: MID, alignment: "center" });
addBox(slide4, { left: 99, top: 377, width: 317, height: 48 }, { geometry: "rect", fill: CHARCOAL, line: { style: "solid", fill: CHARCOAL, width: 0 } });
addText(slide4, "80 steps over the full stage tensor", { left: 99, top: 377, width: 317, height: 48 }, { fontSize: 20, bold: true, color: WHITE, alignment: "center" });

addBox(slide4, { left: 500, top: 254, width: 405, height: 205 }, { fill: PALE_RED, line: { style: "solid", fill: RED, width: 2 } });
addText(slide4, "Ours: four replicated GPUs", { left: 520, top: 266, width: 365, height: 35 }, { fontSize: 24, bold: true, color: DARK_RED, alignment: "center" });
const assignments = [
  ["GPU 0", "Methods 80"], ["GPU 1", "Results 50"], ["GPU 2", "Conclusion 30"], ["GPU 3", "Intro 20"],
];
for (let i = 0; i < assignments.length; i += 1) {
  const left = 520 + (i % 2) * 183;
  const top = 315 + Math.floor(i / 2) * 56;
  addBox(slide4, { left, top, width: 168, height: 44 }, { geometry: "rect", fill: i === 0 ? CHARCOAL : RED, line: { style: "solid", fill: WHITE, width: 1 } });
  addText(slide4, `${assignments[i][0]}  ·  ${assignments[i][1]}`, { left, top, width: 168, height: 44 }, { fontSize: 17, bold: true, color: WHITE, alignment: "center" });
}
addText(slide4, "Rank 0 broadcasts context and gathers chunks", { left: 520, top: 427, width: 365, height: 24 }, { fontSize: 17, color: MID, alignment: "center" });
addBox(slide4, { left: 55, top: 478, width: 850, height: 37 }, { fill: CHARCOAL, line: { style: "solid", fill: CHARCOAL, width: 0 } });
addText(slide4, "Same semantic plan · smaller per-rank sequence · added broadcast and gather", { left: 70, top: 478, width: 820, height: 37 }, { fontSize: 20, bold: true, color: WHITE, alignment: "center" });
slide4.speakerNotes.text = [
  "Timing: 1:15–2:10",
  "Consider a plan with four chunks of predicted lengths 20, 80, 50, and 30. In the original implementation, all four chunks form one sparse sequence on GPU zero. The number of diffusion steps is set by the longest chunk, so this example uses 80 steps, and every step processes the full stage tensor. In our four-GPU LPT execution, rank zero assigns the 80-token Methods chunk first, followed by Results, Conclusion, and Introduction on the other GPUs. Each worker receives the repeated prefix plus only its assigned chunk subset, while preserving global positions. Rank zero gathers the generated chunks in the original order. The semantic plan is identical; the tradeoff is a smaller sequence per rank versus broadcast, gather, and repeated-prefix overhead.",
  "Source: dream/generation_utils.py; semantic_parallel/",
].join("\n\n");

// Slide 5: results
const slide5 = addTitleOnlySlide("Performance depends on workload shape");
repairBrandHeader(slide5);
addText(slide5, "4-GPU geometric-mean speedup", { left: 52, top: 156, width: 560, height: 34 }, { fontSize: 22, bold: true, color: CHARCOAL });
const chartLeft = 82;
const chartTop = 205;
const chartWidth = 510;
const chartHeight = 220;
const maxSpeedup = 1.6;
for (const tick of [0, 0.4, 0.8, 1.2, 1.6]) {
  const y = chartTop + chartHeight - (tick / maxSpeedup) * chartHeight;
  addDivider(slide5, chartLeft, y, chartWidth, tick === 0 ? "#777777" : "#E0E0E0", tick === 0 ? 2 : 1);
  addText(slide5, tick.toFixed(1), { left: 46, top: y - 12, width: 30, height: 24 }, { fontSize: 15, color: MID, alignment: "right" });
}
const baselineY = chartTop + chartHeight - (1 / maxSpeedup) * chartHeight;
addDivider(slide5, chartLeft, baselineY, chartWidth, RED, 2);

const categories = ["Short", "Equal", "Uneven", "Multi-stage", "Long/high"];
const lptValues = [0.69, 1.06, 1.34, 1.22, 1.29];
const rrValues = [0.67, 1.04, 1.23, 1.21, 1.21];
const groupWidth = chartWidth / categories.length;
const barWidth = 25;
for (let i = 0; i < categories.length; i += 1) {
  const center = chartLeft + groupWidth * (i + 0.5);
  for (const [seriesIndex, value] of [[0, lptValues[i]], [1, rrValues[i]]]) {
    const height = (value / maxSpeedup) * chartHeight;
    const left = center + (seriesIndex === 0 ? -barWidth - 3 : 3);
    const top = chartTop + chartHeight - height;
    const color = seriesIndex === 0 ? RED : "#777777";
    addBox(slide5, { left, top, width: barWidth, height }, {
      geometry: "rect",
      fill: color,
      line: { style: "solid", fill: color, width: 0 },
    });
    addText(slide5, value.toFixed(2), { left: left - 8, top: top - 22, width: barWidth + 16, height: 20 }, { fontSize: 13, bold: true, color, alignment: "center" });
  }
  addText(slide5, categories[i], { left: center - 48, top: chartTop + chartHeight + 7, width: 96, height: 28 }, { fontSize: 15, color: CHARCOAL, alignment: "center" });
}
addText(slide5, "1.0× baseline", { left: chartLeft + chartWidth - 122, top: baselineY - 24, width: 122, height: 22 }, { fontSize: 14, bold: true, color: RED, alignment: "right", fill: WHITE });
addBox(slide5, { left: 205, top: 462, width: 14, height: 14 }, { geometry: "rect", fill: RED, line: { style: "solid", fill: RED, width: 0 } });
addText(slide5, "LPT", { left: 225, top: 456, width: 60, height: 26 }, { fontSize: 16, color: CHARCOAL });
addBox(slide5, { left: 319, top: 462, width: 14, height: 14 }, { geometry: "rect", fill: "#777777", line: { style: "solid", fill: "#777777", width: 0 } });
addText(slide5, "Round robin", { left: 339, top: 456, width: 120, height: 26 }, { fontSize: 16, color: CHARCOAL });

addDivider(slide5, 641, 167, 2, "#D0D0D0", 307);
addText(slide5, "1.66×", { left: 676, top: 176, width: 190, height: 55 }, { fontSize: 43, bold: true, color: RED, alignment: "center" });
addText(slide5, "best individual speedup", { left: 670, top: 229, width: 205, height: 34 }, { fontSize: 19, color: MID, alignment: "center" });
addText(slide5, "1.11×", { left: 676, top: 286, width: 190, height: 50 }, { fontSize: 37, bold: true, color: CHARCOAL, alignment: "center" });
addText(slide5, "overall 4-GPU LPT\npilot result", { left: 670, top: 334, width: 205, height: 54 }, { fontSize: 18, color: MID, alignment: "center", verticalAlignment: "top" });
addBox(slide5, { left: 669, top: 405, width: 210, height: 68 }, { fill: PALE_RED, line: { style: "solid", fill: "#E3A1A1", width: 1 } });
addText(slide5, "Correctness check", { left: 682, top: 411, width: 184, height: 24 }, { fontSize: 18, bold: true, color: DARK_RED, alignment: "center" });
addText(slide5, "7 of 10 exact · 33 / 2,620 tokens differed", { left: 682, top: 436, width: 184, height: 31 }, { fontSize: 15, color: CHARCOAL, alignment: "center" });
addText(slide5, "One-repeat pilot campaign on four NVIDIA RTX A6000 GPUs", { left: 52, top: 482, width: 560, height: 28 }, { fontSize: 16, color: MID });
slide5.speakerNotes.text = [
  "Timing: 2:10–3:20",
  "The result is strongly workload dependent. Short controls slow down to 0.69 times because communication and repeated prefix work dominate. Uneven workloads reach 1.34 times with four-GPU LPT, and long high-parallelism workloads reach 1.29 times. The best prompt reaches 1.66 times. Across all 18 prompts, four-GPU LPT reaches 1.11 times geometric-mean speedup. LPT shows its clearest advantage on uneven chunks. Correctness remains a caveat: seven of ten verification runs match exactly, while 33 of 2,620 compared tokens differ overall. These are one-repeat pilot measurements, so we do not claim statistical finality yet.",
  "Source: results/arc_20261115_133441_job341462/summary.csv; output/pdf/resource_aware_planned_diffusion_results_and_future_work.pdf",
].join("\n\n");

// Slide 6: conclusion
const slide6 = addTitleOnlySlide("Workload-aware GPU selection");
repairBrandHeader(slide6);
addText(slide6, "Recommended decision boundary", { left: 58, top: 157, width: 340, height: 35 }, { fontSize: 22, bold: true, color: CHARCOAL });
addDivider(slide6, 100, 270, 760, "#A9A9A9", 4);
const choices = [
  { x: 145, color: CHARCOAL, top: "1 GPU", bottom: "Short stages\nFew chunks" },
  { x: 445, color: "#777777", top: "2 GPUs", bottom: "Medium parallelism\nEnough work to amortize setup" },
  { x: 750, color: RED, top: "4 GPUs + LPT", bottom: "Long, chunk-rich stages\nUneven lengths" },
];
for (const choice of choices) {
  addBox(slide6, { left: choice.x - 15, top: 255, width: 34, height: 34 }, {
    geometry: "ellipse",
    fill: choice.color,
    line: { style: "solid", fill: WHITE, width: 2 },
  });
  addText(slide6, choice.top, { left: choice.x - 85, top: 205, width: 175, height: 40 }, { fontSize: 23, bold: true, color: choice.color, alignment: "center" });
  addText(slide6, choice.bottom, { left: choice.x - 112, top: 305, width: 230, height: 63 }, { fontSize: 19, color: MID, alignment: "center", verticalAlignment: "top" });
}
addText(slide6, "Lower parallelism", { left: 72, top: 377, width: 190, height: 28 }, { fontSize: 17, color: MID });
addText(slide6, "Higher parallelism and length variation", { left: 645, top: 377, width: 260, height: 28 }, { fontSize: 17, color: MID, alignment: "right" });
addDivider(slide6, 58, 416, 844, "#D2D2D2", 2);
addText(slide6, "Next experiment", { left: 58, top: 432, width: 170, height: 32 }, { fontSize: 20, bold: true, color: DARK_RED });
addText(slide6, "Three repetitions, adaptive 1/2/4-GPU selection, and per-rank utilization measurements", { left: 222, top: 426, width: 680, height: 45 }, { fontSize: 20, color: CHARCOAL });
addText(slide6, "Resource-aware scheduling helps when chunk work is large enough to amortize multi-GPU overhead.", { left: 58, top: 479, width: 844, height: 35 }, { fontSize: 22, bold: true, color: RED, alignment: "center" });
slide6.speakerNotes.text = [
  "Timing: 3:20–4:00",
  "The practical conclusion is to choose GPU count from the generated stage. Short or low-chunk stages should stay on one GPU. Medium stages may benefit from two. Four GPUs with LPT make sense when the stage contains many long or uneven chunks. Our next experiment will repeat the campaign three times, add adaptive GPU-count selection, and measure each rank's utilization. Resource-aware scheduling helps when the available chunk work is large enough to pay for the distributed overhead.",
  "Source: output/pdf/resource_aware_planned_diffusion_results_and_future_work.pdf",
].join("\n\n");

const { finalizePresentation } = await import(pathToFileURL(
  path.join(skillDir, "container_tools/artifact_tool_utils.mjs"),
).href);

// Render every slide and a montage before finalization.
for (const [index, slide] of presentation.slides.items.entries()) {
  const stem = `slide-${String(index + 1).padStart(2, "0")}`;
  const png = await presentation.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(path.join(previewDir, `${stem}.png`), new Uint8Array(await png.arrayBuffer()));
  const layout = await slide.export({ format: "layout", detail: "full" });
  await fs.writeFile(path.join(previewDir, `${stem}.layout.json`), await layout.text(), "utf8");
}
const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
await fs.writeFile(path.join(previewDir, "montage.webp"), new Uint8Array(await montage.arrayBuffer()));

const candidatePath = path.join(stagingDir, "candidate.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

const result = await finalizePresentation({
  explicitTotalSlideCount: 6,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
  sourceTemplatePath: inputPath,
  workspaceDir,
  candidatePath,
  finalPath,
  pythonExecutable,
  integrityValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "9144000,5143500",
    "--validate-bullet-geometry",
    "--validate-heading-fit",
  ],
  requiredNativeChartOwnerSlides: [],
  fontPolicy: {
    basis: "design",
    families: [FONT],
  },
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, "resource_aware_planned_diffusion_4min_ncstate_v6.validation.json"),
});

console.log(JSON.stringify({ finalPath, previewDir, result }, null, 2));
