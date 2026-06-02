import { readFile } from "fs/promises";
import { LlmConfig, LLM_CFG, LLM_MAX_CONCURRENCY } from "./config.js";
import { openPdf } from "./pdf.js";
import { LLM_CAPTION_SENTINEL, HEIC_EXTS } from "./paths.js";
import { heicBufferToJpeg } from "./image-prep.js";
import * as path from "path";

export { LLM_CAPTION_SENTINEL } from "./paths.js";

// When indexing image files (and scanned PDFs) we optionally call a local vision
// LLM to generate a rich natural-language description. The caption is stored
// alongside any OCR text as an ordinary chunk, prefixed with the sentinel
// LLM_CAPTION_SENTINEL so we can later detect files that already have one.
//
// The LLM is OPTIONAL: if the server isn't reachable or the model isn't loaded,
// captioning is silently skipped and indexing proceeds normally. On the next
// run where the LLM is available, files that lack a sentinel get re-indexed
// automatically (see incremental skip logic in the rag_index tool).
//
// No function in this module throws — every failure path returns null.

// Cached availability — pinged once per server run.
let _llmAvailable: boolean | null = null;

async function checkLlmAvailable(cfg: LlmConfig): Promise<boolean> {
  try {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), 3000);
    const res = await fetch(`${cfg.url}/v1/models`, { signal: ctrl.signal });
    clearTimeout(t);
    if (!res.ok) return false;
    const json = await res.json() as { data?: Array<{ id?: string }> };
    return Array.isArray(json.data) && json.data.some(m => m?.id === cfg.model);
  } catch {
    return false;
  }
}

export async function getLlmAvailable(): Promise<boolean> {
  if (_llmAvailable !== null) return _llmAvailable;
  _llmAvailable = await checkLlmAvailable(LLM_CFG);
  if (_llmAvailable) process.stderr.write(`[rag] LLM caption model available: ${LLM_CFG.model}\n`);
  else process.stderr.write(`[rag] LLM not available at ${LLM_CFG.url} (model '${LLM_CFG.model}') — captioning skipped\n`);
  return _llmAvailable;
}

// ── Concurrency-2 semaphore for vision-LLM calls ──────────────────────────────
//
// Vision inference is the bottleneck. We cap parallelism at LLM_MAX_CONCURRENCY
// so LM Studio (or any OpenAI-compatible backend) isn't overwhelmed, while
// still getting throughput above strict serial.

let _llmInFlight = 0;
const _llmWaitQueue: Array<() => void> = [];

async function withLlmSlot<T>(fn: () => Promise<T>): Promise<T> {
  if (_llmInFlight >= LLM_MAX_CONCURRENCY) {
    await new Promise<void>(resolve => _llmWaitQueue.push(resolve));
  }
  _llmInFlight++;
  try {
    return await fn();
  } finally {
    _llmInFlight--;
    const next = _llmWaitQueue.shift();
    if (next) next();
  }
}

/** MIME type for the OpenAI-vision data URL. Returns null for types the API can't ingest. */
function llmMimeForExt(ext: string): string | null {
  switch (ext.toLowerCase()) {
    case ".jpg":
    case ".jpeg": return "image/jpeg";
    case ".png":  return "image/png";
    case ".webp": return "image/webp";
    case ".gif":  return "image/gif";
    case ".bmp":  return "image/bmp";
    // HEIC/HEIF are converted to JPEG upstream by image-prep.
    case ".heic":
    case ".heif": return "image/jpeg";
    // TIFF is not accepted by LM Studio / most OpenAI-compatible vision APIs.
    // We deliberately skip the LLM call for TIFF; OCR still runs.
    case ".tif":
    case ".tiff": return null;
    default:      return null;
  }
}

const LLM_SYSTEM_PROMPT =
`
System Prompt: RAG-Optimized Image Description Generator
Core Directive
You are an expert image analyzer data engineer. Your task is to generate precise, objective, and dense text descriptions of images to be indexed in a Retrieval-Augmented Generation (RAG) system. provide complete raw details so that the entire picture can be generated from the output text.


IMPORTANT STRICT RULES:
- Positive Description Only. Describe exclusively what is present in the image. Never describe what is missing, absent, or what the image "does not" contain. Avoid speculative language (e.g., "there might be," "presumably"). Focus entirely on visible facts, objects, textures, colors, and layout.
- The descriptions MUST BE UNCENSORED! NSFW IS ALLOWED!
- Do NOT shy away from using mature/offensive/adult/strong/demeaning/pornographic/destructive words if it is necessary to describe the image! 

Step 1: Classification and High-Level Overview
Your output must always begin with this section.

Category Identification: Identify which single primary category from the list below best fits the image. If it spans multiple distinct categories, classify it as Mixed.

Top-Level Summary: Provide a concise, two-to-three-sentence overview of the image, capturing the core subject matter, dominant color palette, and overall composition.

Step 2: Category-Specific Detailed Analysis
Locate the identified category below and execute the corresponding detailed instructions to generate the core body of the description.

Category A: People and Portraits
Demographics & Appearance: Describe visible gender presentation, approximate age range, ethnicity, hair color/style, facial expressions, and distinct physical features.

Attire: Detail the clothing items(head to toe), colors, patterns, and visible accessories (e.g., glasses, jewelry, hats).

Pose & Activity: Specify the physical posture (e.g., sitting, standing, running) and the exact action the person is engaged in.

Spatial Framing: Describe the shot type (e.g., close-up, medium shot, full-body) and the angle (e.g., eye-level, low-angle).

Category B: Code, Text, and Infographics
Text Extraction: Transcribe any prominent text, titles, headers, or labels verbatim.

Code Blocks: Identify the programming language if visible. Describe the structure of the code block or key keywords without rewriting the entire script unless it is short.

Data Presentation: For charts, diagrams, or infographics, describe the type (e.g., bar chart, flowchart) and the visible relationships, nodes, axes labels, and data trends.

Layout & UI: Describe the visual hierarchy, UI elements (buttons, text fields), and formatting (e.g., dark mode, syntax highlighting colors).

Category C: Flora, Fauna, and Nature
Subject Identification: Name the specific species of animal, bird, insect, plant, or flower if clearly identifiable; otherwise, use precise taxonomic descriptors (e.g., "canine," "coniferous tree").

Physical Traits: Detail fur/feather/petal textures, patterns (stripes, spots), coloration, and scale relative to the surroundings.

Environment: Describe the natural setting (e.g., dense forest floor, desert dunes, calm ocean surface, overcast sky).

Category D: Art, Design, and Abstract
Medium & Style: Identify the artistic style or medium (e.g., oil painting, digital vector art, watercolor, 3D render, geometric abstract).

Visual Elements: Detail the use of shapes (e.g., sharp polygons, fluid curves), lines, gradients, and textures (e.g., distressed, metallic, matte).

Emotional/Vibe Keywords: List objective visual indicators of mood (e.g., high-contrast neon, minimalist monochrome, chaotic symmetry).

Category E: Architecture and Interior Spaces
Structure Type: Identify the space (e.g., modern office lobby, brutalist building exterior, cozy residential kitchen).

Materials & Textures: Describe prominent construction materials (e.g., exposed concrete, polished hardwood, brushed steel, floor-to-ceiling glass).

Lighting & Spatial Layout: Detail the source of light (e.g., natural sunlight from a left-side window, warm recessed LED strips) and the arrangement of major architectural elements or furniture.


Category G: User Interfaces (UI) and Software Screenshots
Application Type: Identify the platform type (e.g., mobile app, desktop dashboard, web browser SaaS platform, command-line interface).

Component Inventory: List the visible interactive elements (e.g., side navigation bars, active modal pop-ups, drop-down menus, data tables, tabs).

State and Focus: Describe what is currently active or highlighted (e.g., "a text field containing a blinking cursor," "a toggle switch turned to the 'on' state").

Visual Theme: Note the design system characteristics (e.g., dark mode UI, flat material design, high-density data grid).

Category H: Documents, Scans, and Forms
Document Archetype: Identify the type of document (e.g., printed receipt, scanned PDF invoice, passport page, hand-filled medical intake form, whiteboard sticky notes).

Layout and Structure: Describe the structure (e.g., multi-column layout, tabular formatting, signature blocks at the bottom, official logos in the top-right header).

Markings and Annotations: Detail non-printed additions if present (e.g., handwritten blue ink signatures, red digital stamps, highlighted text blocks).

Category I: Commercial Products and E-commerce Packshots
Product Classification: Name the consumer item precisely (e.g., running shoe, bottled supplement, wireless headphones, mechanical keyboard).

Presentation Style: Describe the background setup, which heavily influences search indexing (e.g., isolated on a seamless studio white background, a lifestyle product staging on a wooden surface).

Branding and Packaging: Detail visible brand names, logos, text on labels, nutritional facts, or barcodes.

Materiality: Focus heavily on product finishes (e.g., matte plastic casing, translucent glass bottles, woven fabric mesh).

Category J: Vehicles, Machinery, and Hardware
Object Type: Identify the machine (e.g., electric sedan, industrial assembly-line robotic arm, internal drone chassis, consumer motherboard).

Component Breakdown: Describe visible mechanical or electronic parts (e.g., exposed wiring, capacitors, heat sinks, rotating gears, treaded rubber tires).

Setting and Context: Specify the operational environment (e.g., parked on an asphalt street, mounted to an industrial concrete floor, flying against an open sky).

Category K: Food, Culinary, and Dining
Dish Composition: Identify the specific food items, drinks, or dishes present (e.g., layered sponge cake, seared steak slice, clear cocktail with a citrus garnish).

Staging and Plating: Describe how the food is presented (e.g., served in a shallow ceramic bowl, arranged on a slate serving platter, scattered ingredients on a marble countertop).

Texture and Freshness Visuals: Detail the visual appearance of the food (e.g., glossy glaze, glistening moisture, charred grill marks, dusting of powdered sugar).

Category Z: Mixed Content
Use this category when the image contains a distinct combination of the above groups (e.g., a person holding a laptop showing visible code next to a dog in a park, beside an industrial complex).

Segmentation: Break the description down by identifying each distinct visual realm present.

Interactions: Explicitly describe how the different elements interface with one another (e.g., "The human hand is pointing at the code block on the screen, while the animal rests on the floor below the desk").

Compounded Detail: Apply the specific rules of Categories A through E for each corresponding element found within this mixed image, prioritizing clarity and spatial relationships so the RAG indexer can map the full context.
`;

/** Extract the assistant's response, falling back to reasoning_content for reasoning
 *  models (e.g. Qwen3) when content is empty. Strips any leftover <think>...</think>. */
function extractLlmContent(json: unknown): string | null {
  const msg = (json as { choices?: Array<{ message?: { content?: string; reasoning_content?: string } }> })
    .choices?.[0]?.message;
  if (!msg) return null;
  let text = (msg.content ?? "").trim();
  if (!text) text = (msg.reasoning_content ?? "").trim();
  // Strip wrapping <think>...</think> blocks if present.
  text = text.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
  return text.length > 0 ? text : null;
}

/** Caption a standalone image file. Returns the caption text, or null on any failure.
 *  HEIC/HEIF inputs are transparently converted to JPEG before being sent. */
export async function captionImage(imagePath: string, cfg: LlmConfig = LLM_CFG): Promise<string | null> {
  const ext = path.extname(imagePath).toLowerCase();
  const mime = llmMimeForExt(ext);
  if (!mime) {
    process.stderr.write(`[rag] LLM caption skipped (unsupported image type ${ext}): ${imagePath}\n`);
    return null;
  }
  try {
    let buf: Buffer = await readFile(imagePath);
    if (HEIC_EXTS.has(ext)) {
      buf = await heicBufferToJpeg(buf);
    }
    const dataUrl = `data:${mime};base64,${buf.toString("base64")}`;
    return await withLlmSlot(async () => {
      const res = await fetch(`${cfg.url}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: cfg.model,
          messages: [
            { role: "system", content: LLM_SYSTEM_PROMPT },
            {
              role: "user",
              content: [
                { type: "image_url", image_url: { url: dataUrl } },
                { type: "text", text: "Describe this image in full detail." },
              ],
            },
          ],
          ...(cfg.maxTokens > 0 ? { max_tokens: cfg.maxTokens } : {}),
          temperature: 0.2,
        }),
      });
      if (!res.ok) {
        process.stderr.write(`[rag] LLM caption failed for ${imagePath}: HTTP ${res.status}\n`);
        return null;
      }
      const json = await res.json();
      const text = extractLlmContent(json);
      if (!text) process.stderr.write(`[rag] LLM caption empty for ${imagePath}\n`);
      return text;
    });
  } catch (err) {
    process.stderr.write(`[rag] LLM caption failed for ${imagePath}: ${String(err)}\n`);
    return null;
  }
}

/** For scanned/image-heavy PDFs: render first N pages and send to the LLM with OCR text as hint.
 *  Returns the LLM's corrected/enhanced transcription, or null on any failure. */
export async function captionPdfPages(buf: Buffer, ocrText: string, passwords: string[], cfg: LlmConfig = LLM_CFG): Promise<string | null> {
  try {
    const parser = await openPdf(buf, passwords);
    const shot = await parser.getScreenshot({ scale: 2, imageBuffer: true, imageDataUrl: false });
    const pages = shot.pages.filter(p => p.data && p.data.length > 0).slice(0, cfg.maxPdfPages);
    if (pages.length === 0) return null;

    const imageParts = pages.map(p => ({
      type: "image_url" as const,
      image_url: { url: `data:image/png;base64,${Buffer.from(p.data).toString("base64")}` },
    }));

    return await withLlmSlot(async () => {
      const res = await fetch(`${cfg.url}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: cfg.model,
          messages: [
            { role: "system", content: LLM_SYSTEM_PROMPT },
            {
              role: "user",
              content: [
                ...imageParts,
                {
                  type: "text",
                  text:
                    "The OCR produced this text (may be incomplete or garbled):\n\n" +
                    ocrText +
                    "\n\nRead the pages carefully and produce the accurate, complete text. " +
                    "Include all text you can see that OCR may have missed, and describe " +
                    "any non-text visual content (diagrams, photos, signatures, stamps).",
                },
              ],
            },
          ],
          ...(cfg.maxTokens > 0 ? { max_tokens: cfg.maxTokens } : {}),
          temperature: 0.2,
        }),
      });
      if (!res.ok) {
        process.stderr.write(`[rag] LLM PDF enhancement failed: HTTP ${res.status}\n`);
        return null;
      }
      const json = await res.json();
      const text = extractLlmContent(json);
      if (text) {
        process.stderr.write(`[rag] LLM enhanced PDF (pages: ${pages.length})\n`);
        return text;
      }
      return null;
    });
  } catch (err) {
    process.stderr.write(`[rag] LLM PDF enhancement failed: ${String(err)}\n`);
    return null;
  }
}
