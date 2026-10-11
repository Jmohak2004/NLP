// Builds Heatwave_NLP_Presentation.pptx on the Somaiya (KJSCE) slide template.
// Run:  node build_deck.js   (needs pptxgenjs, react, react-dom, react-icons, sharp)
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");

const APPLY_THEME = process.env.PPTX_APPLY_THEME; // path to the pptx skill's apply_theme.js
const DECK_DIR = process.env.DECK_DIR || __dirname; // folder holding assets/ and the output
const ASSETS = path.join(DECK_DIR, "assets");
const OUT = path.join(DECK_DIR, "Heatwave_NLP_Presentation.pptx");

// ---- Theme: Somaiya reds measured from the template --------------------------
const THEME = {
  name: "Somaiya Heatwave NLP",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "262626", lt1: "FFFFFF", dk2: "A83034", lt2: "F4F5F7",
    accent1: "A83034", // template band (dark red)
    accent2: "DA3832", // template bar (bright red)
    accent3: "2E5A7A", // learned models
    accent4: "C9851A", // rules & lexical knowledge
    accent5: "2E8B57", // responsible AI
    accent6: "6B7280", // data / muted text
    hlink: "2E5A7A", folHlink: "6B7280",
  },
};
// IMD warning colours are semantic, so they stay fixed hex values.
const SEV = { red: "C62828", orange: "E07B24", yellow: "E9B824", green: "2E8B57" };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333" x 7.5" (16:9), same ratio as the template
pres.title = "Heatwave NLP: Multilingual Early-Warning Advisories";
pres.subject = "NLP course · AI use case KJS-CES-01";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

// ---- Template frame (positions measured from the Somaiya template) -----------
const FRAME = [
  { rect: { x: 0, y: 7.106, w: 13.333, h: 0.394, fill: { color: C.accent1 }, line: { type: "none" } } },
  { rect: { x: 0, y: 6.937, w: 10.21, h: 0.183, fill: { color: C.accent2 }, line: { type: "none" } } },
  { image: { x: 0.296, y: 0.296, w: 3.092, h: 0.789, path: path.join(ASSETS, "somaiya_vidyavihar_logo.png"),
             altText: "Somaiya Vidyavihar University, K J Somaiya College of Engineering" } },
  { image: { x: 12.03, y: 0.232, w: 1.085, h: 0.817, path: path.join(ASSETS, "somaiya_trust_logo.png"),
             altText: "Somaiya Trust" } },
];
const SLIDE_NUMBER = { x: 12.13, y: 7.15, w: 0.7, h: 0.3, color: C.background1, fontFace: "Arial", fontSize: 12, align: "center" };

pres.defineSlideMaster({
  title: "SOMAIYA_TITLE",
  background: { color: C.background1 },
  objects: [
    ...FRAME,
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 2.1, w: 8.2, h: 1.95, fontSize: 36, bold: true,
                                color: C.text1, align: "left", valign: "top", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "subtitle", type: "body", x: 0.6, y: 4.2, w: 7.9, h: 0.95, fontSize: 18,
                                color: C.accent6, align: "left", valign: "top", margin: 0 }, text: "" } },
  ],
  slideNumber: SLIDE_NUMBER,
});
pres.defineSlideMaster({
  title: "SOMAIYA_CONTENT",
  background: { color: C.background1 },
  objects: [
    ...FRAME,
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 1.2, w: 11.3, h: 0.7, fontSize: 30, bold: true,
                                color: C.text2, align: "left", valign: "middle", margin: 0 }, text: "" } },
  ],
  slideNumber: SLIDE_NUMBER,
});

// ---- Helpers ------------------------------------------------------------------
async function iconPng(Icon, hex = "FFFFFF") {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Icon, { color: "#" + hex, size: 256 }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

async function iconCircle(slide, Icon, x, y, d, color, name) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color }, line: { type: "none" }, objectName: `${name} circle` });
  const pad = d * 0.25;
  slide.addImage({ data: await iconPng(Icon), x: x + pad, y: y + pad, w: d - 2 * pad, h: d - 2 * pad,
                   altText: name, objectName: `${name} icon` });
}

function text(slide, runs, opts) {
  slide.addText(runs, { isTextBox: true, margin: 0, color: C.text1, fontSize: 14, valign: "top", ...opts });
}

function arrow(slide, x, y, w, h, opts = {}) {
  slide.addShape(pres.shapes.LINE, { x, y, w, h, flipV: opts.flipV || false,
    line: { color: opts.color || C.accent6, width: 1.5, dashType: opts.dash || "solid",
            endArrowType: opts.noHead ? undefined : "triangle" }, objectName: opts.name || "connector" });
}

const KIND = {
  model: { color: C.accent3, label: "Learned model" },
  rules: { color: C.accent4, label: "Rules & lexical knowledge" },
  rai: { color: C.accent5, label: "Responsible AI / human review" },
  data: { color: C.accent6, label: "Data / output" },
};

function diagramBox(slide, b, x, y, w, h) {
  const k = KIND[b.kind];
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08,
    fill: { color: k.color, transparency: 88 }, line: { color: k.color, width: 1.25 }, objectName: `${b.title} box` });
  text(slide, [
    { text: b.title, options: { bold: true, fontSize: 12, breakLine: true } },
    { text: b.sub, options: { fontSize: 10.5, color: C.accent6 } },
  ], { x: x + 0.06, y, w: w - 0.12, h, align: "center", valign: "middle", objectName: `${b.title} label` });
}

// ---- Slide 1 · Title ----------------------------------------------------------
async function slideTitle() {
  pres.addSection({ title: "Introduction" });
  const s = pres.addSlide({ masterName: "SOMAIYA_TITLE", sectionTitle: "Introduction" });
  text(s, "NATURAL LANGUAGE PROCESSING  ·  AI USE CASE KJS-CES-01",
       { x: 0.6, y: 1.65, w: 8.2, h: 0.35, fontSize: 14, bold: true, color: C.accent2, charSpacing: 1.5, objectName: "kicker" });
  s.addText([
    { text: "Heatwave NLP", options: { color: C.accent1, breakLine: true } },
    { text: "Multilingual Advisories for Heatwave Early Warning" },
  ], { placeholder: "title" });
  s.addText("Turning IMD heatwave forecasts into fact-checked warnings for citizens, farmers, hospitals and authorities, in English, हिंदी and मराठी",
            { placeholder: "subtitle" });
  text(s, [
    { text: "Use case: ", options: { bold: true, color: C.text1 } },
    { text: "Climate Intelligence for Heatwave Monitoring, Prediction and Early Warning", options: { breakLine: true } },
    { text: "Collaborating organization: ", options: { bold: true, color: C.text1 } },
    { text: "India Meteorological Department (IMD), Mumbai–Pune" },
  ], { x: 0.6, y: 5.5, w: 8.2, h: 0.75, fontSize: 12, color: C.accent6, paraSpaceAfter: 4, objectName: "use case details" });

  // IMD severity ladder
  text(s, "IMD SEVERITY LEVELS USED BY THE SYSTEM", { x: 9.3, y: 1.6, w: 3.55, h: 0.3, fontSize: 10.5, bold: true,
       color: C.accent6, charSpacing: 1, objectName: "ladder heading" });
  const levels = [
    ["SEVERE  ·  Red", "Tmax ≥ 47 °C, or ≥ 6.5 °C above normal", SEV.red, C.background1],
    ["HEATWAVE  ·  Orange", "Tmax ≥ 45 °C, or 4.5–6.4 °C above normal", SEV.orange, C.background1],
    ["WATCH  ·  Yellow", "2.5–4.4 °C above normal (project band)", SEV.yellow, C.text1],
    ["NORMAL  ·  Green", "No heatwave conditions", SEV.green, C.background1],
  ];
  levels.forEach(([label, rule, fill, fg], i) => {
    const y = 2.0 + i * 1.03;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 9.3, y, w: 3.55, h: 0.85, rectRadius: 0.08, fill: { color: fill },
      line: { type: "none" }, objectName: `${label} level` });
    text(s, [
      { text: label, options: { bold: true, fontSize: 14, breakLine: true } },
      { text: rule, options: { fontSize: 10.5 } },
    ], { x: 9.45, y, w: 3.3, h: 0.85, color: fg, valign: "middle", objectName: `${label} text` });
  });
  s.addNotes(
    "Introduce the project: the NLP layer of the IMD heatwave early-warning use case (KJS-CES-01). " +
    "It turns forecasts into warnings people can act on, in English, Hindi and Marathi, and it reads citizen posts to check the forecast on the ground. " +
    "The ladder on the right shows the four severity levels. Heatwave and severe follow IMD's published criteria; watch is a pre-alert band we defined."
  );
}

// ---- Slide 2 · Problem and solution -------------------------------------------
async function slideProblem() {
  const s = pres.addSlide({ masterName: "SOMAIYA_CONTENT", sectionTitle: "Introduction" });
  s.addText("Why NLP for heatwave early warning", { placeholder: "title" });

  text(s, "The problem", { x: 0.5, y: 2.05, w: 5.6, h: 0.4, fontSize: 20, bold: true, fontFace: THEME.headFontFace, objectName: "problem header" });
  const problems = [
    [fa.FaTemperatureHigh, "Forecasts are technical", "“Tmax 46.1 °C, 3.6 °C above normal” means little to a farmer or a hospital."],
    [fa.FaLanguage, "Warnings must be multilingual", "People act on warnings in their own language: English, Hindi, Marathi."],
    [fa.FaHashtag, "Ground signals are noisy", "Citizen reports of heat stress are buried in slang, emojis and hashtags."],
  ];
  for (const [i, [Icon, head, body]] of problems.entries()) {
    const y = 2.65 + i * 1.3;
    await iconCircle(s, Icon, 0.5, y, 0.62, C.accent2, head);
    text(s, [
      { text: head, options: { bold: true, fontSize: 16, breakLine: true } },
      { text: body, options: { color: C.accent6 } },
    ], { x: 1.35, y: y - 0.02, w: 4.75, h: 1.05, paraSpaceAfter: 3, objectName: `${head} text` });
  }

  text(s, "What we built", { x: 6.7, y: 2.05, w: 6.1, h: 0.4, fontSize: 20, bold: true, fontFace: THEME.headFontFace, objectName: "solution header" });
  const cards = [
    [fa.FaBullhorn, "A · Advisory generation", "Phase V",
     "IMD bulletin → facts → IMD severity → advisories for 4 audiences in 3 languages, fact-checked and approved by an expert."],
    [fa.FaUsers, "B · Citizen heat-report mining", "supports Phase IV",
     "Noisy posts → heat or emergency label + place → flags where citizens and the forecast disagree, for weather-station (AWS) checks."],
  ];
  for (const [i, [Icon, head, tag, body]] of cards.entries()) {
    const y = 2.6 + i * 2.05;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.7, y, w: 6.13, h: 1.85, rectRadius: 0.1,
      fill: { color: C.background2 }, line: { type: "none" }, objectName: `${head} card` });
    await iconCircle(s, Icon, 6.95, y + 0.25, 0.62, C.accent1, head);
    text(s, [
      { text: head + "  ", options: { bold: true, fontSize: 16 } },
      { text: tag, options: { fontSize: 12, color: C.accent2, bold: true, breakLine: true } },
      { text: body, options: { color: C.accent6 } },
    ], { x: 7.8, y: y + 0.22, w: 4.8, h: 1.45, paraSpaceAfter: 4, objectName: `${head} text` });
  }
  s.addNotes(
    "Three reasons the use case needs NLP: forecasts are numbers, people need warnings in their own language, and citizen reports are noisy text. " +
    "We built two parts. A generates advisories from bulletins for citizens, farmers, health departments and authorities, in three languages. " +
    "B reads social-media posts, finds heat-stress reports and their places, and flags disagreements with the forecast so they can be checked against the weather stations."
  );
}

// ---- Slide 3 · Architecture ---------------------------------------------------
async function slideArchitecture() {
  pres.addSection({ title: "System" });
  const s = pres.addSlide({ masterName: "SOMAIYA_CONTENT", sectionTitle: "System" });
  s.addText("System architecture", { placeholder: "title" });

  const laneLabel = (t, y) => text(s, t, { x: 0.5, y, w: 6, h: 0.3, fontSize: 12, bold: true, color: C.accent1, charSpacing: 1.5, objectName: t });
  laneLabel("A · ADVISORY GENERATION", 2.0);
  const laneA = [
    { kind: "data", title: "IMD bulletin", sub: "press releases" },
    { kind: "model", title: "CRF NER", sub: "place · temp · date" },
    { kind: "rules", title: "IMD rule engine", sub: "severity + colour" },
    { kind: "model", title: "Seq2Seq + attention", sub: "BiLSTM → LSTM" },
    { kind: "rai", title: "Guardrails", sub: "10 automatic checks" },
    { kind: "rai", title: "Expert review", sub: "Orange/Red approval" },
    { kind: "data", title: "Advisories", sub: "4 audiences × 3 languages" },
  ];
  const aW = 1.55, aG = 0.25, aY = 2.35, aH = 0.95, x0 = 0.5;
  const ax = (i) => x0 + i * (aW + aG);
  laneA.forEach((b, i) => {
    diagramBox(s, b, ax(i), aY, aW, aH);
    if (i) arrow(s, ax(i) - aG, aY + aH / 2, aG, 0, { name: `A arrow ${i}` });
  });
  // templates: train the Seq2Seq, and are the guardrail fallback
  const tX = ax(3), tY = 3.48, tH = 0.6;
  diagramBox(s, { kind: "rules", title: "Templates", sub: "rule-based · en/hi/mr" }, tX, tY, aW, tH);
  arrow(s, tX + aW / 2, aY + aH, 0, tY - aY - aH, { flipV: true, name: "templates train seq2seq" });
  text(s, "trains", { x: tX + aW / 2 + 0.06, y: 3.3, w: 0.6, h: 0.2, fontSize: 10, italic: true, color: C.accent6, objectName: "trains label" });
  arrow(s, tX + aW, tY + tH / 2, ax(4) + aW / 2 - tX - aW, 0, { noHead: true, name: "fallback horizontal" });
  arrow(s, ax(4) + aW / 2, aY + aH, 0, tY + tH / 2 - aY - aH, { flipV: true, name: "fallback to guardrails" });
  text(s, "safe fallback", { x: ax(4) + aW / 2 + 0.06, y: 3.36, w: 1.0, h: 0.2, fontSize: 10, italic: true, color: C.accent6, objectName: "fallback label" });

  laneLabel("B · CITIZEN HEAT-REPORT MINING", 4.35);
  const laneB = [
    { kind: "data", title: "Citizen posts", sub: "English · Hinglish · Devanagari" },
    { kind: "rules", title: "Text normaliser", sub: "emoji · Hinglish · #hashtags" },
    { kind: "model", title: "BiLSTM classifier", sub: "heat / emergency / not heat" },
    { kind: "rules", title: "Place finder", sub: "gazetteer + morphology" },
    { kind: "rai", title: "Forecast vs citizens", sub: "corroborate or flag" },
    { kind: "data", title: "Validation flags", sub: "→ check AWS data" },
  ];
  const bW = 1.83, bG = (12.35 - 6 * 1.83) / 5, bY = 4.7, bH = 0.95;
  const bx = (i) => x0 + i * (bW + bG);
  laneB.forEach((b, i) => {
    diagramBox(s, b, bx(i), bY, bW, bH);
    if (i) arrow(s, bx(i) - bG, bY + bH / 2, bG, 0, { name: `B arrow ${i}` });
  });
  // forecast severity from the rule engine feeds the comparison
  const fy = 4.22, sx = ax(2) + aW / 2, ex = bx(4) + bW / 2;
  arrow(s, sx, aY + aH, 0, fy - aY - aH, { dash: "dash", noHead: true, color: C.accent4, name: "severity down" });
  arrow(s, sx, fy, ex - sx, 0, { dash: "dash", noHead: true, color: C.accent4, name: "severity across" });
  arrow(s, ex, fy, 0, bY - fy, { dash: "dash", color: C.accent4, name: "severity into comparison" });
  text(s, "forecast severity per place", { x: 7.65, y: 4.0, w: 2.0, h: 0.2, fontSize: 10, italic: true, color: C.accent4, objectName: "severity label" });

  // legend + design note
  Object.values(KIND).forEach((k, i) => {
    const x = 0.5 + i * 3.1;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 5.98, w: 0.28, h: 0.2, rectRadius: 0.04,
      fill: { color: k.color, transparency: 88 }, line: { color: k.color, width: 1.25 }, objectName: `${k.label} swatch` });
    text(s, k.label, { x: x + 0.38, y: 5.93, w: 2.6, h: 0.3, fontSize: 12, color: C.accent6, valign: "middle", objectName: `${k.label} legend` });
  });
  text(s, "Design choice: severity is decided by auditable IMD rules; the neural model only writes the words, and real values are filled in from the data.",
       { x: 0.5, y: 6.38, w: 12.33, h: 0.35, fontSize: 12, italic: true, color: C.accent6, objectName: "design note" });
  s.addNotes(
    "Lane A: a CRF extracts place, temperature and date from the IMD bulletin; the IMD rule engine decides the severity; a Seq2Seq model with attention writes the advisory; " +
    "guardrails run ten automatic checks and an expert approves orange and red warnings. The rule-based templates train the Seq2Seq and are the safe fallback when a check fails. " +
    "Lane B: citizen posts are normalised, classified by a BiLSTM and matched to places; the comparison with the forecast severity flags places to check with the weather stations."
  );
}

// ---- Slide 4 · End-to-end flow ------------------------------------------------
async function slideFlow() {
  const s = pres.addSlide({ masterName: "SOMAIYA_CONTENT", sectionTitle: "System" });
  s.addText("End-to-end flow: one bulletin line to twelve advisories", { placeholder: "title" });

  const steps = [
    ["1  Read bulletin", "“The highest maximum temperature of 47.6 °C was recorded at Chandrapur on 18th May.”", "Input: IMD bulletin text"],
    ["2  Extract facts", "LOC   Chandrapur\nTEMP  47.6 °C\nDATE  18 May", "CRF NER · Unit 2.3"],
    ["3  Apply IMD rules", "Plains station,\nTmax ≥ 47 °C\n→ SEVERE · Red alert", "Rule engine"],
    ["4  Write advisories", "“Severe heatwave warning for Chandrapur on 18 May… Call 108 for heat-stroke emergencies.”", "Seq2Seq · Unit 5.3"],
    ["5  Check & approve", "10 / 10 checks passed\nExpert approves the Red warning\n→ released", "Guardrails + human review"],
  ];
  const w = 2.3, g = 0.2;
  steps.forEach(([head, body, caption], i) => {
    const x = 0.5 + i * (w + g);
    s.addShape(i === 0 ? pres.shapes.PENTAGON : pres.shapes.CHEVRON, { x, y: 2.05, w: w + (i < 4 ? 0.12 : 0), h: 0.6,
      fill: { color: i === 4 ? C.accent5 : C.accent1 }, line: { type: "none" }, objectName: `${head} step` });
    text(s, head, { x: x + (i === 0 ? 0.15 : 0.3), y: 2.05, w: w - 0.45, h: 0.6, fontSize: 14, bold: true,
                    color: C.background1, valign: "middle", objectName: `${head} label` });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 2.85, w, h: 1.6, rectRadius: 0.08,
      fill: { color: C.background2 }, line: { type: "none" }, objectName: `${head} example card` });
    text(s, body, { x: x + 0.15, y: 2.95, w: w - 0.3, h: 1.4, fontSize: 12.5, valign: "middle",
                    italic: i === 0 || i === 3, objectName: `${head} example` });
    text(s, caption, { x, y: 4.52, w, h: 0.4, fontSize: 10.5, color: C.accent6, objectName: `${head} caption` });
  });

  // parallel citizen-report path
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 5.1, w: 12.33, h: 1.6, rectRadius: 0.1,
    fill: { color: C.accent3, transparency: 92 }, line: { type: "none" }, objectName: "citizen path panel" });
  text(s, "IN PARALLEL · CITIZEN REPORTS", { x: 0.75, y: 5.2, w: 6, h: 0.3, fontSize: 12, bold: true, color: C.accent3,
       charSpacing: 1.5, objectName: "citizen path label" });
  const pills = [
    ["“my grandmother fainted in the heat at chandrapur market”", 3.3, C.background1, C.text1],
    ["Normalise text", 1.6, C.accent4, C.background1],
    ["HEAT_EMERGENCY", 1.75, C.accent3, C.background1],
    ["Place: Chandrapur", 1.85, C.accent4, C.background1],
    ["Confirms forecast ✓", 2.0, C.accent5, C.background1],
  ];
  let px = 0.75;
  pills.forEach(([label, pw, fill, fg], i) => {
    if (i) { arrow(s, px, 6.07, 0.25, 0, { color: C.accent3, name: `citizen arrow ${i}` }); px += 0.3; }
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: px, y: 5.7, w: pw, h: 0.75, rectRadius: 0.12, fill: { color: fill },
      line: i === 0 ? { color: C.accent3, width: 1 } : { type: "none" }, objectName: `${label} pill` });
    text(s, label, { x: px + 0.1, y: 5.7, w: pw - 0.2, h: 0.75, fontSize: i === 0 ? 12 : 13, bold: i > 0,
                     italic: i === 0, color: fg, align: "center", valign: "middle", objectName: `${label} text` });
    px += pw;
  });
  s.addNotes(
    "Walk through one real example from the demo run. The CRF reads the bulletin sentence and extracts Chandrapur, 47.6 degrees and 18 May — Chandrapur was never seen in training. " +
    "The rule engine applies IMD's criterion: 47 degrees or more on the plains is a severe heatwave, so red. The Seq2Seq writes twelve advisories, four audiences in three languages. " +
    "All ten checks pass and, because it is a red warning, it waits for an expert's approval. In parallel, a citizen post about someone fainting in Chandrapur is classified as an emergency and corroborates the forecast."
  );
}

// ---- Slide 5 · How it works ---------------------------------------------------
async function slideMethods() {
  pres.addSection({ title: "Methods and results" });
  const s = pres.addSlide({ masterName: "SOMAIYA_CONTENT", sectionTitle: "Methods and results" });
  s.addText("How it works: NLP techniques behind each part", { placeholder: "title" });

  const cards = [
    [fa.FaHashtag, "MODULE 1 · UNITS 1.2–1.3", "Word boundary detection",
     "Splits hashtags by argmax over a unigram language model (Viterbi).", "#heatwavealertnagpur → heatwave · alert · nagpur"],
    [fa.FaFont, "MODULE 2 · UNITS 2.1–2.2", "Indian-language morphology",
     "Finite-state analyser strips Marathi/Hindi suffixes, restores stems.", "नागपूरमध्ये → नागपूर + मध्ये  (“in Nagpur”)"],
    [fa.FaTags, "MODULE 2 · UNIT 2.3", "Named entity recognition",
     "CRF with word-shape and context features tags places, dates, temperatures.", "Finds 90% of places never seen in training"],
    [fa.FaBroom, "MODULE 3 · UNIT 3.2", "Noisy web text",
     "Normalises emojis, elongated words, Hinglish and hashtags.", "“burninggg, bahut garmi” → “burning, very heat”"],
    [fa.FaBook, "MODULE 4 · UNITS 4.1–4.3", "WordNet and word senses",
     "Expands heat vocabulary, filters wrong senses; en→hi→mr dictionary.", "“Miami Heat won” is not about the weather"],
    [fa.FaBrain, "MODULE 5 · UNITS 5.1–5.3", "RNN, LSTM and Seq2Seq",
     "BiLSTM classifies posts; Seq2Seq with attention writes advisories.", "Beam search · teacher forcing · gradient clipping"],
  ];
  const w = 3.95, g = 0.24, h = 1.75;
  for (const [i, [Icon, tag, head, body, example]] of cards.entries()) {
    const x = 0.5 + (i % 3) * (w + g), y = 2.05 + Math.floor(i / 3) * (h + 0.2);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: C.background2 },
      line: { type: "none" }, objectName: `${head} card` });
    await iconCircle(s, Icon, x + 0.2, y + 0.22, 0.55, C.accent1, head);
    text(s, [
      { text: tag, options: { fontSize: 10.5, bold: true, color: C.accent2, charSpacing: 1, breakLine: true } },
      { text: head, options: { fontSize: 15, bold: true, breakLine: true } },
      { text: body, options: { fontSize: 13, color: C.text1, breakLine: true } },
      { text: example, options: { fontSize: 11, italic: true, color: C.accent6 } },
    ], { x: x + 0.95, y: y + 0.17, w: w - 1.1, h: h - 0.3, paraSpaceAfter: 3, objectName: `${head} text` });
  }
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 6.0, w: 12.33, h: 0.75, rectRadius: 0.1,
    fill: { color: C.accent5, transparency: 88 }, line: { type: "none" }, objectName: "responsible AI callout" });
  await iconCircle(s, fa.FaShieldAlt, 0.7, 6.1, 0.55, C.accent5, "Responsible AI");
  text(s, [
    { text: "Responsible AI:  ", options: { bold: true, color: C.accent5 } },
    { text: "IMD rules decide severity  ·  real values come from data  ·  10 automatic checks  ·  expert approves Orange/Red" },
  ], { x: 1.45, y: 6.0, w: 11.2, h: 0.75, valign: "middle", objectName: "responsible AI text" });
  s.addNotes(
    "Each part maps to a syllabus module. Module 1: hashtags are split into words by choosing the most likely split. Module 2: a small finite-state analyser handles Marathi and Hindi case endings, " +
    "and a CRF tags entities in bulletins. Module 3: noisy posts are normalised. Module 4: WordNet expands the heat vocabulary and shows why word sense matters. " +
    "Module 5: a BiLSTM classifies posts and a Seq2Seq with attention writes advisories. Across all of it, responsible-AI safeguards keep the warnings factual and human-approved."
  );
}

// ---- Slide 6 · Results and next steps -----------------------------------------
async function slideResults() {
  const s = pres.addSlide({ masterName: "SOMAIYA_CONTENT", sectionTitle: "Methods and results" });
  s.addText("Results and next steps", { placeholder: "title" });

  const stats = [
    ["0.94", "NER F1 on places, dates and temperatures, for unseen places"],
    ["90%", "citizen-post classification accuracy (keyword baseline: 78%)"],
    ["480/480", "generated advisories pass all 10 automatic checks"],
    ["2,366", "real IMD records (Apr–Jun 2024) processed end to end"],
  ];
  const w = 2.9, g = (12.33 - 4 * 2.9) / 3;
  stats.forEach(([num, label], i) => {
    const x = 0.5 + i * (w + g);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 2.05, w, h: 1.55, rectRadius: 0.1, fill: { color: C.background2 },
      line: { type: "none" }, objectName: `${num} tile` });
    text(s, num, { x: x + 0.2, y: 2.15, w: w - 0.4, h: 0.7, fontSize: 36, bold: true, color: C.accent1,
                   fontFace: THEME.headFontFace, valign: "middle", objectName: `${num} value` });
    text(s, label, { x: x + 0.2, y: 2.85, w: w - 0.4, h: 0.65, fontSize: 13, color: C.accent6, objectName: `${num} label` });
  });

  text(s, "Sample output · Chandrapur, 47.6 °C → Red", { x: 0.5, y: 3.85, w: 7.4, h: 0.35, fontSize: 16, bold: true,
       fontFace: THEME.headFontFace, objectName: "sample header" });
  const samples = [
    ["EN", "Severe heatwave warning for Chandrapur on 18 May: temperature may touch 47.6 °C, 4.6 °C above normal.", "en-IN"],
    ["HI", "चंद्रपुर में 18 मई को भीषण लू की चेतावनी, तापमान 47.6 °C तक पहुँच सकता है।", "hi-IN"],
    ["MR", "चंद्रपूर येथे 18 मे रोजी तीव्र उष्णतेच्या लाटेचा इशारा, तापमान 47.6 °C पर्यंत जाऊ शकते.", "mr-IN"],
  ];
  samples.forEach(([tag, line, lang], i) => {
    const y = 4.3 + i * 0.75;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: y + 0.08, w: 0.5, h: 0.36, rectRadius: 0.06,
      fill: { color: C.accent2 }, line: { type: "none" }, objectName: `${tag} tag` });
    text(s, tag, { x: 0.5, y: y + 0.08, w: 0.5, h: 0.36, fontSize: 11, bold: true, color: C.background1,
                   align: "center", valign: "middle", objectName: `${tag} tag text` });
    text(s, line, { x: 1.15, y: y + 0.1, w: 6.75, h: 0.6, fontSize: 13, lang, valign: "top", objectName: `${tag} sample` });
  });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.3, y: 3.85, w: 4.53, h: 2.45, rectRadius: 0.1,
    fill: { color: C.accent1, transparency: 92 }, line: { type: "none" }, objectName: "next steps panel" });
  text(s, "Next steps", { x: 8.55, y: 3.97, w: 4, h: 0.35, fontSize: 16, bold: true, fontFace: THEME.headFontFace,
       color: C.accent1, objectName: "next steps header" });
  text(s, [
    { text: "Train and test on real IMD bulletins and labelled posts", options: { bullet: true, breakLine: true } },
    { text: "LLM-assisted advisories, keeping the same checks", options: { bullet: true, breakLine: true } },
    { text: "Voice alerts for farmers", options: { bullet: true, breakLine: true } },
    { text: "Feed advisories into the heatwave dashboard", options: { bullet: true } },
  ], { x: 8.55, y: 4.4, w: 4.1, h: 1.8, paraSpaceAfter: 6, objectName: "next steps list" });
  text(s, "Scores are on synthetic test data; real-world scores will be lower.", { x: 8.3, y: 6.38, w: 4.53, h: 0.3,
       fontSize: 11, italic: true, color: C.accent6, objectName: "results caveat" });
  s.addNotes(
    "Results from one full run. The CRF reaches 0.94 F1 on places it never saw; the BiLSTM gets 90% on post phrasings it never saw, versus 78% for a WordNet keyword baseline; " +
    "every generated advisory passed all ten checks; and the pipeline also ran on 2,366 real IMD records from April to June 2024. " +
    "Be clear that these test sets are synthetic, so real-world scores will be lower. Next steps: real labelled data, LLM-assisted advisories with the same guardrails, voice alerts, and dashboard integration."
  );
}

(async () => {
  await slideTitle();
  await slideProblem();
  await slideArchitecture();
  await slideFlow();
  await slideMethods();
  await slideResults();
  await pres.writeFile({ fileName: OUT });
  if (APPLY_THEME) await require(APPLY_THEME).applyTheme(OUT, THEME);
  console.log("Wrote", OUT);
})();
