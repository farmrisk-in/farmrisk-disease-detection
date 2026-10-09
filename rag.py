"""
Knowledge-base retrieval + Gemini advisory for crop disease predictions.

Expects in KB_DIR:
    disease_kb.json   structured entries, one per disease/pest
    chunks.jsonl      text chunks, one JSON object per line
    coverage.csv      maps model class labels to KB entry names

Field names are auto-detected from common choices. If detection picks the
wrong field, edit the *_KEYS / *_COLS lists below.
"""
import csv
import difflib
import json
import os
import re
import time
from pathlib import Path

DEBUG = os.getenv("ADVISOR_DEBUG") == "1"

import logging
logging.getLogger("google_genai.models").setLevel(logging.ERROR)  # hide AFC notice

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ---- field-name candidates (checked in order, case-insensitive) -------------
TEXT_KEYS = ["text", "content", "chunk", "page_content", "body"]
NAME_KEYS = ["kb_key", "disease_id", "disease", "disease_name", "name", "title", "key", "label"]
SECTION_KEYS = ["section", "heading", "field", "type"]
SOURCE_KEYS = ["source_url", "url", "source", "link"]

CLASS_COLS = ["class", "model_class", "class_name", "model_label", "label"]
KBKEY_COLS = ["kb_key", "kb_disease", "kb_name", "kb_entry", "matched_disease",
              "tnau_name", "disease", "disease_name", "name"]
STATUS_COLS = ["status", "covered", "coverage", "in_kb", "has_kb"]
NOT_COVERED = {"no", "false", "0", "missing", "not covered", "uncovered", "none", "n"}

SECTIONS = ["Symptoms", "Pathogen", "Favourable Conditions",
            "Survival and Mode of Spread", "Management"]
SECTION_HINTS = {
    "Symptoms": ["symptom", "damage", "identification"],
    "Pathogen": ["pathogen", "causal", "organism", "pest", "scientific"],
    "Favourable Conditions": ["favourable", "favorable", "condition", "weather"],
    "Survival and Mode of Spread": ["survival", "spread", "transmission", "perpetuation"],
    "Management": ["management", "control", "ipm"],
}
QUERY_TERMS = ("symptoms pathogen causal organism favourable conditions weather "
               "survival mode of spread management control")

SYSTEM_PROMPT = """You write crop protection advisories for Indian farmers and extension workers.
Use ONLY the context provided. Do not add facts, chemicals, doses or names that are not in the context.
If the context has nothing for a section, write: "Not available in the knowledge base."
Write short, simple sentences. Use bullet points under each heading.
For insect or mite pests, use the Pathogen section for the pest's common and scientific name.
Output Markdown with exactly these headings, in this order:
### Symptoms
### Pathogen
### Favourable Conditions
### Survival and Mode of Spread
### Management
Do not add an introduction, a conclusion or any other headings."""


# ---- helpers ----------------------------------------------------------------
def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def pick(d, keys):
    low = {k.lower(): k for k in d}
    for k in keys:
        real = low.get(k)
        if real is not None and d[real] not in (None, "", [], {}):
            return d[real]
    return None


def flatten(rec):
    out = dict(rec)
    meta = rec.get("metadata")
    if isinstance(meta, dict):
        for k, v in meta.items():
            out.setdefault(k, v)
    return out


def as_text(v):
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return "\n".join(f"- {as_text(x)}" for x in v)
    if isinstance(v, dict):
        return "\n".join(f"{k}: {as_text(x)}" for k, x in v.items())
    return str(v)


# ---- knowledge base -----------------------------------------------------------
class KnowledgeBase:
    def __init__(self, kb_dir):
        kb_dir = Path(kb_dir)
        self.entries = self._load_entries(kb_dir / "disease_kb.json")
        self.chunks = self._load_chunks(kb_dir / "chunks.jsonl")
        self.coverage = self._load_coverage(kb_dir / "coverage.csv")
        self._norm_keys = {norm(k): k for k in self.entries}

        texts = [f"{c['disease']} {c['section']} {c['text']}" for c in self.chunks]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.mat = self.vec.fit_transform(texts) if texts else None
        print(f"[kb] {len(self.entries)} entries | {len(self.chunks)} chunks | "
              f"{len(self.coverage)} coverage rows")

    # -- loaders
    def _load_entries(self, path):
        if not path.exists():
            print(f"[kb] missing {path}")
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        desc = type(data).__name__
        if isinstance(data, dict):
            desc += f" with {len(data)} keys, first: {list(data)[:5]}"
        elif isinstance(data, list) and data:
            desc += f" of {len(data)} {type(data[0]).__name__}"
            if isinstance(data[0], dict):
                desc += f", fields: {list(data[0])[:10]}"
        print(f"[kb] disease_kb.json: {desc}")

        # Unwrap a container such as {"diseases": [...]} or {"entries": {...}}
        wrappers = {"diseases", "entries", "data", "items", "kb", "records", "pests", "classes"}
        if isinstance(data, dict):
            for k, v in data.items():
                if k.lower() in wrappers and isinstance(v, (dict, list)):
                    data = v
                    break

        entries = {}
        if isinstance(data, dict):
            entries = {str(k): v for k, v in data.items()}
        elif isinstance(data, list):
            for e in data:
                if isinstance(e, dict):
                    name = pick(e, NAME_KEYS) or f"entry_{len(entries)}"
                    entries[str(name)] = e
        # Index entries by any model-class label they carry
        self.entry_by_label = {}
        for k, v in entries.items():
            if isinstance(v, dict):
                lab = pick(v, ["label", "model_class", "class", "class_name"])
                if lab:
                    self.entry_by_label[norm(lab)] = k
        return entries

    def _load_chunks(self, path):
        if not path.exists():
            print(f"[kb] missing {path}")
            return []
        chunks, first_keys = [], None
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = flatten(json.loads(line))
                first_keys = first_keys or sorted(rec.keys())
                text = pick(rec, TEXT_KEYS)
                if not text:
                    continue
                chunks.append({
                    "label": str(pick(rec, ["label", "model_class", "class", "class_name"]) or ""),
                    "text": as_text(text),
                    "disease": str(pick(rec, NAME_KEYS) or ""),
                    "section": str(pick(rec, SECTION_KEYS) or ""),
                    "source": str(pick(rec, SOURCE_KEYS) or ""),
                })
        print(f"[kb] chunk fields found: {first_keys}")
        return chunks

    def _load_coverage(self, path):
        if not path.exists():
            print(f"[kb] missing {path}")
            return {}
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return {}
        cols = list(rows[0].keys())
        low = {c.lower().strip(): c for c in cols}
        class_col = next((low[c] for c in CLASS_COLS if c in low), cols[0])
        # Pick the column whose values best match entry names in disease_kb.json
        entry_norms = {norm(k) for k in self.entries}
        scores = {c: sum(norm(r.get(c) or "") in entry_norms for r in rows)
                  for c in cols if c != class_col}
        kb_col = max(scores, key=scores.get) if scores and max(scores.values()) > 0 else None
        if kb_col is None:
            kb_col = next((low[c] for c in KBKEY_COLS if c in low and low[c] != class_col), None)
        if kb_col is not None:
            print(f"[kb] coverage column match counts: {scores}")
        status_col = next((low[c] for c in STATUS_COLS if c in low), None)
        print(f"[kb] coverage.csv: class={class_col!r} kb_key={kb_col!r} status={status_col!r}")
        return {
            norm(r.get(class_col, "")): {
                "kb_key": (r.get(kb_col) or "").strip() if kb_col else "",
                "status": (r.get(status_col) or "").strip() if status_col else "",
            }
            for r in rows
        }

    # -- lookup
    def resolve(self, label):
        """Map a model class label to a KB entry name, or None."""
        row = self.coverage.get(norm(label))
        if row:
            if norm(row["status"]) in NOT_COVERED:
                return None
            key = row["kb_key"]
            if key:
                if key in self.entries:
                    return key
                return self._norm_keys.get(norm(key), key)
        nl = norm(label)
        if nl in self.entry_by_label:
            return self.entry_by_label[nl]
        if nl in self._norm_keys:
            return self._norm_keys[nl]
        if any(norm(c["label"]) == nl for c in self.chunks):
            return label  # chunks carry this class label directly
        match = difflib.get_close_matches(nl, list(self._norm_keys), n=1, cutoff=0.6)
        return self._norm_keys[match[0]] if match else None

    def entry_text(self, key, limit=8000):
        entry = self.entries.get(key) if key else None
        if entry is None:
            return ""
        return json.dumps(entry, ensure_ascii=False, indent=1)[:limit]

    def retrieve(self, key, label, k=8):
        if self.mat is None:
            return []
        nl, nk = norm(label), norm(key or "")
        idx = [i for i, c in enumerate(self.chunks) if norm(c["label"]) == nl]
        if not idx and nk:
            idx = [i for i, c in enumerate(self.chunks)
                   if (nd := norm(c["disease"])) and (nd == nk or nk in nd)]
        if not idx:
            return []  # no matching chunks: better none than another disease's text
        qv = self.vec.transform([f"{key or ''} {label} {QUERY_TERMS}"])
        sims = linear_kernel(qv, self.mat[idx]).ravel()
        order = sims.argsort()[::-1][:k]
        return [self.chunks[idx[i]] for i in order]


# ---- web fill (Google Search grounding) ------------------------------------------
WEB_PROMPT = """You are an agricultural extension expert for India.
Use Google Search to find information. Prefer sources in this order:
1. ICAR and its institutes (icar.org.in, ICAR-IIMR for maize, ICAR-CICR for cotton,
   ICAR-IIRR for rice, ICAR-IIWBR for wheat, ICAR-SBI and ICAR-IISR for sugarcane,
   ICAR-NCIPM, ICAR-NBAIR).
2. State agricultural universities (TNAU, PAU, others).
3. Government of India (PPQS, Department of Agriculture and Farmers Welfare).
Avoid blogs, forums and commercial product pages.
Write ONLY the sections requested, using exactly the headings given, formatted as '### Heading'.
Use short, simple sentences in bullet points.
Name a chemical only if a source you found recommends it. Give a dose only if a source gives it.
If you cannot find reliable information for a section, write: "Not available in the knowledge base."
Do not add an introduction, a conclusion or other headings."""

AI_PROMPT = """You are an agricultural extension expert for India.
Write from your own knowledge of the crop pest or disease named, as it occurs in India.
Write ONLY the sections requested, using exactly the headings given, formatted as '### Heading'.
Use short, simple sentences in bullet points.
Rules for accuracy:
- Include only what is well established for this pest or disease in India.
- For chemical control, name the active ingredient only. Do not give doses or spray rates.
  End the Management section with: "Follow label dose, or ask your KVK for the dose."
- Do not cite sources, name documents or write URLs.
- If you are not confident about a section, write: "Not available in the knowledge base."
Do not add an introduction, a conclusion or other headings."""

AI_NOTE = ("*🤖 Not in the knowledge base. AI-generated from general knowledge. "
           "Not verified against a source. Please check with your local KVK "
           "or agriculture officer before acting.*")

ICAR_HINTS = ["icar", "iimr", "cicr", "iirr", "iiwbr", "sbi.res", "iisr", "ncipm", "nbair", "nrcpb"]
MISSING_TEXT = "not available in the knowledge base"
WEB_NOTE = ("*🌐 Not in the knowledge base. Filled from web sources listed below. "
            "Please verify with your local KVK or agriculture officer.*")


PATHOGEN_ALIASES = {"causal organism", "causal agent", "pest", "pathogen pest", "pest pathogen",
                    "causal organism pest", "pathogen causal organism"}


def match_heading(line):
    """Return (section, rest_of_line) if this line is a section heading, else None.
    Accepts '### Symptoms', '**Symptoms**', '#### 1. Symptoms', 'Symptoms:' and
    'Symptoms: text on the same line'."""
    raw = line.strip()
    if not raw:
        return None
    if raw[0] in "-•" or (raw.startswith("* ") and not raw.startswith("**")):
        return None  # bullet point, not a heading
    heading_like = raw.startswith(("#", "**", "__")) or re.match(r"^\d+[.)]\s", raw)
    head, sep, rest = raw.partition(":")
    if not heading_like and not sep:
        return None
    h = re.sub(r"^\d+\s*", "", norm(head.lstrip("#")))
    if not h or len(h) > 45:
        return None
    for sec in SECTIONS:
        ns = norm(sec)
        if h == ns or h.startswith(ns[:10]) or (sec == "Pathogen" and h in PATHOGEN_ALIASES):
            return sec, rest.strip().strip("*_").strip()
    return None


def split_sections(md):
    """Split advisory markdown into {canonical section: body}."""
    out, current, buf = {}, None, []
    for line in (md or "").splitlines():
        m = match_heading(line)
        if m:
            if current:
                out[current] = "\n".join(buf).strip()
            current, first = m
            buf = [first] if first else []
        elif current:
            buf.append(line)
    if current:
        out[current] = "\n".join(buf).strip()
    return out


def is_missing(body):
    return (not body) or (MISSING_TEXT in norm(body) and len(body) < 120)


def is_icar(title, uri):
    s = f"{title} {uri}".lower()
    return any(h in s for h in ICAR_HINTS)


def add_citations(text, supports):
    """Insert [n] markers after grounded text segments (offsets are UTF-8 bytes)."""
    try:
        b = text.encode("utf-8")
        items = [s for s in (supports or []) if s.segment and s.grounding_chunk_indices]
        items.sort(key=lambda s: s.segment.end_index or 0, reverse=True)
        for s in items:
            end = s.segment.end_index or 0
            if 0 < end <= len(b):
                marks = "".join(f"[{i + 1}]" for i in sorted(set(s.grounding_chunk_indices)))
                b = b[:end] + f" {marks}".encode() + b[end:]
        return b.decode("utf-8", errors="ignore")
    except Exception:
        return text


# ---- advisory -------------------------------------------------------------------
class Advisor:
    def __init__(self, kb, model_name, fill_mode="ai"):
        """fill_mode: 'off' | 'ai' | 'search' | 'auto' (search, then AI on failure)."""
        self.kb = kb
        self.model_name = model_name
        self.fill_mode = fill_mode
        print(f"[advisor] fill mode: {fill_mode}")
        self.web_paused_until = 0.0
        self.cache = {}
        self.client = None
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key:
            try:
                from google import genai
                from google.genai import types
                self.client = genai.Client(api_key=api_key)
                self.types = types
            except ImportError:
                print("[advisor] google-genai not installed; using KB fallback")
        else:
            print("[advisor] GEMINI_API_KEY not set; using KB fallback")

    def advise(self, label):
        """Return (advisory_markdown, context_markdown, search_suggestions_html)."""
        if label in self.cache:
            return self.cache[label]

        kb_key = self.kb.resolve(label)
        chunks, entry = [], ""
        entry_key = None
        if kb_key is not None:
            entry_key = next((k for k in (kb_key, label) if k in self.kb.entries), None)
            entry = self.kb.entry_text(entry_key)
            chunks = self.kb.retrieve(kb_key, label)

        print(f"[advisor] {label!r} -> KB entry {kb_key!r} | entry {len(entry)} chars "
              f"| {len(chunks)} chunks retrieved")

        # Step 1: KB-grounded sections
        if kb_key is None:
            sections = {s: "" for s in SECTIONS}
        elif self.client is None:
            sections = split_sections(self._fallback(entry_key))
        else:
            try:
                raw = self._call(self._generate, label, kb_key, entry, chunks)
                sections = split_sections(raw)
                if DEBUG:
                    print("[debug] ---- raw KB-step output ----\n" + raw + "\n[debug] ----")
                if raw and not sections:
                    print("[advisor] WARNING: KB-step output has no recognisable headings; "
                          "set ADVISOR_DEBUG=1 to see it")
            except Exception as e:
                print(f"[advisor] KB generation failed: {type(e).__name__}: {e}")
                sections = split_sections(self._fallback(entry_key))
        if DEBUG:
            for sec in SECTIONS:
                state = "MISSING" if is_missing(sections.get(sec)) else "from KB"
                print(f"[debug] {sec:30s} {state}")

        # Step 2: fill missing sections (search and/or AI knowledge)
        missing = [s for s in SECTIONS if is_missing(sections.get(s))]
        fill_secs, web_srcs, search_html, fill_error, fill_kind = {}, [], "", None, None
        if missing and self.client is not None and self.fill_mode != "off":
            if self.fill_mode in ("search", "auto"):
                if time.time() < self.web_paused_until:
                    fill_error = "web search paused after a quota error"
                else:
                    try:
                        fill_secs, web_srcs, search_html = self._web_fill_with_retry(
                            label, kb_key, missing)
                        fill_kind = "web" if fill_secs else None
                    except Exception as e:
                        fill_error = self._short_error(e)
                        print(f"[advisor] web fill failed: {type(e).__name__}: {e}")
                        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                            self.web_paused_until = time.time() + 600
                            print("[advisor] web search paused for 10 minutes")
            if self.fill_mode == "ai" or (self.fill_mode == "auto" and not fill_kind):
                try:
                    fill_secs = self._call(self._ai_fill, label, kb_key, missing)
                    fill_kind, fill_error = "ai", None
                except Exception as e:
                    fill_error = self._short_error(e)
                    print(f"[advisor] AI fill failed: {type(e).__name__}: {e}")

        # Assemble
        note = WEB_NOTE if fill_kind == "web" else AI_NOTE
        parts = [f"## {label}"]
        if kb_key:
            parts.append(f"*Knowledge-base entry: {kb_key}*")
        elif fill_kind == "web":
            parts.append("*No knowledge-base entry for this class. "
                         "All sections below come from web search.*")
        elif fill_kind == "ai":
            parts.append("*No knowledge-base entry for this class. "
                         "All sections below are AI-generated and not source-verified.*")
        else:
            parts.append("*No knowledge-base entry for this class yet.*")
        for sec in SECTIONS:
            body = sections.get(sec, "")
            if sec in missing and not is_missing(fill_secs.get(sec)):
                body = f"{note}\n\n{fill_secs[sec]}"
            elif is_missing(body):
                body = "Not available in the knowledge base."
            parts.append(f"### {sec}\n{body}")
        if fill_error:
            parts.append(f"*Could not fill missing sections ({fill_error}). "
                         "They show knowledge-base content only.*")
        parts.append(self._sources(entry_key, chunks, web_srcs))

        md = "\n\n".join(p for p in parts if p)
        ctx_md = self._context_markdown(kb_key, chunks) if kb_key else ""
        result = (md, ctx_md, search_html)
        if not fill_error:  # don't cache failures, so the next request retries
            self.cache[label] = result
        return result

    def _call(self, fn, *args, waits=(3, 8, 20)):
        """Call fn, retrying transient Gemini errors (429 per-minute, 500, 503)."""
        for attempt in range(len(waits) + 1):
            try:
                return fn(*args)
            except Exception as e:
                msg = str(e)
                transient = any(t in msg for t in ("429", "RESOURCE_EXHAUSTED", "500", "503",
                                                    "UNAVAILABLE", "INTERNAL", "overloaded"))
                if not transient or attempt == len(waits):
                    raise
                print(f"[advisor] {msg[:40]}... retrying in {waits[attempt]}s")
                time.sleep(waits[attempt])

    def _web_fill_with_retry(self, label, kb_key, missing):
        return self._call(self._web_fill, label, kb_key, missing, waits=(5, 20))

    @staticmethod
    def _short_error(e):
        s = str(e)
        if "429" in s or "RESOURCE_EXHAUSTED" in s:
            return "quota reached"
        if "503" in s or "UNAVAILABLE" in s:
            return "Gemini is busy, please try again in a minute"
        return type(e).__name__

    def _generate(self, label, kb_key, entry, chunks):
        passages = "\n\n".join(
            f"[{i + 1}] ({c['section'] or 'general'}) {c['text']}" for i, c in enumerate(chunks))
        prompt = (f"Detected class: {label}\nKnowledge-base entry: {kb_key}\n\n"
                  f"STRUCTURED ENTRY:\n{entry or '(none)'}\n\n"
                  f"RETRIEVED PASSAGES:\n{passages or '(none)'}")
        resp = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self.types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT, temperature=0.2, max_output_tokens=1500),
        )
        return (resp.text or "").strip()

    def _web_fill(self, label, kb_key, missing):
        t = self.types
        heads = "\n".join(f"### {s}" for s in missing)
        name = label if not kb_key or norm(kb_key) == norm(label) else f"{label} ({kb_key})"
        prompt = (f"Crop pest or disease: {name}, as it occurs in India.\n"
                  f"Search ICAR sources first.\n"
                  f"Write only these sections:\n{heads}")
        resp = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=t.GenerateContentConfig(
                system_instruction=WEB_PROMPT,
                temperature=0.3,
                max_output_tokens=1500,
                tools=[t.Tool(google_search=t.GoogleSearch())],
            ),
        )
        cand = resp.candidates[0] if resp.candidates else None
        gm = getattr(cand, "grounding_metadata", None)
        gchunks = (gm.grounding_chunks or []) if gm else []
        if not gchunks:
            print("[advisor] web fill returned no grounded sources; discarded")
            return {}, [], ""
        text = add_citations(resp.text or "", gm.grounding_supports)
        sources = [(c.web.title or c.web.uri, c.web.uri) for c in gchunks if c.web]
        html = ""
        if gm.search_entry_point and gm.search_entry_point.rendered_content:
            html = gm.search_entry_point.rendered_content
        return split_sections(text), sources, html

    def _ai_fill(self, label, kb_key, missing):
        heads = "\n".join(f"### {s}" for s in missing)
        name = label if not kb_key or norm(kb_key) == norm(label) else f"{label} ({kb_key})"
        prompt = (f"Crop pest or disease: {name}, as it occurs in India.\n"
                  f"Write only these sections:\n{heads}")
        resp = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self.types.GenerateContentConfig(
                system_instruction=AI_PROMPT, temperature=0.2, max_output_tokens=1500),
        )
        text = resp.text or ""
        # Safety net: drop any URLs or bracketed citations the model adds anyway
        text = re.sub(r"https?://[^\s)\]]+", "", text)
        text = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", text)
        text = re.sub(r"\(\s*(?:see\s*)?\)", "", text, flags=re.I)      # empty "(see )"
        text = re.sub(r"\s*\b[Ss]ee\s*[:.]?\s*$", "", text, flags=re.M)  # dangling "See"
        text = re.sub(r"[ \t]+([.,;:])", r"\1", text)                    # "weather ." -> "weather."
        return split_sections(text)

    def _fallback(self, kb_key):
        """No LLM: show matching fields of the structured entry under each heading."""
        entry = self.kb.entries.get(kb_key) or {}
        if not isinstance(entry, dict):
            entry = {}
        out = []
        for sec in SECTIONS:
            vals = [as_text(v) for k, v in entry.items()
                    if any(h in norm(k) for h in SECTION_HINTS[sec])]
            out.append(f"### {sec}\n" + ("\n".join(vals) if vals else
                                         "Not available in the knowledge base."))
        return "\n\n".join(out)

    def _sources(self, kb_key, chunks, web_srcs):
        out = []
        entry = (self.kb.entries.get(kb_key) if kb_key else None) or {}
        if not isinstance(entry, dict):
            entry = {}
        kb_srcs = [str(pick(entry, SOURCE_KEYS) or "")] + [c["source"] for c in chunks]
        kb_srcs = list(dict.fromkeys(s for s in kb_srcs if s))
        if kb_srcs:
            out.append("**Knowledge base:** " + " · ".join(kb_srcs))
        if web_srcs:
            lines = [f"{i}. [{title}]({uri})" + (" — **ICAR**" if is_icar(title, uri) else "")
                     for i, (title, uri) in enumerate(web_srcs, 1)]
            out.append("**Web sources:**\n" + "\n".join(lines))
        return ("---\n" + "\n\n".join(out)) if out else ""

    @staticmethod
    def _context_markdown(kb_key, chunks):
        parts = [f"**KB entry:** `{kb_key}`"]
        for i, c in enumerate(chunks, 1):
            parts.append(f"**[{i}]** {c['disease']} · {c['section'] or 'general'}\n\n"
                         f"{c['text'][:600]}")
        return "\n\n".join(parts)
