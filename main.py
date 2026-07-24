"""
Structured extraction with the Claude API.

Flow for each file in input/:
  1. Load the file (PDF or text/HTML)
  2. Ask Claude to call our extract_structured_data tool
  3. Check the tool output looks valid (fill missing lists, one repair retry)
  4. Save JSON to output/
"""

import os
import sys
import json
import base64
from datetime import datetime, timezone
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

MODEL = "claude-sonnet-5"
MAX_TOKENS = 8192  # higher so long PDFs can finish relationships too
INPUT_DIR = "input"
OUTPUT_DIR = "output"

EXTRACTION_TOOL = {
    "name": "extract_structured_data",
    "description": "Extract summary, entities, and relationships from the document.",
    "input_schema": {
        "type": "object",
        "required": ["summary", "entities", "relationships"],
        "properties": {
            "summary": {"type": "string"},
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name", "type"],
                    "properties": {
                        "name": {"type": "string"},
                        "type": {
                            "type": "string",
                            "enum": ["PERSON", "ORGANIZATION", "LOCATION", "PRODUCT", "EVENT", "OTHER"],
                        },
                        "mentions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["text"],
                                "properties": {
                                    "text": {"type": "string"},
                                    "page": {"type": ["integer", "null"]},
                                },
                            },
                        },
                    },
                },
            },
            "relationships": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["subject", "relation", "object"],
                    "properties": {
                        "subject": {"type": "string"},
                        "relation": {"type": "string"},
                        "object": {"type": "string"},
                        "context": {"type": ["string", "null"]},
                        "page": {"type": ["integer", "null"]},
                    },
                },
            },
        },
    },
}

PROMPT = (
    "Read the document and call extract_structured_data exactly once. Determine entities first, "
    "then relationships that only reference those entities — in that order.\n"
    "\n"
    "- summary: 1-2 sentences, neutral and factual, stating what the document is about.\n"
    "\n"
    "- entities: at most 20 total. If more than 20 are central to the document, keep only the 20 "
    "most important and drop the rest — do not exceed 20 under any circumstance. The people, "
    "organizations, locations, products, and events central to the document's content — not "
    "routine bibliography citations. Always include every author named on the byline/title page "
    "as a PERSON entity. type = PERSON, ORGANIZATION, LOCATION, PRODUCT, EVENT, or OTHER. Each "
    "distinct real-world entity appears exactly once: pick one canonical `name` — its fullest/most "
    "formal form, in whichever language the document is predominantly written in (fall back to "
    "English only if the document mixes languages evenly enough that the main language is unclear) "
    "— and put every other surface form it's called by, including the same entity's name in a "
    "different language, in `mentions` rather than creating a second entity for it. Each mention "
    "is an object `{text, page}`: `text` is the surface form as it appears in the source, and "
    "`page` is the 1-indexed page number within the PDF file where that surface form appears "
    "(count pages from the start of the file itself, not any printed page number in a "
    "header/footer) — use null if the source has no pages (HTML/plain text) or the page can't be "
    "determined. If a surface form recurs on more than one page, list it once per page it occurs "
    "on. List entities in order of first appearance.\n"
    "\n"
    "- relationships: subject / relation / object triples (always include this field; use [] if "
    "none). Only extract a relationship if it clearly falls into one of these categories, which "
    "cover the kinds of connections that recur across historiographical scholarship generally (source "
    "criticism, biography, archaeology, institutional/political history) — if a connection doesn't "
    "fit any of them, leave it out rather than forcing it in:\n"
    "  1. creation/authorship — wrote, composed, commissioned, minted, built, translated, edited\n"
    "  2. affiliation — institutional, professional, or organizational membership\n"
    "  3. provenance — discovered, found, or excavated at a place\n"
    "  4. custody — currently held, owned, acquired, or exhibited by an institution\n"
    "  5. depiction — depicts, portrays, symbolizes, or represents\n"
    "  6. scholarly stance — interprets, critiques, supports, corrects, or builds on another's claim, "
    "reading, or source\n"
    "  7. identification — identified as, equated with, or the same entity as\n"
    "  8. historical role — ruled, succeeded, held office, participated in, allied with, or opposed "
    "(people, offices, events, polities)\n"
    "  9. comparison — similar to, compared with, or contrasted with\n"
    "  10. location — located in/near, moved to, part of\n"
    "`subject` and `object` must be copied verbatim from an entity's canonical `name` above — never a "
    "mention/alias or a shortened form, even if that's how the source text refers to it at that "
    "point. Ground every relationship in the specific entity the sentence is actually about — never "
    "substitute a related-but-different entity for it. A location IS a legitimate subject/object "
    "outside categories 3/4/10 when the relationship is genuinely about the place itself — e.g. "
    "identifying an ancient toponym with a modern place, or a polity ruling/acquiring territory, are "
    "real historiographical claims and belong in the output like any other. What's never legitimate "
    "is silently substituting a findspot or holding institution for the actual artifact/document being "
    "discussed: a LOCATION must never be the subject/object of \"published\" or \"depicted on\" — "
    "nothing is ever literally published on, or depicted on, a bare place, so seeing either of those "
    "with a location means you've substituted the findspot for the artifact/document/person that was "
    "actually published or depicted; go back and use that entity instead. Worked example: a seal "
    "depicting a person was found in a village and is now held by a museum. Correct: \"person — "
    "depicted on → seal\" (5), \"seal — found in → village\" (3), \"seal — held by → museum\" (4). "
    "Wrong: \"person — depicted on → village\" — the village never depicted anyone; the seal did. "
    "`relation` is always in English, regardless of what language the document itself is written "
    "in (unlike entity `name`/`mentions`, which follow the document's language) — a short (2-4 "
    "word) lowercase, active-voice verb phrase naming which category applies (e.g. \"published\", "
    "\"critiques\", \"located in\") — phrase the same kind of connection the same way every time "
    "it recurs, in English, even across different documents. Never emit two relationships with the "
    "same subject, relation, and object. Also include `context`: a one-sentence (<=25 words) "
    "English paraphrase of the specific textual evidence for this relationship — always in "
    "English, even when the source document is written in another language; translate the "
    "substance rather than quoting the original-language sentence verbatim (a short original-"
    "language term or title inside the English paraphrase is fine when useful), or null if "
    "nothing concise fits. Also include `page`: the 1-indexed PDF page number (counted "
    "the same way as for mentions above) where this relationship is stated — use the page with "
    "the clearest evidence if the statement spans a page break, or null if the source has no "
    "pages or the page can't be determined.\n"
    "\n"
    "Always include all three top-level fields, using [] for any list with nothing found. Use only "
    "what is stated or directly implied by the text — do not invent facts."
)

REPAIR_PROMPT = (
    "Your previous tool call was incomplete or invalid: {reason}.\n"
    "Call extract_structured_data again now. You MUST include summary, entities, "
    "and relationships (use [] for any list with no items). Keep entities to the "
    "most important ones (max 20)."
)


def load_file(path):
    """
    Turn a local file into a Claude content block.
    Returns (content_block, format_label) or (None, None) on failure.
    """
    ext = os.path.splitext(path)[1].lower()

    try:
        if ext == ".pdf":
            # PDFs go in a "document" block (base64). Claude reads them natively.
            with open(path, "rb") as f:
                data = base64.standard_b64encode(f.read()).decode("utf-8")
            block = {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": data,
                },
            }
            return block, "pdf"

        # .html, .txt, .md, etc. — send as plain text
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        if not text.strip():
            print("   skip: empty file")
            return None, None

        fmt = "html" if ext in (".html", ".htm") else "text"
        return {"type": "text", "text": text}, fmt

    except Exception as e:
        print(f"   skip: could not read file ({e})")
        return None, None


def chat(client, content_block, extra_note=None):
    prompt = PROMPT if not extra_note else f"{extra_note}\n\n{PROMPT}"
    # The document itself (often a multi-page PDF) is identical between the initial
    # call and the repair retry — cache it so the retry doesn't reprocess it from
    # scratch. content_block is copied rather than mutated since callers (e.g. eval.py)
    # reuse the same dict across many chat() calls for the same file.
    cached_block = {**content_block, "cache_control": {"type": "ephemeral"}}
    return client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        tools=[EXTRACTION_TOOL],
        tool_choice={"type": "tool", "name": "extract_structured_data"},
        messages=[
            {
                "role": "user",
                "content": [
                    cached_block,
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )


def get_tool_data(response):
    """
    Find the tool_use block and return its input (the structured JSON).
    Returns None if Claude did not call the tool.
    """
    for block in response.content:
        if block.type == "tool_use" and block.name == "extract_structured_data":
            return block.input  # this is the structured JSON
    return None


def normalize_mentions(data):
    """
    Coerce each entity's `mentions` into {text, page} objects. Older prompt
    revisions (and occasional off-schema model output) return plain strings —
    accepted here as {text: <string>, page: None} rather than rejected.
    Returns True if anything was coerced.
    """
    coerced = False
    for entity in data.get("entities", []) or []:
        if not isinstance(entity, dict):
            continue
        mentions = entity.get("mentions")
        if not isinstance(mentions, list):
            continue
        normalized = []
        for m in mentions:
            if isinstance(m, str):
                normalized.append({"text": m, "page": None})
                coerced = True
            elif isinstance(m, dict) and isinstance(m.get("text"), str):
                normalized.append({"text": m["text"], "page": m.get("page")})
            # anything else (wrong shape entirely) is dropped rather than guessed at
        entity["mentions"] = normalized
    return coerced


def normalize_lists(data):
    """
    If Claude forgot a list field (common when output is truncated),
    fill it with [] and record a warning. Returns (data, warnings).
    """
    warnings = []
    if not isinstance(data, dict):
        return data, warnings

    for key in ("entities", "relationships"):
        if key not in data or data[key] is None:
            data[key] = []
            warnings.append(f"missing `{key}` filled with empty list")
        elif not isinstance(data[key], list):
            pass

    if normalize_mentions(data):
        warnings.append("some `mentions` entries were plain strings, coerced to {text, page: null}")

    return data, warnings


def validation_errors(data):
    """Return a list of human-readable problems, or [] if data looks valid."""
    if data is None:
        return ["Claude did not call the extract_structured_data tool"]
    if not isinstance(data, dict):
        return ["tool output is not a JSON object"]

    errors = []
    for key in ("entities", "relationships"):
        if not isinstance(data.get(key), list):
            errors.append(f"`{key}` must be a list")

    # Stop early if top-level shape is wrong — item checks would crash.
    if errors:
        return errors

    for i, item in enumerate(data["entities"]):
        if not isinstance(item, dict) or "name" not in item or "type" not in item:
            errors.append(f"entities[{i}] needs name and type")

    for i, item in enumerate(data["relationships"]):
        if not isinstance(item, dict):
            errors.append(f"relationships[{i}] must be an object")
        elif "subject" not in item or "relation" not in item or "object" not in item:
            errors.append(f"relationships[{i}] needs subject, relation, and object")

    return errors


# Verbs that are never legitimately paired with a bare LOCATION in any genre: nothing
# is ever literally "published on" or "depicted on" a place. Deliberately narrow —
# things like "identified as", "interprets", or "acquired" are NOT included here
# because a location can legitimately be the object of those (toponym identification,
# a polity acquiring/ruling territory, etc. are real historiographical claims that
# recur across the majority of historiographical works, not just this one).
NEVER_LOCATION_RELATION_KEYWORDS = (
    "publish",
    "depict",
)


def location_conflation_warnings(data):
    """
    Heuristic backstop for a recurring model failure: pairing a LOCATION entity
    (usually a findspot) with "published"/"depicted on" instead of the artifact or
    document actually published/depicted — e.g. "museum -- acquired --> village"
    written as "person -- depicted on --> village" instead of "-- depicted on -->
    seal". Flags via a warning rather than dropping the relationship, since even
    this narrow keyword check can have false positives.
    """
    if not isinstance(data, dict):
        return []

    types = {
        e["name"]: e.get("type")
        for e in data.get("entities", [])
        if isinstance(e, dict) and e.get("name")
    }

    found = []
    for r in data.get("relationships", []):
        if not isinstance(r, dict):
            continue
        subject, relation, obj = r.get("subject"), r.get("relation"), r.get("object")
        if not subject or not relation or not obj:
            continue

        location_side = subject if types.get(subject) == "LOCATION" else None
        if location_side is None and types.get(obj) == "LOCATION":
            location_side = obj
        if location_side is None:
            continue

        if not any(kw in relation.lower() for kw in NEVER_LOCATION_RELATION_KEYWORDS):
            continue

        found.append(
            f'possible artifact/findspot conflation: "{subject}" --{relation}--> "{obj}" '
            f"pairs location `{location_side}` with \"{relation}\", which should name the "
            f"artifact/document actually published or depicted, not its findspot"
        )
    return found


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def extract_one(client, content_block):
    """
    Ask Claude, normalize missing lists, validate.
    If still invalid, try one repair call.
    Returns (data, warnings, error_reason).
    """
    warnings = []

    response = chat(client, content_block)
    data = get_tool_data(response)
    if data is not None:
        data, fill_warnings = normalize_lists(data)
        warnings.extend(fill_warnings)

    errors = validation_errors(data)
    if not errors:
        warnings.extend(location_conflation_warnings(data))
        return data, warnings, None

    # One repair retry with the specific validation reason.
    reason = "; ".join(errors)
    print(f"   retry: {reason}")
    warnings.append("repair retry was attempted")
    response = chat(client, content_block, extra_note=REPAIR_PROMPT.format(reason=reason))
    data = get_tool_data(response)
    if data is not None:
        data, fill_warnings = normalize_lists(data)
        warnings.extend(fill_warnings)

    errors = validation_errors(data)
    if not errors:
        warnings.extend(location_conflation_warnings(data))
        return data, warnings, None

    return data, warnings, "; ".join(errors)


def main():
    if not os.path.isdir(INPUT_DIR):
        print(f"Error: folder '{INPUT_DIR}' not found. Create it and add documents.")
        sys.exit(1)

    files = [
        os.path.join(INPUT_DIR, name)
        for name in sorted(os.listdir(INPUT_DIR))
        if os.path.isfile(os.path.join(INPUT_DIR, name)) and not name.startswith(".")
    ]
    if not files:
        print(f"No files in '{INPUT_DIR}'. Add some and try again.")
        sys.exit(0)

    client = Anthropic()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Processing {len(files)} file(s)...\n")
    ok_count = 0
    fail_count = 0

    for path in files:
        print(f"-> {path}")
        name = os.path.splitext(os.path.basename(path))[0]

        content_block, fmt = load_file(path)
        if content_block is None:
            fail_count += 1
            save_json(
                os.path.join(OUTPUT_DIR, f"{name}.error.json"),
                {"document": {"source": path}, "error": "could not load file"},
            )
            continue

        try:
            data, warnings, error_reason = extract_one(client, content_block)
        except Exception as e:
            print(f"   FAILED: {e}")
            fail_count += 1
            save_json(
                os.path.join(OUTPUT_DIR, f"{name}.error.json"),
                {"document": {"source": path}, "error": str(e)},
            )
            continue

        if error_reason is not None:
            print(f"   FAILED: {error_reason}")
            fail_count += 1
            save_json(
                os.path.join(OUTPUT_DIR, f"{name}.error.json"),
                {
                    "document": {"source": path},
                    "error": error_reason,
                    "raw_payload": data,
                },
            )
            continue

        result = {
            "document": {
                "source": path,
                "format": fmt,
                "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            "summary": data.get("summary"),
            "entities": data["entities"],
            "relationships": data["relationships"],
            "warnings": warnings,
        }
        out_path = os.path.join(OUTPUT_DIR, f"{name}.json")
        save_json(out_path, result)
        ok_count += 1
        print(
            f"   OK: {len(result['entities'])} entities, "
            f"{len(result['relationships'])} relationships"
        )
        if warnings:
            print(f"   warnings: {'; '.join(warnings)}")
        print(f"   -> {out_path}\n")

    print(f"Done. {ok_count} succeeded, {fail_count} failed.")
    sys.exit(1 if ok_count == 0 and fail_count > 0 else 0)


if __name__ == "__main__":
    main()
