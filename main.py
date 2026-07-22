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
    "description": "Extract summary, entities, dates, and relationships from the document.",
    "input_schema": {
        "type": "object",
        "required": ["summary", "entities", "dates", "relationships"],
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
                        "mentions": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "dates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["date", "description"],
                    "properties": {
                        "date": {"type": "string"},
                        "normalized": {"type": ["string", "null"]},
                        "description": {"type": "string"},
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
                    },
                },
            },
        },
    },
}

PROMPT = (
    "Read the document and call extract_structured_data exactly once with:\n"
    "- summary: 1-2 sentences\n"
    "- entities: only the most important people, organizations, locations, products, "
    "or events (max 20). Prefer names central to the article; skip routine bibliography "
    "citations. type = PERSON, ORGANIZATION, LOCATION, PRODUCT, EVENT, or OTHER\n"
    "- dates: only the most important dates (max 20), each with a short description and "
    "normalized YYYY-MM-DD if known\n"
    "- relationships: subject / relation / object triples (always include this field; "
    "use [] if none)\n"
    "Always include all four fields. Use empty lists when nothing is found. Do not invent facts."
)

REPAIR_PROMPT = (
    "Your previous tool call was incomplete or invalid: {reason}.\n"
    "Call extract_structured_data again now. You MUST include summary, entities, dates, "
    "and relationships (use [] for any list with no items). Keep entities/dates to the "
    "most important ones (max 20 each)."
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
    return client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        tools=[EXTRACTION_TOOL],
        tool_choice={"type": "tool", "name": "extract_structured_data"},
        messages=[
            {
                "role": "user",
                "content": [
                    content_block,
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


def normalize_lists(data):
    """
    If Claude forgot a list field (common when output is truncated),
    fill it with [] and record a warning. Returns (data, warnings).
    """
    warnings = []
    if not isinstance(data, dict):
        return data, warnings

    for key in ("entities", "dates", "relationships"):
        if key not in data or data[key] is None:
            data[key] = []
            warnings.append(f"missing `{key}` filled with empty list")
        elif not isinstance(data[key], list):
            pass
    return data, warnings


def validation_errors(data):
    """Return a list of human-readable problems, or [] if data looks valid."""
    if data is None:
        return ["Claude did not call the extract_structured_data tool"]
    if not isinstance(data, dict):
        return ["tool output is not a JSON object"]

    errors = []
    for key in ("entities", "dates", "relationships"):
        if not isinstance(data.get(key), list):
            errors.append(f"`{key}` must be a list")

    # Stop early if top-level shape is wrong — item checks would crash.
    if errors:
        return errors

    for i, item in enumerate(data["entities"]):
        if not isinstance(item, dict) or "name" not in item or "type" not in item:
            errors.append(f"entities[{i}] needs name and type")

    for i, item in enumerate(data["dates"]):
        if not isinstance(item, dict) or "date" not in item:
            errors.append(f"dates[{i}] needs date")

    for i, item in enumerate(data["relationships"]):
        if not isinstance(item, dict):
            errors.append(f"relationships[{i}] must be an object")
        elif "subject" not in item or "relation" not in item or "object" not in item:
            errors.append(f"relationships[{i}] needs subject, relation, and object")

    return errors


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
            "dates": data["dates"],
            "relationships": data["relationships"],
            "warnings": warnings,
        }
        out_path = os.path.join(OUTPUT_DIR, f"{name}.json")
        save_json(out_path, result)
        ok_count += 1
        print(
            f"   OK: {len(result['entities'])} entities, "
            f"{len(result['dates'])} dates, "
            f"{len(result['relationships'])} relationships"
        )
        if warnings:
            print(f"   warnings: {'; '.join(warnings)}")
        print(f"   -> {out_path}\n")

    print(f"Done. {ok_count} succeeded, {fail_count} failed.")
    sys.exit(1 if ok_count == 0 and fail_count > 0 else 0)


if __name__ == "__main__":
    main()
