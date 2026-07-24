"""
Cross-document entity linking.

Scans every ExtractionResult in output/, finds entities whose canonical name or any
mention exactly matches (case/whitespace-insensitive, Unicode-normalized) a name or
mention in another document, and groups them into shared-entity groups spanning 2+
documents. Also flags cross-document entity pairs with similar-but-not-matching
canonical names (spelling/dialect/inflection variants, e.g. "Ljetopis popa
Dukljanina" vs "Letopis Popa Dukljanina") as `possible_matches`, for manual review —
these are never auto-merged into `groups`, since exact matching is what keeps false
merges out. Writes output/cross_document_links.json (shape: links.schema.json) fresh
on each run, alongside main.py's per-document files. Does not overwrite anything
main.py writes.

Usage:
  python link.py
  python link.py --out output/cross_document_links.json
"""

import argparse
import difflib
import glob
import json
import os
import re
import unicodedata
from datetime import datetime, timezone

OUTPUT_DIR = "output"
DEFAULT_OUT = os.path.join(OUTPUT_DIR, "cross_document_links.json")

# Deterministic tiebreak order when member documents disagree on entity type.
# Never used to gate matching, only to pick a representative type.
TYPE_PRIORITY = ["PERSON", "ORGANIZATION", "LOCATION", "PRODUCT", "EVENT", "OTHER"]

# Minimum difflib.SequenceMatcher ratio (on normalized canonical names) for a
# cross-document pair to be flagged as a possible match. Picked so a genuine
# dialect/spelling variant ("ljetopis popa dukljanina" vs "letopis popa
# dukljanina" -> 0.979) clears it with room to spare, while two different real
# entities that just happen to share a lot of characters ("kingdom of croatia"
# vs "kingdom of dalmatia" -> 0.811) stay below it.
FUZZY_MATCH_THRESHOLD = 0.90


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_form(s):
    """NFC-normalize, collapse whitespace, casefold. Empty/None -> ''."""
    s = unicodedata.normalize("NFC", s or "")
    s = re.sub(r"\s+", " ", s.strip())
    return s.casefold()


# ---------------------------------------------------------------------------
# Loading output/*.json
# ---------------------------------------------------------------------------

def is_extraction_result(data):
    return (
        isinstance(data, dict)
        and isinstance(data.get("document"), dict)
        and isinstance(data["document"].get("source"), str)
        and isinstance(data.get("entities"), list)
    )


def load_extraction_results(output_dir, exclude_path=None):
    """
    Returns (docs, skipped).
    docs: list of {"id", "path", "source", "entities"} — one per usable output file,
          in path-sorted order.
    skipped: list of {"path", "reason"}.
    exclude_path, if given, is left out of the scan entirely (used so link.py doesn't
    read back its own previous output file, which now lives in the same directory it scans).
    """
    docs = []
    skipped = []
    by_source = {}  # source -> index into docs, to catch duplicate document.source

    paths = sorted(glob.glob(os.path.join(output_dir, "*.json")))
    if exclude_path is not None:
        exclude_path = os.path.normpath(exclude_path)
        paths = [p for p in paths if os.path.normpath(p) != exclude_path]

    for path in paths:
        if path.endswith(".error.json"):
            skipped.append({"path": path, "reason": "extraction failed for this document (.error.json)"})
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            skipped.append({"path": path, "reason": f"could not parse JSON ({e})"})
            continue

        if not is_extraction_result(data):
            skipped.append({"path": path, "reason": "not an ExtractionResult (missing document.source or entities)"})
            continue

        source = data["document"]["source"]
        doc = {
            "id": os.path.splitext(os.path.basename(path))[0],
            "path": path,
            "source": source,
            "extracted_at": data["document"].get("extracted_at") or "",
            "entities": data["entities"],
        }

        if source in by_source:
            existing = docs[by_source[source]]
            if doc["extracted_at"] >= existing["extracted_at"]:
                docs[by_source[source]] = doc
                skipped.append({"path": existing["path"], "reason": f"duplicate document.source '{source}', superseded by {path}"})
            else:
                skipped.append({"path": path, "reason": f"duplicate document.source '{source}', superseded by {existing['path']}"})
            continue

        by_source[source] = len(docs)
        docs.append(doc)

    return docs, skipped


# ---------------------------------------------------------------------------
# Per-document surface-form indexing
# ---------------------------------------------------------------------------

def entity_surface_forms(entity):
    forms = set()
    if not isinstance(entity, dict):
        return forms
    name = normalize_form(entity.get("name"))
    if name:
        forms.add(name)
    for m in entity.get("mentions") or []:
        # mentions are {"text", "page"} objects; older files may still have plain strings.
        text = m.get("text") if isinstance(m, dict) else m
        text = normalize_form(text)
        if text:
            forms.add(text)
    return forms


def build_form_index(docs):
    """
    Returns (form_index, ambiguous).
    form_index: normalized form -> list of (doc_id, entity_idx), excluding forms
      dropped for within-document ambiguity.
    ambiguous: [{"document", "surface_form", "entities": [name, ...]}] for forms that
      mapped to 2+ distinct entities inside the same document (dropped from matching
      rather than guessing which entity they belong to).
    """
    raw_index = {}  # form -> list of (doc_id, entity_idx)
    for doc in docs:
        for idx, entity in enumerate(doc["entities"]):
            for form in entity_surface_forms(entity):
                raw_index.setdefault(form, []).append((doc["id"], idx))

    ambiguous = []
    form_index = {}
    for form, occurrences in raw_index.items():
        by_doc = {}
        for doc_id, idx in occurrences:
            by_doc.setdefault(doc_id, set()).add(idx)

        drop = False
        for doc_id, idxs in by_doc.items():
            if len(idxs) > 1:
                doc = next(d for d in docs if d["id"] == doc_id)
                names = sorted({doc["entities"][i].get("name", "") for i in idxs})
                ambiguous.append({"document": doc_id, "surface_form": form, "entities": names})
                drop = True

        if not drop:
            form_index[form] = occurrences

    return form_index, ambiguous


# ---------------------------------------------------------------------------
# Union-find grouping
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def group_entities(form_index):
    """
    Nodes = (doc_id, entity_idx). For each surface form present in >=2 distinct
    documents, union all its occurrences together.
    Returns (components: dict[root -> list[node]], evidence: dict[root -> list of
    edges], uf: the UnionFind instance — reused by find_possible_matches to skip
    pairs that are already exactly matched).

    Evidence is bucketed by the FINAL union-find root reached after all unions are
    performed (a second pass, once the component shape is settled), not by which
    documents an edge happens to touch — two different entities that are each
    independently shared between the same two documents must not have their
    evidence cross-attributed to each other's group.
    """
    uf = UnionFind()
    linking_forms = {
        form: occurrences
        for form, occurrences in form_index.items()
        if len({doc_id for doc_id, _ in occurrences}) >= 2
    }

    for occurrences in linking_forms.values():
        first = occurrences[0]
        for other in occurrences[1:]:
            uf.union(first, other)

    components = {}
    for node in uf.parent:
        root = uf.find(node)
        components.setdefault(root, []).append(node)

    evidence = {}
    for form, occurrences in linking_forms.items():
        root = uf.find(occurrences[0])
        doc_ids = sorted({doc_id for doc_id, _ in occurrences})
        edges = evidence.setdefault(root, [])
        for i in range(len(doc_ids) - 1):
            edges.append({"surface_form": form, "documents": [doc_ids[i], doc_ids[i + 1]]})

    return components, evidence, uf


# ---------------------------------------------------------------------------
# Fuzzy candidates (not auto-merged — see module docstring)
# ---------------------------------------------------------------------------

def find_possible_matches(docs, uf, threshold=FUZZY_MATCH_THRESHOLD):
    """
    Flags cross-document entity pairs whose canonical names are similar but not
    exactly matched (spelling/dialect/inflection variants) for manual review.
    Never merged into `groups` — exact matching stays the only thing that actually
    links entities, so this can't introduce a false-positive merge; it only adds
    a suggestion to the output for a human to confirm or ignore.

    Compares canonical `name` only (not mentions), and only entities of the same
    type, across different documents, skipping pairs already in the same
    exact-match component (per `uf.find`). Calling uf.find on a node it hasn't
    seen yet is safe — UnionFind.find() lazily registers unseen nodes as their
    own singleton root.

    Pairs are deduplicated to the single highest-similarity example per distinct
    pair of components, so if three documents already share an exact match and a
    fourth has a spelling variant, that shows up once — not three times.
    Returns a list sorted by descending similarity.
    """
    nodes = []
    for doc in docs:
        for idx, entity in enumerate(doc["entities"]):
            if not isinstance(entity, dict):
                continue
            name = entity.get("name")
            if not name:
                continue
            nodes.append({
                "doc_id": doc["id"],
                "idx": idx,
                "name": name,
                "type": entity.get("type", "OTHER"),
                "norm": normalize_form(name),
            })

    best = {}
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            if a["doc_id"] == b["doc_id"] or a["type"] != b["type"]:
                continue

            root_a = uf.find((a["doc_id"], a["idx"]))
            root_b = uf.find((b["doc_id"], b["idx"]))
            if root_a == root_b:
                continue

            similarity = difflib.SequenceMatcher(None, a["norm"], b["norm"]).ratio()
            if similarity < threshold:
                continue

            key = tuple(sorted([root_a, root_b]))
            candidate = {
                "similarity": round(similarity, 3),
                "a": {"document": a["doc_id"], "name": a["name"], "type": a["type"]},
                "b": {"document": b["doc_id"], "name": b["name"], "type": b["type"]},
            }
            if key not in best or candidate["similarity"] > best[key]["similarity"]:
                best[key] = candidate

    return sorted(best.values(), key=lambda c: -c["similarity"])


# ---------------------------------------------------------------------------
# Group construction
# ---------------------------------------------------------------------------

def pick_representative(members):
    """members: list of {"document", "local_name", "local_type", "local_mentions"}."""
    best_name = sorted(members, key=lambda m: (-len(m["local_name"]), m["local_name"]))[0]["local_name"]

    types = {m["local_type"] for m in members}
    if len(types) == 1:
        return best_name, next(iter(types)), False

    priority = {t: i for i, t in enumerate(TYPE_PRIORITY)}
    rep_type = sorted(types, key=lambda t: priority.get(t, len(TYPE_PRIORITY)))[0]
    return best_name, rep_type, True


def build_groups(docs_by_id, components, evidence):
    groups = []
    for root, nodes in components.items():
        member_doc_ids = sorted({doc_id for doc_id, _ in nodes})
        if len(member_doc_ids) < 2:
            continue

        members = []
        for doc_id, idx in sorted(nodes):
            entity = docs_by_id[doc_id]["entities"][idx]
            members.append({
                "document": doc_id,
                "local_name": entity.get("name", ""),
                "local_type": entity.get("type", "OTHER"),
                "local_mentions": entity.get("mentions") or [],
            })

        rep_name, rep_type, type_conflict = pick_representative(members)
        group_evidence = evidence.get(root, [])

        groups.append({
            "representative_name": rep_name,
            "representative_type": rep_type,
            "type_conflict": type_conflict,
            "documents": member_doc_ids,
            "members": members,
            "link_evidence": group_evidence,
        })

    groups.sort(key=lambda g: (-len(g["documents"]), g["representative_name"]))

    # Re-key each group so "id" comes first, matching the schema's field order.
    ordered = []
    for i, g in enumerate(groups, start=1):
        ordered.append({
            "id": f"group-{i:04d}",
            "representative_name": g["representative_name"],
            "representative_type": g["representative_type"],
            "type_conflict": g["type_conflict"],
            "documents": g["documents"],
            "members": g["members"],
            "link_evidence": g["link_evidence"],
        })
    return ordered


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def build_output(docs, skipped, ambiguous, groups, possible_matches, output_dir):
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_output_dir": output_dir,
        "documents": [
            {"id": d["id"], "source": d["source"], "entity_count": len(d["entities"])}
            for d in docs
        ],
        "skipped_files": skipped,
        "ambiguous_local_matches": ambiguous,
        "groups": groups,
        "possible_matches": possible_matches,
    }


def save_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=DEFAULT_OUT, help=f"Where to write the cross-document links file (default: {DEFAULT_OUT})")
    return p.parse_args()


def main():
    args = parse_args()

    docs, skipped = load_extraction_results(OUTPUT_DIR, exclude_path=args.out)
    if len(docs) < 2:
        print(f"Only {len(docs)} usable document(s) in '{OUTPUT_DIR}' — need at least 2 to find cross-document links.")

    docs_by_id = {d["id"]: d for d in docs}
    form_index, ambiguous = build_form_index(docs)
    components, evidence, uf = group_entities(form_index)
    groups = build_groups(docs_by_id, components, evidence)
    possible_matches = find_possible_matches(docs, uf)

    result = build_output(docs, skipped, ambiguous, groups, possible_matches, OUTPUT_DIR)
    save_json(args.out, result)

    print(f"Scanned {len(docs)} document(s), skipped {len(skipped)}, found {len(groups)} shared-entity group(s).")
    if ambiguous:
        print(f"   {len(ambiguous)} ambiguous local surface form(s) excluded from matching (see ambiguous_local_matches).")
    if possible_matches:
        print(f"   {len(possible_matches)} possible cross-document match(es) flagged for review (see possible_matches).")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
