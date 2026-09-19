#!/usr/bin/env python3
"""Audit script to test if all Markdown documents meet K4-L3A requirements."""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

REQUIRED_KEYS = {"doc_id", "title", "source_url", "retrieved_at", "document_version", "audience"}
VALID_AUDIENCES = {"student", "faculty", "staff", "all"}
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_frontmatter(content: str) -> tuple[dict[str, str], str]:
    if not content.startswith("---"):
        return {}, content
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content
    yaml_block = parts[1]
    body = parts[2]

    metadata = {}
    for line in yaml_block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            # strip inline comments
            if " #" in val:
                val = val.split(" #")[0].strip()
            metadata[key] = val
    return metadata, body.strip()


def run_audit(corpus_dir: Path) -> int:
    print(f"=== Auditing Corpus: {corpus_dir} ===")
    errors = []
    warnings = []

    md_files = sorted(corpus_dir.glob("*.md"))
    print(f"1. Document count: {len(md_files)}")
    if not (5 <= len(md_files) <= 10):
        errors.append(f"Corpus size must be between 5 and 10 documents, got {len(md_files)}")

    doc_ids = set()
    audiences = set()
    parsed_docs = {}

    for file_path in md_files:
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"{file_path.name}: File is not UTF-8 encoded")
            continue

        meta, body = parse_frontmatter(content)
        parsed_docs[file_path.name] = meta

        # Check frontmatter presence
        if not meta:
            errors.append(f"{file_path.name}: Missing YAML frontmatter")
            continue

        # Check required keys
        missing_keys = REQUIRED_KEYS - set(meta.keys())
        if missing_keys:
            errors.append(f"{file_path.name}: Missing required metadata keys: {missing_keys}")

        # Check doc_id
        doc_id = meta.get("doc_id", "")
        if not doc_id:
            errors.append(f"{file_path.name}: Empty doc_id")
        elif doc_id in doc_ids:
            errors.append(f"{file_path.name}: Duplicate doc_id '{doc_id}'")
        else:
            doc_ids.add(doc_id)

        if doc_id != file_path.stem:
            warnings.append(f"{file_path.name}: doc_id '{doc_id}' does not match filename stem '{file_path.stem}'")

        # Check audience
        aud = meta.get("audience", "")
        if aud not in VALID_AUDIENCES:
            errors.append(f"{file_path.name}: Invalid audience '{aud}' (expected one of {VALID_AUDIENCES})")
        else:
            audiences.add(aud)

        # Check retrieved_at
        retrieved_at = meta.get("retrieved_at", "")
        if not DATE_PATTERN.match(retrieved_at):
            errors.append(f"{file_path.name}: Invalid retrieved_at date format '{retrieved_at}' (expected YYYY-MM-DD)")

        # Check source_url
        url = meta.get("source_url", "")
        if not url.startswith("http://") and not url.startswith("https://"):
            errors.append(f"{file_path.name}: Invalid or missing source_url '{url}'")

        # Check document_version
        if not meta.get("document_version"):
            errors.append(f"{file_path.name}: Missing document_version")

        # Check content length
        if len(body) < 80:
            errors.append(f"{file_path.name}: Content too short ({len(body)} chars < 80)")

    # Audience diversity
    print(f"2. Distinct audience values found: {audiences}")
    if len(audiences) < 2:
        errors.append(f"Must have at least 2 distinct audience values for metadata filtering, found only {audiences}")

    # Check sources.csv
    sources_path = corpus_dir / "sources.csv"
    print(f"3. Checking sources.csv manifest: {sources_path}")
    if not sources_path.is_file():
        errors.append("sources.csv is missing in corpus directory")
    else:
        with sources_path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            manifest_rows = list(reader)

        manifest_doc_ids = {row.get("doc_id") for row in manifest_rows}
        if manifest_doc_ids != doc_ids:
            diff1 = doc_ids - manifest_doc_ids
            diff2 = manifest_doc_ids - doc_ids
            if diff1:
                errors.append(f"Files missing from sources.csv: {diff1}")
            if diff2:
                errors.append(f"Extra entries in sources.csv without files: {diff2}")

    # Report
    print("\n=== Audit Summary ===")
    for w in warnings:
        print(f"  [WARN] {w}")
    if errors:
        for e in errors:
            print(f"  [ERROR] {e}")
        print(f"\nFAILED with {len(errors)} error(s).")
        return 1
    else:
        print("\nSUCCESS! All requirements passed perfectly.")
        return 0


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/university")
    sys.exit(run_audit(path))
