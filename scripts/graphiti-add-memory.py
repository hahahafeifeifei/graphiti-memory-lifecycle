#!/usr/bin/env python3
"""
graphiti-add-memory.py

Daily distillation pipeline:
1) Read raw hook memories from memory/YYYY-MM-DD.md
2) Distill per memory type into validated template entries
3) Write distilled files under memory/{TypeDir}/
4) Import only distilled entries to Graphiti
5) Promote key lines to workspace files
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from conf import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    MEMORY_DIR,
    WORKSPACE_DIR,
    add_episode_via_server,
)

MEMORY_TYPES = [
    "Execute error",
    "User profile",
    "Learned knowledge",
    "Project progress",
    "Key decision",
    "Other memory",
]

TYPE_TO_FILE = {
    "Execute error": "Error",
    "User profile": "Profile",
    "Learned knowledge": "Learning",
    "Project progress": "Project",
    "Key decision": "Decision",
    "Other memory": "Others",
}

TYPE_TO_TEMPLATE_FILE = {
    "Execute error": "execute_error.md",
    "User profile": "user_profile.md",
    "Learned knowledge": "learned_knowledge.md",
    "Project progress": "project_progress.md",
    "Key decision": "key_decision.md",
    "Other memory": "other_memory.md",
}

WORKSPACE_FILE_LIMIT = 1024
SCRIPT_DIR = Path(__file__).parent
TEMPLATES_DIR = SCRIPT_DIR.parent / "references"
DISTILL_MAX_TOKENS = 65536
DISTILL_REPAIR_MAX_TOKENS = 65536
PROMOTE_MAX_TOKENS = 65536
PROMOTE_COMPRESS_MAX_TOKENS = 65536
PROMOTE_EXTRACT_MAX_ATTEMPTS = 3
WORKSPACE_TARGETS = [
    "SOUL.md",
    "IDENTITY.md",
    "TOOLS.md",
    "USER.md",
    "PROJECT.md",
    "MEMORY.md",
]
WORKSPACE_TARGET_HINTS = {
    "SOUL.md": "Self perception and stable identity principles.",
    "IDENTITY.md": "Behavioral patterns and response style preferences.",
    "TOOLS.md": "Tool gotchas, execution safeguards, and workflow improvements.",
    "USER.md": "User-specific preferences and interaction constraints.",
    "PROJECT.md": "Ongoing project status, priorities, and stable working direction.",
    "MEMORY.md": "Other broadly useful memory not covered by other targets.",
}


def _to_one_sentence(text: str) -> str:
    compact = " ".join((text or "").replace("\n", " ").split()).strip()
    if not compact:
        return ""
    m = re.search(r"[。！？.!?]", compact)
    if not m:
        return compact
    return compact[: m.end()].strip()


def load_template(mem_type: str) -> str:
    filename = TYPE_TO_TEMPLATE_FILE.get(mem_type, "other_memory.md")
    path = TEMPLATES_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8")
    return (
        f"## {mem_type}: {{Title}}\n\n"
        "### Summary\nOne-line description.\n\n"
        "### Details\n- Details\n\n"
        "### Metadata\n"
        "- Priority: High\n"
        "- Class: General\n"
        "- Keywords: memory\n"
        "- Seen time: YYYY-MM-DD\n"
    )


def _split_blocks(text: str) -> list[str]:
    parts = re.split(r"(?=^##\s+)", text, flags=re.MULTILINE)
    return [p.strip() for p in parts if p.strip()]


def parse_raw_entries(text: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []

    for block in _split_blocks(text):
        lines = block.splitlines()
        if not lines:
            continue

        header = lines[0].strip()
        body = "\n".join(lines[1:])

        # New raw format from hook capture
        time_match = re.search(r"^Time:\s*(.+)$", body, flags=re.MULTILINE)
        type_match = re.search(r"^Type:\s*(.+)$", body, flags=re.MULTILINE)
        details_match = re.search(r"^Details:\s*\n(.+)$", body, flags=re.MULTILINE | re.DOTALL)
        if not details_match:
            details_match = re.search(r"^Details:\s*(.+)$", body, flags=re.MULTILINE | re.DOTALL)

        if type_match and details_match:
            raw_type = type_match.group(1).strip()
            if raw_type not in MEMORY_TYPES:
                continue
            mem_type = raw_type
            title = header.replace("##", "", 1).strip()
            title = re.sub(r"^Memory Candidate\s*-\s*", "", title, flags=re.IGNORECASE)
            if not title:
                title = details_match.group(1).strip().split("\n", 1)[0][:64]

            entries.append(
                {
                    "time": (time_match.group(1).strip() if time_match else datetime.now().strftime("%Y-%m-%d %H:%M")),
                    "type": mem_type,
                    "title": title,
                    "details": details_match.group(1).strip(),
                }
            )
            continue

    return entries


def validate_entry(text: str) -> list[str]:
    errors: list[str] = []
    lines = text.strip().splitlines()
    if not lines:
        return ["Empty"]

    tp = "|".join(re.escape(t) for t in MEMORY_TYPES)
    if not re.match(rf"^##\s+({tp}):\s*.+$", lines[0]):
        errors.append("Bad header")
        return errors

    for section in ("Summary", "Details", "Metadata"):
        if not re.search(rf"###\s+{section}", text, re.IGNORECASE):
            errors.append(f"Missing ### {section}")

    meta_match = re.search(r"###\s+Metadata\s*\n(.+?)(?=\n###|\Z)", text, flags=re.DOTALL | re.IGNORECASE)
    if meta_match:
        meta = meta_match.group(1)
        for field in ("Priority", "Class", "Keywords", "Seen time"):
            if not re.search(rf"[-*]\s*{re.escape(field)}:\s*", meta, flags=re.IGNORECASE):
                errors.append(f"Missing: {field}")
    else:
        errors.append("Missing Metadata block")

    return errors


def parse_entry(text: str) -> dict[str, str]:
    lines = text.strip().splitlines()
    tp = "|".join(re.escape(t) for t in MEMORY_TYPES)
    header_match = re.match(rf"^##\s+({tp}):\s*(.+)$", lines[0])
    if not header_match:
        raise ValueError("invalid header")

    mem_type = header_match.group(1).strip()
    title = header_match.group(2).strip()

    def _section(name: str) -> str:
        m = re.search(rf"###\s+{name}\s*\n(.+?)(?=\n###|\Z)", text, flags=re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    metadata = _section("Metadata")

    def _meta(key: str) -> str:
        m = re.search(rf"[-*]\s*{re.escape(key)}:\s*(.+)", metadata, flags=re.IGNORECASE)
        return m.group(1).strip() if m else ""

    return {
        "type": mem_type,
        "title": title,
        "name": f"{mem_type}: {title}",
        "summary": _section("Summary"),
        "details": _section("Details"),
        "priority": _meta("Priority"),
        "class_": _meta("Class"),
        "keywords": _meta("Keywords"),
        "seen_time": _meta("Seen time"),
    }


def build_fallback_entry(mem_type: str, item: dict[str, str]) -> str:
    seen = (item.get("time", "") or datetime.now().strftime("%Y-%m-%d %H:%M"))[:10]
    details = item.get("details", "").strip()
    title = item.get("title", "Memory").strip() or "Memory"

    return (
        f"## {mem_type}: {title}\n\n"
        "### Summary\n"
        f"{details.splitlines()[0][:200] if details else 'Captured from hook memory.'}\n\n"
        "### Details\n"
        f"- {details.replace(chr(10), chr(10) + '- ')}\n\n"
        "### Metadata\n"
        "- Priority: Medium\n"
        "- Class: HookCapture\n"
        "- Keywords: hook, memory\n"
        f"- Seen time: {seen}\n"
    )


def _strip_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):].strip()
    if cleaned.startswith("```"):
        cleaned = cleaned[3:].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    return cleaned


async def _repair_block_with_llm(
    client: AsyncOpenAI,
    mem_type: str,
    template: str,
    bad_block: str,
    errors: list[str],
) -> str | None:
    repair_prompt = (
        f"Memory type: {mem_type}\n"
        "Fix the markdown entry so it strictly follows the template and validator rules.\n"
        "Rules:\n"
        "- Header must be: ## <Type>: <Title>\n"
        "- Must include sections: Summary, Details, Metadata\n"
        "- Metadata must include: Priority, Class, Keywords, Seen time\n"
        "- Output markdown only, no explanations, no code fences.\n\n"
        f"Template:\n{template}\n\n"
        f"Validation errors:\n- " + "\n- ".join(errors) + "\n\n"
        f"Bad entry:\n{bad_block}\n"
    )

    response = await client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": "You are a strict markdown entry formatter."},
            {"role": "user", "content": repair_prompt},
        ],
        max_tokens=DISTILL_REPAIR_MAX_TOKENS,
    )
    repaired = _strip_fence(response.choices[0].message.content or "")
    return repaired.strip() or None


async def distill_type(mem_type: str, items: list[dict[str, str]]) -> list[str]:
    if not items:
        return []

    template = load_template(mem_type)
    client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    raw_bullets = []
    for idx, item in enumerate(items, start=1):
        raw_bullets.append(
            f"- Candidate #{idx}\n"
            f"  Time: {item.get('time', '')}\n"
            f"  Title: {item.get('title', '')}\n"
            f"  Details: {item.get('details', '')[:1200]}"
        )

    prompt = (
        f"You are distilling '{mem_type}' memories from daily raw candidates.\n\n"
        "Task:\n"
        "1. Group candidates by semantic similarity (same topic/domain/entity).\n"
        "2. Keep unrelated subjects as separate entries.\n"
        "3. Resolve conflicts by preferring newer facts when both cannot be true.\n"
        "4. Extract reusable knowledge, patterns, and decisions for future work.\n"
        "5. Output only complete entries that strictly follow the required template.\n\n"
        "Grouping rules:\n"
        "- Same tool/technology can merge.\n"
        "- Same error type can merge.\n"
        "- Same user preference can merge.\n"
        "- Same project stream can merge.\n"
        "- Different subject/entity must not merge.\n\n"
        "Output constraints:\n"
        f"- Every entry MUST start with: ## {mem_type}: <Title>\n"
        "- Sections required in each entry:\n"
        "  ### Summary\n"
        "  ### Details\n"
        "  ### Metadata\n"
        "- Metadata must include bullet fields exactly:\n"
        "  - Priority:\n"
        "  - Class:\n"
        "  - Keywords:\n"
        "  - Seen time:\n"
        "- Keep each entry self-contained and atomic.\n"
        "- No explanations outside entries, no code fences.\n\n"
        f"Template:\n{template}\n\n"
        "Raw candidates:\n"
        + "\n".join(raw_bullets)
    )

    try:
        response = await client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You produce strict markdown entries only. "
                        "No prose outside entries. No code fences."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=DISTILL_MAX_TOKENS,
        )

        text = _strip_fence(response.choices[0].message.content or "")
        blocks = _split_blocks(text)
        valid: list[str] = []
        invalid_blocks: list[tuple[str, list[str]]] = []
        for block in blocks:
            errors = validate_entry(block)
            if not errors:
                valid.append(block)
            else:
                invalid_blocks.append((block, errors))

        # Retry invalid blocks through a dedicated repair prompt.
        for block, errors in invalid_blocks:
            try:
                repaired = await _repair_block_with_llm(client, mem_type, template, block, errors)
            except Exception:
                repaired = None
            if not repaired:
                continue
            if not validate_entry(repaired):
                valid.append(repaired)

        if valid:
            return valid
    except Exception:
        pass

    # Fallback if LLM distillation failed or invalid
    return [build_fallback_entry(mem_type, item) for item in items]


async def distill_all(grouped: dict[str, list[dict[str, str]]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for mem_type, items in grouped.items():
        out[mem_type] = await distill_type(mem_type, items)
    return out


def write_distilled(
    date_str: str, distilled: dict[str, list[str]]
) -> tuple[list[tuple[str, str]], list[Path]]:
    memory_root = Path(MEMORY_DIR)
    written: list[tuple[str, str]] = []
    written_files: list[Path] = []

    for mem_type, blocks in distilled.items():
        if not blocks:
            continue

        type_dir = TYPE_TO_FILE[mem_type]
        type_path = memory_root / type_dir
        type_path.mkdir(parents=True, exist_ok=True)

        daily_file = type_path / f"{type_dir}.{date_str}.md"
        content = "\n\n".join(blocks).strip() + "\n"
        daily_file.write_text(content, encoding="utf-8")
        written_files.append(daily_file)

        summary_file = type_path / f"{type_dir}.md"
        with summary_file.open("a", encoding="utf-8") as f:
            f.write("\n" + content)
        written_files.append(summary_file)

        for block in blocks:
            written.append((mem_type, block))

    return written, written_files


async def import_distilled(entries: list[dict[str, str]]) -> tuple[int, int]:
    ok = 0
    fail = 0

    for entry in entries:
        body = json.dumps(
            {
                "type": entry["type"],
                "title": entry["title"],
                "summary": entry.get("summary", ""),
                "details": entry.get("details", ""),
                "priority": entry.get("priority", ""),
                "category": entry.get("class_", ""),
                "keywords": entry.get("keywords", ""),
                "seen_time": entry.get("seen_time", ""),
            },
            ensure_ascii=False,
        )

        success, msg = add_episode_via_server(
            name=entry["name"],
            body=body,
            source="json",
            source_description=f"memory-distilled:{entry['type']}",
        )

        if success:
            print(f"  [OK] {entry['name']} — {msg}")
            ok += 1
        else:
            print(f"  [FAIL] {entry['name']} — {msg}")
            fail += 1

    return ok, fail


def _normalize_bullet_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in lines:
        txt = str(raw or "").strip()
        if not txt:
            continue
        if txt.startswith("- "):
            txt = txt[2:].strip()
        else:
            txt = txt.lstrip("-").strip()
        txt = _to_one_sentence(txt)
        if not txt:
            continue
        txt = "- " + txt
        key = txt.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(txt)
    return out


def _type_daily_file(mem_type: str, date_str: str) -> Path:
    type_dir = TYPE_TO_FILE[mem_type]
    return Path(MEMORY_DIR) / type_dir / f"{type_dir}.{date_str}.md"


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = _strip_fence(text)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _build_daily_typed_memory_corpus(date_str: str) -> str:
    sections: list[str] = []
    for mem_type in MEMORY_TYPES:
        daily_file = _type_daily_file(mem_type, date_str)
        if not daily_file.exists():
            continue
        text = daily_file.read_text(encoding="utf-8").strip()
        if not text:
            continue
        sections.append(f"# Type: {mem_type}\n\n{text}")
    return "\n\n".join(sections).strip()


async def _extract_target_candidates_from_corpus(
    client: AsyncOpenAI,
    target_file: str,
    corpus_text: str,
    existing_text: str,
) -> list[str]:
    src = (corpus_text or "").strip()
    if not src:
        return []

    target_hint = WORKSPACE_TARGET_HINTS.get(target_file, "Broadly useful memory.")
    collected_lines: list[str] = []
    instruction = (
        "You are building a single workspace memory file from typed distilled memory.\n"
        f"Target file: {target_file}\n"
        f"Target scope: {target_hint}\n\n"
        "Task:\n"
        "1) Read the complete typed memory input.\n"
        "2) Keep only high-value, broadly reusable rules for the target scope.\n"
        "3) Avoid duplicating existing workspace bullets.\n\n"
        "Return JSON only:\n"
        "{\n"
        '  "lines": ["Always ...", "Prefer ..."]\n'
        "}\n\n"
        "Constraints:\n"
        "- max 15 lines\n"
        "- each line must be exactly one sentence\n"
        "- line length <= 120 chars\n"
        "- no duplicates"
    )

    raw_list: Any = None
    last_reason = ""
    last_output = ""
    for attempt in range(1, PROMOTE_EXTRACT_MAX_ATTEMPTS + 1):
        messages: list[dict[str, str]] = [
            {"role": "system", "content": "You output strict JSON only."},
            {"role": "user", "content": instruction},
            {"role": "user", "content": f"Existing workspace file:\n{existing_text or '(empty)'}"},
            {"role": "user", "content": src},
        ]
        if attempt > 1:
            retry_note = (
                "Your previous output was invalid.\n"
                'Return one JSON object with schema: {"lines":["One sentence rule."]}\n'
            )
            if last_reason:
                retry_note += f"\nValidation issue: {last_reason}\n"
            if last_output:
                retry_note += f"\nPrevious output:\n{last_output}\n"
            messages.append({"role": "user", "content": retry_note})

        try:
            response = await client.chat.completions.create(
                model=LLM_MODEL,
                temperature=0.1 if attempt == 1 else 0.0,
                messages=messages,
                max_tokens=PROMOTE_MAX_TOKENS,
            )
        except Exception as e:
            last_reason = f"request failed: {e}"
            continue

        content = response.choices[0].message.content or ""
        last_output = _strip_fence(content).strip()
        payload = _extract_json_object(content)
        raw_list = payload.get("lines")
        if isinstance(raw_list, list):
            break
        last_reason = "missing or invalid lines list"

    if not isinstance(raw_list, list):
        return []
    for item in raw_list:
        line = _to_one_sentence(str(item or "").strip())
        if line:
            collected_lines.append(line[:120])

    return _normalize_bullet_lines(collected_lines)


def _fallback_merge_lines(existing_text: str, new_lines: list[str]) -> str:
    existing_lines = [
        ln.strip() for ln in (existing_text or "").splitlines()
        if ln.strip().startswith("-")
    ]
    merged = _normalize_bullet_lines(existing_lines + new_lines)
    return ("\n".join(merged) + "\n") if merged else ""


def _truncate_to_limit(content: str, limit_bytes: int = WORKSPACE_FILE_LIMIT) -> str:
    lines = _normalize_bullet_lines(content.splitlines())
    if not lines:
        return ""
    kept: list[str] = []
    for ln in lines:
        candidate = "\n".join(kept + [ln]).strip() + "\n"
        if len(candidate.encode("utf-8")) >= limit_bytes:
            break
        kept.append(ln)
    if not kept:
        seed = lines[0][: max(8, limit_bytes // 4)]
        kept = [seed if seed.startswith("- ") else f"- {seed.lstrip('-').strip()}"]
    return "\n".join(kept).strip() + "\n"


async def _compress_workspace_content(
    client: AsyncOpenAI,
    content: str,
) -> str:
    current = content.strip() + ("\n" if content.strip() else "")
    for _ in range(4):
        if len(current.encode("utf-8")) < WORKSPACE_FILE_LIMIT:
            return current
        response = await client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Compress the user-provided markdown bullet list.\n"
                        f"Hard limit: < {WORKSPACE_FILE_LIMIT} bytes UTF-8.\n"
                        "Keep only critical, broadly applicable prompts/rules.\n"
                        "Each bullet must be exactly one sentence.\n"
                        "Output markdown bullets only. No prose. No headers."
                    ),
                },
                {"role": "user", "content": current},
            ],
            max_tokens=PROMOTE_COMPRESS_MAX_TOKENS,
        )
        current = _strip_fence(response.choices[0].message.content or "").strip()
        if current and not current.endswith("\n"):
            current += "\n"
    return _truncate_to_limit(current, WORKSPACE_FILE_LIMIT)


async def _synthesize_workspace_file(
    client: AsyncOpenAI,
    target_file: str,
    existing_text: str,
    candidate_lines: list[str],
) -> str:
    if not candidate_lines and not existing_text.strip():
        return ""

    payload = json.dumps(
        {
            "existing_workspace_bullets": existing_text or "",
            "new_candidate_bullets": candidate_lines,
            "target_file": target_file,
            "target_scope": WORKSPACE_TARGET_HINTS.get(target_file, ""),
        },
        ensure_ascii=False,
    )
    response = await client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": (
                    "Merge existing workspace prompts with new candidate prompts.\n"
                    "Target file and scope are provided in the JSON payload.\n"
                    "Keep only high-value, reusable, non-duplicated lines.\n"
                    "Prefer imperative guidance ('Always...', 'Prefer...', 'Avoid...').\n"
                    "Every bullet must be exactly one sentence.\n"
                    f"Hard limit: output must be < {WORKSPACE_FILE_LIMIT} bytes UTF-8.\n"
                    "Output markdown bullets only. No heading, no explanation."
                ),
            },
            {"role": "user", "content": payload},
        ],
        max_tokens=PROMOTE_MAX_TOKENS,
    )
    text = _strip_fence(response.choices[0].message.content or "").strip()
    if not text:
        return _fallback_merge_lines(existing_text, candidate_lines)
    normalized = _normalize_bullet_lines(text.splitlines())
    if not normalized:
        return _fallback_merge_lines(existing_text, candidate_lines)
    out = "\n".join(normalized).strip() + "\n"
    if len(out.encode("utf-8")) >= WORKSPACE_FILE_LIMIT:
        out = await _compress_workspace_content(client, out)
    if len(out.encode("utf-8")) >= WORKSPACE_FILE_LIMIT:
        out = _truncate_to_limit(out, WORKSPACE_FILE_LIMIT)
    return out


async def promote_workspace(date_str: str) -> dict[str, int]:
    ws = Path(WORKSPACE_DIR)
    ws.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    typed_corpus = _build_daily_typed_memory_corpus(date_str)
    if not typed_corpus:
        return {}

    promoted: dict[str, int] = {}
    for file_name in WORKSPACE_TARGETS:
        path = ws / file_name
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        try:
            candidates = await _extract_target_candidates_from_corpus(
                client=client,
                target_file=file_name,
                corpus_text=typed_corpus,
                existing_text=existing,
            )
        except Exception:
            candidates = []
        try:
            final_text = await _synthesize_workspace_file(client, file_name, existing, candidates)
        except Exception:
            merged = _fallback_merge_lines(existing, candidates)
            final_text = _truncate_to_limit(merged, WORKSPACE_FILE_LIMIT) if merged else ""

        if final_text and len(final_text.encode("utf-8")) >= WORKSPACE_FILE_LIMIT:
            final_text = _truncate_to_limit(final_text, WORKSPACE_FILE_LIMIT)

        if final_text == existing:
            continue

        if final_text:
            path.write_text(final_text, encoding="utf-8")
            promoted[file_name] = len([ln for ln in final_text.splitlines() if ln.strip()])

    return promoted


def resolve_raw_file(date_str: str, override: str | None) -> Path:
    if override:
        return Path(override)
    return Path(MEMORY_DIR) / f"{date_str}.md"


async def run(date_str: str, raw_file: Path) -> int:
    if not raw_file.exists():
        print(f"No raw file found: {raw_file}")
        return 0

    raw_text = raw_file.read_text(encoding="utf-8")
    raw_entries = parse_raw_entries(raw_text)
    if not raw_entries:
        print(f"No valid raw entries in: {raw_file}")
        return 0

    grouped: dict[str, list[dict[str, str]]] = {t: [] for t in MEMORY_TYPES}
    for item in raw_entries:
        mem_type = item.get("type", "")
        if mem_type in grouped:
            grouped[mem_type].append(item)

    print(f"Distilling {len(raw_entries)} raw entries from {raw_file}")
    distilled = await distill_all(grouped)

    written_blocks, written_files = write_distilled(date_str, distilled)
    if written_files:
        print("Distilled files updated:")
        for path in written_files:
            print(f"  {path}")
    parsed_entries: list[dict[str, str]] = []

    for _, block in written_blocks:
        errs = validate_entry(block)
        if errs:
            continue
        try:
            parsed_entries.append(parse_entry(block))
        except Exception:
            continue

    if not parsed_entries:
        print("No validated distilled entries to import")
        return 1

    print(f"Importing {len(parsed_entries)} distilled entries to Graphiti...")
    ok, fail = await import_distilled(parsed_entries)
    print(f"Graphiti import complete: {ok} success, {fail} failed")

    promoted = await promote_workspace(date_str)
    if promoted:
        print("Workspace promotion:")
        for file_name, count in promoted.items():
            print(f"  {file_name}: +{count}")

    return 0 if fail == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill raw memories and import distilled entries")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    parser.add_argument("--date", default=yesterday, help="Date to process (YYYY-MM-DD)")
    parser.add_argument("--raw-file", help="Optional raw file path")
    args = parser.parse_args()

    raw_file = resolve_raw_file(args.date, args.raw_file)
    rc = asyncio.run(run(args.date, raw_file))
    sys.exit(rc)


if __name__ == "__main__":
    main()
