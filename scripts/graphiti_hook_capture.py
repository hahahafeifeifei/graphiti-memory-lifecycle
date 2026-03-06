#!/usr/bin/env python3
"""
Capture hook pipeline:
- Build transcript from session file/messages
- Ask LLM to insert semantic [SPLIT] markers and extract memory candidates
- Update NOW.md from built-in template
- Append raw memory candidates to memory/YYYY-MM-DD.md
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from conf import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, MEMORY_DIR, WORKSPACE_DIR

DEFAULT_NOW_TEMPLATE_TEXT = """# NOW

## Project Goal
{{project_goal}}

## Current Progress
{{current_progress}}

## Next Work
{{next_work}}

## Session
- Trigger: {{trigger}}
- Session ID: {{session_id}}
- Updated At: {{local_time_yyyy_mm_dd_hh_mm}}
"""

ALLOWED_MEMORY_TYPES = {
    "Execute error",
    "User profile",
    "Learned knowledge",
    "Project progress",
    "Key decision",
    "Other memory",
}

TRANSCRIPT_CHUNK_CHARS = 50000
EXTRACT_MAX_TOKENS = 65536
REPAIR_MAX_TOKENS = 65536
MAX_REPAIR_ATTEMPTS = 5

SPLIT_PROMPT = """# Role
You are a Contextual Architect specializing in Long-term Memory Distillation for AI Agents.

# Task
Analyze the provided conversation transcript and identify semantic breakpoints.
Insert a [SPLIT] marker whenever one of the following occurs:
1. Topic Shift
2. Intent Change
3. Environment Change
4. Milestone Reached

Then extract memory candidates for each segment.

Return JSON only with this schema:
{
  "project_goal": "string",
  "current_progress": ["string"],
  "next_work": ["string"],
  "segmented_transcript": "string with [SPLIT] markers",
  "memories": [
    {
      "type": "Execute error|User profile|Learned knowledge|Project progress|Key decision|Other memory",
      "title": "short title",
      "details": "one self-contained paragraph, should include detailed information that can review the event in the future"
    }
  ]
}

The memory type of event can be determined using this table:
| Memory type         | When to use                                                                                      |
|---------------------|--------------------------------------------------------------------------------------------------|
| Execute error       | Command returns non-zero exit code; exception or stack trace; unexpected output; timeout/failure  |
| User profile        | User preference; user behavior pattern; user important thought and philosophy                     |
| Learned knowledge   | User provides info you didn't know; found better approach; principles from practice               |
| Project progress    | Roadmap/design; what was done recently; current status; next steps                                |
| Key decision        | Important decision made; lessons for future planning                                             |
| Other memory        | Things you like/think; things that don't fit other types                                         |

Constraints:
- Keep memories atomic and reusable.
- One segment can produce multiple memories.
- Avoid duplicates.
- If uncertain, use type "Other memory".
"""


def _extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        texts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type", ""))
            if block_type in {"text", "output_text"} and isinstance(block.get("text"), str):
                texts.append(block["text"].strip())
            elif block_type == "input_text" and isinstance(block.get("text"), str):
                texts.append(block["text"].strip())
        return "\n".join(t for t in texts if t).strip()

    return ""


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        role = msg.get("role")
        if role not in {"user", "assistant"}:
            continue

        content = msg.get("content")
        text = _extract_text_from_content(content)
        if not text and isinstance(msg.get("message"), dict):
            nested = msg["message"]
            role = nested.get("role", role)
            text = _extract_text_from_content(nested.get("content"))

        if text:
            normalized.append({"role": str(role), "text": text})

    return normalized


def _load_from_session_file(session_file: str) -> list[dict[str, str]]:
    path = Path(session_file)
    if not path.exists():
        return []

    messages: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue

            if payload.get("type") != "message":
                continue

            message_obj = payload.get("message")
            if isinstance(message_obj, dict):
                messages.append(message_obj)

    return _normalize_messages(messages)


def _messages_to_transcript(messages: list[dict[str, str]]) -> str:
    parts = [f"{m['role'].upper()}: {m['text']}" for m in messages if m.get("text")]
    return "\n\n".join(parts)


def _split_transcript(transcript: str, max_chars: int = TRANSCRIPT_CHUNK_CHARS) -> list[str]:
    """
    Split full transcript into chunks by message boundary ("\n\n"), keeping all content.
    No truncation is applied; oversized single blocks are hard-split by char length.
    """
    if not transcript:
        return []

    blocks = transcript.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0

    for block in blocks:
        if not block:
            continue
        block_len = len(block)

        # Keep all content: if a single block is too large, split it into slices.
        if block_len > max_chars:
            flush()
            start = 0
            while start < block_len:
                end = start + max_chars
                chunks.append(block[start:end])
                start = end
            continue

        projected = current_len + (2 if current else 0) + block_len
        if current and projected > max_chars:
            flush()

        current.append(block)
        current_len += (2 if current_len > 0 else 0) + block_len

    flush()
    return chunks


def _norm_memories(items: list[dict[str, Any]], fallback_time: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        mem_type = str(item.get("type", "")).strip()
        details = str(item.get("details", "")).strip()
        if not details or mem_type not in ALLOWED_MEMORY_TYPES:
            continue

        title = str(item.get("title", "")).strip()
        if not title:
            title = details.split("\n", 1)[0][:60]

        out.append(
            {
                "time": fallback_time,
                "type": mem_type,
                "title": title,
                "details": details,
            }
        )

    return out


def _strip_fence(content: str) -> str:
    cleaned = (content or "").strip()
    for fence in ("```json", "```"):
        if cleaned.startswith(fence):
            cleaned = cleaned[len(fence):]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def _validate_extracted_payload(payload: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["root must be an object"]

    required = ["project_goal", "current_progress", "next_work", "segmented_transcript", "memories"]
    for key in required:
        if key not in payload:
            errors.append(f"missing key: {key}")

    if "project_goal" in payload and not isinstance(payload.get("project_goal"), str):
        errors.append("project_goal must be string")

    for key in ("current_progress", "next_work"):
        if key in payload and not isinstance(payload.get(key), list):
            errors.append(f"{key} must be list[string]")
        elif key in payload:
            for i, item in enumerate(payload.get(key) or []):
                if not isinstance(item, str):
                    errors.append(f"{key}[{i}] must be string")

    if "segmented_transcript" in payload and not isinstance(payload.get("segmented_transcript"), str):
        errors.append("segmented_transcript must be string")

    memories = payload.get("memories")
    if "memories" in payload and not isinstance(memories, list):
        errors.append("memories must be list[object]")
    elif isinstance(memories, list):
        for i, mem in enumerate(memories):
            if not isinstance(mem, dict):
                errors.append(f"memories[{i}] must be object")
                continue
            for key in ("type", "title", "details"):
                value = mem.get(key)
                if not isinstance(value, str) or not value.strip():
                    errors.append(f"memories[{i}].{key} must be non-empty string")
            mem_type = mem.get("type")
            if isinstance(mem_type, str) and mem_type.strip():
                if mem_type.strip() not in ALLOWED_MEMORY_TYPES:
                    allowed = "|".join(sorted(ALLOWED_MEMORY_TYPES))
                    errors.append(
                        f"memories[{i}].type must be one of: {allowed}"
                    )

    return errors


def _normalize_extracted_payload(payload: dict[str, Any]) -> dict[str, Any]:
    def _list_of_strings(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        out: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        return out

    normalized_memories: list[dict[str, str]] = []
    for mem in payload.get("memories", []) or []:
        if not isinstance(mem, dict):
            continue
        mem_type = str(mem.get("type", "")).strip()
        title = str(mem.get("title", "")).strip()
        details = str(mem.get("details", "")).strip()
        if not mem_type or not title or not details:
            continue
        normalized_memories.append(
            {
                "type": mem_type,
                "title": title,
                "details": details,
            }
        )

    return {
        "project_goal": str(payload.get("project_goal", "")).strip(),
        "current_progress": _list_of_strings(payload.get("current_progress")),
        "next_work": _list_of_strings(payload.get("next_work")),
        "segmented_transcript": str(payload.get("segmented_transcript", "")).strip(),
        "memories": normalized_memories,
    }


async def _repair_extracted_payload(
    client: AsyncOpenAI,
    transcript: str,
    bad_output: str,
    errors: list[str],
) -> str:
    error_block = "\n- ".join(errors)
    repair_prompt = (
        "Fix the following invalid extraction JSON.\n"
        "Return JSON only and keep the same schema:\n"
        "{\n"
        '  "project_goal": "string",\n'
        '  "current_progress": ["string"],\n'
        '  "next_work": ["string"],\n'
        '  "segmented_transcript": "string",\n'
        '  "memories": [\n'
        "    {\n"
        '      "type": "Execute error|User profile|Learned knowledge|Project progress|Key decision|Other memory",\n'
        '      "title": "short title",\n'
        '      "details": "one self-contained paragraph"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Validation errors:\n"
        f"- {error_block}\n\n"
        "Original transcript:\n"
        f"{transcript}\n\n"
        "Invalid JSON output:\n"
        f"{bad_output}\n"
    )

    response = await client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": "You repair structured JSON. Output JSON only."},
            {"role": "user", "content": repair_prompt},
        ],
        max_tokens=REPAIR_MAX_TOKENS,
    )
    return (response.choices[0].message.content or "").strip()


async def _extract_with_llm(transcript: str) -> dict[str, Any]:
    client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    response = await client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SPLIT_PROMPT},
            {"role": "user", "content": transcript},
        ],
        max_tokens=EXTRACT_MAX_TOKENS,
    )

    content = (response.choices[0].message.content or "").strip()
    current = content
    parse_error: str | None = None
    validation_errors: list[str] = []

    for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
        cleaned = _strip_fence(current)
        try:
            payload = json.loads(cleaned)
        except Exception as e:
            parse_error = str(e)
            payload = None
            validation_errors = [f"json parse error: {parse_error}"]
        else:
            validation_errors = _validate_extracted_payload(payload)
            if not validation_errors:
                return _normalize_extracted_payload(payload)

        if attempt >= MAX_REPAIR_ATTEMPTS:
            break

        current = await _repair_extracted_payload(
            client=client,
            transcript=transcript,
            bad_output=cleaned,
            errors=validation_errors,
        )

    joined = "; ".join(validation_errors) if validation_errors else (parse_error or "unknown extraction error")
    raise ValueError(f"Invalid extraction output after {MAX_REPAIR_ATTEMPTS} repair attempts: {joined}")


def _merge_list_unique(dst: list[str], src: Any) -> None:
    if not isinstance(src, list):
        return
    existing = {x.strip().lower() for x in dst if isinstance(x, str)}
    for item in src:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in existing:
            continue
        existing.add(key)
        dst.append(cleaned)


async def _extract_with_llm_chunked(transcript: str) -> dict[str, Any]:
    chunks = _split_transcript(transcript)
    if not chunks:
        return {
            "project_goal": "",
            "current_progress": [],
            "next_work": [],
            "segmented_transcript": "",
            "memories": [],
        }

    merged: dict[str, Any] = {
        "project_goal": "",
        "current_progress": [],
        "next_work": [],
        "segmented_transcript": "",
        "memories": [],
    }
    segmented_parts: list[str] = []

    for chunk in chunks:
        extracted = await _extract_with_llm(chunk)
        goal = str(extracted.get("project_goal", "")).strip()
        if goal and not merged["project_goal"]:
            merged["project_goal"] = goal

        _merge_list_unique(merged["current_progress"], extracted.get("current_progress"))
        _merge_list_unique(merged["next_work"], extracted.get("next_work"))

        segmented = str(extracted.get("segmented_transcript", "")).strip()
        if segmented:
            segmented_parts.append(segmented)
        else:
            segmented_parts.append(chunk)

        memories = extracted.get("memories")
        if isinstance(memories, list):
            merged["memories"].extend(memories)

    merged["segmented_transcript"] = "\n\n[SPLIT]\n\n".join(segmented_parts)
    merged = _normalize_extracted_payload(merged)
    merge_errors = _validate_extracted_payload(merged)
    if merge_errors:
        raise ValueError(f"Merged extraction payload invalid: {'; '.join(merge_errors)}")
    return merged


def _format_list(items: list[str]) -> str:
    cleaned = [i.strip() for i in items if isinstance(i, str) and i.strip()]
    if not cleaned:
        return "- (none)"
    return "\n".join(f"- {line}" for line in cleaned[:6])


def _render_now_content(
    template_text: str,
    project_goal: str,
    current_progress: list[str],
    next_work: list[str],
    trigger: str,
    session_id: str,
    local_time: str,
) -> str:
    rendered = template_text

    replacements = {
        "{{project_goal}}": project_goal or "(not set)",
        "{{current_progress}}": _format_list(current_progress),
        "{{next_work}}": _format_list(next_work),
        "{{trigger}}": trigger,
        "{{session_id}}": session_id or "unknown",
        "{{local_time_yyyy_mm_dd_hh_mm}}": local_time,
    }

    for key, value in replacements.items():
        rendered = rendered.replace(key, value)
    return rendered


def _append_raw_memories(memories: list[dict[str, str]], when: datetime) -> str:
    raw_dir = Path(MEMORY_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)

    raw_file = raw_dir / f"{when.strftime('%Y-%m-%d')}.md"

    with raw_file.open("a", encoding="utf-8") as f:
        for mem in memories:
            f.write(
                "\n"
                f"## Memory Candidate - {mem['title']}\n"
                f"Time: {mem['time']}\n"
                f"Type: {mem['type']}\n"
                "Details:\n"
                f"{mem['details'].strip()}\n"
            )

    return str(raw_file)


async def capture_from_hook(
    trigger: str,
    session_file: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    session_id: str | None = None,
    now_file_path: str | None = None,
) -> dict[str, Any]:
    source_messages: list[dict[str, str]] = []

    if session_file:
        source_messages = _load_from_session_file(session_file)

    if not source_messages and messages:
        source_messages = _normalize_messages(messages)

    if not source_messages:
        return {
            "memory_count": 0,
            "raw_file": "",
            "now_file": now_file_path or str(Path(WORKSPACE_DIR) / "NOW.md"),
            "note": "no session messages",
        }

    transcript = _messages_to_transcript(source_messages)
    if not transcript.strip():
        return {
            "memory_count": 0,
            "raw_file": "",
            "now_file": now_file_path or str(Path(WORKSPACE_DIR) / "NOW.md"),
            "note": "empty transcript",
        }

    now_dt = datetime.now().astimezone()
    now_text = now_dt.strftime("%Y-%m-%d %H:%M")

    extracted = await _extract_with_llm_chunked(transcript)

    memories = _norm_memories(extracted.get("memories", []), now_text)
    raw_file = _append_raw_memories(memories, now_dt)

    now_path = Path(now_file_path or (Path(WORKSPACE_DIR) / "NOW.md"))

    template_text = DEFAULT_NOW_TEMPLATE_TEXT

    now_content = _render_now_content(
        template_text=template_text,
        project_goal=str(extracted.get("project_goal", "")).strip(),
        current_progress=extracted.get("current_progress", []) or [],
        next_work=extracted.get("next_work", []) or [],
        trigger=trigger,
        session_id=session_id or "",
        local_time=now_text,
    )

    now_path.parent.mkdir(parents=True, exist_ok=True)
    now_path.write_text(now_content.strip() + "\n", encoding="utf-8")

    return {
        "memory_count": len(memories),
        "raw_file": raw_file,
        "now_file": str(now_path),
    }
