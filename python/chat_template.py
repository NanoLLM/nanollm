"""
Shared User/Assistant chat template for training and inference.

All chat fine-tune data and interactive prompts should use this format:

    User: <user message>
    Assistant: <assistant message>

    User: <follow-up>
    Assistant:
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

USER_PREFIX = "User: "
ASSISTANT_PREFIX = "Assistant: "
# ByteLevel BPE merges the training-space with the first answer token. Supplying
# a standalone trailing-space token at inference creates an unseen prompt suffix.
ASSISTANT_OPEN = "Assistant:"
EOS_TOKEN = "<EOS>"
RECORD_SEPARATOR = "\n\n"

# Stop generation when the model starts a new dialogue turn mid-stream.
_TURN_BOUNDARY_RE = re.compile(
    r"\n\nUser:|\nUser:|\n\nAssistant:|\nAssistant:|\n\nSystem:|\nSystem:"
)


def find_turn_boundary(text: str) -> Optional[int]:
    """Return the index where a spurious next turn begins, or None."""
    match = _TURN_BOUNDARY_RE.search(text)
    if match is None:
        return None
    return match.start() if match.start() > 0 else None


def trim_assistant_reply(text: str) -> str:
    """Keep only the first assistant turn; drop hallucinated User/Assistant continuations."""
    text = text.strip()
    if not text:
        return text
    boundary = find_turn_boundary(text)
    if boundary is not None:
        text = text[:boundary]
    return text.strip()


def format_turn(user: str, assistant: str) -> str:
    """Format one completed user/assistant exchange."""
    user = " ".join(user.split())
    assistant = "\n".join(
        " ".join(line.split())
        for line in assistant.splitlines()
        if line.strip()
    )
    while assistant.endswith(EOS_TOKEN):
        assistant = assistant[: -len(EOS_TOKEN)].rstrip()
    if not user or not assistant:
        return ""
    return f"{USER_PREFIX}{user}\n{ASSISTANT_PREFIX}{assistant}{EOS_TOKEN}"


def format_prompt(
    user_message: str,
    history: Optional[List[Tuple[str, str]]] = None,
    system: Optional[str] = None,
) -> str:
    """Build a multi-turn prompt ending with an open Assistant turn."""
    segments: List[str] = []
    if system:
        segments.append(f"System: {' '.join(system.split())}")
    for user, assistant in history or []:
        segments.append(f"{USER_PREFIX}{user.strip()}")
        segments.append(f"{ASSISTANT_PREFIX}{assistant.strip()}")
    segments.append(f"{USER_PREFIX}{user_message.strip()}")
    segments.append(ASSISTANT_OPEN)
    return "\n".join(segments)


def _clean_block(block: str) -> str:
    return block.replace("\r\n", "\n").strip()


def _standardize_existing(block: str) -> Optional[str]:
    """Normalize blocks that already contain User:/Assistant: markers."""
    if USER_PREFIX not in block and "User:" not in block:
        return None

    text = block.replace("User:", USER_PREFIX).replace("Assistant:", ASSISTANT_PREFIX)
    lines = text.split("\n")
    turns: List[str] = []
    current_user: Optional[str] = None
    current_assistant: Optional[str] = None

    def flush() -> None:
        nonlocal current_user, current_assistant
        if current_user and current_assistant:
            turn = format_turn(current_user, current_assistant)
            if turn:
                turns.append(turn)
        current_user = None
        current_assistant = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(USER_PREFIX):
            flush()
            current_user = line[len(USER_PREFIX) :].strip()
            continue
        if line.startswith(ASSISTANT_PREFIX):
            if current_assistant is not None:
                flush()
            current_assistant = line[len(ASSISTANT_PREFIX) :].strip()
            continue
        if current_assistant is not None:
            current_assistant = f"{current_assistant}\n{line}".strip()
        elif current_user is None:
            current_user = line
        else:
            current_user = f"{current_user} {line}".strip()

    flush()
    if not turns:
        return None
    return RECORD_SEPARATOR.join(turns)


def _split_question_answer(block: str) -> Optional[str]:
    """Parse 'Question? Answer' style instruction rows."""
    q_idx = block.find("?")
    if q_idx < 0:
        return None
    user = block[: q_idx + 1].strip()
    assistant = block[q_idx + 1 :].strip()
    if len(user) < 8 or len(assistant) < 8:
        return None
    if user.count("\n") > 2:
        return None
    return format_turn(user, assistant)


def _split_instruction_response(block: str) -> Optional[str]:
    """Parse 'Instruction. Response' rows common in Alpaca-style data."""
    if "?" in block[:120]:
        return None
    match = re.match(r"^(.{12,320}?\.)\s+(.+)$", block, flags=re.DOTALL)
    if not match:
        return None
    user = match.group(1).strip()
    assistant = match.group(2).strip()
    if len(assistant) < 8:
        return None
    if assistant.startswith(USER_PREFIX) or user.startswith(ASSISTANT_PREFIX):
        return None
    return format_turn(user, assistant)


def normalize_record(block: str) -> Optional[str]:
    """Convert a raw chat dataset paragraph into canonical User/Assistant text."""
    block = _clean_block(block)
    if not block or len(block) < 16:
        return None

    existing = _standardize_existing(block)
    if existing:
        return existing

    qa = _split_question_answer(block)
    if qa:
        return qa

    instruction = _split_instruction_response(block)
    if instruction:
        return instruction

    return None


def assistant_char_spans(text: str) -> List[Tuple[int, int]]:
    """Return character spans of assistant reply content (after each Assistant: prefix)."""
    spans: List[Tuple[int, int]] = []
    search_from = 0
    while True:
        idx = text.find(ASSISTANT_PREFIX, search_from)
        if idx < 0:
            break
        start = idx + len(ASSISTANT_PREFIX)
        end = len(text)
        for marker in ("\n\nUser:", "\nUser:", "\n\nAssistant:", "\nAssistant:", RECORD_SEPARATOR):
            pos = text.find(marker, start)
            if pos >= 0:
                end = min(end, pos)
        if end > start:
            spans.append((start, end))
        search_from = start
    return spans


def build_assistant_token_mask(text: str, tokenizer) -> List[int]:
    """
    Return 0/1 per token: 1 means the token is assistant content and may be a loss target.

    Uses tokenizer byte offsets so masking stays aligned with BPE tokenization.
    """
    raw = getattr(tokenizer, "tokenizer", None)
    if raw is None:
        raise ValueError("Tokenizer must expose a HuggingFace tokenizers backend for masking")
    encoding = raw.encode(text)
    if not encoding.ids:
        return []

    spans = assistant_char_spans(text)
    if not spans:
        return [0] * len(encoding.ids)

    def char_in_assistant(char_idx: int) -> bool:
        for span_start, span_end in spans:
            if span_start <= char_idx < span_end:
                return True
        return False

    mask: List[int] = []
    for start, end in encoding.offsets:
        if end <= start:
            mask.append(0)
            continue
        train = any(char_in_assistant(char_idx) for char_idx in range(start, end))
        mask.append(1 if train else 0)
    return mask


def normalize_chat_corpus(text: str) -> Tuple[str, dict]:
    """Normalize an entire chat corpus; return formatted text and stats."""
    blocks = re.split(r"\n\s*\n", text)
    formatted_turns: List[str] = []
    stats = {
        "blocks_in": 0,
        "blocks_out": 0,
        "turns_out": 0,
        "skipped": 0,
    }

    for block in blocks:
        block = _clean_block(block)
        if not block:
            continue
        stats["blocks_in"] += 1
        normalized = normalize_record(block)
        if not normalized:
            stats["skipped"] += 1
            continue
        stats["blocks_out"] += 1
        turns = [t for t in normalized.split(RECORD_SEPARATOR) if t.strip()]
        stats["turns_out"] += len(turns)
        formatted_turns.extend(turns)

    output = RECORD_SEPARATOR.join(formatted_turns)
    if output:
        output += RECORD_SEPARATOR
    return output, stats


# ---------------------------------------------------------------------------
# Instruction-following template with optional system prompt.
# ---------------------------------------------------------------------------

SYSTEM_PREFIX = "System: "


def format_instruction_turn(
    system: Optional[str],
    user: str,
    assistant: str,
) -> str:
    """Format one instruction turn with optional system prompt.

    Produces text like:
        System: <system prompt>
        User: <instruction>
        Assistant: <response><EOS>
    """
    segments: List[str] = []
    if system:
        segments.append(f"{SYSTEM_PREFIX}{' '.join(system.split())}")
    segments.append(f"{USER_PREFIX}{' '.join(user.split())}")
    assistant_clean = "\n".join(
        " ".join(line.split())
        for line in assistant.splitlines()
        if line.strip()
    )
    while assistant_clean.endswith(EOS_TOKEN):
        assistant_clean = assistant_clean[: -len(EOS_TOKEN)].rstrip()
    segments.append(f"{ASSISTANT_PREFIX}{assistant_clean}{EOS_TOKEN}")
    return "\n".join(segments)


def format_instruction_corpus(
    records: List[Tuple[Optional[str], str, str]],
) -> Tuple[str, dict]:
    """Format a list of (system, user, assistant) triples.

    Returns (formatted_text, stats_dict).
    """
    formatted_turns: List[str] = []
    stats: dict = {"total": len(records), "skipped": 0, "formatted": 0}

    for system, user, assistant in records:
        if not user or not assistant:
            stats["skipped"] += 1
            continue
        turn = format_instruction_turn(system, user, assistant)
        if turn:
            formatted_turns.append(turn)
            stats["formatted"] += 1

    output = RECORD_SEPARATOR.join(formatted_turns)
    if output:
        output += RECORD_SEPARATOR
    return output, stats


def format_prompt_with_system(
    user_message: str,
    history: Optional[List[Tuple[str, str]]] = None,
    system: Optional[str] = None,
) -> str:
    """Build a multi-turn prompt with system instruction, ending in open Assistant turn.

    This is an alias for format_prompt with explicit system support for instruction
    following workflows.
    """
    return format_prompt(user_message, history=history, system=system)
