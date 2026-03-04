"""
Token-aware chat history trimming.

Keeps as much history as possible under a conservative token budget
to avoid model context overflow without relying on hardcoded turn counts.
"""
from typing import List, Dict, Any


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Fallback heuristic: ~4 chars per token
        return max(1, int(len(text) / 4))


def trim_history_by_token_budget(
    messages: List[Dict[str, Any]],
    max_input_tokens: int = 6500,
    reserve_tokens: int = 1500,
) -> List[Dict[str, Any]]:
    """
    Trim from the front until total tokens <= max_input_tokens - reserve_tokens.
    """
    if not messages:
        return messages
    budget = max(0, max_input_tokens - reserve_tokens)
    if budget <= 0:
        return messages[-2:] if len(messages) >= 2 else messages

    # Compute total tokens
    total = 0
    token_counts = []
    for m in messages:
        t = _estimate_tokens(m.get("content", ""))
        token_counts.append(t)
        total += t

    if total <= budget:
        return messages

    # Trim oldest until within budget
    idx = 0
    while total > budget and idx < len(messages):
        total -= token_counts[idx]
        idx += 1
    return messages[idx:]
