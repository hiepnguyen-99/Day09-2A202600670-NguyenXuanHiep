"""
A2A delegation helper.

Provides:
- delegate(endpoint, question, context_id, trace_id, depth)
- extract_text(response): extract text from an A2A response object
"""

from __future__ import annotations

import logging
from uuid import uuid4

import httpx
from a2a.client import A2AClient
from a2a.types import (
    AgentCard,
    Message,
    MessageSendParams,
    Part,
    Role,
    SendMessageRequest,
    TextPart,
)

logger = logging.getLogger(__name__)


async def delegate(
    endpoint: str,
    question: str,
    context_id: str,
    trace_id: str,
    depth: int,
) -> str:
    """
    Send a question to an A2A agent and return the text response.

    Args:
        endpoint: Base URL of the target agent (e.g. "http://localhost:10101").
        question: The question to ask.
        context_id: Current A2A context_id to propagate.
        trace_id: Trace ID propagated throughout the whole request flow.
        depth: Delegation depth (used to enforce MAX_DELEGATION_DEPTH).

    Returns:
        The agent's text response, or an empty string if none could be extracted.
    """
    async with httpx.AsyncClient(timeout=300.0) as http_client:
        # Fetch agent card
        card_url = f"{endpoint}/.well-known/agent.json"
        card_resp = await http_client.get(card_url)
        card_resp.raise_for_status()
        agent_card = AgentCard.model_validate(card_resp.json())

        # Legacy A2AClient (simple send_message)
        client = A2AClient(httpx_client=http_client, agent_card=agent_card)

        # Build message with trace metadata
        message = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text=question))],
            message_id=str(uuid4()),
            context_id=context_id,
            metadata={
                "trace_id": trace_id,
                "context_id": context_id,
                "delegation_depth": depth,
            },
        )

        request = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(message=message),
        )

        logger.debug("Delegating to %s (depth=%d, trace=%s)", endpoint, depth, trace_id)
        response = await client.send_message(request)

        return _extract_text(response)


def extract_text(response: object) -> str:
    """Public wrapper for response text extraction (useful for test_client.py)."""
    return _extract_text(response)


def _extract_text(response: object) -> str:
    """Walk the response tree and collect all TextPart.text values."""
    # Unwrap RootModel if present
    if hasattr(response, "root"):
        response = response.root  # type: ignore[assignment]

    result = getattr(response, "result", None)
    if result is None:
        return ""

    # Task: text lives in artifacts
    artifacts = getattr(result, "artifacts", None)
    if artifacts:
        text = ""
        for artifact in artifacts:
            for part in getattr(artifact, "parts", []) or []:
                text += _part_text(part)
        if text:
            return text

    # Message: text lives in parts directly
    parts = getattr(result, "parts", None)
    if parts:
        text = ""
        for part in parts:
            text += _part_text(part)
        if text:
            return text

    # Fallback: task history
    history = getattr(result, "history", None)
    if history:
        text = ""
        for msg in history:
            for part in getattr(msg, "parts", []) or []:
                text += _part_text(part)
        return text

    return ""


def _part_text(part: object) -> str:
    """Extract text from Part(root=TextPart) or raw TextPart."""
    inner = getattr(part, "root", part)
    return getattr(inner, "text", "") or ""