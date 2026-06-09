"""
End-to-end test client for the Legal Multi-Agent System.

Sends a legal question to the Customer Agent and prints the response.
Also prints trace_id/context_id so you can follow the flow in logs (Task 5.1).
"""

from __future__ import annotations

import asyncio
import os
import sys
from uuid import uuid4

import httpx
from dotenv import load_dotenv

from a2a.client import A2AClient
from a2a.types import AgentCard, Message, MessageSendParams, Part, Role, SendMessageRequest, TextPart
from common.a2a_client import extract_text

load_dotenv()

CUSTOMER_AGENT_URL = os.getenv("CUSTOMER_AGENT_URL", "http://localhost:10100")

QUESTION = (
    "If a tech startup shares user data without consent and fails to pay taxes on overseas revenue, "
    "what are the legal and regulatory consequences?"
)


async def main() -> None:
    trace_id = str(uuid4())
    context_id = str(uuid4())

    print(f"Connecting to Customer Agent at {CUSTOMER_AGENT_URL}")
    print(f"trace_id:   {trace_id}")
    print(f"context_id: {context_id}")
    print(f"Question: {QUESTION}")
    print("-" * 60)

    async with httpx.AsyncClient(timeout=300.0) as http_client:
        card_url = f"{CUSTOMER_AGENT_URL}/.well-known/agent.json"
        try:
            card_resp = await http_client.get(card_url)
            card_resp.raise_for_status()
        except Exception as exc:
            print(f"ERROR: Could not reach Customer Agent at {card_url}")
            print(f"  {exc}")
            print("Make sure all services are running (./start_all.sh)")
            sys.exit(1)

        agent_card = AgentCard.model_validate(card_resp.json())
        print(f"Connected to agent: {agent_card.name} v{agent_card.version}")
        print("-" * 60)

        client = A2AClient(httpx_client=http_client, agent_card=agent_card)

        message = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text=QUESTION))],
            message_id=str(uuid4()),
            context_id=context_id,
            metadata={
                "trace_id": trace_id,
                "delegation_depth": 0,
            },
        )

        request = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(message=message),
        )

        print("Sending request (this may take 30-60s while agents chain)...\n")
        response = await client.send_message(request)

        result_text = extract_text(response).strip()
        if result_text:
            print("RESPONSE:")
            print("=" * 60)
            print(result_text)
            print("=" * 60)
        else:
            print("No text response received. Raw response:")
            print(response)


if __name__ == "__main__":
    asyncio.run(main())