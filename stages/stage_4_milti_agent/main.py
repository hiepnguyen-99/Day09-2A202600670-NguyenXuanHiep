"""Stage 4: Multi-Agent System (In-Process)

Multiple specialised agents collaborate on a complex legal question.
This mirrors Stage 5's architecture (law_agent/graph.py) but runs
entirely in-process — no HTTP, no A2A protocol, no separate servers.

Graph (main flow):
  law_agent -> check_routing -> parallel [tax_agent, compliance_agent, privacy_agent] -> aggregate_results -> END
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from common.llm import get_llm

# ---------------------------------------------------------------------------
# Tools for specialist sub-agents (toy knowledge bases)
# ---------------------------------------------------------------------------


@tool
def search_tax_law(query: str) -> str:
    """Search tax law knowledge base for relevant statutes and penalties.

    Args:
        query: Natural language query about tax law.
    """
    knowledge = [
        (
            ["tax", "evasion", "fraud", "irs"],
            "Tax evasion (26 U.S.C. § 7201): felony, up to $250K fine and 5 years prison. "
            "Civil fraud penalty: 75% of underpayment (IRC § 6663). Failure to file: up to "
            "$25K fine and 1 year prison.",
        ),
        (
            ["offshore", "overseas", "foreign", "fbar", "fatca"],
            "FBAR penalties: up to $100K or 50% of account balance per violation. "
            "FATCA non-compliance: 30% withholding on US-source payments. "
            "Willful violations may trigger criminal prosecution.",
        ),
        (
            ["transfer", "pricing", "corporate"],
            "Transfer pricing violations (IRC § 482): IRS can reallocate income between "
            "related entities. Penalties: 20-40% of underpayment for substantial/gross "
            "valuation misstatements.",
        ),
    ]

    query_lower = query.lower()
    results: list[str] = []
    for keywords, text in knowledge:
        if any(kw in query_lower for kw in keywords):
            results.append(text)

    return "\n\n".join(results) if results else "No specific tax law matches found."


@tool
def search_compliance_law(query: str) -> str:
    """Search regulatory compliance knowledge base for applicable frameworks.

    Args:
        query: Natural language query about regulatory compliance.
    """
    knowledge = [
        (
            ["sox", "sarbanes", "financial", "sec", "reporting"],
            "SOX § 906: false certification — up to $5M fine, 20 years prison. "
            "§ 802: record destruction — up to 20 years. § 1107: whistleblower "
            "retaliation — up to 10 years. SEC officer/director bars.",
        ),
        (
            ["fcpa", "bribery", "corruption", "foreign"],
            "FCPA anti-bribery: up to $250K fine per violation (individuals), "
            "$2M (corporations). Criminal penalties: up to 5 years prison. "
            "Books and records provisions apply to SEC-reporting companies.",
        ),
        (
            ["aml", "bsa", "money laundering", "sanctions"],
            "BSA/AML failures can trigger civil money penalties, monitorships, and "
            "potential criminal exposure depending on willfulness and concealment.",
        ),
    ]

    query_lower = query.lower()
    results: list[str] = []
    for keywords, text in knowledge:
        if any(kw in query_lower for kw in keywords):
            results.append(text)

    return "\n\n".join(results) if results else "No specific compliance matches found."


@tool
def search_privacy_law(query: str) -> str:
    """Search privacy / data protection knowledge base.

    Args:
        query: Natural language query about privacy/data protection.
    """
    knowledge = [
        (
            ["gdpr", "eu", "personal data", "controller", "processor"],
            "GDPR: requires a lawful basis (e.g., consent/contract/legitimate interests), "
            "transparency notices, data subject rights, DPAs oversight, and breach notification. "
            "Administrative fines can be up to 4% of global annual turnover or EUR 20M (whichever is higher).",
        ),
        (
            ["ccpa", "cpra", "california", "sale", "share", "opt-out"],
            "CCPA/CPRA: obligations around notice, opt-out of sale/sharing, and contractual controls for service "
            "providers/contractors. Civil penalties can apply; private right of action exists for certain breaches "
            "(statutory damages range commonly cited as $100–$750 per consumer per incident).",
        ),
        (
            ["ftc", "deceptive", "unfair", "consent"],
            "FTC Act Section 5: the FTC can pursue unfair/deceptive practices (e.g., misrepresenting privacy practices "
            "or failing to provide promised safeguards). Remedies often include injunctive orders, monitoring, and "
            "sometimes monetary relief depending on the legal theory and posture.",
        ),
    ]

    query_lower = query.lower()
    results: list[str] = []
    for keywords, text in knowledge:
        if any(kw in query_lower for kw in keywords):
            results.append(text)

    return "\n\n".join(results) if results else "No specific privacy law matches found."


# ---------------------------------------------------------------------------
# State definition (matches CODELAB Stage 4 expectations: State(TypedDict))
# ---------------------------------------------------------------------------

from typing import Annotated, TypedDict

from langgraph.constants import Send
from langgraph.graph import END, StateGraph


def _last_wins(a: str, b: str) -> str:
    """Reducer: keep the most recently written value."""
    return b if b else a


class State(TypedDict):
    question: str

    # Routing flags (set by check_routing)
    needs_tax: bool
    needs_compliance: bool
    needs_privacy: bool

    # Agent outputs (parallel writes are safe with reducers)
    law_analysis: Annotated[str, _last_wins]
    tax_analysis: Annotated[str, _last_wins]
    compliance_analysis: Annotated[str, _last_wins]
    privacy_analysis: Annotated[str, _last_wins]

    final_answer: str


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------


async def law_agent(state: State) -> dict:
    """Lead attorney: general legal analysis (contracts/torts/business law)."""
    print("\n  [Node: law_agent] Lead attorney analysing legal aspects...")

    llm = get_llm()
    messages = [
        SystemMessage(
            content=(
                "You are a senior corporate litigation attorney specialising in contract law, "
                "tort law, and general business law. Analyse the legal aspects of the question "
                "thoroughly. Keep your analysis under 200 words."
            )
        ),
        HumanMessage(content=state["question"]),
    ]
    result = await llm.ainvoke(messages)

    print(f"  [Node: law_agent] Done ({len(result.content)} chars)")
    return {"law_analysis": result.content}


def check_routing(state: State) -> dict:
    """Conditional routing (CODELAB Exercise 4.2 style): decide which specialists to call.

    We keep routing deterministic (keyword-based) to make the demo easy to understand and test.
    """
    print("\n  [Node: check_routing] Determining which specialists are needed...")

    q = state["question"].lower()

    needs_tax = any(
        kw in q
        for kw in [
            "tax",
            "irs",
            "evasion",
            "offshore",
            "overseas",
            "foreign income",
            "fbar",
            "fatca",
        ]
    )

    needs_compliance = any(
        kw in q
        for kw in [
            "compliance",
            "sec",
            "sox",
            "sarbanes",
            "reporting",
            "aml",
            "bsa",
            "fcpa",
            "regulation",
            "regulatory",
        ]
    )

    # Exercise 4: privacy_agent + routing via privacy/data keywords
    needs_privacy = any(
        kw in q
        for kw in [
            "data",
            "privacy",
            "gdpr",
            "ccpa",
            "cpra",
            "consent",
            "personal data",
            "user data",
            "data breach",
            "breach notification",
        ]
    )

    print(
        "  [Node: check_routing] "
        f"needs_tax={needs_tax}, needs_compliance={needs_compliance}, needs_privacy={needs_privacy}"
    )

    return {
        "needs_tax": needs_tax,
        "needs_compliance": needs_compliance,
        "needs_privacy": needs_privacy,
    }


def route_to_agents(state: State) -> list[Send]:
    """Routing function: dispatch parallel Send objects based on routing flags."""
    sends: list[Send] = []

    if state.get("needs_tax"):
        sends.append(Send("tax_agent", state))

    if state.get("needs_compliance"):
        sends.append(Send("compliance_agent", state))

    if state.get("needs_privacy"):
        sends.append(Send("privacy_agent", state))

    # If no specialists are needed, go straight to aggregation.
    if not sends:
        sends.append(Send("aggregate_results", state))

    return sends


async def tax_agent(state: State) -> dict:
    """Tax specialist sub-agent (inline ReAct agent grounded by search_tax_law)."""
    from langgraph.prebuilt import create_react_agent

    print("\n  [Node: tax_agent] Tax specialist agent starting...")

    tax_prompt = (
        "You are a specialist tax attorney and CPA with expertise in corporate tax law, "
        "tax evasion vs. avoidance, IRS enforcement, penalties under IRC §§ 6651/6662/6663, "
        "FBAR/FATCA requirements, and tax fraud statutes. "
        "Use the search_tax_law tool to ground your analysis. Keep your response under 200 words."
    )

    llm = get_llm()
    agent = create_react_agent(model=llm, tools=[search_tax_law], prompt=tax_prompt)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": state["question"]}]})

    final_msg = result["messages"][-1].content
    print(f"  [Node: tax_agent] Done ({len(final_msg)} chars)")
    return {"tax_analysis": final_msg}


async def compliance_agent(state: State) -> dict:
    """Compliance specialist sub-agent (inline ReAct agent grounded by search_compliance_law)."""
    from langgraph.prebuilt import create_react_agent

    print("\n  [Node: compliance_agent] Compliance specialist agent starting...")

    compliance_prompt = (
        "You are a senior regulatory compliance officer with expertise in SEC enforcement, "
        "SOX compliance, FCPA, AML/BSA, and corporate governance. "
        "Use the search_compliance_law tool to ground your analysis. Keep your response under 200 words."
    )

    llm = get_llm()
    agent = create_react_agent(model=llm, tools=[search_compliance_law], prompt=compliance_prompt)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": state["question"]}]})

    final_msg = result["messages"][-1].content
    print(f"  [Node: compliance_agent] Done ({len(final_msg)} chars)")
    return {"compliance_analysis": final_msg}


async def privacy_agent(state: State) -> dict:
    """Privacy specialist sub-agent (Exercise 4.1): GDPR/CCPA/FTC privacy issues."""
    from langgraph.prebuilt import create_react_agent

    print("\n  [Node: privacy_agent] Privacy specialist agent starting...")

    privacy_prompt = (
        "You are a privacy and data protection lawyer. Focus on GDPR/CCPA/CPRA, consent/lawful basis, "
        "data sharing, breach notification, regulator actions, class action exposure, and remediation steps. "
        "Use the search_privacy_law tool to ground your analysis. Keep your response under 200 words."
    )

    llm = get_llm()
    agent = create_react_agent(model=llm, tools=[search_privacy_law], prompt=privacy_prompt)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": state["question"]}]})

    final_msg = result["messages"][-1].content
    print(f"  [Node: privacy_agent] Done ({len(final_msg)} chars)")
    return {"privacy_analysis": final_msg}


async def aggregate_results(state: State) -> dict:
    """Combine all analyses into a final comprehensive answer."""
    print("\n  [Node: aggregate_results] Combining all specialist analyses...")

    llm = get_llm()

    sections: list[str] = []
    if state.get("law_analysis"):
        sections.append(f"## Legal Analysis\n{state['law_analysis']}")
    if state.get("tax_analysis"):
        sections.append(f"## Tax Analysis\n{state['tax_analysis']}")
    if state.get("compliance_analysis"):
        sections.append(f"## Compliance Analysis\n{state['compliance_analysis']}")
    if state.get("privacy_analysis"):
        sections.append(f"## Privacy / Data Protection Analysis\n{state['privacy_analysis']}")

    combined = "\n\n---\n\n".join(sections) if sections else "No specialist analyses were produced."

    messages = [
        SystemMessage(
            content=(
                "You are a senior legal counsel synthesising specialist analyses into a "
                "comprehensive, well-structured response. Combine the following analyses "
                "into a cohesive answer with clear sections. Avoid redundancy. "
                "Keep your response under 500 words."
            )
        ),
        HumanMessage(content=combined),
    ]
    result = await llm.ainvoke(messages)

    print(f"  [Node: aggregate_results] Done ({len(result.content)} chars)")
    return {"final_answer": result.content}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def create_graph():
    """Build and compile the multi-agent StateGraph."""
    graph = StateGraph(State)

    graph.add_node("law_agent", law_agent)
    graph.add_node("check_routing", check_routing)
    graph.add_node("tax_agent", tax_agent)
    graph.add_node("compliance_agent", compliance_agent)
    graph.add_node("privacy_agent", privacy_agent)
    graph.add_node("aggregate_results", aggregate_results)

    graph.set_entry_point("law_agent")
    graph.add_edge("law_agent", "check_routing")

    graph.add_conditional_edges(
        "check_routing",
        route_to_agents,
        ["tax_agent", "compliance_agent", "privacy_agent", "aggregate_results"],
    )

    graph.add_edge("tax_agent", "aggregate_results")
    graph.add_edge("compliance_agent", "aggregate_results")
    graph.add_edge("privacy_agent", "aggregate_results")
    graph.add_edge("aggregate_results", END)

    return graph.compile()


QUESTION = (
    "A tech startup with $5M revenue was caught sharing user data without consent "
    "and failed to pay taxes on overseas revenue. What are all the legal consequences?"
)


async def main():
    print("=" * 70)
    print("STAGE 4: Multi-Agent System (In-Process)")
    print("=" * 70)
    print()
    print("[How it works]")
    print("  1. Lead attorney agent analyses the question")
    print("  2. Router decides which specialist agents are needed")
    print("  3. Specialists run IN PARALLEL (LangGraph Send API)")
    print("  4. Aggregator combines all analyses into a final answer")
    print()
    print("[Graph topology]")
    print("  law_agent -> check_routing -> [tax + compliance + privacy] -> aggregate_results -> END")
    print()
    print(f"Question: {QUESTION}")
    print("-" * 70)

    graph = create_graph()

    result = await graph.ainvoke(
        {
            "question": QUESTION,
            "needs_tax": False,
            "needs_compliance": False,
            "needs_privacy": False,
            "law_analysis": "",
            "tax_analysis": "",
            "compliance_analysis": "",
            "privacy_analysis": "",
            "final_answer": "",
        }
    )

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(result["final_answer"])

    print()
    print("-" * 70)
    print("[Improvements over Stage 3]")
    print("  + Specialisation: each agent has domain-specific expertise")
    print("  + Parallel execution: specialists run concurrently")
    print("  + Better quality: specialist prompts produce deeper analysis")
    print("  + Structured flow: explicit graph topology with routing logic")
    print()
    print("[Stage 4 (Monolith) vs Stage 5 (Distributed A2A)]")
    print("  +---------------------------+-------------------------------+")
    print("  | Stage 4 (In-Process)      | Stage 5 (A2A Protocol)        |")
    print("  +---------------------------+-------------------------------+")
    print("  | Single process            | Multiple services (ports)     |")
    print("  | Direct function calls     | HTTP-based A2A protocol       |")
    print("  | Shared memory             | Message passing               |")
    print("  | Simple deployment         | Independent scaling           |")
    print("  | Tight coupling            | Loose coupling                |")
    print("  | Easy to debug             | Service discovery + registry  |")
    print("  | Good for small teams      | Good for large organisations  |")
    print("  +---------------------------+-------------------------------+")
    print()
    print("Stage 5 takes this same topology and deploys each agent as an independent A2A service.")
    print("Run it with:")
    print("  ./start_all.sh && python test_client.py")
    print("=" * 70)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())