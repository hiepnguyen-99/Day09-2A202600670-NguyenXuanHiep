import asyncio
import os
import sys
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.llm import get_llm  # noqa: E402


def _last_wins(left: str | None, right: str | None) -> str:
    return right if right is not None else (left or "")


class State(TypedDict):
    question: str
    law_analysis: Annotated[str, _last_wins]
    tax_analysis: Annotated[str, _last_wins]
    compliance_analysis: Annotated[str, _last_wins]
    privacy_analysis: Annotated[str, _last_wins]
    final_response: str


def law_agent(state: State) -> dict:
    llm = get_llm()
    prompt = (
        "Bạn là chuyên gia pháp lý.\n"
        f"Phân tích câu hỏi sau: {state['question']}\n"
        "Tập trung: hợp đồng, trách nhiệm dân sự, quyền và nghĩa vụ."
    )
    res = llm.invoke([HumanMessage(content=prompt)])
    return {"law_analysis": res.content}


def check_routing(state: State) -> list[Send]:
    q = state["question"].lower()
    tasks: list[Send] = []

    if any(kw in q for kw in ["tax", "irs", "thuế", "fbar", "fatca"]):
        tasks.append(Send("tax_agent", state))

    if any(kw in q for kw in ["compliance", "sec", "regulation", "sox", "fCPA".lower()]):
        tasks.append(Send("compliance_agent", state))

    if any(kw in q for kw in ["data", "privacy", "gdpr", "dữ liệu", "rò rỉ", "breach"]):
        tasks.append(Send("privacy_agent", state))

    return tasks if tasks else [Send("aggregate_results", state)]


def tax_agent(state: State) -> dict:
    llm = get_llm()
    prompt = (
        "Bạn là chuyên gia thuế.\n"
        f"Câu hỏi: {state['question']}\n"
        f"Phân tích pháp lý: {state.get('law_analysis', 'N/A')}\n"
        "Tập trung: IRS, tax evasion, penalties, FBAR, FATCA."
    )
    res = llm.invoke([HumanMessage(content=prompt)])
    return {"tax_analysis": res.content}


def compliance_agent(state: State) -> dict:
    llm = get_llm()
    prompt = (
        "Bạn là chuyên gia compliance.\n"
        f"Câu hỏi: {state['question']}\n"
        f"Phân tích pháp lý: {state.get('law_analysis', 'N/A')}\n"
        "Tập trung: SEC, SOX, FCPA, AML, regulatory violations."
    )
    res = llm.invoke([HumanMessage(content=prompt)])
    return {"compliance_analysis": res.content}


def privacy_agent(state: State) -> dict:
    llm = get_llm()
    prompt = (
        "Bạn là chuyên gia GDPR/privacy.\n"
        f"Câu hỏi gốc: {state['question']}\n"
        f"Phân tích pháp lý: {state.get('law_analysis', 'N/A')}\n"
        "Tập trung: GDPR, bảo vệ dữ liệu cá nhân, data breach, nghĩa vụ thông báo, chế tài."
    )
    res = llm.invoke([HumanMessage(content=prompt)])
    return {"privacy_analysis": res.content}


def aggregate_results(state: State) -> dict:
    llm = get_llm()
    sections = []
    if state.get("law_analysis"):
        sections.append(f"PHÂN TÍCH PHÁP LÝ:\n{state['law_analysis']}")
    if state.get("tax_analysis"):
        sections.append(f"PHÂN TÍCH THUẾ:\n{state['tax_analysis']}")
    if state.get("compliance_analysis"):
        sections.append(f"PHÂN TÍCH TUÂN THỦ:\n{state['compliance_analysis']}")
    if state.get("privacy_analysis"):
        sections.append(f"PHÂN TÍCH PRIVACY/GDPR:\n{state['privacy_analysis']}")

    combined = "\n\n".join(sections)
    prompt = (
        "Tổng hợp các phân tích sau thành báo cáo pháp lý ngắn gọn, có cấu trúc rõ ràng.\n\n"
        f"{combined}\n\nCâu hỏi gốc: {state['question']}"
    )
    res = llm.invoke([HumanMessage(content=prompt)])
    return {"final_response": res.content}


def build_graph():
    g = StateGraph(State)
    g.add_node("law_agent", law_agent)
    g.add_node("check_routing", check_routing)
    g.add_node("tax_agent", tax_agent)
    g.add_node("compliance_agent", compliance_agent)
    g.add_node("privacy_agent", privacy_agent)
    g.add_node("aggregate_results", aggregate_results)

    g.add_edge(START, "law_agent")
    g.add_edge("law_agent", "check_routing")
    g.add_conditional_edges("check_routing", lambda x: x)
    g.add_edge("tax_agent", "aggregate_results")
    g.add_edge("compliance_agent", "aggregate_results")
    g.add_edge("privacy_agent", "aggregate_results")
    g.add_edge("aggregate_results", END)
    return g.compile()


async def main() -> None:
    load_dotenv()
    question = "Nếu công ty bị rò rỉ dữ liệu khách hàng và có dấu hiệu trốn thuế, hậu quả pháp lý là gì?"
    graph = build_graph()
    out = await graph.ainvoke(
        {
            "question": question,
            "law_analysis": "",
            "tax_analysis": "",
            "compliance_analysis": "",
            "privacy_analysis": "",
            "final_response": "",
        }
    )
    print(out["final_response"])


if __name__ == "__main__":
    asyncio.run(main())