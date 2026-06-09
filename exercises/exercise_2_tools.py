import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.llm import get_llm  # noqa: E402

LEGAL_KNOWLEDGE = [
    {
        "id": "ucc_breach",
        "keywords": ["breach", "contract", "remedies", "damages", "ucc", "hợp đồng", "vi phạm"],
        "text": (
            "UCC Article 2 remedies include expectation and consequential damages; "
            "statute of limitations is typically 4 years (UCC § 2-725)."
        ),
    },
    {
        "id": "labor_law",
        "keywords": ["lao động", "sa thải", "hợp đồng lao động", "labor", "termination"],
        "text": (
            "Theo Bộ luật Lao động Việt Nam 2019, NSDLĐ có thể đơn phương chấm dứt HĐLĐ "
            "trong một số trường hợp như NLĐ thường xuyên không hoàn thành công việc, "
            "ốm đau kéo dài, thiên tai/hỏa hoạn, hoặc đủ tuổi nghỉ hưu."
        ),
    },
]


@tool
def search_legal_knowledge(query: str) -> str:
    q = query.lower()
    for e in LEGAL_KNOWLEDGE:
        if any(kw in q for kw in e["keywords"]):
            return f"[{e['id']}] {e['text']}"
    return "Không tìm thấy thông tin liên quan."


@tool
def check_statute_of_limitations(case_type: str) -> str:
    limits = {
        "contract": "4 năm (UCC § 2-725)",
        "tort": "2-3 năm tùy bang",
        "property": "5 năm",
    }
    return limits.get(case_type.lower(), "Không xác định")


async def main() -> None:
    load_dotenv()
    llm = get_llm()

    tools = [search_legal_knowledge, check_statute_of_limitations]
    llm_tools = llm.bind_tools(tools)

    question = "Thời hiệu khởi kiện vụ vi phạm hợp đồng là bao lâu?"
    messages = [
        SystemMessage(content="Bạn là chuyên gia pháp lý. Sử dụng tools để tra cứu."),
        HumanMessage(content=question),
    ]

    first = await llm_tools.ainvoke(messages)
    messages.append(first)

    tool_map = {t.name: t for t in tools}
    for tc in first.tool_calls or []:
        fn = tool_map.get(tc["name"])
        if not fn:
            continue
        out = await fn.ainvoke(tc["args"])
        messages.append(ToolMessage(content=out, tool_call_id=tc["id"]))

    final = await llm_tools.ainvoke(messages)
    print(final.content)


if __name__ == "__main__":
    asyncio.run(main())