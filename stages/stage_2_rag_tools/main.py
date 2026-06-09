import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.llm import get_llm


LEGAL_KNOWLEDGE = [
    {
        "id": "ucc_breach",
        "keywords": ["breach", "contract", "remedies", "damages", "ucc", "hợp đồng", "vi phạm"],
        "text": (
            "UCC Article 2 remedies: expectation damages, consequential damages, "
            "specific performance. Statute of limitations: 4 years (UCC § 2-725)."
        ),
    },
    {
        "id": "nda_trade_secret",
        "keywords": ["nda", "non-disclosure", "confidential", "trade secret", "bí mật"],
        "text": (
            "NDA breach: contractual remedies + DTSA (18 U.S.C. § 1836): injunction, "
            "actual damages, up to 2x exemplary damages for willful acts, attorney fees."
        ),
    },
    {
        "id": "labor_law",
        "keywords": ["lao động", "sa thải", "hợp đồng lao động", "labor", "termination"],
        "text": (
            "Bộ luật Lao động 2019: NSDLĐ được đơn phương chấm dứt HĐLĐ trong các trường hợp "
            "NLĐ thường xuyên không hoàn thành công việc, ốm đau kéo dài, thiên tai, hoặc đủ tuổi nghỉ hưu."
        ),
    },
]


@tool
def search_legal_database(query: str) -> str:
    """Tìm kiếm thông tin pháp lý liên quan đến query."""
    q = query.lower()
    scored = []
    for e in LEGAL_KNOWLEDGE:
        overlap = sum(1 for kw in e["keywords"] if kw in q)
        if overlap:
            scored.append((overlap, e))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [e for _, e in scored[:2]]
    if not top:
        return "No relevant legal sources found."
    return "\n\n".join(f"[{e['id']}] {e['text']}" for e in top)


@tool
def calculate_damages(breach_type: str, contract_value: float) -> str:
    """Tính toán thiệt hại ước tính dựa trên loại vi phạm và giá trị hợp đồng."""
    t = breach_type.lower()
    if "willful" in t or "intentional" in t:
        m, label = 2.0, "Willful (2x)"
    elif "negligent" in t:
        m, label = 1.0, "Negligent (1x)"
    else:
        m, label = 1.5, "Standard (1.5x)"
    base = contract_value * m
    fees = contract_value * 0.15
    total = base + fees
    return (
        f"Breach type: {label}\n"
        f"Contract value: ${contract_value:,.2f}\n"
        f"Estimated damages: ${base:,.2f}\n"
        f"Attorney fees: ${fees:,.2f}\n"
        f"Total exposure: ${total:,.2f}"
    )


@tool
def check_statute_of_limitations(case_type: str) -> str:
    """Kiểm tra thời hiệu khởi kiện theo loại vụ việc."""
    limits = {
        "contract": "4 năm (UCC § 2-725)",
        "tort": "2-3 năm tùy bang",
        "property": "5 năm",
        "labor": "1 năm kể từ ngày vi phạm",
    }
    return limits.get(case_type.lower(), "Không xác định")


TOOLS = [search_legal_database, calculate_damages, check_statute_of_limitations]


async def main() -> None:
    load_dotenv()
    llm = get_llm()
    llm_tools = llm.bind_tools(TOOLS)
    tool_map = {t.name: t for t in TOOLS}

    question = (
        "For a breach of contract dispute with a $100,000 contract value: "
        "what remedies apply and what is the statute of limitations? "
        "Also estimate damages for a standard breach."
    )

    messages = [
        SystemMessage(content="Bạn là luật sư chuyên nghiệp. Sử dụng tools khi cần. Trả lời rõ ràng, có cấu trúc."),
        HumanMessage(content=question),
    ]

    first = await llm_tools.ainvoke(messages)
    messages.append(first)

    for tc in first.tool_calls or []:
        tool_func = tool_map.get(tc["name"])
        if tool_func:
            result = await tool_func.ainvoke(tc["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    final = await llm_tools.ainvoke(messages)
    print("\n" + "="*60)
    print(final.content)
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())