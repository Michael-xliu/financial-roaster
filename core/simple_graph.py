from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import os

import boto3
from langgraph.graph import StateGraph, END
from langchain_core.output_parsers import StrOutputParser
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate


@dataclass
class Transaction:
    id: str
    date: str
    amount: float
    merchant: str
    memo: Optional[str] = None
    category: Optional[str] = None
    source: str = "upload"
    is_recurring: bool = False


@dataclass
class GraphState:
    transactions: List[Transaction] = field(default_factory=list)
    tone: Literal["roast", "coach"] = "roast"
    advice: str = ""
    chaos_score: int = 0
    flags: List[str] = field(default_factory=list)
    patterns: Dict[str, List[str]] = field(default_factory=dict)
    red_flags: List[str] = field(default_factory=list)


DEFAULT_BEDROCK_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"


@lru_cache(maxsize=1)
def _bedrock_runtime_client():
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    if not region:
        raise RuntimeError("AWS_REGION is not set; Bedrock needs a region.")
    return boto3.client("bedrock-runtime", region_name=region)


def llm_client(temp: float = 0.2):
    model_id = os.getenv("BEDROCK_MODEL_ID", DEFAULT_BEDROCK_MODEL_ID)
    max_tokens = int(os.getenv("BEDROCK_MAX_TOKENS", "700"))
    return ChatBedrock(
        client=_bedrock_runtime_client(),
        model_id=model_id,
        model_kwargs={"temperature": temp, "max_tokens": max_tokens},
    )


def parse_agent_json(text: str) -> Dict[str, Any]:
    import json

    try:
        return json.loads(text)
    except Exception:
        return {}


def txns_payload(transactions: List[Transaction], max_txns: int = 200) -> Dict[str, Any]:
    """Compact payload for LLMs: truncated transactions + simple stats."""
    txns_list = [t.__dict__ for t in transactions]
    sample = txns_list[-max_txns:] if len(txns_list) > max_txns else txns_list

    per_merchant: Dict[str, float] = {}
    for t in transactions:
        per_merchant[t.merchant] = per_merchant.get(t.merchant, 0.0) + float(t.amount or 0)
    top_merchants = sorted(per_merchant.items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]

    stats = {
        "count": len(transactions),
        "total_amount": sum(float(t.amount or 0) for t in transactions),
        "top_merchants": [{"merchant": m, "total": amt} for m, amt in top_merchants],
    }
    return {"stats": stats, "transactions": sample}


def cleaner_agent_llm(transactions: List[Transaction]) -> Dict[str, Any]:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are the Data Cleaner/Normalizer agent.\n"
                "Tasks:\n"
                "1) Normalize merchant names (e.g., remove noise like UBER *TRIP -> Uber).\n"
                "2) Fill missing categories (guess sensibly).\n"
                "3) Flag suspicious items: huge charges, duplicates.\n"
                "Return JSON only: {{\"transactions\": [...], \"flags\": [...]}}\n"
                "Each transaction: id, date, amount, merchant, memo, category, source.",
            ),
            ("user", "{txns}"),
        ]
    )
    chain = prompt | llm_client(0.2) | StrOutputParser()
    text = chain.invoke({"txns": txns_payload(transactions)})
    data = parse_agent_json(text)
    txns_out = []
    for t in data.get("transactions", []):
        try:
            txns_out.append(
                Transaction(
                    id=str(t.get("id") or ""),
                    date=str(t.get("date") or ""),
                    amount=float(t.get("amount") or 0),
                    merchant=str(t.get("merchant") or "Unknown"),
                    memo=t.get("memo"),
                    category=t.get("category"),
                    source=t.get("source") or "upload",
                )
            )
        except Exception:
            continue
    flags = data.get("flags") if isinstance(data.get("flags"), list) else []
    return {"transactions": txns_out or transactions, "flags": flags}


def pattern_miner_agent_llm(transactions: List[Transaction]) -> Dict[str, Any]:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are the Pattern Miner agent.\n"
                "Find: recurring amounts not marked recurring; end-of-month spikes; lots of small charges at same place.\n"
                "Return JSON only: {{\"recurring\": [...], \"eom\": [...], \"cuts\": [...]}} with short strings.",
            ),
            ("user", "{txns}"),
        ]
    )
    chain = prompt | llm_client(0.2) | StrOutputParser()
    text = chain.invoke({"txns": txns_payload(transactions)})
    data = parse_agent_json(text)
    return {
        "recurring": data.get("recurring") or [],
        "eom": data.get("eom") or [],
        "cuts": data.get("cuts") or [],
    }


def risk_agent_llm(transactions: List[Transaction]) -> Dict[str, Any]:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are the Risk/Red Flag Detector.\n"
                "Only care about: overdraft-ish behavior, large high-risk purchases, credit card dependence.\n"
                "Return JSON only: {{\"red_flags\": [up to 3 short items]}}",
            ),
            ("user", "{txns}"),
        ]
    )
    chain = prompt | llm_client(0.2) | StrOutputParser()
    text = chain.invoke({"txns": txns_payload(transactions)})
    data = parse_agent_json(text)
    return {"red_flags": data.get("red_flags") or []}


def chaos_score_agent_llm(transactions: List[Transaction]) -> Dict[str, Any]:
    """
    Calculate a financial chaos score from 0-100 based on spending patterns.
    Higher score = more chaotic financial behavior.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are the Financial Chaos Score Calculator.\n"
                "Analyze spending patterns and calculate a chaos score from 0-100:\n"
                "- 0-20: Boring/stable spending\n"
                "- 21-40: Some questionable choices\n"
                "- 41-60: Moderately chaotic\n"
                "- 61-80: Very messy finances\n"
                "- 81-100: Complete financial mayhem\n\n"
                "Consider: spending variance, impulse purchases, frequency, categories, amounts.\n"
                "Return JSON only: {{\"chaos_score\": number, \"reasoning\": \"brief explanation\"}}",
            ),
            ("user", "{txns}"),
        ]
    )
    chain = prompt | llm_client(0.3) | StrOutputParser()
    text = chain.invoke({"txns": txns_payload(transactions)})
    data = parse_agent_json(text)

    # Ensure we get a valid score between 0-100
    score = data.get("chaos_score", 0)
    try:
        score = int(score)
        score = max(0, min(100, score))  # Clamp between 0-100
    except (ValueError, TypeError):
        score = 50  # Default moderate chaos if parsing fails

    return {
        "chaos_score": score,
        "reasoning": data.get("reasoning", "Unable to calculate reasoning")
    }


def ingest_node(state: GraphState) -> GraphState:
    return state


def categorize_node(state: GraphState) -> GraphState:
    base_txns = state.transactions or sample_transactions()
    cleaned = cleaner_agent_llm(base_txns)
    txns = cleaned["transactions"]
    return GraphState(transactions=txns, tone=state.tone, flags=cleaned["flags"])


def advisor_node(state: GraphState) -> GraphState:
    txns = state.transactions or sample_transactions()

    # Run the three analysis sub-agents in parallel to reduce latency.
    with ThreadPoolExecutor(max_workers=3) as executor:
        patterns_future = executor.submit(pattern_miner_agent_llm, txns)
        risks_future = executor.submit(risk_agent_llm, txns)
        chaos_future = executor.submit(chaos_score_agent_llm, txns)
        patterns = patterns_future.result()
        risks = risks_future.result()
        chaos = chaos_future.result()

    agent_context = {
        "suspicious": state.flags,
        "recurring": patterns["recurring"],
        "eom": patterns["eom"],
        "cuts": patterns["cuts"],
        "red_flags": risks["red_flags"],
        "chaos_score": chaos["chaos_score"],
    }

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a savage financial roaster. Write EXACTLY ONE SHORT PARAGRAPH (max 3-4 sentences). "
                "Be brutal but concise. Reference specific merchants/amounts. No mercy, no profanity.",
            ),
            (
                "user",
                "Roast their worst spending pattern: {txns}\nFlags: {flags}",
            ),
        ]
    )
    chain = prompt | llm_client(0.6) | StrOutputParser()
    advice = chain.invoke({"txns": txns_payload(txns), "flags": agent_context})
    return GraphState(
        transactions=txns,
        tone=state.tone,
        advice=advice,
        chaos_score=chaos["chaos_score"],
        flags=state.flags,
        patterns=patterns,
        red_flags=risks.get("red_flags") or [],
    )


def build_graph(_nodes):
    sg = StateGraph(GraphState)
    sg.add_node("ingest", ingest_node)
    sg.add_node("categorize", categorize_node)
    sg.add_node("advisor", advisor_node)
    sg.set_entry_point("ingest")
    sg.add_edge("ingest", "categorize")
    sg.add_edge("categorize", "advisor")
    sg.add_edge("advisor", END)
    return sg.compile()


def sample_transactions() -> List[Transaction]:
    return [
        Transaction(id="t1", date="2024-04-01", amount=12.5, merchant="Starbucks"),
        Transaction(id="t2", date="2024-04-02", amount=55.0, merchant="Uber"),
    ]
