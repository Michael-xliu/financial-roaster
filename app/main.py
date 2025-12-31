from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import shutil
import threading
import time
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from core.simple_graph import build_graph, GraphState, Transaction
import core.simple_graph as nodes
from core.parsers import parse_csv_transactions, parse_document_transactions, is_financial_document


class TransactionModel(BaseModel):
    id: str
    date: str
    amount: float
    merchant: str
    memo: Optional[str] = None
    category: Optional[str] = None
    source: str = "upload"
    is_recurring: bool = False


class AnalyzeRequest(BaseModel):
    tone: Literal["roast", "coach"] = Field(default="roast")
    transactions: Optional[List[TransactionModel]] = None


class ManualAnalyzeRequest(BaseModel):
    spending_text: str = Field(..., min_length=10, max_length=2000)
    tone: Literal["roast", "coach"] = Field(default="roast")


class AnalyzeResponse(BaseModel):
    tone: str
    advice: str
    chaos_score: int = 0
    lines: Optional[List[str]] = None


app = FastAPI(title="Financial Roaster V2", version="0.1.0")  # Sample files added

BASE_DIR = Path(__file__).resolve().parent.parent
SAVED_DIR = BASE_DIR / "saved_files"
SAVED_DIR.mkdir(exist_ok=True)
CACHE_DIR = SAVED_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)
FRONTEND_DIR = BASE_DIR / "frontend"

if FRONTEND_DIR.exists():
    # Serve static assets (js/css/images) from /static to avoid intercepting API routes.
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.on_event("startup")
async def startup_event():
    """Clean any old personal data on startup."""
    cleanup_personal_data()


def save_lines_to_file(lines: List[str]):
    if not lines:
        return None
    timestamp = dt.datetime.utcnow().isoformat().replace(":", "-")
    fname = SAVED_DIR / f"ocr_lines_{timestamp}.txt"
    fname.write_text("\n".join(lines), encoding="utf-8")
    return str(fname)


def compute_file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_cache_path(file_hash: str) -> Path:
    return CACHE_DIR / f"{file_hash}.parse.json"


def _advice_cache_path(file_hash: str, tone: str) -> Path:
    return CACHE_DIR / f"{file_hash}.{tone}.advice.json"


def load_parse_cache(file_hash: str):
    path = _parse_cache_path(file_hash)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        txns_raw = payload.get("transactions") or []
        lines = payload.get("lines") or []
        txns = []
        for t in txns_raw:
            try:
                txns.append(Transaction(**t))
            except Exception:
                continue
        return txns, lines
    except Exception:
        return None


def save_parse_cache(file_hash: str, transactions: List[Transaction], lines: List[str]):
    path = _parse_cache_path(file_hash)
    payload = {
        "transactions": [t.__dict__ for t in transactions],
        "lines": lines,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_advice_cache(file_hash: str, tone: str):
    path = _advice_cache_path(file_hash, tone)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload.get("advice")
    except Exception:
        return None


def save_advice_cache(file_hash: str, tone: str, advice: str):
    path = _advice_cache_path(file_hash, tone)
    payload = {"tone": tone, "advice": advice}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def cleanup_personal_data(file_hash: str = None):
    """Remove personal data from cache and saved files."""
    try:
        if file_hash:
            # Clean specific file hash
            parse_path = _parse_cache_path(file_hash)
            if parse_path.exists():
                parse_path.unlink()

            advice_path_roast = _advice_cache_path(file_hash, "roast")
            if advice_path_roast.exists():
                advice_path_roast.unlink()

            advice_path_coach = _advice_cache_path(file_hash, "coach")
            if advice_path_coach.exists():
                advice_path_coach.unlink()
        else:
            # Clean all cached data older than 1 hour
            current_time = time.time()

            for cache_file in CACHE_DIR.glob("*.json"):
                if current_time - cache_file.stat().st_mtime > 3600:  # 1 hour
                    cache_file.unlink()

            for ocr_file in SAVED_DIR.glob("ocr_lines_*.txt"):
                if current_time - ocr_file.stat().st_mtime > 3600:  # 1 hour
                    ocr_file.unlink()

    except Exception as e:
        print(f"Warning: Failed to cleanup data: {e}")


def schedule_cleanup(file_hash: str, delay_seconds: int = 300):  # 5 minutes
    """Schedule cleanup of personal data after a delay."""
    def delayed_cleanup():
        time.sleep(delay_seconds)
        cleanup_personal_data(file_hash)

    threading.Thread(target=delayed_cleanup, daemon=True).start()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/cleanup")
def cleanup_all_data():
    """Manually trigger cleanup of all personal data."""
    cleanup_personal_data()
    return {"status": "cleaned"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest):
    txns = [Transaction(**t.dict()) for t in payload.transactions] if payload.transactions else []
    state = GraphState(transactions=txns, tone=payload.tone)
    graph = build_graph(nodes)
    result = graph.invoke(state)
    tone = result.tone if hasattr(result, "tone") else result.get("tone")
    advice = result.advice if hasattr(result, "advice") else result.get("advice")
    chaos_score = result.chaos_score if hasattr(result, "chaos_score") else result.get("chaos_score", 0)
    return AnalyzeResponse(tone=tone, advice=advice, chaos_score=chaos_score, lines=None)


@app.post("/analyze-manual", response_model=AnalyzeResponse)
def analyze_manual_spending(payload: ManualAnalyzeRequest):
    """Analyze manually entered spending text and generate a roast."""
    spending_text = payload.spending_text.strip()

    if not spending_text:
        raise HTTPException(status_code=400, detail="Spending text cannot be empty")

    # Parse the text to create mock transactions for the AI to analyze
    # This is a simple parser - in production you might want more sophisticated NLP
    lines = spending_text.split('\n')
    transactions = []

    # Create transactions from the text
    import re
    transaction_id = 1

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        # Look for amounts like $50, $123.45, etc.
        amount_matches = re.findall(r'\$(\d+(?:\.\d{2})?)', line)

        if amount_matches:
            # Use the first amount found
            amount = float(amount_matches[0])

            # Extract merchant/description (remove the amount and clean up)
            description = re.sub(r'\$\d+(?:\.\d{2})?', '', line).strip()
            description = re.sub(r'[•\-\*]', '', description).strip()

            if description:
                # Try to guess merchant from common keywords
                merchant = "Unknown"
                if any(word in description.lower() for word in ["starbucks", "coffee"]):
                    merchant = "Starbucks"
                elif any(word in description.lower() for word in ["doordash", "uber eats", "grubhub"]):
                    merchant = "DoorDash"
                elif any(word in description.lower() for word in ["amazon", "online"]):
                    merchant = "Amazon"
                elif any(word in description.lower() for word in ["target", "store"]):
                    merchant = "Target"
                elif any(word in description.lower() for word in ["netflix", "spotify", "subscription"]):
                    merchant = "Streaming Service"
                else:
                    # Use first few words as merchant
                    words = description.split()[:2]
                    merchant = " ".join(words) if words else "Miscellaneous"

                transaction = Transaction(
                    id=f"m{transaction_id}",
                    date="2024-01-01",  # Use current month
                    amount=amount,
                    merchant=merchant,
                    memo=description[:50] if description != merchant else None,
                    category="Manual Entry",
                    source="manual"
                )
                transactions.append(transaction)
                transaction_id += 1

    # If we couldn't parse specific transactions, create a general one
    if not transactions:
        # Create a single transaction representing the whole text
        transaction = Transaction(
            id="m1",
            date="2024-01-01",
            amount=0.0,  # No specific amount
            merchant="Manual Entry",
            memo=spending_text[:100],
            category="Confessions",
            source="manual"
        )
        transactions.append(transaction)

    # Analyze using the same graph system
    state = GraphState(transactions=transactions, tone=payload.tone)
    graph = build_graph(nodes)
    result = graph.invoke(state)

    tone_out = result.tone if hasattr(result, "tone") else result.get("tone")
    advice = result.advice if hasattr(result, "advice") else result.get("advice")
    chaos_score = result.chaos_score if hasattr(result, "chaos_score") else result.get("chaos_score", 0)

    return AnalyzeResponse(tone=tone_out, advice=advice, chaos_score=chaos_score, lines=None)


@app.post("/upload", response_model=AnalyzeResponse)
async def upload_and_analyze(
    file: UploadFile = File(...), tone: Literal["roast", "coach"] = "roast"
):
    allowed = {
        "text/csv",
        "application/vnd.ms-excel",
        "application/pdf",
        "image/jpeg",
        "image/png",
    }
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Use CSV, PDF, JPEG, or PNG.",
        )
    data = await file.read()
    file_hash = compute_file_hash(data)

    cached_parse = load_parse_cache(file_hash)
    if cached_parse:
        transactions, ocr_lines = cached_parse
    else:
        if file.content_type in {"text/csv", "application/vnd.ms-excel"}:
            text_preview = data.decode("utf-8", errors="ignore")
            preview_lines = text_preview.splitlines()
            if not is_financial_document(preview_lines):
                raise HTTPException(
                    status_code=400,
                    detail="File not recognized as a financial/expense statement.",
                )
            transactions = parse_csv_transactions(data)
            ocr_lines = preview_lines
        else:
            transactions, ocr_lines = parse_document_transactions(data, file.content_type)
            if not is_financial_document(ocr_lines):
                raise HTTPException(
                    status_code=400,
                    detail="File not recognized as a financial/expense statement.",
                )

        if not transactions:
            raise HTTPException(status_code=400, detail="No transactions could be parsed.")
        save_parse_cache(file_hash, transactions, ocr_lines)

    if not transactions:
        raise HTTPException(status_code=400, detail="No transactions could be parsed.")

    cached_advice = load_advice_cache(file_hash, tone)
    if cached_advice:
        save_lines_to_file(ocr_lines)
        # For cached results, we don't have chaos score, so calculate it quickly
        try:
            from core.simple_graph import chaos_score_agent_llm
            chaos_result = chaos_score_agent_llm(transactions)
            chaos_score = chaos_result.get("chaos_score", 50)
        except:
            chaos_score = 50

        # Schedule automatic cleanup of personal data in 5 minutes even for cached results
        schedule_cleanup(file_hash)

        return AnalyzeResponse(tone=tone, advice=cached_advice, chaos_score=chaos_score, lines=None)

    state = GraphState(transactions=transactions, tone=tone)
    graph = build_graph(nodes)
    result = graph.invoke(state)
    tone_out = result.tone if hasattr(result, "tone") else result.get("tone")
    advice = result.advice if hasattr(result, "advice") else result.get("advice")
    chaos_score = result.chaos_score if hasattr(result, "chaos_score") else result.get("chaos_score", 0)
    save_advice_cache(file_hash, tone, advice)
    save_lines_to_file(ocr_lines)

    # Schedule automatic cleanup of personal data in 5 minutes
    schedule_cleanup(file_hash)

    return AnalyzeResponse(tone=tone_out, advice=advice, chaos_score=chaos_score, lines=None)


def get_sample_transactions(sample_type: str) -> List[Transaction]:
    """Generate different sample transaction sets for testing."""

    if sample_type == "tech_bro":
        return [
            Transaction(id="s1", date="2024-01-05", amount=15.99, merchant="Soylent", memo="Meal replacement for productivity", category="Food", source="sample"),
            Transaction(id="s2", date="2024-01-06", amount=299.99, merchant="Apple Store", memo="AirPods Max for focus", category="Electronics", source="sample"),
            Transaction(id="s3", date="2024-01-08", amount=89.99, merchant="Whole Foods", memo="Organic everything", category="Groceries", source="sample"),
            Transaction(id="s4", date="2024-01-10", amount=12.50, merchant="Blue Bottle Coffee", memo="Third wave coffee", category="Food", source="sample"),
            Transaction(id="s5", date="2024-01-12", amount=450.00, merchant="WeWork", memo="Hot desk membership", category="Office", source="sample"),
            Transaction(id="s6", date="2024-01-15", amount=199.99, merchant="Patagonia", memo="Startup uniform vest", category="Clothing", source="sample"),
            Transaction(id="s7", date="2024-01-18", amount=75.00, merchant="ClassPass", memo="Optimize my body", category="Fitness", source="sample"),
            Transaction(id="s8", date="2024-01-20", amount=29.99, merchant="Notion", memo="Life OS subscription", category="Software", source="sample"),
        ]

    elif sample_type == "college_student":
        return [
            Transaction(id="s1", date="2024-01-02", amount=3.50, merchant="7-Eleven", memo="Instant ramen dinner", category="Food", source="sample"),
            Transaction(id="s2", date="2024-01-03", amount=45.00, merchant="Amazon", memo="Textbook I'll never open", category="Education", source="sample"),
            Transaction(id="s3", date="2024-01-05", amount=8.99, merchant="Netflix", memo="Procrastination fuel", category="Entertainment", source="sample"),
            Transaction(id="s4", date="2024-01-07", amount=25.00, merchant="Venmo", memo="Split pizza from last night", category="Food", source="sample"),
            Transaction(id="s5", date="2024-01-10", amount=2.50, merchant="McDonald's", memo="Dollar menu splurge", category="Food", source="sample"),
            Transaction(id="s6", date="2024-01-12", amount=150.00, merchant="Campus Bookstore", memo="Overpriced supplies", category="Education", source="sample"),
            Transaction(id="s7", date="2024-01-15", amount=12.99, merchant="Spotify", memo="Study playlist premium", category="Entertainment", source="sample"),
            Transaction(id="s8", date="2024-01-18", amount=35.00, merchant="Uber Eats", memo="Too lazy to walk to cafeteria", category="Food", source="sample"),
        ]

    elif sample_type == "impulse_shopper":
        return [
            Transaction(id="s1", date="2024-01-03", amount=89.99, merchant="Amazon", memo="Thing I saw in TikTok ad", category="Shopping", source="sample"),
            Transaction(id="s2", date="2024-01-04", amount=234.50, merchant="Target", memo="Went for shampoo, bought everything", category="Shopping", source="sample"),
            Transaction(id="s3", date="2024-01-06", amount=15.99, merchant="As Seen On TV", memo="Miracle cleaning product", category="Shopping", source="sample"),
            Transaction(id="s4", date="2024-01-08", amount=129.99, merchant="QVC", memo="Limited time offer panic buy", category="Shopping", source="sample"),
            Transaction(id="s5", date="2024-01-10", amount=67.99, merchant="Amazon", memo="Same thing different color", category="Shopping", source="sample"),
            Transaction(id="s6", date="2024-01-12", amount=199.99, merchant="Best Buy", memo="Gadget I'll use once", category="Electronics", source="sample"),
            Transaction(id="s7", date="2024-01-15", amount=45.00, merchant="Etsy", memo="Handmade thing I don't need", category="Shopping", source="sample"),
            Transaction(id="s8", date="2024-01-18", amount=299.99, merchant="HSN", memo="3 easy payments seemed reasonable", category="Shopping", source="sample"),
        ]

    elif sample_type == "coffee_addict":
        return [
            Transaction(id="s1", date="2024-01-01", amount=6.45, merchant="Starbucks", memo="New Year resolution lasted 1 hour", category="Food", source="sample"),
            Transaction(id="s2", date="2024-01-01", amount=4.75, merchant="Starbucks", memo="Afternoon coffee", category="Food", source="sample"),
            Transaction(id="s3", date="2024-01-02", amount=8.50, merchant="Local Coffee Co", memo="Support local (still overpriced)", category="Food", source="sample"),
            Transaction(id="s4", date="2024-01-03", amount=12.99, merchant="Starbucks", memo="Venti with 5 shots", category="Food", source="sample"),
            Transaction(id="s5", date="2024-01-04", amount=5.25, merchant="Dunkin'", memo="Settling for cheaper addiction", category="Food", source="sample"),
            Transaction(id="s6", date="2024-01-05", amount=15.99, merchant="Blue Bottle", memo="Fancy single origin", category="Food", source="sample"),
            Transaction(id="s7", date="2024-01-06", amount=7.85, merchant="Starbucks", memo="Mobile order to skip shame", category="Food", source="sample"),
            Transaction(id="s8", date="2024-01-07", amount=89.99, merchant="Nespresso", memo="Home setup to save money (lol)", category="Shopping", source="sample"),
        ]

    else:  # default
        return [
            Transaction(id="s1", date="2024-01-15", amount=4.85, merchant="Starbucks", memo="Daily coffee addiction", category="Food", source="sample"),
            Transaction(id="s2", date="2024-01-16", amount=47.82, merchant="DoorDash", memo="Late night food delivery", category="Food", source="sample"),
            Transaction(id="s3", date="2024-01-17", amount=129.99, merchant="Amazon", memo="Random stuff I don't need", category="Shopping", source="sample"),
            Transaction(id="s4", date="2024-01-18", amount=299.99, merchant="Best Buy", memo="Tech I didn't need", category="Electronics", source="sample"),
        ]


@app.get("/samples")
def get_sample_files():
    """Get list of available sample financial profiles."""
    return {
        "samples": [
            {
                "id": "tech_bro",
                "name": "💻 Silicon Valley Tech Bro",
                "description": "Optimizing life through overpriced productivity tools",
                "preview": "Soylent, WeWork, Patagonia vest..."
            },
            {
                "id": "college_student",
                "name": "🎓 Broke College Student",
                "description": "Living on ramen and student loans",
                "preview": "Dollar menu, textbooks, Netflix..."
            },
            {
                "id": "impulse_shopper",
                "name": "🛍️ Impulse Shopping Queen",
                "description": "TikTok ads are my financial weakness",
                "preview": "Target hauls, As Seen On TV, QVC..."
            },
            {
                "id": "coffee_addict",
                "name": "☕ Coffee Shop Regular",
                "description": "My blood type is espresso",
                "preview": "Multiple Starbucks visits daily..."
            }
        ]
    }


@app.post("/analyze-sample/{sample_id}", response_model=AnalyzeResponse)
def analyze_sample(sample_id: str, tone: Literal["roast"] = "roast"):
    """Analyze a sample financial profile."""
    try:
        print(f"DEBUG: Analyzing sample {sample_id} with tone {tone}")
        transactions = get_sample_transactions(sample_id)
        print(f"DEBUG: Generated {len(transactions)} transactions")
        if not transactions:
            raise HTTPException(status_code=404, detail="Sample not found")

        state = GraphState(transactions=transactions, tone=tone)
        print(f"DEBUG: Created GraphState")
        graph = build_graph(nodes)
        print(f"DEBUG: Built graph")
        result = graph.invoke(state)
        print(f"DEBUG: Graph invocation completed")
        tone_out = result.tone if hasattr(result, "tone") else result.get("tone")
        advice = result.advice if hasattr(result, "advice") else result.get("advice")
        chaos_score = result.chaos_score if hasattr(result, "chaos_score") else result.get("chaos_score", 0)
        print(f"DEBUG: Extracted tone={tone_out}, advice length={len(advice) if advice else 0}, chaos_score={chaos_score}")

        return AnalyzeResponse(tone=tone_out, advice=advice, chaos_score=chaos_score, lines=None)
    except Exception as e:
        print(f"ERROR in analyze_sample: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/")
def root():
    if FRONTEND_DIR.exists():
        return FileResponse(FRONTEND_DIR / "index.html")
    return {"message": "Financial Roaster minimal API", "routes": ["/health", "/analyze", "/upload", "/samples"]}
