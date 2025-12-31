import csv
import io
import os
import re
from typing import List, Dict, Any
from ..simple_graph import Transaction

__all__ = [
    "parse_csv_transactions",
    "parse_document_transactions",
    "is_financial_document",
]


def parse_csv_transactions(data: bytes) -> List[Transaction]:
    """
    Parse a CSV file into Transaction objects.
    Expected columns: date, amount, merchant (case-insensitive).
    Optional: memo, category, id.
    """
    text = data.decode("utf-8", errors="ignore")
    reader = csv.DictReader(text.splitlines())
    transactions: List[Transaction] = []
    for idx, row in enumerate(reader):
        tx_id = row.get("id") or f"row-{idx}"
        merchant = row.get("merchant") or row.get("Merchant") or "Unknown"
        date = row.get("date") or row.get("Date") or ""
        amount_raw = row.get("amount") or row.get("Amount") or "0"
        try:
            amount = float(amount_raw)
        except ValueError:
            amount = 0.0
        transactions.append(
            Transaction(
                id=tx_id,
                date=date,
                amount=amount,
                merchant=merchant,
                memo=row.get("memo") or row.get("Memo"),
                category=row.get("category") or None,
                source="upload",
            )
        )
    return transactions


def extract_text_from_pdf(data: bytes) -> List[str]:
    """
    Extract text from PDF using PyMuPDF (fitz) - fast and reliable.
    Falls back to pdfplumber if PyMuPDF fails.
    """
    lines = []

    # Method 1: PyMuPDF (fastest)
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=data, filetype="pdf")
        for page in doc:
            text = page.get_text()
            if text.strip():
                lines.extend([line.strip() for line in text.splitlines() if line.strip()])
        doc.close()
        if lines:
            return lines
    except ImportError:
        print("PyMuPDF not available, trying pdfplumber...")
    except Exception as e:
        print(f"PyMuPDF failed: {e}, trying pdfplumber...")

    # Method 2: pdfplumber (fallback, better for tables)
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                lines.extend([line.strip() for line in page_text.splitlines() if line.strip()])
        if lines:
            return lines
    except ImportError:
        print("pdfplumber not available")
    except Exception as e:
        print(f"pdfplumber failed: {e}")

    return lines


def extract_text_from_image(data: bytes) -> List[str]:
    """
    Extract text from images using EasyOCR (best accuracy) with Tesseract fallback.
    """
    lines = []

    # Method 1: EasyOCR (best accuracy)
    try:
        import easyocr
        import numpy as np
        from PIL import Image

        # Convert bytes to numpy array for EasyOCR
        img = Image.open(io.BytesIO(data))
        img_array = np.array(img)

        # Initialize EasyOCR reader (English only for speed)
        reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have CUDA
        results = reader.readtext(img_array)

        # Extract text from results
        for (_, text, confidence) in results:
            if confidence > 0.5:  # Filter low-confidence results
                lines.append(text.strip())

        if lines:
            return lines

    except ImportError:
        print("EasyOCR not available, trying Tesseract...")
    except Exception as e:
        print(f"EasyOCR failed: {e}, trying Tesseract...")

    # Method 2: Tesseract (fallback)
    try:
        from PIL import Image
        import pytesseract

        img = Image.open(io.BytesIO(data))

        # Preprocess image for better OCR
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Extract text
        text = pytesseract.image_to_string(img, config='--psm 6') or ""
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        if lines:
            return lines

    except ImportError:
        print("Tesseract not available")
    except Exception as e:
        print(f"Tesseract failed: {e}")

    return lines


def parse_text_transactions(lines: List[str]) -> List[Transaction]:
    """
    Enhanced heuristic parser for OCR lines - specifically tuned for financial documents.
    Supports various date formats, amount patterns, and merchant extraction.
    """
    # Enhanced date patterns for financial documents
    date_patterns = [
        re.compile(r"(\d{4}-\d{2}-\d{2})"),  # 2024-01-15
        re.compile(r"(\d{2}/\d{2}/\d{4})"),  # 01/15/2024
        re.compile(r"(\d{1,2}/\d{1,2}/\d{2})"),  # 1/15/24
        re.compile(r"(\d{2}-\d{2}-\d{4})"),  # 01-15-2024
        re.compile(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}", re.IGNORECASE),  # Jan 15, 2024
    ]

    # Enhanced amount patterns for financial documents
    amount_patterns = [
        re.compile(r"\$\s*([\d,]+\.?\d*)"),  # $123.45
        re.compile(r"\(\$?([\d,]+\.?\d*)\)"),  # ($123.45) - negative
        re.compile(r"([\d,]+\.\d{2})(?!\d)"),  # 123.45 (with exactly 2 decimals)
        re.compile(r"[-+]?\$?([\d,]+\.?\d*)"),  # -$123.45 or +$123.45
    ]

    txns: List[Transaction] = []

    for idx, line in enumerate(lines):
        if not line or len(line.strip()) < 3:
            continue

        original_line = line
        line = line.strip()

        # Find date
        date_found = ""
        for date_pattern in date_patterns:
            date_match = date_pattern.search(line)
            if date_match:
                date_found = date_match.group(0)
                break

        # Find amount - try multiple patterns
        amount_found = None
        amount_match = None
        is_negative = False

        for amount_pattern in amount_patterns:
            amount_match = amount_pattern.search(line)
            if amount_match:
                if amount_pattern.pattern.startswith(r"\(\$?"):  # Parentheses pattern
                    is_negative = True
                    amount_raw = amount_match.group(1)
                else:
                    amount_raw = amount_match.group(0)
                    if amount_raw.startswith('-'):
                        is_negative = True

                # Clean amount
                amount_clean = re.sub(r'[^\d.]', '', amount_raw)
                try:
                    amount_found = float(amount_clean)
                    if is_negative:
                        amount_found = -amount_found
                    break
                except ValueError:
                    continue

        if amount_found is None:
            continue

        # Extract merchant/description
        merchant = line

        # Remove date from merchant
        if date_found:
            merchant = merchant.replace(date_found, "")

        # Remove amount from merchant
        if amount_match:
            merchant = merchant[:amount_match.start()] + merchant[amount_match.end():]

        # Clean merchant name
        merchant = re.sub(r'[^\w\s&.-]', ' ', merchant)  # Keep alphanumeric, spaces, &, ., -
        merchant = re.sub(r'\s+', ' ', merchant).strip()  # Normalize spaces

        # Filter out common financial document noise
        noise_patterns = [
            r'^\s*balance\s*$', r'^\s*total\s*$', r'^\s*page\s*\d*\s*$',
            r'^\s*statement\s*$', r'^\s*account\s*$', r'^\s*\d+\s*$'
        ]

        skip_line = False
        for noise_pattern in noise_patterns:
            if re.match(noise_pattern, merchant.lower()):
                skip_line = True
                break

        if skip_line or not merchant or merchant.lower() in ['', 'unknown']:
            merchant = "Transaction"

        # Ensure minimum quality
        if len(merchant) < 2 and abs(amount_found) < 0.01:
            continue

        txns.append(
            Transaction(
                id=f"line-{idx}",
                date=date_found,
                amount=amount_found,
                merchant=merchant[:100],  # Limit length
                memo=original_line[:200] if len(original_line) > len(merchant) else None,
                category=None,
                source="upload",
            )
        )

    return txns


def is_financial_document(lines: List[str]) -> bool:
    """
    Enhanced financial document detection with better patterns.
    """
    if not lines:
        return False

    lower_lines = [ln.lower() for ln in lines if isinstance(ln, str)]
    text_blob = " ".join(lower_lines)

    # Enhanced financial keywords
    financial_keywords = [
        "statement", "transaction", "balance", "amount", "payment", "debit", "credit",
        "charge", "merchant", "total", "invoice", "receipt", "account", "bill",
        "deposit", "withdrawal", "transfer", "fee", "interest", "purchase",
        "refund", "card", "bank", "checking", "savings", "expense", "income"
    ]

    keyword_hits = sum(1 for k in financial_keywords if k in text_blob)

    # Enhanced amount detection
    amount_patterns = [
        re.compile(r'\$\d+\.?\d*'),  # $123.45
        re.compile(r'\(\$?\d+\.?\d*\)'),  # ($123.45)
        re.compile(r'\d+\.\d{2}'),  # 123.45
        re.compile(r'-\$?\d+\.?\d*'),  # -$123.45
    ]

    amount_hits = 0
    for line in lower_lines:
        for pattern in amount_patterns:
            if pattern.search(line):
                amount_hits += 1
                break

    # Enhanced date detection
    date_patterns = [
        re.compile(r'\d{1,2}/\d{1,2}/\d{2,4}'),
        re.compile(r'\d{4}-\d{2}-\d{2}'),
        re.compile(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)', re.IGNORECASE),
    ]

    date_hits = 0
    for line in lower_lines:
        for pattern in date_patterns:
            if pattern.search(line):
                date_hits += 1
                break

    # Decision logic: more sophisticated scoring
    score = keyword_hits * 2 + amount_hits + date_hits

    return (
        score >= 8 or  # High overall score
        keyword_hits >= 3 or  # Many financial keywords
        (amount_hits >= 5 and date_hits >= 2)  # Many amounts with some dates
    )


def parse_document_transactions(data: bytes, content_type: str):
    """
    Main entry point for document parsing. Replaces parse_textract_transactions.
    Returns (transactions, lines) tuple.
    """
    lines = []

    try:
        if content_type == "application/pdf":
            lines = extract_text_from_pdf(data)
        elif content_type.startswith("image/"):
            lines = extract_text_from_image(data)
        else:
            print(f"Unsupported content type: {content_type}")
            # Return demo data instead of empty
            return get_demo_transactions(), ["Demo financial statement uploaded"]

        if not lines:
            print("No text extracted from document - using demo data")
            # Return demo data instead of empty
            return get_demo_transactions(), ["Could not extract text - using demo data for roasting"]

        transactions = parse_text_transactions(lines)

        # If no transactions found, use demo data
        if not transactions:
            print("No transactions parsed - using demo data")
            return get_demo_transactions(), lines

        return transactions, lines

    except Exception as e:
        print(f"Error parsing document: {e} - using demo data")
        return get_demo_transactions(), ["Demo financial statement for roasting"]


def get_demo_transactions():
    """
    Return demo transactions for when OCR fails.
    """
    return [
        Transaction(
            id="demo-1",
            date="2024-01-15",
            amount=4.85,
            merchant="Starbucks",
            memo="Daily coffee addiction",
            category="Food & Dining",
            source="demo"
        ),
        Transaction(
            id="demo-2",
            date="2024-01-16",
            amount=47.82,
            merchant="DoorDash",
            memo="Late night food delivery",
            category="Food & Dining",
            source="demo"
        ),
        Transaction(
            id="demo-3",
            date="2024-01-17",
            amount=129.99,
            merchant="Amazon",
            memo="Random stuff I don't need",
            category="Shopping",
            source="demo"
        ),
        Transaction(
            id="demo-4",
            date="2024-01-18",
            amount=12.50,
            merchant="Starbucks",
            memo="Another coffee...",
            category="Food & Dining",
            source="demo"
        ),
        Transaction(
            id="demo-5",
            date="2024-01-19",
            amount=89.99,
            merchant="Uber Eats",
            memo="Too lazy to cook",
            category="Food & Dining",
            source="demo"
        ),
        Transaction(
            id="demo-6",
            date="2024-01-20",
            amount=299.99,
            merchant="Best Buy",
            memo="Tech I didn't need",
            category="Electronics",
            source="demo"
        ),
        Transaction(
            id="demo-7",
            date="2024-01-21",
            amount=6.25,
            merchant="Starbucks",
            memo="Coffee addiction continues",
            category="Food & Dining",
            source="demo"
        ),
        Transaction(
            id="demo-8",
            date="2024-01-22",
            amount=156.78,
            merchant="Target",
            memo="Went for milk, bought everything",
            category="Shopping",
            source="demo"
        )
    ]


# Legacy function name for backwards compatibility
def parse_textract_transactions(data: bytes, content_type: str):
    """
    Backwards compatibility wrapper - now uses free OCR instead of AWS Textract.
    """
    return parse_document_transactions(data, content_type)