from typing import List
from .normalize import normalize_rows


def parse_pdf_bytes(data: bytes) -> List[dict]:
    # TODO: plug in pdfplumber/tabula extraction. Stub returns empty list.
    return []


def parse_pdf(data: bytes):
    rows = parse_pdf_bytes(data)
    return normalize_rows(rows)
