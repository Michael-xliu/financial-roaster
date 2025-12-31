from typing import List
from ..graph import Transaction


def normalize_rows(rows: List[dict]) -> List[Transaction]:
    normalized = []
    for idx, row in enumerate(rows):
        normalized.append(
            Transaction(
                id=row.get("id") or f"row-{idx}",
                date=row.get("date") or row.get("Date") or "",
                amount=float(row.get("amount") or row.get("Amount") or 0),
                merchant=row.get("merchant") or row.get("Merchant") or "Unknown",
                memo=row.get("memo") or row.get("Memo"),
                category=row.get("category") or None,
                source=row.get("source") or "upload",
            )
        )
    return normalized
