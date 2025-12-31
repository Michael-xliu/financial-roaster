import csv
from io import StringIO
from typing import List
from .normalize import normalize_rows


def parse_csv_text(text: str) -> List[dict]:
    reader = csv.DictReader(StringIO(text))
    return [row for row in reader]


def parse_csv(data: bytes):
    text = data.decode("utf-8", errors="ignore")
    rows = parse_csv_text(text)
    return normalize_rows(rows)
