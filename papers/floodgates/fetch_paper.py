"""Retrieve the attributed, CC BY 4.0 arXiv v1 PDF without modification."""
from pathlib import Path
from urllib.request import Request, urlopen
import hashlib
import json

URL = "https://arxiv.org/pdf/2611.33896v1"
MAX_BYTES = 16 * 1024 * 1024
HERE = Path(__file__).resolve().parent

def main():
    pdf = HERE / "2611.33896v1.pdf"
    receipt = HERE / "pdf_receipt.json"
    if pdf.exists() or receipt.exists():
        raise SystemExit("Refusing to overwrite an existing paper or receipt.")
    request = Request(URL, headers={"User-Agent": "MoralityLab-paper-archive/1.0"})
    with urlopen(request, timeout=60) as response:
        data = response.read(MAX_BYTES + 1)
        final_url = response.geturl()
    if len(data) > MAX_BYTES or not data.startswith(b"%PDF-") or b"%%EOF" not in data[-4096:]:
        raise ValueError("Unexpected or incomplete PDF")
    with pdf.open("xb") as f:
        f.write(data)
    record = {"arxiv": "2611.33896v1", "source": URL, "final_url": final_url,
              "license": "CC-BY-4.0", "title": "Looped Models are Fast Multi-Token Predictors",
              "authors": ["Ayhan Suleymanzade", "Sean McLeish", "Sehoon Kim", "Jinwoo Kim", "Antoine Bosselut"],
              "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
              "modified": False}
    with receipt.open("x", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    print(json.dumps(record, sort_keys=True))

if __name__ == "__main__":
    main()
