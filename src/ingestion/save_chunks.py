import json
from pdf_loader import load_pdf


PDF_PATH = "data/raw/pakistan_penal_code.pdf"
OUTPUT_PATH = "data/processed/chunks.json"


documents = load_pdf(PDF_PATH)

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(documents, file, ensure_ascii=False, indent=2)

print(f"Saved {len(documents)} chunks to {OUTPUT_PATH}")