from pathlib import Path
from pypdf import PdfReader
from text_cleaner import clean_text
from chunker import chunk_text


def load_pdf(file_path: str):
    reader = PdfReader(file_path)

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text and text.strip():
            cleaned_text = clean_text(text)

            chunks = chunk_text(cleaned_text)

            for chunk_number, chunk in enumerate(chunks, start=1):
                documents.append({
                    "text": chunk,
                    "page": page_number,
                    "chunk": chunk_number,
                    "source": Path(file_path).name
                })

    return documents


if __name__ == "__main__":
    pdf_path = "data/raw/pakistan_penal_code.pdf"

    documents = load_pdf(pdf_path)

    print(f"Total chunks: {len(documents)}")

    if documents:
        print("\nFirst chunk:\n")
        print(documents[0]["text"])