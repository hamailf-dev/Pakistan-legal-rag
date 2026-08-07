from pathlib import Path

def load_documents(data_dir: str):
    data_path = Path(data_dir)

    documents = []

    for file_path in data_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in [".txt", ".pdf", ".docx"]:
            documents.append(file_path)

    return documents


if __name__ == "__main__":
    files = load_documents("data/raw")

    print(f"Found {len(files)} documents:")

    for file in files:
        print(file)