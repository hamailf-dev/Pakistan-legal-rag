import json
from sentence_transformers import SentenceTransformer


INPUT_PATH = "data/processed/chunks.json"
OUTPUT_PATH = "data/processed/embeddings.json"

MODEL_NAME = "BAAI/bge-m3"


with open(INPUT_PATH, "r", encoding="utf-8") as file:
    documents = json.load(file)


model = SentenceTransformer(MODEL_NAME)

texts = [document["text"] for document in documents]

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    normalize_embeddings=True
)


for document, embedding in zip(documents, embeddings):
    document["embedding"] = embedding.tolist()


with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(documents, file, ensure_ascii=False)

print(f"Generated embeddings for {len(documents)} chunks.")
print(f"Saved embeddings to {OUTPUT_PATH}")