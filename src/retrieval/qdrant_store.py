import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.json"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings.json"
QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant"

COLLECTION_NAME = "pakistan_legal_code"
VECTOR_SIZE = 1024


def load_data():
    print("Loading chunks and embeddings...")

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
        embeddings = json.load(f)

    print(f"Loaded {len(chunks)} chunks")
    print(f"Loaded {len(embeddings)} embeddings")

    return chunks, embeddings


def create_qdrant_collection(client):
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        print("Deleted old collection.")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )

    print(f"Created collection: {COLLECTION_NAME}")


def insert_vectors(client, chunks, embeddings):
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) do not match."
        )

    points = []

    for i, item in enumerate(embeddings):
        vector = item["embedding"]
        chunk = chunks[i]

        points.append(
            PointStruct(
                id=i,
                vector=vector,
                payload={
                    "text": chunk.get("text", ""),
                    "page": chunk.get("page", ""),
                    "chunk": chunk.get("chunk", i),
                    "source": chunk.get(
                        "source",
                        "pakistan_penal_code.pdf"
                    )
                }
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print(f"Inserted {len(points)} vectors into Qdrant")


def main():
    print("=" * 60)
    print("Starting Qdrant migration...")
    print("=" * 60)

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_PATH}")

    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {EMBEDDINGS_PATH}"
        )

    client = QdrantClient(path=str(QDRANT_PATH))

    chunks, embeddings = load_data()

    first_vector = embeddings[0]["embedding"]

    print(f"Embedding dimension: {len(first_vector)}")

    if len(first_vector) != VECTOR_SIZE:
        raise ValueError(
            f"Expected {VECTOR_SIZE} dimensions, "
            f"found {len(first_vector)}"
        )

    create_qdrant_collection(client)

    insert_vectors(client, chunks, embeddings)

    collection_info = client.get_collection(COLLECTION_NAME)

    print("\n" + "=" * 60)
    print("QDRANT COLLECTION INFORMATION")
    print("=" * 60)
    print(collection_info)

    print("\n" + "=" * 60)
    print("QDRANT MIGRATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()