import sys
from pathlib import Path
import re

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.retrieval.text_cleaner import clean_legal_text

QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant"
COLLECTION_NAME = "pakistan_legal_code"
MODEL_NAME = "BAAI/bge-m3"

LEGAL_SECTIONS = {
    "theft": "379",
    "murder": "302",
    "robbery": "392",
    "cheating": "420",
    "forgery": "465",
    "dacoity": "395"
}

print("Loading Qdrant...")
client = QdrantClient(path=str(QDRANT_PATH))

print("Loading BGE-M3...")
model = SentenceTransformer(MODEL_NAME)

print("Retriever ready.")


def is_actual_legal_text(document):
    text = document.get("text", "").strip()

    if len(text) < 80:
        return False

    try:
        page = int(document.get("page", ""))
    except (ValueError, TypeError):
        return False

    if page < 50:
        return False

    return True


def get_all_documents():
    all_points = []
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=None,
            limit=100,
            offset=offset,
            with_payload=True
        )

        all_points.extend(points)

        if next_offset is None:
            break

        offset = next_offset

    return all_points


def find_exact_section(section_number):
    points = get_all_documents()

    pattern = re.compile(
        rf"^\s*{re.escape(section_number)}\."
    )

    matches = []

    for point in points:
        payload = point.payload

        if not payload:
            continue

        text = payload.get("text", "").strip()

        if not text:
            continue

        document = {
            "text": text,
            "page": payload.get("page", ""),
            "chunk": payload.get("chunk", ""),
            "source": payload.get(
                "source",
                "pakistan_penal_code.pdf"
            ),
            "score": 10.0
        }

        if not is_actual_legal_text(document):
            continue

        if pattern.match(text):
            document["text"] = clean_legal_text(text)
            matches.append(document)

    return matches


def detect_section(query):
    query_lower = query.lower()

    match = re.search(
        r"\b(?:section|sec\.?)\s*(\d+)\b",
        query_lower
    )

    if match:
        return match.group(1)

    for offence, section in LEGAL_SECTIONS.items():
        if re.search(
            rf"\b{re.escape(offence)}\b",
            query_lower
        ):
            return section

    return None


def semantic_search(query, limit=20):
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    print("Searching Qdrant...")

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        limit=limit,
        with_payload=True
    )

    print(
        f"Qdrant returned {len(response.points)} results."
    )

    candidates = []

    for result in response.points:
        payload = result.payload

        if not payload:
            continue

        document = {
            "text": payload.get("text", ""),
            "page": payload.get("page", ""),
            "chunk": payload.get("chunk", ""),
            "source": payload.get(
                "source",
                "pakistan_penal_code.pdf"
            ),
            "score": float(result.score)
        }

        if not is_actual_legal_text(document):
            continue

        document["text"] = clean_legal_text(
            document["text"]
        )

        candidates.append(document)

    return candidates


def retrieve(query, top_k=5):
    detected_section = detect_section(query)

    if detected_section:
        print(
            f"Detected Section: {detected_section}"
        )

    candidates = semantic_search(
        query,
        limit=20
    )

    if detected_section:
        pattern = re.compile(
            rf"^\s*{re.escape(detected_section)}\."
        )

        for document in candidates:
            if pattern.match(document["text"]):
                document["score"] += 10.0

    if detected_section:
        exact_matches = find_exact_section(
            detected_section
        )

        candidates.extend(exact_matches)

    unique_candidates = {}

    for document in candidates:
        key = (
            str(document.get("page", "")),
            str(document.get("chunk", "")),
            document.get("text", "")
        )

        if key not in unique_candidates:
            unique_candidates[key] = document
        elif document["score"] > unique_candidates[key]["score"]:
            unique_candidates[key] = document

    candidates = list(unique_candidates.values())

    candidates.sort(
        key=lambda document: document["score"],
        reverse=True
    )

    return candidates[:top_k]


if __name__ == "__main__":

    test_queries = [
        "What is the punishment for theft?",
        "What is the punishment for murder?",
        "What is the punishment for robbery?",
        "What does Section 302 of the Pakistan Penal Code say?"
    ]

    for query in test_queries:

        print("\n" + "=" * 70)
        print(f"Query: {query}")
        print("=" * 70)

        try:
            results = retrieve(
                query,
                top_k=3
            )

            print(
                f"Retrieved {len(results)} documents\n"
            )

            for index, result in enumerate(
                results,
                start=1
            ):
                print(f"RESULT {index}")
                print(
                    f"Score: {result['score']:.4f}"
                )
                print(
                    f"Page: {result['page']}"
                )
                print(
                    f"Chunk: {result['chunk']}"
                )
                print(
                    f"Source: {result['source']}"
                )
                print(result["text"][:700])
                print("-" * 70)

        except Exception as e:
            print(
                f"ERROR: {type(e).__name__}: {e}"
            )