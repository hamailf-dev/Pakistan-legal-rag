import sys
from pathlib import Path
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.retrieval.retriever import retrieve

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading Qwen model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
    device_map="cpu"
)

print("Qwen model ready.")


def clean_answer_text(text):
    text = text.replace("\\_", "_")
    text = text.replace("\\*", "*")
    text = text.replace("\\[", "[")
    text = text.replace("\\]", "]")
    text = text.replace("qat l-i-amd", "qatl-i-amd")
    text = text.replace("qatleamd", "qatl-e-amd")
    text = text.replace("twentyfive", "twenty-five")
    text = text.replace("an d", "and")

    text = re.sub(
        r"\b\d+\s*Ins\.\s*by Act.*?(?=\n|$)",
        "",
        text,
        flags=re.DOTALL
    )

    text = re.sub(
        r"\b\d+\s*Subs\.\s*by Act.*?(?=\n|$)",
        "",
        text,
        flags=re.DOTALL
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def get_section_number(text):
    match = re.match(
        r"^\s*(\d+)\.",
        text
    )

    if match:
        return match.group(1)

    return "Unknown"


def generate_answer(query, top_k=3):

    results = retrieve(
        query,
        top_k=top_k
    )

    print(
        "DEBUG RESULTS:",
        len(results)
    )

    if not results:
        return (
            "I could not find this information in the provided legal document.",
            []
        )

    query_lower = query.lower()

    section_match = re.search(
        r"\b(?:section|sec\.?)\s*(\d+)\b",
        query_lower
    )

    requested_section = None

    if section_match:
        requested_section = section_match.group(1)

    offence_sections = {
        "theft": "379",
        "murder": "302",
        "robbery": "392",
        "cheating": "420",
        "forgery": "465",
        "dacoity": "395"
    }

    if requested_section is None:

        for offence, section in offence_sections.items():

            if re.search(
                rf"\b{re.escape(offence)}\b",
                query_lower
            ):
                requested_section = section
                break

    if requested_section:

        pattern = re.compile(
            rf"^\s*{re.escape(requested_section)}\."
        )

        exact_results = []

        for result in results:

            text = result["text"].strip()

            if pattern.match(text):
                exact_results.append(result)

        if exact_results:

            exact_results.sort(
                key=lambda result: len(result["text"]),
                reverse=True
            )

            result = exact_results[0]

            cleaned_text = clean_answer_text(
                result["text"]
            )

            answer = (
                f"Section {requested_section}: "
                f"{cleaned_text}"
            )

            return (
                answer,
                [result]
            )

        return (
            "I could not find this information in the provided legal document.",
            []
        )

    stop_words = {
        "what",
        "does",
        "is",
        "the",
        "for",
        "under",
        "this",
        "that",
        "are",
        "can",
        "how",
        "who",
        "which",
        "from",
        "with",
        "about",
        "tell",
        "provide",
        "information",
        "according",
        "pakistan",
        "penal",
        "code"
    }

    query_keywords = {
        word
        for word in re.findall(
            r"\b[a-zA-Z]{4,}\b",
            query_lower
        )
        if word not in stop_words
    }

    relevant_results = []

    for result in results:

        text_lower = result["text"].lower()

        matches = sum(
            1
            for keyword in query_keywords
            if keyword in text_lower
        )

        if matches >= 2:
            relevant_results.append(
                result
            )

    if not relevant_results:

        return (
            "I could not find this information in the provided legal document.",
            []
        )

    context_parts = []

    for result in relevant_results:

        cleaned_text = clean_answer_text(
            result["text"]
        )

        context_parts.append(
            f"Section Text:\n"
            f"{cleaned_text}\n"
            f"Page: {result['page']}"
        )

    context = "\n\n".join(
        context_parts
    )

    prompt = f"""
You are a Pakistan legal information assistant.

Answer ONLY using the legal text provided below.

Rules:

- Use only the provided legal text.
- Do not use outside knowledge.
- Do not invent facts.
- Do not guess.
- Do not interpret the law.
- Do not change numbers.
- Do not change imprisonment periods.
- Do not change fines.
- Keep the answer concise.
- Use clean and readable language.
- Preserve the legal meaning exactly.

If the legal text does not answer the question, say:

I could not find this information in the provided legal document.

Legal Text:

{context}

Question:

{query}

Answer:
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise Pakistan legal information assistant. "
                "Use only the supplied legal text."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False
        )

    generated_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    answer = clean_answer_text(
        answer
    )

    if not answer:

        answer = (
            "I could not find this information in the provided legal document."
        )

    return (
        answer,
        relevant_results
    )


if __name__ == "__main__":

    test_queries = [
        "What is the punishment for theft?",
        "What is the punishment for murder?",
        "What is the punishment for robbery?",
        "What does Section 302 of the Pakistan Penal Code say?"
    ]

    for query in test_queries:

        print("\n" + "=" * 70)
        print("Query:", query)
        print("=" * 70)

        try:

            answer, sources = generate_answer(
                query,
                top_k=3
            )

            print("\nAnswer:")
            print(answer)

            print("\nSources:")

            if sources:

                for source in sources:

                    section = get_section_number(
                        source["text"]
                    )

                    print(
                        f"- Pakistan Penal Code, "
                        f"Section {section}, "
                        f"Page {source['page']}"
                    )

            else:

                print("No sources found.")

        except Exception as e:

            print(
                f"ERROR: {type(e).__name__}: {e}"
            )