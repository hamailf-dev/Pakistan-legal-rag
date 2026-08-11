import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.generation.generator import generate_answer


st.set_page_config(
    page_title="Pakistan Legal RAG Assistant",
    page_icon="⚖️",
    layout="centered"
)

st.title("⚖️ Pakistan Legal RAG Assistant")

st.write(
    "Ask questions about the Pakistan Penal Code. "
    "Answers are generated using retrieved legal text."
)

query = st.text_input(
    "Enter your legal question:",
    placeholder="e.g. What is the punishment for theft?"
)

if st.button("Search", type="primary"):

    if not query.strip():
        st.warning("Please enter a legal question.")

    else:

        with st.spinner("Searching legal document..."):

            try:

                answer, sources = generate_answer(
                    query,
                    top_k=3
                )

                st.subheader("Answer")
                st.write(answer)

                st.subheader("Sources")

                if sources:

                    for source in sources:

                        text = source.get("text", "")
                        page = source.get("page", "")
                        source_name = source.get(
                            "source",
                            "pakistan_penal_code.pdf"
                        )

                        st.markdown(
                            f"""
**Pakistan Penal Code**

- **Page:** {page}
- **Source:** {source_name}

<details>
<summary>View Retrieved Legal Text</summary>

{text}

</details>
""",
                            unsafe_allow_html=True
                        )

                else:
                    st.info("No sources found.")

            except Exception as e:

                st.error(
                    f"An error occurred: {type(e).__name__}: {e}"
                )