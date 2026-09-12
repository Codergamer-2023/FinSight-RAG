import requests
import streamlit as st


import os

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000/api/v1",
)
QUERY_URL = f"{API_BASE_URL}/query"
UPLOAD_URL = f"{API_BASE_URL}/upload"


st.set_page_config(
    page_title="FinSight",
    page_icon="📊",
    layout="centered",
)


st.title("📊 FinSight")
st.caption("AI-powered financial research assistant")


st.markdown(
    """
    Search financial documents using semantic retrieval,
    cross-encoder reranking, and grounded LLM generation.
    """
)


st.divider()


st.subheader("📤 Upload a Financial Document")

uploaded_file = st.file_uploader(
    "Choose a PDF document",
    type=["pdf"],
)

if uploaded_file is not None:

    upload = st.button(
        "Upload Document",
        type="primary",
        use_container_width=True,
    )

    if upload:
        with st.spinner("Processing document..."):
            try:
                response = requests.post(
                    UPLOAD_URL,
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf",
                        )
                    },
                    timeout=180,
                )

                response.raise_for_status()

                data = response.json()

                st.success(
                    f"Successfully processed **{data['document']}**."
                )

                st.info(
                    f"Pages: {data['pages']}  |  "
                    f"Chunks: {data['chunks']}  |  "
                    f"Total vectors: {data['total_vectors']}"
                )

            except requests.exceptions.Timeout:
                st.error(
                    "Document processing timed out. "
                    "Please try again."
                )

            except requests.exceptions.RequestException:
                st.error(
                    "Unable to upload the document. "
                    "Make sure the FinSight API is running."
                )


st.divider()


st.subheader("Ask a financial question")


examples = [
    "What was NVIDIA's total revenue in fiscal 2026?",
    "What drove NVIDIA's revenue growth in fiscal 2026?",
    "How much did NVIDIA's Data Center revenue grow in fiscal 2026?",
]

selected_example = st.selectbox(
    "Example questions",
    ["Select an example..."] + examples,
)


question = st.text_area(
    "Your question",
    value=(
        ""
        if selected_example == "Select an example..."
        else selected_example
    ),
    placeholder="Ask a question about the financial documents...",
    height=120,
)


ask = st.button(
    "Ask FinSight",
    type="primary",
    use_container_width=True,
)


if ask:

    if not question.strip():
        st.warning("Please enter a question.")

    else:

        with st.spinner("Researching financial documents..."):

            try:

                response = requests.post(
                    QUERY_URL,
                    json={"question": question},
                    timeout=120,
                )

                response.raise_for_status()

                data = response.json()

                st.divider()

                st.subheader("Answer")

                st.write(data["answer"])


                st.subheader("Sources")

                if data["sources"]:

                    for source in data["sources"]:

                        st.info(
                            f"📄 {source['document']}  |  "
                            f"Page {source['page']}"
                        )

                else:

                    st.caption(
                        "No supporting sources found."
                    )


            except requests.exceptions.Timeout:

                st.error(
                    "The request timed out. Please try again."
                )


            except requests.exceptions.RequestException:

                st.error(
                    "Unable to connect to the FinSight API. "
                    "Make sure the FastAPI server is running."
                )