
import streamlit as st
import PyPDF2
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from google import genai

st.set_page_config(
    page_title="Document Q&A AI",
    page_icon="📄"
)

st.title("📄 Document Q&A AI")
st.write("Upload a PDF and ask questions about its content.")

# Gemini
api_key = st.secrets.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=api_key)

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def clean_text(text):
    return " ".join(text.split())


def create_chunks(text, chunk_size=800, chunk_overlap=150):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap

    return chunks


def generate_answer(question, context):
    prompt = f"""
Answer the question using only the information from the uploaded PDF.

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.models.generate_content(
        model="models/gemini-3.1-flash-lite",
        contents=prompt
    )

    return response.text


uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    reader = PyPDF2.PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    text = clean_text(text)

    chunks = create_chunks(text)

    embeddings = embedder.encode(
        chunks,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    st.success("Document processed successfully!")

    question = st.text_input(
        "Ask a question about your document:"
    )

    if st.button("Ask"):

        if question:

            query_embedding = embedder.encode(
                [question],
                convert_to_numpy=True
            ).astype("float32")

            distances, indices = index.search(
                query_embedding,
                min(3, len(chunks))
            )

            context = "\n\n".join(
                chunks[i] for i in indices[0]
            )

            answer = generate_answer(
                question,
                context
            )

            st.subheader("Answer")
            st.write(answer)

        else:
            st.warning("Please enter a question.")
