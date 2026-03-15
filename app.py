import streamlit as st
import faiss
import os
from io import BytesIO
from docx import Document
import numpy as np
from PyPDF2 import PdfReader
from huggingface_hub import InferenceClient

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

# ========================
#  YOUR TOKEN
# ========================
HF_TOKEN = "hf_jho*******************************cvsQw"
client = InferenceClient(api_key=HF_TOKEN)


# ==========================
#  Document to VectorStore
# ==========================
def process_input(input_type, input_data):

    if input_type == "Link":
        loader = WebBaseLoader(input_data)
        docs = loader.load()
        full_text = "\n".join([d.page_content for d in docs])

    elif input_type == "PDF":
        pdf = PdfReader(BytesIO(input_data.read()))
        full_text = ""
        for p in pdf.pages:
            full_text += p.extract_text() or ""

    elif input_type == "Text":
        full_text = input_data

    elif input_type == "DOCX":
        doc = Document(BytesIO(input_data.read()))
        full_text = "\n".join([p.text for p in doc.paragraphs])

    elif input_type == "TXT":
        full_text = input_data.read().decode("utf-8")

    else:
        raise ValueError("Unsupported input type")

    # Split text
    split = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = split.split_text(full_text)

    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": None}
    )

    # FAISS INDEX SETUP
    sample = np.array(embed_model.embed_query("hello"))
    dim = sample.shape[0]
    index = faiss.IndexFlatL2(dim)

    vs = FAISS(
        embedding_function=embed_model.embed_query,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    vs.add_texts(chunks)

    return vs


# ======================
#  Answer Generator (Chat Mode)
# ======================
def answer_question(vectorstore, query):

    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([d.page_content for d in docs])

    messages = [
        {"role": "system", "content": "You are an intelligent AI assistant."},
        {"role": "user", "content": f"Answer using ONLY the context.\n\nCONTEXT:\n{context}\n\nQUESTION: {query}"}
    ]

    result = client.chat_completion(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=messages,
        max_tokens=2000,
        temperature=0.6
    )

    return result.choices[0].message["content"]


# ======================
#  Streamlit UI
# ======================
def main():
    st.title("RAG Q&A App (HF Chat API)")

    input_type = st.selectbox("Input Type", ["Text", "PDF", "DOCX", "TXT", "Link"])

    if input_type == "Text":
        input_data = st.text_area("Enter text")

    elif input_type == "Link":
        input_data = st.text_input("Enter URL")

    else:
        input_data = st.file_uploader("Upload File", type=["pdf", "txt", "docx"])

    if st.button("Proceed"):
        if not input_data:
            st.error("Please provide input!")
            return
        st.session_state["vectorstore"] = process_input(input_type, input_data)
        st.success("Knowledge base created successfully!")

    if "vectorstore" in st.session_state:
        query = st.text_input("Ask your question")
        if st.button("Submit"):
            ans = answer_question(st.session_state["vectorstore"], query)
            st.write("### Answer:")
            st.write(ans)


if __name__ == "__main__":
    main()
