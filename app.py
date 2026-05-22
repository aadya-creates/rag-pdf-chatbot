import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile

# Load env variables
load_dotenv()

# Streamlit page config
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 PDF RAG Chatbot")
st.markdown("Upload a PDF and ask questions from it.")

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type="pdf"
)

if uploaded_file is not None:

    with st.spinner("Processing PDF..."):

        # Save uploaded PDF temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        # Load PDF
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        # Split text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)

        # Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Vector Store
        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

        # Retriever
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3}
        )

        # LLM
        llm = ChatMistralAI(
            model="mistral-small-latest"
        )

        # Prompt Template
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
                ),
                (
                    "human",
                    """Context:
{context}

Question:
{question}
"""
                )
            ]
        )

    st.success("PDF processed successfully ✅")

    # Question input
    query = st.text_input("Enter your question")

    if st.button("Ask"):

        if query.strip() == "":
            st.warning("Please enter a question.")

        else:
            with st.spinner("Generating answer..."):

                docs = retriever.invoke(query)

                context = "\n\n".join(
                    [doc.page_content for doc in docs]
                )

                final_prompt = prompt.invoke({
                    "context": context,
                    "question": query
                })

                response = llm.invoke(final_prompt)

                st.subheader("Answer")
                st.write(response.content)

                with st.expander("Retrieved Context"):
                    st.write(context)
