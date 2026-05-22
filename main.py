
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()
file_path="document loaders/bio.pdf"
data= PyPDFLoader(file_path)
docs=data.load()
splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
chunks=splitter.split_documents(docs)
embeddings=HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2")
vectorstore=FAISS.from_documents(chunks, embeddings)
retriever=vectorstore.as_retriever(
     search_type="mmr",
     search_kwargs={"k": 3}
 )
llm=ChatMistralAI(
    model="mistral-small-latest")
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

while True:
    query = input("Enter your question (or 'exit' to quit): ")
    if query.lower() == "exit":
        break
docs = retriever.invoke(query)

context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

final_prompt = prompt.invoke({
            "context": context,
            "question": query
        })

response = llm.invoke(final_prompt)
print(response.content)



