from dotenv import load_dotenv
load_dotenv()

import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient


def create_vector_store(texts):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_texts(texts, embeddings)
    return vector_db


def answer_question(vector_db, query):
    retriever = vector_db.as_retriever()
    docs = retriever.invoke(query)
    context = "\n".join([d.page_content for d in docs])
    prompt = f"You are a customer support analyst. Use the context below to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

    client = InferenceClient(api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
    result = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        model="Qwen/Qwen2.5-72B-Instruct",
        max_tokens=300
    )
    return result.choices[0].message.content
