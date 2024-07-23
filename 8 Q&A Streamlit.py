%%writefile app.py
import streamlit as st
from pymilvus import Collection, connections
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import re


@st.cache_resource
def init_milvus():
    connections.connect("default", host="localhost", port="19530")
    collection = Collection("raptor_index")
    collection.load()
    return collection

@st.cache_resource
def init_models():
    dense_model = SentenceTransformer('all-MiniLM-L6-v2')
    genai.configure(api_key="AIzaSyAvGySbPqXTrwg_Tkp1UMoJIBWt667dnIw")
    gen_model = genai.GenerativeModel('gemini-pro')
    return dense_model, gen_model

collection = init_milvus()
dense_model, gen_model = init_models()

def extract_book_title(text):
    words = text.split()
    if len(words) > 5:
        potential_title = ' '.join(words[:5])
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of'}
        title_words = [word for word in potential_title.split() if word.lower() not in stop_words]
        return ' '.join(title_words) if title_words else "Unknown Book"
    return "Unknown Book"

def hybrid_retrieval(query, top_k=10):
    query_embedding = dense_model.encode([query])[0]
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_embedding.tolist()],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["summary", "text"]
    )
    return [hit.id for hits in results for hit in hits]

def rerank_results(query, doc_ids, top_k=5):
    texts = collection.query(
        expr=f"id in {doc_ids}",
        output_fields=["id", "text"]
    )
    return [doc['id'] for doc in texts][:top_k]

def generate_answer(query, context):
    try:
        prompt = f"Question: {query}\n\nContext: {context}\n\nAnswer:"
        response = gen_model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return "Sorry, I couldn't generate an answer due to a technical issue."

def ask_question(question, top_k=5):
    expanded_question = question
    initial_results = hybrid_retrieval(expanded_question, top_k=top_k*2)
    reranked_results = rerank_results(expanded_question, initial_results, top_k=top_k)
    context_docs = collection.query(
        expr=f"id in {reranked_results}",
        output_fields=["text"]
    )
    context = " ".join([doc['text'] for doc in context_docs])
    answer = generate_answer(question, context)
    return answer, reranked_results

def main():
    st.title("AI Textbook Assistant")

    question = st.text_input("Please enter your question:")
    
    if st.button("Get Answer"):
        if question:
            with st.spinner("Generating answer..."):
                answer, source_ids = ask_question(question)
            
            st.subheader("Answer:")
            st.write(answer)
            
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()