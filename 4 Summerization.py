import google.generativeai as genai
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from sentence_transformers import SentenceTransformer

genai.configure(api_key="AIzaSyAvGySbPqXTrwg_Tkp1UMoJIBWt667dnIw")  # Replace with your actual API key

generation_config = {
    "temperature": 0,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

gen_model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                                  generation_config=generation_config,
                                  safety_settings=safety_settings)

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5))
def summarize_cluster(cluster_texts):
    combined_text = "\n\n".join(cluster_texts[:5])  
    prompt = f"Summarize the following texts concisely:\n\n{combined_text}"

    try:
        response = gen_model.generate_content(prompt)
        if response.parts:
            return response.text
        else:
            return "Summary could not be generated due to content restrictions."
    except Exception as e:
        print(f"Error in summarize_cluster: {str(e)}")
        return "Error in summarization process."

def batch_summarize_clusters(all_cluster_texts, batch_size=5):
    summaries = []
    for i in range(0, len(all_cluster_texts), batch_size):
        batch = all_cluster_texts[i:i+batch_size]
        batch_summaries = [summarize_cluster(cluster) for cluster in batch]
        summaries.extend(batch_summaries)
        time.sleep(5)  
    return summaries


all_cluster_texts = []
for cluster_id in set(cluster_probs):
    cluster_texts = [all_chunks[i] for i, prob in enumerate(cluster_probs) if prob == cluster_id]
    all_cluster_texts.append(cluster_texts)

cluster_summaries = batch_summarize_clusters(all_cluster_texts)

summary_embeddings = embed_model.encode(cluster_summaries)

print("Cluster summaries:", cluster_summaries)
print("Summary embeddings shape:", summary_embeddings.shape)