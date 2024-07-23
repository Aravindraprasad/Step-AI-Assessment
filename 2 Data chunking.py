import os
import nltk
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

nltk.download('punkt', quiet=True)

def chunk_text(text, tokenizer, target_chunk_size=100):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_chunk_size = 0

    for sentence in sentences:
        sentence_tokens = tokenizer.tokenize(sentence)
        sentence_token_count = len(sentence_tokens)

        if current_chunk_size + sentence_token_count > target_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_chunk_size = 0

        current_chunk.append(sentence)
        current_chunk_size += sentence_token_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def process_folder(input_folder):
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    all_chunks = []
    processed_files = []

    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            input_path = os.path.join(input_folder, filename)

            print(f"Processing file: {filename}")
            processed_files.append(filename)

            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read()

            chunks = chunk_text(text, tokenizer)
            all_chunks.extend(chunks)

    print(f"Total files processed: {len(processed_files)}")
    print(f"Files processed: {', '.join(processed_files)}")
    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks

def embed_chunks(chunks, model):
    return model.encode(chunks)


input_folder = r'C:/Users/jaykh/OneDrive/Desktop/Assessment/processed_textbooks'

all_chunks = process_folder(input_folder)

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embed_chunks(all_chunks, model)

print(f"Shape of embeddings: {embeddings.shape}")
print("First few embeddings:")
embeddings