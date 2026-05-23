from sentence_transformers import SentenceTransformer
from openai import OpenAI
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')
client = OpenAI(api_key="sk-d73a7156217942958bc18d9ebdbc9259", base_url="https://api.deepseek.com")

with open("sample.txt", "r", encoding="utf-8") as f:
    text = f.read()
chunk_size = 200
chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
embeddings = model.encode(chunks)

query = "Python有什么特点？"
query_emb = model.encode([query])[0]
scores = [np.dot(query_emb, emb) for emb in embeddings]
top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:2]
context = "\n".join([chunks[i] for i in top_idx])

messages = [
    {"role": "system", "content": f"基于资料回答：\n{context}"},
    {"role": "user", "content": query}
]
response = client.chat.completions.create(model="deepseek-chat", messages=messages)
print(response.choices[0].message.content)