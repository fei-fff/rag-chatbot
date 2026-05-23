import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 1. 读取文档并切块（你已完成）
with open("sample.txt", "r", encoding="utf-8") as f:
    text = f.read()
chunk_size = 200
chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# 2. 向量化（使用 sentence-transformers）
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
chunk_embeddings = model.encode(chunks)   # 所有块的向量

# 3. 检索函数
def retrieve(query, top_k=2):
    query_emb = model.encode([query])[0]   # 问题的向量
    # 计算余弦相似度（这里用点积代替，因为模型输出已归一化）
    scores = [query_emb @ emb for emb in chunk_embeddings]
    # 取相似度最高的 top_k 个块的索引
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [chunks[i] for i in top_idx]

# 4. 生成函数（调用大模型 API）
from openai import OpenAI
client = OpenAI(api_key="sk-d73a7156217942958bc18d9ebdbc9259", base_url="https://api.deepseek.com")

def generate_answer(query, context_chunks):
    context = "\n".join(context_chunks)
    messages = [
        {"role": "system", "content": "你是一个基于给定资料回答问题的助手。"},
        {"role": "user", "content": f"资料：\n{context}\n\n问题：{query}"}
    ]
    response = client.chat.completions.create(model="deepseek-chat", messages=messages)
    return response.choices[0].message.content

# 5. 整合
query = input("请输入问题：")
context = retrieve(query)
answer = generate_answer(query, context)
print("答案：", answer)