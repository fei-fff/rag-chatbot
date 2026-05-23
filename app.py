import gradio as gr
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import numpy as np

# ---------- 配置 ----------
API_KEY = "sk-d73a7156217942958bc18d9ebdbc9259"   # 请替换
BASE_URL = "https://api.deepseek.com"

model = SentenceTransformer('all-MiniLM-L6-v2')
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

current_chunks = []
current_embeddings = []

def process_file(file_path):
    global current_chunks, current_embeddings
    print(f"处理文件: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    chunk_size = 200
    current_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    current_embeddings = model.encode(current_chunks)
    return f"文件已加载，共 {len(current_chunks)} 个文本块"

def search(query, top_k=2):
    if len(current_embeddings) == 0:
        return []
    query_emb = model.encode([query])[0]
    scores = [np.dot(query_emb, emb) for emb in current_embeddings]
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [current_chunks[i] for i in top_idx]

def chat_stream(message, history_messages, file_path):
    """流式生成回答，history_messages 是 Gradio 的消息列表（字典格式）"""
    if file_path is None:
        yield "请先上传一个 .txt 文件。"
        return

    process_file(file_path)
    relevant = search(message)
    if not relevant:
        yield "未在文档中找到相关内容。"
        return

    context = "\n".join(relevant)
    # 构建 OpenAI 消息，包含历史对话（不包括当前用户消息，因为还没加入）
    messages = [
        {"role": "system", "content": f"基于资料回答：\n{context}"}
    ]
    for msg in history_messages:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=True,
            temperature=0.3
        )
        full = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                full += chunk.choices[0].delta.content
                yield full
    except Exception as e:
        yield f"API错误: {e}"

def respond(message, history, file):
    """处理用户输入并更新对话"""
    if file is None:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "请先上传 .txt 文件"})
        yield history
        return

    # 添加用户消息
    history.append({"role": "user", "content": message})
    yield history

    # 获取助手回复（流式）
    full_reply = ""
    # 注意：chat_stream 的 history_messages 是除当前 user 消息外的历史
    for chunk in chat_stream(message, history[:-1], file):
        full_reply = chunk
        # 更新或添加 assistant 消息
        if history and history[-1]["role"] == "assistant":
            history[-1]["content"] = full_reply
        else:
            history.append({"role": "assistant", "content": full_reply})
        yield history

# ---------- 界面 ----------
with gr.Blocks(title="RAG 聊天机器人") as demo:
    gr.Markdown("# 📚 多轮对话 RAG 助手")
    with gr.Row():
        file_input = gr.File(label="上传 .txt 文档", type="filepath")
    file_status = gr.Textbox(label="文件状态", interactive=False)
    chatbot = gr.Chatbot(label="对话记录")
    msg = gr.Textbox(label="输入问题", placeholder="先上传文件，再提问...")
    clear = gr.Button("清除历史")

    file_input.change(process_file, inputs=[file_input], outputs=[file_status])
    msg.submit(respond, [msg, chatbot, file_input], [chatbot])
    clear.click(lambda: [], None, chatbot)

demo.launch()