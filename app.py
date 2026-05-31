import gradio as gr
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
import numpy as np
import os
import re
import json
import pdfplumber

# ========== 配置 ==========
API_KEY = "sk-d73a7156217942958bc18d9ebdbc9259"
BASE_URL = "https://api.deepseek.com"
HISTORY_FILE = "history.json"

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = None

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

current_chunks = []
current_embeddings = []

# ========== 1. 多角色系统提示词 ==========
ROLE_PROMPTS = {
    "心理辅导员": (
        "你是一位温暖、专业的心理辅导员。你会认真倾听用户的问题，"
        "用共情、理解和温和的语气给予支持和建议。"
        "基于以下资料回答问题，保持专业和关怀的语气："
    ),
    "学习搭子": (
        "你是一个积极、鼓励的学习搭子。你会陪伴用户学习，"
        "用轻松、有动力的语气帮助用户理解知识，像朋友一样交流。"
        "基于以下资料回答问题，保持热情和鼓励的语气："
    ),
    "树洞": (
        "你是一个安静的树洞。你只倾听，不评判，简短而温暖的回应，"
        "让用户感受到被接纳和理解。"
        "基于以下资料简短回应（不超过3句话），保持安静陪伴的语气："
    ),
}

# ========== 2. 情绪关键词 ==========
EMOTION_KEYWORDS = [
    "烦", "累", "焦虑", "emo", "压力", "难过", "伤心",
    "崩溃", "绝望", "迷茫", "沮丧", "紧张", "害怕",
    "孤独", "无聊", "生气", "暴躁", "疲惫", "抑郁",
    "不安", "失眠", "烦躁", "痛苦", "无助", "委屈",
]

def add_empathy_prefix(message):
    msg_lower = message.lower()
    matched = [kw for kw in EMOTION_KEYWORDS if kw in msg_lower]
    if matched:
        return "（检测到你提到" + "、".join(matched) + "，先给你一个温暖的回应）"
    return ""

# ========== 3. 按句子切块 ==========
def split_sentences(text):
    sentences = re.split(r"(?<=[。！？；\n])\s*", text)
    result = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(s) > 300:
            sub_parts = re.split(r"(?<=[，])\s*", s)
            for sp in sub_parts:
                sp = sp.strip()
                if sp and len(sp) > 10:
                    result.append(sp)
        else:
            result.append(s)
    return result

# ========== 4. 多格式文件处理 ==========
def process_file(file_path):
    global current_chunks, current_embeddings
    if file_path is None:
        return "请先上传文件"
    print("处理文件:", file_path)
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        try:
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            return "PDF 解析失败: " + str(e)
    elif ext in (".txt", ".md"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        return "不支持的文件格式: " + ext + "，请上传 .txt 或 .pdf 文件"

    sentences = split_sentences(text)
    current_chunks = sentences
    current_embeddings = embed_model.encode(current_chunks)
    return "文件已加载，共 " + str(len(current_chunks)) + " 个文本块"

# ========== 5. 向量检索 ==========
def search(query, top_k=10):
    if len(current_embeddings) == 0:
        return []
    query_emb = embed_model.encode([query])[0]
    scores = [np.dot(query_emb, emb) for emb in current_embeddings]
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [current_chunks[i] for i in top_idx]

# ========== 6. 重排序 ==========
def get_reranker():
    global reranker
    if reranker is None:
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return reranker

def rerank(query, candidates, top_n=3):
    if not candidates:
        return []
    if len(candidates) <= top_n:
        return candidates
    rk = get_reranker()
    pairs = [[query, c] for c in candidates]
    scores = rk.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_n]]

# ========== 7. 对话历史持久化 ==========
def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_history(history):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("保存历史失败:", e)

def clear_history_file():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    return []

# ========== 8. 流式生成 ==========
def chat_stream(message, history_messages, file_path, role):
    if file_path is None:
        yield "请先上传一个 .txt 或 .pdf 文件"
        return

    if len(current_embeddings) == 0:
        status = process_file(file_path)
        if "失败" in status or "不支持" in status:
            yield status
            return

    candidates = search(message, top_k=10)
    relevant = rerank(message, candidates, top_n=3)
    if not relevant:
        yield "未在文档中找到相关内容"
        return

    context = "\n".join(relevant)
    system_prompt = ROLE_PROMPTS.get(role, ROLE_PROMPTS["学习搭子"])

    empathy_prefix = add_empathy_prefix(message)
    if empathy_prefix:
        system_prompt = empathy_prefix + "\n" + system_prompt

    messages = [
        {"role": "system", "content": system_prompt + "\n\n资料：\n" + context}
    ]
    for msg in history_messages:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=True,
            temperature=0.3,
        )
        full = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                full += chunk.choices[0].delta.content
                yield full
    except Exception as e:
        yield "API 错误: " + str(e)

# ========== 9. 响应入口 ==========
def respond(message, history, file, role):
    if file is None:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "请先上传 .txt 或 .pdf 文件"})
        save_history(history)
        yield history, history
        return

    history.append({"role": "user", "content": message})
    yield history, history

    full_reply = ""
    for chunk in chat_stream(message, history[:-1], file, role):
        full_reply = chunk
        if history and history[-1]["role"] == "assistant":
            history[-1]["content"] = full_reply
        else:
            history.append({"role": "assistant", "content": full_reply})
        save_history(history)
        yield history, history

# ========== 10. Gradio 界面 ==========
CUSTOM_CSS = """
.gradio-container { max-width: 900px !important; margin: auto !important; }
footer { display: none !important; }
"""

EXAMPLES = [
    ["Python 有什么特点？"],
    ["什么是 RAG？"],
    ["我最近学习很累，怎么办？"],
    ["给我总结一下文档内容"],
    ["压力好大，感觉学不进去"],
]

with gr.Blocks(title="RAG 多轮聊天机器人") as demo:
    gr.Markdown(
        """
        # 多轮对话 RAG 助手
        > 支持 PDF / TXT | 流式输出 | 情绪感知 | 多角色 | 历史保存
        """
    )

    chat_state = gr.State(load_history())

    with gr.Row():
        with gr.Column(scale=2):
            file_input = gr.File(
                label="上传文档（支持 .txt / .pdf）",
                file_types=[".txt", ".pdf"],
                type="filepath",
            )
        with gr.Column(scale=1):
            role_radio = gr.Radio(
                label="角色切换",
                choices=["学习搭子", "心理辅导员", "树洞"],
                value="学习搭子",
                interactive=True,
            )

    file_status = gr.Textbox(label="文件状态", interactive=False, lines=2)

    chatbot = gr.Chatbot(
        label="对话记录",
        height=450,
        buttons=["copy"],
    )

    with gr.Row():
        msg = gr.Textbox(
            label="输入问题",
            placeholder="先上传文件，再提出问题...",
            scale=8,
            lines=2,
        )
        send_btn = gr.Button("发送", variant="primary", scale=1)

    with gr.Row():
        clear_btn = gr.Button("清空历史", variant="secondary", size="sm")
        load_btn = gr.Button("加载历史", variant="secondary", size="sm")

    gr.Examples(
        examples=EXAMPLES,
        inputs=[msg],
        label="试试这些问题",
    )

    # 事件绑定
    file_input.change(process_file, inputs=[file_input], outputs=[file_status])

    msg.submit(
        respond,
        inputs=[msg, chat_state, file_input, role_radio],
        outputs=[chat_state, chatbot],
    )
    send_btn.click(
        respond,
        inputs=[msg, chat_state, file_input, role_radio],
        outputs=[chat_state, chatbot],
    )

    clear_btn.click(
        fn=lambda: (clear_history_file(), []),
        outputs=[chatbot, msg],
    )
    load_btn.click(
        fn=lambda: (load_history(), "历史已加载"),
        outputs=[chatbot, msg],
    )

    demo.load(fn=load_history, outputs=[chatbot])

demo.launch(theme=gr.themes.Soft(), css=CUSTOM_CSS)