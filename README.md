# 📚 RAG 多轮对话聊天机器人

基于检索增强生成（RAG）的智能多轮对话机器人，支持多格式文档问答、情绪感知、多角色切换。

## ✨ 功能特点

- 📄 **多格式支持** — 上传 .txt / .pdf 文件，自动提取文本
- ✂️ **按句子切块** — 按句号/问号/感叹号切分，保持语义完整性
- 🤖 **流式多轮对话** — 集成 DeepSeek API，逐字流式输出，记住对话历史
- 🔍 **语义检索 + 重排序** — 向量召回 top-10 后用交叉编码器重排取 top-3
- 💚 **情绪关键词识别** — 检测负面情绪词，自动添加共情回应
- 🎭 **多角色切换** — 心理辅导员 / 学习搭子 / 树洞，一键切换人设
- 💾 **对话历史持久化** — 自动保存到本地 JSON，启动时恢复
- 🎨 **Gradio 现代界面** — 软主题、示例问题、复制按钮、自定义 CSS

## 🛠 技术栈

Python, Sentence-Transformers, CrossEncoder, DeepSeek API, Gradio, pdfplumber

## 🚀 快速开始

`ash
git clone https://github.com/fei-fff/rag-chatbot.git
cd rag-chatbot
pip install -r requirements.txt
# 在 app.py 中配置你的 API Key
python app.py
`

## 📂 项目结构

`
├── app.py              # 主程序（Gradio 界面 + 全部功能）
├── requirements.txt    # Python 依赖
├── sample.txt          # 示例文本文件
├── test_pdf.pdf        # 示例 PDF 文件
├── chunk_demo.py       # 切块演示
├── rag_demo.py         # RAG 命令行演示
├── test_rag.py         # 检索测试
└── test_pdf.py         # PDF 解析测试
`

## ⚙️ 配置说明

在 pp.py 中修改：

`python
API_KEY = "your-api-key"          # DeepSeek API Key
BASE_URL = "https://api.deepseek.com"  # API 地址
`

首次运行会自动下载模型：
- ll-MiniLM-L6-v2（向量嵌入）
- cross-encoder/ms-marco-MiniLM-L-6-v2（重排序）

已设置 HF_ENDPOINT = "https://hf-mirror.com" 国内镜像加速下载。