# 📚 RAG 智能问答助手

基于检索增强生成（RAG）的多轮对话机器人，支持用户上传自定义文本文件，并根据文档内容进行智能问答。

## ✨ 功能特点

- 📄 支持上传 `.txt` 文档，自动切块并向量化
- 🔍 使用 `sentence-transformers` 进行语义检索，返回最相关的文本段落
- 💬 集成 **DeepSeek API** 生成流式回答，支持多轮对话记忆
- 🎨 基于 **Gradio** 构建友好的 Web 界面，无需前端代码
- 🐙 项目代码已托管至 GitHub，具备版本管理

## 🛠 技术栈

- **编程语言**: Python 3.12
- **向量化模型**: `sentence-transformers/all-MiniLM-L6-v2`
- **大模型 API**: DeepSeek API (OpenAI 兼容接口)
- **Web 界面**: Gradio
- **版本控制**: Git + GitHub

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/fei-fff/rag-chatbot.git
cd rag-chatbot
