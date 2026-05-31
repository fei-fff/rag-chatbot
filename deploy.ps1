# ========== RAG Chatbot GitHub 部署脚本 ==========
Set-Location E:\test

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  RAG Chatbot —— 一键部署到 GitHub" -ForegroundColor Yellow
Write-Host "  支持 PDF/DOCX/图片/视频/代码等任意文件" -ForegroundColor Gray
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[1/6] 覆盖 app.py（新版支持万能文件上传）..." -ForegroundColor Green
Copy-Item -Path "app_v3.py" -Destination "app.py" -Force
Write-Host "       app.py 已更新" -ForegroundColor Gray

Write-Host "[2/6] 覆盖 README.md + requirements.txt..." -ForegroundColor Green
Copy-Item -Path "README_v3.md" -Destination "README.md" -Force
Copy-Item -Path "requirements_new.txt" -Destination "requirements.txt" -Force
Write-Host "       README.md / requirements.txt 已更新" -ForegroundColor Gray

Write-Host "[3/6] 清理临时文件..." -ForegroundColor Green
Remove-Item -Path "app_fixed.py","app_new.py","app_v2.py","app_v3.py","README_new.md","README_v3.md","requirements_new.txt" -Force -ErrorAction SilentlyContinue
Write-Host "       中间文件已清理" -ForegroundColor Gray

Write-Host "[4/6] git add..." -ForegroundColor Green
git add .gitignore app.py requirements.txt README.md deploy.ps1

Write-Host "[5/6] git commit..." -ForegroundColor Green
git commit -m "feat: 支持任意格式文件上传 + 图片视频 AI 识别

- 万能文件上传：PDF/DOCX/图片/视频/代码/表格等任意格式
- 图片识别：多模态 API 自动生成文字描述
- 视频识别：OpenCV 提取关键帧 + 逐帧多模态描述
- DOCX 解析：python-docx 读取 Word 文档
- 代码文件：支持 py/js/cpp/java等 30+ 种格式
- 情绪关键词检测 + 三角色切换
- 向量检索 + 交叉编码器重排序
- 流式输出 + 对话历史持久化
- Gradio 6 现代界面"

Write-Host "[6/6] git push..." -ForegroundColor Green
git push origin main

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  部署完成！" -ForegroundColor Green
Write-Host "  仓库地址: https://github.com/fei-fff/rag-chatbot" -ForegroundColor Yellow
Write-Host "============================================" -ForegroundColor Green