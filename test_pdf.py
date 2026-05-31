from pypdf import PdfReader

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# 测试：你需要自己准备一个简单的 PDF 文件（比如用 Word 导出或下载一个示例）
pdf_text = extract_text_from_pdf("test_pdf.pdf")
print("提取到的文本长度:", len(pdf_text))
print("前200个字符:", pdf_text[:200])