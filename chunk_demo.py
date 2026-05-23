with open("sample.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("文件长度:", len(text))

chunk_size = 200
chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

print("切块数量:", len(chunks))
for i, chunk in enumerate(chunks):
    print(f"块{i}: 长度{len(chunk)}，开头：{chunk[:30]}...")