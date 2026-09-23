import os, json, glob, requests
import fitz

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"

def extract_text_chunks(pdf_path, chunk_size=800):
    doc = fitz.open(pdf_path)
    chunks = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text").strip()
        if not text:
            continue
        for i in range(0, len(text), chunk_size):
            piece = text[i:i+chunk_size].strip()
            if len(piece) > 100:
                chunks.append({"doc_id": os.path.basename(pdf_path), "page": page_num, "text": piece})
    return chunks

def generate_qa(context, n=3):
    prompt = f"""Dua vao doan van ban sau, hay sinh ra {n} cap cau hoi-tra loi bang tieng Viet.
Cau hoi phai tra loi duoc chi bang thong tin trong doan van, khong suy dien them.
Chi tra ve dung dinh dang JSON list, khong giai thich gi them:
[{{"question": "...", "answer": "..."}}]

Doan van ban:
\"\"\"{context}\"\"\"
"""
    resp = requests.post(OLLAMA_URL, json={
        "model": MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.3}
    })
    resp.raise_for_status()
    raw = resp.json()["response"]
    try:
        start, end = raw.find("["), raw.rfind("]") + 1
        return json.loads(raw[start:end])
    except Exception:
        print("  !! Khong parse duoc JSON, bo qua doan nay")
        return []

def main():
    pdf_files = glob.glob("data/raw/domain_docs/*.pdf")
    if not pdf_files:
        print("Khong tim thay file PDF nao trong data/raw/domain_docs/")
        return
    all_synthetic = []
    for pdf_path in pdf_files:
        print(f"Dang xu ly {pdf_path} ...")
        for chunk in extract_text_chunks(pdf_path):
            qa_list = generate_qa(chunk["text"], n=3)
            for qa in qa_list:
                all_synthetic.append({
                    "doc_id": chunk["doc_id"], "page": chunk["page"],
                    "question": qa.get("question", "").strip(),
                    "answer": qa.get("answer", "").strip(),
                    "context": chunk["text"],
                })
            print(f"  trang {chunk['page']}: sinh duoc {len(qa_list)} cau")
    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/synthetic_qa_raw.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_synthetic, f, ensure_ascii=False, indent=2)
    print(f"\nHoan tat. Da sinh {len(all_synthetic)} cap QA tho, luu tai {out_path}")
    print("BUOC BAT BUOC TIEP THEO: mo file len, doc lai, XOA cau sai truoc khi dung fine-tune.")

if __name__ == "__main__":
    main()
