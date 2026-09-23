import json, csv, random, os

random.seed(42)

def load_manual_csv(path):
    items = []
    if not os.path.exists(path):
        print(f"  (Chua co {path} - hay xuat CSV tu Google Sheet truoc)")
        return items
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = (row.get("question") or "").strip()
            a = (row.get("answer") or "").strip()
            if not q or not a:
                continue
            items.append({
                "doc_id": row.get("doc_id", "").strip(),
                "page": row.get("page", "").strip(),
                "question": q, "answer": a,
                "loai": (row.get("loai") or "text").strip().lower(),
            })
    return items

def load_synthetic_json(path):
    items = []
    if not os.path.exists(path):
        print(f"  (Chua co {path} - hay chay Buoc 2.3 va loc thu cong truoc)")
        return items
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for row in data:
        q, a = (row.get("question") or "").strip(), (row.get("answer") or "").strip()
        if not q or not a:
            continue
        items.append({
            "doc_id": row.get("doc_id", ""), "page": row.get("page", ""),
            "question": q, "answer": a, "loai": "text",
            "context": row.get("context", ""),
        })
    return items

def write_llm_jsonl(items, path):
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            ctx = it.get("context") or ""
            input_text = f"Ngu canh: {ctx}\nCau hoi: {it['question']}" if ctx else f"Cau hoi: {it['question']}"
            record = {
                "instruction": "Tra loi cau hoi dua tren ngu canh sau.",
                "input": input_text,
                "output": it["answer"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def main():
    manual = load_manual_csv("data/raw/domain_docs/qa_thu_cong.csv")
    synthetic = load_synthetic_json("data/processed/synthetic_qa_filtered.json")
    print(f"Thu cong: {len(manual)} cau | Tong hop da loc: {len(synthetic)} cau")

    text_items = [x for x in manual if x["loai"] == "text"] + synthetic
    image_items = [x for x in manual if x["loai"] == "image"]

    random.shuffle(text_items)
    split_idx = int(len(text_items) * 0.9)
    train_items, val_items = text_items[:split_idx], text_items[split_idx:]

    os.makedirs("data/processed", exist_ok=True)
    write_llm_jsonl(train_items, "data/processed/llm_train.jsonl")
    write_llm_jsonl(val_items, "data/processed/llm_val.jsonl")
    print(f"\nDa ghi {len(train_items)} cau vao llm_train.jsonl, {len(val_items)} cau vao llm_val.jsonl")

    if image_items:
        with open("data/processed/vlm_pending_images.json", "w", encoding="utf-8") as f:
            json.dump(image_items, f, ensure_ascii=False, indent=2)
        print(f"Co {len(image_items)} cau loai 'image' -> luu tam vao vlm_pending_images.json,")
        print("se ghep voi anh thuc te sau khi lam xong Buoc 3 (Ingestion).")
    else:
        print("Chua co cau hoi loai 'image' nao trong file thu cong.")

if __name__ == "__main__":
    main()
