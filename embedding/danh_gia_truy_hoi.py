# -*- coding: utf-8 -*-
"""
danh_gia_truy_hoi.py - Giai doan 5: do Recall@k cua Qdrant tren 3 bo cau hoi (HUONG_DAN_GIAI_DOAN_3.md, muc 8).

Chay tu thu muc goc repo, sau khi nap Qdrant bang embedding/nap_qdrant.py:
    python embedding/danh_gia_truy_hoi.py
    python embedding/danh_gia_truy_hoi.py --path data/processed/qdrant_local

  1. Tim chu        : cau loai text cua QA_Data (qa_thu_cong.csv) -> chunk, dung khi cung doc_id va co trang ghi tren sheet
  2. Tim hinh bang chu: qa_anh_truy_hoi.jsonl (cau image) -> hinh, dung khi dung image_id (hoac dung trang)
  3. Tim bang anh   : anh_hoi_truy_hoi.jsonl (cau image_query, anh "nguoi dung chup") -> hinh, dung image_id (hoac trang)
Dau ra: data/processed/danh_gia_truy_hoi.md   bang Recall@1/5/10, MRR@10, tach theo tai lieu
        data/processed/danh_gia_truy_hoi.jsonl tung cau: hang cua ket qua dung, top 3 tim duoc
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import nhung  # noqa: E402
from tim_kiem import TimKiem  # noqa: E402

KS = (1, 5, 10)


def first_rank(hits, ok):
    for i, h in enumerate(hits, start=1):
        if ok(h):
            return i
    return None


def hit_pages(h):
    return set(h.get("pages") or [h.get("page_number")])


def same_page(doc_id, pages):
    doc_id, pages = doc_id.lower(), set(pages)
    return lambda h: h["doc_id"].lower() == doc_id and bool(hit_pages(h) & pages)


def load_text_questions(qa_path):
    import build_training_data as btd  # doc QA_Data giong het buoc 2.4 (sua so trang Sheets lam mat so 0, ...)

    items, _ = btd.load_manual_csv(qa_path)
    return [it for it in items if it["loai"] == "text" and it["pages"] and it["doc_id"]]


def summarize(rows, key="rank"):
    n = len(rows)
    out = {"n": n}
    for k in KS:
        out[f"R@{k}"] = sum(1 for r in rows if r[key] and r[key] <= k) / n if n else 0.0
    out["MRR"] = sum(1 / r[key] for r in rows if r[key]) / n if n else 0.0
    return out


def fmt_row(name, s, extra=""):
    return (f"| {name} | {s['n']} | {s['R@1']:.1%} | {s['R@5']:.1%} | {s['R@10']:.1%} | {s['MRR']:.3f} |{extra}"
            .replace(".", ","))


def main(argv=None):
    nhung.utf8_stdout()
    p = argparse.ArgumentParser(description="Do Recall@k cua Qdrant (Giai doan 5, Nhom 7)")
    p.add_argument("--qa", default="data/raw/domain_docs/qa_thu_cong.csv", help="CSV xuat tu sheet QA_Data")
    p.add_argument("--processed", default="data/processed", help="thu muc co qa_anh_truy_hoi.jsonl, anh_hoi_truy_hoi.jsonl")
    p.add_argument("--url", default=nhung.QDRANT_URL)
    p.add_argument("--path", default=None, help="Qdrant luu trong thu muc (nap bang nap_qdrant.py --path)")
    p.add_argument("--collection", default=nhung.COLLECTION)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="data/processed/danh_gia_truy_hoi")
    args = p.parse_args(argv)

    t0 = time.time()
    tk = TimKiem(args.url, args.path, args.collection, device=args.device)
    k_max = max(KS)
    rows = []

    qa_path = nhung.repo_path(args.qa)
    if os.path.exists(qa_path):
        qs = load_text_questions(qa_path)
        print(f"1. Tim chu: {len(qs)} cau text cua QA_Data")
        all_hits = tk.tim_chu_nhieu([q["question"] for q in qs], k_max, loai="text")
        for q, hits in zip(qs, all_hits):
            lech1 = {p + d for p in q["pages"] for d in (-1, 0, 1)}
            rows.append({"bo": "chu", "stt": q["stt"], "doc_id": q["doc_id"], "dap_an": q["page_raw"],
                         "question": q["question"],
                         "rank": first_rank(hits, same_page(q["doc_id"], q["pages"])),
                         "rank_lech1": first_rank(hits, same_page(q["doc_id"], lech1)),
                         "top3": [f"{h['chunk_id']} tr.{h['page_number']}" for h in hits[:3]]})
    else:
        print(f"  !! Khong thay {args.qa} - bo qua phan tim chu")

    anh_path = os.path.join(nhung.repo_path(args.processed), "qa_anh_truy_hoi.jsonl")
    if os.path.exists(anh_path):
        qs = nhung.read_jsonl(anh_path)
        print(f"2. Tim hinh bang chu: {len(qs)} cau image")
        all_hits = tk.tim_chu_nhieu([q["question"] for q in qs], k_max, loai="image")
        for q, hits in zip(qs, all_hits):
            rows.append({"bo": "hinh_bang_chu", "stt": q["stt"], "doc_id": q["doc_id"], "dap_an": q["image_id"],
                         "question": q["question"], "tap": q.get("tap"),
                         "rank": first_rank(hits, lambda h: h["image_id"] == q["image_id"]),
                         "rank_trang": first_rank(hits, same_page(q["doc_id"], [q["page"]])),
                         "top3": [h["image_id"] for h in hits[:3]]})
    else:
        print(f"  !! Khong thay {anh_path} - bo qua phan tim hinh bang chu")

    hoi_path = os.path.join(nhung.repo_path(args.processed), "anh_hoi_truy_hoi.jsonl")
    if os.path.exists(hoi_path):
        qs = [q for q in nhung.read_jsonl(hoi_path) if os.path.exists(nhung.repo_path(q["query_image"]))]
        print(f"3. Tim bang anh: {len(qs)} anh hoi")
        all_hits = tk.tim_bang_anh_nhieu([q["query_image"] for q in qs], k_max)
        for q, hits in zip(qs, all_hits):
            rows.append({"bo": "bang_anh", "stt": q["stt"], "doc_id": q["doc_id"], "dap_an": q["image_id"],
                         "question": q["query_image"], "tap": q.get("tap"),
                         "rank": first_rank(hits, lambda h: h["image_id"] == q["image_id"]),
                         "rank_trang": first_rank(hits, same_page(q["doc_id"], [q["page"]])),
                         "top3": [h["image_id"] for h in hits[:3]]})
    else:
        print(f"  !! Khong thay {hoi_path} - bo qua phan tim bang anh")
    tk.close()
    if not rows:
        sys.exit("Khong co cau hoi nao de danh gia")

    by_bo = defaultdict(list)
    for r in rows:
        by_bo[r["bo"]].append(r)
    head = "| Bộ câu hỏi | Số câu | Recall@1 | Recall@5 | Recall@10 | MRR@10 |\n|---|---|---|---|---|---|"
    L = ["# Đánh giá tìm kiếm Qdrant – Giai đoạn 5", "",
         f"Mô hình chữ `{nhung.TEXT_MODEL}`, mô hình ảnh `{nhung.IMAGE_MODEL}`, collection `{args.collection}`.", "",
         head]
    ten = {"chu": "Tìm chữ: câu text → chunk đúng doc_id + trang",
           "hinh_bang_chu": "Tìm hình bằng câu chữ → đúng `image_id`",
           "bang_anh": "Tìm bằng ảnh chụp → đúng `image_id`"}
    for bo in ("chu", "hinh_bang_chu", "bang_anh"):
        if by_bo[bo]:
            L.append(fmt_row(ten[bo], summarize(by_bo[bo])))
            if bo != "chu":
                L.append(fmt_row("&nbsp;&nbsp;… chỉ cần đúng trang", summarize(by_bo[bo], "rank_trang")))
    if by_bo["chu"]:
        L += ["", "## Tìm chữ theo tài liệu", "",
              "Cột cuối tính đúng cả khi chunk lệch ±1 trang so với sheet: SP_001 và TL_001 có trang bìa không đánh số, "
              "nhiều câu ghi theo số in ở chân trang, nhỏ hơn số trang PDF 1 đơn vị (`HUONG_DAN_GIAI_DOAN_3.md`, mục 9).",
              "",
              "| Tài liệu | Số câu | Recall@1 | Recall@5 | Recall@10 | MRR@10 | Recall@5 (±1 trang) |",
              "|---|---|---|---|---|---|---|"]
        docs = defaultdict(list)
        for r in by_bo["chu"]:
            docs[r["doc_id"]].append(r)
        for d in sorted(docs):
            s1 = summarize(docs[d], "rank_lech1")
            L.append(fmt_row(d, summarize(docs[d]), f" {s1['R@5']:.1%} |".replace(".", ",")))
    for bo in ("hinh_bang_chu", "bang_anh"):
        if by_bo[bo]:
            L += ["", f"## {ten[bo]} theo tài liệu", "", head.replace("Bộ câu hỏi", "Tài liệu")]
            docs = defaultdict(list)
            for r in by_bo[bo]:
                docs[r["doc_id"]].append(r)
            for d in sorted(docs):
                L.append(fmt_row(d, summarize(docs[d])))
    L += ["", f"Chi tiết từng câu (hạng của kết quả đúng, top 3 tìm được): `{args.out}.jsonl`."]

    out = nhung.repo_path(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(out + ".jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print()
    for bo in ("chu", "hinh_bang_chu", "bang_anh"):
        if by_bo[bo]:
            s = summarize(by_bo[bo])
            print(f"  {bo:<14} n={s['n']:<4} R@1 {s['R@1']:.1%}  R@5 {s['R@5']:.1%}  R@10 {s['R@10']:.1%}  "
                  f"MRR {s['MRR']:.3f}")
    print(f"Bao cao: {args.out}.md ({time.time() - t0:.0f} giay)")


if __name__ == "__main__":
    main()
