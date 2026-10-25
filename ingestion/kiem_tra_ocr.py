# -*- coding: utf-8 -*-
"""
kiem_tra_ocr.py - Giai doan 3: do do chinh xac OCR tieng Viet tren trang tai lieu cua nhom.

Cach lam: chon trang PDF co lop chu, render thanh anh (co the lam gia hieu ung scan), cho OCR doc,
roi so voi chu goc trong PDF (dap an chuan co san, khong can go tay).

    python ingestion/kiem_tra_ocr.py                              # engine tu chon, 12 trang mac dinh
    python ingestion/kiem_tra_ocr.py --engine paddle tesseract    # so sanh 2 engine
    python ingestion/kiem_tra_ocr.py --pages TRIET_001:10,50 XL_001:7 --scan

Chi so (cang thap cang tot):
    CER          ti le loi ky tu = so phep sua (them/xoa/thay) / so ky tu dap an
    CER bo dau   nhu tren nhung bo het dau tieng Viet -> chenh lech voi CER = loi do dau
    WER          ti le loi am tiet
Dau ra: data/processed/ingest/kiem_tra_ocr.md (+ .json chi tiet)
"""
import argparse
import json
import os
import re
import sys
import time
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from pdf_parser import PdfDocument  # noqa: E402
from ocr import get_engine, sort_reading_order  # noqa: E402

DEFAULT_PAGES = ["TRIET_001:20,120", "TTHCM_001:30,90", "XL_001:10,40", "AI_001:6,60", "SP_001:6,10", "LX_003:8,40"]


def norm(s, strip_accents=False):
    s = unicodedata.normalize("NFC", s or "").lower()
    s = re.sub(r"[•▪●◦■□►➢✓\-–—_]+", " ", s)
    if strip_accents:
        s = unicodedata.normalize("NFD", s).replace("đ", "d")
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s).strip()


def edit_distance(a, b):
    try:
        from rapidfuzz.distance import Levenshtein
        return Levenshtein.distance(a, b)
    except ImportError:
        pass
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def metrics(ref, hyp):
    r, h = norm(ref), norm(hyp)
    r0, h0 = norm(ref, True), norm(hyp, True)
    rw, hw = r.split(), h.split()
    return {"cer": edit_distance(r, h) / max(1, len(r)),
            "cer_bo_dau": edit_distance(r0, h0) / max(1, len(r0)),
            "wer": edit_distance(rw, hw) / max(1, len(rw)),
            "n_chars": len(r)}


def page_reference(parser, pno):
    lines = sorted(parser.lines[pno - 1], key=lambda l: (round(l.bbox.y0 / 3), l.bbox.x0))
    return "\n".join(l.text for l in lines)


def pick_pages(specs, input_dir):
    out = []
    for spec in specs:
        doc, pages = spec.split(":")
        out.append((doc, [int(p) for p in pages.split(",") if p.strip()]))
    return out


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    p = argparse.ArgumentParser(description="Do do chinh xac OCR tieng Viet (Giai doan 3)")
    p.add_argument("--engine", nargs="+", default=["auto"], help="auto | paddle | tesseract (co the nhieu)")
    p.add_argument("--pages", nargs="*", default=DEFAULT_PAGES, help="vd TRIET_001:10,50 XL_001:7")
    p.add_argument("--input", default="data/raw/domain_docs")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--scan", action="store_true", help="lam gia hieu ung scan (nghieng, nhieu, JPEG) truoc khi OCR")
    p.add_argument("--out", default="data/processed/ingest/kiem_tra_ocr")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    from PIL import Image

    engines = []
    for name in args.engine:
        eng = get_engine(name, device=args.device)
        if eng:
            engines.append(eng)
    if not engines:
        sys.exit("Khong khoi tao duoc engine OCR nao. Xem HUONG_DAN_GIAI_DOAN_3.md muc cai OCR.")
    rows = []
    for doc_id, pages in pick_pages(args.pages, args.input):
        path = os.path.join(args.input, f"{doc_id}.pdf")
        if not os.path.exists(path):
            print(f"  (bo qua {doc_id}: khong co file)")
            continue
        parser = PdfDocument(path, doc_id)
        for pno in pages:
            if not 1 <= pno <= parser.n_pages:
                continue
            ref = page_reference(parser, pno)
            pix = parser.render_page(pno, dpi=args.dpi)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            if args.scan:
                from tao_ban_scan_thu import scan_effect
                img = scan_effect(img, seed=pno)
            for eng in engines:
                t = time.time()
                lines = sort_reading_order([l for l in eng.read(img) if l.score >= 0.3])
                secs = time.time() - t
                hyp = "\n".join(l.text for l in lines)
                m = metrics(ref, hyp)
                m.update(doc_id=doc_id, page=pno, engine=eng.detail, seconds=round(secs, 2),
                         ref_head=norm(ref)[:150], hyp_head=norm(hyp)[:150])
                rows.append(m)
                print(f"  {doc_id} tr.{pno:<4} {eng.name:<10} CER {100 * m['cer']:5.1f}%  bo dau {100 * m['cer_bo_dau']:5.1f}%"
                      f"  WER {100 * m['wer']:5.1f}%  {secs:.1f}s")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    L = ["# Kiểm tra OCR tiếng Việt — Giai đoạn 3", ""]
    L.append(f"- Thời điểm: {time.strftime('%Y-%m-%d %H:%M')} · ảnh {args.dpi} dpi"
             + (" · có giả lập hiệu ứng scan (nghiêng, nhiễu, JPEG)" if args.scan else " · ảnh render sạch từ PDF"))
    L.append("- Đáp án chuẩn = chữ có sẵn trong PDF của chính trang đó. So sánh không phân biệt hoa thường.")
    L.append("")
    L.append("| Engine | Số trang | CER | CER bỏ dấu | WER (âm tiết) | Giây/trang |")
    L.append("|---|---|---|---|---|---|")
    for eng in engines:
        rs = [r for r in rows if r["engine"] == eng.detail]
        if not rs:
            continue
        tot = sum(r["n_chars"] for r in rs)
        w = lambda k: sum(r[k] * r["n_chars"] for r in rs) / max(1, tot)
        L.append(f"| {eng.detail} | {len(rs)} | {100 * w('cer'):.1f}% | {100 * w('cer_bo_dau'):.1f}% | "
                 f"{100 * w('wer'):.1f}% | {sum(r['seconds'] for r in rs) / len(rs):.1f} |")
    L.append("")
    L.append("CER, WER tính trung bình có trọng số theo số ký tự. Chênh lệch giữa CER và CER bỏ dấu là phần lỗi do dấu thanh, dấu mũ.")
    L.append("")
    L.append("| Tài liệu | Trang | Engine | CER | CER bỏ dấu | WER |")
    L.append("|---|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['doc_id']} | {r['page']} | {r['engine'].split(' lang')[0]} | {100 * r['cer']:.1f}% | "
                 f"{100 * r['cer_bo_dau']:.1f}% | {100 * r['wer']:.1f}% |")
    with open(args.out + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"\nBao cao: {args.out}.md")


if __name__ == "__main__":
    main()
