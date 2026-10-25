# -*- coding: utf-8 -*-
"""
ingest.py - Giai doan 3: chay ingestion cho cac file PDF -> chunk van ban + hinh anh kem metadata.

Chay tu thu muc goc repo:
    python ingestion/ingest.py                              # 10 tai lieu trong data/raw/domain_docs
    python ingestion/ingest.py --docs XL_001 AI_001         # chi chay lai vai tai lieu
    python ingestion/ingest.py --input data/raw/scan_test --out data/processed/ingest_scan   # thu OCR file scan

Dau ra (data/processed/ingest/):
    chunks.jsonl     moi dong 1 chunk  {chunk_id, doc_id, type, text, section, page_number, pages, ...}
    images.jsonl     moi dong 1 hinh   {image_id, doc_id, page_number, bbox, caption_hint, image_path, ...}
    images/*.png     anh cat tu PDF (ca hinh ve vector va nhan chu)
    pages.jsonl      thong ke tung trang (so ky tu, co OCR khong, ly do)
    bao_cao_ingest.md  bang tong hop + canh bao + kiem tra chunk co chua dap an QA khong

Chay lai voi --docs chi thay phan cua tai lieu do, giu nguyen ket qua cac tai lieu khac.
"""
import argparse
import base64
import csv
import glob
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from pdf_parser import Element, PdfDocument, suspicious_ratio, nfc  # noqa: E402
from chunker import Chunker  # noqa: E402

STOPWORDS = set("""
là của và các những được có trong cho với không một này đó thì mà để như theo từ về khi ra
nào gì sao thế hay hoặc cũng đã sẽ đang rất nhiều phải bị do vì nên nếu lại còn chỉ tại trên
dưới ai đâu người cái việc đến đi vào cùng nhau ở hơn sự gồm
""".split())


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Ingestion PDF -> chunk + hinh (Giai doan 3, Nhom 7)")
    p.add_argument("--input", default="data/raw/domain_docs", help="thu muc chua PDF (ten file = doc_id)")
    p.add_argument("--docs", nargs="*", default=None, help="chi xu ly cac doc_id nay")
    p.add_argument("--out", default="data/processed/ingest")
    p.add_argument("--ocr", choices=["auto", "off", "force"], default="auto",
                   help="auto: chi OCR trang scan/chu loi; force: OCR moi trang; off: tat")
    p.add_argument("--ocr_engine", choices=["auto", "paddle", "tesseract"], default="auto")
    p.add_argument("--ocr_device", default=None, help="vd cpu, gpu:0 (PaddleOCR)")
    p.add_argument("--ocr_hinh", choices=["auto", "on", "off"], default="auto",
                   help="OCR chu trong anh raster/slide (auto: bat neu co engine)")
    p.add_argument("--dpi_ocr", type=int, default=200)
    p.add_argument("--dpi_hinh", type=int, default=150)
    p.add_argument("--max_chars", type=int, default=1200)
    p.add_argument("--min_chars", type=int, default=300)
    p.add_argument("--overlap", type=int, default=150)
    p.add_argument("--base64", action="store_true", help="ghi them image_base64 vao images.jsonl (file se lon)")
    p.add_argument("--qa", default="data/raw/domain_docs/qa_thu_cong.csv", help="QA de kiem tra chunk ('' = bo qua)")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------
class LazyOcr:
    def __init__(self, args):
        self.args, self.engine, self.tried = args, None, False

    def get(self):
        if not self.tried:
            self.tried = True
            if self.args.ocr != "off" or self.args.ocr_hinh == "on":
                from ocr import get_engine
                print(f"  Khoi tao OCR ({self.args.ocr_engine}) ...")
                self.engine = get_engine(self.args.ocr_engine, device=self.args.ocr_device)
                if self.engine:
                    print(f"  OCR: {self.engine.detail}")
        return self.engine


def pix_to_pil(pix):
    from PIL import Image
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def ocr_reason(parser, pno, mode):
    if mode == "off":
        return None
    if mode == "force":
        return "ep_ocr"
    text = " ".join(l.text for l in parser.lines[pno - 1])
    n = len(text)
    cover = parser.page_image_cover(pno)
    if n < 50 and cover >= 0.4:
        return "trang_scan"
    if n >= 50 and suspicious_ratio(text) >= 0.15:
        return "chu_loi_font"
    return None


def ocr_page_elements(parser, pno, engine, dpi):
    from ocr import lines_to_paragraphs
    from pdf_parser import CAPTION_RE, heading_level

    pix = parser.render_page(pno, dpi=dpi)
    img = pix_to_pil(pix)
    t = time.time()
    lines = engine.read(img)
    secs = time.time() - t
    paras = lines_to_paragraphs(lines)
    scale = 72.0 / dpi
    els = []
    import pymupdf
    for p in paras:
        bbox = pymupdf.Rect(*[v * scale for v in p.box])
        txt = nfc(p.text)
        if not txt:
            continue
        if getattr(p, "is_heading", False) and not CAPTION_RE.match(txt):
            els.append(Element("heading", txt, pno, bbox, level=heading_level(txt, 99, 0) if re.match(r"^\S", txt) else 3,
                               extra={"source": "ocr"}))
        elif CAPTION_RE.match(txt):
            els.append(Element("caption", txt, pno, bbox, extra={"source": "ocr"}))
        else:
            els.append(Element("para", txt, pno, bbox, extra={"source": "ocr"}))
    return els, len(lines), secs


# ---------------------------------------------------------------------------
# 1 tai lieu
# ---------------------------------------------------------------------------
def process_doc(path, doc_id, args, ocr, out_dir, repo_root):
    t0 = time.time()
    parser = PdfDocument(path, doc_id)
    chunker = Chunker(doc_id, args.max_chars, args.min_chars, args.overlap)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    for old in glob.glob(os.path.join(img_dir, f"{doc_id}_p*.png")):
        os.remove(old)
    images, pages_info, warnings = [], [], []
    stats = Counter()
    for pno in range(1, parser.n_pages + 1):
        reason = ocr_reason(parser, pno, args.ocr)
        info = {"doc_id": doc_id, "page": pno, "n_chars": parser.page_chars[pno - 1],
                "image_cover": round(parser.page_image_cover(pno), 3), "ocr": None}
        figures = []
        if reason and ocr.get():
            els, n_lines, secs = ocr_page_elements(parser, pno, ocr.get(), args.dpi_ocr)
            info.update(ocr=reason, ocr_lines=n_lines, ocr_seconds=round(secs, 2))
            stats["ocr_pages"] += 1
        else:
            if reason:
                warnings.append(f"trang {pno}: can OCR ({reason}) nhung chua co engine OCR -> dung chu cua PDF")
            els, figures = parser.analyze_page(pno)
            if pno in parser.toc_pages:
                info["toc"] = True
                stats["toc_pages"] += 1
        # duong dan muc tai vi tri tung phan tu (de gan "section" cho hinh)
        marks, slide_path, cur_slide, titled = [], {}, None, set()
        for el in els:
            chunker.add(el)
            if el.kind == "slide":
                cur_slide = el.extra.get("slide_number")
                slide_path[cur_slide] = chunker.path()
                continue
            if el.kind == "heading" and cur_slide is not None and cur_slide not in titled:
                titled.add(cur_slide)  # muc cua 1 slide = duong dan sau tieu de cua chinh slide do
                slide_path[cur_slide] = chunker.path()
            if el.bbox is not None:
                marks.append((el.bbox.y1, chunker.path()))
        base_path = marks[0][1] if marks else chunker.path()
        for k, fig in enumerate(figures, 1):
            image_id = f"{doc_id}_p{pno}_{k}"
            rel = f"{out_dir}/images/{image_id}.png".replace("\\", "/")
            w, h = parser.render_crop(fig, os.path.join(img_dir, f"{image_id}.png"), dpi=args.dpi_hinh)
            above = [p for y1, p in marks if y1 <= fig.bbox.y0 + 2]
            if fig.kind == "slide" and fig.slide_number in slide_path:
                above = [slide_path[fig.slide_number]]
            rec = {
                "image_id": image_id, "doc_id": doc_id, "page_number": pno,
                "bbox": [round(v, 1) for v in fig.bbox], "page_size": [round(parser.doc[pno - 1].rect.width, 1),
                                                                      round(parser.doc[pno - 1].rect.height, 1)],
                "kind": fig.kind,
                "caption_hint": fig.caption or fig.slide_title or "",
                "caption_source": "chu_thich" if fig.caption else ("tieu_de_slide" if fig.slide_title else ""),
                "slide_number": fig.slide_number,
                "section": above[-1] if above else base_path,
                "text_in_image": fig.text_in_image,
                "ocr_text": "",
                "context_before": fig.context_before,
                "context_after": fig.context_after,
                "image_path": rel, "width": w, "height": h,
            }
            if args.base64:
                with open(os.path.join(img_dir, f"{image_id}.png"), "rb") as f:
                    rec["image_base64"] = base64.b64encode(f.read()).decode("ascii")
            images.append(rec)
            stats[f"fig_{fig.kind}"] += 1
        info.update(n_elements=len(els), n_figures=len(figures))
        pages_info.append(info)

    chunks = chunker.finish()
    stats["spaces_added"] = parser.spaces_added
    stats["footnotes"] = parser.footnotes_dropped
    # OCR chu trong anh (anh raster, slide) -> giup tim anh theo noi dung
    want_fig_ocr = args.ocr_hinh == "on" or (args.ocr_hinh == "auto" and args.ocr != "off")
    if want_fig_ocr and images:
        eng = ocr.get()
        if eng:
            from PIL import Image
            from ocr import lines_to_paragraphs
            for rec in images:
                if rec["kind"] == "vector":
                    continue  # chu cua hinh ve vector da co san trong PDF (text_in_image)
                try:
                    with Image.open(os.path.join(img_dir, f"{rec['image_id']}.png")) as im:
                        paras = lines_to_paragraphs(eng.read(im.convert("RGB")))
                    rec["ocr_text"] = " | ".join(p.text for p in paras)[:1500]
                    stats["fig_ocr"] += 1
                except Exception as e:
                    warnings.append(f"OCR hinh {rec['image_id']} loi: {type(e).__name__}: {e}")
    link_images_to_chunks(images, chunks)
    stats["seconds"] = round(time.time() - t0, 1)
    meta = {"doc_id": doc_id, "n_pages": parser.n_pages, "is_slides": parser.is_slides,
            "body_size": parser.body_size, "header_footer": sorted(parser.header_footer)[:10]}
    return chunks, images, pages_info, warnings, stats, meta


def link_images_to_chunks(images, chunks):
    """Gan moi hinh voi chunk chua chu thich/ngu canh cua no -> backend tra kem anh khi chunk duoc tim thay."""
    by_page = defaultdict(list)
    for c in chunks:
        for p in c["pages"]:
            by_page[p].append(c)
    pages_with_text = sorted(by_page)
    for rec in images:
        pno = rec["page_number"]
        cands = by_page.get(pno, [])
        keys = [k for k in (rec["caption_hint"], rec["context_before"][-60:], rec["context_after"][:60]) if k]
        hit = [c["chunk_id"] for c in cands if any(k[:60] in c["text"] for k in keys)]
        if not hit and rec.get("slide_number") is not None:
            hit = [c["chunk_id"] for c in cands if c.get("slide_number") == rec["slide_number"]]
        if not hit and not cands and pages_with_text:
            # trang chi co hinh (khong co chu): lay chunk cuoi cua trang truoc / chunk dau cua trang sau gan nhat
            near = min(pages_with_text, key=lambda p: (abs(p - pno), p > pno))
            cands = by_page[near][-1:] if near < pno else by_page[near][:1]
        rec["chunk_ids"] = hit or [c["chunk_id"] for c in cands][:2]


# ---------------------------------------------------------------------------
# Kiem tra: chunk o dung trang co chua dap an QA khong
# ---------------------------------------------------------------------------
def _words(s):
    return re.findall(r"\w+", nfc(s).lower())


def _content(s):
    return {w for w in _words(s) if w not in STOPWORDS}


def _pages(raw):
    s = str(raw or "").strip()
    m = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", s)
    if m and int(m.group(1)) <= int(m.group(2)) <= int(m.group(1)) + 10:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    nums = [int(x) for x in re.findall(r"\d+", s)]
    if len(nums) == 2 and re.fullmatch(r"\d+[.,]\d+", s) and nums[1] < nums[0]:
        b = nums[1]
        while 0 < b < nums[0]:
            b *= 10
        nums[1] = b
    return nums


def qa_check(qa_path, chunks):
    if not qa_path or not os.path.exists(qa_path):
        return None
    by_doc_page = defaultdict(list)
    for c in chunks:
        for p in c["pages"]:
            by_doc_page[(c["doc_id"].lower(), p)].append(c)
    res = defaultdict(lambda: {"n": 0, "hit": 0, "cov": 0.0, "miss": []})
    docs = {c["doc_id"].lower() for c in chunks}
    with open(qa_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("loai") or "text").strip().lower() != "text":
                continue
            doc = (row.get("doc_id") or "").strip()
            q, a = row.get("question") or "", row.get("answer") or ""
            if doc.lower() not in docs or not a.strip():
                continue
            pages = _pages(row.get("page"))
            cands = [c for p in pages for c in by_doc_page.get((doc.lower(), p), [])]
            if not cands:
                r = res[doc]
                r["n"] += 1
                r["miss"].append((row.get("STT"), row.get("page"), q[:80]))
                continue
            aw = _content(a)
            best = max(len(aw & _content(c["text"])) / max(1, len(aw)) for c in cands)
            r = res[doc]
            r["n"] += 1
            r["cov"] += best
            if best >= 0.5:
                r["hit"] += 1
            else:
                r["miss"].append((row.get("STT"), row.get("page"), q[:80]))
    return res


# ---------------------------------------------------------------------------
# Ghi ket qua
# ---------------------------------------------------------------------------
def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_report(path, args, metas, all_stats, all_warn, chunks, images, qa, ocr_detail):
    L = ["# Báo cáo ingestion — Giai đoạn 3 (Nhóm 7)", ""]
    L.append(f"- Thời điểm chạy: {time.strftime('%Y-%m-%d %H:%M')}")
    L.append(f"- Chunk: tối đa {args.max_chars} ký tự, tối thiểu {args.min_chars}, lặp {args.overlap} ký tự giữa 2 chunk liền nhau")
    L.append(f"- OCR trang: `{args.ocr}` · OCR chữ trong hình: `{args.ocr_hinh}` · engine: {ocr_detail or 'không dùng'}")
    L.append("")
    L.append("## Tổng hợp theo tài liệu")
    L.append("")
    L.append("| doc_id | Trang | Bỏ (mục lục) | OCR | Chunk chữ | Chunk bảng | Độ dài TB | Hình | Hình có chú thích | Thêm dấu cách | Thời gian (s) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for m in metas:
        d = m["doc_id"]
        st = all_stats[d]
        cs = [c for c in chunks if c["doc_id"] == d]
        ims = [i for i in images if i["doc_id"] == d]
        nt = sum(1 for c in cs if c["type"] == "text")
        nb = sum(1 for c in cs if c["type"] == "table")
        avg = sum(c["n_chars"] for c in cs if c["type"] == "text") / max(1, nt)
        kinds = Counter(i["kind"] for i in ims)
        ktxt = ", ".join(f"{v} {k}" for k, v in kinds.items()) or "0"
        L.append(f"| {d}{' (slide)' if m['is_slides'] else ''} | {m['n_pages']} | {st.get('toc_pages', 0)} | "
                 f"{st.get('ocr_pages', 0)} | {nt} | {nb} | {avg:.0f} | {len(ims)} ({ktxt}) | "
                 f"{sum(1 for i in ims if i['caption_hint'])} | {st.get('spaces_added', 0)} | {st.get('seconds', 0)} |")
    tot_c = len(chunks)
    L.append("")
    L.append(f"Tổng: **{tot_c} chunk**, **{len(images)} hình**. \"Thêm dấu cách\" = số chỗ PDF bị mất dấu cách giữa hai "
             "âm tiết (hay gặp ở PDF xuất từ LaTeX) đã được thêm lại.")
    if qa:
        L.append("")
        L.append("## Kiểm tra: chunk ở đúng trang có chứa đáp án QA không")
        L.append("")
        L.append("Với mỗi câu QA loại text, lấy các chunk chứa trang ghi trong QA, tính tỉ lệ từ khoá của đáp án "
                 "xuất hiện trong chunk tốt nhất. Đạt nếu ≥ 50%.")
        L.append("")
        L.append("| doc_id | Số câu | Đạt | Tỉ lệ đạt | Độ phủ TB |")
        L.append("|---|---|---|---|---|")
        for d in sorted(qa):
            r = qa[d]
            if r["n"]:
                L.append(f"| {d} | {r['n']} | {r['hit']} | {100 * r['hit'] / r['n']:.0f}% | {100 * r['cov'] / r['n']:.0f}% |")
        misses = [(d, m) for d in sorted(qa) for m in qa[d]["miss"]]
        if misses:
            L.append("")
            L.append(f"<details><summary>{len(misses)} câu chưa đạt (thường do ghi sai số trang hoặc đáp án diễn đạt lại)</summary>")
            L.append("")
            for d, (stt, page, q) in misses[:200]:
                L.append(f"- {d} · STT {stt} · tr.{page} · {q}")
            L.append("")
            L.append("</details>")
    warn = [(d, w) for d in all_warn for w in all_warn[d]]
    if warn:
        L.append("")
        L.append("## Cảnh báo")
        L.append("")
        for d, w in warn[:100]:
            L.append(f"- {d}: {w}")
    L.append("")
    L.append("## Định dạng đầu ra (data contract)")
    L.append("")
    L.append("- `chunks.jsonl`: `chunk_id, doc_id, type (text|table), text, section, page_number, pages, slide_number, source (pdf|ocr), n_chars`")
    L.append("- `images.jsonl`: `image_id, doc_id, page_number, bbox [x0,y0,x1,y1 theo điểm PDF], caption_hint, kind (raster|vector|mixed|slide), "
             "text_in_image, ocr_text, context_before, context_after, section, image_path, chunk_ids` (+ `image_base64` khi chạy với `--base64`)")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    args = parse_args(argv)
    out_dir = args.out.rstrip("/\\")
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input, "*.pdf")) + glob.glob(os.path.join(args.input, "*.PDF")))
    files = list(dict.fromkeys(files))
    if args.docs:
        want = {d.lower() for d in args.docs}
        files = [f for f in files if os.path.splitext(os.path.basename(f))[0].lower() in want]
    if not files:
        sys.exit(f"Khong tim thay PDF nao trong {args.input}")
    ocr = LazyOcr(args)
    all_chunks, all_images, all_pages = [], [], []
    metas, all_stats, all_warn = [], {}, {}
    print(f"Ingestion {len(files)} tai lieu -> {out_dir}")
    for path in files:
        doc_id = os.path.splitext(os.path.basename(path))[0]
        chunks, images, pages, warns, stats, meta = process_doc(path, doc_id, args, ocr, out_dir, ".")
        all_chunks += chunks
        all_images += images
        all_pages += pages
        metas.append(meta)
        all_stats[doc_id], all_warn[doc_id] = stats, warns
        print(f"  {doc_id:<12} {meta['n_pages']:>4} trang | {len(chunks):>4} chunk | {len(images):>3} hinh"
              f" | OCR {stats.get('ocr_pages', 0)} trang | {stats['seconds']}s")
    done = {m["doc_id"] for m in metas}
    keep = lambda rows: [r for r in rows if r["doc_id"] not in done]
    cpath, ipath, ppath = (os.path.join(out_dir, n) for n in ("chunks.jsonl", "images.jsonl", "pages.jsonl"))
    chunks = keep(read_jsonl(cpath)) + all_chunks
    images = keep(read_jsonl(ipath)) + all_images
    pages = keep(read_jsonl(ppath)) + all_pages
    order = {d: i for i, d in enumerate(sorted({r["doc_id"] for r in chunks + images}))}
    chunks.sort(key=lambda r: (order[r["doc_id"]], r["chunk_id"]))
    images.sort(key=lambda r: (order[r["doc_id"]], r["page_number"], r["image_id"]))
    write_jsonl(chunks, cpath)
    write_jsonl(images, ipath)
    write_jsonl(pages, ppath)
    qa = qa_check(args.qa, all_chunks) if args.qa else None
    eng = ocr.engine
    write_report(os.path.join(out_dir, "bao_cao_ingest.md"), args, metas, all_stats, all_warn, all_chunks,
                 all_images, qa, eng.detail if eng else "")
    print(f"\nDa ghi {len(chunks)} chunk, {len(images)} hinh vao {out_dir} (bao cao: bao_cao_ingest.md)")


if __name__ == "__main__":
    main()
