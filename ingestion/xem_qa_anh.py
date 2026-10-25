# -*- coding: utf-8 -*-
"""
xem_qa_anh.py - Giai doan 3: kiem tra va xem lai cac cau hoi anh trong sheet QA_Data.

    python ingestion/xem_qa_anh.py
    python ingestion/xem_qa_anh.py --qa data/raw/domain_docs/qa_thu_cong.csv --out data/processed/ingest/qa_anh_xem_lai.html

Doc CSV xuat tu sheet QA_Data, lay cac dong co image_id:
  loai = image        cau hoi bang chu, dap an la 1 hinh trong tai lieu
  loai = image_query  cau hoi la anh: nguoi dung gui anh chup 1 hinh roi hoi (anh hoi tao bang tao_anh_hoi.py)
(Van doc duoc file qa_anh.csv cu cua tab QA_Anh: moi dong coi la loai image.)

Viec script lam:
  1. Doi chieu tung dong voi images.jsonl cua ingestion: image_id co ton tai khong, doc_id/page tren sheet
     co khop voi hinh khong, cau hoi/dap an co trong khong.
  2. Tao 1 trang HTML tu chua (anh nhung san) de ca nhom soat: anh (voi image_query: anh nguoi dung gui + hinh goc)
     + cau hoi + dap an + "Ngữ cảnh" mo hinh se nhan. Loc theo loai, tai lieu, tap train/val, tim theo chu.
     Sua cau hoi/dap an thi sua tren sheet QA_Data roi xuat lai CSV.
Can: pip install pillow
"""
import argparse
import base64
import csv
import html
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "finetune"))
sys.path.insert(0, HERE)
try:
    import prompt_format as pf  # image_context: dong "Ngữ cảnh" mo hinh nhan kem hinh
except ImportError:  # chay rieng thu muc ingestion
    pf = None
try:
    import tao_anh_hoi
except ImportError:
    tao_anh_hoi = None

LOAI_TEN = {"image": "đáp án là hình", "image_query": "câu hỏi là ảnh"}


def thumb_b64(path, max_px):
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_px, max_px))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=82, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


CSS = """
:root{--bg:#f6f6f3;--card:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e2e1dc;--accent:#2f5d8a;--val:#a5531c;--bad:#b3261e;
--q:#2e7d4f}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#1f1f1d;--ink:#ecebe6;--muted:#a3a29b;--line:#34332f;
--accent:#8db7e0;--val:#e9a06b;--bad:#f2867c;--q:#7fcf9f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 16px}
h1{font-size:18px;margin:0 0 8px}.bar{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
button{font:inherit;font-size:13px;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:999px;
padding:3px 11px;cursor:pointer}button.on{background:var(--accent);border-color:var(--accent);color:var(--bg)}
input{font:inherit;font-size:14px;padding:4px 10px;border:1px solid var(--line);border-radius:8px;background:var(--card);
color:var(--ink);min-width:220px;flex:1;max-width:360px}.count{color:var(--muted);font-size:13px;margin-left:6px}
main{max-width:1180px;margin:0 auto;padding:16px}
.card{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.05fr);gap:18px;background:var(--card);
border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:14px}
.card img{width:100%;height:auto;border:1px solid var(--line);border-radius:6px;background:#fff}
.imgs figure{margin:0 0 8px}.imgs figcaption{font-size:12px;color:var(--muted)}
.imgs .goc img{max-width:55%}
.meta{font-size:12.5px;color:var(--muted);display:flex;flex-wrap:wrap;gap:4px 10px;margin-bottom:6px}
.tag{font-weight:600;color:var(--accent)}.tag.val{color:var(--val)}.loai{font-weight:600;color:var(--q)}
.q{font-weight:600;margin:4px 0 6px}
.a{margin:0 0 8px}.ctx{font-size:13px;color:var(--muted);border-left:3px solid var(--line);padding-left:8px}
.warn{color:var(--bad);font-size:13px;margin-top:6px}
@media (max-width:760px){.card{grid-template-columns:1fr}}
"""

JS = """
const cards=[...document.querySelectorAll('.card')];let doc='*',tap='*',loai='*';
function apply(){const q=document.getElementById('q').value.toLowerCase();let n=0;
cards.forEach(c=>{const ok=(doc==='*'||c.dataset.doc===doc)&&(tap==='*'||c.dataset.tap===tap)&&
(loai==='*'||c.dataset.loai===loai)&&(!q||c.textContent.toLowerCase().includes(q));
c.style.display=ok?'':'none';if(ok)n++});document.getElementById('n').textContent=n+' câu';}
function group(attr,set){document.querySelectorAll('button['+attr+']').forEach(b=>b.onclick=()=>{set(b.getAttribute(attr));
document.querySelectorAll('button['+attr+']').forEach(x=>x.classList.toggle('on',x===b));apply()});}
group('data-doc',v=>doc=v);group('data-tap',v=>tap=v);group('data-loai',v=>loai=v);
document.getElementById('q').oninput=apply;apply();
"""


def load_rows(path):
    """Dong co image_id cua CSV QA_Data (hoac qa_anh.csv cu). Tra ve list dict da chuan hoa."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for k, row in enumerate(rows, start=2):
        iid = (row.get("image_id") or "").strip()
        loai = (row.get("loai") or "image").strip().lower()
        if not iid or loai not in ("image", "image_query"):
            continue
        out.append({"line": k, "stt": (row.get("STT") or "").strip(), "image_id": iid, "loai": loai,
                    "doc_id": (row.get("doc_id") or "").strip(), "page": str(row.get("page") or "").strip(),
                    "question": (row.get("question") or "").strip(), "answer": (row.get("answer") or "").strip(),
                    "tap": (row.get("tap") or "").strip().lower(), "nguoi_soan": (row.get("nguoi_soan") or "").strip(),
                    "trang_thai": (row.get("trang_thai") or row.get("ghi_chu") or "").strip()})
    return out


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    p = argparse.ArgumentParser(description="Kiem tra + trang HTML xem lai cac cau hoi anh cua QA_Data")
    p.add_argument("--qa", default="data/raw/domain_docs/qa_thu_cong.csv", help="CSV xuat tu sheet QA_Data")
    p.add_argument("--ingest", default="data/processed/ingest", help="thu muc dau ra cua ingestion/ingest.py")
    p.add_argument("--out", default="data/processed/ingest/qa_anh_xem_lai.html")
    p.add_argument("--max_px", type=int, default=820, help="canh dai nhat cua anh nhung vao HTML")
    args = p.parse_args(argv)

    if not os.path.exists(args.qa):
        sys.exit(f"Khong thay {args.qa} - xuat CSV tu sheet QA_Data truoc")
    figs = {}
    with open(os.path.join(args.ingest, "images.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                figs[r["image_id"]] = r
    rows = load_rows(args.qa)
    if not rows:
        sys.exit(f"{args.qa} khong co dong nao loai image/image_query kem image_id")
    query_imgs = {}
    need = [r["image_id"] for r in rows if r["loai"] == "image_query"]
    if need and tao_anh_hoi is not None:
        query_imgs = tao_anh_hoi.ensure_query_images(need, args.ingest, quiet=True)

    problems, cards, docs = [], [], []
    e = html.escape
    for row in rows:
        stt, iid, q, a, loai = row["stt"], row["image_id"], row["question"], row["answer"], row["loai"]
        fig = figs.get(iid)
        path = os.path.join(args.ingest, "images", f"{iid}.png")
        warn = []
        if not q or not a:
            warn.append("thiếu câu hỏi hoặc đáp án")
        if fig is None or not os.path.exists(path):
            problems.append(f"dong {row['line']} (STT {stt}): khong co hinh {iid!r} - chay lai ingestion/ingest.py "
                            f"hoac sua image_id")
            continue
        if row["doc_id"].lower() != fig["doc_id"].lower() or row["page"] != str(fig["page_number"]):
            warn.append(f"sheet ghi {row['doc_id']} tr.{row['page']}, hình thật ở {fig['doc_id']} tr.{fig['page_number']}")
        if loai == "image_query" and iid not in query_imgs:
            warn.append("chưa có ảnh hỏi - chạy python ingestion/tao_anh_hoi.py")
        for w in warn:
            problems.append(f"dong {row['line']} (STT {stt}): {w}")
        ctx = ""
        if pf:
            ctx = pf.image_context(fig.get("caption_hint", ""), fig["page_number"], fig.get("section", ""),
                                   fig.get("caption_source", ""), fig.get("slide_number"))
            if loai == "image_query":
                ctx += " + đoạn văn cùng trang (build_training_data.py lấy từ PDF)"
        tap = row["tap"] or "?"
        if fig["doc_id"] not in docs:
            docs.append(fig["doc_id"])
        if loai == "image_query" and iid in query_imgs:
            imgs = (f'<figure><img loading="lazy" alt="ảnh hỏi {e(iid)}" '
                    f'src="data:image/jpeg;base64,{thumb_b64(query_imgs[iid], args.max_px)}">'
                    f'<figcaption>Ảnh người dùng gửi (mô phỏng ảnh chụp)</figcaption></figure>'
                    f'<figure class="goc"><img loading="lazy" alt="{e(iid)}" '
                    f'src="data:image/jpeg;base64,{thumb_b64(path, args.max_px // 2)}">'
                    f'<figcaption>Hình gốc trong tài liệu: {e(iid)}</figcaption></figure>')
        else:
            imgs = (f'<figure><img loading="lazy" alt="{e(iid)}" '
                    f'src="data:image/jpeg;base64,{thumb_b64(path, args.max_px)}"></figure>')
        cards.append(
            f'<section class="card" data-doc="{e(fig["doc_id"])}" data-tap="{e(tap)}" data-loai="{e(loai)}">'
            f'<div class="imgs">{imgs}</div>'
            f'<div><div class="meta"><span class="tag{" val" if tap == "val" else ""}">STT {e(stt)} · {e(tap)}</span>'
            f'<span class="loai">{e(LOAI_TEN[loai])}</span><span>{e(iid)}</span><span>{e(fig["kind"])}</span>'
            f'<span>{e(row["nguoi_soan"])}</span><span>{e(row["trang_thai"])}</span></div>'
            f'<p class="q">{e(q)}</p><p class="a">{e(a)}</p>'
            + (f'<div class="ctx">Ngữ cảnh mô hình nhận: {e(ctx)}</div>' if ctx else "")
            + "".join(f'<div class="warn">⚠ {e(w)}</div>' for w in warn) + "</div></section>")

    n_val = sum(1 for r in rows if r["tap"] == "val")
    n_q = sum(1 for r in rows if r["loai"] == "image_query")
    buttons = '<button class="on" data-doc="*">Tất cả</button>' + "".join(
        f'<button data-doc="{e(d)}">{e(d)}</button>' for d in docs)
    taps = ('<button class="on" data-tap="*">train + val</button><button data-tap="train">train</button>'
            '<button data-tap="val">val</button>')
    loais = ('<button class="on" data-loai="*">Mọi loại</button><button data-loai="image">Đáp án là hình</button>'
             '<button data-loai="image_query">Câu hỏi là ảnh</button>')
    page = (f'<!doctype html><html lang="vi"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>Soát câu hỏi ảnh</title>'
            f'<style>{CSS}</style></head><body><header><h1>Câu hỏi ảnh trong QA_Data — {len(cards)} câu '
            f'({len(cards) - n_q} đáp án là hình, {n_q} câu hỏi là ảnh; {n_val} val)</h1>'
            f'<div class="bar">{loais}</div><div class="bar" style="margin-top:6px">{buttons}</div>'
            f'<div class="bar" style="margin-top:6px">{taps}'
            f'<input id="q" placeholder="Tìm trong câu hỏi, đáp án…"><span class="count" id="n"></span></div>'
            f'</header><main>{"".join(cards)}</main><script>{JS}</script></body></html>')
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"{len(rows)} dong co image_id ({n_q} image_query), {len(cards)} co hinh, {len(problems)} van de")
    for msg in problems[:50]:
        print("  -", msg)
    print(f"Trang xem lai: {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
