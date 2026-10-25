# -*- coding: utf-8 -*-
"""
tao_ban_scan_thu.py - Giai doan 3: tao file PDF "scan" de thu nhanh OCR.

Ca 10 tai lieu cua nhom deu la PDF co lop chu, nen de kiem tra nhanh OCR can them vai file scan.
Script nay render vai trang PDF thanh anh, lam gia hieu ung scan (nghieng nhe, nhieu hat, nen JPEG)
roi ghep lai thanh PDF CHI CO ANH (khong con lop chu). Neu co ban scan/chup that thi dung ban that.

    python ingestion/tao_ban_scan_thu.py --doc TRIET_001 --pages 10-12
    python ingestion/tao_ban_scan_thu.py --doc XL_001 --pages 7,8 --nghieng 1.5 --nhieu 10
    python ingestion/ingest.py --input data/raw/scan_test --out data/processed/ingest_scan   # tu chuyen sang OCR

Dau ra mac dinh: data/raw/scan_test/SCAN_<doc>_p<trang>.pdf
"""
import argparse
import io
import os
import random

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf


def scan_effect(img, angle=0.8, noise=6.0, jpeg=70, seed=0):
    """Lam anh trang PDF trong giong ban scan: xam, nghieng, nhieu, nen JPEG."""
    from PIL import Image

    rnd = random.Random(seed)
    g = img.convert("L")
    if angle:
        g = g.rotate(rnd.uniform(-angle, angle), resample=Image.BICUBIC, expand=False, fillcolor=255)
    if noise:
        try:
            import numpy as np

            arr = np.asarray(g).astype("float32")
            arr += np.random.default_rng(seed).normal(0, noise, arr.shape)
            g = Image.fromarray(arr.clip(0, 255).astype("uint8"))
        except ImportError:
            pass
    buf = io.BytesIO()
    g.save(buf, format="JPEG", quality=jpeg)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def parse_pages(spec, n):
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return [p for p in out if 1 <= p <= n]


def main(argv=None):
    p = argparse.ArgumentParser(description="Tao PDF scan gia lap de thu OCR")
    p.add_argument("--doc", required=True, help="doc_id, vd TRIET_001")
    p.add_argument("--pages", required=True, help="vd 10-12 hoac 7,8")
    p.add_argument("--input", default="data/raw/domain_docs")
    p.add_argument("--out_dir", default="data/raw/scan_test")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--nghieng", type=float, default=0.8, help="goc nghieng toi da (do)")
    p.add_argument("--nhieu", type=float, default=6.0, help="do lech chuan nhieu hat (0 = tat)")
    p.add_argument("--jpeg", type=int, default=70)
    args = p.parse_args(argv)
    from PIL import Image

    src = pymupdf.open(os.path.join(args.input, f"{args.doc}.pdf"))
    pages = parse_pages(args.pages, len(src))
    if not pages:
        raise SystemExit("Khong co trang hop le")
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{pages[0]}-{pages[-1]}" if len(pages) > 1 else str(pages[0])
    out = os.path.join(args.out_dir, f"SCAN_{args.doc}_p{tag}.pdf")
    dst = pymupdf.open()
    for i, pno in enumerate(pages):
        page = src[pno - 1]
        pix = page.get_pixmap(dpi=args.dpi, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        img = scan_effect(img, args.nghieng, args.nhieu, args.jpeg, seed=pno)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=args.jpeg)
        newp = dst.new_page(width=page.rect.width, height=page.rect.height)
        newp.insert_image(newp.rect, stream=buf.getvalue())
    dst.set_metadata({"title": f"Ban scan gia lap {args.doc} trang {tag}", "producer": "tao_ban_scan_thu.py (Nhom 7)"})
    dst.save(out, garbage=3, deflate=True)
    print(f"Da tao {out} ({len(pages)} trang, chi co anh, khong co lop chu)")


if __name__ == "__main__":
    main()
