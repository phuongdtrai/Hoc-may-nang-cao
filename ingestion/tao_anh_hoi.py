# -*- coding: utf-8 -*-
"""
tao_anh_hoi.py - Giai doan 3: tao "anh nguoi dung gui" cho cau hoi la anh (loai = image_query).

Cau hoi loai image_query tren sheet QA_Data: nguoi dung chup/chup man hinh 1 hinh trong tai lieu
roi hoi ve no. Anh nguoi dung gui khong giong het hinh cat trong images/ (bi nghieng, mo, toi sang
khac, nen JPEG...), nen script nay bien hinh cat thanh 1 anh "giong anh chup" de:
  - lam anh dau vao khi train / danh gia (vlm_train.jsonl, vlm_val.jsonl),
  - lam truy van khi do tim anh o Giai doan 5 (anh_hoi_truy_hoi.jsonl: anh -> image_id / trang).
Ket qua co dinh theo image_id (cung seed -> cung anh), chay lai bao nhieu lan cung ra file nhu cu.

    python ingestion/tao_anh_hoi.py                       # tao anh cho moi dong image_query trong qa_thu_cong.csv
    python ingestion/tao_anh_hoi.py --all                 # tao cho tat ca hinh trong images.jsonl
    python ingestion/tao_anh_hoi.py --ids XL_001_p7_2 AI_001_p31_2 --force

Dau ra: data/processed/ingest/anh_hoi/<image_id>.jpg + anh_hoi.jsonl (thong so bien doi tung anh)
Can: pip install pillow
"""
import argparse
import csv
import hashlib
import json
import os
import random
import sys

INGEST_DIR = "data/processed/ingest"
QA_CSV = "data/raw/domain_docs/qa_thu_cong.csv"
OUT_SUBDIR = "anh_hoi"
SEED = 42
PAPER = (246, 244, 238)  # mau giay


def query_image_path(image_id, ingest_dir=INGEST_DIR):
    """Duong dan anh hoi (tuong doi so voi goc repo) cua 1 hinh."""
    return os.path.join(ingest_dir, OUT_SUBDIR, f"{image_id}.jpg")


def _perspective_coeffs(src, dst):
    """He so cho Image.transform(PERSPECTIVE): anh dau ra lay mau tai diem src ung voi dst (giai he 8x8)."""
    rows, rhs = [], []
    for (x, y), (u, v) in zip(dst, src):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rhs.append(u)
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        rhs.append(v)
    n = 8
    a = [r[:] + [b] for r, b in zip(rows, rhs)]
    for c in range(n):  # khu Gauss co chon phan tu truc
        p = max(range(c, n), key=lambda r: abs(a[r][c]))
        a[c], a[p] = a[p], a[c]
        for r in range(n):
            if r != c and a[c][c]:
                f = a[r][c] / a[c][c]
                a[r] = [x - f * y for x, y in zip(a[r], a[c])]
    return [a[i][n] / a[i][i] for i in range(n)]


def params_for(image_id, seed=SEED):
    """Thong so bien doi co dinh theo image_id."""
    h = int(hashlib.sha1(f"{seed}:{image_id}".encode("utf-8")).hexdigest()[:12], 16)
    rnd = random.Random(h)
    return {
        "le": round(rnd.uniform(0.04, 0.10), 3),          # le giay quanh hinh (ti le canh)
        "meo": round(rnd.uniform(0.0, 0.035), 3),         # do meo phoi canh toi da (ti le canh)
        "xoay": round(rnd.uniform(-2.5, 2.5), 2),         # do
        "sang": round(rnd.uniform(0.90, 1.08), 3),
        "tuong_phan": round(rnd.uniform(0.88, 1.05), 3),
        "am": round(rnd.uniform(-0.03, 0.03), 3),         # am/lanh (lech kenh do - xanh)
        "mo": round(rnd.uniform(0.2, 0.8), 2),            # ban kinh Gaussian blur
        "nhieu": round(rnd.uniform(0.03, 0.08), 3),       # muc tron nhieu
        "ti_le": round(rnd.uniform(0.85, 1.10), 3),       # phong to / thu nho
        "jpeg": rnd.randint(58, 82),
        "rnd": h,
    }


def make_query_image(src_path, out_path, image_id, seed=SEED, max_side=1600):
    """Bien hinh cat src_path thanh anh giong anh chup, luu JPEG out_path. Tra ve thong so da dung."""
    from PIL import Image, ImageEnhance, ImageFilter

    prm = params_for(image_id, seed)
    rnd = random.Random(prm["rnd"])
    with Image.open(src_path) as im:
        im = im.convert("RGB")
    w, h = im.size
    # 1) le giay
    m = max(8, int(prm["le"] * max(w, h)))
    canvas = Image.new("RGB", (w + 2 * m, h + 2 * m), PAPER)
    canvas.paste(im, (m, m))
    im, (w, h) = canvas, canvas.size
    # 2) meo phoi canh: dich 4 goc ngau nhien
    d = prm["meo"] * min(w, h)
    dst = [(0, 0), (w, 0), (w, h), (0, h)]
    src = [(x + rnd.uniform(-d, d), y + rnd.uniform(-d, d)) for x, y in dst]
    im = im.transform((w, h), Image.PERSPECTIVE, _perspective_coeffs(src, dst), Image.BICUBIC, fillcolor=PAPER)
    # 3) xoay nhe
    im = im.rotate(prm["xoay"], resample=Image.BICUBIC, expand=True, fillcolor=PAPER)
    # 4) anh sang, tuong phan, mau
    im = ImageEnhance.Brightness(im).enhance(prm["sang"])
    im = ImageEnhance.Contrast(im).enhance(prm["tuong_phan"])
    r, g, b = im.split()
    r = r.point(lambda v: min(255, max(0, int(v * (1 + prm["am"])))))
    b = b.point(lambda v: min(255, max(0, int(v * (1 - prm["am"])))))
    im = Image.merge("RGB", (r, g, b))
    # 5) mo + nhieu
    im = im.filter(ImageFilter.GaussianBlur(prm["mo"]))
    noise = Image.effect_noise(im.size, 40).convert("RGB")
    im = Image.blend(im, noise, prm["nhieu"])
    # 6) kich thuoc
    s = prm["ti_le"]
    nw, nh = max(32, int(im.width * s)), max(32, int(im.height * s))
    if max(nw, nh) > max_side:
        k = max_side / max(nw, nh)
        nw, nh = int(nw * k), int(nh * k)
    im = im.resize((nw, nh), Image.LANCZOS)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    im.save(out_path, format="JPEG", quality=prm["jpeg"], optimize=True)
    out = {k: v for k, v in prm.items() if k != "rnd"}
    out.update(width=nw, height=nh)
    return out


def ensure_query_images(image_ids, ingest_dir=INGEST_DIR, force=False, seed=SEED, quiet=False):
    """Tao anh hoi con thieu cho cac image_id. Tra ve {image_id: duong_dan} cua cac anh tao duoc/da co."""
    out_dir = os.path.join(ingest_dir, OUT_SUBDIR)
    index_path = os.path.join(ingest_dir, "anh_hoi.jsonl")
    index = {}
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    index[r["image_id"]] = r
    done, made, missing = {}, 0, []
    for iid in dict.fromkeys(image_ids):
        src = os.path.join(ingest_dir, "images", f"{iid}.png")
        dst = os.path.join(out_dir, f"{iid}.jpg")
        if not os.path.exists(src):
            missing.append(iid)
            continue
        if force or not os.path.exists(dst):
            prm = make_query_image(src, dst, iid, seed)
            index[iid] = {"image_id": iid, "path": dst.replace("\\", "/"), "nguon": src.replace("\\", "/"), **prm}
            made += 1
        done[iid] = dst
    if made:
        os.makedirs(ingest_dir, exist_ok=True)
        with open(index_path, "w", encoding="utf-8") as f:
            for iid in sorted(index):
                f.write(json.dumps(index[iid], ensure_ascii=False) + "\n")
    if not quiet:
        print(f"Anh hoi: {len(done)} anh ({made} anh moi tao) trong {out_dir}"
              + (f" | thieu hinh cat: {', '.join(missing[:8])}" if missing else ""))
    return done


def image_query_ids(csv_path):
    ids = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("loai") or "").strip().lower() == "image_query" and (row.get("image_id") or "").strip():
                ids.append(row["image_id"].strip())
    return ids


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    p = argparse.ArgumentParser(description="Tao anh hoi (giong anh chup) cho cau hoi la anh")
    p.add_argument("--csv", default=QA_CSV, help="CSV xuat tu sheet QA_Data")
    p.add_argument("--ingest", default=INGEST_DIR)
    p.add_argument("--all", action="store_true", help="tao cho tat ca hinh trong images.jsonl")
    p.add_argument("--ids", nargs="*", help="chi tao cho cac image_id nay")
    p.add_argument("--force", action="store_true", help="tao lai ca anh da co")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args(argv)
    if args.ids:
        ids = args.ids
    elif args.all:
        with open(os.path.join(args.ingest, "images.jsonl"), encoding="utf-8") as f:
            ids = [json.loads(l)["image_id"] for l in f if l.strip()]
    else:
        if not os.path.exists(args.csv):
            sys.exit(f"Khong thay {args.csv} - xuat CSV tu sheet QA_Data truoc (hoac dung --all)")
        ids = image_query_ids(args.csv)
        print(f"{len(ids)} dong image_query trong {args.csv}")
    ensure_query_images(ids, args.ingest, force=args.force, seed=args.seed)


if __name__ == "__main__":
    main()
