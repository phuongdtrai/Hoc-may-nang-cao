# -*- coding: utf-8 -*-
"""
nap_qdrant.py - Giai doan 5: nhung chunk + hinh cua ingestion roi nap vao Qdrant.

Chay tu thu muc goc repo:
    docker compose -f docker/docker-compose.yml up -d qdrant         # Qdrant o http://localhost:6333
    python embedding/nap_qdrant.py
    python embedding/nap_qdrant.py --path data/processed/qdrant_local  # khong co Docker: Qdrant luu trong thu muc

Dau vao: data/processed/ingest/chunks.jsonl, images.jsonl, images/*.png
         (giai nen du_lieu_xu_ly_v2.zip, hoac chay python ingestion/ingest.py)
Dau ra : collection "nhom7_tai_lieu", xoa va tao lai moi lan chay de khong con diem cu khi chay lai ingestion
  moi dong chunks.jsonl -> 1 diem type "text",  vector "text"
  moi dong images.jsonl -> 1 diem type "image", vector "text" (phan chu mo ta hinh) + "image" (CLIP)
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import nhung  # noqa: E402

PAYLOAD_INDEXES = (("type", "keyword"), ("doc_id", "keyword"), ("chunk_id", "keyword"), ("image_id", "keyword"),
                   ("chunk_ids", "keyword"), ("page_number", "integer"), ("pages", "integer"))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Nhung chunk + hinh va nap Qdrant (Giai doan 5, Nhom 7)")
    p.add_argument("--ingest_dir", default="data/processed/ingest", help="thu muc dau ra cua ingestion/ingest.py")
    p.add_argument("--url", default=nhung.QDRANT_URL, help="dia chi Qdrant server")
    p.add_argument("--path", default=None, help="dung Qdrant luu trong thu muc nay thay cho server")
    p.add_argument("--collection", default=nhung.COLLECTION)
    p.add_argument("--text_model", default=nhung.TEXT_MODEL)
    p.add_argument("--image_model", default=nhung.IMAGE_MODEL)
    p.add_argument("--device", default="auto", help="auto | cuda | cpu")
    p.add_argument("--batch", type=int, default=16)
    return p.parse_args(argv)


def main(argv=None):
    nhung.utf8_stdout()
    args = parse_args(argv)
    from qdrant_client import models

    ingest_dir = nhung.repo_path(args.ingest_dir)
    chunks_path = os.path.join(ingest_dir, "chunks.jsonl")
    images_path = os.path.join(ingest_dir, "images.jsonl")
    if not os.path.exists(chunks_path):
        sys.exit(f"Khong thay {chunks_path} - giai nen du_lieu_xu_ly_v2.zip hoac chay python ingestion/ingest.py")
    chunks = nhung.read_jsonl(chunks_path)
    figs = nhung.read_jsonl(images_path) if os.path.exists(images_path) else []
    thieu = [f["image_id"] for f in figs if not os.path.exists(nhung.repo_path(f["image_path"]))]
    if thieu:
        print(f"  !! {len(thieu)} hinh khong co file anh, bo qua: {', '.join(thieu[:5])}{' ...' if len(thieu) > 5 else ''}")
        figs = [f for f in figs if os.path.exists(nhung.repo_path(f["image_path"]))]

    t0 = time.time()
    client = nhung.connect(args.url, args.path)
    try:
        client.get_collections()  # bao loi ngay neu chua bat Qdrant, truoc khi mat vai phut nhung
    except Exception as exc:
        sys.exit(f"Khong ket noi duoc Qdrant {args.url}: {exc}\n"
                 "Bat Qdrant: docker compose -f docker/docker-compose.yml up -d qdrant (hoac dung --path)")
    text_model = nhung.load_text_model(args.text_model, args.device)
    print(f"Nhung chu: {len(chunks)} chunk + {len(figs)} hinh bang {args.text_model} ({text_model.device})")
    texts = [nhung.chunk_embed_text(c) for c in chunks] + [nhung.image_embed_text(f) for f in figs]
    text_vecs = nhung.encode_texts(text_model, texts, args.batch, show_progress=True)
    del text_model

    image_vecs = []
    if figs:
        image_model = nhung.load_image_model(args.image_model, args.device)
        print(f"Nhung anh: {len(figs)} hinh bang {args.image_model} ({image_model.device})")
        image_vecs = nhung.encode_images(image_model, [f["image_path"] for f in figs], args.batch, show_progress=True)
        del image_model

    if client.collection_exists(args.collection):
        client.delete_collection(args.collection)
    vectors_config = {nhung.VEC_TEXT: models.VectorParams(size=len(text_vecs[0]), distance=models.Distance.COSINE)}
    if image_vecs:
        vectors_config[nhung.VEC_IMAGE] = models.VectorParams(size=len(image_vecs[0]),
                                                              distance=models.Distance.COSINE)
    client.create_collection(args.collection, vectors_config=vectors_config)
    if not args.path:  # Qdrant luu trong thu muc khong dung chi muc payload
        for field, schema in PAYLOAD_INDEXES:
            client.create_payload_index(args.collection, field_name=field, field_schema=schema)

    points = [models.PointStruct(id=nhung.point_id(c["chunk_id"]), vector={nhung.VEC_TEXT: v},
                                 payload=nhung.chunk_payload(c))
              for c, v in zip(chunks, text_vecs)]
    for i, f in enumerate(figs):
        vector = {nhung.VEC_TEXT: text_vecs[len(chunks) + i], nhung.VEC_IMAGE: image_vecs[i]}
        points.append(models.PointStruct(id=nhung.point_id(f["image_id"]), vector=vector,
                                         payload=nhung.image_payload(f)))
    for i in range(0, len(points), 128):
        client.upsert(args.collection, points=points[i:i + 128])

    n = client.count(args.collection, exact=True).count
    client.close()
    where = os.path.normpath(nhung.repo_path(args.path)) if args.path else args.url
    print(f"Da nap {n} diem ({len(chunks)} text, {len(figs)} image) vao '{args.collection}' tai {where} "
          f"trong {time.time() - t0:.0f} giay")
    if n != len(points):
        sys.exit(f"!! So diem trong Qdrant ({n}) khac so diem da gui ({len(points)})")
    print("Tiep theo: python embedding/danh_gia_truy_hoi.py" + (f" --path {args.path}" if args.path else ""))


if __name__ == "__main__":
    main()
