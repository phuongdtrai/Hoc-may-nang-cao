# -*- coding: utf-8 -*-
"""
tim_kiem.py - Giai doan 5-6: tim trong Qdrant bang cau hoi chu hoac bang anh. Backend Giai doan 6 import file nay.

Thu nhanh (tai thu muc goc repo, Qdrant da nap bang nap_qdrant.py):
    python embedding/tim_kiem.py "Sơ đồ tổng quát của một hệ thống xử lý ảnh"
    python embedding/tim_kiem.py "Hình này minh họa gì?" --anh data/processed/ingest/anh_hoi/XL_001_p13_1.jpg
    them --path data/processed/qdrant_local neu nap vao thu muc thay cho server

Trong backend:
    from embedding.tim_kiem import TimKiem
    tk = TimKiem()                                            # hoac TimKiem(path="data/processed/qdrant_local")
    chunks = tk.tim_chu(cau_hoi, k=3, loai="text")            # payload chunks.jsonl + "score"
    hinh = tk.tim_chu(cau_hoi, k=1, loai="image")             # hinh co phan chu mo ta khop cau hoi
    hinh_kem = tk.hinh_cua_chunk([c["chunk_id"] for c in chunks])
    fig = tk.tim_bang_anh(anh_nguoi_dung, k=1)[0]             # cau hoi la anh: hinh trong tai lieu khop anh chup
    doan_van = tk.chunk_cung_trang(fig["doc_id"], fig["page_number"], cau_hoi)
Ket qua la payload nguyen dong chunks.jsonl / images.jsonl, dua thang vao finetune/prompt_format.py
(image_context, image_query_contexts, build_rag_messages) nhu HUONG_DAN_GIAI_DOAN_3.md muc 8.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import nhung  # noqa: E402


class TimKiem:
    def __init__(self, url=nhung.QDRANT_URL, path=None, collection=nhung.COLLECTION, text_model=nhung.TEXT_MODEL,
                 image_model=nhung.IMAGE_MODEL, device=None):
        self.client = nhung.connect(url, path)
        self.collection = collection
        try:
            co = self.client.collection_exists(collection)
        except Exception as exc:
            raise RuntimeError(f"Khong ket noi duoc Qdrant {url}: {exc} - bat Qdrant: "
                               "docker compose -f docker/docker-compose.yml up -d qdrant (hoac dung path=...)") from exc
        if not co:
            raise RuntimeError(f"Qdrant chua co collection '{collection}' - chay python embedding/nap_qdrant.py truoc")
        self.device = nhung.pick_device(device)
        self._ten_model = {"text": text_model, "image": image_model}
        self._model = {}

    def _get_model(self, kind):
        if kind not in self._model:  # chi tai mo hinh khi can (tim bang chu khong can CLIP)
            load = nhung.load_text_model if kind == "text" else nhung.load_image_model
            self._model[kind] = load(self._ten_model[kind], self.device)
        return self._model[kind]

    @staticmethod
    def _filter(loai=None, doc_id=None, extra=()):
        from qdrant_client import models

        must = list(extra)
        if loai:
            must.append(models.FieldCondition(key="type", match=models.MatchValue(value=loai)))
        if doc_id:
            must.append(models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)))
        return models.Filter(must=must) if must else None

    def _query(self, vectors, using, k, flt):
        from qdrant_client import models

        out = []
        for i in range(0, len(vectors), 64):
            reqs = [models.QueryRequest(query=v, using=using, filter=flt, limit=k, with_payload=True)
                    for v in vectors[i:i + 64]]
            for res in self.client.query_batch_points(self.collection, requests=reqs):
                out.append([dict(p.payload, score=p.score) for p in res.points])
        return out

    def tim_chu_nhieu(self, cau_hoi, k=5, loai=None, doc_id=None):
        """Nhieu cau hoi mot luc. loai: "text" (chunk), "image" (hinh) hoac None (ca hai)."""
        vecs = nhung.encode_texts(self._get_model("text"), cau_hoi, batch_size=32)
        return self._query(vecs, nhung.VEC_TEXT, k, self._filter(loai, doc_id))

    def tim_chu(self, cau_hoi, k=5, loai=None, doc_id=None):
        return self.tim_chu_nhieu([cau_hoi], k, loai, doc_id)[0]

    def tim_bang_anh_nhieu(self, anh, k=5):
        """anh: duong dan hoac PIL.Image. Tra ve cac hinh trong tai lieu giong anh nhat."""
        vecs = nhung.encode_images(self._get_model("image"), list(anh))
        return self._query(vecs, nhung.VEC_IMAGE, k, self._filter("image"))

    def tim_bang_anh(self, anh, k=5):
        return self.tim_bang_anh_nhieu([anh], k)[0]

    def hinh_cua_chunk(self, chunk_ids, k=3):
        """Hinh nam trong doan van cua cac chunk tim duoc (truong chunk_ids cua images.jsonl), xep theo thu tu chunk."""
        from qdrant_client import models

        chunk_ids = list(chunk_ids)
        if not chunk_ids:
            return []
        flt = self._filter("image", extra=[models.FieldCondition(key="chunk_ids",
                                                                 match=models.MatchAny(any=chunk_ids))])
        points, _ = self.client.scroll(self.collection, scroll_filter=flt, limit=100, with_payload=True)
        rank = {cid: i for i, cid in enumerate(chunk_ids)}
        figs = [p.payload for p in points]
        figs.sort(key=lambda f: (min(rank[c] for c in f["chunk_ids"] if c in rank), f["image_id"]))
        return figs[:k]

    def chunk_cung_trang(self, doc_id, page, cau_hoi=None):
        """Chunk o cung trang voi 1 hinh (page_chunk cua prompt_format.image_query_contexts).
        Co cau_hoi thi lay chunk cua trang khop cau hoi nhat, khong thi lay chunk dau trang."""
        from qdrant_client import models

        flt = self._filter("text", doc_id, [models.FieldCondition(key="pages", match=models.MatchValue(value=int(page)))])
        if cau_hoi:
            vecs = nhung.encode_texts(self._get_model("text"), [cau_hoi])
            hits = self._query(vecs, nhung.VEC_TEXT, 1, flt)[0]
            return hits[0] if hits else None
        points, _ = self.client.scroll(self.collection, scroll_filter=flt, limit=100, with_payload=True)
        return min((p.payload for p in points), key=lambda c: c["chunk_id"], default=None)

    def close(self):
        self.client.close()


def _dong(hit):
    if hit["type"] == "image":
        mo_ta = hit.get("caption_hint") or hit.get("section") or ""
        ma = hit["image_id"]
    else:
        mo_ta = hit.get("text", "")
        ma = hit["chunk_id"]
    mo_ta = " ".join(mo_ta.split())
    return f"{hit['score']:.3f}  {hit['type']:<5}  {ma:<16} tr.{hit['page_number']:<4} {mo_ta[:90]}"


def main(argv=None):
    nhung.utf8_stdout()
    p = argparse.ArgumentParser(description="Tim trong Qdrant bang cau hoi chu hoac anh (Giai doan 5, Nhom 7)")
    p.add_argument("cau_hoi")
    p.add_argument("--anh", default=None, help="anh nguoi dung gui (cau hoi la anh)")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--loai", choices=["text", "image"], default=None, help="chi tim chunk hoac chi tim hinh")
    p.add_argument("--url", default=nhung.QDRANT_URL)
    p.add_argument("--path", default=None, help="Qdrant luu trong thu muc (nap bang nap_qdrant.py --path)")
    p.add_argument("--collection", default=nhung.COLLECTION)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    tk = TimKiem(args.url, args.path, args.collection, device=args.device)
    if args.anh:
        hinh = tk.tim_bang_anh(args.anh, args.k)
        print(f"Hinh giong anh {args.anh}:")
        for h in hinh:
            print("  " + _dong(h))
        if hinh:
            c = tk.chunk_cung_trang(hinh[0]["doc_id"], hinh[0]["page_number"], args.cau_hoi)
            if c:
                print("Doan van cung trang voi hinh dau tien:\n  " + _dong(c))
    else:
        print(f"Ket qua cho: {args.cau_hoi}")
        hits = tk.tim_chu(args.cau_hoi, args.k, args.loai)
        for h in hits:
            print("  " + _dong(h))
        kem = tk.hinh_cua_chunk([h["chunk_id"] for h in hits if h["type"] == "text"])
        if kem:
            print("Hinh nam trong cac chunk tren:")
            for h in kem:
                print(f"  {h['image_id']:<16} tr.{h['page_number']:<4} {' '.join((h.get('caption_hint') or '').split())[:90]}")
    tk.close()


if __name__ == "__main__":
    main()
