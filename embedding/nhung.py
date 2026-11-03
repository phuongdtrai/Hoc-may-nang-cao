# -*- coding: utf-8 -*-
"""
nhung.py - Giai doan 5: phan dung chung cho nap Qdrant, tim kiem va danh gia.

  - Mo hinh nhung chu: BAAI/bge-m3 (da ngon ngu, doc tot tieng Viet, 1024 chieu, khong can tien to "query:").
    Dung cho ca chunk van ban lan phan chu mo ta hinh, nen cau hoi chu tim duoc ca chunk lan hinh.
  - Mo hinh nhung anh: CLIP ViT-B/32 (512 chieu), chi dung de so anh voi anh:
    anh nguoi dung chup  ->  hinh cat trong tai lieu (cau hoi la anh).
  - Moi diem Qdrant: {id, vector: {"text": ..., "image": ... (chi hinh)}, payload: nguyen 1 dong
    chunks.jsonl hoac images.jsonl + type "text" | "image"} (data contract o HUONG_DAN_GIAI_DOAN_3.md, muc 3).

Can: pip install -r requirements-embed.txt
"""
import json
import os
import sys
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COLLECTION = "nhom7_tai_lieu"
QDRANT_URL = "http://localhost:6333"          # docker/docker-compose.yml
TEXT_MODEL = "BAAI/bge-m3"
IMAGE_MODEL = "clip-ViT-B-32"                 # sentence-transformers tai tu Hugging Face
TEXT_MAX_TOKENS = 1024                        # chunk dai nhat ~1.700 ky tu, mo ta hinh ~1.300 ky tu
VEC_TEXT = "text"
VEC_IMAGE = "image"

# Phan chu cua 1 hinh dem di nhung, dung thu tu goi y trong HUONG_DAN_GIAI_DOAN_3.md (muc 3).
# Hinh it chu nen section quan trong: bo section thi Recall@1 tim hinh bang chu giam 66,1% -> 57,4%.
IMAGE_TEXT_FIELDS = ("caption_hint", "section", "context_before", "context_after", "text_in_image", "ocr_text")


def utf8_stdout():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def repo_path(path):
    """Duong dan trong jsonl tinh tu thu muc goc repo (vd data/processed/ingest/images/x.png)."""
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def point_id(key):
    """chunk_id / image_id -> UUID co dinh, nap lai van ra dung id cu."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"nhom7/{key}"))


def chunk_embed_text(chunk):
    """Chu dem di nhung cua 1 chunk: chi noi dung. Them duong dan muc vao truoc lam Recall@1 tim chu giam
    80,9% -> 77,2% (nhieu chunk chung 1 muc, nhat la Triet hoc), xem HUONG_DAN_GIAI_DOAN_5.md muc 6."""
    return (chunk.get("text") or "").strip()


def image_embed_text(fig):
    """Chu dem di nhung cua 1 hinh: caption_hint + section + context_before + context_after + text_in_image + ocr_text."""
    parts = []
    for key in IMAGE_TEXT_FIELDS:
        s = " ".join(str(fig.get(key) or "").split())
        if s and s not in parts:
            parts.append(s)
    return "\n".join(parts) or f"Hình ở trang {fig.get('page_number')} của {fig.get('doc_id')}"


def chunk_payload(chunk):
    # chunks.jsonl da co type = text | table -> giu lai o chunk_type, type cua Qdrant luon la "text"
    return dict(chunk, type="text", chunk_type=chunk.get("type", "text"))


def image_payload(fig):
    return dict(fig, type="image")


def pick_device(device=None):
    if device and device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_text_model(name=TEXT_MODEL, device=None):
    from sentence_transformers import SentenceTransformer

    device = pick_device(device)
    model = SentenceTransformer(name, device=device)
    model.max_seq_length = min(model.max_seq_length or TEXT_MAX_TOKENS, TEXT_MAX_TOKENS)
    if device.startswith("cuda"):
        model.half()  # bge-m3 fp16 chiem ~1,1 GB VRAM
    return model


def load_image_model(name=IMAGE_MODEL, device=None):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device=pick_device(device))


def encode_texts(model, texts, batch_size=16, show_progress=False):
    vecs = model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=show_progress)
    return vecs.astype("float32").tolist()


def open_image(path_or_image):
    from PIL import Image

    img = path_or_image if isinstance(path_or_image, Image.Image) else Image.open(repo_path(path_or_image))
    return img.convert("RGB") if img.mode != "RGB" else img


def encode_images(model, images, batch_size=16, show_progress=False):
    vecs = []
    for i in range(0, len(images), batch_size):
        batch = [open_image(x) for x in images[i:i + batch_size]]
        vecs.extend(model.encode(batch, batch_size=batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=show_progress))
    return [v.astype("float32").tolist() for v in vecs]


def connect(url=QDRANT_URL, path=None):
    """Qdrant server (Docker) theo url, hoac Qdrant chay ngay trong Python luu o thu muc path (khong can Docker)."""
    from qdrant_client import QdrantClient

    if path:
        return QdrantClient(path=repo_path(path))
    return QdrantClient(url=url, timeout=60)
