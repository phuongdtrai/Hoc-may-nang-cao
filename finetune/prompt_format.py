# -*- coding: utf-8 -*-
"""
prompt_format.py - Giai doan 4: "hop dong prompt" dung chung cho train, danh gia va backend.

Moi noi goi mo hinh da fine-tune deu phai dung cac ham o day de prompt luc chay
giong het prompt luc train:
  - finetune/train_qwen2vl.py    (train)
  - finetune/eval_truoc_sau.py   (danh gia truoc/sau)
  - backend Giai doan 6          (goi build_rag_messages)

File nay chi dung thu vien chuan (+ Pillow khi doc anh), khong import torch/transformers.
"""
import json
import math
import os

# ---------------------------------------------------------------------------
# 1. Prompt
# ---------------------------------------------------------------------------
# System prompt co dinh. Backend Giai doan 6 phai gui DUNG chuoi nay.
SYSTEM_PROMPT = (
    "Bạn là trợ lý hỏi đáp tài liệu học tập. Hãy trả lời bằng tiếng Việt, ngắn gọn và chính xác. "
    "Khi có ngữ cảnh hoặc hình ảnh, chỉ dựa vào đó để trả lời."
)

# Ba cau chi dan, khop voi build_training_data.py (Buoc 2.4)
INSTR_CTX = "Trả lời câu hỏi dựa trên ngữ cảnh sau."
INSTR_NOCTX = "Trả lời câu hỏi sau."
INSTR_IMAGE = "Trả lời câu hỏi dựa trên hình ảnh sau."
INSTR_IMAGE_CTX = "Trả lời câu hỏi dựa trên hình ảnh và ngữ cảnh sau."


def image_context(caption="", page=None, section="", caption_source="", slide=None):
    """Dong "Ngữ cảnh" mo ta 1 hinh cat tu tai lieu (lay tu images.jsonl cua ingestion, Giai doan 3).

    Dung chung cho build_training_data.py (cau hoi co dap an la hinh) va backend Giai doan 6:
    khi tra ve 1 hinh, them dong nay vao contexts roi goi build_rag_messages(question, contexts, images).
        image_context(r["caption_hint"], r["page_number"], r["section"], r["caption_source"], r["slide_number"])
    Vd: 'Hình ảnh ở trang 7, mục "1.1.1. Xử lý ảnh là gì?", chú thích: Hình 1.2. Các bước cơ bản ...'
    """
    caption = " ".join((caption or "").split())
    parts = [f"Hình ảnh ở trang {page}" if page else "Hình ảnh"]
    if slide:
        parts[0] += f" (slide {slide})"
    sec = " ".join((section or "").split(" > ")[-1].split())
    if sec and sec != caption:
        parts.append(f'mục "{sec[:100]}"')
    if caption:
        label = "tiêu đề slide" if caption_source == "tieu_de_slide" else "chú thích"
        parts.append(f"{label}: {caption}")
    text = ", ".join(parts)
    return text if text.endswith((".", "?", "!", "…")) else text + "."


def image_query_contexts(fig, page_chunk=""):
    """Ngữ cảnh cho CÂU HỎI LÀ ẢNH (nguoi dung gui anh chup 1 hinh trong tai lieu roi hoi).

    fig: ban ghi images.jsonl cua hinh khop voi anh nguoi dung (backend tim bang anh/OCR o Giai doan 6),
    page_chunk: doan van cung trang voi hinh (chunk Qdrant). Tra ve list de dua vao build_rag_messages:
        build_rag_messages(cau_hoi, image_query_contexts(fig, chunk), [anh_nguoi_dung])
    build_training_data.py dung dung ham nay cho cac cau loai image_query.
    """
    out = [image_context(fig.get("caption_hint", ""), fig.get("page_number"), fig.get("section", ""),
                         fig.get("caption_source", ""), fig.get("slide_number"))]
    page_chunk = " ".join((page_chunk or "").split())
    if page_chunk:
        out.append(page_chunk)
    return out


# ---------------------------------------------------------------------------
# 2. Anh: Qwen2-VL cat anh thanh o 28x28 pixel, moi o la 1 token anh
# ---------------------------------------------------------------------------
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
DEFAULT_MAX_PIXELS = 1024 * 28 * 28  # toi da ~1024 token anh cho 1 trang tai lieu


def smart_resize(height, width, factor=IMAGE_FACTOR, min_pixels=MIN_PIXELS, max_pixels=DEFAULT_MAX_PIXELS):
    """Giong qwen_vl_utils / unsloth_zoo: dua 2 canh ve boi so cua 28, tong pixel trong [min, max]."""
    if height <= 0 or width <= 0:
        raise ValueError(f"Kich thuoc anh khong hop le: {width}x{height}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"Anh qua dai/hep (ti le > 200): {width}x{height}")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def image_token_count(width, height, max_pixels=DEFAULT_MAX_PIXELS):
    """So token anh Qwen2-VL sinh ra cho 1 anh (sau khi gop 2x2)."""
    h, w = smart_resize(height, width, max_pixels=max_pixels)
    return (h // IMAGE_FACTOR) * (w // IMAGE_FACTOR)


def load_image(path_or_image, max_pixels=DEFAULT_MAX_PIXELS):
    """Mo anh, doi sang RGB va thu nho dung nhu luc train (cung thuat toan voi collator Unsloth)."""
    from PIL import Image

    img = path_or_image if isinstance(path_or_image, Image.Image) else Image.open(path_or_image)
    if img.mode != "RGB":
        img = img.convert("RGB")
    h, w = smart_resize(img.height, img.width, max_pixels=max_pixels)
    if (w, h) != img.size:
        img = img.resize((w, h))
    return img


# ---------------------------------------------------------------------------
# 3. Doc du lieu Buoc 2.4
# ---------------------------------------------------------------------------
def read_jsonl(path):
    """Doc file .jsonl, bao loi kem so dong neu co dong hong."""
    records = []
    with open(path, encoding="utf-8-sig") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}, dong {i}: JSON loi ({e})") from None
    return records


def is_image_record(rec):
    return bool(rec.get("image"))


def answer_text(rec):
    return (rec.get("output") if "output" in rec else rec.get("answer", "")) or ""


def user_text(rec):
    """Noi dung chu cua luot user, dung dinh dang Buoc 2.4: instruction + xuong dong + input."""
    if is_image_record(rec):
        instr = rec.get("instruction") or INSTR_IMAGE
        body = rec.get("input") or f"Câu hỏi: {rec['question']}"
        return f"{instr}\n{body}"
    return f"{rec['instruction']}\n{rec['input']}"


def check_record(rec):
    """Tra ve chuoi mo ta loi, hoac None neu ban ghi dung dinh dang."""
    if is_image_record(rec):
        need = ("image", "question", "answer")
    else:
        need = ("instruction", "input", "output")
    for k in need:
        if not isinstance(rec.get(k), str) or not rec[k].strip():
            return f"thieu hoac rong truong '{k}'"
    return None


def resolve_image_path(path, search_dirs=()):
    """Duong dan anh trong vlm_train.jsonl la tuong doi so voi thu muc goc repo."""
    if os.path.isabs(path) and os.path.exists(path):
        return path
    for base in ("",) + tuple(search_dirs):
        cand = os.path.normpath(os.path.join(base, path))
        if os.path.exists(cand):
            return os.path.abspath(cand)
    return None


# ---------------------------------------------------------------------------
# 4. Dung tin nhan (messages) theo chat template Qwen2-VL
# ---------------------------------------------------------------------------
def _text(t):
    return {"type": "text", "text": t}


def build_messages(rec, with_answer=True, image=None, max_pixels=DEFAULT_MAX_PIXELS,
                   system_prompt=SYSTEM_PROMPT):
    """Ban ghi Buoc 2.4 -> danh sach tin nhan [system, user, (assistant)].

    image: anh da mo (PIL) hoac duong dan; mac dinh lay rec["image"].
    """
    user = []
    if is_image_record(rec):
        user.append({"type": "image", "image": image if image is not None else rec["image"],
                     "max_pixels": max_pixels})
    user.append(_text(user_text(rec)))
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": [_text(system_prompt)]})
    messages.append({"role": "user", "content": user})
    if with_answer:
        messages.append({"role": "assistant", "content": [_text(answer_text(rec))]})
    return messages


def build_rag_messages(question, contexts=None, images=None, max_pixels=DEFAULT_MAX_PIXELS,
                       system_prompt=SYSTEM_PROMPT):
    """Dung cho backend Giai doan 6.

    question : cau hoi cua nguoi dung
    contexts : list cac doan van tim duoc tu Qdrant (str), co the rong
    images   : list anh (duong dan hoac PIL) tim duoc, co the rong
    Dinh dang giong het luc train: chi dan + "\\nNgữ cảnh: ...\\nCâu hỏi: ...".
    """
    contexts = [c.strip() for c in (contexts or []) if c and c.strip()]
    images = list(images or [])
    ctx = "\n".join(contexts)
    if images and ctx:
        text = f"{INSTR_IMAGE_CTX}\nNgữ cảnh: {ctx}\nCâu hỏi: {question}"
    elif images:
        text = f"{INSTR_IMAGE}\nCâu hỏi: {question}"
    elif ctx:
        text = f"{INSTR_CTX}\nNgữ cảnh: {ctx}\nCâu hỏi: {question}"
    else:
        text = f"{INSTR_NOCTX}\nCâu hỏi: {question}"
    user = [{"type": "image", "image": im, "max_pixels": max_pixels} for im in images]
    user.append(_text(text))
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": [_text(system_prompt)]})
    messages.append({"role": "user", "content": user})
    return messages


def message_images(messages):
    """Lay cac phan tu anh trong messages theo thu tu xuat hien."""
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            out.extend(p for p in content if isinstance(p, dict) and p.get("type") == "image")
    return out


def to_openai_messages(messages, max_pixels=DEFAULT_MAX_PIXELS):
    """Chuyen messages sang dinh dang OpenAI chat (llama.cpp llama-server, vLLM... deu nhan).

    Anh duoc thu nho dung nhu luc train roi gui dang data URL base64 (PNG).
    """
    import base64
    import io

    out = []
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        parts = []
        for p in content:
            if p.get("type") == "image":
                img = load_image(p["image"], p.get("max_pixels") or max_pixels)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
                parts.append({"type": "image_url", "image_url": {"url": url}})
            else:
                parts.append({"type": "text", "text": p.get("text", "")})
        if all(x["type"] == "text" for x in parts):
            out.append({"role": m["role"], "content": "".join(x["text"] for x in parts)})
        else:
            out.append({"role": m["role"], "content": parts})
    return out


def clean_messages(messages):
    """Bo cac khoa co gia tri None (xuat hien khi messages di qua datasets.Dataset/Arrow).

    Neu khong bo, phan tu chu co {"image": None} se bi chat template hieu nham la anh.
    """
    out = []
    for m in messages:
        m = dict(m)
        if isinstance(m.get("content"), list):
            m["content"] = [{k: v for k, v in p.items() if v is not None} if isinstance(p, dict) else p
                            for p in m["content"]]
        out.append(m)
    return out
