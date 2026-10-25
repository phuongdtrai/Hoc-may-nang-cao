# -*- coding: utf-8 -*-
"""
eval_truoc_sau.py - Giai doan 4: so sanh Qwen2-VL-2B goc va ban da fine-tune.

Chay tu thu muc goc repo, sau khi train xong:
    python finetune/eval_truoc_sau.py                 # ca 2 mo hinh, toan bo tap val + cau anh
    python finetune/eval_truoc_sau.py --limit 10      # thu nhanh 10 cau
    python finetune/eval_truoc_sau.py --chi finetune  # chi chay lai mo hinh fine-tune

Dau vao: data/processed/llm_val.jsonl (cau chua dung khi train)
         data/processed/vlm_val.jsonl   (cau anh CHUA dung khi train - do kha nang nhin anh; bao cao tach 2 nhom:
                                         dap an la hinh (loai image) va cau hoi la anh (loai image_query))
         data/processed/vlm_train.jsonl (cau hoi ve anh thuoc tap train, lay toi da --anh_toi_da cau de minh hoa)
Dau ra (finetune/outputs/danh_gia/):
  du_doan_goc.jsonl, du_doan_finetune.jsonl  cau tra loi cua tung mo hinh (lan sau dung lai, khong sinh lai)
  so_sanh_truoc_sau.csv                      tung cau: dap an chuan, 2 cau tra loi, diem (mo bang Excel)
  bao_cao_danh_gia.md                        bang tong hop + vi du, dua vao bao cao cuoi ky

Cach cham: so cau tra loi voi dap an chuan theo tung am tiet (chu thuong, bo dau cau):
  F1      - can bang giua dung va du (chi so chinh)
  ROUGE-L - muc trung khop theo thu tu tu
  Do phu  - ti le am tiet cua dap an chuan xuat hien trong cau tra loi
"""
import argparse
import csv
import gc
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import prompt_format as pf  # noqa: E402

UNSLOTH_MODEL = "unsloth/Qwen2-VL-2B-Instruct-unsloth-bnb-4bit"
DEFAULT_ADAPTER = "finetune/outputs/qwen2vl-2b-nhom7-lora"
GROUPS = {"val_ctx": "Val – có ngữ cảnh", "val_noctx": "Val – không ngữ cảnh",
          "anh_moi": "Ảnh – đáp án là hình, tập val", "anh_hoi_moi": "Ảnh – câu hỏi là ảnh, tập val",
          "anh": "Ảnh – thuộc tập train"}
GROUPS_ASCII = {"val_ctx": "val co ngu canh", "val_noctx": "val khong ngu canh",
                "anh_moi": "anh dap an la hinh, tap val", "anh_hoi_moi": "cau hoi la anh, tap val",
                "anh": "anh thuoc tap train"}
GROUP_ORDER = ("val_ctx", "val_noctx", "anh_moi", "anh_hoi_moi", "anh")


def anh_ca_trang(rec):
    """Cau anh vong 1 (anh ca trang PDF, khong co image_id)."""
    meta = rec.get("meta", {})
    return not meta.get("image_id") and meta.get("source") != "qa_anh"
MODELS = {"goc": "goc", "finetune": "fine-tune"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Danh gia truoc/sau fine-tune - Giai doan 4, Nhom 7")
    p.add_argument("--adapter", default=DEFAULT_ADAPTER, help="thu muc LoRA adapter da train")
    p.add_argument("--base", default=None, help="mo hinh goc (mac dinh: doc tu adapter_config.json)")
    p.add_argument("--val", default="data/processed/llm_val.jsonl")
    p.add_argument("--anh_val", default="data/processed/vlm_val.jsonl",
                   help="cau hoi anh CHUA dung khi train ('' = bo qua)")
    p.add_argument("--anh", default="data/processed/vlm_train.jsonl", help="cau hoi anh thuoc tap train ('' = bo qua)")
    p.add_argument("--anh_toi_da", type=int, default=20,
                   help="so cau anh thuoc tap train toi da (chon deu trong file; 0 = tat ca)")
    p.add_argument("--out_dir", default="finetune/outputs/danh_gia")
    p.add_argument("--chi", choices=["ca_hai", "goc", "finetune"], default="ca_hai", help="chay mo hinh nao")
    p.add_argument("--limit", type=int, default=0, help="chi lay N cau dau (0 = tat ca)")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--rep_penalty", type=float, default=1.05)
    p.add_argument("--max_pixels", type=int, default=pf.DEFAULT_MAX_PIXELS)
    p.add_argument("--lam_lai", action="store_true", help="sinh lai cau tra loi, khong dung ket qua cu")
    p.add_argument("--backend", choices=["unsloth", "hf"], default="unsloth")
    p.add_argument("--no_4bit", action="store_true", help="nap 16-bit (backend hf)")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Cham diem
# ---------------------------------------------------------------------------
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def words(s):
    s = unicodedata.normalize("NFC", s or "").lower().replace("_", " ")
    return _PUNCT.sub(" ", s).split()


def f1_score(pred, ref):
    p, r = words(pred), words(ref)
    if not p or not r:
        return float(p == r)
    same = sum((Counter(p) & Counter(r)).values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(r)
    return 2 * prec * rec / (prec + rec)


def recall_score(pred, ref):
    p, r = words(pred), words(ref)
    if not r:
        return float(not p)
    return sum((Counter(p) & Counter(r)).values()) / len(r)


def rouge_l(pred, ref):
    p, r = words(pred), words(ref)
    if not p or not r:
        return float(p == r)
    prev = [0] * (len(r) + 1)
    for a in p:
        cur = [0]
        for j, b in enumerate(r, 1):
            cur.append(prev[j - 1] + 1 if a == b else max(prev[j], cur[j - 1]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(p), lcs / len(r)
    return 2 * prec * rec / (prec + rec)


def score(pred, ref):
    return {"f1": f1_score(pred, ref), "rougeL": rouge_l(pred, ref), "recall": recall_score(pred, ref),
            "len": len(words(pred))}


# ---------------------------------------------------------------------------
# Du lieu
# ---------------------------------------------------------------------------
def question_of(rec):
    if pf.is_image_record(rec):
        return rec["question"]
    text = rec["input"]
    i = text.rfind("Câu hỏi:")
    return text[i + len("Câu hỏi:"):].strip() if i >= 0 else text


def spread(rows, n):
    """Chon n phan tu cach deu nhau (giu thu tu) - de lay mau cau anh tap train tu nhieu tai lieu."""
    if n <= 0 or len(rows) <= n:
        return rows
    return [rows[round(i * (len(rows) - 1) / (n - 1))] for i in range(n)] if n > 1 else rows[:1]


def load_items(args):
    items = []
    sources = [(args.val, None)]
    if args.anh_val:
        sources.append((args.anh_val, "anh_moi"))
    if args.anh:
        sources.append((args.anh, "anh"))
    for path, forced in sources:
        if not path:
            continue
        if not os.path.exists(path):
            if forced:
                print(f"  (chua co {path} - bo qua nhom {GROUPS_ASCII[forced]})")
                continue
            sys.exit(f"Khong tim thay {path}. Chay build_training_data.py (Buoc 2.4) truoc.")
        got = []
        for rec in pf.read_jsonl(path):
            if pf.check_record(rec):
                continue
            if pf.is_image_record(rec):
                found = pf.resolve_image_path(rec["image"], (ROOT,))
                if not found:
                    print(f"  !! khong thay anh {rec['image']} - bo qua")
                    continue
                rec = dict(rec, image=found)
                group = forced or "anh"
                if group == "anh_moi" and rec.get("meta", {}).get("loai") == "image_query":
                    group = "anh_hoi_moi"
            else:
                group = "val_ctx" if rec.get("instruction") == pf.INSTR_CTX else "val_noctx"
            ident = group + "\n" + pf.user_text(rec)
            if pf.is_image_record(rec):  # doi anh (vd sua so trang) -> sinh lai cau tra loi
                ident += "\n" + os.path.basename(rec["image"])
            key = hashlib.sha1(ident.encode("utf-8")).hexdigest()[:16]
            got.append({"key": key, "group": group, "rec": rec})
        if forced == "anh" and args.anh_toi_da > 0:
            # giu du cac cau anh ca trang (so sanh duoc voi vong 1), phan con lai lay mau deu tu cau co image_id
            old = [g for g in got if anh_ca_trang(g["rec"])]
            new = [g for g in got if not anh_ca_trang(g["rec"])]
            got = old + spread(new, max(0, args.anh_toi_da - len(old)))
        if args.limit:
            got = got[:args.limit] if forced is None else got[:max(1, args.limit // 5)]
        items += got
    return items


# ---------------------------------------------------------------------------
# Mo hinh
# ---------------------------------------------------------------------------
def adapter_base(adapter_dir):
    path = os.path.join(adapter_dir, "adapter_config.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("base_model_name_or_path")
    return None


def model_signature(args, which, base):
    if which == "goc":
        ident = base
    else:
        f = os.path.join(args.adapter, "adapter_model.safetensors")
        st = os.stat(f) if os.path.exists(f) else None
        ident = f"{os.path.abspath(args.adapter)}|{st.st_size if st else 0}|{int(st.st_mtime) if st else 0}"
    raw = json.dumps([ident, args.max_new_tokens, args.rep_penalty, args.max_pixels, pf.SYSTEM_PROMPT,
                      args.backend, args.no_4bit])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def load_model(args, which, base):
    """which = 'goc' | 'finetune'. Tra ve (model, processor, device)."""
    if args.backend == "unsloth":
        from unsloth import FastVisionModel

        name = base if which == "goc" else args.adapter
        model, processor = FastVisionModel.from_pretrained(name, load_in_4bit=not args.no_4bit)
        FastVisionModel.for_inference(model)
        return model, processor, "cuda"

    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else (torch.float16 if cuda else torch.float32)
    from train_qwen2vl import dtype_kwargs
    kw = dict(attn_implementation="sdpa", **dtype_kwargs(dtype))
    if cuda:
        kw["device_map"] = {"": 0}
    if not args.no_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype, llm_int8_skip_modules=["visual", "lm_head"])
    model = Qwen2VLForConditionalGeneration.from_pretrained(base, **kw)
    proc_src = args.adapter if (which == "finetune" and os.path.exists(
        os.path.join(args.adapter, "tokenizer_config.json"))) else base
    processor = AutoProcessor.from_pretrained(proc_src)
    if which == "finetune":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    return model, processor, ("cuda" if cuda else "cpu")


def generate(model, processor, device, rec, args):
    import torch

    msgs = pf.build_messages(rec, with_answer=False, max_pixels=args.max_pixels)
    images = [pf.load_image(p["image"], args.max_pixels) for p in pf.message_images(msgs)]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    kw = dict(text=[text], return_tensors="pt", add_special_tokens=False)
    if images:
        kw["images"] = images
    inputs = processor(**kw).to(device)
    tok = processor.tokenizer
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                             temperature=None, top_p=None, top_k=None,
                             repetition_penalty=args.rep_penalty, use_cache=True,
                             pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    new = out[0, inputs["input_ids"].shape[1]:]
    return tok.decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()


def free_gpu():
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_model(args, which, base, items):
    """Sinh cau tra loi cho tung cau, dung lai ket qua cu neu cung cau hinh."""
    path = os.path.join(args.out_dir, f"du_doan_{which}.jsonl")
    sig = model_signature(args, which, base)
    cache = {}
    if os.path.exists(path) and not args.lam_lai:
        for row in pf.read_jsonl(path):
            if row.get("sig") == sig:
                cache[row["key"]] = row
    todo = [it for it in items if it["key"] not in cache]
    print(f"\n[{MODELS[which]}] {len(items) - len(todo)} cau da co ket qua, can sinh {len(todo)} cau")
    if todo:
        model, processor, device = load_model(args, which, base)
        mode = "a" if (cache and not args.lam_lai) else "w"
        t_start = time.time()
        with open(path, mode, encoding="utf-8") as f:
            for i, it in enumerate(todo, 1):
                t = time.time()
                ans = generate(model, processor, device, it["rec"], args)
                row = {"key": it["key"], "sig": sig, "answer": ans, "seconds": round(time.time() - t, 2)}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                cache[it["key"]] = row
                if i % 10 == 0 or i == len(todo):
                    eta = (time.time() - t_start) / i * (len(todo) - i)
                    print(f"  {i}/{len(todo)} cau | con khoang {eta / 60:.1f} phut")
        del model, processor  # giai phong VRAM truoc khi nap mo hinh thu hai
        free_gpu()
    return cache


# ---------------------------------------------------------------------------
# Bao cao
# ---------------------------------------------------------------------------
def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def fmt(x, pct=True):
    if x != x:  # NaN
        return "-"
    return f"{100 * x:.1f}" if pct else f"{x:.1f}"


def build_rows(items, preds):
    rows = []
    for it in items:
        rec, ref = it["rec"], pf.answer_text(it["rec"])
        meta = rec.get("meta", {})
        row = {"key": it["key"], "group": it["group"], "doc_id": meta.get("doc_id", ""), "page": meta.get("page", ""),
               "question": question_of(rec), "ref": ref}
        for m in MODELS:
            p = preds.get(m, {}).get(it["key"])
            if p is None:
                continue
            row[f"ans_{m}"] = p["answer"]
            row[f"sec_{m}"] = p.get("seconds")
            for k, v in score(p["answer"], ref).items():
                row[f"{k}_{m}"] = v
        rows.append(row)
    return rows


def write_csv(rows, path):
    cols = [("group", "nhom"), ("doc_id", "doc_id"), ("page", "trang"), ("question", "cau_hoi"),
            ("ref", "dap_an_chuan"), ("ans_goc", "tra_loi_goc"), ("ans_finetune", "tra_loi_finetune"),
            ("f1_goc", "f1_goc"), ("f1_finetune", "f1_finetune"), ("rougeL_goc", "rougeL_goc"),
            ("rougeL_finetune", "rougeL_finetune"), ("recall_goc", "do_phu_goc"),
            ("recall_finetune", "do_phu_finetune"), ("sec_goc", "giay_goc"), ("sec_finetune", "giay_finetune")]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig: Excel doc dung tieng Viet
        w = csv.writer(f)
        w.writerow([c for _, c in cols])
        for r in rows:
            out = []
            for k, _ in cols:
                v = r.get(k, "")
                if k == "group":
                    v = GROUPS.get(v, v)
                elif isinstance(v, float) and not k.startswith("sec_"):
                    v = round(100 * v, 1)
                out.append(v)
            w.writerow(out)


def summary_line(name, rs, both):
    cells = [name, str(len(rs))]
    for metric in ("f1", "rougeL", "recall"):
        for m in MODELS:
            cells.append(fmt(mean(r[f"{metric}_{m}"] for r in rs if f"{metric}_{m}" in r)))
    for m in MODELS:
        cells.append(fmt(mean(r[f"len_{m}"] for r in rs if f"len_{m}" in r), pct=False))
    if both:
        win = sum(1 for r in rs if r["f1_finetune"] > r["f1_goc"] + 0.05)
        loss = sum(1 for r in rs if r["f1_finetune"] < r["f1_goc"] - 0.05)
        cells.append(f"{win} / {len(rs) - win - loss} / {loss}")
    else:
        cells.append("-")
    return "| " + " | ".join(cells) + " |"


def short(s, n=400):
    s = " ".join((s or "").split())
    if not s:
        return "(trống)"
    return s if len(s) <= n else s[:n].rstrip() + " …"


def write_report(rows, args, base, path):
    both = all("f1_goc" in r and "f1_finetune" in r for r in rows) and rows
    L = ["# Đánh giá trước/sau fine-tune — Qwen2-VL-2B (Nhóm 7)", ""]
    L.append(f"- Thời điểm chạy: {time.strftime('%Y-%m-%d %H:%M')}")
    L.append(f"- Mô hình gốc: `{base}`")
    L.append(f"- Adapter fine-tune: `{args.adapter}`")
    L.append(f"- Sinh câu trả lời: greedy, tối đa {args.max_new_tokens} token, repetition_penalty {args.rep_penalty}")
    L.append("- Cách chấm: so với đáp án chuẩn theo âm tiết (chữ thường, bỏ dấu câu). "
             "F1 là chỉ số chính; Độ phủ = tỉ lệ âm tiết của đáp án chuẩn có trong câu trả lời.")
    L.append("")
    info_path = os.path.join(args.adapter, "thong_tin_train.json")
    if os.path.exists(info_path):
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
        vb, va = info.get("val_loss_truoc"), info.get("val_loss_sau")
        L.append("## Quá trình train")
        L.append("")
        L.append("| Chỉ số | Giá trị |")
        L.append("|---|---|")
        if vb is not None and va is not None:
            L.append(f"| Val loss trước → sau | {vb:.4f} → {va:.4f} |")
        L.append(f"| Số bước / thời gian train | {info.get('so_buoc')} bước / {info.get('thoi_gian_train_phut')} phút |")
        if info.get("vram_dinh_gb") is not None:
            L.append(f"| VRAM đỉnh (PyTorch) | {info['vram_dinh_gb']} GB trên {(info.get('gpu') or {}).get('ten', '?')} |")
        tp = info.get("tham_so", {})
        if tp:
            L.append(f"| Tham số LoRA được train | {tp.get('trainable', 0):,} ({tp.get('ti_le_phan_tram')}% tổng) |")
        L.append("")
    L.append("## Tổng hợp (điểm 0–100)")
    L.append("")
    L.append("| Tập | Số câu | F1 gốc | F1 FT | ROUGE-L gốc | ROUGE-L FT | Độ phủ gốc | Độ phủ FT | "
             "Độ dài TB gốc | Độ dài TB FT | FT thắng / hòa / thua |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    by_group = defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
    for g in ("val_ctx", "val_noctx"):
        if by_group[g]:
            L.append(summary_line(GROUPS[g], by_group[g], both))
    val_rows = by_group["val_ctx"] + by_group["val_noctx"]
    if val_rows:
        L.append(summary_line("**Val – tất cả**", val_rows, both))
    for g in ("anh_moi", "anh_hoi_moi", "anh"):
        if by_group[g]:
            L.append(summary_line(GROUPS[g], by_group[g], both))
    L.append("")
    L.append("Độ dài TB tính bằng số âm tiết. Thắng/thua: F1 của bản fine-tune cao/thấp hơn bản gốc quá 5 điểm.")
    L.append("")

    if val_rows:
        L.append("## Theo tài liệu (tập val)")
        L.append("")
        L.append("| doc_id | Số câu | F1 gốc | F1 FT |")
        L.append("|---|---|---|---|")
        by_doc = defaultdict(list)
        for r in val_rows:
            by_doc[r["doc_id"]].append(r)
        for d in sorted(by_doc):
            rs = by_doc[d]
            L.append(f"| {d} | {len(rs)} | {fmt(mean(r['f1_goc'] for r in rs if 'f1_goc' in r))} | "
                     f"{fmt(mean(r['f1_finetune'] for r in rs if 'f1_finetune' in r))} |")
        L.append("")

    secs = {m: mean(r[f"sec_{m}"] for r in rows if r.get(f"sec_{m}") is not None) for m in MODELS}
    if any(v == v for v in secs.values()):
        L.append("## Thời gian sinh câu trả lời (giây/câu, GPU hiện tại)")
        L.append("")
        L.append(f"| Gốc | Fine-tune |\n|---|---|\n| {fmt(secs['goc'], pct=False)} | {fmt(secs['finetune'], pct=False)} |")
        L.append("")

    if both:
        ranked = sorted(rows, key=lambda r: r["f1_finetune"] - r["f1_goc"])
        sections = [("Ví dụ fine-tune cải thiện rõ nhất", [r for r in reversed(ranked) if r["f1_finetune"] > r["f1_goc"]][:5]),
                    ("Ví dụ fine-tune kém hơn bản gốc", [r for r in ranked if r["f1_finetune"] < r["f1_goc"]][:3])]
        for title, rs in sections:
            if not rs:
                continue
            L.append(f"## {title}")
            L.append("")
            for i, r in enumerate(rs, 1):
                L.append(f"**{i}. [{r['doc_id']} tr.{r['page']} · {GROUPS[r['group']]}] {short(r['question'], 300)}**")
                L.append("")
                L.append(f"- Đáp án chuẩn: {short(r['ref'])}")
                L.append(f"- Gốc (F1 {fmt(r['f1_goc'])}): {short(r['ans_goc'])}")
                L.append(f"- Fine-tune (F1 {fmt(r['f1_finetune'])}): {short(r['ans_finetune'])}")
                L.append("")

    L.append("## Ghi chú khi đưa vào báo cáo")
    L.append("")
    L.append("- Tập val gồm các câu chưa dùng khi train, chia theo trang nên không trùng ngữ cảnh với tập train.")
    if by_group["anh_moi"]:
        L.append(f"- Nhóm \"{GROUPS['anh_moi']}\" ({len(by_group['anh_moi'])} câu, `vlm_val.jsonl`): câu hỏi có đáp án "
                 "là một hình cắt từ tài liệu, chia theo trang và không dùng khi train. Đây là nhóm đo đúng khả năng "
                 "đọc hình/sơ đồ của mô hình. Mỗi câu kèm 'Ngữ cảnh' là trang, mục, chú thích của hình.")
    if by_group["anh_hoi_moi"]:
        L.append(f"- Nhóm \"{GROUPS['anh_hoi_moi']}\" ({len(by_group['anh_hoi_moi'])} câu, `vlm_val.jsonl`): người dùng "
                 "gửi ảnh chụp một hình trong tài liệu (ảnh đã làm nghiêng, mờ, nén JPEG cho giống ảnh chụp) rồi hỏi. "
                 "'Ngữ cảnh' gồm mô tả hình tìm được và đoạn văn cùng trang, giống lúc backend trả về kết quả tìm bằng ảnh.")
    if by_group["anh"]:
        L.append(f"- Nhóm \"{GROUPS['anh']}\" ({len(by_group['anh'])} câu lấy từ `vlm_train.jsonl`) chỉ để minh hoạ: "
                 "với adapter đã train trên các câu này, điểm cao chỉ cho thấy mô hình đã học được dữ liệu đó.")
    L.append("- Điểm F1/ROUGE-L đo mức trùng từ ngữ, nên câu trả lời đúng ý nhưng diễn đạt khác vẫn bị điểm thấp. "
             "Nên đọc thêm các ví dụ ở trên và file so_sanh_truoc_sau.csv.")
    L.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    base = args.base or adapter_base(args.adapter) or UNSLOTH_MODEL
    if args.chi != "goc" and not os.path.exists(os.path.join(args.adapter, "adapter_config.json")):
        sys.exit(f"Chua co adapter o {args.adapter}. Chay finetune/train_qwen2vl.py truoc (hoac dung --chi goc).")

    items = load_items(args)
    counts = Counter(it["group"] for it in items)
    print(f"Danh gia {len(items)} cau: " + ", ".join(f"{GROUPS_ASCII[g]} {n}" for g, n in counts.items()))
    print(f"Mo hinh goc: {base} | adapter: {args.adapter} | backend: {args.backend}")

    preds = {}
    order = ["goc", "finetune"] if args.chi == "ca_hai" else [args.chi]
    if args.backend == "unsloth":
        import unsloth  # noqa: F401  (import truoc transformers)
    for which in order:
        preds[which] = run_model(args, which, base, items)
    # nap ket qua cu cua mo hinh con lai (neu co) de bao cao van so sanh duoc
    for which in MODELS:
        if which not in preds:
            path = os.path.join(args.out_dir, f"du_doan_{which}.jsonl")
            if os.path.exists(path):
                sig = model_signature(args, which, base)
                preds[which] = {r["key"]: r for r in pf.read_jsonl(path) if r.get("sig") == sig}

    rows = build_rows(items, preds)
    csv_path = os.path.join(args.out_dir, "so_sanh_truoc_sau.csv")
    md_path = os.path.join(args.out_dir, "bao_cao_danh_gia.md")
    write_csv(rows, csv_path)
    write_report(rows, args, base, md_path)

    print("\n" + "=" * 70)
    for g in GROUP_ORDER:
        rs = [r for r in rows if r["group"] == g]
        if not rs:
            continue
        parts = [f"{MODELS[m]} {fmt(mean(r['f1_' + m] for r in rs if 'f1_' + m in r))}" for m in MODELS
                 if any(f"f1_{m}" in r for r in rs)]
        print(f"F1 {GROUPS_ASCII[g]} ({len(rs)} cau): " + " | ".join(parts))
    print(f"Chi tiet: {csv_path}\nBao cao: {md_path}")


if __name__ == "__main__":
    main()
