# -*- coding: utf-8 -*-
"""
export_model.py - Giai doan 4: gop LoRA adapter vao Qwen2-VL-2B thanh 1 mo hinh 16-bit day du.

Chay tu thu muc goc repo, sau khi train:
    python finetune/export_model.py                     # gop bang transformers + peft (chay ca tren CPU, can ~10 GB RAM)
    python finetune/export_model.py --kiem_tra          # gop xong hoi thu 1 cau trong tap val
    python finetune/export_model.py --backend unsloth   # gop bang Unsloth (GPU)

Dau ra: finetune/outputs/qwen2vl-2b-nhom7-merged/  (safetensors ~4.5 GB + processor + chat template)
Ban gop nay dung de: chuyen sang GGUF chay bang llama.cpp (xem HUONG_DAN_GIAI_DOAN_4.md, muc 6),
hoac nap thang bang transformers ma khong can peft.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import prompt_format as pf  # noqa: E402

DEFAULT_ADAPTER = "finetune/outputs/qwen2vl-2b-nhom7-lora"
DEFAULT_OUT = "finetune/outputs/qwen2vl-2b-nhom7-merged"
HF_BASE = "Qwen/Qwen2-VL-2B-Instruct"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Gop LoRA vao Qwen2-VL-2B (Giai doan 4)")
    p.add_argument("--adapter", default=DEFAULT_ADAPTER)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--base", default=None, help="mo hinh goc 16-bit (mac dinh suy ra tu adapter_config.json)")
    p.add_argument("--backend", choices=["hf", "unsloth"], default="hf")
    p.add_argument("--gguf", default=None, help="(backend unsloth) them ban GGUF, vd q8_0 - chi nen dung tren Linux/WSL/Docker")
    p.add_argument("--kiem_tra", action="store_true", help="gop xong nap lai va tra loi thu 1 cau val")
    p.add_argument("--val", default="data/processed/llm_val.jsonl")
    return p.parse_args(argv)


def base_16bit(adapter_dir, override=None):
    """Adapter train tren ban 4-bit -> gop vao ban 16-bit cung kien truc."""
    if override:
        return override
    name = None
    cfg = os.path.join(adapter_dir, "adapter_config.json")
    if os.path.exists(cfg):
        with open(cfg, encoding="utf-8") as f:
            name = json.load(f).get("base_model_name_or_path")
    if not name:
        return HF_BASE
    for suffix in ("-unsloth-bnb-4bit", "-bnb-4bit"):
        if name.endswith(suffix):
            return name[: -len(suffix)]  # vd unsloth/Qwen2-VL-2B-Instruct (ban 16-bit)
    return name


def merge_hf(args):
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from train_qwen2vl import dtype_kwargs

    base = base_16bit(args.adapter, args.base)
    print(f"Nap mo hinh goc 16-bit: {base} (CPU)")
    model = Qwen2VLForConditionalGeneration.from_pretrained(base, low_cpu_mem_usage=True,
                                                            **dtype_kwargs(torch.bfloat16))
    print(f"Gan adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    print(f"Luu ban gop vao {args.out} ...")
    model.save_pretrained(args.out, safe_serialization=True)
    src = args.adapter if os.path.exists(os.path.join(args.adapter, "tokenizer_config.json")) else base
    AutoProcessor.from_pretrained(src).save_pretrained(args.out)
    return base


def merge_unsloth(args):
    from unsloth import FastVisionModel

    model, processor = FastVisionModel.from_pretrained(args.adapter, load_in_4bit=True)
    print(f"Luu ban gop 16-bit vao {args.out} ...")
    model.save_pretrained_merged(args.out, processor, save_method="merged_16bit")
    if args.gguf:
        gdir = args.out.rstrip("/\\") + "-gguf"
        try:
            model.save_pretrained_gguf(gdir, processor, quantization_method=args.gguf)
            print(f"GGUF: {gdir}")
        except Exception as e:
            print(f"!! Khong tao duoc GGUF bang Unsloth ({type(e).__name__}: {e}).\n"
                  "   Dung cach thu cong voi llama.cpp o muc 6 cua HUONG_DAN_GIAI_DOAN_4.md.")
    return base_16bit(args.adapter, args.base)


def smoke_test(args):
    """Nap ban gop bang transformers thuan va hoi 1 cau val."""
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from train_qwen2vl import dtype_kwargs

    if not os.path.exists(args.val):
        print("(bo qua kiem tra: khong co file val)")
        return
    rec = next(r for r in pf.read_jsonl(args.val) if not pf.check_record(r))
    cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else (torch.float16 if cuda else torch.float32)
    kw = dtype_kwargs(dtype)
    if cuda:
        kw["device_map"] = {"": 0}
    model = Qwen2VLForConditionalGeneration.from_pretrained(args.out, **kw).eval()
    processor = AutoProcessor.from_pretrained(args.out)
    msgs = pf.build_messages(rec, with_answer=False)
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt", add_special_tokens=False).to(model.device)
    t = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False, temperature=None, top_p=None,
                             top_k=None, repetition_penalty=1.05)
    ans = processor.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    q = rec["input"].rsplit("Câu hỏi:", 1)[-1].strip()
    print(f"\n[kiem tra] ({time.time() - t:.1f} giay)\n  Cau hoi : {q}\n  Dap an  : {pf.answer_text(rec)}\n  Mo hinh : {ans}")


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    args = parse_args(argv)
    if not os.path.exists(os.path.join(args.adapter, "adapter_config.json")):
        sys.exit(f"Khong thay adapter o {args.adapter}. Chay finetune/train_qwen2vl.py truoc.")
    t = time.time()
    base = merge_unsloth(args) if args.backend == "unsloth" else merge_hf(args)
    info = {"adapter": os.path.abspath(args.adapter), "mo_hinh_goc_16bit": base, "backend": args.backend,
            "thoi_diem": time.strftime("%Y-%m-%d %H:%M:%S"), "system_prompt": pf.SYSTEM_PROMPT}
    with open(os.path.join(args.out, "nguon_goc_ban_gop.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"Xong sau {(time.time() - t) / 60:.1f} phut: {args.out}")
    if args.kiem_tra:
        smoke_test(args)


if __name__ == "__main__":
    main()
