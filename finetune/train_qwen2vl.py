
import argparse
import csv
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import prompt_format as pf  # noqa: E402

UNSLOTH_MODEL = "unsloth/Qwen2-VL-2B-Instruct-unsloth-bnb-4bit"
HF_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
DEFAULT_TRAIN = ["data/processed/llm_train.jsonl", "data/processed/vlm_train.jsonl"]
OPTIONAL_TRAIN = {"data/processed/vlm_train.jsonl"}
DEFAULT_VAL = "data/processed/llm_val.jsonl"
DEFAULT_OUT = "finetune/outputs/qwen2vl-2b-nhom7-lora"
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
VISION_TARGETS = ["qkv", "attn.proj", "fc1", "fc2"]  # chi dung khi --finetune_vision (backend hf)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Fine-tune Qwen2-VL-2B bang QLoRA - Giai doan 4, Nhom 7")
    p.add_argument("--model", default=None,
                   help=f"mo hinh goc (mac dinh {UNSLOTH_MODEL}; backend hf: {HF_MODEL})")
    p.add_argument("--train", nargs="+", default=DEFAULT_TRAIN, help="cac file .jsonl dung de train")
    p.add_argument("--val", default=DEFAULT_VAL, help="file .jsonl validation ('' = khong dung)")
    p.add_argument("--output_dir", default=DEFAULT_OUT)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--max_steps", type=int, default=-1, help="> 0: chi chay so buoc nay (de chay thu)")
    p.add_argument("--batch", type=int, default=2, help="so mau moi lan dua vao GPU")
    p.add_argument("--grad_accum", type=int, default=4, help="so lan tich luy gradient truoc moi buoc")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=float, default=0.05, help="< 1: ti le so buoc; >= 1: so buoc")
    p.add_argument("--max_len", type=int, default=2048, help="do dai toi da (token) cua 1 mau")
    p.add_argument("--max_pixels", type=int, default=pf.DEFAULT_MAX_PIXELS,
                   help="so pixel toi da cua 1 anh (1024*28*28 ~ 1024 token anh)")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--finetune_vision", action="store_true", help="train ca phan thi giac (can nhieu anh)")
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--no_eval", action="store_true", help="khong tinh val loss")
    p.add_argument("--resume", action="store_true", help="train tiep tu checkpoint gan nhat trong output_dir")
    p.add_argument("--save_merged", action="store_true", help="luu them ban gop 16-bit (~4.5 GB)")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--backend", choices=["unsloth", "hf"], default="unsloth",
                   help="hf = transformers + peft + bitsandbytes, du phong khi Unsloth loi")
    p.add_argument("--no_4bit", action="store_true", help="nap mo hinh 16-bit thay vi 4-bit")
    p.add_argument("--dry_run", action="store_true", help="chi kiem tra du lieu, do dai va prompt")
    args = p.parse_args(argv)
    if args.model is None:
        args.model = HF_MODEL if args.backend == "hf" else UNSLOTH_MODEL
    return args


def setup_console():
    """Tranh loi UnicodeEncodeError khi chuyen log ra file tren Windows."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ---------------------------------------------------------------------------
# Du lieu
# ---------------------------------------------------------------------------
def load_records(paths, search_dirs=(ROOT,)):
    records, problems = [], []
    for path in paths:
        if not path:
            continue
        if not os.path.exists(path):
            if path.replace("\\", "/") in OPTIONAL_TRAIN:
                print(f"  (chua co {path} - train khong co mau anh)")
                continue
            sys.exit(f"Khong tim thay file du lieu: {path}\n-> Chay build_training_data.py (Buoc 2.4) truoc.")
        for i, rec in enumerate(pf.read_jsonl(path), 1):
            err = pf.check_record(rec)
            if err is None and pf.is_image_record(rec):
                found = pf.resolve_image_path(rec["image"], search_dirs)
                if found is None:
                    err = f"khong tim thay anh '{rec['image']}'"
                else:
                    rec = dict(rec, image=found)
            if err:
                problems.append(f"{os.path.basename(path)} dong {i}: {err}")
            else:
                records.append(rec)
    return records, problems


def describe(records):
    n_img = sum(1 for r in records if pf.is_image_record(r))
    n_ctx = sum(1 for r in records if not pf.is_image_record(r) and r.get("instruction") == pf.INSTR_CTX)
    return {"tong": len(records), "co_ngu_canh": n_ctx,
            "khong_ngu_canh": len(records) - n_img - n_ctx, "anh": n_img}


def token_length(processor, messages, max_pixels):
    """So token cua 1 mau sau khi processor chen token anh."""
    from PIL import Image

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    tok = getattr(processor, "tokenizer", processor)
    n = len(tok(text, add_special_tokens=False)["input_ids"])
    for part in pf.message_images(messages):
        img = part["image"]
        if isinstance(img, str):
            with Image.open(img) as im:
                w, h = im.size
        else:
            w, h = img.size
        n += pf.image_token_count(w, h, part.get("max_pixels", max_pixels)) - 1  # 1 <|image_pad|> -> N token
    return n


def percentile(sorted_vals, q):
    if not sorted_vals:
        return 0
    k = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[k]


def filter_by_length(processor, records, max_len, max_pixels, name):
    kept, dropped, lengths = [], [], []
    for r in records:
        n = token_length(processor, pf.build_messages(r, max_pixels=max_pixels), max_pixels)
        lengths.append(n)
        (kept if n <= max_len else dropped).append(r)
    s = sorted(lengths)
    stats = {"min": s[0] if s else 0, "trung_vi": percentile(s, 0.5), "p95": percentile(s, 0.95),
             "max": s[-1] if s else 0}
    print(f"  {name}: do dai token min {stats['min']} | trung vi {stats['trung_vi']} | "
          f"p95 {stats['p95']} | max {stats['max']}")
    if dropped:
        print(f"  !! Bo {len(dropped)} mau {name} dai hon --max_len {max_len} "
              f"(tang --max_len hoac giam --max_pixels neu muon giu)")
    return kept, dropped, stats


def to_dataset(records, max_pixels):
    """Moi phan tu: {"messages": [...]} - dinh dang ma UnslothVisionDataCollator doc."""
    return [{"messages": pf.build_messages(r, max_pixels=max_pixels)} for r in records]


class Qwen2VLCollator:
    """Collator cho backend hf (va cho --dry_run): chi giu nhan o phan tra loi cua assistant.

    Ket qua giong UnslothVisionDataCollator(train_on_responses_only=True): nhan = cau tra loi
    + <|im_end|>; system, user, token anh va padding deu bi gan -100.
    """

    def __init__(self, processor, max_pixels=pf.DEFAULT_MAX_PIXELS):
        self.processor = processor
        self.max_pixels = max_pixels
        tok = processor.tokenizer
        self.im_start = tok.convert_tokens_to_ids("<|im_start|>")
        self.im_end = tok.convert_tokens_to_ids("<|im_end|>")
        self.header = tok.encode("assistant\n", add_special_tokens=False)

    def __call__(self, examples):
        import torch

        texts, images = [], []
        for ex in examples:
            msgs = pf.clean_messages(ex["messages"])
            texts.append(self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))
            for part in pf.message_images(msgs):
                images.append(pf.load_image(part["image"], part.get("max_pixels", self.max_pixels)))
        kwargs = dict(text=texts, padding=True, return_tensors="pt", add_special_tokens=False)
        if images:
            kwargs["images"] = images
        batch = self.processor(**kwargs)
        ids = batch["input_ids"]
        labels = torch.full_like(ids, -100)
        h, L = self.header, len(self.header)
        for i, row in enumerate(ids.tolist()):
            j, n = 0, len(row)
            while j < n:
                if row[j] == self.im_start and row[j + 1:j + 1 + L] == h:
                    start = end = j + 1 + L
                    while end < n and row[end] != self.im_end:
                        end += 1
                    stop = min(end + 1, n)  # giu ca <|im_end|> de mo hinh hoc diem dung
                    labels[i, start:stop] = ids[i, start:stop]
                    j = stop
                else:
                    j += 1
        if "attention_mask" in batch:
            labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# Mo hinh
# ---------------------------------------------------------------------------
def load_unsloth(args):
    from unsloth import FastVisionModel  # phai import truoc transformers/trl/peft

    model, processor = FastVisionModel.from_pretrained(
        args.model,
        load_in_4bit=not args.no_4bit,
        max_seq_length=args.max_len,
        use_gradient_checkpointing="unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=args.finetune_vision,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        random_state=args.seed,
        use_gradient_checkpointing="unsloth",
    )
    return model, processor


def dtype_kwargs(dtype):
    import transformers
    from packaging import version

    key = "dtype" if version.parse(transformers.__version__) >= version.parse("4.56.0") else "torch_dtype"
    return {key: dtype}


def load_hf(args):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else (torch.float16 if cuda else torch.float32)
    kw = dict(attn_implementation="sdpa", **dtype_kwargs(dtype))
    if cuda:
        kw["device_map"] = {"": 0}
    if not args.no_4bit:
        if not cuda:
            sys.exit("Luong tu hoa 4-bit can GPU NVIDIA. Them --no_4bit de chay tren CPU.")
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype, llm_int8_skip_modules=["visual", "lm_head"])
    model = Qwen2VLForConditionalGeneration.from_pretrained(args.model, **kw)
    processor = AutoProcessor.from_pretrained(args.model)
    model.config.use_cache = False
    model.enable_input_require_grads()
    targets = LORA_TARGETS + (VISION_TARGETS if args.finetune_vision else [])
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      bias="none", target_modules=targets, task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    return model, processor


def trainable_params(model):
    if hasattr(model, "get_nb_trainable_parameters"):
        trainable, total = model.get_nb_trainable_parameters()
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
    return {"trainable": int(trainable), "tong": int(total), "ti_le_phan_tram": round(100 * trainable / max(total, 1), 3)}


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
def build_config(config_cls, values):
    """Tao SFTConfig/TrainingArguments, tu doi ten tham so giua cac phien ban TRL/transformers."""
    import dataclasses

    fields = {f.name for f in dataclasses.fields(config_cls)}
    aliases = {"max_length": ["max_seq_length"], "eval_strategy": ["evaluation_strategy"]}
    kw = {}
    for key, val in values.items():
        if key not in fields:
            key = next((a for a in aliases.get(key, []) if a in fields), None)
            if key is None:
                continue
        kw[key] = val
    return config_cls(**kw)


def schedule(args, n_train):
    steps_per_epoch = math.ceil(math.ceil(n_train / args.batch) / args.grad_accum)
    total = args.max_steps if args.max_steps > 0 else math.ceil(steps_per_epoch * args.epochs)
    warmup = max(1, round(total * args.warmup)) if args.warmup < 1 else int(args.warmup)
    return total, warmup


def training_values(args, n_train, do_eval, cuda, bf16):
    total, warmup = schedule(args, n_train)
    full_run = args.max_steps <= 0
    return dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        warmup_steps=warmup,
        lr_scheduler_type="linear",
        weight_decay=0.01,
        max_grad_norm=1.0,
        optim="adamw_8bit" if cuda else "adamw_torch",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        eval_strategy="epoch" if (do_eval and full_run) else "no",
        prediction_loss_only=True,  # chi can val loss, khong gom logits (rat ton bo nho)
        save_strategy="epoch" if full_run else "no",
        save_total_limit=2,
        bf16=bf16,
        fp16=cuda and not bf16,
        seed=args.seed,
        data_seed=args.seed,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    ), total


def make_sft_trainer(args, model, processor, collator, train_ds, val_ds, values):
    from trl import SFTConfig, SFTTrainer

    cfg = build_config(SFTConfig, dict(
        values,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_num_proc=1,
        max_length=args.max_len,
    ))
    kw = dict(model=model, processing_class=processor, data_collator=collator, args=cfg)
    try:
        return SFTTrainer(train_dataset=train_ds, eval_dataset=val_ds, **kw)
    except TypeError as e:
        if "Dataset" not in str(e):
            raise
        # TRL moi chi nhan datasets.Dataset (Unsloth thuong tu va loi nay); chuyen doi roi thu lai
        from datasets import Dataset
        print("  (TRL yeu cau datasets.Dataset - chuyen danh sach mau sang Dataset)")
        return SFTTrainer(train_dataset=Dataset.from_list(train_ds),
                          eval_dataset=Dataset.from_list(val_ds) if val_ds else None, **kw)


def build_trainer_unsloth(args, model, processor, train_ds, val_ds, values):
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator

    FastVisionModel.for_training(model)
    collator = UnslothVisionDataCollator(
        model, processor,
        max_seq_length=args.max_len,
        resize="max",  # anh da duoc gioi han bang max_pixels trong messages
        train_on_responses_only=True,  # chi tinh loss tren cau tra loi
        instruction_part=INSTRUCTION_PART,
        response_part=RESPONSE_PART,
    )
    return make_sft_trainer(args, model, processor, collator, train_ds, val_ds, values)


def build_trainer_hf(args, model, processor, train_ds, val_ds, values):
    from transformers import Trainer, TrainingArguments

    processor.tokenizer.padding_side = "right"
    cfg = build_config(TrainingArguments, dict(
        values, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}))
    return Trainer(model=model, args=cfg, train_dataset=train_ds, eval_dataset=val_ds,
                   data_collator=Qwen2VLCollator(processor, args.max_pixels))


# ---------------------------------------------------------------------------
# Ket qua
# ---------------------------------------------------------------------------
def write_loss_log(log_history, out_dir, val_before=None):
    rows = [h for h in log_history if "loss" in h or "eval_loss" in h]
    if val_before is not None:  # trainer.train() xoa log cu, nen them lai val loss ban dau o buoc 0
        rows = [{"step": 0, "epoch": 0.0, "eval_loss": val_before}] + rows
    path = os.path.join(out_dir, "loss_log.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "epoch", "loss", "eval_loss", "learning_rate", "grad_norm"])
        for h in rows:
            w.writerow([h.get("step"), h.get("epoch"), h.get("loss"), h.get("eval_loss"),
                        h.get("learning_rate"), h.get("grad_norm")])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return path, None
    tr = [(h["step"], h["loss"]) for h in rows if "loss" in h and h.get("step") is not None]
    ev = [(h["step"], h["eval_loss"]) for h in rows if "eval_loss" in h and h.get("step") is not None]
    fig, ax = plt.subplots(figsize=(7, 4))
    if tr:
        ax.plot(*zip(*tr), label="train loss")
    if ev:
        ax.plot(*zip(*ev), "o-", label="val loss")
    ax.set_xlabel("Bước (step)")
    ax.set_ylabel("loss")
    ax.set_title("Qwen2-VL-2B QLoRA – Nhóm 7")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    png = os.path.join(out_dir, "loss.png")
    fig.savefig(png, dpi=120)
    plt.close(fig)
    return path, png


def gpu_info():
    import torch

    if not torch.cuda.is_available():
        return None
    p = torch.cuda.get_device_properties(0)
    return {"ten": p.name, "vram_gb": round(p.total_memory / 1024 ** 3, 2),
            "compute_capability": f"{p.major}.{p.minor}"}


def dry_run(args, train_records, val_records):
    from transformers import AutoProcessor

    print(f"\n[dry_run] Nap processor tu {args.model} (khong nap trong so mo hinh)...")
    processor = AutoProcessor.from_pretrained(args.model)
    train_records, dropped, _ = filter_by_length(processor, train_records, args.max_len, args.max_pixels, "train")
    if val_records:
        filter_by_length(processor, val_records, args.max_len, args.max_pixels, "val")
    sample = next((r for r in train_records if not pf.is_image_record(r)), train_records[0])
    msgs = pf.build_messages(sample, max_pixels=args.max_pixels)
    print("\n--- Prompt mau dua vao mo hinh (dung chat template Qwen2-VL) ---")
    print(processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))
    batch = Qwen2VLCollator(processor, args.max_pixels)([{"messages": msgs}])
    lab = batch["labels"][0]
    kept = batch["input_ids"][0][lab != -100]
    print("--- Phan duoc tinh loss (chi cau tra loi) ---")
    print(processor.tokenizer.decode(kept))
    imgs = [r for r in train_records if pf.is_image_record(r)]
    if imgs:
        b = Qwen2VLCollator(processor, args.max_pixels)([{"messages": pf.build_messages(imgs[0], max_pixels=args.max_pixels)}])
        print(f"\nMau anh: {os.path.basename(imgs[0]['image'])} -> {int(b['image_grid_thw'].prod() // 4)} token anh, "
              f"tong {b['input_ids'].shape[1]} token")
    print("\n[dry_run] Du lieu va prompt on. Bo --dry_run de train.")


def main(argv=None):
    setup_console()
    args = parse_args(argv)
    t0 = time.time()

    # 1) du lieu
    print("=" * 70)
    print("GIAI DOAN 4 - Fine-tune Qwen2-VL-2B (QLoRA) | Nhom 7")
    print("=" * 70)
    train_records, problems = load_records(args.train)
    val_records, vproblems = load_records([args.val]) if (args.val and not args.no_eval) else ([], [])
    for msg in (problems + vproblems)[:20]:
        print("  !! bo qua", msg)
    if not train_records:
        sys.exit("Khong co mau train hop le.")
    d_train, d_val = describe(train_records), describe(val_records)
    print(f"Train: {d_train['tong']} mau ({d_train['co_ngu_canh']} co ngu canh, "
          f"{d_train['khong_ngu_canh']} khong ngu canh, {d_train['anh']} anh) | Val: {d_val['tong']} mau")

    if args.dry_run:
        dry_run(args, train_records, val_records)
        return

    # 2) mo hinh
    if args.backend == "unsloth":
        model, processor = load_unsloth(args)  # import unsloth truoc torch/transformers
    import torch

    cuda = torch.cuda.is_available()
    if args.backend == "hf":
        model, processor = load_hf(args)
    gpu = gpu_info()
    if gpu:
        print(f"GPU: {gpu['ten']} | VRAM {gpu['vram_gb']} GB | compute {gpu['compute_capability']}")
    params = trainable_params(model)
    print(f"Tham so LoRA duoc train: {params['trainable']:,} / {params['tong']:,} ({params['ti_le_phan_tram']}%)")

    # 3) do dai + dataset
    train_records, dropped, len_stats = filter_by_length(processor, train_records, args.max_len, args.max_pixels, "train")
    if val_records:
        val_records, _, _ = filter_by_length(processor, val_records, args.max_len, args.max_pixels, "val")
    train_ds = to_dataset(train_records, args.max_pixels)
    val_ds = to_dataset(val_records, args.max_pixels) if val_records else None

    bf16 = cuda and torch.cuda.is_bf16_supported()
    values, total_steps = training_values(args, len(train_ds), val_ds is not None, cuda, bf16)
    build = build_trainer_unsloth if args.backend == "unsloth" else build_trainer_hf
    trainer = build(args, model, processor, train_ds, val_ds, values)
    print(f"Lich train: {total_steps} buoc | batch {args.batch} x tich luy {args.grad_accum} = "
          f"{args.batch * args.grad_accum} mau/buoc | lr {args.lr} | {'bf16' if bf16 else 'fp16/fp32'}")

    # 4) val loss truoc khi train
    val_before = None
    if val_ds is not None:
        try:
            print("\nTinh val loss cua mo hinh goc...")
            val_before = trainer.evaluate().get("eval_loss")
            print(f"  val loss truoc fine-tune: {val_before:.4f}")
        except Exception as e:  # khong de buoc phu lam hong ca lan train
            print(f"  (bo qua val loss ban dau: {type(e).__name__}: {e})")
        if args.backend == "unsloth":
            from unsloth import FastVisionModel
            FastVisionModel.for_training(model)

    # 5) train
    resume = None
    if args.resume:
        from transformers.trainer_utils import get_last_checkpoint
        resume = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
        print(f"Train tiep tu: {resume or '(khong co checkpoint - train tu dau)'}")
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    interrupted = False
    t_train = time.time()
    try:
        result = trainer.train(resume_from_checkpoint=resume)
        train_loss = result.training_loss
    except KeyboardInterrupt:
        interrupted = True
        train_loss = None
        print("\nDung giua chung (Ctrl+C) - van luu adapter hien tai.")
    except torch.cuda.OutOfMemoryError:
        print("\n!! HET VRAM (CUDA out of memory). Thu lan luot:\n"
              "   --batch 1 --grad_accum 8\n"
              "   --max_pixels 602112   (768*28*28, anh nho hon)\n"
              "   --max_len 1536\n"
              "   va tat cac ung dung dung GPU khac (trinh duyet, game...).")
        sys.exit(1)
    train_minutes = (time.time() - t_train) / 60
    peak_vram = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2) if cuda else None

    # 6) val loss sau khi train
    val_after = None
    if val_ds is not None and not interrupted:
        try:
            val_after = trainer.evaluate().get("eval_loss")
        except Exception as e:
            print(f"  (khong tinh duoc val loss sau train: {type(e).__name__}: {e})")

    # 7) luu
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    loss_csv, loss_png = write_loss_log(trainer.state.log_history, args.output_dir, val_before)
    info = {
        "thoi_diem": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backend": args.backend,
        "mo_hinh_goc": args.model,
        "cau_hinh": {k: v for k, v in vars(args).items()},
        "du_lieu": {"train": describe(train_records), "val": describe(val_records),
                    "bo_vi_qua_dai": len(dropped), "do_dai_token_train": len_stats},
        "tham_so": params,
        "so_buoc": trainer.state.global_step,
        "train_loss_trung_binh": train_loss,
        "val_loss_truoc": val_before,
        "val_loss_sau": val_after,
        "thoi_gian_train_phut": round(train_minutes, 1),
        "vram_dinh_gb": peak_vram,
        "gpu": gpu,
        "dung_giua_chung": interrupted,
        "system_prompt": pf.SYSTEM_PROMPT,
    }
    with open(os.path.join(args.output_dir, "thong_tin_train.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    if args.save_merged and not interrupted:
        out = args.output_dir.rstrip("/\\")
        merged = (out[:-5] if out.endswith("-lora") else out) + "-merged"
        if args.backend == "unsloth":
            print(f"\nLuu ban gop 16-bit vao {merged} ...")
            model.save_pretrained_merged(merged, processor, save_method="merged_16bit")
        else:
            print(f"\n(backend hf) Gop bang: python finetune/export_model.py --adapter {args.output_dir}")

    print("\n" + "=" * 70)
    print(f"XONG sau {(time.time() - t0) / 60:.1f} phut (train {train_minutes:.1f} phut, {trainer.state.global_step} buoc)")
    if val_before is not None and val_after is not None:
        print(f"Val loss: {val_before:.4f} -> {val_after:.4f}")
    if peak_vram is not None:
        print(f"VRAM dinh (PyTorch): {peak_vram} GB")
    print(f"Adapter: {args.output_dir}")
    print(f"Loss: {loss_csv}" + (f" , {loss_png}" if loss_png else ""))
    print("Buoc tiep: python finetune/eval_truoc_sau.py")


if __name__ == "__main__":
    main()
