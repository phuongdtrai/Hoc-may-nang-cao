# -*- coding: utf-8 -*-
"""
check_env.py - Giai doan 4: kiem tra moi truong truoc khi train.

    python finetune/check_env.py

Kiem tra lan luot: Python, PyTorch + CUDA (co ho tro GPU Blackwell RTX 50xx khong), bitsandbytes 4-bit,
Unsloth, transformers/trl/peft, du lieu Buoc 2.4 va dung luong o dia. Moi dong FAIL kem cach sua.
"""
import importlib
import os
import platform
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = []


def report(ok, name, detail="", fix=""):
    RESULTS.append(ok)
    tag = "OK  " if ok else ("WARN" if ok is None else "FAIL")
    print(f"[{tag}] {name}" + (f": {detail}" if detail else ""))
    if fix and ok is not True:
        print(f"       -> {fix}")


def version_of(mod):
    try:
        return importlib.import_module(mod).__version__
    except Exception:
        return None


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    print(f"He dieu hanh: {platform.system()} {platform.release()} | Python {platform.python_version()}")
    py = sys.version_info
    report((3, 10) <= (py.major, py.minor) <= (3, 13), "Python", platform.python_version(),
           "Dung Python 3.12 (Unsloth ho tro 3.10-3.13)")

    # PyTorch + GPU
    try:
        import torch
    except Exception as e:
        report(False, "PyTorch", str(e), "Cai torch ban CUDA, xem HUONG_DAN_GIAI_DOAN_4.md muc 2")
        return finish()
    report(True, "PyTorch", f"{torch.__version__} (CUDA build {torch.version.cuda})")
    if torch.cuda.is_available():
        check_gpu(torch)
    else:
        report(False, "GPU CUDA", "torch khong thay GPU",
               "Cap nhat driver NVIDIA; cai lai torch ban cu128/cu130 (khong phai ban cpu)")
    check_libs_and_data()
    return finish()


def check_gpu(torch):
    p = torch.cuda.get_device_properties(0)
    cap = f"sm_{p.major}{p.minor}"
    free, total = torch.cuda.mem_get_info()
    report(True, "GPU", f"{p.name} | {total / 1024**3:.1f} GB VRAM (trong {free / 1024**3:.1f} GB) | {cap}")
    archs = torch.cuda.get_arch_list()
    if p.major >= 12:
        report(cap in archs or "sm_120" in archs, "Ho tro Blackwell (sm_120)", " ".join(archs[-4:]),
               "Can torch dung CUDA >= 12.8: pip install torch torchvision --index-url "
               "https://download.pytorch.org/whl/cu130")
    try:
        x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        (x @ x).sum().item()
        report(True, "Phep tinh bf16 tren GPU")
    except Exception as e:
        report(False, "Phep tinh tren GPU", str(e).splitlines()[0],
               "Loi 'no kernel image' = torch qua cu cho GPU nay, cai lai torch cu128/cu130")
    report(torch.cuda.is_bf16_supported() or None, "bfloat16", str(torch.cuda.is_bf16_supported()),
           "GPU khong ho tro bf16 - script se tu dung fp16")
    if free / 1024**3 < 6:
        report(None, "VRAM trong", f"chi con {free / 1024**3:.1f} GB",
               "Tat trinh duyet/game/app dung GPU truoc khi train")

    # bitsandbytes 4-bit
    try:
        import bitsandbytes as bnb
        lin = bnb.nn.Linear4bit(64, 64, compute_dtype=torch.bfloat16, quant_type="nf4").cuda()
        lin(torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)).sum().item()
        report(True, "bitsandbytes 4-bit", bnb.__version__)
    except Exception as e:
        report(False, "bitsandbytes 4-bit", f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}",
               "pip install -U bitsandbytes (>= 0.45.5 cho RTX 50xx)")

    # Unsloth (import truoc transformers)
    try:
        import unsloth  # noqa: F401
        from unsloth import FastVisionModel  # noqa: F401
        report(True, "Unsloth", version_of("unsloth") or "?")
    except Exception as e:
        report(False, "Unsloth", f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}",
               "pip install unsloth (hoac dung Docker unsloth/unsloth). Tam thoi co the train bang --backend hf")


def check_libs_and_data():
    for mod in ("transformers", "trl", "peft", "accelerate", "datasets"):
        v = version_of(mod)
        report(v is not None, mod, v or "chua cai", f"pip install {mod}")
    report(version_of("matplotlib") is not None or None, "matplotlib (ve bieu do loss)",
           version_of("matplotlib") or "chua cai", "pip install matplotlib (khong bat buoc)")

    # Du lieu
    sys.path.insert(0, HERE)
    import prompt_format as pf
    for rel, need in (("data/processed/llm_train.jsonl", True), ("data/processed/llm_val.jsonl", True),
                      ("data/processed/vlm_train.jsonl", False)):
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            report(False if need else None, rel, "chua co", "Chay python build_training_data.py (Buoc 2.4)")
            continue
        recs = pf.read_jsonl(path)
        bad = sum(1 for r in recs if pf.check_record(r))
        missing = sum(1 for r in recs if pf.is_image_record(r) and not pf.resolve_image_path(r["image"], (ROOT,)))
        report(bad == 0 and missing == 0, rel, f"{len(recs)} mau" + (f", {bad} loi" if bad else "")
               + (f", thieu {missing} anh" if missing else ""), "Chay lai build_training_data.py")

    # O dia
    free_gb = shutil.disk_usage(ROOT).free / 1024**3
    report(free_gb >= 15 or None, "O dia trong", f"{free_gb:.0f} GB",
           "Can ~3 GB tai mo hinh + ~5 GB ban gop + ~3 GB GGUF")


def finish():
    fails = sum(1 for r in RESULTS if r is False)
    print("\n" + ("San sang train: python finetune/train_qwen2vl.py --max_steps 10" if fails == 0
                  else f"Con {fails} muc FAIL - sua theo goi y o tren roi chay lai."))
    return fails


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
