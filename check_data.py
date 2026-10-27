# -*- coding: utf-8 -*-
"""Check that the shared data files are in place and match their published hashes."""

import argparse
import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DOC_IDS = (
    "AI_001",
    "LX_001",
    "LX_002",
    "LX_003",
    "RP_001",
    "SP_001",
    "TL_001",
    "TRIET_001",
    "TTHCM_001",
    "XL_001",
)

HASHES = {
    "tai_lieu_va_qa.zip": "85c570cd591ce99406f70926a0d5fd28cefa3d3b59acbdfa4db3ef99b9ba4963",
    "du_lieu_xu_ly_v2.zip": "131c8dff66ed8ab1583116f91c750ad4dda47c4354d21b0a78595989ae5e1c24",
    "adapter_lora_v2.zip": "ff5e1419d63cf7211b625e51962ce193ceb304693e217697e7e4795ce718af96",
    "qwen2vl-2b-nhom7-q8_0.gguf": "2ee9f8e7e9f2e8c3e9f8c7cb0a400565199e40fe02d557e5d4a32763f6778371",
    "mmproj-qwen2vl-2b-nhom7-f16.gguf": "ee09b928111299277e66d479405d13e818045d0ff779a823ce412f069c3174c6",
}


class Report:
    def __init__(self):
        self.failures = 0
        self.warnings = 0

    def ok(self, message):
        print(f"[OK]   {message}")

    def warn(self, message):
        self.warnings += 1
        print(f"[WARN] {message}")

    def fail(self, message):
        self.failures += 1
        print(f"[FAIL] {message}")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_hash(path, report):
    expected = HASHES[path.name]
    try:
        actual = sha256_file(path)
    except OSError as exc:
        report.fail(f"Không đọc được để kiểm tra SHA256: {path}: {exc}")
        return
    if actual.lower() == expected:
        report.ok(f"SHA256 khớp: {path}")
    else:
        report.fail(
            f"SHA256 không khớp: {path}\n"
            f"       Mong đợi: {expected}\n"
            f"       Thực tế:  {actual}\n"
            "       Hãy tải lại file từ Drive nhóm."
        )


def check_downloads(downloads_dir, repo_dir, report):
    model_targets = {
        "qwen2vl-2b-nhom7-q8_0.gguf": repo_dir / "finetune" / "outputs" / "gguf",
        "mmproj-qwen2vl-2b-nhom7-f16.gguf": repo_dir / "finetune" / "outputs" / "gguf",
    }

    for name in HASHES:
        candidates = [downloads_dir / name]
        if name in model_targets:
            candidates.append(model_targets[name] / name)
        found = [path for path in candidates if path.is_file()]
        if not found:
            report.warn(f"Chưa tìm thấy {name} trong {downloads_dir} hoặc thư mục đích (nếu có).")
            continue
        for path in found:
            check_hash(path, report)


def check_raw_data(repo_dir, report):
    docs_dir = repo_dir / "data" / "raw" / "domain_docs"
    qa_path = docs_dir / "qa_thu_cong.csv"

    if not docs_dir.is_dir():
        report.fail(f"Thiếu thư mục dữ liệu gốc: {docs_dir}")
        return
    report.ok(f"Tìm thấy thư mục dữ liệu gốc: {docs_dir}")

    if not qa_path.is_file():
        report.fail(f"Thiếu {qa_path} — hãy giải nén tai_lieu_va_qa.zip hoặc tải CSV từ sheet QA_Data.")
    else:
        try:
            with qa_path.open("r", encoding="utf-8-sig", newline="") as file:
                reader = csv.DictReader(file)
                required_columns = {"doc_id", "page", "question", "answer", "loai"}
                missing_columns = required_columns.difference(reader.fieldnames or [])
                if missing_columns:
                    report.fail(f"{qa_path} thiếu cột: {', '.join(sorted(missing_columns))}")
                else:
                    row_count = sum(1 for _ in reader)
                    if row_count:
                        report.ok(f"CSV QA_Data đọc được, có {row_count} dòng: {qa_path}")
                    else:
                        report.fail(f"CSV QA_Data không có dòng dữ liệu: {qa_path}")
        except (OSError, UnicodeError, csv.Error) as exc:
            report.fail(f"Không đọc được {qa_path}: {exc}")

    missing_docs = [doc_id for doc_id in DOC_IDS if not (docs_dir / f"{doc_id}.pdf").is_file()]
    if missing_docs:
        report.fail(
            "Thiếu PDF trong data/raw/domain_docs/: "
            + ", ".join(f"{doc_id}.pdf" for doc_id in missing_docs)
        )
    else:
        report.ok(f"Đủ {len(DOC_IDS)} PDF tài liệu trong {docs_dir}")


def check_processed_data(repo_dir, report):
    processed_dir = repo_dir / "data" / "processed"
    ingest_dir = processed_dir / "ingest"
    required_ingest_files = ("chunks.jsonl", "images.jsonl", "pages.jsonl")
    missing_ingest = [name for name in required_ingest_files if not (ingest_dir / name).is_file()]
    if missing_ingest:
        report.warn(
            f"Thiếu dữ liệu ingestion trong {ingest_dir}: {', '.join(missing_ingest)}. "
            "Giải nén du_lieu_xu_ly_v2.zip hoặc chạy python ingestion/ingest.py."
        )
    else:
        report.ok(f"Đủ file ingestion trong {ingest_dir}")

    if not (ingest_dir / "images").is_dir():
        report.warn(
            f"Chưa có thư mục ảnh {ingest_dir / 'images'}; "
            "cần dữ liệu xử lý sẵn hoặc chạy python ingestion/ingest.py."
        )

    training_files = ("llm_train.jsonl", "llm_val.jsonl", "vlm_train.jsonl", "vlm_val.jsonl")
    missing_training = [name for name in training_files if not (processed_dir / name).is_file()]
    if missing_training:
        report.warn(
            f"Thiếu dữ liệu train/val trong {processed_dir}: {', '.join(missing_training)}. "
            "Giải nén du_lieu_xu_ly_v2.zip hoặc chạy python build_training_data.py."
        )
    else:
        report.ok(f"Đủ file train/val trong {processed_dir}")


def check_optional_model_files(repo_dir, report):
    optional_files = (
        repo_dir / "finetune" / "outputs" / "gguf" / "qwen2vl-2b-nhom7-q8_0.gguf",
        repo_dir / "finetune" / "outputs" / "gguf" / "mmproj-qwen2vl-2b-nhom7-f16.gguf",
        repo_dir / "finetune" / "outputs" / "qwen2vl-2b-nhom7-lora-v2" / "adapter_model.safetensors",
    )
    present = [path for path in optional_files if path.is_file()]
    if present:
        report.ok("Tìm thấy file mô hình/adapter tùy chọn: " + ", ".join(str(path) for path in present))
    else:
        report.warn("Chưa có GGUF hoặc adapter trong finetune/outputs/ (chỉ cần nếu chạy mô hình/đánh giá).")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Kiểm tra vị trí dữ liệu dự án và SHA256 các gói/mô hình đã biết."
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=ROOT,
        help="thư mục gốc repo (mặc định: thư mục chứa check_data.py)",
    )
    parser.add_argument(
        "--downloads-dir",
        type=Path,
        default=Path.home() / "Downloads",
        help="thư mục chứa file tải từ Drive (mặc định: ~/Downloads)",
    )
    args = parser.parse_args(argv)
    repo_dir = args.repo_dir.expanduser().resolve()
    downloads_dir = args.downloads_dir.expanduser().resolve()
    report = Report()

    print(f"Repo:       {repo_dir}")
    print(f"Downloads:  {downloads_dir}")
    check_raw_data(repo_dir, report)
    check_processed_data(repo_dir, report)
    check_downloads(downloads_dir, repo_dir, report)
    check_optional_model_files(repo_dir, report)

    print(f"\nTổng kết: {report.failures} lỗi, {report.warnings} cảnh báo.")
    return 1 if report.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
