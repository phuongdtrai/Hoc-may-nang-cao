# Dữ liệu, mô hình và môi trường – những gì không có trên GitHub

Repo này để công khai, nên GitHub chỉ chứa code và hướng dẫn. Những thứ còn lại nằm ở hai chỗ:

- **Thư mục Google Drive chung của nhóm** (`Nhom7-du-lieu`): tài liệu PDF, dữ liệu đã xử lý, mô hình. Link chỉ gửi trong nhóm chat, **không ghi vào repo**: repo công khai, còn giáo trình có bản quyền.
- **Google Sheet QA_Data**: nơi duy nhất để thêm và sửa câu hỏi QA.

`.gitignore` chặn `data/raw/`, `data/processed/`, `finetune/outputs/`, `.venv/`, `unsloth_compiled_cache/`, các file `*.safetensors`, `*.gguf`… để không ai đẩy nhầm file nặng hoặc tài liệu có bản quyền lên GitHub.

## 1. Mỗi thứ lấy ở đâu, đặt vào đâu

| Cần gì | Lấy ở đâu | Đặt vào (tính từ thư mục gốc repo) | Ai cần |
|---|---|---|---|
| 10 tài liệu PDF + `qa_thu_cong.csv` đã dùng cho vòng 2 | Drive: `tai_lieu_va_qa.zip` | giải nén tại thư mục gốc → `data/raw/domain_docs/` | Mọi người |
| Câu hỏi QA mới nhất | Sheet QA_Data → Tệp → Tải xuống → CSV | lưu đè `data/raw/domain_docs/qa_thu_cong.csv` | Người chạy `build_training_data.py` hoặc đánh giá |
| Kết quả ingestion và dữ liệu train/val vòng 2 (chunk, hình cắt, ảnh hỏi, các file `.jsonl`) | Drive: `du_lieu_xu_ly_v2.zip` | giải nén tại thư mục gốc → `data/processed/` | Thành viên 3 (nạp Qdrant), Thành viên 1 (train) |
| Mô hình đã fine-tune vòng 2 để chạy bằng llama-server (2,8 GB) | Drive: `qwen2vl-2b-nhom7-q8_0.gguf`, `mmproj-qwen2vl-2b-nhom7-f16.gguf` | `finetune/outputs/gguf/` | Thành viên 3 (backend), Thành viên 4 (giao diện, đo hiệu năng) |
| Adapter LoRA vòng 2 | Drive: `adapter_lora_v2.zip` | giải nén tại thư mục gốc → `finetune/outputs/qwen2vl-2b-nhom7-lora-v2/` | Chỉ khi đánh giá lại hoặc gộp lại mô hình |
| Thư viện Python | `requirements-ingest.txt`, `requirements-train.txt` trong repo | mỗi người tự tạo `.venv` (mục 3) | Mọi người |

**Không cần chia sẻ:**
- `unsloth_compiled_cache/`: Unsloth tự tạo khi train.
- `docker/qdrant_storage/`: Qdrant tự tạo khi nạp dữ liệu.
- `finetune/outputs/*-merged*`: bản gộp 4,2 GB, chỉ dùng để tạo lại GGUF.
- `.venv/`: mỗi máy tự tạo.

**Tự tạo lại `data/processed/` thay vì tải:** chạy `python ingestion/ingest.py` rồi `python build_training_data.py`. Có hai lưu ý:
- Chữ OCR trong hình chỉ có khi máy đã cài Tesseract. Chưa cài mà chạy lại thì `ocr_text` bị xoá trắng.
- Người nạp Qdrant nên dùng bản zip, để `chunk_id`, `image_id` và chữ OCR giống hệt bản cả nhóm đang dùng.

## 2. Kiểm tra đã tải đúng bản

Sau khi tải file về và giải nén vào repo, chạy script kiểm tra:

```powershell
python check_data.py
```

Script kiểm tra CSV QA_Data (đủ cột, kể cả `image_id` và `tap` có từ bản xuất 21/10), đủ 10 PDF, các file ingestion và train/val đã giải nén cùng mọi ảnh mà `images.jsonl`, `vlm_train.jsonl`, `vlm_val.jsonl` trỏ tới; đồng thời đối chiếu SHA256 cho gói ZIP hoặc GGUF nếu tìm thấy trong `"$HOME\Downloads"` hay thư mục GGUF đích. Dữ liệu xử lý và mô hình là tùy chọn theo công việc: nếu chưa có hoặc thiếu ảnh, script sẽ báo `WARN` kèm lệnh tạo lại/tải về. Mã thoát khác 0 nghĩa là dữ liệu gốc bắt buộc bị thiếu, CSV không hợp lệ hoặc hash sai.

Nếu cần kiểm tra thủ công các file đang ở Downloads:

```powershell
Get-FileHash "$HOME\Downloads\*.zip", "$HOME\Downloads\*.gguf" | Format-Table Hash, Path
```

So cột Hash với bảng dưới (không phân biệt chữ hoa, chữ thường):

| File | Dung lượng | SHA256 |
|---|---|---|
| `tai_lieu_va_qa.zip` | 10,4 MB | `85c570cd591ce99406f70926a0d5fd28cefa3d3b59acbdfa4db3ef99b9ba4963` |
| `du_lieu_xu_ly_v2.zip` | 38,6 MB | `131c8dff66ed8ab1583116f91c750ad4dda47c4354d21b0a78595989ae5e1c24` |
| `adapter_lora_v2.zip` | 76,1 MB | `ff5e1419d63cf7211b625e51962ce193ceb304693e217697e7e4795ce718af96` |
| `qwen2vl-2b-nhom7-q8_0.gguf` | 1,65 GB | `2ee9f8e7e9f2e8c3e9f8c7cb0a400565199e40fe02d557e5d4a32763f6778371` |
| `mmproj-qwen2vl-2b-nhom7-f16.gguf` | 1,33 GB | `ee09b928111299277e66d479405d13e818045d0ff779a823ce412f069c3174c6` |

## 3. Thiết lập máy mới (PowerShell, Python 3.11)

```powershell
git clone https://github.com/phuongdtrai/Hoc-may-nang-cao.git "C:\Machine Learning\nhom7-multimodal-rag"
cd "C:\Machine Learning\nhom7-multimodal-rag"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-ingest.txt                  # máy có GPU NVIDIA để train: requirements-train.txt
Expand-Archive "$HOME\Downloads\tai_lieu_va_qa.zip" . -Force
Expand-Archive "$HOME\Downloads\du_lieu_xu_ly_v2.zip" . -Force
python ingestion/xem_qa_anh.py                          # chạy được và tạo trang HTML là dữ liệu đã đúng chỗ
```

- Máy có GPU NVIDIA: sau khi cài `requirements-train.txt`, chạy `python finetune/check_env.py`; hết dòng FAIL là train được.
- Cần chạy mô hình: tạo thư mục rồi chuyển hai file GGUF vào:

  ```powershell
  mkdir finetune\outputs\gguf
  Move-Item "$HOME\Downloads\*.gguf" finetune\outputs\gguf\
  ```

## 4. Chạy mô hình bằng llama-server

```powershell
$b = "b11669"; $dst = "C:\Machine Learning\llama-bin"; $ProgressPreference = "SilentlyContinue"
New-Item -ItemType Directory -Force $dst | Out-Null
Invoke-WebRequest "https://github.com/ggml-org/llama.cpp/releases/download/$b/llama-$b-bin-win-cuda-13.4-x64.zip" -OutFile "$dst\llama.zip"
Invoke-WebRequest "https://github.com/ggml-org/llama.cpp/releases/download/$b/cudart-llama-bin-win-cuda-13.4-x64.zip" -OutFile "$dst\cudart.zip"
Expand-Archive "$dst\llama.zip" $dst -Force
Expand-Archive "$dst\cudart.zip" $dst -Force
$env:Path = "C:\Machine Learning\llama-bin;$env:Path"      # mỗi lần mở terminal mới
llama-server -m finetune/outputs/gguf/qwen2vl-2b-nhom7-q8_0.gguf --mmproj finetune/outputs/gguf/mmproj-qwen2vl-2b-nhom7-f16.gguf -ngl 99 -c 4096 --port 8080
```

- **Máy không có GPU NVIDIA:** tải `llama-$b-bin-win-vulkan-x64.zip` (GPU AMD hoặc Intel) hoặc `llama-$b-bin-win-cpu-x64.zip` (chạy bằng CPU, chậm hơn) thay cho hai file trên, không cần file cudart.
- **Thử nhanh:** server chạy xong thì mở `http://localhost:8080` để hỏi thử.
- **Khi gọi từ backend:** phải dựng prompt bằng `finetune/prompt_format.py`, như mục 7 của `HUONG_DAN_GIAI_DOAN_4.md`.

## 5. Quy ước để cả nhóm dùng cùng một bản

- **Code** chỉ đi qua GitHub. Trước khi làm thì `git pull`; làm xong thì commit với nội dung rõ ràng rồi `git push`.
- **Câu hỏi QA** chỉ sửa trên sheet QA_Data. Không sửa file CSV trên máy rồi gửi cho nhau.
- **Dữ liệu hoặc mô hình mới:**
  - tải lên Drive với tên có số vòng (ví dụ `du_lieu_xu_ly_v3.zip`), không ghi đè bản người khác đang dùng;
  - cập nhật bảng ở mục 1–2 của file này và commit, để cả nhóm biết bản nào mới nhất.
- **Không commit** PDF, CSV, `.jsonl`, ảnh hay mô hình. Nếu `git status` hiện file trong `data/` hoặc `finetune/outputs/` thì dừng lại, báo cả nhóm.
