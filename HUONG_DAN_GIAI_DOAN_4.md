# Giai đoạn 4 – Fine-tune Qwen2-VL-2B bằng QLoRA (Unsloth) trên RTX 5060 8 GB

Gói này nối tiếp Bước 2.4: lấy `llm_train.jsonl`, `llm_val.jsonl`, `vlm_train.jsonl` để fine-tune Qwen2-VL-2B-Instruct, so sánh trước/sau và xuất mô hình cho Giai đoạn 6.

## Trong gói này có gì

| Đường dẫn | Việc |
|---|---|
| `finetune/check_env.py` | Kiểm tra môi trường: GPU, PyTorch có hỗ trợ RTX 50xx không, bitsandbytes, Unsloth, dữ liệu |
| `finetune/train_qwen2vl.py` | Train QLoRA, tính val loss trước/sau, lưu adapter + thông tin train |
| `finetune/eval_truoc_sau.py` | Cho mô hình gốc và mô hình đã fine-tune cùng trả lời tập val, chấm điểm, xuất CSV + báo cáo |
| `finetune/export_model.py` | Gộp LoRA vào mô hình gốc thành bản 16-bit đầy đủ (để chuyển GGUF / chạy không cần peft) |
| `finetune/prompt_format.py` | "Hợp đồng prompt" dùng chung cho train, đánh giá và backend Giai đoạn 6 |
| `finetune/cham_tay.py` | Tạo phiếu chấm thủ công (ẩn tên mô hình) và tổng hợp F1 theo ý, độ đồng thuận (mục 10) |
| `finetune/y_dap_an_cham_tay.csv` | 84 câu val chọn để chấm tay, đáp án chuẩn đã tách sẵn thành các ý |
| `finetune/colab_finetune.ipynb` | Train và đánh giá trên Google Colab (GPU T4 miễn phí), cho thành viên không có máy GPU |

Các file nằm trong thư mục gốc repo (cạnh `build_training_data.py`). Mọi lệnh bên dưới đều chạy **tại thư mục gốc repo**. Kết quả được ghi vào `finetune/outputs/`. Thư mục này đã có sẵn trong `.gitignore` của repo, nên các file mô hình nặng vài GB sẽ không bị đẩy lên Git.

## 1. Năm lệnh chính

```powershell
python finetune/check_env.py                       # 1. môi trường ổn chưa
python finetune/train_qwen2vl.py --dry_run          # 2. xem dữ liệu + prompt (không cần GPU)
python finetune/train_qwen2vl.py --max_steps 10     # 3. chạy thử vài phút, xem VRAM
python finetune/train_qwen2vl.py                    # 4. train đầy đủ (2 epoch)
python finetune/eval_truoc_sau.py                   # 5. so sánh trước/sau
```

Sau đó, nếu cần đóng gói: `python finetune/export_model.py --kiem_tra` (mục 6).

## 2. Cài môi trường

RTX 5060 là GPU thế hệ Blackwell nên cần **PyTorch bản CUDA 12.8 trở lên**. Bản PyTorch cũ hơn sẽ báo lỗi `no kernel image is available`.

Trước hết cập nhật driver NVIDIA bản mới nhất, mở PowerShell gõ `nvidia-smi` và nhìn ô **CUDA Version** ở góc phải:
- 13.x → dùng `cu130` trong các lệnh bên dưới
- 12.8 hoặc 12.9 → thay bằng `cu128`

### Cách 1 – Windows + Conda (đơn giản nhất, nên thử trước)

1. Cài [Miniconda](https://www.anaconda.com/download/success), mở **Anaconda PowerShell Prompt**.
2. Tạo môi trường và cài:
   ```powershell
   conda create -n unsloth_env python=3.12 -y
   conda activate unsloth_env
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   pip install unsloth
   pip install matplotlib pymupdf
   ```
   Dòng `python -c ...` phải in ra `True NVIDIA GeForce RTX 5060`. Không cài riêng `triton`, vì `pip install unsloth` tự cài bản `triton-windows` phù hợp.
3. Vào repo và kiểm tra:
   ```powershell
   cd "C:\Machine Learning\nhom7-multimodal-rag"
   python finetune/check_env.py
   ```

### Cách 2 – Docker (image `unsloth/unsloth` dựng sẵn, cả nhóm dùng giống hệt nhau)

Cần Docker Desktop bật "Use the WSL 2 based engine", driver NVIDIA ≥ 570.26, và đã chạy `wsl --update`. Tại thư mục repo trong PowerShell:

```powershell
docker run -d --name unsloth --gpus all --ipc=host -p 8888:8888 -e JUPYTER_PASSWORD="nhom7" `
  -v "${PWD}:/workspace/host" `
  -v "${HOME}/.cache/huggingface:/workspace/.cache/huggingface" `
  unsloth/unsloth
docker exec -it unsloth bash
# trong container:
cd /workspace/host
pip install matplotlib pymupdf
python finetune/check_env.py
```

Kết quả train ghi thẳng vào thư mục repo trên Windows. Lần sau chỉ cần `docker start unsloth` rồi `docker exec -it unsloth bash`.

### Cách 3 – WSL2 Ubuntu

```bash
# PowerShell: wsl --install Ubuntu-24.04   rồi mở Ubuntu
sudo apt update && sudo apt install -y python3-venv python3-pip
python3 -m venv ~/unsloth-env && source ~/unsloth-env/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install unsloth matplotlib pymupdf
cd "/mnt/c/Machine Learning/nhom7-multimodal-rag"
python finetune/check_env.py
```

### Cách 4 – Google Colab (không cần máy có GPU)

Trên Colab chọn **Tệp > Tải sổ tay lên**, mở `finetune/colab_finetune.ipynb`, chọn GPU T4 rồi chạy các ô từ trên xuống. Notebook tự lấy code từ GitHub và lấy dữ liệu `du_lieu_xu_ly_v2.zip` trong thư mục `nhom7` trên Drive. Kết quả lưu ở `nhom7/ket_qua/<tên lần train>/`. Notebook cài theo `requirements-colab.txt` thay cho `requirements-train.txt`: cùng phiên bản thư viện train, nhưng giữ nguyên torch có sẵn của Colab. T4 không có bf16 nên các script tự train bằng fp16. Mỗi lần train, `thong_tin_train.json` ghi mục `moi_truong` (phiên bản thư viện, GPU, bf16/fp16, commit code, mã băm dữ liệu) để đối chiếu kết quả giữa các máy.

## 3. Kiểm tra trước khi train

`check_env.py` in từng dòng `OK / WARN / FAIL`, dòng FAIL có kèm cách sửa. Khi hết FAIL, chạy:

```powershell
python finetune/train_qwen2vl.py --dry_run
```

Lệnh này chỉ tải processor (vài MB) và in ra:
- số mẫu train/val, có bao nhiêu mẫu có ngữ cảnh, bao nhiêu mẫu ảnh;
- độ dài token (với dữ liệu hiện tại: trung vị khoảng 490, mẫu ảnh khoảng 1.150, dưới giới hạn 2.048);
- **nguyên văn prompt đưa vào mô hình** và **phần được tính loss** (chỉ câu trả lời + `<|im_end|>`).

## 4. Train

### Chạy thử

```powershell
python finetune/train_qwen2vl.py --max_steps 10
```

Lần đầu sẽ tải mô hình `unsloth/Qwen2-VL-2B-Instruct-unsloth-bnb-4bit` (vài GB). Trong lúc chạy, mở một cửa sổ khác gõ `nvidia-smi -l 1` để xem VRAM, đúng như kế hoạch ở lộ trình. Nếu chạy hết 10 bước và in `XONG`, mọi thứ đã ổn.

### Train đầy đủ

```powershell
python finetune/train_qwen2vl.py
```

Script sẽ:
1. Tính **val loss của mô hình gốc**.
2. Train 2 epoch. Vòng 1 có 770 mẫu train (763 câu chữ + 7 câu ảnh) và mỗi bước dùng 8 mẫu (batch 2 × tích luỹ 4), nên mỗi epoch khoảng 97 bước, tổng cộng gần 200 bước. Từ Giai đoạn 3 có 862 mẫu (thêm 92 câu hỏi có đáp án là hình), khoảng 108 bước mỗi epoch.
3. Tính val loss sau mỗi epoch, lưu checkpoint mỗi epoch.
4. Lưu kết quả vào `finetune/outputs/qwen2vl-2b-nhom7-lora/`:
   - `adapter_model.safetensors`: LoRA adapter, vài chục MB;
   - `thong_tin_train.json`: cấu hình, số mẫu, thời gian, VRAM đỉnh, val loss trước/sau;
   - `loss_log.csv` và `loss.png`: diễn biến loss, dùng cho báo cáo.

Cách đọc loss:
- Train loss giảm nhanh trong epoch 1 rồi chững lại.
- Val loss sau train phải thấp hơn val loss ban đầu.
- Nếu val loss ở epoch 2 cao hơn epoch 1 rõ rệt là mô hình bắt đầu học thuộc. Khi đó chạy lại với `--epochs 1` hoặc `--lr 1e-4`. Lộ trình cũng dặn không train quá nhiều epoch trên tập hẹp.

Bấm Ctrl+C giữa chừng thì script vẫn lưu adapter hiện tại. Muốn train tiếp từ checkpoint gần nhất thì thêm `--resume`.

### Tham số hay dùng

| Tham số | Mặc định | Khi nào đổi |
|---|---|---|
| `--epochs` | 2 | 1 nếu val loss tăng lại; 3 nếu loss còn giảm đều |
| `--lr` | 2e-4 | 1e-4 nếu loss dao động mạnh |
| `--batch` / `--grad_accum` | 2 / 4 | `--batch 1 --grad_accum 8` nếu hết VRAM |
| `--max_pixels` | 802816 (≈1.024 token ảnh) | `602112` (768·28·28) để giảm VRAM khi có nhiều ảnh |
| `--max_len` | 2048 | Giữ nguyên; script tự bỏ (và báo) mẫu dài hơn |
| `--lora_r` / `--lora_alpha` | 16 / 16 | 8 / 8 nếu muốn nhẹ hơn |
| `--finetune_vision` | tắt | Chỉ bật khi đã có hàng trăm câu hỏi ảnh |
| `--save_merged` | tắt | Lưu thêm bản gộp 16-bit ngay sau khi train |
| `--backend hf` | `unsloth` | Dự phòng khi Unsloth lỗi trên máy: dùng transformers + peft + bitsandbytes, cùng dữ liệu và cùng prompt. Bản này tốn VRAM hơn Unsloth, nên chạy kèm `--batch 1 --grad_accum 8` |

### Nếu hết VRAM (`CUDA out of memory`)

Thử lần lượt:
1. Tắt trình duyệt, game và các ứng dụng đang dùng GPU.
2. `--batch 1 --grad_accum 8`
3. `--max_pixels 602112`
4. `--max_len 1536`

## 5. Đánh giá trước/sau

```powershell
python finetune/eval_truoc_sau.py              # đầy đủ: câu val chữ + 54 câu ảnh val + 20 câu ảnh train, mỗi câu 2 mô hình
python finetune/eval_truoc_sau.py --limit 10   # thử nhanh
```

Script nạp lần lượt mô hình gốc rồi mô hình fine-tune, cho cả hai trả lời cùng một câu hỏi theo cùng một prompt (greedy, tối đa 256 token), rồi chấm điểm với đáp án chuẩn:
- **F1** (chỉ số chính): khớp theo âm tiết, cân bằng giữa đúng và đủ;
- **ROUGE-L**: mức trùng theo thứ tự từ;
- **Độ phủ**: tỉ lệ âm tiết của đáp án chuẩn có trong câu trả lời.

Kết quả ở `finetune/outputs/danh_gia/`:
- `bao_cao_danh_gia.md`: bảng tổng hợp theo nhóm câu và theo tài liệu, thời gian trả lời mỗi câu, val loss, các ví dụ cải thiện hoặc kém đi. Có thể đưa thẳng vào báo cáo cuối kỳ.
- `so_sanh_truoc_sau.csv`: từng câu kèm hai câu trả lời và điểm. Mở được bằng Excel và hiển thị đúng tiếng Việt.
- `du_doan_goc.jsonl` và `du_doan_finetune.jsonl`: câu trả lời đã sinh. Lần chạy sau script dùng lại, không sinh lại. Nếu train lại adapter, script tự nhận ra và sinh lại cho bản mới. Muốn ép sinh lại thì thêm `--lam_lai`.

**Lưu ý khi viết báo cáo:**
- Ở vòng 1, 7 câu hỏi về ảnh đều đã nằm trong tập train, nên kết quả nhóm ảnh chỉ cho thấy mô hình đã học được dữ liệu đó. Từ Giai đoạn 3 đã có 54 câu hỏi ảnh chưa dùng khi train (`vlm_val.jsonl`); script tự tách thành hai nhóm "Ảnh – đáp án là hình, tập val" (26 câu) và "Ảnh – câu hỏi là ảnh, tập val" (28 câu, người dùng gửi ảnh chụp rồi hỏi). Đây là nhóm dùng để chứng minh "trả lời câu hỏi về ảnh/biểu đồ tốt hơn rõ rệt" như điều kiện hoàn thành của lộ trình (xem `HUONG_DAN_GIAI_DOAN_3.md`, mục 7).
- F1 và ROUGE-L chỉ đo mức trùng từ ngữ, nên câu trả lời đúng ý nhưng diễn đạt khác vẫn bị điểm thấp. Hãy đọc thêm phần ví dụ, hoặc chấm thủ công theo mục 10.

## 6. Xuất mô hình

### 6.1 Gộp LoRA thành mô hình 16-bit

```powershell
python finetune/export_model.py --kiem_tra
```

Script tải bản gốc 16-bit `unsloth/Qwen2-VL-2B-Instruct` (khoảng 4,5 GB), gộp adapter trên CPU (cần khoảng 10 GB RAM), rồi lưu vào `finetune/outputs/qwen2vl-2b-nhom7-merged/`. Tuỳ chọn `--kiem_tra` nạp lại bản gộp và hỏi thử một câu. Bản gộp nạp được bằng `transformers` thuần, không cần peft.

### 6.2 Chuyển sang GGUF để chạy bằng llama.cpp

Chạy trong cùng môi trường train (đã có torch và transformers). **Không** cài `requirements` của llama.cpp vào môi trường này, vì file đó ép cài torch bản CPU. Clone llama.cpp ra **ngoài** repo (vào `C:\Machine Learning\llama.cpp`) để repo không chứa thêm một repo Git khác:

```powershell
git clone --depth 1 https://github.com/ggml-org/llama.cpp ..\llama.cpp
mkdir finetune\outputs\gguf
python ..\llama.cpp\convert_hf_to_gguf.py finetune/outputs/qwen2vl-2b-nhom7-merged --outtype q8_0 --outfile finetune/outputs/gguf/qwen2vl-2b-nhom7-q8_0.gguf
python ..\llama.cpp\convert_hf_to_gguf.py finetune/outputs/qwen2vl-2b-nhom7-merged --mmproj --outtype f16 --outfile finetune/outputs/gguf/mmproj-qwen2vl-2b-nhom7-f16.gguf
```

Lệnh đầu tạo phần ngôn ngữ (khoảng 1,6 GB). Lệnh sau tạo phần thị giác (mmproj). Thử bằng bản llama.cpp dựng sẵn có CUDA (trang Releases của llama.cpp, hoặc Docker image `ghcr.io/ggml-org/llama.cpp:server-cuda`):

```powershell
llama-server -m finetune/outputs/gguf/qwen2vl-2b-nhom7-q8_0.gguf --mmproj finetune/outputs/gguf/mmproj-qwen2vl-2b-nhom7-f16.gguf -ngl 99 -c 4096 --port 8080
```

`llama-server` mở API kiểu OpenAI tại `http://localhost:8080/v1/chat/completions` và nhận được cả ảnh. Chuỗi lệnh này đã được chạy thử trên một mô hình Qwen2-VL thu nhỏ: llama.cpp nạp được cả hai file, và prompt mà server dựng ra trùng khớp từng ký tự với prompt lúc train.

**Về Ollama:** tài liệu nhập mô hình hiện tại của Ollama không hướng dẫn cách nạp mô hình thị giác kèm file mmproj. Riêng Qwen2-VL-2B còn có báo lỗi ảnh không được nhận dù phần chữ chạy bình thường ([ollama#14388](https://github.com/ollama/ollama/issues/14388)). Vì vậy, với câu hỏi có ảnh ở Giai đoạn 6, nên gọi mô hình qua `llama-server` như trên thay vì Ollama.

## 7. "Hợp đồng prompt" cho Giai đoạn 6

Mô hình được train với **system prompt cố định** cùng định dạng câu hỏi của Bước 2.4. Backend phải gửi đúng như vậy thì kết quả mới giống lúc đánh giá. Cách an toàn nhất là import thẳng từ `finetune/prompt_format.py`:

```python
from finetune.prompt_format import build_rag_messages, to_openai_messages

msgs = build_rag_messages(question, contexts=[chunk1, chunk2], images=["data/.../trang_8.png"])
payload = {"model": "qwen2vl-nhom7", "messages": to_openai_messages(msgs), "temperature": 0.2}
# requests.post("http://localhost:8080/v1/chat/completions", json=payload)
```

- `build_rag_messages` dựng tin nhắn **giống hệt lúc train**. Đã kiểm tra trùng khớp trên toàn bộ 854 mẫu.
  - Có ngữ cảnh: `Trả lời câu hỏi dựa trên ngữ cảnh sau.` + `Ngữ cảnh: …` + `Câu hỏi: …`.
  - Không có ngữ cảnh: `Trả lời câu hỏi sau.`
  - Có ảnh: `Trả lời câu hỏi dựa trên hình ảnh sau.`
  - Có ảnh và ngữ cảnh: `Trả lời câu hỏi dựa trên hình ảnh và ngữ cảnh sau.` Khi Qdrant trả về một hình, thêm dòng `image_context(...)` (trang, mục, chú thích của hình) vào `contexts`, đúng như lúc train câu hỏi ảnh (xem `HUONG_DAN_GIAI_DOAN_3.md`, mục 8).
- `to_openai_messages` thu nhỏ ảnh đúng như lúc train rồi đổi sang dạng `image_url` base64 mà `llama-server` nhận.
- System prompt nằm ở hằng `SYSTEM_PROMPT`. Nếu sửa thì phải train lại.

## 8. Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `no kernel image is available for execution on the device` | PyTorch quá cũ cho RTX 50xx: cài lại bằng `--index-url .../cu130` (hoặc `cu128`) |
| `torch.cuda.is_available()` ra `False` | Lỡ cài bản CPU: `pip uninstall torch torchvision`, rồi cài lại theo mục 2 |
| Lỗi bitsandbytes | `pip install -U bitsandbytes` (cần ≥ 0.45.5 cho RTX 50xx) |
| Lỗi có chữ `triton`, `cl.exe`, `compile` trên Windows | Chuyển sang Cách 2 (Docker) hoặc Cách 3 (WSL). Tạm thời vẫn train được bằng `--backend hf` |
| Tên người dùng Windows có dấu hoặc khoảng trắng làm Triton lỗi | `setx TRITON_CACHE_DIR C:\triton_cache` rồi mở lại terminal |
| `UnicodeEncodeError` khi ghi log ra file | `$env:PYTHONIOENCODING="utf-8"` trước khi chạy |
| Tải mô hình chậm hoặc đứt giữa chừng | Chạy lại lệnh (sẽ tải tiếp phần còn thiếu), hoặc tải trước bằng `hf download unsloth/Qwen2-VL-2B-Instruct-unsloth-bnb-4bit` |
| `Khong tim thay anh ...` | Chạy lại `python build_training_data.py` để render ảnh trang vào `data/processed/images/` |

## 9. Cấu hình đã chọn và lý do (dùng cho báo cáo)

- **QLoRA 4-bit** với bản lượng tử hoá động của Unsloth. Mô hình 2B chỉ chiếm vài GB VRAM, phần còn lại đủ cho batch 2 trên RTX 5060 8 GB.
- **LoRA r = 16, alpha = 16** trên toàn bộ lớp attention và MLP của phần ngôn ngữ. Số tham số được train dưới 1% tổng số (số chính xác ghi trong `thong_tin_train.json`).
- **Đóng băng phần thị giác**, vì cả bộ dữ liệu chỉ có 7 câu hỏi ảnh. Train phần thị giác với từng ấy dữ liệu dễ làm hỏng khả năng nhìn ảnh vốn có của mô hình.
- **Chỉ tính loss trên câu trả lời.** System prompt, ngữ cảnh, câu hỏi và token ảnh đều bị che (−100). Mô hình học cách trả lời bám ngữ cảnh chứ không học thuộc ngữ cảnh.
- **Tham số train:**
  - lr 2e-4 giảm tuyến tính, có 5% bước khởi động (warmup);
  - 2 epoch, mỗi bước 8 mẫu;
  - optimizer AdamW 8-bit;
  - gradient checkpointing của Unsloth để tiết kiệm VRAM.
- **Ảnh tối đa khoảng 1.024 token.** Một trang A4 render ở 120 DPI được thu về khoảng 728×1064 pixel, đủ đọc chữ trên slide và sơ đồ mà vẫn vừa giới hạn 2.048 token.
- **Tập val chia theo trang** (Bước 2.4), nên câu hỏi val không trùng ngữ cảnh với tập train.
- **Đã kiểm thử trên CPU bằng mô hình Qwen2-VL thu nhỏ và bộ tách từ thật của Qwen2:**
  - collator tự viết (`--backend hf`) cho ra token, nhãn và ảnh **trùng khớp từng phần tử** với `UnslothVisionDataCollator`, kể cả với padding trái hoặc phải;
  - nhãn chỉ gồm câu trả lời và `<|im_end|>`;
  - ước lượng độ dài token khớp đúng với processor;
  - toàn bộ luồng train → đánh giá → gộp → GGUF chạy thông.

## 10. Chấm thủ công: F1 theo ý, theo ngữ nghĩa và ngữ cảnh

F1 tự động ở mục 5 chỉ đếm âm tiết trùng với đáp án chuẩn. `finetune/cham_tay.py` tạo phiếu để người chấm theo **ngữ nghĩa** (ý đúng nghĩa, diễn đạt khác vẫn tính) và theo **ngữ cảnh** (đối chiếu với đúng ngữ cảnh mô hình được đưa), rồi tổng hợp lại sau khi nhóm chấm xong.

```powershell
pip install openpyxl==3.1.5                # một lần (đã thêm vào requirements-ingest.txt)
python finetune/cham_tay.py tao           # tạo 4 phiếu + khoá A/B từ danh_gia_v2/so_sanh_truoc_sau.csv
python finetune/cham_tay.py tong_hop      # sau khi chép các phiếu đã chấm vào da_cham/
```

**Chọn câu.** Có 84 câu tập val, cố định trong `finetune/y_dap_an_cham_tay.csv`:
- 26 câu "Ảnh – đáp án là hình" (toàn bộ);
- 28 câu "Ảnh – câu hỏi là ảnh" (toàn bộ);
- 30 câu chữ: 14 TRIET_001, 13 TTHCM_001, 3 tài liệu khác, lấy ngẫu nhiên với seed 42.

Đáp án chuẩn của mỗi câu đã được tách sẵn thành 2–6 ý, tổng cộng 314 ý. Cả nhóm dùng chung một bản tách để R có cùng mẫu số. Vòng sau vẫn dùng file này nếu tập val không đổi. Nếu đáp án trên sheet bị sửa, `tao` sẽ báo câu nào cần tách lại.

**Phiếu chấm** (`finetune/outputs/cham_tay_v2/`):
- **Chia câu cho người chấm.** 84 câu được chia thành 4 phần đều nhau, phần nào cũng có đủ 3 nhóm câu. Phiếu i gồm phần i và phần i+1, nên mỗi câu có 2 người chấm, tạo thành 4 cặp khác nhau. Mỗi phiếu có 42 câu, tức 84 câu trả lời, mất khoảng 1,5–2 giờ.
- **Ẩn tên mô hình.** Hai câu trả lời của mỗi câu được ghi A/B theo thứ tự ngẫu nhiên. `khoa.json` ghi A/B ứng với mô hình nào; người điều phối giữ file này, không gửi đi và không mở trước khi cả nhóm chấm xong.
- **Ba sheet trong phiếu:**
  - Huong_dan: cách chấm và 5 ví dụ;
  - Cham: mỗi câu 2 dòng, có sẵn ngữ cảnh mô hình nhận được (ảnh, chú thích, đoạn văn) và các ý của đáp án, chỉ điền vào ô vàng;
  - Tham_khao: đáp án và ngữ cảnh đầy đủ.

**Cách chấm** (chi tiết trong sheet Huong_dan):
- Ngữ nghĩa: đánh dấu từng ý của đáp án mà câu trả lời nêu đúng nghĩa (`x` là đủ ý, `p` là một phần). Từ đó R = ý đạt / số ý của đáp án.
- Ngữ cảnh: đếm các ý còn lại của câu trả lời vào 3 cột: thêm trong ngữ cảnh, thêm ngoài ngữ cảnh, sai/bịa. Từ đó P = (ý đạt + ý thêm trong ngữ cảnh) / tổng số ý.
- F1 = 2PR/(P+R). Ngoài ra chấm thêm:
  - nhãn đúng 0/1/2;
  - điểm Diễn đạt 0/1/2;
  - câu nào tốt hơn (A, B hoặc Ngang).

**Tổng hợp.** Chép các phiếu đã chấm vào `finetune/outputs/cham_tay_v2/da_cham/` (file .xlsx, hoặc .csv tải từ Google Sheets) rồi chạy `tong_hop`. Kết quả gồm:
- `bao_cao_cham_tay.md`:
  - bảng chỉ số tự động đặt cạnh bản chấm tay;
  - F1 thủ công theo nhóm câu, kèm khoảng tin cậy 95%;
  - tỷ lệ người chấm chọn v2;
  - nhãn và diễn đạt;
  - tỷ lệ ý bịa, tỷ lệ ý ngoài ngữ cảnh;
  - tương quan với F1 tự động;
  - độ đồng thuận (Krippendorff α, kappa);
  - các câu hai người chấm lệch nhau cần xem lại.
- `ket_qua_cham_tay.csv`: điểm từng câu trả lời, đã gắn lại tên mô hình.

| Chỉ số tự động (mục 5) | Bản chấm tay |
|---|---|
| F1 | F1 theo ý (P theo ngữ cảnh, R theo nghĩa) |
| Độ phủ | R theo nghĩa |
| ROUGE-L | Không đếm tay được: ROUGE-L chỉ đếm chuỗi chữ trùng, đếm tay chỉ ra lại con số của máy. Thay bằng điểm Diễn đạt 0/1/2 |
| Thắng/hòa/thua | Tính từ F1 thủ công, và từ lựa chọn A/B của người chấm |
| Thời gian trả lời | Vẫn đo bằng máy |
| (không có) | Nhãn đúng 0/1/2; tỷ lệ câu có ý sai/bịa; tỷ lệ câu có ý ngoài ngữ cảnh |

Nếu độ đồng thuận dưới 0,67, cả nhóm họp thống nhất lại cách tách ý và cách xếp ý vào trong hay ngoài ngữ cảnh. Sau đó hai người chấm mỗi câu trong mục "Cần xem lại" cùng xem lại câu đó, sửa phiếu và chạy lại `tong_hop`.
