# Giai đoạn 5 – Embedding và nạp Qdrant

Gói này làm phần Giai đoạn 5 của lộ trình: nhúng (embedding) **1.694 chunk** và **137 hình** do ingestion Giai đoạn 3 tạo ra, nạp vào Qdrant đúng data contract ở `HUONG_DAN_GIAI_DOAN_3.md` (mục 3 và 8), kèm **module tìm kiếm** cho backend Giai đoạn 6 và **đo Recall@k** trên 3 bộ câu hỏi: câu chữ, câu hỏi đáp án là hình, câu hỏi là ảnh chụp.

## Trong gói này có gì

| Đường dẫn | Việc |
|---|---|
| `embedding/nhung.py` | Phần dùng chung: tên mô hình, collection, chữ đem đi nhúng của chunk và hình, kết nối Qdrant |
| `embedding/nap_qdrant.py` | Nhúng `chunks.jsonl` + `images.jsonl` rồi nạp vào Qdrant (xoá và tạo lại collection mỗi lần chạy) |
| `embedding/tim_kiem.py` | Lớp `TimKiem` cho backend: tìm bằng câu chữ, tìm bằng ảnh, hình kèm chunk, chunk cùng trang với hình; chạy riêng để thử nhanh |
| `embedding/danh_gia_truy_hoi.py` | Đo Recall@1/5/10 và MRR@10, ghi `data/processed/danh_gia_truy_hoi.md` + `.jsonl` |
| `requirements-embed.txt` | Thư viện: `sentence-transformers`, `qdrant-client` (kèm `requirements-ingest.txt`) |

Mọi lệnh chạy **tại thư mục gốc repo**. Cần có `data/processed/` (giải nén `du_lieu_xu_ly_v2.zip`) và `data/raw/domain_docs/qa_thu_cong.csv` (giải nén `tai_lieu_va_qa.zip`), như `HUONG_DAN_DU_LIEU.md`.

## 1. Lệnh chính

```powershell
docker compose -f docker/docker-compose.yml up -d qdrant   # 1. bật Qdrant ở http://localhost:6333
python embedding/nap_qdrant.py                             # 2. nhúng + nạp, khoảng 2 phút trên GPU
python embedding/danh_gia_truy_hoi.py                      # 3. đo Recall@k, khoảng 40 giây
python embedding/tim_kiem.py "Cho xem sơ đồ tổng quát của một hệ thống xử lý ảnh"   # 4. thử tìm
```

- Lần đầu chạy bước 2, máy tự tải mô hình về `%USERPROFILE%\.cache\huggingface`: `BAAI/bge-m3` khoảng 2,3 GB, CLIP khoảng 0,6 GB. Bước này cần Internet.
- Chỉ bật service `qdrant`. Service `ollama` trong cùng file compose không dùng tới.
- Xem dữ liệu đã nạp: mở `http://localhost:6333/dashboard`.
- Chạy lại bước 2 sau mỗi lần chạy lại ingestion hoặc đổi mô hình. Collection được tạo lại từ đầu, nên không còn điểm cũ.

**Không có Docker:** thêm `--path data/processed/qdrant_local` vào cả 4 lệnh. Qdrant khi đó chạy ngay trong Python và lưu vào thư mục này (đã nằm trong `.gitignore` vì thuộc `data/processed/`). Kết quả giống hệt bản server, nhưng chỉ một tiến trình được mở thư mục đó tại một thời điểm, và đánh giá chậm hơn (khoảng 3 phút).

## 2. Cài môi trường

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements-embed.txt
```

- **Máy có GPU NVIDIA:** cài `requirements-train.txt` trước (để có torch bản CUDA), rồi mới cài file trên vào cùng `.venv`. `sentence-transformers==6.2.0` dùng được với `transformers==5.17.0` của môi trường train. Trên RTX 3050 Laptop 6 GB, bước nạp mất 112 giây, tính cả thời gian tải mô hình lên GPU. bge-m3 chạy fp16 nên chiếm khoảng 1,1 GB VRAM.
- **Máy không có GPU:** cài thẳng file trên, pip tự cài torch bản CPU. Đo trên i5-13450HX:
  - nạp Qdrant mất khoảng **28 phút** (0,9 giây mỗi chunk), chỉ cần làm một lần;
  - tìm kiếm vẫn nhanh, khoảng 0,12 giây mỗi câu hỏi.
- Sau khi đã tải mô hình, có thể đặt `$env:HF_HUB_OFFLINE="1"` để script không hỏi lại Hugging Face mỗi lần chạy.

## 3. Nhúng gì, nạp thế nào

| Loại điểm | Số điểm | Vector `text` (bge-m3, 1024 chiều, cosine) | Vector `image` (CLIP ViT-B/32, 512 chiều, cosine) | Payload |
|---|---|---|---|---|
| `type: "text"` | 1.694 | trường `text` của chunk | – | nguyên một dòng `chunks.jsonl`. Trường `type` gốc (`text`/`table`) được chuyển sang `chunk_type` |
| `type: "image"` | 137 | `caption_hint + section + context_before + context_after + text_in_image + ocr_text` | ảnh `images/<image_id>.png` | nguyên một dòng `images.jsonl` |

- **id điểm:** UUID tính từ `chunk_id` / `image_id` (`nhung.point_id`). Nạp lại vẫn ra đúng id cũ. Payload vẫn giữ `chunk_id`, `image_id` để backend dùng.
- **Chỉ mục payload (bản server):** `type`, `doc_id`, `chunk_id`, `image_id`, `chunk_ids` (từ khoá) và `page_number`, `pages` (số nguyên), giúp lọc nhanh.
- **Câu hỏi chữ** dùng vector `text`, nên tìm được cả chunk lẫn hình qua phần chữ mô tả hình. Lọc `type` khi chỉ muốn một loại.
- **Câu hỏi là ảnh** dùng vector `image`, so ảnh người dùng chụp với hình cắt trong tài liệu.

## 4. Kết quả đo

Chạy `python embedding/danh_gia_truy_hoi.py` trên dữ liệu vòng 2 (`du_lieu_xu_ly_v2.zip`, QA_Data xuất 21/10):

| Bộ câu hỏi | Số câu | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
|---|---|---|---|---|---|
| Tìm chữ: câu `text` → chunk đúng `doc_id` + trang | 847 | 80,9% | 94,8% | 97,3% | 0,869 |
| Tìm hình bằng câu chữ (`qa_anh_truy_hoi.jsonl`) → đúng `image_id` | 115 | 64,3–66,1% | 97,4% | 99,1% | 0,783–0,792 |
| … chỉ cần đúng trang | 115 | 71,3% | 99,1% | 100% | 0,836 |
| Tìm bằng ảnh chụp (`anh_hoi_truy_hoi.jsonl`) → đúng `image_id` | 127 | 86,6% | 97,6% | 98,4% | 0,911 |
| … chỉ cần đúng trang | 127 | 89,0% | 99,2% | 100% | 0,931 |

Tìm chữ theo từng tài liệu (cột cuối: tính đúng cả khi chunk lệch ±1 trang so với sheet):

| Tài liệu | Số câu | Recall@1 | Recall@5 | Recall@5 (±1 trang) |
|---|---|---|---|---|
| TRIET_001 | 400 | 86,5% | 97,5% | 98,5% |
| TTHCM_001 | 340 | 79,7% | 95,3% | 97,4% |
| XL_001 | 18 | 94,4% | 100% | 100% |
| LX_001 / LX_002 / LX_003 | 12 / 13 / 13 | 75,0% / 84,6% / 92,3% | 100% | 100% |
| RP_001 | 10 | 60,0% | 90,0% | 90,0% |
| SP_001 | 10 | 20,0% | 80,0% | 100% |
| TL_001 | 11 | 9,1% | 36,4% | 100% |
| AI_001 | 20 | 50,0% | 60,0% | 75,0% |

- **SP_001 và TL_001** thấp chỉ vì số trang trên sheet lệch 1 so với số trang PDF, như đã ghi ở `HUONG_DAN_GIAI_DOAN_3.md`, mục 9. Tính lệch ±1 trang thì cả hai lên 100%. Sửa sheet theo bảng ở mục đó rồi chạy lại để có số chuẩn.
- **AI_001** là tài liệu yếu nhất. Mục 4 của `HUONG_DAN_GIAI_DOAN_3.md` cũng đo được chunk ở đúng trang trên sheet chỉ chứa khoảng 70% từ khoá đáp án, nên nên rà lại số trang các câu AI_001.
- **Tìm hình:** phần lớn câu trượt top 1 là do nhầm với hình anh em cùng trang hoặc trang kề (`AI_001_p42_1` với `p42_2`, các cây tìm kiếm giống nhau). Recall@5 đạt khoảng 97% ở cả hai cách tìm. Vì vậy backend nên lấy 3–5 hình rồi để mô hình hoặc người dùng chọn.
- **Recall@1 tìm hình bằng chữ dao động 64,3–66,1% giữa các lần chạy:**
  - 4 cặp hình của AI_001 (trang 12, 42, 43) không có chú thích và cùng đoạn văn xung quanh, nên phần chữ mô tả giống hệt nhau;
  - hai hình trong mỗi cặp hoà điểm, Qdrant trả về theo thứ tự bất kỳ, và có 4 câu hỏi rơi vào các cặp này;
  - muốn phân biệt phải nhìn ảnh, hoặc thêm chú thích cho các hình này ở ingestion.
- Chi tiết từng câu (hạng của kết quả đúng, top 3 tìm được) nằm ở `data/processed/danh_gia_truy_hoi.jsonl`, dùng để soát các câu trượt.

## 5. Dùng ở Giai đoạn 6 – backend

```python
from embedding.tim_kiem import TimKiem
from finetune.prompt_format import build_rag_messages, image_context, image_query_contexts, to_openai_messages

tk = TimKiem()                      # nạp mô hình khi cần; TimKiem(path="data/processed/qdrant_local") nếu không dùng Docker

# Câu hỏi bằng chữ
text_hits = tk.tim_chu(question, k=3, loai="text")        # list payload chunks.jsonl, thêm "score"
img = tk.tim_chu(question, k=1, loai="image")[0]          # hoặc tk.hinh_cua_chunk([c["chunk_id"] for c in text_hits])
ctx = [image_context(img["caption_hint"], img["page_number"], img["section"], img["caption_source"], img["slide_number"])]
ctx += [c["text"] for c in text_hits]
msgs = build_rag_messages(question, contexts=ctx, images=[img["image_path"]])
payload = {"model": "qwen2vl-nhom7", "messages": to_openai_messages(msgs)}

# Câu hỏi là ảnh: tìm hình trong tài liệu giống ảnh người dùng, rồi lấy đoạn văn cùng trang
fig = tk.tim_bang_anh(anh_nguoi_dung, k=1)[0]
chunk = tk.chunk_cung_trang(fig["doc_id"], fig["page_number"], question)
msgs = build_rag_messages(question, image_query_contexts(fig, chunk["text"] if chunk else ""), [anh_nguoi_dung])
```

- Đoạn trên đã chạy thử với Qdrant 1.20 và `prompt_format.py` hiện tại.
- Mỗi kết quả là nguyên payload kèm `score` (cosine, càng cao càng giống). Có thể đặt ngưỡng `score` để quyết định có gửi hình kèm câu trả lời hay không.
- Khi chỉ cần một loại, lọc bằng `loai="text"` hoặc `loai="image"`, và lọc theo tài liệu bằng `doc_id="XL_001"`.
- `hinh_cua_chunk(chunk_ids)` trả về các hình nằm trong đoạn văn của các chunk tìm được (trường `chunk_ids` của `images.jsonl`), xếp theo thứ tự chunk.
- Backend và `nap_qdrant.py` phải dùng cùng mô hình. Hằng `TEXT_MODEL`, `IMAGE_MODEL` nằm ở `embedding/nhung.py`. Đổi mô hình thì phải nạp lại.

## 6. Cấu hình đã chọn và lý do (dùng cho báo cáo)

| Lựa chọn | Lý do |
|---|---|
| `BAAI/bge-m3` cho chữ | Đa ngôn ngữ, đọc tốt tiếng Việt, nhận tới 8.192 token (chunk dài nhất ~1.700 ký tự). Không cần tiền tố `query:`/`passage:`, chạy offline trên máy nhóm |
| CLIP ViT-B/32 cho ảnh | Chỉ dùng để so ảnh với ảnh. Ảnh hỏi bị xoay, mờ, nén JPEG mà vẫn tìm đúng hình 86,6% ở top 1 và 97,6% ở top 5. Mô hình nhẹ (~0,6 GB) |
| Một collection, hai vector có tên | Câu hỏi chữ tìm được cả chunk lẫn hình trong một lần gọi; câu hỏi là ảnh chỉ so với các điểm có vector `image` |
| Chunk chỉ nhúng nội dung, không ghép đường dẫn mục | Bảng dưới: ghép mục làm tìm chữ kém đi, vì nhiều chunk chung một mục (nhất là Triết học) |
| Hình nhúng đủ 6 trường như mục 3 của Giai đoạn 3 | Hình ít chữ nên `section` là manh mối quan trọng. Bỏ `section` thì Recall@1 giảm gần 9 điểm |

So sánh cách nhúng (cùng mô hình, cùng bộ câu hỏi; cột tìm hình R@1 có thể lệch ±2 điểm do hoà điểm, xem mục 4):

| Cách nhúng | Tìm chữ R@1 | Tìm chữ R@5 | Tìm hình bằng chữ R@1 | Tìm hình bằng chữ R@5 |
|---|---|---|---|---|
| Chunk = đường dẫn mục + nội dung | 77,2% | 92,4% | 66,1% | 97,4% |
| **Chunk = nội dung (đang dùng)** | **80,9%** | **94,8%** | **66,1%** | **97,4%** |
| Như trên nhưng bỏ `section` khỏi chữ mô tả hình | 80,9% | 94,8% | 57,4% | 88,7% |

## 7. Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `Khong thay .../chunks.jsonl` | Chưa giải nén `du_lieu_xu_ly_v2.zip` vào thư mục gốc repo (hoặc chưa chạy `python ingestion/ingest.py`) |
| `Qdrant chua co collection 'nhom7_tai_lieu'` | Chưa chạy `python embedding/nap_qdrant.py`, hoặc đang trỏ sai chỗ: nạp bằng `--path` thì tìm/đánh giá cũng phải có `--path` |
| `Khong ket noi duoc Qdrant http://localhost:6333` (`WinError 10061`) | Docker Desktop chưa bật, hoặc chưa chạy `docker compose -f docker/docker-compose.yml up -d qdrant` |
| `Storage folder ... is already accessed by another instance` | Chế độ `--path` chỉ cho một tiến trình mở thư mục. Tắt tiến trình kia, hoặc chuyển sang Qdrant server (Docker) |
| Tải mô hình báo lỗi mạng | Lần đầu cần Internet. Đã tải xong mà vẫn lỗi thì đặt `$env:HF_HUB_OFFLINE="1"` |
| Bước nạp rất chậm | Máy đang chạy bằng CPU: kiểm tra `python -c "import torch; print(torch.cuda.is_available())"`, cài torch bản CUDA theo `HUONG_DAN_GIAI_DOAN_4.md` |
| `CUDA out of memory` khi nạp | Giảm `--batch 4`, hoặc tắt `llama-server` trong lúc nạp |
| `pip install` báo `WinError 32 ... being used by another process` | Phần mềm diệt virus đang quét file wheel lớn (torch). Chạy lại lệnh, hoặc `pip download` trước rồi cài từ file |
| Chữ tiếng Việt lỗi khi in ra PowerShell | `$env:PYTHONIOENCODING="utf-8"` |
