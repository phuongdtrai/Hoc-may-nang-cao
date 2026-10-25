# Giai đoạn 3 – Ingestion, OCR tiếng Việt và câu hỏi ảnh

Gói này làm phần Giai đoạn 3 của lộ trình: đọc 10 tài liệu PDF thành **chunk văn bản giữ cấu trúc** và **hình cắt kèm metadata** (đúng data contract để nạp Qdrant ở Giai đoạn 5), thêm **nhánh OCR tiếng Việt** cho trang scan và chữ trong ảnh, và bổ sung **242 câu hỏi ảnh** trên sheet QA_Data (115 câu có đáp án là hình, 127 câu hỏi là ảnh) để fine-tune vòng 2 và đo khả năng tìm ảnh.

## Trong gói này có gì

| Đường dẫn | Việc |
|---|---|
| `ingestion/ingest.py` | Chạy ingestion: PDF → `chunks.jsonl`, `images.jsonl`, `images/*.png`, `pages.jsonl`, `bao_cao_ingest.md` |
| `ingestion/pdf_parser.py` | Đọc PDF bằng PyMuPDF: dòng chữ, tiêu đề, bảng, hình (ảnh raster + hình vẽ vector), chú thích, slide |
| `ingestion/chunker.py` | Chia chunk theo mục, tối đa 1.200 ký tự, lặp 150 ký tự giữa hai chunk liền nhau |
| `ingestion/ocr.py` | OCR tiếng Việt: PaddleOCR (engine chính) hoặc Tesseract (dự phòng) |
| `ingestion/kiem_tra_ocr.py` | Đo CER/WER của OCR trên trang có sẵn lớp chữ (đáp án chuẩn lấy luôn từ PDF) |
| `ingestion/tao_ban_scan_thu.py` | Tạo PDF "giả scan" (chỉ có ảnh) để thử nhánh OCR từ đầu đến cuối |
| `ingestion/xem_qa_anh.py` | Kiểm tra các câu ảnh trong `qa_thu_cong.csv` (cột `image_id`) và tạo trang HTML để cả nhóm soát |
| `ingestion/tao_anh_hoi.py` | Tạo ảnh "người dùng chụp" cho câu hỏi là ảnh (`data/processed/ingest/anh_hoi/`); `build_training_data.py` tự gọi |
| `data/raw/domain_docs/qa_thu_cong.csv` | Xuất từ QA_Data ngày 21/10: 1.096 câu (847 `text`, 122 `image`, 127 `image_query`), thêm cột `image_id`, `tap` |
| `build_training_data.py` (bản 4) | Đọc câu `image`/`image_query` ngay trong QA_Data → `vlm_train.jsonl`, `vlm_val.jsonl`, `qa_anh_truy_hoi.jsonl`, `anh_hoi_truy_hoi.jsonl`; chia train/val theo trang đồng bộ cho mọi loại câu |
| `finetune/eval_truoc_sau.py` | Báo cáo tách hai nhóm ảnh val: "đáp án là hình" và "câu hỏi là ảnh" |
| `finetune/prompt_format.py` | `image_context()` và `image_query_contexts()`: dòng "Ngữ cảnh" mô tả hình, dùng chung cho train và backend |
| `data/processed/ingest/` | Kết quả đã chạy sẵn trên 10 tài liệu (có OCR chữ trong hình bằng Tesseract) |

Mọi lệnh chạy **tại thư mục gốc repo**, trong `.venv` đang dùng để train (đã có PyMuPDF 1.28.2 và Pillow).

## 1. Lệnh chính

```powershell
pip install rapidfuzz pytesseract                 # rapidfuzz: tính CER nhanh; pytesseract: chỉ cần khi dùng Tesseract
python ingestion/ingest.py                        # 1. ingestion 10 tài liệu, khoảng 2–3 phút
python ingestion/xem_qa_anh.py                    # 2. kiểm tra câu ảnh trong QA_Data + trang HTML để soát
python build_training_data.py                     # 3. dữ liệu train/val (tự tạo ảnh hỏi nếu thiếu)
python finetune/eval_truoc_sau.py                 # 4. đo adapter vòng 1 trên 54 câu ảnh val (mốc so sánh)
```

`data/processed/ingest/` đã có kết quả chạy sẵn, nên có thể bắt đầu từ bước 2. Chỉ chạy lại bước 1 khi sửa code hoặc thêm tài liệu. Cùng PyMuPDF 1.28.2 thì chạy lại sẽ ra đúng các `image_id` cũ, nên cột `image_id` trên sheet vẫn khớp. Nếu máy chưa cài OCR, bước 1 vẫn chạy bình thường, chỉ để trống `ocr_text` của hình (ghi đè phần `ocr_text` đang có sẵn).

Chỉ chạy lại vài tài liệu: `python ingestion/ingest.py --docs XL_001 AI_001` (kết quả các tài liệu khác giữ nguyên).

## 2. Ingestion làm những gì

Với mỗi trang PDF:

1. **Đọc chữ theo từng ký tự** để sửa lỗi của chính file PDF:
   - PDF xuất từ LaTeX/Foxit hay mất dấu cách sau chữ có dấu ("hệthống"). Script đo khoảng trống giữa hai ký tự và thêm lại dấu cách: RP_001 498 chỗ, SP_001 499, TL_001 392, LX_001 191, XL_001 69.
   - Font Symbol (¬ ∧ ∨ ⇒ ∀ ∃, chữ Hy Lạp) được đổi về ký tự Unicode đúng.
2. **Bỏ phần không phải nội dung:** đầu trang/chân trang lặp lại, số trang, trang mục lục (18 trang), chú thích chân trang chèn giữa đoạn (Triết học), mảnh công thức rời.
3. **Nhận tiêu đề và cấp mục:** Phần, Chương, mục La Mã, mục số `1.2.3`, mục chữ `a.`, hoặc chữ to hơn thân bài. Mỗi chunk mang đường dẫn mục, ví dụ `1.1. Giới thiệu về hệ điều hành Linux > 1.1.2 Linus Torvalds`.
4. **Bảng** → chunk riêng dạng markdown; bảng dài cắt theo hàng, lặp lại dòng tiêu đề.
5. **Hình:**
   - Ảnh raster: bỏ logo/hình trang trí lặp ở ≥ 3 trang.
   - Hình vẽ vector (sơ đồ, cây tìm kiếm…): gom các nét vẽ gần nhau. Vùng công thức, chữ tô nền và khung chữ không bị nhận nhầm là hình.
   - Chú thích: ưu tiên dòng "Hình x.y …" ngay trên/dưới hình, sau đó đến dòng ngắn căn giữa sát hình.
   - Ảnh cắt được phóng to tới ít nhất khoảng 1.000 điểm ảnh ở cạnh dài để chữ trong hình đọc được.
6. **Slide in 4 slide/trang (LX_001):** nhận từng khung slide và số slide, lấy tiêu đề slide. Mỗi slide là 1 chunk, và slide có hình được cắt nguyên slide.
7. **OCR (nếu cần):** trang scan hoặc chữ lỗi font được đọc lại bằng OCR. Chữ trong ảnh raster/slide được OCR vào `ocr_text` để tìm ảnh theo nội dung chữ trong ảnh.
8. **Chia chunk** theo mục, tối đa 1.200 ký tự, bằng đúng độ dài "Ngữ cảnh" lúc fine-tune. Nhờ vậy chunk Qdrant trả về có cùng kích thước với ngữ cảnh mô hình đã học.

## 3. Đầu ra và data contract

`data/processed/ingest/`:

| File | Nội dung |
|---|---|
| `chunks.jsonl` | `chunk_id, doc_id, type (text\|table), text, section, page_number, pages, slide_number, source (pdf\|ocr), n_chars` |
| `images.jsonl` | `image_id, doc_id, page_number, bbox [x0,y0,x1,y1 theo điểm PDF], page_size, kind (raster\|vector\|mixed\|slide), caption_hint, caption_source, slide_number, section, text_in_image, ocr_text, context_before, context_after, image_path, width, height, chunk_ids` |
| `images/<image_id>.png` | Hình cắt, `image_id` = `<doc_id>_p<trang>_<thứ tự>` |
| `pages.jsonl` | Từng trang: số ký tự, tỉ lệ ảnh phủ trang, có OCR không và vì sao |
| `bao_cao_ingest.md` | Bảng tổng hợp, kiểm tra QA, cảnh báo |
| `kiem_tra_ocr*.md/.json` | Kết quả đo OCR (mục 5) |
| `qa_anh_xem_lai.html` | Trang soát câu hỏi ảnh (mục 6) |

Khớp với data contract trong lộ trình:
- chunk `{text, page_number, section, doc_id}`: có đủ, thêm `chunk_id`, `pages`, `type`.
- ảnh `{image_base64, page_number, bbox, caption_hint, doc_id}`: có đủ. `image_base64` chỉ ghi khi chạy `--base64` (file nặng thêm khoảng 25 MB), mặc định dùng `image_path`.
- Qdrant ở Giai đoạn 5: `{id, vector, payload:{type:"text"|"image", ...}}`. Payload lấy nguyên một dòng của `chunks.jsonl` hoặc `images.jsonl`. Với ảnh, nên nhúng chuỗi `caption_hint + section + context_before + context_after + text_in_image + ocr_text`.
- `chunk_ids` của mỗi hình trỏ tới chunk chứa đoạn văn quanh hình, để khi tìm thấy chunk thì trả kèm hình.

## 4. Kết quả trên 10 tài liệu

| Tài liệu | Trang | Chunk chữ / bảng | Hình | Hình có chú thích |
|---|---|---|---|---|
| AI_001 | 111 | 152 / 7 | 36 (21 raster, 15 vector) | 6 |
| LX_001 (slide) | 12 | 32 / 0 | 30 slide | 7 tiêu đề slide |
| LX_002 | 2 | 7 / 11 | 2 | 0 |
| LX_003 | 92 | 184 / 2 | 41 ảnh chụp màn hình | 3 |
| RP_001 | 14 | 31 / 1 | 0 | – |
| SP_001 | 15 | 27 / 1 | 2 sơ đồ vector | 0 |
| TL_001 | 12 | 26 / 1 | 0 | – |
| TRIET_001 | 214 | 673 / 0 | 0 | – |
| TTHCM_001 | 120 | 435 / 0 | 0 | – |
| XL_001 | 76 | 101 / 3 | 26 (23 vector) | 24 |
| **Tổng** | **668** | **1.668 / 26** | **137** | |

Chạy hết khoảng 3 phút trên CPU, tính cả OCR chữ trong 97 hình raster/slide (73 hình đọc được chữ).

**Kiểm tra chunk có chứa đáp án QA không:** với mỗi câu QA loại text, lấy các chunk ở đúng trang ghi trên sheet và đo tỉ lệ từ khoá của đáp án có trong chunk. Triết học đạt 100%, Tư tưởng Hồ Chí Minh 99%, LX_001/LX_002/LX_003 100%, XL_001 89%, RP_001 90%, AI_001 70%. Hai tài liệu thấp là SP_001 (40%) và TL_001 (55%). Lý do không nằm ở ingestion mà ở **số trang ghi trên sheet lệch 1**: hai file này có trang bìa không đánh số, nên số in ở chân trang nhỏ hơn số trang PDF 1 đơn vị, và người soạn đã ghi theo số in. Danh sách cần sửa ở mục 9.

## 5. OCR tiếng Việt

### Khi nào OCR chạy

| Trường hợp | Cách nhận biết | Kết quả |
|---|---|---|
| Trang scan / ảnh chụp | < 50 ký tự chữ và ảnh phủ ≥ 40% trang | Cả trang đọc bằng OCR, chunk có `source = "ocr"` |
| Chữ lỗi font | ≥ 15% ký tự lạ | Cả trang đọc lại bằng OCR |
| Chữ trong ảnh raster, slide | luôn chạy khi có engine | Lưu vào `ocr_text` của hình |
| Ép OCR | `--ocr force` | Mọi trang đọc bằng OCR |

Cả 10 tài liệu hiện tại đều có lớp chữ tốt, nên không trang nào phải OCR. Nhánh OCR dùng cho tài liệu scan nhóm thêm sau này và cho chữ trong ảnh.

### Cài engine

**PaddleOCR (engine chính, như báo cáo giữa kỳ)** – nên cài trong môi trường riêng, đặt **ngoài** thư mục repo (giống `llama.cpp`), để không đụng PyTorch của môi trường train và không lọt vào Git:

```powershell
python -m venv "C:\Machine Learning\venv-ocr"
& "C:\Machine Learning\venv-ocr\Scripts\Activate.ps1"
pip install paddlepaddle==3.3.1 paddleocr pymupdf==1.28.2 pillow rapidfuzz
python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='vi')"     # lần đầu tự tải mô hình (cần Internet)
```

`paddlepaddle` bản CPU là đủ: OCR chỉ chạy cho trang scan và hình, không cần GPU.

**Tesseract (dự phòng, nhẹ):** cài bản Windows của UB Mannheim (trang [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)), tick *Vietnamese* trong mục ngôn ngữ, rồi `pip install pytesseract`. Script tự tìm `C:\Program Files\Tesseract-OCR\tesseract.exe`; nếu cài chỗ khác thì đặt biến `TESSERACT_CMD`. Kết quả đo dưới đây dùng gói `vie` bản *best*: tải [vie.traineddata](https://github.com/tesseract-ocr/tessdata_best/raw/main/vie.traineddata) chép đè vào `C:\Program Files\Tesseract-OCR\tessdata\`.

`--ocr_engine auto` (mặc định) dùng PaddleOCR nếu có, không thì Tesseract, không có gì thì bỏ qua OCR và ghi cảnh báo.

### Độ chính xác đo được

`kiem_tra_ocr.py` render 12 trang của 6 tài liệu thành ảnh 200 dpi, cho OCR đọc rồi so với chữ gốc trong PDF:

| Engine | Ảnh | CER | CER bỏ dấu | WER (âm tiết) | Giây/trang |
|---|---|---|---|---|---|
| Tesseract 5.3.4 + vie | render sạch | 1,3% | 0,9% | 3,5% | 9,5 |
| Tesseract 5.3.4 + vie | giả scan (nghiêng, nhiễu, JPEG) | 1,3% | 0,8% | 3,7% | 10,1 |
| PaddleOCR | | *chạy trên máy nhóm* | | | |

- Trang văn bản thường (Triết học, Tư tưởng HCM, AI_001 tr.6) có CER 0,3–0,7%, gần như không sai.
- Trang khó nhất là LX_003 tr.40 (ảnh chụp menu, chữ nhỏ trên nền màu), CER 5,7%.
- Chênh lệch giữa CER và CER bỏ dấu chỉ khoảng 0,4 điểm, tức lỗi dấu thanh chiếm phần nhỏ.

Đo PaddleOCR (trong `venv-ocr`, tại thư mục repo) để điền dòng còn trống:

```powershell
python ingestion/kiem_tra_ocr.py --engine paddle tesseract
python ingestion/kiem_tra_ocr.py --engine paddle tesseract --scan --out data/processed/ingest/kiem_tra_ocr_scan
```

### Thử nhánh OCR từ đầu đến cuối

```powershell
python ingestion/tao_ban_scan_thu.py --doc TRIET_001 --pages 20-22
python ingestion/ingest.py --input data/raw/scan_test --out data/processed/ingest_scan
```

File giả scan không còn lớp chữ. Ingestion tự nhận ra (`ocr = "trang_scan"` trong `pages.jsonl`), đọc bằng OCR rồi chia chunk như bình thường. Nhớ dùng `--out` riêng để không trộn vào kết quả chính.

## 6. Câu hỏi ảnh trong QA_Data (loai `image` và `image_query`)

Từ ngày 21/10 mọi câu hỏi nằm trong **một tab QA_Data** (tab QA_Anh cũ đã gộp vào và xoá). Sheet có thêm 2 cột:

| Cột | Ý nghĩa |
|---|---|
| `image_id` | Mã hình do ingestion cắt ra, dạng `<doc_id>_p<trang>_<số thứ tự>` (vd `XL_001_p7_2`). Bắt buộc với `image` và `image_query` |
| `tap` | `train` hoặc `val`. Để trống thì script tự chia theo trang |

Có hai kiểu câu hỏi ảnh, dùng chung các hình trong `data/processed/ingest/images/`:

| `loai` | Người dùng làm gì | Mô hình nhận | Dùng để đo |
|---|---|---|---|
| `image` (115 câu, STT 855–969) | Hỏi bằng chữ, đáp án là một hình: *"Cho xem sơ đồ tổng quát của một hệ thống xử lý ảnh"* | hình cắt + "Ngữ cảnh" = trang, mục, chú thích của hình | tìm ảnh bằng câu chữ (`qa_anh_truy_hoi.jsonl`: câu hỏi → `image_id`) |
| `image_query` (127 câu, STT 970–1096) | Gửi ảnh chụp một hình trong tài liệu rồi hỏi: *"Hình này minh họa thuật toán gì?"* | ảnh "người dùng chụp" + "Ngữ cảnh" = mô tả hình + đoạn văn cùng trang | tìm trang/hình từ ảnh (`anh_hoi_truy_hoi.jsonl`: ảnh → `image_id`, trang) |
| `image` không có `image_id` (7 câu cũ) | như vòng 1 | ảnh cả trang PDF (120 DPI) | giữ để so sánh với vòng 1 |

**Ảnh "người dùng chụp".** Người dùng thật không gửi đúng hình cắt sạch mà gửi ảnh chụp bằng điện thoại hoặc chụp màn hình: hơi nghiêng, mờ, sáng tối khác, nén JPEG, có viền giấy. `ingestion/tao_anh_hoi.py` biến mỗi hình cắt thành một ảnh như vậy (xoay ±2°, méo phối cảnh nhẹ, chỉnh sáng/tương phản, làm mờ, nhiễu, JPEG chất lượng 55–80) và lưu ở `data/processed/ingest/anh_hoi/<image_id>.jpg`, thông số từng ảnh ghi trong `anh_hoi.jsonl`. Ảnh cố định theo `image_id`, chạy lại vẫn ra đúng file cũ. `build_training_data.py` tự gọi script này khi thiếu ảnh. Mỗi ảnh hỏi chiếm 196–1.024 token ảnh (trung vị 980).

**Chia train/val theo trang, đồng bộ cho mọi loại câu.** Một trang không bao giờ vừa ở train vừa ở val, kể cả giữa câu chữ và câu ảnh. Thứ tự ưu tiên khi chia câu ảnh:
1. trang đã train ở vòng 1 bằng ảnh cả trang (7 câu cũ) → train;
2. trang có câu text → cùng tập với câu text của trang đó;
3. cột `tap` trên sheet;
4. còn lại: chia 80/20 theo trang, seed 42.

Nếu cột `tap` trên sheet khác với (1)–(2), script dùng (1)–(2) và ghi vào `bao_cao_du_lieu.txt` để sửa sheet. Cột `tap` hiện đã khớp, báo cáo không có cảnh báo nào loại này.

| Tài liệu | `image` train | `image` val | `image_query` train | `image_query` val | Loại hình |
|---|---|---|---|---|---|
| XL_001 | 17 | 7 | 17 | 9 | sơ đồ, hình minh hoạ thuật toán có số "Hình x.y" |
| AI_001 | 25 | 7 | 25 | 7 | cây tìm kiếm, bàn cờ, bản đồ, bảng vết thuật toán |
| LX_001 + LX_002 | 18 | 4 | 22 | 4 | slide: biểu đồ, infographic, ảnh chụp màn hình cài đặt |
| LX_003 | 28 | 7 | 34 | 7 | ảnh chụp màn hình Ubuntu |
| SP_001 | 1 | 1 | 1 | 1 | sơ đồ tấn công |
| **Tổng** | **89** | **26** | **99** | **28** | |

**Mẫu train của câu hỏi là ảnh** (một dòng trong `vlm_val.jsonl`, rút gọn):

```json
{"image": "data/processed/ingest/anh_hoi/XL_001_p13_1.jpg",
 "instruction": "Trả lời câu hỏi dựa trên hình ảnh và ngữ cảnh sau.",
 "input": "Ngữ cảnh: Hình ảnh ở trang 13, mục \"1.2.2. Biểu diễn ảnh\", chú thích: Hình 1.4. Quá trình hiển thị và chỉnh sửa, lưu trữ ảnh thông qua DIB.\nHiện nay trên thế giới có trên 50 khuôn dạng ảnh thông dụng ...\nCâu hỏi: Sơ đồ có các khối BMP, PCC, DIB và Cửa sổ trong ảnh mô tả quá trình gì?",
 "answer": "Đây là Hình 1.4 (trang 13): quá trình hiển thị, chỉnh sửa và lưu trữ ảnh thông qua DIB. ...",
 "meta": {"doc_id": "XL_001", "page": "13", "loai": "image_query", "image_id": "XL_001_p13_1", "tap": "val", "stt": "973",
          "kieu": "cau_hoi_la_anh", "hinh_goc": "data/processed/ingest/images/XL_001_p13_1.png"}}
```

"Ngữ cảnh" của câu hỏi là ảnh sinh bằng `prompt_format.image_query_contexts(hình, đoạn văn cùng trang)`: đúng thứ backend sẽ có sau khi tìm ra hình khớp với ảnh người dùng. Câu loại `image` vẫn dùng `image_context()` như trước.

**Quy trình duyệt (cả nhóm):**
1. Xuất QA_Data: mở tab QA_Data → Tệp → Tải xuống → CSV, lưu đè `data/raw/domain_docs/qa_thu_cong.csv`.
2. `python ingestion/xem_qa_anh.py` rồi mở `data/processed/ingest/qa_anh_xem_lai.html`: mỗi thẻ có hình gốc, ảnh hỏi (với `image_query`), câu hỏi, đáp án; lọc theo tài liệu, loại câu, tập train/val.
3. Câu sai thì sửa ngay trên QA_Data. Các câu ảnh do AI hỗ trợ soạn được giao cho Nguyen Duc Phuong rà soát (cột `nguoi_soan`).
4. Xuất lại CSV và chạy `python build_training_data.py`.

**Thêm câu mới:** tìm `image_id` trong trang HTML hoặc `images.jsonl`, thêm dòng với `loai` = `image` hoặc `image_query`, ghi `image_id`, để trống `tap`. `qa_anh.csv` cũ không còn dùng; nếu file vẫn nằm trong thư mục thì script bỏ qua và báo một dòng.

## 7. Fine-tune vòng 2 với câu hỏi ảnh

```powershell
python build_training_data.py                       # llm 761/86, vlm_train 195, vlm_val 54
python finetune/eval_truoc_sau.py                   # (a) mốc: adapter vòng 1 trên 54 câu ảnh val
python finetune/train_qwen2vl.py --dry_run          # 956 mẫu train = 761 câu chữ + 195 câu ảnh
python finetune/train_qwen2vl.py --output_dir finetune/outputs/qwen2vl-2b-nhom7-lora-v2
python finetune/eval_truoc_sau.py --adapter finetune/outputs/qwen2vl-2b-nhom7-lora-v2 --out_dir finetune/outputs/danh_gia_v2
```

- `vlm_train.jsonl` 195 câu = 7 câu ảnh cả trang + 89 câu đáp án là hình + 99 câu hỏi là ảnh; `vlm_val.jsonl` 54 câu = 26 + 28, không dùng khi train.
- Báo cáo đánh giá tách hai dòng: **"Ảnh – đáp án là hình, tập val"** và **"Ảnh – câu hỏi là ảnh, tập val"**. Hai dòng này đo đúng phần đọc hình, thay cho nhóm 7 câu ảnh đã dùng khi train ở vòng 1.
- Phần câu chữ: 761 train, 86 val (vòng 1 là 763/84). Chênh 2 câu vì một câu TTHCM_001 đã được sửa số trang trên sheet nên cả nhóm trang đổi tập. Khi so sánh với vòng 1, dùng đúng tập val mới cho cả mô hình gốc và hai adapter.
- Mẫu ảnh dài nhất khoảng 1.024 token ảnh + dưới 1.200 ký tự ngữ cảnh, vẫn trong giới hạn 2.048 token; `--dry_run` in ra số chính xác. Mỗi epoch khoảng 120 bước. Nếu hết VRAM, chạy thêm `--batch 1 --grad_accum 8`.
- Vẫn đóng băng phần thị giác ở vòng 2. 195 câu ảnh là đủ để bắt đầu thử bật lại ở vòng 3 (so sánh hai cấu hình trên cùng 54 câu val).
- Bộ dữ liệu vòng 1 vẫn lưu ở `data/processed/vong1/`.

## 8. Dùng ở Giai đoạn 5–6

**Giai đoạn 5 – nạp Qdrant:**
- Mỗi dòng `chunks.jsonl` là một điểm `type: "text"`.
- Mỗi dòng `images.jsonl` là một điểm `type: "image"`, nhúng phần chữ mô tả hình như ở mục 3.
- Đo tìm ảnh bằng `qa_anh_truy_hoi.jsonl` (câu chữ → `image_id`, Recall@1, Recall@5), tìm từ ảnh bằng `anh_hoi_truy_hoi.jsonl` (ảnh hỏi → đúng `image_id` hoặc đúng trang) và tìm chữ bằng QA_Data (đúng `doc_id` + trang).

**Giai đoạn 6 – backend:** khi kết quả có hình, gửi kèm dòng mô tả hình như lúc train:

```python
from finetune.prompt_format import build_rag_messages, image_context, image_query_contexts, to_openai_messages

ctx = [image_context(img["caption_hint"], img["page_number"], img["section"], img["caption_source"], img["slide_number"])]
ctx += [c["text"] for c in text_hits]
msgs = build_rag_messages(question, contexts=ctx, images=[img["image_path"]])
payload = {"model": "qwen2vl-nhom7", "messages": to_openai_messages(msgs)}

# Câu hỏi là ảnh: tìm hình khớp với ảnh người dùng (OCR/nhúng ảnh), rồi
# msgs = build_rag_messages(question, image_query_contexts(fig, page_chunk), [anh_nguoi_dung])
```

## 9. Vấn đề dữ liệu phát hiện được (sửa trên sheet QA_Data)

| Câu | Đang ghi | Nên sửa | Căn cứ |
|---|---|---|---|
| STT 3 (XL_001, loại image) | trang 8 | **7** | "Hình 1.2" nằm ở trang 7 của PDF |
| STT 20 (XL_001, loại image) | trang 67 | **66** | "Hình 5.10" nằm ở trang 66 |
| SP_001 STT 95, 96, 98, 100, 101, 102 | 5, 5, 7, 8, 9, 10 | **6, 6, 8, 9, 10, 11** | Đáp án nằm ở trang liền sau (độ phủ 15–42% → 94–100%) |
| TL_001 STT 84, 85, 86, 87, 89 | 2, 4, 4, 4, 6 | **3, 5, 5, 5, 7** | Như trên (0–38% → 95–100%) |
| RP_001 STT 113 | 11 | **12** | Như trên (32% → 100%) |
| XL_001 STT 19 | 65 | **66** | 48% → 86% |
| AI_001 STT 24 | 7 | **6** | 24% → 96% |

- Hai câu ảnh đầu bảng: trong lúc chưa sửa sheet, `build_training_data.py` tự nhận ra hình nằm ở trang khác và render đúng trang. Báo cáo `bao_cao_du_lieu.txt` có mục nhắc sửa. Ở vòng 1, hai câu này đã được train với ảnh trang sai.
- Sửa xong chạy lại `python build_training_data.py`. Đổi số trang có thể làm vài câu ở 5 tài liệu nhỏ chuyển giữa train và val, nên sửa sheet **trước** khi train vòng 2, rồi so sánh vòng 2 với mô hình gốc trên tập val mới. Nhóm câu ảnh val (`vlm_val.jsonl`) không bị ảnh hưởng.

## 10. Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `Khong khoi tao duoc engine OCR nao` | Chưa cài PaddleOCR/Tesseract (mục 5). Ingestion vẫn chạy, chỉ bỏ OCR |
| `Tesseract chua co goi ngon ngu 'vie'` | Cài lại Tesseract và tick Vietnamese, hoặc chép `vie.traineddata` vào thư mục `tessdata` |
| PaddleOCR báo lỗi khi tải mô hình | Lần đầu cần Internet. Chạy lại lệnh `PaddleOCR(lang='vi')` ở mục 5 |
| `xem_qa_anh.py` báo "khong co hinh …" | `image_id` trên sheet không có trong `images.jsonl`: chạy lại `python ingestion/ingest.py`, hoặc sửa `image_id` theo trang HTML |
| Báo cáo có mục "Cột tap trên sheet khác với cách chia theo trang" | Sửa cột `tap` của các câu được liệt kê cho khớp, rồi xuất lại CSV |
| `build_training_data.py` báo "chờ ảnh" | Chưa có `data/processed/ingest/`: chạy ingestion trước |
| Chữ tiếng Việt lỗi khi in ra PowerShell | `$env:PYTHONIOENCODING="utf-8"` |
| Muốn file `images.jsonl` có ảnh base64 | `python ingestion/ingest.py --base64` (file nặng thêm khoảng 25 MB) |
