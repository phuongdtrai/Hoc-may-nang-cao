# Hoc-may-nang-cao – Nhóm 7: Multimodal RAG on-premise với Qwen2-VL-2B

**Bắt đầu từ đây:** tài liệu PDF, dữ liệu đã xử lý và mô hình không nằm trên GitHub. Làm theo [`HUONG_DAN_DU_LIEU.md`](HUONG_DAN_DU_LIEU.md) để lấy về đúng chỗ và tạo môi trường trước khi chạy bất kỳ script nào.

| Đường dẫn | Nội dung |
|---|---|
| `ingestion/` | Giai đoạn 3: đọc PDF thành chunk và hình cắt, OCR tiếng Việt, tạo ảnh hỏi — [`HUONG_DAN_GIAI_DOAN_3.md`](HUONG_DAN_GIAI_DOAN_3.md) |
| `build_training_data.py` | Chuẩn hoá câu hỏi trên sheet QA_Data thành dữ liệu train/val |
| `finetune/` | Giai đoạn 4: fine-tune Qwen2-VL-2B bằng QLoRA, đánh giá trước/sau, gộp và xuất GGUF — [`HUONG_DAN_GIAI_DOAN_4.md`](HUONG_DAN_GIAI_DOAN_4.md) |
| `finetune/prompt_format.py` | "Hợp đồng prompt": backend Giai đoạn 6 phải dựng prompt bằng file này |
| `docker/docker-compose.yml` | Qdrant cho Giai đoạn 5 |
| `requirements-ingest.txt`, `requirements-train.txt` | Thư viện cho máy thường và máy có GPU NVIDIA |
