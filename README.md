# Mammography V26 — VS Code Streamlit

## Cài đặt

Mở thư mục bằng VS Code, mở Terminal:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Checkpoint

Tải `final_model.pt` của chính bạn vào `checkpoints/final_model.pt`. Không kèm file 109 MB trong ZIP để tiện tải.

## Chạy

```bash
streamlit run app.py
```

Trên sidebar, nhập đường dẫn tuyệt đối đến folder NPZ local (thư mục bao gồm file `.npz`, tự quét thư mục con). Ví dụ `D:\CBIS\processed_cache\npz`. Ứng dụng dò filename theo `image_key` từ `assets/prototype_sources.json`, KHÔNG dùng đường dẫn máy chủ GPU trước đây. Bạn không phải upload NPZ.

## Diễn giải đúng

- **NPZ local + mask có sẵn:** mô hình chạy cả Global/topology + prototype. Đây là *ROI-assisted inference* như đánh giá trong code, KHÔNG chứng minh auto-detection.
- **DICOM mới, không ROI:** nhánh topology sử dụng fallback trong code model, không tương đương mô hình được đánh giá. Không được coi kết quả là valid clinical prediction.
- **DICOM + ROI PNG cung cấp:** mặt nạ phải được căn đúng hình học ảnh, và preprocessing DICOM phải được kiểm chứng tương đương với pipeline train. Đoạn ứng dụng chỉ cung cấp pipeline thử nghiệm.
- Heatmap là prototype similarity, **không phải segmentation**. Đường viền xanh chỉ là ground-truth nếu NPZ có mask.
- `signed Δlogit` giải thích quyết định của *prototype branch*, không giải thích ensemble/Global toàn diện.
- Chỉ sử dụng cho mục tiêu nghiên cứu, không chẩn đoán.

## Kiểm tra nhanh

```bash
python -m compileall -q app.py inference.py model_v26.py
python smoke_test.py
```
