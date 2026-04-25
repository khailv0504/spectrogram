# project_spectrogram

Dự án huấn luyện mô hình phân loại ảnh spectrogram tín hiệu radar bằng PyTorch.

Luồng chính hiện tại:
1. Đọc config từ `config/config.yaml`
2. Tạo train/validation loader từ ảnh `.png`
3. Train model CNN + CBAM
4. Vẽ report/curve/confusion matrix
5. Export TorchScript model (`.pt`)

## 1. Cấu trúc chính

```text
project_spectrogram/
├── config/
│   ├── __init__.py         # load_config()
│   └── config.yaml         # cấu hình train/data/path
├── datasets/
│   ├── processing.py       # split + transform + dataloader
│   └── spectrogram_dataset.py
├── module/                 # model blocks + SpectrumExpertModel
├── train/
│   └── training.py         # vòng lặp train/val
├── utils/
│   └── visualize.py        # report + curve + confusion matrix
├── eval/
│   └── evaluating.py       # script evaluate riêng (thủ công MODEL_PATH)
└── main.py                 # entrypoint train chính
```

## 2. Yêu cầu môi trường

- Python 3.10+ (khuyến nghị dùng venv)
- GPU CUDA (tùy chọn, code tự fallback CPU)

Package cần có:
- `torch`, `torchvision`, `torchaudio`
- `timm`
- `numpy`, `pandas`, `scikit-learn`
- `matplotlib`, `seaborn`, `tqdm`, `Pillow`
- `pyyaml` (chỉ cần nếu bạn muốn dùng cú pháp YAML đầy đủ thay vì JSON-style)

Cài nhanh:

```powershell
cd D:\deep_learning
.\venv\Scripts\activate
pip install torch torchvision torchaudio timm numpy pandas scikit-learn matplotlib seaborn tqdm pillow pyyaml
```

## 3. Định dạng dữ liệu đầu vào

`train_root_dir` phải có cấu trúc kiểu:

```text
train_root_dir/
├── class_a/
│   ├── sample_snr10_doppler3_xxx.png
│   └── ...
├── class_b/
│   └── ...
```

Lưu ý quan trọng:
- Loader chỉ quét file `*.png`
- Tên file phải chứa được pattern:
  - `snr(\d+)`
  - `doppler(\d+)`
- Thư mục cha trực tiếp của ảnh được dùng làm label class

## 4. Cấu hình

File config: `config/config.yaml`

Hiện tại file đang dùng JSON-style (vẫn hợp lệ vì loader đọc JSON trước):

```json
{
  "paths": {
    "train_root_dir": "D:/.../radarcommunsignaldata2026train/versions/1",
    "output_dir": "output",
    "model_output_path": "output/TrainedModel.pt",
    "train_log_path": "output/train_log.csv"
  },
  "model": {
    "num_classes": 12
  },
  "data": {
    "val_split": 0.2,
    "batch_size_train": 64,
    "batch_size_val": 32
  },
  "training": {
    "num_epochs": 30,
    "learning_rate": 0.0003
  },
  "visualization": {
    "save_dir": "output",
    "confusion_threshold": 3
  }
}
```

Ghi chú:
- Path tương đối sẽ được resolve theo root dự án.
- Thư mục output sẽ tự tạo nếu chưa tồn tại.

## 5. Cách chạy train

Chạy từ thư mục cha `D:\deep_learning` để import `project_spectrogram.*` ổn định:

```powershell
cd D:\deep_learning
.\venv\Scripts\activate
python -m project_spectrogram.main
```

## 6. Kết quả đầu ra

Mặc định sinh trong `project_spectrogram/output/`:
- `train_log.csv`
- `loss_curve.pdf`
- `accuracy_curve.pdf`
- `raw_cm.pdf`
- `confusion_matrix.pdf`
- `TrainedModel.pt`

## 7. Evaluate riêng trên test set

Script: `eval/evaluating.py`

Trước khi chạy, cần sửa:
- `MODEL_PATH`
- `TEST_DATASET_DIR` (nếu khác đường dẫn mặc định)

Rồi chạy:

```powershell
cd D:\deep_learning\project_spectrogram
python eval\evaluating.py
```

## 8. Lỗi thường gặp

1. `ModuleNotFoundError: No module named 'project_spectrogram'`
- Nguyên nhân: chạy sai thư mục.
- Cách đúng: chạy `python -m project_spectrogram.main` từ `D:\deep_learning`.

2. `ModuleNotFoundError: No module named 'torchaudio'`
- Cài thêm `torchaudio` đúng theo version `torch`.

3. `No .png files found under ...`
- Kiểm tra lại `paths.train_root_dir` và định dạng dữ liệu.
