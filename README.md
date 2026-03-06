# TransTResearch - Visual Object Tracking Research Project

Dự án nghiên cứu và đánh giá các thuật toán theo dõi đối tượng (Visual Object Tracking) sử dụng **PyTracking framework** và **OTB benchmark**.

## :fire: Tổng quan

Dự án này tích hợp các công cụ và framework mạnh mẽ để nghiên cứu về visual object tracking:

- **PyTracking**: Framework Python cho visual object tracking và video object segmentation dựa trên PyTorch
- **OTB Toolkit**: Công cụ đánh giá hiệu suất tracker trên Online Tracking Benchmark (OTB-2013, OTB-2015, OTB-100)
- **ToMP Tracker**: Triển khai và tối ưu hóa các biến thể của ToMP (Transformer-based Model-free Tracking via Relative Position Encoding)

## 📁 Cấu trúc thư mục

```
TransTResearch/
├── pytracking/              # PyTracking framework
│   ├── ltr/                # Learning Tracking Representations - training framework
│   │   ├── actors/         # Training actors
│   │   ├── dataset/        # Training datasets
│   │   ├── models/         # Network models
│   │   └── train_settings/ # Training configurations
│   ├── pytracking/         # Tracking evaluation and inference
│   │   ├── tracker/        # Tracker implementations (ToMP, DiMP, ATOM, etc.)
│   │   ├── parameter/      # Tracker parameters
│   │   ├── evaluation/     # Evaluation scripts
│   │   ├── experiments/    # Experiment configurations
│   │   └── tracking_results/ # Tracking results
│   └── pretrained_network/ # Pre-trained model weights
├── otb-toolkit/            # OTB benchmark evaluation toolkit
│   ├── configs/            # Tracker and sequence configurations
│   ├── sequences/          # OTB video sequences
│   ├── results/            # Tracking results
│   ├── perfmat/            # Performance matrices
│   └── figs/               # Performance plots
├── otb100/                 # OTB100 benchmark results
│   ├── plots/              # Performance plots for different runs
│   └── otb_matlab_export/  # Exported MATLAB results
└── venv312/                # Python virtual environment

```

## 🚀 Tính năng chính

### PyTracking Framework
- Hỗ trợ nhiều tracker state-of-the-art: **ToMP**, **KeepTrack**, **DiMP**, **PrDiMP**, **ATOM**, **KYS**, **LWL**
- Framework training hoàn chỉnh với LTR (Learning Tracking Representations)
- Tích hợp các dataset phổ biến: OTB, VOT, LaSOT, GOT-10k, TrackingNet, NFS, UAV123
- Công cụ phân tích và đánh giá hiệu suất tracker
- Hỗ trợ training và inference trên GPU

### OTB Toolkit
- Đánh giá tracker trên OTB benchmark (OTB-2013, OTB-2015, OTB-100)
- Tự động download và cấu hình video sequences
- Vẽ success plots và precision plots
- So sánh hiệu suất với các tracker state-of-the-art

### ToMP Tracker Variants
Dự án bao gồm nhiều biến thể của ToMP tracker được tối ưu hóa:
- `tomp50`: Base model với ResNet-50 backbone
- `tomp101`: Model với ResNet-101 backbone
- `tomp50_auc60`: Tối ưu cho AUC score
- `tomp50_realtime`: Tối ưu cho real-time tracking
- `tomp50_realtime_balanced`: Cân bằng giữa tốc độ và độ chính xác
- `tomp50_auc60_plus`, `plus2`, `plus3`: Các biến thể cải tiến

## 📋 Yêu cầu hệ thống

### Python Environment
- Python 3.7 hoặc cao hơn
- CUDA-capable GPU (NVIDIA)
- Conda hoặc pip để quản lý packages

### MATLAB (cho OTB Toolkit)
- MATLAB R2016b hoặc cao hơn

## 🔧 Cài đặt

### 1. Clone repository

```bash
git clone <repository-url>
cd TransTResearch
```

### 2. Tạo và kích hoạt virtual environment

**Windows:**
```powershell
python -m venv venv312
.\venv312\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
python -m venv venv312
source venv312/bin/activate
```

### 3. Cài đặt PyTracking

#### Cài đặt PyTorch
```bash
# CUDA 10.2
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu102

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Cài đặt dependencies
```bash
pip install matplotlib pandas tqdm opencv-python visdom tb-nightly scikit-image tikzplotlib gdown
pip install cython pycocotools lvis
```

Chi tiết xem: [pytracking/INSTALL.md](pytracking/INSTALL.md) hoặc [pytracking/INSTALL_win.md](pytracking/INSTALL_win.md)

### 4. Download pretrained models

```bash
cd pytracking
python pytracking/util_scripts/download_models.py
```

Hoặc download thủ công từ [Model Zoo](pytracking/MODEL_ZOO.md) và đặt vào `pytracking/pretrained_network/`

## 🎯 Sử dụng

### Chạy tracker trên video/webcam

```bash
cd pytracking

# Chạy trên video
python pytracking/run_video.py tomp tomp50 --video_name <path_to_video>

# Chạy trên webcam
python pytracking/run_webcam.py tomp tomp50

# Chạy với optional display
python pytracking/run_video.py tomp tomp50 --video_name video.mp4 --optional_box 100,100,200,200
```

### Đánh giá trên OTB benchmark

```bash
cd pytracking

# Đánh giá trên toàn bộ OTB100
python pytracking/run_tracker.py tomp tomp50 --dataset_name otb --threads 4

# Đánh giá trên sequence cụ thể
python pytracking/run_tracker.py tomp tomp50 --dataset_name otb --sequence Basketball

# Với debug mode
python pytracking/run_tracker.py tomp tomp50 --dataset_name otb --debug 1
```

### Sử dụng OTB Toolkit (MATLAB)

```matlab
cd otb-toolkit

% Chạy evaluation
matlab -r "run_OPE;exit;"

% Hoặc trong MATLAB
run_OPE  % Đánh giá toàn bộ benchmark
```

### Training tracker mới

```bash
cd pytracking

# Training ToMP
python ltr/run_training.py tomp tomp50

# Training với custom settings
python ltr/run_training.py <tracker_name> <settings_name>
```

## 📊 Kết quả

Kết quả tracking được lưu tại:
- **PyTracking results**: `pytracking/pytracking/tracking_results/`
- **OTB Toolkit results**: `otb-toolkit/results/OPE/`
- **Performance plots**: `otb100/plots/`

### Success plots

Các performance plots được tạo tự động sau khi chạy evaluation:
- Success plots (AUC metrics)
- Precision plots
- Comparison với state-of-the-art trackers

## 🔬 Cấu hình nâng cao

### Tùy chỉnh tracker parameters

Chỉnh sửa file trong `pytracking/pytracking/parameter/<tracker_name>/`:
```python
# Ví dụ: tomp50_custom.py
def parameters():
    params = TrackerParams()
    
    # Network settings
    params.net = NetWithBackbone(net_path='tomp50.pth.tar', use_gpu=True)
    
    # Search area settings
    params.search_area_scale = 5.0
    params.target_area_scale = 2.0
    
    # Score prediction settings
    params.score_threshold = 0.5
    
    return params
```

### Thêm tracker mới vào OTB Toolkit

Edit `otb-toolkit/configs/config_trackers.m`:
```matlab
trackers = {
    struct('name', 'MyTracker', 'path', '../trackers/MyTracker/')
};
```

## 📚 Tài liệu tham khảo

### Papers
- **ToMP** (CVPR 2022): [Transforming Model Prediction for Tracking](https://arxiv.org/abs/2203.11192)
- **DiMP** (ICCV 2019): [Learning Discriminative Model Prediction for Tracking](https://arxiv.org/abs/1904.07220)
- **ATOM** (CVPR 2019): [ATOM: Accurate Tracking by Overlap Maximization](https://arxiv.org/abs/1811.07628)

### Resources
- [PyTracking Documentation](pytracking/README.md)
- [Model Zoo](pytracking/MODEL_ZOO.md)
- [OTB Benchmark](http://cvlab.hanyang.ac.kr/tracker_benchmark/)

## 🐛 Troubleshooting

### Lỗi CUDA out of memory
```python
# Giảm batch size trong training settings
params.train_batch_size = 16  # Giảm xuống từ 32
```

### Lỗi import module
```bash
# Thêm pytracking vào PYTHONPATH
export PYTHONPATH=/path/to/TransTResearch/pytracking:$PYTHONPATH

# Windows PowerShell
$env:PYTHONPATH = "E:\Programming\C\C2P\Project\TransTResearch\pytracking;$env:PYTHONPATH"
```

### Lỗi visualize với Visdom
```bash
# Start Visdom server
python -m visdom.server

# Trong browser: http://localhost:8097
```

## 📝 License

Dự án này sử dụng các thành phần từ:
- PyTracking: MIT License
- OTB Toolkit: Academic use

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on the repository.

---

**Last updated**: March 2026
