# MyTracker (ECO) Runner Scripts

Scripts và hướng dẫn để chạy MyTracker (ECO Tracker) trên Jetson Nano và máy local.

## 📋 Mô tả

**MyTracker** là tên gọi của **ECO Tracker** được tối ưu hóa với parameter set `verified_otb936`.

- **Tracker Type**: ECO (Efficient Convolution Operators)
- **Parameter Set**: verified_otb936 (run 936)
- **Supported Datasets**: LaSOT, OTB, OTB-100
- **Framework**: PyTracking

## 🚀 Cách sử dụng

### 1. Chạy trên Local Machine (Windows/Linux/Mac)

**Yêu cầu**:
- Python 3.7+
- PyTorch với CUDA hoặc CPU
- OpenCV, matplotlib, pandas, tqdm, scikit-image, visdom
- Pre-trained networks (ECO)

**Chạy trực tiếp**:

```bash
# Chạy trên LaSOT (head 20 + tail 20 sequences) - nhanh nhất
python run_mytracker.py

# Chạy trên LaSOT (all sequences)
python run_mytracker.py --dataset lasot

# Chạy trên LaSOT (first 20 sequences)
python run_mytracker.py --dataset lasot_first20

# Chạy trên OTB dataset
python run_mytracker.py --dataset otb

# Với options khác
python run_mytracker.py --dataset lasot --debug 1 --threads 4
```

**Options**:
- `--dataset`: Dataset để chạy (mặc định: lasot_headtail40)
- `--debug`: Debug level 0-2 (mặc định: 0)
- `--threads`: Số threads (mặc định: 0 = auto)

### 2. Chạy trên Jetson Nano (via SSH)

#### Windows PowerShell

```powershell
# Chạy với cài đặt mặc định
.\run_mytracker_jetson.ps1

# Với custom settings
.\run_mytracker_jetson.ps1 -JetsonHost "helios@192.168.1.162" -JetsonPort 22
```

#### Linux/Mac Bash

```bash
# Chạy với cài đặt mặc định
bash run_mytracker_jetson.sh

# Với custom settings
bash run_mytracker_jetson.sh "helios@192.168.1.162" 22
```

### 3. Chạy từ PowerShell (Windows)

```powershell
# Activate Python environment
.\venv312\Scripts\Activate.ps1

# Chạy MyTracker
python run_mytracker.py

# Hoặc với dataset khác
python run_mytracker.py --dataset lasot --debug 1
```

## 📊 Output

Results được lưu tại:
```
MyECOTracker/lasot/result/           # LaSOT results
MyECOTracker/otb100result/           # OTB results
```

Mỗi sequence sẽ tạo ra:
- `{sequence_name}.txt` - Bounding boxes
- `{sequence_name}_time.txt` - Timing info
- `{sequence_name}_object_presence_scores.txt` - Confidence scores

## 🔧 Tùy chỉnh

### Chỉnh sửa ECO Parameters

File: `MyECOTracker/pytracking/pytracking/parameter/eco/verified_otb936.py`

### Chỉnh sửa Experiment Settings

File: `MyECOTracker/pytracking/pytracking/experiments/myexperiments.py`

Thêm experiment mới:
```python
def eco_custom_experiment():
    trackers = [Tracker('eco', 'verified_otb936', 936, 'MyTrackerCustom')]
    dataset = get_dataset('lasot')  # hoặc 'otb', etc.
    return trackers, dataset
```

Sau đó chạy:
```bash
python run_mytracker.py --experiment eco_custom_experiment
```

## ⚙️ Jetson Nano Setup

Nếu chưa cài đặt environment trên Jetson, hãy cài đặt:

```bash
# 1. Create virtual environment
python3 -m venv ~/TransTResearch/venv

# 2. Activate it
source ~/TransTResearch/venv/bin/activate

# 3. Install PyTorch for Jetson (important!)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Install other dependencies
pip install opencv-python matplotlib pandas tqdm scikit-image visdom tensorboard
pip install pycocotools lvis jpeg4py gdown

# 5. Download pre-trained ECO network (if not already present)
cd ~/TransTResearch/MyECOTracker/pytracking
mkdir -p pretrained_network
gdown https://drive.google.com/uc?id=1aWC4waLv_te-BULoy0k-n_zS-ONms21S -O pretrained_network/resnet18_vggmconv1.pth
```

## 📝 Ghi chú

- Các script này **không xóa hay sửa** mã nguồn gốc
- Tất cả kết quả được lưu trong folder kết quả riêng
- Jetson Nano version có thể chạy chậm hơn tùy vào phần cứng
- Để tăng tốc, có thể chạy multi-threaded hoặc các subset của dataset

## 🐛 Troubleshooting

### CUDA/GPU issues
```bash
# Kiểm tra PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Nếu không detect GPU, cài PyTorch CPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Memory issues
- Giảm số threads: `python run_mytracker.py --threads 1`
- Chạy trên subset: `python run_mytracker.py --dataset lasot_first20`

### SSH connection issues
- Đảm bảo SSH key authentication: `ssh-keygen -t rsa`
- Thêm public key vào Jetson: `cat ~/.ssh/id_rsa.pub | ssh user@host 'cat >> .ssh/authorized_keys'`

## 📚 Tài liệu tham khảo

- PyTracking: https://github.com/visionml/pytracking
- ECO Paper: http://openaccess.thecvf.com/ICCV2017
- LaSOT Dataset: http://vision.cs.stonybrook.edu/~lasot/
- OTB Benchmark: http://cvlab.hanyang.ac.kr/tracker_benchmark/

---
Created: 2026-04-02
Updated: Script for MyTracker ECO Tracker
