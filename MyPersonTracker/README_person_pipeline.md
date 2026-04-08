# MyPerson Face Pipeline

Pipeline moi cho bai toan tracking nguoi:

- `YOLO` detect person
- `MTCNN` detect khuon mat
- `InceptionResnetV1` tao embedding khuon mat va so khop cosine similarity
- `MyECO` ban copy nguyen goc tu `MyECOTracker`, dung param `verified_otb936`

## Face model va khoa muc tieu

- Face detector va face embedder da duoc tach rieng trong `MyPersonTracker/face_reid_models.py`
- `MTCNN` chi tim khuon mat trong frame
- `InceptionResnetV1 (vggface2)` bien moi khuon mat thanh vector embedding
- Pipeline se so khuon mat dang ky voi cac khuon mat detect duoc, chon khuon mat giong nhat
- Bounding box `person` cua `YOLO` chi duoc giu lai neu tam khuon mat nam trong bbox do
- Bbox nguoi nao khop mat dang ky thi duoc dung de `init/reinit MyECO`
- Co the doi sang checkpoint embedding fine-tuned bang `--face-embedding-pretrained` va `--face-embedding-weights`

## YOLO hien tai

- Hien tai pipeline da chi lay `person` o infer, vi script goi YOLO voi `classes=[0]`.
- Nghia la ngay ca khi dung `yolov8n.pt` goc COCO, no van chi detect nguoi trong pipeline nay.
- Neu muon tot hon cho bai toan cua anh, da co them script fine-tune thanh detector person-only.

## Workspace tach biet

- Nguon `MyECO` copy sang: `MyPersonTracker/myeco_otb936`
- Venv rieng: `MyPersonTracker/.venv_person_pipeline`
- `venv312` khong can dung cho pipeline nay

## Setup

Chay:

```powershell
pwsh -File MyPersonTracker/setup_person_pipeline_env.ps1
```

## Chay webcam va dang ky khuon mat truc tiep

```powershell
MyPersonTracker\.venv_person_pipeline\Scripts\python.exe `
    MyPersonTracker\run_person_face_pipeline.py `
    --source 0 `
    --display
```

Trong luc registration:

- Nhan `1-9` de chon candidate theo so dang hien tren man hinh
- Nhan `r` de lay candidate co face confidence cao nhat
- Nhan `q` de thoat

Sau khi dang ky xong, pipeline se:

1. Khoi tao `MyECO verified_otb936`
2. Theo doi nguoi da dang ky
3. Dinh ky chay lai `YOLO + face match` de reacquire khi drift

## Chay video va dung anh mat mau

```powershell
MyPersonTracker\.venv_person_pipeline\Scripts\python.exe `
    MyPersonTracker\run_person_face_pipeline.py `
    --source path\\to\\video.mp4 `
    --register-face-image path\\to\\target_face.jpg `
    --display `
    --face-embedding-pretrained casia-webface `
    --face-embedding-weights MyPersonTracker\\weights\\face_embedding_casia_balanced_256x20x4.pt `
    --output MyPersonTracker\\outputs\\person_track.mp4
```

## Ghi chu

- Tracker dang dung source copy `MyECO` goc, khong dung nhanh `pytracking` da sua trong `MyPersonTracker`.
- `YOLO` mac dinh la `yolov8n.pt`; neu can model khac thi truyen qua `--yolo-model`.
- Neu co file `MyPersonTracker/weights/yolo_person_only.pt` thi pipeline se uu tien dung weight nay.
- Neu tracker mat muc tieu, pipeline se uu tien tim lai bang `face ReID` roi moi khoi tao lai `MyECO`.

## Fine-tune YOLO thanh person-only

Template dataset yaml:

- [person_only_dataset_template.yaml](e:/Programming/Python/Project/AI/CV/Tracking/TransTResearch/MyPersonTracker/person_only_dataset_template.yaml)

Train:

```powershell
MyPersonTracker\.venv_person_pipeline\Scripts\python.exe `
    MyPersonTracker\train_yolo_person_only.py `
    --data MyPersonTracker\person_only_dataset_template.yaml `
    --model yolov8n.pt `
    --epochs 50 `
    --imgsz 960 `
    --batch 16 `
    --device 0
```

Sau khi train xong, script se copy `best.pt` thanh:

```text
MyPersonTracker\weights\yolo_person_only.pt
```

Luc do pipeline se tu uu tien detector fine-tuned nay.
