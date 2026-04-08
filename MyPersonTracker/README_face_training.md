# Face Training

Muc tieu cua workspace nay:

- chon dataset cho face detection va face embedding
- fine-tune face embedding rieng cho bai toan tracking nguoi
- giu tach biet voi `.venv_person_pipeline`

## Dataset da chon

- Face detection cho MTCNN: `WIDER FACE` cho bbox face, va `CelebA` cho landmark
- Face embedding: `CASIA-WebFace`

Ly do:

- `WIDER FACE` la benchmark face detection phu hop cho scale, occlusion, clutter
- `CASIA-WebFace` co identity label, phu hop de fine-tune model embedding

## Luu y ve MTCNN

- `facenet-pytorch` chi cung cap `MTCNN` pretrained de infer, khong co train pipeline san
- Vi vay phan embedding da duoc dung script train rieng trong repo nay
- Neu muon train detector `MTCNN` dung nghia, huong dung la `WIDER FACE + CelebA landmarks` voi code training ngoai stack `facenet-pytorch`

## Tao venv rieng

```powershell
pwsh -File MyPersonTracker/setup_face_training_env.ps1
```

## Chuan bi subset CASIA can bang

```powershell
MyPersonTracker\.venv_face_training\Scripts\python.exe `
    MyPersonTracker\prepare_casia_embedding_subset.py `
    --max-identities 256 `
    --train-per-identity 20 `
    --val-per-identity 4
```

Output mac dinh:

- `MyPersonTracker/datasets/casia_webface_balanced/train`
- `MyPersonTracker/datasets/casia_webface_balanced/val`
- `MyPersonTracker/datasets/casia_webface_balanced/manifest.json`

## Fine-tune embedding

```powershell
MyPersonTracker\.venv_face_training\Scripts\python.exe `
    MyPersonTracker\train_face_embedding_classifier.py `
    --data-root MyPersonTracker\datasets\casia_webface_balanced `
    --pretrained casia-webface `
    --epochs 1 `
    --batch-size 64 `
    --device cuda:0
```

Checkpoint mac dinh:

- `MyPersonTracker/weights/face_embedding_casia_balanced.pt`

## Dung checkpoint trong pipeline tracking

```powershell
MyPersonTracker\.venv_person_pipeline\Scripts\python.exe `
    MyPersonTracker\run_person_face_pipeline.py `
    --source 0 `
    --display `
    --face-embedding-pretrained casia-webface `
    --face-embedding-weights MyPersonTracker\weights\face_embedding_casia_balanced.pt
```
