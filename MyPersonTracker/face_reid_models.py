from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN


BBoxXYXY = Tuple[int, int, int, int]


def _clip_xyxy(box_xyxy: Sequence[float], image_shape: Tuple[int, int]) -> BBoxXYXY:
    image_h, image_w = image_shape
    x1, y1, x2, y2 = [int(round(float(v))) for v in box_xyxy]
    x1 = max(0, min(x1, image_w - 1))
    y1 = max(0, min(y1, image_h - 1))
    x2 = max(x1 + 1, min(x2, image_w))
    y2 = max(y1 + 1, min(y2, image_h))
    return x1, y1, x2, y2


@dataclass
class DetectedFace:
    bbox_xyxy: BBoxXYXY
    confidence: float
    aligned_tensor: torch.Tensor


class FaceDetectorModel:
    def __init__(
        self,
        device: torch.device,
        image_size: int = 160,
        margin: int = 14,
        min_face_size: int = 24,
        confidence_threshold: float = 0.90,
    ):
        self.confidence_threshold = confidence_threshold
        self.model = MTCNN(
            image_size=image_size,
            margin=margin,
            min_face_size=min_face_size,
            post_process=True,
            keep_all=True,
            device=device,
        )

    def detect(self, frame_rgb: np.ndarray) -> List[DetectedFace]:
        image_pil = Image.fromarray(frame_rgb)
        return self._detect_from_image(image_pil, frame_rgb.shape[:2])

    def detect_image(self, image_path: Path) -> List[DetectedFace]:
        image_pil = Image.open(image_path).convert("RGB")
        image_w, image_h = image_pil.size
        return self._detect_from_image(image_pil, (image_h, image_w))

    def _detect_from_image(self, image_pil: Image.Image, image_shape: Tuple[int, int]) -> List[DetectedFace]:
        face_boxes, face_probs = self.model.detect(image_pil)
        if face_boxes is None or face_probs is None:
            return []

        face_tensors, _ = self.model(image_pil, return_prob=True)
        if face_tensors is None:
            return []
        if face_tensors.ndim == 3:
            face_tensors = face_tensors.unsqueeze(0)

        detections: List[DetectedFace] = []
        for face_box, face_prob, face_tensor in zip(face_boxes, face_probs, face_tensors):
            if face_prob is None or float(face_prob) < self.confidence_threshold:
                continue

            clipped_box = _clip_xyxy(face_box.tolist(), image_shape)
            detections.append(
                DetectedFace(
                    bbox_xyxy=clipped_box,
                    confidence=float(face_prob),
                    aligned_tensor=face_tensor.detach().cpu(),
                )
            )
        return detections


class FaceEmbeddingModel:
    def __init__(
        self,
        device: torch.device,
        pretrained: str = "vggface2",
        checkpoint_path: Optional[Path] = None,
    ):
        self.device = device
        self.model = InceptionResnetV1(pretrained=pretrained).eval().to(device)
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint.get("embedding_state_dict", checkpoint.get("state_dict", checkpoint))
            model_state_dict = self.model.state_dict()
            filtered_state_dict = {
                key: value
                for key, value in state_dict.items()
                if key in model_state_dict and tuple(value.shape) == tuple(model_state_dict[key].shape)
            }
            state_dict = filtered_state_dict
            self.model.load_state_dict(state_dict, strict=False)

    def encode_faces(self, face_tensors: Sequence[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if isinstance(face_tensors, torch.Tensor):
            batch = face_tensors
        else:
            tensors = list(face_tensors)
            if not tensors:
                raise ValueError("FaceEmbeddingModel.encode_faces() received an empty batch.")
            batch = torch.stack(tensors, dim=0)

        with torch.inference_mode():
            embeddings = self.model(batch.to(self.device))
            embeddings = F.normalize(embeddings, dim=1).detach().cpu()
        return embeddings

    def encode_best_face_image(self, image_path: Path, face_detector: FaceDetectorModel) -> torch.Tensor:
        detections = face_detector.detect_image(image_path)
        if not detections:
            raise RuntimeError(f"No face found in registration image: {image_path}")

        best_detection = max(detections, key=lambda detection: detection.confidence)
        return self.encode_faces(best_detection.aligned_tensor.unsqueeze(0))[0]
