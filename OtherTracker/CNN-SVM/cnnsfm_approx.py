from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import random
import time
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import SGDClassifier
from torchvision.models import AlexNet_Weights, alexnet


def _clip_box(box: np.ndarray, image_shape: tuple[int, int, int]) -> np.ndarray:
    h, w = image_shape[:2]
    x, y, bw, bh = box.astype(np.float32)
    bw = float(np.clip(bw, 8.0, w))
    bh = float(np.clip(bh, 8.0, h))
    x = float(np.clip(x, 0.0, max(0.0, w - bw)))
    y = float(np.clip(y, 0.0, max(0.0, h - bh)))
    return np.array([x, y, bw, bh], dtype=np.float32)


def _bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return float(inter / max(union, 1e-6))


def _crop_and_resize(image: np.ndarray, box: np.ndarray, size: int) -> np.ndarray:
    x, y, w, h = box.astype(np.float32)
    x1 = int(math.floor(x))
    y1 = int(math.floor(y))
    x2 = int(math.ceil(x + w))
    y2 = int(math.ceil(y + h))

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - image.shape[1])
    pad_bottom = max(0, y2 - image.shape[0])

    if pad_left or pad_top or pad_right or pad_bottom:
        image = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_REPLICATE,
        )
    x1 += pad_left
    x2 += pad_left
    y1 += pad_top
    y2 += pad_top

    patch = image[y1:y2, x1:x2]
    if patch.size == 0:
        patch = np.zeros((size, size, 3), dtype=np.uint8)
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_LINEAR)


def _l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.clip(norms, 1e-6, None)


@dataclass
class TrackerConfig:
    input_size: int = 224
    template_size: int = 64
    num_candidates: int = 64
    saliency_topk: int = 3
    template_history: int = 30
    saliency_weight: float = 0.35
    translation_std_factor: float = 0.35
    scale_std: float = 0.04
    train_pos_iou: float = 0.7
    train_neg_iou: float = 0.3
    random_seed: int = 1337


class AlexNetFC6Extractor:
    def __init__(self, device: torch.device):
        self.device = device
        weights = AlexNet_Weights.IMAGENET1K_V1
        model = alexnet(weights=weights).to(device).eval()
        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = nn.Sequential(*model.classifier[:3])
        self.mean = torch.tensor(weights.transforms().mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(weights.transforms().std, device=device).view(1, 3, 1, 1)

    def _preprocess(self, patches: Iterable[np.ndarray]) -> torch.Tensor:
        arrays = []
        for patch in patches:
            rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            arrays.append(rgb.astype(np.float32) / 255.0)
        tensor = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).to(self.device)
        return (tensor - self.mean) / self.std

    def encode(self, patches: list[np.ndarray]) -> np.ndarray:
        if not patches:
            return np.empty((0, 4096), dtype=np.float32)
        with torch.inference_mode():
            x = self._preprocess(patches)
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
        return x.detach().cpu().numpy().astype(np.float32)

    def saliency(self, patch: np.ndarray, positive_weights: np.ndarray) -> np.ndarray:
        x = self._preprocess([patch]).requires_grad_(True)
        for module in (self.features, self.avgpool, self.classifier):
            module.zero_grad(set_to_none=True)
        features = self.features(x)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        features = self.classifier(features)

        weight_tensor = torch.from_numpy(positive_weights.astype(np.float32)).to(self.device)
        score = (features[0] * weight_tensor).sum()
        if torch.isclose(score, torch.tensor(0.0, device=self.device)):
            score = features[0].sum()
        score.backward()
        grad = x.grad[0].detach().abs().amax(dim=0)
        saliency = grad.cpu().numpy()
        saliency -= saliency.min()
        saliency /= max(float(saliency.max()), 1e-6)
        return saliency.astype(np.float32)


class CNNSVMApproxTracker:
    def __init__(self, device: str | None = None, config: TrackerConfig | None = None):
        self.config = config or TrackerConfig()
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.extractor = AlexNetFC6Extractor(self.device)
        self.classifier = SGDClassifier(
            loss="hinge",
            alpha=1e-4,
            learning_rate="optimal",
            average=True,
            fit_intercept=True,
            random_state=self.config.random_seed,
        )
        self.template_history: deque[np.ndarray] = deque(maxlen=self.config.template_history)
        self.template: np.ndarray | None = None
        self.initialized = False

    def _sample_candidates(self, box: np.ndarray, image_shape: tuple[int, int, int]) -> list[np.ndarray]:
        x, y, w, h = box.astype(np.float32)
        cx = x + w / 2.0
        cy = y + h / 2.0
        sigma = self.config.translation_std_factor * math.sqrt(max(w * h, 1.0))

        candidates = [_clip_box(box.copy(), image_shape)]
        for _ in range(self.config.num_candidates - 1):
            dx = np.random.normal(0.0, sigma)
            dy = np.random.normal(0.0, sigma)
            scale = float(np.clip(np.random.lognormal(mean=0.0, sigma=self.config.scale_std), 0.85, 1.15))
            nw = w * scale
            nh = h * scale
            candidate = np.array(
                [cx + dx - nw / 2.0, cy + dy - nh / 2.0, nw, nh],
                dtype=np.float32,
            )
            candidates.append(_clip_box(candidate, image_shape))
        return candidates

    def _boxes_to_patches(self, image: np.ndarray, boxes: list[np.ndarray]) -> list[np.ndarray]:
        return [_crop_and_resize(image, box, self.config.input_size) for box in boxes]

    def _train_from_boxes(self, image: np.ndarray, target_box: np.ndarray, candidate_boxes: list[np.ndarray] | None = None) -> None:
        boxes: list[np.ndarray] = [target_box.copy()]
        labels: list[int] = [1]

        if candidate_boxes is None:
            candidate_boxes = self._sample_candidates(target_box, image.shape)

        positives = 0
        negatives = 0
        for candidate in candidate_boxes:
            iou = _bbox_iou(candidate, target_box)
            if iou >= self.config.train_pos_iou and positives < 8:
                boxes.append(candidate)
                labels.append(1)
                positives += 1
            elif iou <= self.config.train_neg_iou and negatives < 32:
                boxes.append(candidate)
                labels.append(-1)
                negatives += 1

        patches = self._boxes_to_patches(image, boxes)
        features = _l2_normalize(self.extractor.encode(patches))
        self.classifier.partial_fit(features, np.asarray(labels, dtype=np.int32), classes=np.asarray([-1, 1], dtype=np.int32))

    def _positive_weights(self) -> np.ndarray:
        coef = getattr(self.classifier, "coef_", None)
        if coef is None:
            return np.ones((4096,), dtype=np.float32)
        weights = coef[0].astype(np.float32)
        weights = np.where(weights > 0.0, weights, 0.0)
        if not np.any(weights > 0):
            weights = np.abs(coef[0].astype(np.float32))
        return weights

    def _match_template(self, saliency_patch: np.ndarray) -> float:
        if self.template is None:
            return 0.0
        a = saliency_patch.reshape(-1)
        b = self.template.reshape(-1)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 1e-6:
            return 0.0
        return float(np.dot(a, b) / denom)

    def initialize(self, image: np.ndarray, init_box: tuple[float, float, float, float]) -> None:
        box = _clip_box(np.asarray(init_box, dtype=np.float32), image.shape)
        init_candidates = self._sample_candidates(box, image.shape)
        self._train_from_boxes(image, box, init_candidates)

        init_patch = _crop_and_resize(image, box, self.config.input_size)
        init_saliency = self.extractor.saliency(init_patch, self._positive_weights())
        init_saliency = cv2.resize(init_saliency, (self.config.template_size, self.config.template_size))
        self.template_history.append(init_saliency)
        self.template = init_saliency.copy()
        self.initialized = True

    def update(self, image: np.ndarray, prev_box: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        if not self.initialized:
            raise RuntimeError("Tracker must be initialized before update().")

        t0 = time.perf_counter()
        candidates = self._sample_candidates(prev_box, image.shape)
        patches = self._boxes_to_patches(image, candidates)
        features = _l2_normalize(self.extractor.encode(patches))
        svm_scores = self.classifier.decision_function(features).astype(np.float32)

        positive_indices = np.flatnonzero(svm_scores > 0)
        if positive_indices.size == 0:
            positive_indices = np.argsort(svm_scores)[-self.config.saliency_topk :]
        top_indices = positive_indices[np.argsort(svm_scores[positive_indices])[-self.config.saliency_topk :]]

        positive_weights = self._positive_weights()
        saliency_scores = np.zeros((len(candidates),), dtype=np.float32)
        saliency_cache: dict[int, np.ndarray] = {}
        for idx in top_indices.tolist():
            saliency = self.extractor.saliency(patches[idx], positive_weights)
            saliency_small = cv2.resize(saliency, (self.config.template_size, self.config.template_size))
            saliency_cache[idx] = saliency_small
            saliency_scores[idx] = self._match_template(saliency_small)

        svm_norm = svm_scores - float(svm_scores.mean())
        svm_scale = float(np.std(svm_norm))
        if svm_scale > 1e-6:
            svm_norm /= svm_scale

        combined = svm_norm + self.config.saliency_weight * saliency_scores
        best_idx = int(np.argmax(combined))
        best_box = candidates[best_idx]

        best_saliency = saliency_cache.get(best_idx)
        if best_saliency is None:
            saliency = self.extractor.saliency(_crop_and_resize(image, best_box, self.config.input_size), positive_weights)
            best_saliency = cv2.resize(saliency, (self.config.template_size, self.config.template_size))

        self.template_history.append(best_saliency)
        self.template = np.mean(np.stack(list(self.template_history), axis=0), axis=0).astype(np.float32)
        self._train_from_boxes(image, best_box, candidates)

        return best_box, {
            "total": time.perf_counter() - t0,
            "svm_max": float(svm_scores.max()),
            "combined_max": float(combined.max()),
        }


def track_sequence(
    image_paths: list[str],
    init_box: tuple[float, float, float, float],
    device: str | None = None,
    log_prefix: str = "[CNN-SVM-Approx]",
) -> tuple[list[tuple[float, float, float, float]], list[float]]:
    tracker = CNNSVMApproxTracker(device=device)
    boxes: list[tuple[float, float, float, float]] = []
    times: list[float] = []

    first = cv2.imread(image_paths[0])
    if first is None:
        raise FileNotFoundError(image_paths[0])
    tracker.initialize(first, init_box)
    boxes.append(tuple(float(v) for v in init_box))
    times.append(0.0)

    start = time.perf_counter()
    last_log = start
    print(f"{log_prefix} start frames={len(image_paths)} device={tracker.device}")

    prev_box = np.asarray(init_box, dtype=np.float32)
    for frame_idx, image_path in enumerate(image_paths[1:], start=2):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(image_path)
        prev_box, metrics = tracker.update(image, prev_box)
        boxes.append(tuple(float(v) for v in prev_box))
        times.append(float(metrics["total"]))

        now = time.perf_counter()
        if frame_idx == 2 or frame_idx == len(image_paths) or frame_idx % 25 == 0 or (now - last_log) >= 30.0:
            elapsed = now - start
            avg_fps = max((frame_idx - 1) / max(sum(times[1:frame_idx]), 1e-6), 1e-6)
            eta = (len(image_paths) - frame_idx) / avg_fps
            print(
                f"{log_prefix} frame {frame_idx}/{len(image_paths)} "
                f"({100.0 * frame_idx / len(image_paths):.1f}%) "
                f"elapsed {elapsed:.1f}s avg_fps {avg_fps:.3f} eta {eta:.1f}s"
            )
            last_log = now

    total_time = sum(times[1:])
    fps = (len(image_paths) - 1) / max(total_time, 1e-6)
    print(f"{log_prefix} done total_time={total_time:.1f}s fps={fps:.4f}")
    return boxes, times
