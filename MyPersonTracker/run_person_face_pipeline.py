import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from face_reid_models import FaceDetectorModel, FaceEmbeddingModel


REPO_ROOT = Path(__file__).resolve().parents[1]
PERSON_ROOT = Path(__file__).resolve().parent
MYECO_WORKSPACE = PERSON_ROOT / "myeco_otb936"
WEIGHTS_ROOT = PERSON_ROOT / "weights"


def default_person_yolo_model() -> str:
    preferred = WEIGHTS_ROOT / "yolo_person_only.pt"
    if preferred.is_file():
        return str(preferred)
    return "yolov8n.pt"


def _bootstrap_myeco_workspace() -> None:
    workspace_str = str(MYECO_WORKSPACE)
    if workspace_str not in sys.path:
        sys.path.insert(0, workspace_str)

    pytracking_pkg_root = MYECO_WORKSPACE / "pytracking"
    os.environ.setdefault("PERSON_MYECO_NETWORK_PATH", str(MYECO_WORKSPACE / "pretrained_network"))
    os.environ.setdefault("PERSON_MYECO_RESULTS_PATH", str(pytracking_pkg_root / "tracking_results"))
    os.environ.setdefault("PERSON_MYECO_RESULT_PLOT_PATH", str(pytracking_pkg_root / "result_plots"))
    os.environ.setdefault("PERSON_MYECO_SEGMENTATION_PATH", str(pytracking_pkg_root / "segmentation_results"))
    os.environ.setdefault("PERSON_MYECO_OTB_PATH", str(REPO_ROOT / "otb" / "otb100"))
    os.environ.setdefault("PERSON_MYECO_LASOT_PATH", str(REPO_ROOT / "ls" / "lasot"))


_bootstrap_myeco_workspace()

from pytracking.evaluation import Tracker as PyTrackingTracker  # noqa: E402


Color = Tuple[int, int, int]


@dataclass
class FaceCandidate:
    person_bbox_xywh: Tuple[int, int, int, int]
    person_bbox_xyxy: Tuple[int, int, int, int]
    face_bbox_xyxy: Tuple[int, int, int, int]
    embedding: torch.Tensor
    person_confidence: float
    face_confidence: float
    similarity: Optional[float] = None


def parse_source(source: str) -> int | str:
    if source.isdigit():
        return int(source)
    return source


def xyxy_to_xywh(box_xyxy: Sequence[float]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(float(v))) for v in box_xyxy]
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def clip_xyxy(box_xyxy: Sequence[float], image_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    h, w = image_shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in box_xyxy]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return x1, y1, x2, y2


def bbox_iou(box_a_xywh: Sequence[float], box_b_xywh: Sequence[float]) -> float:
    ax1, ay1, aw, ah = box_a_xywh
    bx1, by1, bw, bh = box_b_xywh
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return float(inter_area / denom)


def cosine_similarity(embedding_a: torch.Tensor, embedding_b: torch.Tensor) -> float:
    return float(torch.sum(embedding_a * embedding_b).item())


def draw_text_block(frame: np.ndarray, lines: Sequence[str], origin: Tuple[int, int]) -> None:
    x, y = origin
    line_height = 24
    width = max(220, max((len(line) for line in lines), default=0) * 9 + 20)
    height = line_height * len(lines) + 10
    cv2.rectangle(frame, (x, y), (x + width, y + height), (32, 32, 32), -1)
    for idx, line in enumerate(lines):
        baseline_y = y + 24 + idx * line_height
        cv2.putText(frame, line, (x + 10, baseline_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2)


class MyECOAdapter:
    def __init__(self, tracker_name: str = "eco", parameter_name: str = "verified_otb936", debug: int = 0):
        wrapper = PyTrackingTracker(tracker_name, parameter_name)
        params = wrapper.get_parameters()
        params.debug = debug
        params.visualization = False
        self.tracker = wrapper.create_tracker(params)
        if hasattr(self.tracker, "initialize_features"):
            self.tracker.initialize_features()
        self.initialized = False

    def initialize(self, frame_rgb: np.ndarray, bbox_xywh: Sequence[float]) -> None:
        self.tracker.initialize(frame_rgb, {"init_bbox": list(map(float, bbox_xywh))})
        self.initialized = True

    def track(self, frame_rgb: np.ndarray) -> Tuple[Tuple[float, float, float, float], float]:
        out = self.tracker.track(frame_rgb, {})
        bbox = tuple(float(v) for v in out["target_bbox"])
        score = float(getattr(self.tracker, "last_max_score", 0.0))
        return bbox, score


class FaceReIDFinder:
    def __init__(
        self,
        yolo_model: str,
        person_confidence: float,
        face_confidence: float,
        yolo_imgsz: int,
        device: torch.device,
        face_embedding_pretrained: str,
        face_embedding_weights: str,
    ):
        self.device = device
        self.yolo_device = "0" if device.type == "cuda" else "cpu"
        self.person_confidence = person_confidence
        self.face_confidence = face_confidence
        self.yolo_imgsz = yolo_imgsz
        self.person_detector = YOLO(yolo_model)
        self.face_detector = FaceDetectorModel(device=device, confidence_threshold=face_confidence)
        checkpoint_path = Path(face_embedding_weights) if face_embedding_weights else None
        self.face_embedder = FaceEmbeddingModel(
            device=device,
            pretrained=face_embedding_pretrained,
            checkpoint_path=checkpoint_path,
        )

    def _detect_people(self, frame_bgr: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        result = self.person_detector.predict(
            source=frame_bgr,
            verbose=False,
            conf=self.person_confidence,
            classes=[0],
            imgsz=self.yolo_imgsz,
            device=self.yolo_device,
        )[0]
        people: List[Tuple[Tuple[int, int, int, int], float]] = []
        if result.boxes is None:
            return people

        boxes_xyxy = result.boxes.xyxy.detach().cpu().tolist()
        scores = result.boxes.conf.detach().cpu().tolist()
        for box_xyxy, score in zip(boxes_xyxy, scores):
            clipped = clip_xyxy(box_xyxy, frame_bgr.shape)
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                continue
            people.append((clipped, float(score)))
        return people

    def detect_candidates(self, frame_bgr: np.ndarray, frame_rgb: np.ndarray) -> List[FaceCandidate]:
        people = self._detect_people(frame_bgr)
        if not people:
            return []

        detected_faces = self.face_detector.detect(frame_rgb)
        if not detected_faces:
            return []

        embeddings = self.face_embedder.encode_faces([face.aligned_tensor for face in detected_faces])

        best_by_person: dict[int, FaceCandidate] = {}
        for detected_face, embedding in zip(detected_faces, embeddings):
            fx1, fy1, fx2, fy2 = detected_face.bbox_xyxy
            face_center_x = 0.5 * (fx1 + fx2)
            face_center_y = 0.5 * (fy1 + fy2)

            matched_person_idx: Optional[int] = None
            for person_idx, (person_box, _) in enumerate(people):
                px1, py1, px2, py2 = person_box
                if px1 <= face_center_x <= px2 and py1 <= face_center_y <= py2:
                    matched_person_idx = person_idx
                    break
            if matched_person_idx is None:
                continue

            person_box, person_score = people[matched_person_idx]
            candidate = FaceCandidate(
                person_bbox_xywh=xyxy_to_xywh(person_box),
                person_bbox_xyxy=person_box,
                face_bbox_xyxy=(fx1, fy1, fx2, fy2),
                embedding=embedding,
                person_confidence=float(person_score),
                face_confidence=detected_face.confidence,
            )

            previous = best_by_person.get(matched_person_idx)
            if previous is None or candidate.face_confidence > previous.face_confidence:
                best_by_person[matched_person_idx] = candidate

        return list(best_by_person.values())

    def encode_face_image(self, image_path: Path) -> torch.Tensor:
        return self.face_embedder.encode_best_face_image(image_path, self.face_detector)

    def best_match(
        self,
        candidates: Sequence[FaceCandidate],
        registered_embedding: torch.Tensor,
        similarity_threshold: float,
    ) -> Optional[FaceCandidate]:
        best: Optional[FaceCandidate] = None
        for candidate in candidates:
            candidate.similarity = cosine_similarity(candidate.embedding, registered_embedding)
            if candidate.similarity < similarity_threshold:
                continue
            if best is None or candidate.similarity > (best.similarity or -1.0):
                best = candidate
        return best


class PersonFaceTrackingPipeline:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
        self.finder = FaceReIDFinder(
            yolo_model=args.yolo_model,
            person_confidence=args.person_confidence,
            face_confidence=args.face_confidence,
            yolo_imgsz=args.yolo_imgsz,
            device=self.device,
            face_embedding_pretrained=args.face_embedding_pretrained,
            face_embedding_weights=args.face_embedding_weights,
        )
        self.tracker = MyECOAdapter(parameter_name=args.tracker_param, debug=args.debug)
        self.registered_embedding: Optional[torch.Tensor] = None
        self.registered = False
        self.tracker_score = 0.0
        self.frame_index = 0
        self.last_bbox_xywh: Optional[Tuple[float, float, float, float]] = None
        self.lost_counter = 0

    def _open_source(self) -> cv2.VideoCapture:
        source = parse_source(self.args.source)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open source: {self.args.source}")
        return cap

    def _build_writer(self, cap: cv2.VideoCapture) -> Optional[cv2.VideoWriter]:
        if not self.args.output:
            return None

        output_path = Path(self.args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    def _register_from_image(self) -> None:
        if not self.args.register_face_image:
            return
        self.registered_embedding = self.finder.encode_face_image(Path(self.args.register_face_image))
        self.registered = True

    def _register_from_candidates(
        self,
        key: int,
        candidates: Sequence[FaceCandidate],
    ) -> Optional[FaceCandidate]:
        if not candidates:
            return None

        if key == ord("r") or (self.args.auto_register and len(candidates) == 1):
            return max(candidates, key=lambda c: c.face_confidence)

        if ord("1") <= key <= ord("9"):
            candidate_idx = key - ord("1")
            if candidate_idx < len(candidates):
                return candidates[candidate_idx]
        return None

    def _run_detection(self, frame_bgr: np.ndarray, frame_rgb: np.ndarray) -> List[FaceCandidate]:
        return self.finder.detect_candidates(frame_bgr, frame_rgb)

    def _maybe_reacquire(self, frame_bgr: np.ndarray, frame_rgb: np.ndarray) -> Optional[FaceCandidate]:
        if self.registered_embedding is None:
            return None

        should_detect = (
            not self.tracker.initialized
            or self.frame_index % self.args.detect_interval == 0
            or self.tracker_score < self.args.reacquire_score_threshold
            or self.lost_counter >= self.args.max_lost_frames
        )
        if not should_detect:
            return None

        candidates = self._run_detection(frame_bgr, frame_rgb)
        return self.finder.best_match(candidates, self.registered_embedding, self.args.reid_threshold)

    def _draw_candidates(self, frame: np.ndarray, candidates: Sequence[FaceCandidate]) -> None:
        for idx, candidate in enumerate(candidates[:9], start=1):
            px1, py1, px2, py2 = candidate.person_bbox_xyxy
            fx1, fy1, fx2, fy2 = candidate.face_bbox_xyxy
            cv2.rectangle(frame, (px1, py1), (px2, py2), (90, 170, 255), 2)
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 255), 2)
            label = f"{idx}: person {candidate.person_confidence:.2f} face {candidate.face_confidence:.2f}"
            cv2.putText(frame, label, (px1, max(24, py1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 170, 255), 2)

    def _draw_tracking(self, frame: np.ndarray, candidate: Optional[FaceCandidate]) -> None:
        if self.last_bbox_xywh is not None:
            x, y, w, h = [int(round(v)) for v in self.last_bbox_xywh]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 3)
            score_label = f"MyECO score={self.tracker_score:.3f}"
            cv2.putText(frame, score_label, (x, max(24, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2)

        if candidate is not None:
            px1, py1, px2, py2 = candidate.person_bbox_xyxy
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 140, 255), 2)
            if candidate.similarity is not None:
                label = f"reid={candidate.similarity:.3f}"
                cv2.putText(frame, label, (px1, py2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)

    def run(self) -> None:
        self._register_from_image()
        cap = self._open_source()
        writer = self._build_writer(cap)

        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    break

                self.frame_index += 1
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                display_frame = frame_bgr.copy()

                matched_candidate: Optional[FaceCandidate] = None
                registration_candidates: List[FaceCandidate] = []

                if not self.registered:
                    registration_candidates = self._run_detection(frame_bgr, frame_rgb)
                    self._draw_candidates(display_frame, registration_candidates)

                    lines = [
                        "Registration mode",
                        "Press 1-9 to register a face/person",
                        "Press r to register the strongest visible face",
                        "Press q to quit",
                    ]
                    draw_text_block(display_frame, lines, (12, 12))
                else:
                    if self.tracker.initialized:
                        tracked_bbox, self.tracker_score = self.tracker.track(frame_rgb)
                        self.last_bbox_xywh = tracked_bbox
                        if self.tracker_score < self.args.reacquire_score_threshold:
                            self.lost_counter += 1
                        else:
                            self.lost_counter = 0

                    matched_candidate = self._maybe_reacquire(frame_bgr, frame_rgb)
                    if matched_candidate is not None:
                        reinit_required = (
                            not self.tracker.initialized
                            or self.last_bbox_xywh is None
                            or self.tracker_score < self.args.reacquire_score_threshold
                            or bbox_iou(self.last_bbox_xywh, matched_candidate.person_bbox_xywh) < self.args.reacquire_iou_threshold
                        )
                        if reinit_required:
                            self.tracker.initialize(frame_rgb, matched_candidate.person_bbox_xywh)
                            self.last_bbox_xywh = matched_candidate.person_bbox_xywh
                            self.tracker_score = 1.0
                            self.lost_counter = 0

                    self._draw_tracking(display_frame, matched_candidate)

                    lines = [
                        "Tracking mode",
                        f"MyECO param: {self.args.tracker_param}",
                        f"Tracker score: {self.tracker_score:.3f}",
                        f"Face ReID threshold: {self.args.reid_threshold:.2f}",
                        "Press q to quit",
                    ]
                    draw_text_block(display_frame, lines, (12, 12))

                if self.args.display:
                    cv2.imshow("MyPerson Face Pipeline", display_frame)
                    key = cv2.waitKey(1) & 0xFF
                else:
                    key = -1
                if writer is not None:
                    writer.write(display_frame)

                if not self.registered:
                    selected = self._register_from_candidates(key, registration_candidates)
                    if selected is not None:
                        self.registered_embedding = selected.embedding
                        self.registered = True
                        self.tracker.initialize(frame_rgb, selected.person_bbox_xywh)
                        self.last_bbox_xywh = selected.person_bbox_xywh
                        self.tracker_score = 1.0
                        self.lost_counter = 0
                    elif key == ord("q"):
                        break
                elif key == ord("q"):
                    break

        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if self.args.display:
                cv2.destroyAllWindows()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO + face registration/ReID + MyECO otb936 person tracking pipeline.")
    parser.add_argument("--source", default="0", help="Video source. Use 0 for webcam or pass a video file path.")
    parser.add_argument("--output", default="", help="Optional output video path.")
    parser.add_argument("--display", action="store_true", default=False, help="Show the live preview window.")
    parser.add_argument("--cpu", action="store_true", default=False, help="Force CPU mode.")
    parser.add_argument("--debug", type=int, default=0, help="MyECO debug level.")
    parser.add_argument("--tracker-param", default="verified_otb936", help="Tracker parameter name from the copied MyECO workspace.")
    parser.add_argument("--yolo-model", default=default_person_yolo_model(), help="Ultralytics YOLO model or local weight path.")
    parser.add_argument("--yolo-imgsz", type=int, default=960, help="YOLO inference size.")
    parser.add_argument("--person-confidence", type=float, default=0.25, help="Minimum YOLO person confidence.")
    parser.add_argument("--face-confidence", type=float, default=0.90, help="Minimum face detector confidence.")
    parser.add_argument("--face-embedding-pretrained", default="vggface2", choices=["vggface2", "casia-webface"], help="Backbone checkpoint for face embeddings.")
    parser.add_argument("--face-embedding-weights", default="", help="Optional fine-tuned face embedding checkpoint.")
    parser.add_argument("--reid-threshold", type=float, default=0.78, help="Minimum cosine similarity to accept a face match.")
    parser.add_argument("--detect-interval", type=int, default=10, help="Run person/face reacquisition every N frames.")
    parser.add_argument("--reacquire-score-threshold", type=float, default=0.22, help="Force reacquisition when MyECO score drops below this value.")
    parser.add_argument("--reacquire-iou-threshold", type=float, default=0.35, help="Reinitialize MyECO if the matched detection drifts this far from the tracker.")
    parser.add_argument("--max-lost-frames", type=int, default=4, help="Treat the target as unstable after this many low-score frames.")
    parser.add_argument("--register-face-image", default="", help="Optional face image path to skip live registration.")
    parser.add_argument("--auto-register", action="store_true", default=False, help="Auto-register when only one visible face candidate exists.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.display and not args.register_face_image:
        raise SystemExit("Use --display for live face registration, or pass --register-face-image to run headless.")

    pipeline = PersonFaceTrackingPipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
