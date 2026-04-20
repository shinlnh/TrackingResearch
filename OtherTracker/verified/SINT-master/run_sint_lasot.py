import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import Ridge

_NET = None
_RUNTIME_DESC = "CPU"


def func_iou(bb, gtbb):
    iw = min(bb[2], gtbb[2]) - max(bb[0], gtbb[0]) + 1
    ih = min(bb[3], gtbb[3]) - max(bb[1], gtbb[1]) + 1
    if iw <= 0 or ih <= 0:
        return 0.0
    ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1)
    ua += (gtbb[2] - gtbb[0] + 1) * (gtbb[3] - gtbb[1] + 1)
    ua -= iw * ih
    return float(iw * ih) / float(ua)


def sample_regions_precompute(rad, nr_ang, stepsize, scales=(0.7071, 1.0, 1.4142)):
    nr_step = int(rad / stepsize)
    cos_values = np.cos(np.arange(0, 2 * np.pi, 2 * np.pi / nr_ang))
    sin_values = np.sin(np.arange(0, 2 * np.pi, 2 * np.pi / nr_ang))

    dxdys = np.zeros((2, nr_step * nr_ang + 1), dtype=np.float32)
    count = 0
    for ir in range(1, nr_step + 1):
        offset = stepsize * ir
        for ia in range(1, nr_ang + 1):
            dx = offset * cos_values[ia - 1]
            dy = offset * sin_values[ia - 1]
            dxdys[0, count] = dx
            dxdys[1, count] = dy
            count += 1

    samples = np.zeros((4, (nr_ang * nr_step + 1) * len(scales)), dtype=np.float32)
    jump = nr_step * nr_ang + 1
    for idx, scale in enumerate(scales):
        samples[0:2, idx * jump : (idx + 1) * jump] = dxdys
        samples[2, idx * jump : (idx + 1) * jump] = scale
        samples[3, idx * jump : (idx + 1) * jump] = scale
    return samples


def sample_regions(x, y, w, h, im_w, im_h, samples_template):
    samples = samples_template.copy()
    samples[0, :] += x
    samples[1, :] += y
    samples[2, :] *= w
    samples[3, :] *= h

    samples[2, :] = samples[0, :] + samples[2, :] - 1
    samples[3, :] = samples[1, :] + samples[3, :] - 1
    samples = np.round(samples)

    flags = np.logical_and.reduce(
        (samples[0, :] > 0, samples[1, :] > 0, samples[2, :] < im_w, samples[3, :] < im_h)
    )
    return samples[:, flags]


def preprocess_image(image_bgr, image_sz, mean_bgr):
    resized = cv2.resize(image_bgr, (image_sz, image_sz), interpolation=cv2.INTER_LINEAR)
    blob = resized.astype(np.float32)
    blob -= mean_bgr.reshape(1, 1, 3)
    blob = np.transpose(blob, (2, 0, 1))
    return np.expand_dims(blob, axis=0)


def extract_feat(net, image_bgr, rois, image_sz, mean_bgr):
    blob = preprocess_image(image_bgr, image_sz, mean_bgr)
    net.setInput(blob, "data")
    net.setInput(rois.astype(np.float32), "rois")
    out = net.forward("feat_l2")
    out = np.array(out, copy=True)
    if out.ndim == 1:
        out = out.reshape(1, -1)
    return out


def load_lasot_sequence(seq_dir):
    seq_dir = Path(seq_dir)
    gt = np.genfromtxt(seq_dir / "groundtruth.txt", delimiter=",")
    gt = np.atleast_2d(gt).astype(np.float32)
    frame_paths = sorted((seq_dir / "img").glob("*.jpg"))
    if not frame_paths:
        frame_paths = sorted((seq_dir / "img").glob("*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No image frames found in {seq_dir / 'img'}")
    return gt, frame_paths


def get_net():
    global _NET, _RUNTIME_DESC
    if _NET is None:
        model_dir = Path(__file__).resolve().parent
        cache_dir = model_dir / ".ocl4dnn_cache"
        cache_dir.mkdir(exist_ok=True)
        os.environ.setdefault("OPENCV_OCL4DNN_CONFIG_PATH", str(cache_dir))
        os.environ.setdefault("OPENCV_OPENCL_DEVICE", "NVIDIA:GPU:")
        prototxt = model_dir / "protos" / "SINT_deploy.prototxt"
        caffemodel = model_dir / "SINT_similarity.caffemodel"
        _NET = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
        _NET.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        _NET.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        _RUNTIME_DESC = "CPU"
        try:
            cv2.ocl.setUseOpenCL(True)
            if cv2.ocl.haveOpenCL() and cv2.ocl.useOpenCL():
                device = cv2.ocl.Device_getDefault()
                if device.name() and "NVIDIA" in device.vendorName().upper():
                    _NET.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
                    _RUNTIME_DESC = f"GPU OpenCL ({device.name()})"
        except Exception:
            pass
    return _NET


def get_runtime_desc():
    if _NET is None:
        get_net()
    return _RUNTIME_DESC


def run_sequence(seq_dir, results_path=None, time_path=None, log_every=100, save_every=100):
    seq_dir = Path(seq_dir)
    net = get_net()

    gtboxes, frame_paths = load_lasot_sequence(seq_dir)
    image_sz = 512
    mean_bgr = np.array([102.9801, 115.9465, 122.7717], dtype=np.float32)

    init_box = gtboxes[0, :].copy()
    firstframe = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    first_h, first_w = firstframe.shape[:2]

    init_roi = np.zeros((1, 5), dtype=np.float32)
    init_roi[0, 1] = init_box[0] * image_sz / first_w
    init_roi[0, 2] = init_box[1] * image_sz / first_h
    init_roi[0, 3] = (init_box[0] + init_box[2] - 1) * image_sz / first_w
    init_roi[0, 4] = (init_box[1] + init_box[3] - 1) * image_sz / first_h
    init_roi[0, 1:] -= 1

    init_start = time.perf_counter()
    qfeat = extract_feat(net, firstframe, init_roi, image_sz, mean_bgr).squeeze()

    base_rad = 30
    nr_angles = 20
    ov_thresh = 0.6
    stepsize = 3
    samples_templates = sample_regions_precompute(
        float(base_rad) / 512 * first_w, nr_angles, 1, scales=(0.7, 0.8, 0.9, 1.0, 1 / 0.9, 1 / 0.8, 1 / 0.7)
    )
    samples = sample_regions(init_box[0], init_box[1], init_box[2], init_box[3], first_w, first_h, samples_templates)

    ov_samples = np.zeros((samples.shape[1],), dtype=np.float32)
    init_box_xyxy = init_box.copy()
    init_box_xyxy[2] = init_box_xyxy[2] + init_box_xyxy[0] - 1
    init_box_xyxy[3] = init_box_xyxy[3] + init_box_xyxy[1] - 1
    for ii in range(samples.shape[1]):
        ov_samples[ii] = func_iou(samples[:, ii], init_box_xyxy)

    sel_samples = samples[:, ov_samples > ov_thresh]
    sel_rois = np.zeros((sel_samples.shape[1], 5), dtype=np.float32)
    sel_rois[:, 1:] = sel_samples.T.copy()
    sel_rois[:, 1] = sel_rois[:, 1] * image_sz / first_w - 1
    sel_rois[:, 3] = sel_rois[:, 3] * image_sz / first_w - 1
    sel_rois[:, 2] = sel_rois[:, 2] * image_sz / first_h - 1
    sel_rois[:, 4] = sel_rois[:, 4] * image_sz / first_h - 1

    br_feats = extract_feat(net, firstframe, sel_rois, image_sz, mean_bgr)
    br_feats = br_feats[:, 0:25088]

    br_coor = sel_samples.copy()
    br_coor[2, :] = br_coor[2, :] - br_coor[0, :] + 1
    br_coor[3, :] = br_coor[3, :] - br_coor[1, :] + 1
    br_coor[0, :] = br_coor[0, :] + 0.5 * br_coor[2, :]
    br_coor[1, :] = br_coor[1, :] + 0.5 * br_coor[3, :]

    gt_coor = init_box.copy()
    gt_coor[0] = gt_coor[0] + 0.5 * gt_coor[2]
    gt_coor[1] = gt_coor[1] + 0.5 * gt_coor[3]

    target_x = (gt_coor[0] - br_coor[0, :]) / br_coor[2, :]
    target_y = (gt_coor[1] - br_coor[1, :]) / br_coor[3, :]
    target_w = np.log(gt_coor[2] / br_coor[2, :])
    target_h = np.log(gt_coor[3] / br_coor[3, :])

    regr_x = Ridge(alpha=1, fit_intercept=False).fit(br_feats, target_x)
    regr_y = Ridge(alpha=1, fit_intercept=False).fit(br_feats, target_y)
    regr_w = Ridge(alpha=1, fit_intercept=False).fit(br_feats, target_w)
    regr_h = Ridge(alpha=1, fit_intercept=False).fit(br_feats, target_h)
    init_time = time.perf_counter() - init_start

    samples_tmpl = sample_regions_precompute(float(base_rad) / 512 * first_w, 10, stepsize)
    prev_box = init_box.copy()
    num_frames = len(frame_paths)
    bboxes = np.zeros((num_frames, 4), dtype=np.float32)
    times = np.zeros((num_frames,), dtype=np.float64)
    bboxes[0, :] = init_box
    times[0] = init_time

    for frame_idx in range(1, num_frames):
        if frame_idx % log_every == 0 or frame_idx == num_frames - 1:
            print(f"[sint] frame {frame_idx + 1}/{num_frames}", flush=True)
        frame_start = time.perf_counter()

        image = cv2.imread(str(frame_paths[frame_idx]), cv2.IMREAD_COLOR)
        im_h, im_w = image.shape[:2]
        samples = sample_regions(prev_box[0], prev_box[1], init_box[2], init_box[3], im_w, im_h, samples_tmpl)
        nr_samples = samples.shape[1]

        rois = np.zeros((nr_samples, 5), dtype=np.float32)
        rois[:, 1:] = samples.T.copy()
        rois[:, 1] = rois[:, 1] * image_sz / im_w - 1
        rois[:, 3] = rois[:, 3] * image_sz / im_w - 1
        rois[:, 2] = rois[:, 2] * image_sz / im_h - 1
        rois[:, 4] = rois[:, 4] * image_sz / im_h - 1
        tfeats = extract_feat(net, image, rois, image_sz, mean_bgr)

        scores = np.dot(tfeats, qfeat)
        max_idx = int(np.argmax(scores))
        candidate_box = samples[:, max_idx].copy()

        prev_box = candidate_box.copy()
        prev_box[2] = prev_box[2] - prev_box[0] + 1
        prev_box[3] = prev_box[3] - prev_box[1] + 1

        box_feat = tfeats[max_idx, 0:25088].copy()
        p_x = float(regr_x.predict(box_feat.reshape(1, -1))[0])
        p_y = float(regr_y.predict(box_feat.reshape(1, -1))[0])
        p_w = float(regr_w.predict(box_feat.reshape(1, -1))[0])
        p_h = float(regr_h.predict(box_feat.reshape(1, -1))[0])

        new_x = p_x * prev_box[2] + prev_box[0] + 0.5 * prev_box[2]
        new_y = p_y * prev_box[3] + prev_box[1] + 0.5 * prev_box[3]
        new_w = prev_box[2] * np.exp(p_w)
        new_h = prev_box[3] * np.exp(p_h)
        new_x = new_x - 0.5 * new_w
        new_y = new_y - 0.5 * new_h

        bboxes[frame_idx, :] = [new_x, new_y, new_w, new_h]

        # Conservative propagation like the original code: use the raw best sample for next search.
        prev_box = candidate_box.copy()
        prev_box[2] = prev_box[2] - prev_box[0] + 1
        prev_box[3] = prev_box[3] - prev_box[1] + 1

        times[frame_idx] = time.perf_counter() - frame_start
        if results_path is not None and (frame_idx % save_every == 0 or frame_idx == num_frames - 1):
            np.savetxt(results_path, bboxes[: frame_idx + 1, :], fmt="%.6f", delimiter=",")
        if time_path is not None and (frame_idx % save_every == 0 or frame_idx == num_frames - 1):
            np.savetxt(time_path, times[: frame_idx + 1].reshape(-1, 1), fmt="%.10f", delimiter="\t")

    fps = float(num_frames / np.sum(times)) if np.sum(times) > 0 else 0.0
    if results_path is not None:
        np.savetxt(results_path, bboxes, fmt="%.6f", delimiter=",")
    if time_path is not None:
        np.savetxt(time_path, times.reshape(-1, 1), fmt="%.10f", delimiter="\t")
    print(f"fps={fps:.6f}", flush=True)
    return bboxes, times, fps


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-dir", required=True)
    parser.add_argument("--results-path")
    parser.add_argument("--time-path")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    run_sequence(
        args.seq_dir,
        results_path=args.results_path,
        time_path=args.time_path,
        log_every=args.log_every,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
