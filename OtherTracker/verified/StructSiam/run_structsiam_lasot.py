import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import cv2
import numpy as np
import tensorflow.compat.v1 as tf

from region_to_bbox import region_to_bbox
from siamese_net import SiameseNet

tf.disable_v2_behavior()


def configure_runtime():
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("[structsiam] runtime=CPU")
        return "/CPU:0"

    try:
        # Keep only the first GPU, which maps to the NVIDIA adapter in this env.
        tf.config.set_visible_devices(gpus[0], "GPU")
    except Exception:
        pass

    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except Exception:
        pass

    print(f"[structsiam] runtime=GPU device={gpus[0]}")
    return "/GPU:0"


def get_subwindow_tracking(img, pos, model_sz, original_sz, avg_chans):
    if original_sz is None:
        original_sz = model_sz

    sz = np.array(original_sz, dtype=np.int32)
    im_sz = img.shape
    assert min(im_sz[:2]) > 2, "image size is too small"
    c = (sz + 1) / 2.0

    context_xmin = round(pos[1] - c[1])
    context_xmax = context_xmin + sz[1] - 1
    context_ymin = round(pos[0] - c[0])
    context_ymax = context_ymin + sz[0] - 1
    left_pad = max(0, int(-context_xmin))
    top_pad = max(0, int(-context_ymin))
    right_pad = max(0, int(context_xmax - im_sz[1] + 1))
    bottom_pad = max(0, int(context_ymax - im_sz[0] + 1))

    context_xmin = int(context_xmin + left_pad)
    context_xmax = int(context_xmax + left_pad)
    context_ymin = int(context_ymin + top_pad)
    context_ymax = int(context_ymax + top_pad)

    if top_pad or left_pad or bottom_pad or right_pad:
        padded = np.zeros(
            (im_sz[0] + top_pad + bottom_pad, im_sz[1] + left_pad + right_pad, 3),
            dtype=img.dtype,
        )
        padded[:, :] = avg_chans
        padded[top_pad : top_pad + im_sz[0], left_pad : left_pad + im_sz[1], :] = img
        img = padded

    patch_original = img[context_ymin : context_ymax + 1, context_xmin : context_xmax + 1, :]
    if tuple(model_sz) != tuple(original_sz):
        patch = cv2.resize(patch_original, tuple(model_sz), interpolation=cv2.INTER_LINEAR)
    else:
        patch = patch_original
    return patch, patch_original


def tracker_eval(score, sx, target_position, window, hp, design):
    response_maps = score[:, :, :, 0]
    upsz = design["score_sz"] * hp["response_up"]
    response_maps_up = []

    if hp["scale_num"] > 1:
        current_scale_id = int(hp["scale_num"] / 2)
        best_scale = current_scale_id
        best_peak = -float("inf")

        for scale_idx in range(hp["scale_num"]):
            if hp["response_up"] > 1:
                response_maps_up.append(
                    cv2.resize(
                        response_maps[scale_idx, :, :],
                        (upsz, upsz),
                        interpolation=cv2.INTER_CUBIC,
                    )
                )
            else:
                response_maps_up.append(response_maps[scale_idx, :, :])

            response = response_maps_up[-1]
            if scale_idx != current_scale_id:
                response = response * hp["scale_penalty"]

            peak = np.max(response)
            if peak > best_peak:
                best_peak = peak
                best_scale = scale_idx

        response_map = response_maps_up[best_scale]
    else:
        response_map = cv2.resize(response_maps[0, :, :], (upsz, upsz), interpolation=cv2.INTER_CUBIC)
        best_scale = 0

    response_map = response_map - np.min(response_map)
    denom = np.sum(response_map)
    if denom > 0:
        response_map = response_map / denom
    else:
        response_map = np.full_like(response_map, 1.0 / response_map.size)

    if window.shape != response_map.shape:
        window_use = cv2.resize(
            window.astype(np.float32),
            (response_map.shape[1], response_map.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        window_sum = np.sum(window_use)
        if window_sum > 0:
            window_use = window_use / window_sum
    else:
        window_use = window

    response_map = (1 - hp["window_influence"]) * response_map + hp["window_influence"] * window_use
    r_max, c_max = np.unravel_index(response_map.argmax(), response_map.shape)
    p_corr = np.array((r_max, c_max))
    disp_instance_final = p_corr - int(upsz / 2)
    disp_instance_input = disp_instance_final * design["tot_stride"] / hp["response_up"]
    disp_instance_frame = disp_instance_input * sx / design["search_sz"]
    new_target_position = target_position + disp_instance_frame

    return new_target_position, best_scale


def make_scale_pyramid(im, target_position, in_side_scaled, out_side, avg_chans, num_scale):
    in_side_scaled = np.round(in_side_scaled)
    max_target_side = int(round(in_side_scaled[-1]))
    min_target_side = int(round(in_side_scaled[0]))
    beta = out_side / float(min_target_side)
    search_side = int(round(beta * max_target_side))
    search_region, _ = get_subwindow_tracking(
        im,
        target_position,
        (search_side, search_side),
        (max_target_side, max_target_side),
        avg_chans,
    )

    tmp_list = []
    tmp_pos = ((search_side - 1) / 2.0, (search_side - 1) / 2.0)
    for scale_idx in range(num_scale):
        target_side = round(beta * in_side_scaled[scale_idx])
        tmp_region, _ = get_subwindow_tracking(
            search_region,
            tmp_pos,
            (out_side, out_side),
            (target_side, target_side),
            avg_chans,
        )
        tmp_list.append(tmp_region)

    return np.stack(tmp_list)


def get_opts():
    return {
        "numScale": 3,
        "scaleStep": 1.0375,
        "scalePenalty": 0.9745,
        "scaleLr": 0.59,
        "responseUp": 16,
        "windowing": "cosine",
        "wInfluence": 0.176,
        "exemplarSize": 127,
        "instanceSize": 255,
        "scoreSize": 17,
        "totalStride": 8,
        "contextAmount": 0.5,
        "trainWeightDecay": 5e-4,
        "stddev": 0.03,
        "subMean": False,
    }


def build_tracker_graph(ckpt_path):
    hp = {
        "scale_min": 0.2,
        "window_influence": 0.25,
        "z_lr": 0.01,
        "scale_max": 5,
        "scale_step": 1.04,
        "scale_num": 3,
        "scale_penalty": 0.97,
        "response_up": 8,
        "scale_lr": 0.59,
    }
    design = {
        "exemplar_sz": 127,
        "search_sz": 255,
        "tot_stride": 4,
        "context": 0.5,
        "pad_with_image_mean": True,
        "windowing": "cosine_sum",
        "score_sz": 33,
        "trainBatchSize": 8,
    }
    opts = get_opts()

    device = configure_runtime()
    config = tf.ConfigProto(allow_soft_placement=True)

    with tf.device(device):
        exemplar_op = tf.placeholder(tf.float32, [1, design["exemplar_sz"], design["exemplar_sz"], 3])
        instance_op = tf.placeholder(tf.float32, [hp["scale_num"], design["search_sz"], design["search_sz"], 3])
        exemplar_op_bak = tf.placeholder(
            tf.float32,
            [design["trainBatchSize"], design["exemplar_sz"], design["exemplar_sz"], 3],
        )
        instance_op_bak = tf.placeholder(
            tf.float32,
            [design["trainBatchSize"], design["search_sz"], design["search_sz"], 3],
        )
        is_training_op = tf.convert_to_tensor(False, dtype="bool", name="is_training")

        sn = SiameseNet()
        _ = sn.buildTrainNetwork(exemplar_op_bak, instance_op_bak, opts, isTraining=False)
        saver = tf.train.Saver()
        zfeat_op = sn.buildExemplarSubNetwork(exemplar_op, opts, is_training_op)

    sess = tf.Session(config=config)
    saver.restore(sess, str(ckpt_path))

    return {
        "sess": sess,
        "sn": sn,
        "opts": opts,
        "hp": hp,
        "design": design,
        "exemplar_op": exemplar_op,
        "instance_op": instance_op,
        "is_training_op": is_training_op,
        "zfeat_op": zfeat_op,
    }


def init_lasot_video(seq_dir):
    seq_dir = Path(seq_dir)
    img_dir = seq_dir / "img"
    frame_paths = sorted(img_dir.glob("*.jpg"))
    if not frame_paths:
        frame_paths = sorted(img_dir.glob("*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No image files found in {img_dir}")

    gt = np.genfromtxt(seq_dir / "groundtruth.txt", delimiter=",")
    gt = np.atleast_2d(gt)
    return gt, frame_paths


def run_sequence(seq_dir, tracker_bundle, results_path=None, time_path=None, log_every=100):
    gt, frame_paths = init_lasot_video(seq_dir)
    hp = tracker_bundle["hp"]
    design = tracker_bundle["design"]
    opts = tracker_bundle["opts"]
    sess = tracker_bundle["sess"]
    sn = tracker_bundle["sn"]
    exemplar_op = tracker_bundle["exemplar_op"]
    instance_op = tracker_bundle["instance_op"]
    is_training_op = tracker_bundle["is_training_op"]
    zfeat_op = tracker_bundle["zfeat_op"]

    pos_x, pos_y, target_w, target_h = region_to_bbox(gt[0], center=True)
    num_frames = len(frame_paths)

    scale_factors = hp["scale_step"] ** np.linspace(
        -np.ceil(hp["scale_num"] / 2), np.ceil(hp["scale_num"] / 2), hp["scale_num"]
    )
    final_score_sz = hp["response_up"] * (design["score_sz"] - 1) + 1
    hann_1d = np.expand_dims(np.hanning(final_score_sz), axis=0)
    window = hann_1d.T.dot(hann_1d)
    window = window / np.sum(window)

    bboxes = np.zeros((num_frames, 4), dtype=np.float32)
    times = np.zeros((num_frames,), dtype=np.float64)

    init_start = time.perf_counter()
    im = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    avg_chans = np.mean(im, axis=(0, 1))
    context = design["context"] * (target_w + target_h)
    z_sz = np.sqrt(np.prod((target_w + context) * (target_h + context)))
    scalez = design["exemplar_sz"] / z_sz
    z_crop, _ = get_subwindow_tracking(
        im,
        [pos_y, pos_x],
        (design["exemplar_sz"], design["exemplar_sz"]),
        (np.around(z_sz), np.around(z_sz)),
        avg_chans,
    )
    d_search = (design["search_sz"] - design["exemplar_sz"]) / 2
    pad = d_search / scalez
    sx = z_sz + 2 * pad
    min_sx = 0.2 * sx
    max_sx = 5.0 * sx
    target_size = np.array([target_h, target_w], dtype=np.float32)

    z_crop = np.expand_dims(z_crop, axis=0)
    z_feat = sess.run(zfeat_op, feed_dict={exemplar_op: z_crop.astype(np.float32)})
    z_feat = np.transpose(z_feat, [1, 2, 3, 0])
    z_feat_constant_op = tf.constant(z_feat, dtype=tf.float32)
    score_op = sn.buildInferenceNetwork(instance_op, z_feat_constant_op, opts, is_training_op)
    times[0] = time.perf_counter() - init_start

    bboxes[0, :] = [pos_x - target_w / 2, pos_y - target_h / 2, target_w, target_h]

    for frame_idx in range(1, num_frames):
        if frame_idx % log_every == 0 or frame_idx == num_frames - 1:
            print(f"[structsiam] frame {frame_idx + 1}/{num_frames}", flush=True)

        frame_start = time.perf_counter()
        im = cv2.imread(str(frame_paths[frame_idx]), cv2.IMREAD_COLOR)
        scaled_instance = sx * scale_factors
        scaled_target = np.array([target_size * scale_i for scale_i in scale_factors])
        x_crops = make_scale_pyramid(
            im,
            [pos_y, pos_x],
            scaled_instance,
            design["search_sz"],
            avg_chans,
            hp["scale_num"],
        )
        score = sess.run(score_op, feed_dict={instance_op: x_crops.astype(np.float32)})
        new_target_position, new_scale = tracker_eval(
            score, round(sx), np.array([pos_y, pos_x]), window, hp, design
        )
        sx = max(min_sx, min(max_sx, (1 - hp["scale_lr"]) * sx + hp["scale_lr"] * scaled_instance[new_scale]))
        target_size = (1 - hp["scale_lr"]) * target_size + hp["scale_lr"] * scaled_target[new_scale]

        rect_position = new_target_position - target_size / 2.0
        tl = tuple(np.round(rect_position).astype(int)[::-1])
        br = tuple(np.round(rect_position + target_size).astype(int)[::-1])
        bboxes[frame_idx, :] = [tl[0], tl[1], br[0] - tl[0], br[1] - tl[1]]
        pos_y = tl[1] + (br[1] - tl[1]) / 2.0
        pos_x = tl[0] + (br[0] - tl[0]) / 2.0
        times[frame_idx] = time.perf_counter() - frame_start

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
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt_path = Path(__file__).resolve().parent / "ckpt" / "ckpt" / "model_epoch49.ckpt"
    tracker_bundle = build_tracker_graph(ckpt_path)
    try:
        run_sequence(
            args.seq_dir,
            tracker_bundle,
            results_path=args.results_path,
            time_path=args.time_path,
            log_every=args.log_every,
        )
    finally:
        tracker_bundle["sess"].close()


if __name__ == "__main__":
    main()
