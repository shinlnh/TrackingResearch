import os
from pathlib import Path

os.environ.setdefault("OPENCV_OPENCL_DEVICE", "NVIDIA:GPU:")

import cv2
import numpy as np


_NET = None
_MODEL_KEY = None
_RUNTIME_DESC = "CPU"


def _configure_runtime(net, root):
    global _RUNTIME_DESC

    cache_dir = root / ".ocl4dnn_cache"
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("OPENCV_OCL4DNN_CONFIG_PATH", str(cache_dir))

    try:
        cv2.ocl.setUseOpenCL(True)
    except Exception:
        pass

    try:
        if cv2.ocl.haveOpenCL() and cv2.ocl.useOpenCL():
            device = cv2.ocl.Device_getDefault()
            device_name = device.name()
            device_vendor = device.vendorName()
            if device_name and "NVIDIA" in device_vendor.upper():
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                if hasattr(cv2.dnn, "DNN_TARGET_OPENCL_FP16"):
                    net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)
                    _RUNTIME_DESC = f"GPU OpenCL FP16 ({device_name})"
                else:
                    net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
                    _RUNTIME_DESC = f"GPU OpenCL ({device_name})"
                return
    except Exception:
        pass

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    _RUNTIME_DESC = "CPU"


def init_backend(prototxt=None, caffemodel=None):
    global _NET, _MODEL_KEY
    root = Path(__file__).resolve().parent
    prototxt = Path(prototxt) if prototxt else root / "siamese_networks" / "deploy.prototxt"
    caffemodel = Path(caffemodel) if caffemodel else root / "siamese_networks" / "similarity.caffemodel"
    model_key = (str(prototxt), str(caffemodel))
    if _NET is None or _MODEL_KEY != model_key:
        _NET = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
        _configure_runtime(_NET, root)
        _MODEL_KEY = model_key
    return True


def get_runtime_desc():
    if _NET is None:
        init_backend()
    return _RUNTIME_DESC


def extract_features(image, rois, image_sz=512, pixel_means=None):
    if _NET is None:
        init_backend()

    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    arr = arr.astype(np.float32)
    if pixel_means is None:
        pixel_means = np.array([102.9801, 115.9465, 122.7717], dtype=np.float32)
    else:
        pixel_means = np.asarray(pixel_means, dtype=np.float32).reshape(1, 1, 3)

    # MATLAB images arrive as RGB; OpenCV DNN expects BGR.
    arr = arr[:, :, ::-1]
    resized = cv2.resize(arr, (int(image_sz), int(image_sz)), interpolation=cv2.INTER_LINEAR)
    blob = resized - pixel_means
    blob = np.transpose(blob, (2, 0, 1))[None, :, :, :]

    rois = np.asarray(rois, dtype=np.float32)
    if rois.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if rois.ndim == 1:
        rois = rois.reshape(1, -1)
    if rois.shape[0] == 1 and rois.shape[1] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if rois.shape[0] == 5 and rois.shape[1] != 5:
        rois = rois.T
    if rois.shape[1] != 5:
        raise ValueError(f"Expected rois to have 5 columns, got shape {rois.shape}")
    rois = rois.copy()
    rois[:, 0] = 0.0
    _NET.setInput(blob, "data")
    _NET.setInput(rois, "rois")
    out = _NET.forward("feat_l2")
    out = np.asarray(out, dtype=np.float32)
    if out.ndim == 1:
        out = out.reshape(1, -1)
    return out.T
