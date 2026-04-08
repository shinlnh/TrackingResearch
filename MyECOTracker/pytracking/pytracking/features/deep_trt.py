import os

import tensorrt as trt
import torch

from pytracking import TensorList
from pytracking.evaluation.environment import env_settings
from pytracking.features.featurebase import MultiFeatureBase


_TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
def _binding_dtype_to_torch(dtype: trt.DataType) -> torch.dtype:
    if dtype == trt.DataType.FLOAT:
        return torch.float32
    if dtype == trt.DataType.HALF:
        return torch.float16
    if dtype == trt.DataType.INT8:
        return torch.int8
    if dtype == trt.DataType.INT32:
        return torch.int32
    if hasattr(trt.DataType, "BOOL") and dtype == trt.DataType.BOOL:
        return torch.bool
    raise TypeError("Unsupported TensorRT dtype: {}".format(dtype))


class ResNet18m1TensorRT(MultiFeatureBase):
    """TensorRT runtime for the ECO ResNet18+VGGm backbone."""

    def __init__(self, output_layers, engine_path=None, use_gpu=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        valid_layers = {"vggconv1", "conv1", "layer1", "layer2", "layer3", "layer4", "fc"}
        for layer in output_layers:
            if layer not in valid_layers:
                raise ValueError("Unknown layer: {}".format(layer))

        self.output_layers = list(output_layers)
        self.use_gpu = use_gpu
        self.engine_path = (
            "resnet18_vggmconv1/resnet18_vggmconv1_otb_dyn_fp16.engine"
            if engine_path is None
            else engine_path
        )

        self.layer_stride = {
            "vggconv1": 2,
            "conv1": 2,
            "layer1": 4,
            "layer2": 8,
            "layer3": 16,
            "layer4": 32,
            "fc": None,
        }
        self.layer_dim = {
            "vggconv1": 96,
            "conv1": 64,
            "layer1": 64,
            "layer2": 128,
            "layer3": 256,
            "layer4": 512,
            "fc": None,
        }

        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, -1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, -1, 1, 1)

        self.device = torch.device("cuda")
        self.runtime = None
        self.engine = None
        self.context = None
        self.input_binding_index = None
        self.output_binding_indices = []

    def _resolve_engine_path(self) -> str:
        if os.path.isabs(self.engine_path):
            return self.engine_path

        root_paths = env_settings().network_path
        if isinstance(root_paths, str):
            root_paths = [root_paths]

        for root in root_paths:
            candidate = os.path.join(root, self.engine_path)
            if os.path.isfile(candidate):
                return candidate

        raise FileNotFoundError("Did not find TensorRT engine {}".format(self.engine_path))

    def initialize(self):
        if not self.use_gpu or not torch.cuda.is_available():
            raise RuntimeError("TensorRT ECO requires CUDA")

        engine_path = self._resolve_engine_path()

        self.runtime = trt.Runtime(_TRT_LOGGER)
        with open(engine_path, "rb") as fh:
            self.engine = self.runtime.deserialize_cuda_engine(fh.read())
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine {}".format(engine_path))

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context for {}".format(engine_path))

        for binding_index in range(self.engine.num_bindings):
            if self.engine.binding_is_input(binding_index):
                self.input_binding_index = binding_index
                break
        if self.input_binding_index is None:
            raise RuntimeError("TensorRT engine has no input binding")

        self.output_binding_indices = []
        for layer in self.output_layers:
            binding_index = self.engine.get_binding_index(layer)
            if binding_index < 0:
                raise RuntimeError("TensorRT engine missing output binding '{}'".format(layer))
            self.output_binding_indices.append(binding_index)

        self.mean = self.mean.to(self.device)
        self.std = self.std.to(self.device)
        self.use_gpu = True

    def dim(self):
        return TensorList([self.layer_dim[layer] for layer in self.output_layers])

    def stride(self):
        return TensorList([self.pool_stride[idx] * self.layer_stride[layer] for idx, layer in enumerate(self.output_layers)])

    def extract(self, im: torch.Tensor):
        if self.context is None:
            self.initialize()

        im = im.to(self.device, non_blocking=True).contiguous()
        im = im / 255.0
        im = (im - self.mean) / self.std
        im = im.contiguous()

        input_shape = tuple(int(dim) for dim in im.shape)
        self.context.set_binding_shape(self.input_binding_index, input_shape)

        bindings = [0] * self.engine.num_bindings
        bindings[self.input_binding_index] = int(im.data_ptr())

        outputs = []
        for binding_index in self.output_binding_indices:
            output_shape = tuple(int(dim) for dim in self.context.get_binding_shape(binding_index))
            if any(dim < 0 for dim in output_shape):
                raise RuntimeError("TensorRT returned unresolved output shape {}".format(output_shape))

            output_dtype = _binding_dtype_to_torch(self.engine.get_binding_dtype(binding_index))
            output_tensor = torch.empty(output_shape, device=self.device, dtype=output_dtype)
            bindings[binding_index] = int(output_tensor.data_ptr())
            outputs.append(output_tensor)

        if not self.context.execute_v2(bindings):
            raise RuntimeError("TensorRT execute_v2 failed for input shape {}".format(input_shape))

        return TensorList([tensor.float().contiguous() for tensor in outputs])
