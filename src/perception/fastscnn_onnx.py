import onnxruntime as ort
import cv2
import numpy as np
import warnings


class FastSCNN:
    """ONNX‑Runtime wrapper for Fast‑SCNN (3‑class)."""

    def __init__(self, onnx_path: str, input_size=(640, 360)):
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._device = "GPU (CUDA)"
        else:
            providers = ["CPUExecutionProvider"]
            self._device = "CPU"
            warnings.warn(
                "CUDAExecutionProvider not available — falling back to CPU. "
                f"Available providers: {available}"
            )

        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_size = input_size
        self.input_name = self.session.get_inputs()[0].name
        print(f"FastSCNN ONNX session created — device: {self._device}, "
              f"providers: {self.session.get_providers()}")

    def infer(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Returns a (H,W) uint8 mask with values {0,1,2}."""
        img = cv2.resize(bgr_frame, self.input_size).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None]  # NCHW
        out = self.session.run(None, {self.input_name: img})[0]
        mask = out.argmax(1)[0].astype(np.uint8)
        return mask