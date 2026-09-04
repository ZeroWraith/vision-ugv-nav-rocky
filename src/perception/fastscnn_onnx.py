import onnxruntime as ort
import cv2
import numpy as np


class FastSCNN:
    """ONNX-Runtime wrapper for Fast-SCNN (3-class). Requires CUDA GPU."""

    def __init__(self, onnx_path: str, input_size=(640, 360)):
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                f"CUDAExecutionProvider not available. "
                f"Available providers: {available}. "
                f"Install onnxruntime-gpu and ensure CUDA is configured."
            )

        options = ort.SessionOptions()
        options.log_severity_level = 0  # verbose — shows which ops fall back

        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=options,
            providers=["CUDAExecutionProvider"],
        )

        actual = self.session.get_providers()
        if "CUDAExecutionProvider" not in actual:
            raise RuntimeError(
                f"ONNX session failed to use CUDA. "
                f"Requested: ['CUDAExecutionProvider'], Actual: {actual}. "
                f"Check CUDA runtime libraries (cuDNN, cuBLAS)."
            )

        self.input_size = input_size
        self.input_name = self.session.get_inputs()[0].name
        print(f"FastSCNN ONNX — device: GPU (CUDA)")

    def infer(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Returns a (H,W) uint8 mask with values {0,1,2}."""
        img = cv2.resize(bgr_frame, self.input_size).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None]  # NCHW
        out = self.session.run(None, {self.input_name: img})[0]
        mask = out.argmax(1)[0].astype(np.uint8)
        return mask
