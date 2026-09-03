import onnxruntime as ort
import cv2
import numpy as np

class FastSCNN:
    """ONNX‑Runtime wrapper for Fast‑SCNN (3‑class)."""
    def __init__(self, onnx_path: str, input_size=(640, 360)):
        self.session = ort.InferenceSession(
            onnx_path, providers=["CUDAExecutionProvider"]
        )
        self.input_size = input_size
        self.input_name = self.session.get_inputs()[0].name

    def infer(self, bgr_frame: np.ndarray) -> np.ndarray:
        """
        Returns a (H,W) uint8 mask with values {0,1,2}.
        """
        img = cv2.resize(bgr_frame, self.input_size).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None]          # NCHW
        out = self.session.run(None, {self.input_name: img})[0]
        mask = out.argmax(1)[0].astype(np.uint8)
        return mask