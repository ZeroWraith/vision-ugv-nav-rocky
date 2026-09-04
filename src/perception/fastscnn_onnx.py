import cv2
import numpy as np
import torch
import torchvision


# ImageNet normalization
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# VOC class -> navigation class mapping
_TRAVERSABLE = {0, 7, 10, 11, 12, 13, 14}  # bg, road, grass, sidewalk, terrain, fence, dirt
_SKY = {17}


class DeepLabV3Seg:
    """PyTorch DeepLabV3-ResNet50 segmentation on CUDA GPU."""

    def __init__(self, input_size=(640, 360)):
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(
            weights="DeepLabV3_ResNet50_Weights.DEFAULT"
        ).cuda().eval()
        self.input_size = input_size  # (W, H)
        self.mean = _IMAGENET_MEAN.cuda()
        self.std = _IMAGENET_STD.cuda()

    def infer(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Returns a (H,W) uint8 mask: 0=traversable, 1=obstacle, 2=sky."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self.input_size)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
        tensor = (tensor - self.mean) / self.std

        with torch.no_grad():
            out = self.model(tensor)["out"]

        raw = out.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        nav = np.ones_like(raw, dtype=np.uint8)  # default: obstacle
        for c in _TRAVERSABLE:
            nav[raw == c] = 0
        for c in _SKY:
            nav[raw == c] = 2
        return nav
