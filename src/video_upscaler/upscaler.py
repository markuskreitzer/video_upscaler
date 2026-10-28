from pathlib import Path
from shutil import copyfile

import numpy as np
from PIL import Image
import torch
from huggingface_hub import hf_hub_download
from spandrel import ImageModelDescriptor, ModelLoader


class RealESRGAN:
    def __init__(self, device: torch.device, scale: int = 4):
        self.device = device
        self.scale = scale
        self.model: ImageModelDescriptor | None = None

    def load_weights(self, model_path: Path, download: bool = True) -> None:
        path = Path(model_path)
        if not path.exists() and download:
            if self.scale not in (2, 4, 8):
                raise ValueError("Automatic weights support scales 2, 4 and 8")
            cached = hf_hub_download(
                repo_id="sberbank-ai/Real-ESRGAN",
                filename=f"RealESRGAN_x{self.scale}.pth",
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            copyfile(cached, path)
        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict):
            state = state.get("params", state.get("params_ema", state))
        model = ModelLoader().load_from_state_dict(state)
        if not isinstance(model, ImageModelDescriptor):
            raise ValueError("Weights must describe an image upscaling model")
        if model.scale != self.scale or model.input_channels != 3 or model.output_channels != 3:
            raise ValueError("Weights must match the requested RGB upscaling scale")
        self.model = model.to(self.device).eval()

    def predict(
        self,
        image: Image.Image,
        batch_size: int = 4,
        patches_size: int = 192,
        padding: int = 24,
        pad_size: int = 15,
    ) -> Image.Image:
        if self.model is None:
            raise RuntimeError("Load weights before upscaling")
        if batch_size < 1 or patches_size < 1 or padding < 0 or pad_size < 0:
            raise ValueError("Invalid patch dimensions")
        pixels = np.asarray(image.convert("RGB"))
        pixels = np.pad(pixels, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode="symmetric")
        height, width = pixels.shape[:2]
        extra_h, extra_w = -height % patches_size, -width % patches_size
        extended = np.pad(pixels, ((0, extra_h), (0, extra_w), (0, 0)), mode="edge")
        padded = np.pad(extended, ((padding, padding), (padding, padding), (0, 0)), mode="edge")
        result = np.empty((extended.shape[0] * self.scale, extended.shape[1] * self.scale, 3), dtype=np.uint8)
        border = padding * self.scale
        size = patches_size * self.scale
        positions = [(y, x) for y in range(0, extended.shape[0], patches_size) for x in range(0, extended.shape[1], patches_size)]
        with torch.inference_mode(), torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
            for offset in range(0, len(positions), batch_size):
                batch = positions[offset:offset + batch_size]
                patches = np.stack([padded[y:y + patches_size + 2 * padding, x:x + patches_size + 2 * padding] for y, x in batch])
                tensor = torch.from_numpy(np.ascontiguousarray(patches)).permute(0, 3, 1, 2)
                outputs = self.model(tensor.to(self.device, dtype=torch.float32) / 255)
                for (y, x), output in zip(batch, outputs):
                    tile = output[:, border:border + size, border:border + size]
                    tile = tile.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
                    result[y * self.scale:(y + patches_size) * self.scale, x * self.scale:(x + patches_size) * self.scale] = (tile * 255).astype(np.uint8)
        start = pad_size * self.scale
        return Image.fromarray(result[start:(height - pad_size) * self.scale, start:(width - pad_size) * self.scale])
