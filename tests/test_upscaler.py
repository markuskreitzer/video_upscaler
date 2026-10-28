import pickle

import av
import numpy as np
from PIL import Image
import pytest
import torch
from spandrel.architectures.ESRGAN.__arch.RRDB import RRDBNet

from video_upscaler.main import upscale_frame, upscale_video
from video_upscaler.upscaler import RealESRGAN


class UnsupportedCheckpoint:
    pass


@pytest.fixture
def upscaler(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = RRDBNet(num_filters=4, num_blocks=1, scale=4)
    weights = tmp_path / "weights.pth"
    torch.save({"params": model.state_dict()}, weights)
    engine = RealESRGAN(torch.device("cpu"), scale=4)
    engine.load_weights(weights, download=False)
    return engine


def test_local_weights_preserve_rgb_image_dimensions(upscaler):
    image = Image.fromarray(np.full((17, 19, 3), 96, dtype=np.uint8))
    result = upscaler.predict(image, patches_size=16, padding=4, pad_size=3)
    assert result.mode == "RGB"
    assert result.size == (76, 68)


def test_checkpoint_loading_rejects_python_objects(tmp_path):
    weights = tmp_path / "unsupported.pth"
    torch.save(UnsupportedCheckpoint(), weights)
    engine = RealESRGAN(torch.device("cpu"))
    with pytest.raises(pickle.UnpicklingError):
        engine.load_weights(weights, download=False)


def test_weights_must_match_requested_scale(upscaler, tmp_path):
    weights = tmp_path / "weights.pth"
    incompatible = RealESRGAN(torch.device("cpu"), scale=2)
    with pytest.raises(ValueError, match="scale"):
        incompatible.load_weights(weights, download=False)


def test_video_frame_and_encoding_round_trip(upscaler, tmp_path, monkeypatch):
    image = Image.new("RGB", (16, 16), "blue")
    frame = upscale_frame(av.VideoFrame.from_image(image), upscaler)
    assert (frame.width, frame.height) == (64, 64)
    source = tmp_path / "input.mp4"
    with av.open(str(source), "w") as container:
        stream = container.add_stream("mpeg4", rate=5)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        for _ in range(2):
            for packet in stream.encode(av.VideoFrame.from_image(image)):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    monkeypatch.chdir(tmp_path)
    upscale_video(source, upscaler)
    with av.open("input_HD.mp4") as container:
        frames = list(container.decode(video=0))
    assert len(frames) == 2
    assert all((frame.width, frame.height) == (64, 64) for frame in frames)
