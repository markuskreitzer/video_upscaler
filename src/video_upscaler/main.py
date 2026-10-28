import sys
import torch
from PIL import Image
import numpy as np
import av
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from video_upscaler.upscaler import RealESRGAN

def upscale_image(image: Image, device: torch.device, weights: Path, scale: int = 4) -> Image:
    """Upscale a single image."""
    upscaler = get_upscaler(device, weights, scale)
    return upscaler.predict(image)

def upscale_frame(frame: av.VideoFrame, upscaler: RealESRGAN) -> av.VideoFrame:
    """Upscale a single frame."""
    img = frame.to_image().convert('RGB')
    sr_img = upscaler.predict(img)
    return av.VideoFrame.from_image(sr_img)

def get_upscaler(device: torch.device, model_weights_path: Path, scale: int = 4) -> RealESRGAN:
    """Get the upscaler model."""
    upscaler = RealESRGAN(device, scale)
    upscaler.load_weights(model_weights_path, download=True)
    return upscaler

def upscale_video(input_path: Path, upscaler: RealESRGAN, scale: int = 4) -> None:
    """Upscale video frames and reassemble the video."""

    input_container = av.open(str(input_path))
    output_container = av.open(f"{input_path.stem}_HD.mp4",  'w')

    stream = input_container.streams.video[0]
    output_stream = output_container.add_stream('mpeg4', rate=stream.average_rate)
    output_stream.width = stream.width * scale
    output_stream.height = stream.height * scale
    output_stream.pix_fmt = 'yuv420p'

    total_frames = input_container.streams.video[0].frames

    print("Upscaling video...")
    progress_bar = tqdm(total=total_frames, unit='frames', desc='Upscaling')

    for frame in input_container.decode(stream):
        sr_frame = upscale_frame(frame, upscaler)
        packet = output_stream.encode(sr_frame)
        output_container.mux(packet)
        progress_bar.update()

    progress_bar.close()

    print("Flush and close the containers")
    progress_bar = tqdm(total=total_frames, unit='frames', desc='Encoding/Muxing')
    for packet in output_stream.encode():
        output_container.mux(packet)
        progress_bar.update()

    progress_bar.close()

    input_container.close()
    output_container.close()

def get_device() -> torch.device:
    if torch.cuda.is_available():
        print("Using CUDA.")
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        print("Using MPS.")
        return torch.device('mps')
    else:
        print("Using CPU.")
        return torch.device('cpu')
    
def main():
    parser = ArgumentParser(description="Upscale video files.")
    parser.add_argument('input_file', type=Path, help='Path to the input video file.')
    parser.add_argument('--scale', type=int, default=4, help='Upscaling factor.')
    parser.add_argument('--model_weights', type=Path, default=Path('weights/RealESRGAN_x4.pth'), help='Path to the model weights file.')
    parser.add_argument('--download-weights', action='store_true', default=True, help='Download the model weights if they are not found locally.')
    args = parser.parse_args()
    device = get_device()
    upscaler = get_upscaler(device, args.model_weights, args.scale)
    upscale_video(args.input_file, upscaler, args.scale)

if __name__ == "__main__":
    main()

