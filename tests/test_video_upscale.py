from pathlib import Path
from unittest.mock import Mock

from video_upscaler import main


def test_cli_passes_selected_weights_and_scale(monkeypatch):
    engine = Mock()
    get_upscaler = Mock(return_value=engine)
    upscale_video = Mock()
    monkeypatch.setattr(main, "get_upscaler", get_upscaler)
    monkeypatch.setattr(main, "upscale_video", upscale_video)
    monkeypatch.setattr(main, "get_device", lambda: "cpu")
    monkeypatch.setattr(main.sys, "argv", ["upscale", "input.mp4", "--scale", "2", "--model_weights", "local.pth"])
    main.main()
    get_upscaler.assert_called_once_with("cpu", Path("local.pth"), 2)
    upscale_video.assert_called_once_with(Path("input.mp4"), engine, 2)
