"""Frame sources: image, video file, webcam index, or OAK-D (DepthAI v3)."""
from __future__ import annotations

import json
from pathlib import Path

import cv2

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def open_source(spec: str, width: int = 640, height: int = 480, loop_image: bool = False):
    """Yields BGR frames. spec = 'oak', a webcam index like '0', or a file path."""
    if spec == "oak":
        yield from _oak_frames(width, height)
        return
    if Path(spec).suffix.lower() in IMG_EXT:
        img = cv2.imread(spec)
        if img is None:
            raise FileNotFoundError(spec)
        while True:
            yield img.copy()
            if not loop_image:
                return
    cap = cv2.VideoCapture(int(spec) if spec.isdigit() else spec)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {spec}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield frame
    finally:
        cap.release()


def _oak_frames(width: int, height: int):
    # NOTE: if your detector pipeline already owns the OAK-D, don't open a second one.
    # Add another cam.requestOutput(...) to that pipeline and feed its frames to LaneDetector.
    import depthai as dai

    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        q = cam.requestOutput((width, height), dai.ImgFrame.Type.BGR888p).createOutputQueue(
            maxSize=4, blocking=False
        )
        pipeline.start()
        while pipeline.isRunning():
            yield q.get().getCvFrame()


def save_oak_intrinsics(path: str, width: int = 640, height: int = 480) -> None:
    """Dump the factory RGB calibration (K, D) so lane_detector can undistort.
    Use a 4:3 output size (e.g. 640x480) so it matches the sensor without cropping."""
    import depthai as dai

    with dai.Device() as dev:
        calib = dev.readCalibration()
        K = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, width, height)
        D = calib.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A)
    Path(path).write_text(json.dumps({"K": K, "D": D, "size": [width, height]}, indent=2), encoding="utf-8")
    print(f"Saved intrinsics to {path}")
