"""
oak_obstacles.py -- OAK-D -> obstacle list of (label, forward_m, lateral_m) per frame.

Uses DepthAI v3's SpatialDetectionNetwork: your converted model runs on the camera,
and each detection gets an (x, y, z) from stereo depth. This script logs those as an
obstacle list the controller can consume later. Distance source is swappable: when the
lidar is fused in, only the code that fills forward_m/lateral_m changes.

Run:
    python oak_obstacles.py              # headless: prints + logs to $AKS_DATA/obstacles.csv
    python oak_obstacles.py --show       # also opens an OpenCV window (q to quit)
Stop headless runs with Ctrl+C.
"""
import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import depthai as dai

sys.path.append(str(Path(__file__).resolve().parents[1]))   # repo root, for autonomy.paths
from autonomy.paths import DATA, MODELS  # noqa: E402
from autonomy.storage import DataBudget  # noqa: E402

# ---------------- CONFIG ----------------
MODEL_ARCHIVE = MODELS / "aks_obj_detection.rvc2.tar.xz"   # override with --model
FPS = 10                 # camera/depth rate; your model runs ~7 FPS at 640x640 anyway
CONF = 0.5               # detection confidence threshold
DEPTH_MIN_MM = 200       # ignore depth readings closer than this
DEPTH_MAX_MM = 8000      # ignore depth readings farther than this (stereo gets noisy)
BOX_SCALE = 0.5          # use the center 50% of each box for depth (cones are thin)
NN_SHAVES = 7            # RVC2 has 16 shaves; stereo depth etc. leave ~7 for the network. Try 6 if it still errors
LATERAL_SIGN = 1         # flip to -1 so that LEFT is positive in the kart frame
# ----------------------------------------


@dataclass
class Obstacle:
    ts: float            # host time (seconds)
    label: str           # "cone" / "person" / "kart"
    conf: float
    forward_m: float     # distance ahead of the camera
    lateral_m: float     # sideways offset (sign set by LATERAL_SIGN)


class ObstacleNode(dai.node.HostNode):
    """Host-side node: turns spatial detections into an obstacle list each frame."""

    def __init__(self):
        dai.node.HostNode.__init__(self)
        self.sendProcessingToPipeline(True)
        self.show = False
        self.writer = None
        self.budget = None        # DataBudget; when it runs out, CSV logging stops and detection carries on
        self.log_stopped = False
        self.latest = []          # most recent obstacle list; the controller will read this

    def build(self, rgb: dai.Node.Output, detections: dai.Node.Output):
        self.link_args(rgb, detections)   # must match process() arguments

    def process(self, rgb, detections):
        now = time.time()
        obstacles = []
        for d in detections.detections:
            c = d.spatialCoordinates
            if c.z <= 0:                  # no valid depth for this detection
                continue
            label = getattr(d, "labelName", None) or str(d.label)
            obstacles.append(Obstacle(
                ts=now,
                label=label,
                conf=float(d.confidence),
                forward_m=c.z / 1000.0,
                lateral_m=LATERAL_SIGN * c.x / 1000.0,
            ))
        obstacles.sort(key=lambda o: o.forward_m)   # nearest first
        self.latest = obstacles

        if obstacles and self.writer and not self.log_stopped and self.budget is not None:
            status = self.budget.check()
            if not status.ok:
                self.log_stopped = True
                print(f"WARNING: obstacle CSV logging stopped ({status.reason}); detection keeps running")
        for o in obstacles:
            print(f"{o.label:7s} conf={o.conf:.2f} fwd={o.forward_m:5.2f} m  lat={o.lateral_m:+5.2f} m")
            if self.writer and not self.log_stopped:
                row = [f"{o.ts:.3f}", o.label, f"{o.conf:.3f}", f"{o.forward_m:.3f}", f"{o.lateral_m:.3f}"]
                self.writer.writerow(row)
                if self.budget is not None:
                    self.budget.add(sum(len(v) for v in row) + len(row))
        if not obstacles:
            print("-- no obstacles --")

        if self.show:
            frame = rgb.getCvFrame()
            h, w = frame.shape[:2]
            for d in detections.detections:
                if d.spatialCoordinates.z <= 0:
                    continue
                x1, y1 = int(d.xmin * w), int(d.ymin * h)
                x2, y2 = int(d.xmax * w), int(d.ymax * h)
                name = getattr(d, "labelName", None) or str(d.label)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
                cv2.putText(frame, f"{name} {d.spatialCoordinates.z / 1000:.1f}m",
                            (x1 + 4, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("obstacles", frame)
            if cv2.waitKey(1) == ord("q"):
                self.stopPipeline()


def make_depth(p):
    """Stereo depth source. Newer DepthAI v3 has dai.node.Depth; older builds use StereoDepth."""
    if hasattr(dai.node, "Depth"):
        return p.create(dai.node.Depth).build(dai.node.Depth.Algorithm.AUTO, FPS, (640, 400))
    left = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B, sensorFps=FPS)
    right = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C, sensorFps=FPS)
    stereo = p.create(dai.node.StereoDepth)
    left.requestOutput((640, 400)).link(stereo.left)
    right.requestOutput((640, 400)).link(stereo.right)
    return stereo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="open an OpenCV window")
    ap.add_argument("--log", type=Path, default=DATA / "obstacles.csv", help="CSV log path")
    ap.add_argument("--model", type=Path, default=MODEL_ARCHIVE, help=".rvc2.tar.xz archive (not extracted)")
    args = ap.parse_args()

    if not args.model.is_file():
        raise SystemExit(f"Model archive not found: {args.model}\n"
                         "Copy it into $AKS_MODELS (default ~/aks_models) or pass --model.")
    nn_archive = dai.NNArchive(str(args.model))
    args.log.parent.mkdir(parents=True, exist_ok=True)
    print(f"Logging to {args.log}")

    with dai.Pipeline() as p, open(args.log, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ts", "label", "conf", "forward_m", "lateral_m"])

        # RGB camera on an OAK-D is CAM_A (this DepthAI version's getConnectedCameras() takes no args)
        cam = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A, sensorFps=FPS)
        depth = make_depth(p)

        snn = p.create(dai.node.SpatialDetectionNetwork).build(cam, depth, nn_archive)
        try:
            snn.setNNArchive(nn_archive, NN_SHAVES)   # pick a blob that fits alongside stereo depth
        except TypeError as e:
            print(f"Could not set shave count ({e}); using the default")
        snn.setConfidenceThreshold(CONF)
        snn.setBoundingBoxScaleFactor(BOX_SCALE)
        snn.setDepthLowerThreshold(DEPTH_MIN_MM)
        snn.setDepthUpperThreshold(DEPTH_MAX_MM)
        snn.input.setBlocking(False)

        node = p.create(ObstacleNode)
        node.show = args.show
        node.writer = writer
        node.budget = DataBudget(DATA)
        node.build(snn.passthrough, snn.out)

        print("Starting pipeline (Ctrl+C to stop)...")
        p.run()


if __name__ == "__main__":
    main()