import depthai as dai
import sys
from pathlib import Path

from depthai_nodes.node import SnapsUploader
from depthai_nodes.node.parsing_neural_network import ParsingNeuralNetwork
from utils.snaps_producer import SnapsProducer
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[1]))   # repo root, for autonomy.paths
from autonomy.paths import MODELS, REPO  # noqa: E402

# Assumes `DEPTHAI_HUB_API_KEY` is defined in the workspace root `.env` file.
# Load environment variables before initializing the pipeline.
load_dotenv(REPO / ".env", override=True)

# Path to the converted model in $AKS_MODELS. Do NOT extract it: point at the .tar.xz file itself.
MODEL_ARCHIVE = MODELS / "aks_obj_detection.rvc2.tar.xz"   # update file name here
time_interval = 10.0  # min nr of seconds between snaps uploading

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device()

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    if not MODEL_ARCHIVE.is_file():
        raise SystemExit(f"Model archive not found: {MODEL_ARCHIVE}\n"
                         "Copy it into $AKS_MODELS (default ~/aks_models).")
    nn_archive = dai.NNArchive(str(MODEL_ARCHIVE))

    input_node = pipeline.create(dai.node.Camera).build()

    nn_with_parser = pipeline.create(ParsingNeuralNetwork).build(
        input_node, nn_archive
    )

    visualizer.addTopic("Video", nn_with_parser.passthrough, "images")
    visualizer.addTopic("Visualizations", nn_with_parser.out, "images")

    #snaps_producer = pipeline.create(SnapsProducer).build(
        #frame=nn_with_parser.passthrough,
        #detections=nn_with_parser.out,
        #time_interval=time_interval
    #)

    #snaps_uploader = pipeline.create(SnapsUploader).build(snaps_producer.out)

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break