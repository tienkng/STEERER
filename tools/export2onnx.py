import os
import onnx
import torch
import argparse

from lib.models.build_counter import Baseline_Counter
from mmcv import Config


class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.model = model

    def forward(self, x):
        result = self.model(x)[0]

        return result


def parse_args():
    parser = argparse.ArgumentParser(description="Export crowd counting model to ONNX")
    parser.add_argument(
        "--model-cfg",
        help="experiment configure file name",
        required=True,
        type=str,
        default="configs/SHHB_final.py",
    )
    parser.add_argument(
        "--checkpoint",
        help="checkpoint file to load weights",
        required=True,
        type=str,
        default="weights/SHHB_mae_5.8_mse_8.5.pth",
    )
    parser.add_argument(
        "--input-shape",
        help="input shape for the model (height, width)",
        default=[1024, 2048],
        type=int,
        nargs="+",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    args = parser.parse_args()

    return args


def export_to_onnx(
    model_cfg,
    checkpoint,
    input_shape=(1024, 2048),
    batch_size=1,  # batch size
    simplify=True,
    cleanup=True,
):
    # Load configuration
    config = Config.fromfile(model_cfg)

    # Initialize model
    model = Baseline_Counter(
        config.network, config.dataset.den_factor, config.train.route_size, "cpu"
    )
    pretrained_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(pretrained_dict, strict=True)
    model.eval()

    model = WrappedModel(model)

    # Create dummy input
    batch_size = 1
    im = torch.zeros(batch_size, 3, *input_shape)
    for _ in range(2):
        y = model(im)  # dry runs

    f = os.path.splitext(checkpoint)[0] + ".onnx"
    torch.onnx.export(
        model,
        im,
        f,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=["image"],
        # output_names=["pre_den_1", "pre_den_4", "pre_den_8"],
        output_names=["pre_den_1"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "pre_den_1": {0: "batch_size"},
            # "pre_den_4": {0: "batch_size"},
            # "pre_den_8": {0: "batch_size"},
        },
    )

    model_onnx = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    if simplify:
        try:
            import onnxsim

            print("\nStarting to simplify ONNX...")
            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, "assert check failed"
        except Exception as e:
            print(f"Simplifier failure: {e}")

        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        onnx.save(model_onnx, f)
        print("ONNX export success, saved as %s" % f)

    if cleanup:
        try:
            import importlib.util

            # Check if package is installed
            if importlib.util.find_spec("onnx_graphsurgeon") is None:
                os.system("pip install onnx_graphsurgeon")

            import onnx_graphsurgeon as gs

            print("\nStarting to cleanup ONNX using onnx_graphsurgeon...")
            graph = gs.import_onnx(model_onnx)
            graph = graph.cleanup().toposort()
            model_onnx = gs.export_onnx(graph)
        except Exception as e:
            print(f"Cleanup failure: {e}")

    inputs = [node for node in model_onnx.graph.input]
    print("Inputs: ", inputs, "\n")
    outputs = [node for node in model_onnx.graph.output]
    print("Outputs: ", outputs)


if __name__ == "__main__":
    args = parse_args()
    export_to_onnx(**vars(args))
