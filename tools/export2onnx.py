import os
import torch
import torch.onnx
from mmcv import Config
import numpy as np
import logging
from lib.models.build_counter import Baseline_Counter
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Export crowd counting model to ONNX')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--checkpoint',
                        help='checkpoint file to load weights',
                        required=True,
                        type=str)
    parser.add_argument('--output',
                        help='output ONNX file path',
                        default='model.onnx',
                        type=str)
    parser.add_argument('--input-shape',
                        help='input shape for the model (channels, height, width)',
                        default=(3, 1024, 2048),
                        type=int,
                        nargs=3)
    args = parser.parse_args()
    return args

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, image, label, mode):
        result = self.model(image, label, mode)
        # Nếu result là list (out_list), lấy 3 tensor đầu tiên
        if isinstance(result, (list, tuple)):
            if len(result) < 3:
                raise ValueError(f"Expected at least 3 outputs, got {len(result)}")
            # Giả sử out_list[0] là pre_den_1, out_list[-2] là pre_den_4, out_list[-1] là pre_den_8
            return result[0], result[-2], result[-1]
        # Nếu result là dictionary, lấy từ pre_den
        elif isinstance(result, dict):
            return (
                result['pre_den']['1'],  # out_list[0]
                result['pre_den']['4'],  # out_list[-2]
                result['pre_den']['8']   # out_list[-1]
            )
        else:
            raise ValueError(f"Unexpected result type: {type(result)}")

def export_to_onnx(cfg_file, checkpoint_file, output_file, input_shape):
    # Thiết lập logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Load config
    try:
        config = Config.fromfile(cfg_file)
        logger.info(f"Loaded config from {cfg_file}")
    except Exception as e:
        logger.error(f"Failed to load config {cfg_file}: {e}")
        return

    # Thiết lập device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Khởi tạo model
    model = Baseline_Counter(
        config.network,
        config.dataset.den_factor,
        config.train.route_size,
        device
    )
    logger.info("Model initialized")

    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    logger.info("Model loaded and set to evaluation mode")

    # Tạo wrapper để chỉ xuất 3 đầu ra
    wrapped_model = ModelWrapper(model)

    # Tạo dummy input
    batch_size = 1
    channels, height, width = input_shape
    dummy_input = torch.randn(batch_size, channels, height, width).to(device)
    dummy_label = []  # Giả lập label rỗng
    mode = 'val'
    logger.info(f"Dummy input shape: {dummy_input.shape}")

    # Xuất mô hình sang ONNX
    logger.info(f"Exporting model to {output_file}")
    try:
        torch.onnx.export(
            wrapped_model,
            (dummy_input, dummy_label, mode),
            output_file,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input', 'label', 'mode'],
            output_names=['pre_den_1', 'pre_den_4', 'pre_den_8'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'pre_den_1': {0: 'batch_size'},
                'pre_den_4': {0: 'batch_size'},
                'pre_den_8': {0: 'batch_size'}
            }
        )
        logger.info(f"Model exported successfully to {output_file}")
    except Exception as e:
        logger.error(f"Error exporting ONNX model: {e}")
        raise

    # Kiểm tra file ONNX
    try:
        import onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model is valid!")
    except ImportError:
        logger.warning("ONNX package not installed. Skipping model validation.")
    except Exception as e:
        logger.error(f"Error validating ONNX model: {e}")

def main():
    args = parse_args()
    export_to_onnx(
        cfg_file=args.cfg,
        checkpoint_file=args.checkpoint,
        output_file=args.output,
        input_shape=args.input_shape
    )

if __name__ == '__main__':
    main()
