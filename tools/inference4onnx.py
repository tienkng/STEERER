import os
import argparse
import logging
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort
from mmcv import Config
import torch
from tqdm import tqdm
import time
import glob

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference using ONNX model for crowd counting')
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--onnx-model', help='path to ONNX model file', required=True, type=str)
    parser.add_argument('--input-dir', help='directory containing input images', required=True, type=str)
    parser.add_argument('--output-dir', help='directory to save output images', default='output', type=str)
    parser.add_argument('--input-shape', help='input shape for the model (channels, height, width)', 
                       default=(3, 1024, 2048), type=int, nargs=3)
    parser.add_argument('--providers', help='ONNX runtime providers (comma-separated)', 
                       default='CUDAExecutionProvider,CPUExecutionProvider', type=str)
    parser.add_argument('--device', help='Device to run inference on (cpu or cuda)', default='cuda', type=str)
    parser.add_argument('--loc-threshold', help='threshold for local maximum points', default=0.1, type=float)
    args = parser.parse_args()
    return args

def local_maximum_points(density_map, gaussian_maximum=1.0, patch_size=32, den_scale=1, threshold=0.1):
    density_map = density_map.squeeze()
    h, w = density_map.shape
    points = []
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = density_map[i:i+patch_size, j:j+patch_size]
            if patch.shape[0] == 0 or patch.shape[1] == 0:
                continue
            max_val = np.max(patch)
            if max_val > threshold * gaussian_maximum:
                max_idx = np.unravel_index(np.argmax(patch), patch.shape)
                point_y = (i + max_idx[0]) * den_scale
                point_x = (j + max_idx[1]) * den_scale
                points.append([point_x, point_y])
    points = np.array(points, dtype=np.float32) if points else np.array([], dtype=np.float32).reshape(-1, 2)
    return {'points': points, 'num': len(points)}

def save_results(name, sv_dir, input_img, pre_points):
    os.makedirs(sv_dir, exist_ok=True)
    if input_img.max() <= 1.0:
        input_img = (input_img * 255).astype(np.uint8)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    img_with_points = input_img.copy()
    RGB_G = (0, 255, 0)
    thickness = 5
    marker_size = 20
    for point in pre_points:
        point = point.astype(np.int32)
        x, y = point[0], point[1]
        cv2.drawMarker(img_with_points, (x, y), RGB_G, markerType=cv2.MARKER_CROSS, 
                      markerSize=marker_size, thickness=thickness)
    pil_img = Image.fromarray(img_with_points)
    output_path = os.path.join(sv_dir, f'{name}.jpg')
    pil_img.save(output_path)
    logging.info(f'Saved result to {output_path}')

def preprocess_image(image_path, input_shape, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    orig_img = img.copy()
    orig_h, orig_w = img.shape[:2]
    channels, height, width = input_shape
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - np.array(mean)) / np.array(std)
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img, orig_img, (orig_h, orig_w)

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Determine device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using device: CUDA")
    else:
        device = torch.device("cpu")
        logger.info("Using device: CPU")
    
    # Convert providers
    provider_list = [p.strip() for p in args.providers.split(',')]
    if device.type == "cpu":
        provider_list = ["CPUExecutionProvider"]
        logger.info("Overriding ONNX providers to CPUExecutionProvider")

    # Load config
    try:
        config = Config.fromfile(args.cfg)
        logger.info(f'Loaded config from {args.cfg}')
    except Exception as e:
        logger.error(f'Failed to load config {args.cfg}: {e}')
        return

    # Load ONNX model
    try:
        session = ort.InferenceSession(args.onnx_model, providers=provider_list)
        logger.info(f'Loaded ONNX model from {args.onnx_model}')
        logger.info(f'ONNX Runtime providers: {session.get_providers()}')
    except Exception as e:
        logger.error(f'Failed to load ONNX model: {e}')
        return

    # Get input images
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))
    image_paths.sort()
    if not image_paths:
        logger.error(f'No images found in {args.input_dir}')
        return

    total_inference_time = 0
    count = 0
    logger.info('Running warm-up to initialize model...')
    dummy_input = np.random.randn(1, *args.input_shape).astype(np.float32)
    try:
        _ = session.run(None, {'input': dummy_input})
        logger.info('Warm-up completed.')
    except Exception as e:
        logger.error(f'Warm-up failed: {e}')
        return

    for img_path in tqdm(image_paths, desc='Processing images'):
        name = os.path.splitext(os.path.basename(img_path))[0]
        logger.info(f'Processing image: {name}')
        input_tensor, orig_img, orig_size = preprocess_image(img_path, args.input_shape)
        inputs = {'input': input_tensor.astype(np.float32)}

        start_time = time.time()
        try:
            outputs = session.run(None, inputs)
        except Exception as e:
            logger.error(f'Failed to run inference on {img_path}: {e}')
            continue
        end_time = time.time()

        inference_time = end_time - start_time
        total_inference_time += inference_time
        count += 1
        logger.info(f'Inference time for {name}: {inference_time:.4f} seconds')

        pre_den_1, pre_den_4, pre_den_8 = outputs
        pred_data = local_maximum_points(pre_den_1, gaussian_maximum=1.0, patch_size=32, threshold=args.loc_threshold)
        orig_h, orig_w = orig_size
        input_h, input_w = args.input_shape[1], args.input_shape[2]
        adjusted_points = pred_data['points'].copy()
        if adjusted_points.shape[0] > 0:
            adjusted_points[:, 0] = adjusted_points[:, 0] * orig_w / input_w
            adjusted_points[:, 1] = adjusted_points[:, 1] * orig_h / input_h
            adjusted_points[:, 0] = np.clip(adjusted_points[:, 0], 0, orig_w)
            adjusted_points[:, 1] = np.clip(adjusted_points[:, 1], 0, orig_h)

        save_results(name, args.output_dir, orig_img, adjusted_points)

    if count > 0:
        logger.info(f'Average inference time: {total_inference_time / count:.4f} seconds')
    else:
        logger.warning('No successful inferences were made.')

if __name__ == '__main__':
    main()
