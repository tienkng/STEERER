import cv2
import argparse
import numpy as np
import onnxruntime as ort


norm_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
norm_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference using ONNX model for crowd counting"
    )
    parser.add_argument(
        "--onnx-model", help="path to ONNX model file", required=True, type=str
    )
    parser.add_argument(
        "--input-dir", help="directory containing input images", required=True, type=str
    )
    parser.add_argument(
        "--output-dir",
        help="directory to save output images",
        default="output",
        type=str,
    )
    args = parser.parse_args()

    return args


def local_maximum_points(
    density_map, gaussian_maximum=1.0, patch_size=32, den_scale=1, threshold=0.1
):
    density_map = density_map.squeeze()
    h, w = density_map.shape
    points = []
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = density_map[i : i + patch_size, j : j + patch_size]
            if patch.shape[0] == 0 or patch.shape[1] == 0:
                continue
            max_val = np.max(patch)
            if max_val > threshold * gaussian_maximum:
                max_idx = np.unravel_index(np.argmax(patch), patch.shape)
                point_y = (i + max_idx[0]) * den_scale
                point_x = (j + max_idx[1]) * den_scale
                points.append([point_x, point_y])
    points = (
        np.array(points, dtype=np.float32)
        if points
        else np.array([], dtype=np.float32).reshape(-1, 2)
    )
    return {"points": points, "num": len(points)}


class STEERERONNX:
    def __init__(self, model_path):
        self.model = ort.InferenceSession(
            model_path, providers=["CUDAExecutionProvider"]
        )

        self.inp_name = [x.name for x in self.model.get_inputs()]
        self.opt_name = [x.name for x in self.model.get_outputs()]
        _, _, h, w = self.model.get_inputs()[0].shape
        self.model_inpsize = (w, h)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Resize and prepare the image for ONNX inference.

        Args:
            image (np.ndarray): Input image in shape (H, W, C) - OpenCV format

        Returns:
            input_tensor (np.ndarray): (1, C, H, W), float32
            ratio (tuple): (w_ratio, h_ratio) for resizing back (postprocess)
        """
        h_model, w_model = self.model_inpsize[1], self.model_inpsize[0]
        h_original, w_original = image.shape[:2]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.model_inpsize, interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        image = (image - norm_mean) / norm_std
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)

        ratio_w = w_original / w_model
        ratio_h = h_original / h_model

        return image, (ratio_w, ratio_h)

    def postprocess(
        self,
        output: np.ndarray,
        org_image: np.ndarray,
        ratio: tuple,
        threshold: float = 0.1,
    ) -> np.ndarray:
        """
        Postprocess the output from ONNX inference.

        Args:
            output (np.ndarray): Output from the model
            ratio (tuple): (w_ratio, h_ratio) for resizing back

        Returns:
            points (np.ndarray): Detected points in original image scale
        """
        orig_h, orig_w = org_image.shape[:2]
        pred_data = local_maximum_points(
            output[0], gaussian_maximum=1.0, patch_size=32, threshold=threshold
        )
        adjusted_points = pred_data["points"].copy()
        if adjusted_points.shape[0] > 0:
            adjusted_points[:, 0] *= ratio[0]  # x
            adjusted_points[:, 1] *= ratio[1]  # y
            adjusted_points[:, 0] = np.clip(adjusted_points[:, 0], 0, orig_w)
            adjusted_points[:, 1] = np.clip(adjusted_points[:, 1], 0, orig_h)

        return adjusted_points

    def run(self, image: np.ndarray, thresold=0.1) -> np.ndarray:
        """
        Run inference on the input image.

        Args:
            image (np.ndarray): Input image in shape (H, W, C) - OpenCV format

        Returns:
            points (np.ndarray): Detected points in original image scale
        """
        input_tensor, ratio = self.preprocess(image)
        outputs = self.model.run(self.opt_name, {self.inp_name[0]: input_tensor})
        points = self.postprocess(outputs[0], image, ratio, thresold)
        return points


if __name__ == "__main__":
    import os
    import glob

    args = parse_args()
    model = STEERERONNX(args.onnx_model)

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    for img_path in glob.glob(f"{input_dir}/*.jpg"):
        img_name = os.path.basename(img_path)
        if not os.path.isfile(img_path):
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to read image: {img_name}")
            break

        points = model.run(image, thresold=0.1)

        if len(points) > 0:
            for point in points:
                cv2.drawMarker(
                    image,
                    (int(point[0]), int(point[1])),
                    color=(0, 255, 0),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=10,
                    thickness=5,
                )
            cv2.imwrite(os.path.join(output_dir, img_name), image)
    print("Inference completed.")
