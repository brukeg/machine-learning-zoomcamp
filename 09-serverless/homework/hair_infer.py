import io
import json
from urllib import request
from typing import Tuple, Dict, Any

import numpy as np
from PIL import Image
import onnxruntime as ort


def download_image(url: str) -> Image.Image:
    with request.urlopen(url) as resp:
        buf = resp.read()
    img = Image.open(io.BytesIO(buf))
    return img


def prepare_image(img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img.resize(target_size, Image.NEAREST)


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def to_numpy(img: Image.Image, target_size=(200, 200), use_imagenet_norm=True) -> np.ndarray:
    """
    Returns CHW float32 ready for ONNX (batch-free).
    """
    img = prepare_image(img, target_size)
    x = np.array(img).astype(np.float32) / 255.0  # HWC, [0,1]
    if use_imagenet_norm:
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
    # HWC -> CHW
    x = np.transpose(x, (2, 0, 1))
    return x


def load_session(model_path: str) -> ort.InferenceSession:
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return sess


def get_io_names(model_path: str) -> Dict[str, Any]:
    sess = load_session(model_path)
    inputs = [i.name for i in sess.get_inputs()]
    outputs = [o.name for o in sess.get_outputs()]
    return {"inputs": inputs, "outputs": outputs}


def run_model(model_path: str, x_chw: np.ndarray, input_name: str | None = None, output_name: str | None = None) -> np.ndarray:
    sess = load_session(model_path)
    if input_name is None:
        input_name = sess.get_inputs()[0].name
    if output_name is None:
        output_name = sess.get_outputs()[0].name
    
    xb = np.expand_dims(x_chw, axis=0)
    out = sess.run([output_name], {input_name: xb})[0]
    return out


def cli_q1(model_path: str = "hair_classifier_v1.onnx"):
    io_names = get_io_names(model_path)
    print(json.dumps(io_names, indent=2))


def cli_q2_print_target_hint():
    print("Target size should be 200x200.")


def cli_q3_first_pixel_r(url: str, target=(200, 200), imagenet=True):
    img = download_image(url)
    x = to_numpy(img, target_size=target, use_imagenet_norm=imagenet)  # CHW
    r0 = x[0, 0, 0].item()
    print(f"First pixel, R channel: {r0:.3f}")


def cli_q4_score(url: str, model_path="hair_classifier_v1.onnx", target=(200, 200), imagenet=True):
    img = download_image(url)
    x = to_numpy(img, target_size=target, use_imagenet_norm=imagenet)
    y = run_model(model_path, x)

    val = float(np.array(y).reshape(-1)[0])
    print(f"Model output: {val:.2f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("q1")
    s1.add_argument("--model", default="hair_classifier_v1.onnx")

    s3 = sub.add_parser("q3")
    s3.add_argument("--url", required=True)
    s3.add_argument("--size", default="200x200")
    s3.add_argument("--imagenet", action="store_true", default=True)

    s4 = sub.add_parser("q4")
    s4.add_argument("--url", required=True)
    s4.add_argument("--model", default="hair_classifier_v1.onnx")
    s4.add_argument("--size", default="200x200")
    s4.add_argument("--imagenet", action="store_true", default=True)

    args = p.parse_args()
    if args.cmd == "q1":
        cli_q1(args.model)
    elif args.cmd == "q3":
        w, h = map(int, args.size.split("x"))
        cli_q3_first_pixel_r(args.url, target=(w, h), imagenet=args.imagenet)
    elif args.cmd == "q4":
        w, h = map(int, args.size.split("x"))
        cli_q4_score(args.url, model_path=args.model, target=(w, h), imagenet=args.imagenet)
