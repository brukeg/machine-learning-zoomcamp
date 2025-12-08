import json
from hair_infer import download_image, to_numpy, run_model


MODEL_PATH = "hair_classifier_empty.onnx"
TARGET = (200, 200)
IMAGENET = True

def lambda_handler(event, context=None):
    url = event.get("url")
    if not url:
        return {"statusCode": 400, "body": json.dumps({"error": "missing 'url'"})}

    img = download_image(url)
    x = to_numpy(img, target_size=TARGET, use_imagenet_norm=IMAGENET)
    y = run_model(MODEL_PATH, x)
    score = float(y.reshape(-1)[0])
    return {"statusCode": 200, "body": json.dumps({"score": score})}
