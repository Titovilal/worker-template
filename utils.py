import base64
import json
from PIL import Image
import io
import random

def get_input_data(event: dict) -> dict:
    """Gets the input data from the event"""
    input_data = event["input"]

    if not input_data or not input_data.get("image") or not input_data.get("mask"):
        return {"error": "Both 'image' and 'mask' are required"}

    positive_prompt = "closeup, portrait, epic, movie scene, naked, nude"
    negative_prompt = "watermark, ugly, clothes"

    return {
        "mask": input_data.get("mask"),
        "image": input_data.get("image"),
        "positive_prompt": input_data.get("positive_prompt") or positive_prompt,
        "negative_prompt": input_data.get("negative_prompt") or negative_prompt,
        "seed": input_data.get("seed") or random.randint(0, 2**32 - 1),
        "steps": input_data.get("steps") or 20,
        "cfg": input_data.get("cfg") or 8,
        "denoise": input_data.get("denoise") or 1,
    }


def save_image(image_data: str | bytes, path: str) -> None:
    """Saves an image to a filepath"""
    if isinstance(image_data, str):  # base64
        image_bytes = base64.b64decode(image_data)
    else:  # binary
        image_bytes = image_data

    image = Image.open(io.BytesIO(image_bytes))
    image.save(path)


def create_test_input(image_path: str, mask_path: str, save_path: str) -> None:
    """Creates a test input JSON file for the handler"""
    with open(image_path, "rb") as img_file:
        image_base64 = base64.b64encode(img_file.read()).decode()

    with open(mask_path, "rb") as mask_file:
        mask_base64 = base64.b64encode(mask_file.read()).decode()

    test_input = {"input": {"image": image_base64, "mask": mask_base64}}

    with open(save_path, "w") as f:
        json.dump(test_input, f, indent=2)

    print(f"JSON saved to {save_path}")
