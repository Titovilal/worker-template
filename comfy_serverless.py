import json
from urllib import request
import websocket
import uuid
import io
from PIL import Image
import base64
import random
from dotenv import load_dotenv
import os

load_dotenv()

# Is loaded as string to improve efficiency and reduce I/O load.
workflow_dump = """
{
  "4": {
    "inputs": {
      "ckpt_name": "juggernaut-xl-inpainting.safetensors"
    },
    "class_type": "CheckpointLoaderSimple",
    "_meta": {
      "title": "Load Checkpoint"
    }
  },
  "36": {
    "inputs": {
      "noise_mask": true,
      "positive": [
        "37",
        0
      ],
      "negative": [
        "38",
        0
      ],
      "vae": [
        "4",
        2
      ],
      "pixels": [
        "43",
        1
      ],
      "mask": [
        "43",
        2
      ]
    },
    "class_type": "InpaintModelConditioning",
    "_meta": {
      "title": "InpaintModelConditioning"
    }
  },
  "37": {
    "inputs": {
      "text": "closeup, portrait, epic, movie scene, naked, nude",
      "clip": [
        "4",
        1
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "38": {
    "inputs": {
      "text": "watermark, ugly, clothes",
      "clip": [
        "4",
        1
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "39": {
    "inputs": {
      "seed": 996178653109355,
      "steps": 30,
      "cfg": 8,
      "sampler_name": "dpmpp_2m",
      "scheduler": "karras",
      "denoise": 1,
      "model": [
        "4",
        0
      ],
      "positive": [
        "36",
        0
      ],
      "negative": [
        "36",
        1
      ],
      "latent_image": [
        "36",
        2
      ]
    },
    "class_type": "KSampler",
    "_meta": {
      "title": "KSampler"
    }
  },
  "40": {
    "inputs": {
      "samples": [
        "39",
        0
      ],
      "vae": [
        "4",
        2
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "43": {
    "inputs": {
      "context_expand_pixels": 20,
      "context_expand_factor": 1,
      "fill_mask_holes": true,
      "blur_mask_pixels": 16,
      "invert_mask": false,
      "blend_pixels": 16,
      "rescale_algorithm": "bicubic",
      "mode": "ranged size",
      "force_width": 1024,
      "force_height": 1024,
      "rescale_factor": 1,
      "min_width": 512,
      "min_height": 512,
      "max_width": 768,
      "max_height": 768,
      "padding": 8,
      "image": [
        "47",
        0
      ],
      "mask": [
        "48",
        0
      ]
    },
    "class_type": "InpaintCrop",
    "_meta": {
      "title": "✂️ Inpaint Crop"
    }
  },
  "45": {
    "inputs": {
      "rescale_algorithm": "bislerp",
      "stitch": [
        "43",
        0
      ],
      "inpainted_image": [
        "40",
        0
      ]
    },
    "class_type": "InpaintStitch",
    "_meta": {
      "title": "✂️ Inpaint Stitch"
    }
  },
  "47": {
    "inputs": {
      "image": "tmp.png",
      "upload": "image"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Image"
    }
  },
  "48": {
    "inputs": {
      "image": "tmp-mask.png",
      "channel": "alpha",
      "upload": "image"
    },
    "class_type": "LoadImageMask",
    "_meta": {
      "title": "Load Image (as Mask)"
    }
  },
  "save_image_websocket_node": {
    "class_type": "SaveImageWebsocket",
    "inputs": {
      "images": [
        "45",
        0
      ]
    }
  }
}
"""


def validate_api_key(api_key: str) -> tuple[bool, str]:
    """Validates the API key"""
    local_api_key = os.getenv("API_KEY")

    if not local_api_key:
        return False, "API_KEY not found in env"

    if api_key != local_api_key:
        return False, "API_KEY does not match"

    return True, "OK"


def validate_input(event: dict) -> dict:
    """Validates the input data from the event"""
    input_data = event["input"]

    is_valid, message = validate_api_key(input_data.get("api_key"))
    if not is_valid:
        return {"error": message}

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


def execute_workflow(
    positive_prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    cfg: int,
    denoise: int,
) -> dict:
    """Executes the inpainting workflow"""
    workflow = modify_workflow_dump(
        workflow_dump, positive_prompt, negative_prompt, seed, steps, cfg, denoise
    )
    server_address = "127.0.0.1:8188"
    client_id = str(uuid.uuid4())
    ws = websocket.WebSocket()
    ws.connect(f"ws://{server_address}/ws?clientId={client_id}")
    prompt_id = send_prompt_to_comfy(workflow, client_id)["prompt_id"]
    images = receive_generated_images(ws, prompt_id)
    ws.close()
    return images


def save_image_to_path(image_data: str | bytes, path: str) -> None:
    """Saves an image to a filepath"""
    if isinstance(image_data, str):  # base64
        image_bytes = base64.b64decode(image_data)
    else:  # binary
        image_bytes = image_data

    image = Image.open(io.BytesIO(image_bytes))
    image.save(path)


def send_prompt_to_comfy(prompt: str, client_id: str) -> dict:
    """Sends a prompt to the ComfyUI server and returns the prompt ID"""
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode("utf-8")
    req = request.Request("http://127.0.0.1:8188/prompt", data=data)
    return json.loads(request.urlopen(req).read())


def receive_generated_images(ws: websocket.WebSocket, prompt_id: str) -> dict:
    """Receives generated images from the ComfyUI server"""
    output_images = {}
    current_node = ""
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message["type"] == "executing":
                data = message["data"]
                if data["prompt_id"] == prompt_id:
                    if data["node"] is None:
                        break  # Execution is done
                    else:
                        current_node = data["node"]
                        print(f"Processing node: {current_node}")
        else:
            if current_node == "save_image_websocket_node":
                images_output = output_images.get(current_node, [])
                # Important: The first 8 bytes need to be skipped as they contain binary header info
                images_output.append(out[8:])
                output_images[current_node] = images_output

    return output_images


def modify_workflow_dump(
    workflow_dump: str,
    positive_prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    cfg: int,
    denoise: int,
) -> dict:
    """Sets up the inpainting workflow"""
    workflow = json.loads(workflow_dump)
    workflow["37"]["inputs"]["text"] = positive_prompt
    workflow["38"]["inputs"]["text"] = negative_prompt
    workflow["39"]["inputs"]["seed"] = seed
    workflow["39"]["inputs"]["steps"] = steps
    workflow["39"]["inputs"]["cfg"] = cfg
    workflow["39"]["inputs"]["denoise"] = denoise
    return workflow


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
