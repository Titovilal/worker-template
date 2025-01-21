import runpod
import os
import base64
import orjson
from urllib import request
import websocket
import uuid
import io
from PIL import Image

# -------------------------------------------------
# region Main
# -------------------------------------------------


def handler(event: dict) -> dict:
    """Handler for the RunPod serverless API"""

    # 1. Get input data
    input_data = event["input"]

    # 2. Prepare paths for input images
    input_dir = "ComfyUI/input"
    image_path = os.path.join(input_dir, "tmp.png")
    mask_path = os.path.join(input_dir, "tmp-mask.png")

    try:
        # 3. Save input images
        save_image(input_data.get("image"), image_path)
        save_image(input_data.get("mask"), mask_path)

        # 4. Clean dictionary of input data
        input_data["image"] = ""
        input_data["mask"] = ""

        # 5. Setup inpainting workflow
        workflow = setup_inpainting_workflow(workflow_dump, input_data)

        # 6. Connect to ComfyUI server
        server_address = "127.0.0.1:8188"
        client_id = str(uuid.uuid4())
        ws = websocket.WebSocket()
        ws.connect(f"ws://{server_address}/ws?clientId={client_id}")

        # 7. Send prompt to ComfyUI server
        prompt_id = send_prompt_to_comfy(workflow, client_id)["prompt_id"]

        # 8. Receive generated images
        images = receive_generated_images(ws, prompt_id)

        # 9. Close connection to ComfyUI server
        ws.close()

        # 10. Convert binary images to PIL images
        output_images = {}
        for node_id, image_list in images.items():
            output_images[node_id] = [
                base64.b64encode(img).decode("utf-8") for img in image_list
            ]

        # 11. Return output images
        return {"images": output_images}

    except Exception as e:
        print("Error executing inpainting workflow:", e)
        return {"error": str(e)}


# -------------------------------------------------
# region ComfyUI
# -------------------------------------------------


def save_image(image_data: str | bytes, path: str) -> None:
    """Saves an image to a filepath"""
    if isinstance(image_data, str):  # base64
        image_bytes = base64.b64decode(image_data)
    else:  # binary
        image_bytes = image_data

    image = Image.open(io.BytesIO(image_bytes))
    image.save(path)


def receive_generated_images(ws: websocket.WebSocket, prompt_id: str) -> dict:
    """Receives generated images from the ComfyUI server"""
    output_images = {}
    current_node = ""
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = orjson.loads(out)
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


def send_prompt_to_comfy(prompt: str, client_id: str) -> dict:
    """Sends a prompt to the ComfyUI server and returns the prompt ID"""
    p = {"prompt": prompt, "client_id": client_id}
    data = orjson.dumps(p)
    req = request.Request("http://127.0.0.1:8188/prompt", data=data)
    return orjson.loads(request.urlopen(req).read())


def setup_inpainting_workflow(
    workflow_dump: str,
    input_data: dict,
) -> dict:
    """Sets up the inpainting workflow"""
    workflow = orjson.loads(workflow_dump)
    workflow["37"]["inputs"]["text"] = input_data["positive_prompt"]
    workflow["38"]["inputs"]["text"] = input_data["negative_prompt"]
    workflow["39"]["inputs"]["seed"] = input_data["seed"]
    workflow["39"]["inputs"]["steps"] = input_data["steps"]
    workflow["39"]["inputs"]["cfg"] = input_data["cfg"]
    workflow["39"]["inputs"]["denoise"] = input_data["denoise"]
    return workflow


def convert_binary_to_pil_image(images: dict) -> Image.Image | None:
    """Converts binary images to PIL images"""
    for node_id in images:
        for image_data in images[node_id]:
            return Image.open(io.BytesIO(image_data))
    return None


# -------------------------------------------------
# region Workflow
# -------------------------------------------------

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


# -------------------------------------------------
# region Runner
# -------------------------------------------------


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
