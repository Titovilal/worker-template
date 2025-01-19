import json
from urllib import request
import websocket
import uuid
import io
from PIL import Image

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


def send_prompt_to_comfy(prompt: str, client_id: str) -> dict:
    """Sends a prompt to the ComfyUI server and returns the prompt ID"""
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode("utf-8")
    req = request.Request("http://127.0.0.1:8188/prompt", data=data)
    return json.loads(request.urlopen(req).read())


def receive_generated_images(
    ws: websocket.WebSocket, prompt: str, client_id: str
) -> dict:
    """Receives generated images from the ComfyUI server"""
    prompt_id = send_prompt_to_comfy(prompt, client_id)["prompt_id"]
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


def setup_inpainting_workflow(
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


def convert_binary_to_pil_image(images: dict) -> Image.Image | None:
    """Converts binary images to PIL images"""
    for node_id in images:
        for image_data in images[node_id]:
            return Image.open(io.BytesIO(image_data))
    return None


def execute_inpainting_workflow(
    positive_prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    cfg: int,
    denoise: int,
) -> dict:
    """Executes the inpainting workflow"""
    workflow = setup_inpainting_workflow(
        workflow_dump, positive_prompt, negative_prompt, seed, steps, cfg, denoise
    )
    server_address = "127.0.0.1:8188"
    client_id = str(uuid.uuid4())
    ws = websocket.WebSocket()
    ws.connect(f"ws://{server_address}/ws?clientId={client_id}")
    images = receive_generated_images(ws, workflow, client_id)
    ws.close()
    return images
