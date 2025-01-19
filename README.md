# ComfyUI Inpainting Worker

A RunPod serverless worker that provides AI-powered inpainting capabilities using ComfyUI and the Juggernaut XL model.

## Overview

This worker provides an API endpoint for image inpainting tasks. It uses ComfyUI as the backend and the Juggernaut XL model for high-quality inpainting results. The project includes a complete ComfyUI installation in the `/ComfyUI` directory, which runs alongside the serverless worker.

## Features

- Image inpainting with mask support
- Customizable parameters (steps, CFG, denoise, etc.)
- Custom prompt support
- Base64 image input/output
- Secure API key authentication
- Built-in ComfyUI server

## Requirements

- NVIDIA GPU with CUDA support
- Docker
- RunPod account (for deployment)
- Juggernaut XL Inpainting model

## Setup

1. Clone this repository:

```bash
git clone <repository-url>
cd worker-template
```

2. Download the Juggernaut XL Inpainting model:
   - Download the model from [Civitai](https://civitai.com/models/133005)
   - Create a directory: `mkdir -p ComfyUI/models/checkpoints/`
   - Place the downloaded model `juggernaut-xl-inpainting.safetensors` in `ComfyUI/models/checkpoints/`

3. Create a `.env` file with your API key:

```bash
API_KEY=your_secret_key_here
```

4. Build the Docker image:

```bash
docker build -t your-image-name .
```

### ComfyUI Server

The worker runs a ComfyUI server internally on port 8188. When deployed, this server is only accessible within the container. The server:
- Loads automatically when the container starts
- Handles the inpainting workflow
- Communicates with the worker through WebSocket

## API Usage

The worker accepts POST requests with the following JSON structure:

```json
{
  "input": {
    "api_key": "your_api_key", // set in env variable
    "image": "base64_encoded_image",
    "mask": "base64_encoded_mask",
    "positive_prompt": "custom positive prompt", // optional
    "negative_prompt": "custom negative prompt", // optional
    "seed": 123456,  // optional
    "steps": 20,     // optional
    "cfg": 8,        // optional
    "denoise": 1     // optional
  }
}
```

### Parameters

> The optional parameters are have default values hardcoded in the `comfy_serverless.py` file.

- `api_key`: Your authentication key
- `image`: Base64 encoded image to be inpainted
- `mask`: Base64 encoded mask (white areas will be inpainted)
- `positive_prompt`: Text describing what to generate (default: "closeup, portrait, epic, movie scene, naked, nude")
- `negative_prompt`: Text describing what to avoid (default: "watermark, ugly, clothes")
- `seed`: Random seed for generation (optional)
- `steps`: Number of sampling steps (optional, default: 20)
- `cfg`: Classifier free guidance scale (optional, default: 8)
- `denoise`: Denoising strength (optional, default: 1)

### Response

The API returns a JSON object with the generated images in base64 format:

```json
{
  "images": {
    "save_image_websocket_node": [
      "base64_encoded_image"
    ]
  }
}
```

## Development

The project consists of several key components:

- `handler.py`: Main RunPod serverless handler
- `comfy_serverless.py`: ComfyUI workflow management and execution
- `Dockerfile`: Container configuration
- `requirements.txt`: Python dependencies

## License

[Your license here]

## Credits

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [RunPod](https://www.runpod.io/)
- [Juggernaut XL Model](https://civitai.com/models/133005)
