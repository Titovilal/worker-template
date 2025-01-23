# ComfyUI Inpainting Serverless Endpoint

A RunPod serverless endpoint for image inpainting using ComfyUI and the Juggernaut XL model.

Note: The project includes a `ComfyUI/extra_model_paths.yaml` file configured to use `runpod-volume` as the models' location, enabling the use of RunPod network volumes for model storage.

## Features

- Image inpainting using ComfyUI workflows
- Juggernaut XL inpainting model
- Serverless deployment on RunPod
- WebSocket-based image generation
- Customizable parameters for inpainting

## Prerequisites

- RunPod account
- Docker installed (for local testing)
- Python

## Project Structure

```
.
├── ComfyUI/               # ComfyUI installation
├── Dockerfile             # Container configuration
├── handler.py            # Main serverless handler
├── requirements.txt      # Python dependencies
└── test_input.json      # Sample input for testing
```

## Deployment

1. Build the Docker image:
```bash
docker build -t your-registry/comfyui-inpainting:latest .
```

2. Push to your container registry:
```bash
docker push your-registry/comfyui-inpainting:latest
```

3. Create a new RunPod serverless endpoint using your image

## API Usage

Send POST requests to your endpoint with the following JSON structure:

```json
{
    "input": {
        "image": "base64_encoded_image",
        "mask": "base64_encoded_mask",
        "positive_prompt": "your positive prompt",
        "negative_prompt": "your negative prompt",
        "seed": 123456789,
        "steps": 30,
        "cfg": 8,
        "denoise": 1.0
    }
}
```

### Parameters

- `image`: Base64 encoded input image
- `mask`: Base64 encoded mask image
- `positive_prompt`: Prompt describing desired outcome
- `negative_prompt`: Prompt describing elements to avoid
- `seed`: Random seed for generation
- `steps`: Number of sampling steps
- `cfg`: Classifier-free guidance scale
- `denoise`: Denoising strength (0.0 to 1.0)

### Response

```json
{
    "images": {
        "save_image_websocket_node": [
            "base64_encoded_output_image"
        ]
    }
}
```

## Local Testing

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start ComfyUI:
```bash
python ComfyUI/main.py
```

3. Run the handler in a separate terminal:
```bash
python handler.py
```

## Environment Variables

- None required for basic operation
- ComfyUI runs on port 8188 internally

## How the Handler Works

The handler processes requests in the following steps:

1. Receives input data containing the image, mask, and generation parameters
2. Saves the base64-encoded images as temporary files in `ComfyUI/input`
3. Sets up the inpainting workflow with user parameters (prompts, seed, steps, etc.)
4. Connects to ComfyUI via WebSocket
5. Sends the workflow to ComfyUI and waits for image generation
6. Receives the generated image through WebSocket
7. Encodes the result in base64 and returns it

The workflow uses the Juggernaut XL model specifically trained for inpainting, with a predefined node structure that handles:
- Image loading and masking
- Prompt encoding
- Inpainting generation
- Result stitching and output
