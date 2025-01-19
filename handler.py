import runpod
import os
import base64
from comfy_serverless import execute_workflow, save_image_to_path, validate_input


def handler(event: dict) -> dict:
    """Handler for the RunPod serverless API"""
    input_data = validate_input(event)

    input_dir = "ComfyUI/input"
    image_path = os.path.join(input_dir, "tmp.png")
    mask_path = os.path.join(input_dir, "tmp-mask.png")

    try:
        save_image_to_path(input_data.get("image"), image_path)
        save_image_to_path(input_data.get("mask"), mask_path)

        images = execute_workflow(
            input_data.get("positive_prompt"),
            input_data.get("negative_prompt"),
            input_data.get("seed"),
            input_data.get("steps"),
            input_data.get("cfg"),
            input_data.get("denoise"),
        )
        
        output_images = {}
        for node_id, image_list in images.items():
            output_images[node_id] = [base64.b64encode(img).decode('utf-8') for img in image_list]

        return {"images": output_images}

    except Exception as e:
        print("Error executing inpainting workflow:", e)
        return {"error": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
