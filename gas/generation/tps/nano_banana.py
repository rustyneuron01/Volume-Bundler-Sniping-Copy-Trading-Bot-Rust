import requests
import base64
import os
import time
import io
from PIL import Image
from gas.types import Modality, MediaType


def generate_image(prompt: str, model: str = "google/gemini-2.5-flash-image-preview:free"):
    """
    Generates an image based on the given prompt using an API call.
    Raises an exception if the API key is not present.

    Example response structure:

        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "I've generated a beautiful sunset image for you.",
                    "images": [
                        {
                            "type": "image_url",
                            "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
                            }
                        }
                    ]
                }
            }
        ]


    Args:
        prompt (str): The text prompt to generate the image.
        model (str): The model to use for image generation. Defaults to a specific model.

    Returns:
        Dict shaped like GenerationPipeline output or None on failure, e.g.:
        {
            "image": PIL.Image.Image,
            "modality": Modality.IMAGE,
            "media_type": MediaType.SYNTHETIC,
            "prompt": str,
            "model_name": str,
            "time": float,
            "gen_duration": float,
            "gen_args": dict
        }
    """
    API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
    if not API_KEY:
        raise RuntimeError("API key is not set. Please set the 'OPEN_ROUTER_API_KEY' environment variable in .env.validator.")

    URL = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "modalities": ["image", "text"]
    }
    
    start_time = time.time()
    response = requests.post(URL, headers=headers, json=payload)

    if response.status_code != 200:
        return None

    result = response.json()

    if 'choices' not in result or not result['choices']:
        return None

    choice = result['choices'][0]
    if 'message' not in choice or 'images' not in choice['message']:
        return None

    images = choice['message']['images']
    if not images:
        return None

    # Use the first image
    image_url = images[0]['image_url']['url']
    # Format: "data:image/<fmt>;base64,<data>"
    if ',' not in image_url:
        return None
    base64_data = image_url.split(',')[1]
    image_binary = base64.b64decode(base64_data)
    pil_image = Image.open(io.BytesIO(image_binary))

    gen_time = time.time() - start_time

    output = {
        "image": pil_image,
        "modality": Modality.IMAGE,
        "media_type": MediaType.SYNTHETIC,
        "prompt": prompt,
        "model_name": model,
        "time": time.time(),
        "gen_duration": gen_time,
        "gen_args": {
            "provider": "openrouter",
            "model": model,
            "modalities": ["image", "text"],
        },
    }

    return output
