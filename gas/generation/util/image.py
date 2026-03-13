import numpy as np
import PIL
import os
from PIL import Image, ImageDraw
from typing import Tuple, Union, List


def resize_image(
    image: PIL.Image.Image, max_width: int, max_height: int
) -> PIL.Image.Image:
    """Resize the image to fit within specified dimensions while maintaining aspect ratio."""
    original_width, original_height = image.size

    # Calculate the aspect ratio and determine new dimensions
    aspect_ratio = original_width / original_height
    new_width = min(max_width, original_width)
    new_height = int(new_width / aspect_ratio)

    if new_height > max_height:
        new_height = max_height
        new_width = int(new_height * aspect_ratio)

    # Resize the image using the high-quality LANCZOS filter
    resized_image = image.resize((new_width, new_height), PIL.Image.LANCZOS)
    return resized_image


def resize_images_in_directory(directory, target_width, target_height):
    """
    Resize all images in the specified directory to the target width and height.

    Args:
    directory (str): Path to the directory containing images.
    target_width (int): Target width for resizing the images.
    target_height (int): Target height for resizing the images.
    """
    # List all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(
            (".png", ".jpg", ".jpeg", ".bmp", ".gif")
        ):  # Check for image file extensions
            filepath = os.path.join(directory, filename)
            with PIL.Image.open(filepath) as img:
                # Resize the image and save back to the file location
                resized_img = resize_image(
                    img, max_width=target_width, max_height=target_height
                )
                resized_img.save(filepath)


def save_images_to_disk(
    image_dataset, start_index, num_images, save_directory, resize=True
):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    for i in range(start_index, start_index + num_images):
        try:
            image_data = image_dataset[i]  # Retrieve image using the __getitem__ method
            image = image_data["image"]  # Extract the image
            image_id = image_data["id"]  # Extract the image ID
            file_path = os.path.join(
                save_directory, f"{image_id}.jpg"
            )  # Construct file path
            # if resize:
            #    image = resize_image(image, TARGET_IMAGE_SIZE[0], TARGET_IMAGE_SIZE[1])
            image.save(file_path, "JPEG")  # Save the image
            print(f"Saved: {file_path}")
        except Exception as e:
            print(f"Failed to save image {i}: {e}")


def ensure_mask_3d(mask: np.ndarray) -> np.ndarray:
    """
    Ensure the mask is 3D (H, W, 1) if it's 2D (H, W).
    """
    if mask.ndim == 2:
        return mask[:, :, None]
    return mask


def create_random_mask(
    size: Tuple[int, int],
    min_size_ratio: float = 0.15,
    max_size_ratio: float = 0.5,
    allow_multiple: bool = True,
    allowed_shapes: list = ["rectangle", "circle", "ellipse", "triangle"],
) -> "Image.Image":
    """
    Create a random mask (or masks) for i2i/inpainting with more variety.
    Returns a single-channel ("L" mode) mask image.
    """
    w, h = size
    allowed_shapes = [s for s in allowed_shapes]
    max_retries = 5
    for attempt in range(max_retries):
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        n_masks = np.random.randint(1, 5) if allow_multiple else 1
        for _ in range(n_masks):
            shape = np.random.choice(allowed_shapes)
            min_dim = min(w, h)
            min_pixel_size = 64
            min_mask_size = max(int(min_size_ratio * min_dim), min_pixel_size)
            max_mask_size = max(int(max_size_ratio * min_dim), min_pixel_size)
            if min_mask_size >= max_mask_size:
                width = min_mask_size
                height = min_mask_size
            else:
                width = np.random.randint(min_mask_size, max_mask_size)
                height = np.random.randint(min_mask_size, max_mask_size)
            width = min(width, w)
            height = min(height, h)
            if shape == "circle":
                r = min(width, height) // 2
                if r < 1:
                    r = 1
                cx = np.random.randint(r, w - r + 1)
                cy = np.random.randint(r, h - r + 1)
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
            elif shape == "rectangle":
                x = np.random.randint(0, w - width + 1)
                y = np.random.randint(0, h - height + 1)
                draw.rectangle([x, y, x + width, y + height], fill=255)
            elif shape == "ellipse":
                x = np.random.randint(0, w - width + 1)
                y = np.random.randint(0, h - height + 1)
                x0, y0, x1, y1 = x, y, x + width, y + height
                draw.ellipse([x0, y0, x1, y1], fill=255)
            elif shape == "triangle":
                min_triangle_size = max(96, min_mask_size)
                max_triangle_size = max(128, max_mask_size)
                min_triangle_size = min(min_triangle_size, w, h)
                max_triangle_size = min(max_triangle_size, w, h)
                if min_triangle_size >= max_triangle_size:
                    width = max_triangle_size
                    height = max_triangle_size
                else:
                    width = np.random.randint(min_triangle_size, max_triangle_size)
                    height = np.random.randint(min_triangle_size, max_triangle_size)
                x = np.random.randint(0, w - width + 1)
                y = np.random.randint(0, h - height + 1)
                jitter = lambda v, maxv: max(
                    0,
                    min(v + np.random.randint(-width // 10, width // 10 + 1), maxv - 1),
                )
                pt1 = (jitter(x, w), jitter(y, h))
                pt2 = (jitter(x + width - 1, w), jitter(y, h))
                pt3 = (jitter(x, w), jitter(y + height - 1, h))
                pts = [pt1, pt2, pt3]
                draw.polygon(pts, fill=255)
        if np.array(mask).max() > 0:
            return mask
    return mask


def is_black_output(
    modality: str, output: Union[List[Image.Image], Image.Image], threshold: int = 10
) -> bool:
    """
    Returns True if the image or frames are (almost) completely black.
    """
    try:
        if modality == "image":
            # Handle different image output formats
            if hasattr(output, "images"):
                arr = np.array(output.images[0])
            elif isinstance(output, Image.Image):
                arr = np.array(output)
            else:
                arr = np.array(output)
            return np.mean(arr) < threshold
        elif modality == "video":
            # Handle different video output formats
            if hasattr(output, "frames"):
                frames = output.frames[0]
            else:
                frames = output

            # Check first few frames to avoid processing all frames unnecessarily
            sample_frames = frames[: min(5, len(frames))] if len(frames) > 5 else frames

            for frame in sample_frames:
                try:
                    # Convert frame to numpy array if it's a PIL Image
                    if hasattr(frame, "mode"):  # PIL Image
                        arr = np.array(frame)
                    elif hasattr(frame, "numpy"):  # Tensor
                        arr = (
                            frame.numpy()
                            if hasattr(frame, "numpy")
                            else np.array(frame)
                        )
                    else:
                        arr = np.array(frame)

                    # Ensure we have numeric data
                    if arr.dtype.kind not in [
                        "i",
                        "u",
                        "f",
                    ]:  # not integer, unsigned, or float
                        continue  # Skip non-numeric frames

                    # If any frame has reasonable brightness, it's not black
                    if np.mean(arr) >= threshold:
                        return False
                except Exception:
                    # If we can't process a frame, assume it's not black
                    continue

            return True  # All processable frames are black
    except Exception:
        # If we can't analyze the output, assume it's not black to avoid false positives
        return False
