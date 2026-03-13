import hashlib
import os
import zipfile
from pathlib import Path


def calculate_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_onnx_directory(onnx_dir: str) -> bool:
    """Validate that the directory contains all required ONNX files"""
    
    required_files = [
        "image_detector.onnx",
        "video_detector.onnx",
    ]

    if not os.path.exists(onnx_dir):
        print(f"Error: Directory does not exist: {onnx_dir}")
        return False

    if not os.path.isdir(onnx_dir):
        print(f"Error: Path is not a directory: {onnx_dir}")
        return False

    missing_files = []
    for filename in required_files:
        file_path = os.path.join(onnx_dir, filename)
        if not os.path.exists(file_path):
            missing_files.append(filename)

    if missing_files:
        print(f"Error: Missing required ONNX files: {missing_files}")
        print(f"Expected files in {onnx_dir}:")
        for filename in required_files:
            print(f"  - {filename}")
        return False

    print(f"✅ Found all required ONNX files in {onnx_dir}:")
    for filename in required_files:
        file_path = os.path.join(onnx_dir, filename)
        file_size = os.path.getsize(file_path)
        print(f"  - {filename} ({file_size} bytes)")

    return True


def create_model_zip(onnx_dir):
    """Create zip file containing multiple ONNX models

    Args:
        onnx_dir: Path to directory containing ONNX model files

    Raises:
        FileNotFoundError: If ONNX directory or files do not exist
        ValueError: If required files missing
    """
    
    if not os.path.exists(onnx_dir):
        raise FileNotFoundError(f"ONNX directory not found: {onnx_dir}")

    required_files = [
        "image_detector.onnx",
        "video_detector.onnx",
    ]
    missing_files = []
    for filename in required_files:
        file_path = os.path.join(onnx_dir, filename)
        if not os.path.exists(file_path):
            missing_files.append(filename)

    if missing_files:
        raise FileNotFoundError(f"Missing required ONNX files: {missing_files}")

    zip_path = os.path.join(onnx_dir, "models.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for filename in required_files:
            file_path = os.path.join(onnx_dir, filename)
            zipf.write(file_path, filename)

    return zip_path


 