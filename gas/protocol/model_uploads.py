import hashlib
import json
import re
from pathlib import Path
from typing import Optional

import bittensor as bt
import requests

from gas.protocol.epistula import generate_header


def calculate_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def generate_presigned_url(
    wallet: bt.wallet, 
    upload_endpoint: str, 
    filename: str, 
    file_size: int, 
    file_hash: str, 
    content_type: Optional[str] = None,
    modality: Optional[str] = None
) -> dict:
    """Generate presigned upload URL from the API with optional modality parameter."""
    
    payload = {
        'filename': filename,
        'file_size': file_size,
        'expected_hash': file_hash,
    }
    if content_type:
        payload['content_type'] = content_type
    if modality:
        payload['modality'] = modality
    
    payload_json = json.dumps(payload, separators=(',', ':'))
    payload_bytes = payload_json.encode('utf-8')
    
    headers = generate_header(wallet.hotkey, payload_bytes)
    headers['Content-Type'] = 'application/json'
    
    try:
        presigned_endpoint = upload_endpoint.rstrip('/') + '/presigned'
        response = requests.post(
            presigned_endpoint,
            data=payload_bytes,
            headers=headers,
            timeout=30
        )

        try:
            result = response.json()
        except json.JSONDecodeError:
            result = {"error": "Invalid JSON response", "text": response.text}
        
        return {
            "status_code": response.status_code,
            "success": response.status_code == 200,
            "response": result
        }
        
    except requests.exceptions.RequestException as e:
        return {
            "status_code": 0,
            "success": False,
            "response": {"error": f"Request failed: {str(e)}"}
        }


def upload_to_r2(presigned_url: str, file_content: bytes, content_type: str = 'application/octet-stream') -> dict:
    """Upload file directly to R2 using presigned URL."""
    try:
        response = requests.put(
            presigned_url,
            data=file_content,
            headers={'Content-Type': content_type},
            timeout=300  # 5 minutes for large files
        )
        
        error_detail = None
        if response.status_code != 200:
            text = response.text or ""
            match = re.search(r'<Message>(.*?)</Message>', text) or \
                    re.search(r'<Code>(.*?)</Code>', text)
            error_detail = match.group(1) if match else (text[:200] or "Upload failed")

        return {
            "status_code": response.status_code,
            "success": response.status_code == 200,
            "response": {
                "message": "Upload successful" if response.status_code == 200 else error_detail,
                "etag": response.headers.get('ETag', ''),
            }
        }
        
    except requests.exceptions.RequestException as e:
        return {
            "status_code": 0,
            "success": False,
            "response": {"error": f"Upload failed: {str(e)}"}
        }


def confirm_upload(wallet: bt.wallet, upload_endpoint: str, model_id: int, file_hash: str) -> dict:
    """Confirm file upload and finalize model record."""
    
    payload = {
        'model_id': model_id,
        'file_hash': file_hash
    }
    
    payload_json = json.dumps(payload, separators=(',', ':'))
    payload_bytes = payload_json.encode('utf-8')
    
    headers = generate_header(wallet.hotkey, payload_bytes)
    headers['Content-Type'] = 'application/json'
    
    try:
        confirm_endpoint = upload_endpoint.rstrip('/') + '/confirm'
        response = requests.post(
            confirm_endpoint,
            data=payload_bytes,
            headers=headers,
            timeout=30
        )
        
        try:
            result = response.json()
        except json.JSONDecodeError:
            result = {"error": "Invalid JSON response", "text": response.text}
        
        return {
            "status_code": response.status_code,
            "success": response.status_code == 200,
            "response": result
        }
        
    except requests.exceptions.RequestException as e:
        return {
            "status_code": 0,
            "success": False,
            "response": {"error": f"Request failed: {str(e)}"}
        }


def upload_single_modality(
    wallet: bt.wallet,
    file_path: str,
    modality: str,
    upload_endpoint: str
) -> dict:
    """Upload a single modality file (image or video model)."""
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path_obj, 'rb') as f:
        file_content = f.read()

    file_hash = calculate_sha256(file_content)
    file_size = len(file_content)
    filename = file_path_obj.name

    print(f"  File: {filename} ({file_size / 1024 / 1024:.2f} MB)")
    print(f"  Hash: {file_hash}")

    def extract_error(result: dict) -> str:
        resp = result.get('response', {})
        msg = resp.get('detail') or resp.get('error') or resp.get('message') or str(resp)
        status = result.get('status_code', 0)
        return f"HTTP {status}: {msg}" if status else str(msg)

    print(f"  [1/3] Requesting presigned URL...", end=' ', flush=True)
    presigned_result = generate_presigned_url(
        wallet,
        upload_endpoint,
        filename,
        file_size,
        file_hash,
        'application/octet-stream',
        modality
    )

    if not presigned_result['success']:
        print("FAILED")
        return {
            "success": False,
            "modality": modality,
            "step": "presigned_url_generation",
            "error": extract_error(presigned_result),
            "response": presigned_result['response']
        }
    print("done")

    presigned_data = presigned_result['response']['data']
    model_id = presigned_data['model_id']
    presigned_url = presigned_data['presigned_url']
    r2_key = presigned_data['r2_key']
    submissions_used = presigned_data.get('submissions_used')
    submissions_max = presigned_data.get('submissions_max')

    print(f"  [2/3] Uploading to R2...", end=' ', flush=True)
    upload_result = upload_to_r2(presigned_url, file_content, 'application/octet-stream')

    if not upload_result['success']:
        print("FAILED")
        return {
            "success": False,
            "modality": modality,
            "step": "r2_upload",
            "model_id": model_id,
            "error": extract_error(upload_result),
            "response": upload_result['response']
        }
    print("done")

    print(f"  [3/3] Confirming upload...", end=' ', flush=True)
    confirm_result = confirm_upload(wallet, upload_endpoint, model_id, file_hash)

    if not confirm_result['success']:
        print("FAILED")
        return {
            "success": False,
            "modality": modality,
            "step": "upload_confirmation",
            "model_id": model_id,
            "error": extract_error(confirm_result),
            "response": confirm_result['response']
        }
    print("done")

    return {
        "success": True,
        "modality": modality,
        "model_id": model_id,
        "r2_key": r2_key,
        "file_hash": file_hash,
        "file_size": file_size,
        "submissions_used": submissions_used,
        "submissions_max": submissions_max,
    } 