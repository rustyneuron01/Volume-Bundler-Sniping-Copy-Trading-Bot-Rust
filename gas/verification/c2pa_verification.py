"""
C2PA Content Credentials verification for miner-submitted media.

Validates that content has authentic provenance from trusted AI generators
like OpenAI (DALL-E), Google (Gemini/Imagen), Adobe Firefly, etc.
"""

import json
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

import bittensor as bt

import c2pa


TRUSTED_CERT_ISSUERS = {
    "OpenAI",           # DALL-E, Sora
    "Google",           # Imagen, Gemini, Veo
    "Adobe",            # Firefly
    "Microsoft",        # Copilot, Designer
    "Meta",             # Imagine
    "Runway",           # Gen-3
    "Stability AI",     # Stable Diffusion, Stable Video
    "Black Forest Labs",  # FLUX
    "Midjourney",
    "Pika",             # Pika Labs
    "Luma",             # Dream Machine
    "Ideogram",
    "Leonardo",         # Leonardo.AI
    "Kuaishou", "Kling",  # Kling AI
    "MiniMax",
    "Haiper",
    "Lightricks",       # LTX Studio
    "Shutterstock",
    "ByteDance",        # Seedream, Seedance
}

TRUSTED_CA_ISSUERS = {
    # Traditional Certificate Authorities
    "DigiCert", "GlobalSign", "Entrust", "Sectigo", "Let's Encrypt",
    "Amazon", "Google", "Microsoft", "Apple", "Cloudflare",
    "Comodo", "GeoTrust", "Thawte", "VeriSign", "GoDaddy",
    "IdenTrust", "Symantec", "QuoVadis", "SwissSign",
    # AI companies (may appear as cert issuers in C2PA manifests)
    *TRUSTED_CERT_ISSUERS,
}

AI_SOURCE_TYPES = {
    "trainedAlgorithmicMedia",
    "compositeWithTrainedAlgorithmicMedia",
    "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
    "http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia",
}


class C2PAVerificationResult:
    def __init__(
        self,
        verified: bool = False,
        signature_valid: bool = False,
        issuer: Optional[str] = None,
        cert_issuer: Optional[str] = None,
        is_trusted_issuer: bool = False,
        is_self_signed: bool = False,
        ai_generated: bool = False,
        model_name: Optional[str] = None,
        manifest_data: Optional[Dict[str, Any]] = None,
        validation_errors: Optional[List[str]] = None,
        error: Optional[str] = None,
    ):
        self.verified = verified
        self.signature_valid = signature_valid
        self.issuer = issuer
        self.cert_issuer = cert_issuer
        self.is_trusted_issuer = is_trusted_issuer
        self.is_self_signed = is_self_signed
        self.ai_generated = ai_generated
        self.model_name = model_name
        self.manifest_data = manifest_data or {}
        self.validation_errors = validation_errors or []
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verified": self.verified,
            "signature_valid": self.signature_valid,
            "issuer": self.issuer,
            "cert_issuer": self.cert_issuer,
            "is_trusted_issuer": self.is_trusted_issuer,
            "is_self_signed": self.is_self_signed,
            "ai_generated": self.ai_generated,
            "model_name": self.model_name,
            "validation_errors": self.validation_errors,
            "error": self.error,
        }


def verify_c2pa(media_data: Union[bytes, str, Path]) -> C2PAVerificationResult:
    """Verify C2PA credentials with full cryptographic validation."""
    temp_file = None
    try:
        # Always read bytes to detect actual format (file extension might be wrong)
        if isinstance(media_data, bytes):
            data = media_data
        else:
            data = Path(media_data).read_bytes()

        # Create temp file with correct extension based on magic bytes
        suffix = detect_media_format(data)
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        temp_file.write(data)
        temp_file.close()
        file_path = temp_file.name

        try:
            with c2pa.Reader(file_path) as reader:
                manifest_json = reader.json()
        except Exception as e:
            return C2PAVerificationResult(verified=False, error=f"No C2PA manifest found: {str(e)}")

        if not manifest_json:
            return C2PAVerificationResult(verified=False, error="Empty C2PA manifest")

        try:
            manifest_data = json.loads(manifest_json) if isinstance(manifest_json, str) else manifest_json
        except json.JSONDecodeError as e:
            return C2PAVerificationResult(verified=False, error=f"Invalid manifest JSON: {e}")

        validation_errors = _check_validation_status(manifest_data)
        if validation_errors:
            bt.logging.warning(f"C2PA signature validation failed: {validation_errors}")
            return C2PAVerificationResult(
                verified=False,
                signature_valid=False,
                validation_errors=validation_errors,
                error=f"Signature validation failed: {'; '.join(validation_errors)}"
            )

        cert_info = _extract_certificate_info(manifest_data)
        is_self_signed = cert_info.get("is_self_signed", True)
        cert_issuer = cert_info.get("cert_issuer")
        cert_subject = cert_info.get("cert_subject")
        cert_chain_length = cert_info.get("cert_chain_length", 0)

        bt.logging.debug(
            f"C2PA cert info: issuer={cert_issuer}, subject={cert_subject}, "
            f"chain_len={cert_chain_length}, is_self_signed={is_self_signed}"
        )

        if is_self_signed:
            bt.logging.warning(f"C2PA rejected: self-signed certificate (subject: {cert_subject})")
            return C2PAVerificationResult(
                verified=False,
                signature_valid=True,
                is_self_signed=True,
                cert_issuer=cert_issuer,
                error="Self-signed certificates are not accepted"
            )

        if not _is_trusted_ca(cert_issuer):
            bt.logging.warning(f"C2PA rejected: untrusted CA: {cert_issuer}")
            return C2PAVerificationResult(
                verified=False,
                signature_valid=True,
                is_self_signed=False,
                cert_issuer=cert_issuer,
                error=f"Certificate issuer not from trusted CA: {cert_issuer}"
            )

        claim_generator = _extract_claim_generator(manifest_data)
        is_trusted = _is_trusted_issuer(cert_subject, cert_issuer, claim_generator)

        if not is_trusted:
            bt.logging.warning(
                f"C2PA rejected: untrusted generator. Subject: {cert_subject}, Issuer: {cert_issuer}, Claim: {claim_generator}"
            )
            return C2PAVerificationResult(
                verified=False,
                signature_valid=True,
                is_self_signed=False,
                issuer=claim_generator,
                cert_issuer=cert_issuer,
                is_trusted_issuer=False,
                error=f"Not from a trusted AI generator: {cert_subject or claim_generator}"
            )

        ai_generated = _check_ai_generated(manifest_data)
        model_name = _extract_model_name(manifest_data)

        bt.logging.info(
            f"C2PA verified: issuer={claim_generator}, model_name={model_name}, "
            f"cert_subject={cert_subject}, cert_issuer={cert_issuer}, ai_generated={ai_generated}"
        )

        return C2PAVerificationResult(
            verified=True,
            signature_valid=True,
            issuer=claim_generator,
            cert_issuer=cert_issuer,
            is_trusted_issuer=True,
            is_self_signed=False,
            ai_generated=ai_generated,
            model_name=model_name,
            manifest_data={"raw": manifest_json},
        )

    except Exception as e:
        bt.logging.warning(f"C2PA verification error: {e}")
        return C2PAVerificationResult(verified=False, error=str(e))
    finally:
        if temp_file:
            try:
                Path(temp_file.name).unlink(missing_ok=True)
            except Exception:
                pass


def detect_media_format(data: bytes) -> str:
    """
    Detect media format from magic bytes.

    Supports:
        - Images: PNG, JPEG, WebP, GIF, BMP, TIFF, HEIC/HEIF, AVIF
        - Video: MP4/M4V/MOV (ftyp-based), WebM, AVI, MKV
        - Audio: MP3, WAV, FLAC, OGG, AAC/M4A

    Args:
        data: Raw bytes of the media file (at least first 12 bytes recommended)

    Returns:
        File extension string (e.g., ".png", ".mp4") or ".bin" if unknown
    """
    if len(data) < 4:
        return ".bin"

    # === Images ===

    # PNG: 89 50 4E 47 0D 0A 1A 0A
    if len(data) >= 8 and data[:8] == b'\x89PNG\r\n\x1a\n':
        return ".png"

    # JPEG: FF D8 FF
    if data[:3] == b'\xff\xd8\xff':
        return ".jpg"

    # WebP: RIFF....WEBP
    if len(data) >= 12 and data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return ".webp"

    # GIF: GIF87a or GIF89a
    if data[:6] in (b'GIF87a', b'GIF89a'):
        return ".gif"

    # BMP: 42 4D (BM)
    if data[:2] == b'BM':
        return ".bmp"

    # TIFF: 49 49 2A 00 (little-endian) or 4D 4D 00 2A (big-endian)
    if data[:4] in (b'II*\x00', b'MM\x00*'):
        return ".tiff"

    # === ftyp-based formats (ISO Base Media File Format) ===
    # MP4, M4A, MOV, 3GP, HEIC, AVIF all use ftyp box at offset 4
    # Must check all brands in a single block to avoid unreachable code
    if len(data) >= 12 and data[4:8] == b'ftyp':
        brand = data[8:12]

        # Images: HEIC/HEIF
        if brand in (b'heic', b'heix', b'hevc', b'mif1', b'msf1'):
            return ".heic"
        if brand == b'avif':
            return ".avif"

        # Audio: M4A/AAC
        if brand in (b'M4A ', b'M4B '):
            return ".m4a"

        # Video: QuickTime MOV
        if brand == b'qt  ':
            return ".mov"

        # Video: 3GP/3G2
        if brand in (b'3gp4', b'3gp5', b'3gp6', b'3g2a'):
            return ".3gp"

        # Default: MP4 for other ftyp-based formats (isom, mp41, mp42, etc.)
        return ".mp4"

    # WebM/MKV: EBML header 1A 45 DF A3
    if data[:4] == b'\x1a\x45\xdf\xa3':
        # Both WebM and MKV use this header; WebM is a subset of MKV
        # For C2PA purposes, treat as WebM (more common for web video)
        return ".webm"

    # AVI: RIFF....AVI
    if len(data) >= 12 and data[:4] == b'RIFF' and data[8:12] == b'AVI ':
        return ".avi"

    # === Audio ===

    # MP3: ID3 tag or frame sync (FF FB, FF FA, FF F3, FF F2)
    if data[:3] == b'ID3':
        return ".mp3"
    if data[:2] == b'\xff\xfb' or data[:2] == b'\xff\xfa':
        return ".mp3"
    if data[:2] == b'\xff\xf3' or data[:2] == b'\xff\xf2':
        return ".mp3"

    # WAV: RIFF....WAVE
    if len(data) >= 12 and data[:4] == b'RIFF' and data[8:12] == b'WAVE':
        return ".wav"

    # FLAC: 66 4C 61 43 (fLaC)
    if data[:4] == b'fLaC':
        return ".flac"

    # OGG: 4F 67 67 53 (OggS)
    if data[:4] == b'OggS':
        return ".ogg"

    return ".bin"


def _check_validation_status(manifest_data: Dict[str, Any]) -> List[str]:
    errors = []
    try:
        validation_status = manifest_data.get("validation_status", [])
        critical_codes = [
            "assertion.hashedURI.mismatch",
            "assertion.dataHash.mismatch",
            "claim.signature.mismatch",
            "signingCredential.invalid",
            "signingCredential.expired",
            "signingCredential.revoked",
            "signingCredential.untrusted",
            "manifest.inaccessible",
            "assertion.inaccessible",
        ]
        for status in validation_status:
            code = status.get("code", "")
            explanation = status.get("explanation", "")
            if any(critical in code for critical in critical_codes):
                errors.append(f"{code}: {explanation}")
            elif code and "error" in code.lower():
                errors.append(f"{code}: {explanation}")
    except Exception as e:
        bt.logging.debug(f"Error checking validation status: {e}")
        errors.append(f"Failed to parse validation status: {e}")
    return errors


def _extract_certificate_info(manifest_data: Dict[str, Any]) -> Dict[str, Any]:
    result = {
        "cert_issuer": None,
        "cert_subject": None,
        "is_self_signed": True,
        "cert_chain_length": 0,
    }
    try:
        manifests = manifest_data.get("manifests", {})
        active_manifest_id = manifest_data.get("active_manifest")

        manifest = None
        if active_manifest_id and active_manifest_id in manifests:
            manifest = manifests[active_manifest_id]
        elif manifests:
            manifest = next(iter(manifests.values()))

        if not manifest:
            return result

        signature_info = manifest.get("signature_info", {})
        cert_issuer = signature_info.get("issuer")
        result["cert_issuer"] = cert_issuer

        cert_chain = signature_info.get("cert_chain", [])
        result["cert_chain_length"] = len(cert_chain) if cert_chain else 0

        cert_subject = signature_info.get("subject")
        if not cert_subject:
            cert_serial = signature_info.get("cert_serial_number")
            if cert_serial:
                result["cert_subject"] = f"serial:{cert_serial}"
        else:
            result["cert_subject"] = cert_subject

        # Check if cert_issuer is from a trusted CA first
        if cert_issuer and _is_trusted_ca(cert_issuer):
            result["is_self_signed"] = False
        elif cert_issuer and result["cert_subject"]:
            # Only mark as self-signed if issuer == subject (actual self-signed)
            # Don't use cert_chain_length - many valid certs don't include full chain
            result["is_self_signed"] = cert_issuer.lower() == result["cert_subject"].lower()
        elif result["cert_chain_length"] > 1:
            result["is_self_signed"] = False
        # else: default is True (assume self-signed if we can't determine)

    except Exception as e:
        bt.logging.debug(f"Error extracting certificate info: {e}")

    return result


def _extract_claim_generator(manifest_data: Dict[str, Any]) -> Optional[str]:
    try:
        manifests = manifest_data.get("manifests", {})
        active_manifest_id = manifest_data.get("active_manifest")

        manifest = None
        if active_manifest_id and active_manifest_id in manifests:
            manifest = manifests[active_manifest_id]
        elif manifests:
            manifest = next(iter(manifests.values()))

        if not manifest:
            return None

        # Older c2pa-rs format: plain string
        claim_generator = manifest.get("claim_generator")
        if claim_generator:
            return claim_generator

        # Newer C2PA spec format (e.g. Google): structured list
        info_list = manifest.get("claim_generator_info", [])
        if info_list:
            name = info_list[0].get("name")
            if name:
                return name

        return None
    except Exception as e:
        bt.logging.debug(f"Error extracting claim generator: {e}")
        return None


def _extract_model_name(manifest_data: Dict[str, Any]) -> Optional[str]:
    """
    Extract a specific model name from a C2PA manifest.

    Only returns a value when the manifest contains an unambiguous model
    identifier — not generic library/infrastructure names or marketing text.

    Checks (in priority order):
    1. `softwareAgent` on c2pa.actions assertions (e.g. older Adobe/Firefly)
    2. `claim_generator` UA string, filtered to non-infrastructure tokens
       (e.g. "DALL-E 3/1.0 c2pa-rs/0.36.1" -> "DALL-E 3")

    Google's manifests don't expose the specific model (Imagen/Veo/etc.) in
    either field, so this returns None for them.
    """
    try:
        manifests = manifest_data.get("manifests", {})
        active_manifest_id = manifest_data.get("active_manifest")

        manifest = None
        if active_manifest_id and active_manifest_id in manifests:
            manifest = manifests[active_manifest_id]
        elif manifests:
            manifest = next(iter(manifests.values()))

        if not manifest:
            return None

        for assertion in manifest.get("assertions", []):
            if "c2pa.actions" not in assertion.get("label", ""):
                continue
            for action in assertion.get("data", {}).get("actions", []):
                software_agent = action.get("softwareAgent")
                if software_agent:
                    return _parse_product_name(software_agent)

        claim_generator = manifest.get("claim_generator")
        if claim_generator:
            return _parse_product_name(claim_generator)

    except Exception as e:
        bt.logging.debug(f"Error extracting model name: {e}")

    return None


def _parse_product_name(ua_string: str) -> Optional[str]:
    """
    Extract the primary product name from a UA-style string.

    Examples:
        "DALL-E 3/1.0 c2pa-rs/0.36.1"  -> "DALL-E 3"
        "Adobe Firefly/3.0"             -> "Adobe Firefly"
        "Google Imagen/3"               -> "Google Imagen"
        "c2pa-rs/0.36.1"               -> None  (infrastructure-only)
    """
    if not ua_string:
        return None

    # Skip known C2PA infrastructure library tokens
    INFRA_PREFIXES = ("c2pa", "libc2pa", "contentauth", "trustedpublisher")

    tokens = ua_string.strip().split()
    for token in tokens:
        product = token.split("/")[0].strip()
        if not product:
            continue
        if any(product.lower().startswith(p) for p in INFRA_PREFIXES):
            continue
        return product

    return None


def _is_trusted_ca(issuer: Optional[str]) -> bool:
    if not issuer:
        return False
    issuer_lower = issuer.lower()
    for trusted_ca in TRUSTED_CA_ISSUERS:
        if trusted_ca.lower() in issuer_lower:
            return True
    return False


def _check_ai_generated(manifest_data: Dict[str, Any]) -> bool:
    try:
        for manifest in manifest_data.get("manifests", {}).values():
            for assertion in manifest.get("assertions", []):
                label = assertion.get("label", "")
                if "c2pa.ai_generated" in label or "c2pa.ai" in label:
                    return True
                data = assertion.get("data", {})
                source_type = data.get("digitalSourceType", "")
                if source_type in AI_SOURCE_TYPES:
                    return True
                for action in data.get("actions", []):
                    action_source = action.get("digitalSourceType", "")
                    if action_source in AI_SOURCE_TYPES:
                        return True
        return False
    except Exception as e:
        bt.logging.debug(f"Error checking AI generation: {e}")
        return False


def _is_trusted_issuer(
    cert_subject: Optional[str],
    cert_issuer: Optional[str],
    claim_generator: Optional[str]
) -> bool:
    if not cert_subject and not cert_issuer:
        return False

    def check_against_trusted(value: str) -> bool:
        if not value:
            return False
        value_lower = value.lower()
        for trusted in TRUSTED_CERT_ISSUERS:
            if trusted.lower() in value_lower or value_lower in trusted.lower():
                return True
        return False

    if cert_subject and check_against_trusted(cert_subject):
        return True

    if cert_issuer:
        for trusted in TRUSTED_CERT_ISSUERS:
            if trusted.lower() in cert_issuer.lower():
                return True

    if claim_generator and (cert_subject or cert_issuer):
        claim_lower = claim_generator.lower()
        for trusted in TRUSTED_CERT_ISSUERS:
            if trusted.lower() in claim_lower:
                return True

    return False


def is_from_trusted_generator(media_data: Union[bytes, str, Path], require_ai_label: bool = False) -> bool:
    """Check if media is from a trusted AI generator with valid cryptographic proof."""
    result = verify_c2pa(media_data)
    if not result.verified or not result.signature_valid:
        return False
    if result.is_self_signed or not result.is_trusted_issuer:
        return False
    if require_ai_label and not result.ai_generated:
        return False
    return True
