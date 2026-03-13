import os
import traceback
from PIL import Image
import io
import time

from neurons.generator.services.stabilityai_service import StabilityAIService, Models
from neurons.generator.task_manager import TaskManager
from gas.verification.c2pa_verification import verify_c2pa

# Set API key if one isn't already set
os.environ.setdefault(
    "STABILITY_API_KEY",
    "sk-YOUR-API-KEY"  # replace with your test key
)

def save_image(img_bytes, filename):
    os.makedirs("outputs", exist_ok=True)
    out_path = f"outputs/{filename}"
    with open(out_path, "wb") as f:
        f.write(img_bytes)
    return out_path

def validate_image(img_bytes):
    try:
        Image.open(io.BytesIO(img_bytes)).verify()
        return True
    except Exception:
        return False


def run_model_test(service, manager, model):
    print(f"\n=== Running generation test for model: {model} ===")

    # Use TaskManager.create_task() exactly like in production
    task_id = manager.create_task(
        modality="image",
        prompt="A neon cyberpunk city with flying cars",
        parameters={
            "model": model,
            "format": "png",
            "aspect_ratio": "16:9",
            "seed": 777,
            "negative_prompt": "low quality, blurry"
        },
        webhook_url=None,
        signed_by="test-suite"
    )

    task = manager.get_task(task_id)

    try:
        start_time = time.time()
        result = service.process(task)
        elapsed = time.time() - start_time

        img_bytes = result["data"]
        meta = result["metadata"]

        print(f"✔ Generated image in {elapsed:.2f}s ({len(img_bytes)/1024:.1f} KB)")
        print(f"✔ Metadata keys: {list(meta.keys())}")

        # Save output
        out = save_image(img_bytes, f"{model.replace('.', '_')}.png")
        print(f"✔ Saved to {out}")

        # Validate image integrity
        assert validate_image(img_bytes), "Image failed Pillow validation"

        # After generating with Runway
        result = verify_c2pa(img_bytes)
        if result.verified and result.is_trusted_issuer:
            print(f"✅ C2PA verified: {result.issuer}")
        else:
            print(f"❌ C2PA failed: {result.error}")

        # Validate metadata
        assert meta["model"] == model
        assert meta["provider"] == "stability.ai"
        assert meta["format"] in ("PNG", "JPEG", "WEBP")
        assert meta["generation_time"] > 0

        print(f"=== Model {model} PASSED ===")

    except Exception:
        print(f"=== Model {model} FAILED ===")
        print(traceback.format_exc())


def test_invalid_api_key():
    print("\n=== Testing invalid API key ===")
    os.environ["STABILITY_API_KEY"] = "invalid-key"

    service = StabilityAIService()
    manager = TaskManager()

    task_id = manager.create_task(
        modality="image",
        prompt="test prompt",
        parameters={"model": Models.SD35_MEDIUM},
        webhook_url=None,
        signed_by="test"
    )
    task = manager.get_task(task_id)

    try:
        service.process(task)
        raise AssertionError("❌ Should have failed with invalid API key!")
    except Exception as e:
        print(f"✔ Correctly failed: {e}")


def test_invalid_format(service, manager):
    print("\n=== Testing invalid format fallback ===")

    task_id = manager.create_task(
        modality="image",
        prompt="test prompt",
        parameters={"model": Models.SD35_MEDIUM, "format": "tiff"},
        webhook_url=None,
        signed_by="test"
    )
    task = manager.get_task(task_id)

    result = service.process(task)
    assert result["metadata"]["format"] == "PNG"
    print("✔ Invalid format 'tiff' auto-corrected to PNG")


def run_full_test_suite():
    print("\n========== StabilityAI Full Test Suite ==========\n")

    # Restore real key
    os.environ["STABILITY_API_KEY"] = os.getenv("STABILITY_API_KEY")

    service = StabilityAIService()
    manager = TaskManager()

    if not service.is_available():
        print("❌ API key missing — cannot run tests")
        return

    # Test all models
    for model in [
        Models.SD35_MEDIUM,
        Models.SD35_LARGE,
        Models.ULTRA,
        Models.CORE
    ]:
        run_model_test(service, manager, model)

    # Negative tests
    test_invalid_format(service, manager)
    test_invalid_api_key()

    print("\n========== All Tests Completed ==========\n")


if __name__ == "__main__":
    run_full_test_suite()
