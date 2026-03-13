# ONNX Model Creation Guide

> **⚠️ DEPRECATED**: ONNX format is no longer accepted for Subnet 34 competition submissions. 
> Please use **safetensors format** instead. See the [Safetensors Guide](https://github.com/bitmind-ai/gasbench/blob/main/docs/Safetensors.md) for requirements.

---

This guide explains how to create ONNX models for discriminative mining using the example scripts in `neurons/discriminator/onnx_examples`.

## Key Requirements

**⚠️ IMPORTANT**: Your ONNX models must meet these requirements:

1. **Input Shape**: Specify fixed spatial dimensions for optimal batching
   - Image models: `['batch_size', 3, H, W]` where H and W are fixed (e.g., 224)
   - Video models: `['batch_size', 'frames', 3, H, W]` where H and W are fixed
   - ⚠️ If you use dynamic H/W axes, gasbench will default to 224x224

2. **Pixel Range**: Accept raw pixel values in range `[0-255]`
   - Gasbench handles preprocessing (shortest-edge resize, center crop, augmentations)
   - Your model wrapper should only normalize to [0, 1] and apply model-specific normalization

3. **Output Format**: Return logits for 3 classes `[real, synthetic, semisynthetic]`
   - Image models: `(batch_size, 3)`
   - Video models: `(batch_size, 3)` after temporal aggregation


## Example Scripts

The `neurons/discriminator/onnx_examples` directory contains working examples:

### PyTorch Models
- `pytorch_models/image_model.py` - Custom image classification model
- `pytorch_models/video_model.py` - Custom video classification model

### HuggingFace Models  
- `huggingface_models/image_model.py` - Convert HuggingFace image models (e.g., ResNet50)
- `huggingface_models/video_model.py` - Convert HuggingFace video models (e.g., VideoMAE)

## Quick Start

1. **Navigate to the examples directory:**
   ```bash
   cd neurons/discriminator/onnx_examples
   ```

2. **Run an example script:**
   ```bash
   # For custom PyTorch models
   python pytorch_models/image_model.py
   python pytorch_models/video_model.py
   
   # For HuggingFace models
   python huggingface_models/image_model.py
   python huggingface_models/video_model.py
   ```

3. **Check the output:**
   ```bash
   ls models/
   # Should see: image_detector.onnx, video_detector.onnx
   ```

## Preprocessing Pipeline

Gasbench performs the following preprocessing before passing data to your model:
1. **Resize shortest edge** to target size (preserving aspect ratio)
2. **Center crop** to exact target size (H x W from your model spec)
3. **Random augmentations** (rotation, flip, crop, color jitter, etc.)
4. **Batching** with configurable batch size

Your ONNX wrapper should only handle:
- Normalization: `[0-255]` → `[0-1]`
- Model-specific transforms (e.g., ImageNet mean/std)
- **Video models**: Temporal aggregation

## Custom Models

To create your own model:

1. **Inherit from `nn.Module`** and implement your architecture
2. **Wrap with preprocessing** (normalize, model-specific transforms, temporal aggregation)
3. **Export with fixed spatial dimensions** as shown in the examples
4. **Test with batched uint8 inputs** to ensure compatibility

## Example Export Code

### Image Model
```python
# Create dummy input with raw pixel values (fixed spatial dims)
dummy_input = torch.randint(0, 256, (1, 3, 224, 224), dtype=torch.uint8)

# Export with fixed spatial dimensions for batching
torch.onnx.export(
    wrapped_model,
    dummy_input,
    "models/image_detector.onnx",
    input_names=['input'],
    output_names=['logits'],
    dynamic_axes={
        'input': {0: 'batch_size'},  # Only batch_size is dynamic
        'logits': {0: 'batch_size'}
    },
    opset_version=11,
    do_constant_folding=True,
    export_params=True,
    keep_initializers_as_inputs=False
)
```

### Video Model
```python
# Create dummy input (B, T, C, H, W) with fixed spatial dims
dummy_input = torch.randint(0, 256, (1, 8, 3, 224, 224), dtype=torch.uint8)

# Export with dynamic batch and frames, fixed spatial dims
torch.onnx.export(
    wrapped_model,
    dummy_input,
    "models/video_detector.onnx",
    input_names=['input'],
    output_names=['logits'],
    dynamic_axes={
        'input': {0: 'batch_size', 1: 'frames'},  # H, W are fixed
        'logits': {0: 'batch_size'}
    },
    opset_version=11,
    do_constant_folding=True,
    export_params=True,
    keep_initializers_as_inputs=False
)
```

## Model Preprocessing Wrapper

To perform input preprocessing or ouptut postprocessing, you can wrap your model:

```python
class PreprocessingWrapper(nn.Module):
    def __init__(self, model, is_video=False):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        # Input x: (B, T, C, H, W) for video, (B, C, H, W) for image
        # Values in range [0, 255]
        
        # Normalize to [0, 1]
        x = x.float() / 255.0
        
        # Apply model
        outputs = self.model(x)
        
        # Any necessary output postprocessing

        return outputs
```

The temporal aggregation prevents single-frame anomalies from dominating predictions.

## Packaging Your Models

After creating your ONNX models, you need to package them into zip files before pushing to the network (keeps the system flexible in case we need to add supplemental files in the future):

```bash
# Package image model
zip image_detector.zip image_detector.onnx

# Package video model
zip video_detector.zip video_detector.onnx
```

Each zip file should contain only the corresponding ONNX model file.

## Next Steps

Once you have your ONNX files packaged as zip files, follow the [Discriminative Mining Guide](Discriminative-Mining.md) to push them to the network.
