#!/usr/bin/env python3
"""
Alternative exporter for A/B comparison.

This keeps the original export pipeline intact, but modifies the generated
standalone model code to align inference with HF/training behavior:
backbone output -> mean pool (if token sequence) -> fc_norm -> classifier.
"""

import export_videomae_v2 as base


def create_standalone_model(temperature, hidden_size=1408, input_format="RGB"):
    """Create standalone model code aligned with HF/training inference path."""
    code = base.create_standalone_model(
        temperature=temperature,
        hidden_size=hidden_size,
        input_format=input_format,
    )

    old_backbone_return = """        for blk in self.blocks:
            x = blk(x)
        
        # Match HF: norm applied to first token only, returns [B, embed_dim]
        return self.norm(x[:, 0])
"""
    new_backbone_return = """        for blk in self.blocks:
            x = blk(x)
        
        # HF/training-aligned: return token sequence [B, N, C]
        # Pooling + fc_norm are handled in VideoDetector, like training.
        return x
"""

    old_pool_classifier = """        # Backbone - returns norm(x[:, 0]) already [B, embed_dim]
        pooled = self.backbone(x)
        
        # Classifier
        pooled = self.fc_norm(pooled)
        logits = self.classifier(pooled)
"""
    new_pool_classifier = """        # Backbone output can be token sequence [B, N, C] or pooled [B, C]
        features = self.backbone(x)
        if features.dim() == 3:
            pooled = features.mean(dim=1)
        else:
            pooled = features
        
        # Classifier head: fc_norm -> linear
        pooled = self.fc_norm(pooled)
        logits = self.classifier(pooled)
"""

    if old_backbone_return not in code:
        raise RuntimeError("Patch failed: backbone return block not found in generated model code.")
    if old_pool_classifier not in code:
        raise RuntimeError("Patch failed: pooling/classifier block not found in generated model code.")

    code = code.replace(old_backbone_return, new_backbone_return, 1)
    code = code.replace(old_pool_classifier, new_pool_classifier, 1)
    return code


def main():
    # Monkeypatch base exporter to reuse all checkpoint/packaging logic unchanged.
    original = base.create_standalone_model
    base.create_standalone_model = create_standalone_model
    try:
        base.main()
    finally:
        base.create_standalone_model = original


if __name__ == "__main__":
    main()
