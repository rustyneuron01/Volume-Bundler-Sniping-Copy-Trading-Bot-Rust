#!/usr/bin/env python3
"""Export ReStraV checkpoint to model.safetensors for gasbench."""
import argparse
from pathlib import Path
import torch
from safetensors.torch import save_file
from model import create_model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="checkpoints/restrav/best.pt")
    p.add_argument("--output", type=str, default="checkpoints/restrav/model.safetensors")
    args = p.parse_args()
    ckpt = torch.load(args.input, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("state_dict", ckpt)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, out_path)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
