"""Portable single-patch inference; frozen Phikon-v2 BF16 CUDA + FP32 MLP."""
from __future__ import annotations

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

from verify_package import ROOT, sha256, verify, verify_weights


def select_device(torch, requested):
    if requested == "cpu":
        raise ValueError("CPU inference is not supported by the original numerical contract: "
                         "the encoder requires CUDA BF16, with an FP32 head. Use --device cuda:0 "
                         "on a compatible NVIDIA GPU. No automatic precision change is performed.")
    if requested != "cuda" and not (requested.startswith("cuda:") and requested[5:].isdigit()):
        raise ValueError("Use --device cuda, cuda:0, cuda:1, ... (CPU is unsupported).")
    if not torch.cuda.is_available():
        raise ValueError("Requested CUDA is unavailable. Install requirements.lock.txt and an "
                         "NVIDIA driver compatible with CUDA 13.0; check nvidia-smi.")
    index = int(requested.split(":")[1]) if ":" in requested else 0
    if index >= torch.cuda.device_count():
        raise ValueError(f"Requested cuda:{index}, but only {torch.cuda.device_count()} GPU(s) are visible.")
    torch.cuda.set_device(index)
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError("Selected GPU lacks native BF16 support required by the original encoder.")
    return torch.device(f"cuda:{index}")


def strict_image(path):
    from PIL import Image
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != (224, 224):
            raise ValueError(f"Expected PNG RGB uint8 224x224; got {image.format}/{image.mode}/{image.size}. "
                             "No resize, crop or color conversion is applied.")
        image.load()
        return image.copy()


def preprocess(image, config):
    from torchvision import transforms
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config["input"]["mean"], config["input"]["std"]),
    ])(image).unsqueeze(0)


def seed_all(torch, np, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(8)


def load_models(config, device):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModel
    from model import MLPHead
    encoder, info = AutoModel.from_pretrained(
        ROOT / config["encoder"]["directory"], local_files_only=True,
        use_safetensors=True, output_loading_info=True, attn_implementation="sdpa",
    )
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if info.get(key):
            raise ValueError(f"Incompatible encoder: {key}={info[key]}")
    if encoder.config.hidden_size != 1024 or encoder.config.image_size != 224:
        raise ValueError("Encoder shape mismatch")
    encoder.requires_grad_(False).eval().to(device)
    head = MLPHead(1024, 2).to(device)
    head.load_state_dict(load_file(ROOT / config["head"]["file"]), strict=True)
    head.requires_grad_(False).eval()
    if any(p.dtype != torch.float32 for m in (encoder, head) for p in m.parameters()):
        raise ValueError("Model parameters must remain float32")
    return encoder, head


def forward(encoder, head, images, device):
    import torch
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        features = encoder(pixel_values=images.to(device)).last_hidden_state[:, 0, :].float()
    with torch.no_grad():
        logits = head(features)
        probabilities = logits.softmax(dim=1)
    if tuple(features.shape) != (1, 1024):
        raise ValueError("Invalid CLS dimensions")
    for name, tensor in (("features", features), ("logits", logits), ("probabilities", probabilities)):
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Nonfinite {name}")
    return features.cpu(), logits.cpu(), probabilities.cpu()


def save_result(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if path.suffix.lower() == ".json":
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    else:
        row = {"image_sha256": result["image_sha256"], **result["classification"],
               "device": result["runtime"]["device"], "model_version": result["model_version"],
               "output_scope": "patch", "calibrated": False,
               "qc_accepted": result["quality"]["accepted"],
               "qc_flags": json.dumps(result["quality"]["qc_flags"]),
               "qc_rejection_reasons": json.dumps(result["quality"]["rejection_reasons"])}
        for name in result["classes"]:
            row[f"logit_{name}"] = result["logits"][name]
            row[f"probability_{name}"] = result["softmax_scores"][name]
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Native PNG RGB 224x224")
    parser.add_argument("--output", type=Path, required=True, help="Output .json or .csv")
    parser.add_argument("--device", default="cuda:0", help="cuda:0 (default), cuda:N; CPU unsupported")
    args = parser.parse_args()
    try:
        if sys.version_info[:3] != (3, 11, 9):
            raise ValueError("This release is verified with Python 3.11.9; install that version.")
        if args.output.suffix.lower() not in (".json", ".csv"):
            raise ValueError("--output must have .json or .csv extension")
        if args.image.resolve() == args.output.resolve():
            raise ValueError("Input and output must differ")
        manifest = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
        protected = {(ROOT / i["path"]).resolve() for i in manifest["files"]}
        required = json.loads((ROOT / "WEIGHTS_REQUIRED.json").read_text(encoding="utf-8"))
        protected.update((ROOT / i["path"]).resolve() for i in required["files"])
        protected.add((ROOT / "MANIFEST.json").resolve())
        if args.output.resolve() in protected:
            raise ValueError("Output would overwrite a release file")
        verify()
        verify_weights()
        image = strict_image(args.image)
        import numpy as np
        import torch
        from qc_pixels import analyze_rgb
        device = select_device(torch, args.device)
        print(f"Actual device: {device} / {torch.cuda.get_device_name(device)}; encoder BF16 autocast, head FP32", flush=True)
        config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        seed_all(torch, np, config["seed"])
        started = time.perf_counter()
        inputs = preprocess(image, config)
        quality = analyze_rgb(image, json.loads((ROOT / config["qc_config"]).read_text(encoding="utf-8")))
        encoder, head = load_models(config, device)
        _, logits, probabilities = forward(encoder, head, inputs, device)
        classes = config["classes"]
        predicted = int(probabilities[0, 1] > config["output"]["threshold"])
        result = {
            "schema_version": 1, "model_id": config["model_id"], "model_version": config["model_version"],
            "image_sha256": sha256(args.image), "classes": classes,
            "logits": dict(zip(classes, logits[0].tolist())),
            "softmax_scores": dict(zip(classes, probabilities[0].tolist())),
            "classification": {"class_index": predicted, "label": classes[predicted]},
            "calibrated_probabilities": None, "output_scope": "patch", "quality": quality,
            "runtime": {"device": str(next(encoder.parameters()).device),
                        "gpu_name": torch.cuda.get_device_name(device), "torch": torch.__version__,
                        "cuda_runtime": torch.version.cuda, "encoder_autocast": "bfloat16",
                        "parameter_dtype": "float32", "head_dtype": "float32", "batch_size": 1,
                        "elapsed_seconds": round(time.perf_counter()-started, 6)},
            "limitations": ["Research only; uncalibrated model scores, no specimen diagnosis.",
                            "QC is a technical heuristic; no saved dataset/manual QC records are used.",
                            "A synthetic input tests execution only, not model quality."],
        }
        save_result(args.output.resolve(), result)
        print(f"Saved: {args.output.resolve()}")
        print(json.dumps(result["classification"], ensure_ascii=False))
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
