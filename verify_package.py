"""Verify public source files; optionally verify privately supplied model files."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _verify_items(root, items, *, private_weights=False):
    root = root.resolve()
    missing = []
    for item in items:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Invalid path outside the package")
        if not path.is_file():
            missing.append(item["path"])
            continue
        if path.stat().st_size != item["size_bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Size/SHA-256 mismatch: {item['path']}. See RUN.md for the exact required files.")
    if missing:
        if private_weights:
            raise ValueError("This public package contains no model weights. Inference requires privately "
                             "supplied files listed in WEIGHTS_REQUIRED.json; see RUN.md. Missing: "
                             + ", ".join(missing))
        raise ValueError("Missing public release files: " + ", ".join(missing))
    return len(items)


def verify(root=ROOT):
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    return _verify_items(root, manifest["files"])


def verify_weights(root=ROOT):
    required = json.loads((root / "WEIGHTS_REQUIRED.json").read_text(encoding="utf-8"))
    return _verify_items(root, required["files"], private_weights=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-weights", action="store_true", help="Also check privately supplied model files")
    args = parser.parse_args()
    try:
        print(f"OK: {verify()} public files verified (size and SHA-256).")
        if args.with_weights:
            print(f"OK: {verify_weights()} private model files verified.")
        else:
            print("Model weights are not part of this public release and were not checked.")
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
