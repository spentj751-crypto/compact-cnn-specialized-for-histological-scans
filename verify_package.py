"""Check the shipped files without importing ML packages or accessing the network."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(root=ROOT):
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Invalid manifest path")
        if not path.is_file():
            raise ValueError(f"Missing file: {item['path']}. Restore the complete release / run git lfs pull.")
        if path.stat().st_size != item["size_bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Size/SHA-256 mismatch: {item['path']}. Restore the release; check Git LFS.")
    return len(manifest["files"])


if __name__ == "__main__":
    try:
        print(f"OK: {verify()} files verified (size and SHA-256).")
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
