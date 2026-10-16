"""Verify the distributed snapshot without loading data or external packages."""
from pathlib import Path
import hashlib
import json


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "archive_manifest.json").read_text(encoding="utf-8"))
    failed = []
    for name, expected in manifest["sha256"].items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failed.append(name)
    if failed:
        raise ValueError("Snapshot files missing/changed: " + ", ".join(failed))
    print(f"Archive verified: {len(manifest['sha256'])} files.")


if __name__ == "__main__":
    main()
