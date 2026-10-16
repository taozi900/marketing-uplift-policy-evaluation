"""Fetch the exact source gzip, or restore it from a local archive."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import urllib.request


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--archive", type=Path, help="Use an already downloaded original gzip.")
    args = parser.parse_args()
    raw_dir = args.project_dir.resolve() / "data" / "raw"
    manifest = json.loads((raw_dir / "source_manifest.json").read_text(encoding="utf-8"))
    csv_path = raw_dir / "hillstrom.csv"
    if csv_path.exists():
        if digest(csv_path.read_bytes()) != manifest["csv_sha256"]:
            raise ValueError("Existing CSV differs from the source manifest; keep it for investigation.")
        print(f"CSV already present and SHA256 verified: {csv_path}")
        return
    if args.archive:
        payload = args.archive.read_bytes()
    else:
        with urllib.request.urlopen(manifest["download_url"], timeout=60) as response:
            payload = response.read(10_000_001)
    if len(payload) > 10_000_000 or digest(payload) != manifest["archive_sha256"]:
        raise ValueError("Source gzip SHA256/size check failed; no source data written.")
    plain = gzip.decompress(payload)
    if len(plain) != manifest["csv_bytes"] or digest(plain) != manifest["csv_sha256"]:
        raise ValueError("Decompressed source CSV SHA256/size check failed.")
    archive_path = raw_dir / "hillstrom_original.csv.gz"
    if archive_path.exists() and digest(archive_path.read_bytes()) != manifest["archive_sha256"]:
        raise ValueError("Existing gzip differs; refusing to replace it.")
    archive_path.write_bytes(payload)
    csv_path.write_bytes(plain)
    if digest(csv_path.read_bytes()) != manifest["csv_sha256"]:
        raise ValueError("Saved CSV failed verification.")
    print(f"Verified original CSV: {csv_path}")
    print(f"SHA256: {manifest['csv_sha256']}")


if __name__ == "__main__":
    main()
