#!/usr/bin/env python3
"""Build or verify the portable full/self diffraction server bundle."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile

ROOT = Path(__file__).resolve().parents[1]
TOP = "diffraction_observables"
MANIFEST_NAME = f"{TOP}/bundle_manifest.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_paths(root=ROOT):
    """Explicit workflow inputs only: no runs, dotfiles or server history."""
    root = Path(root)
    paths = list((root / "upc_deformation").rglob("*.py"))
    paths += [root / name for name in (
        "upc_deformation/requirements.txt",
        "upc_deformation/vendor/provenance.json",
        "configs/diffraction_observables_a100.yaml",
        "configs/panel_d_precision_a100.yaml",
        "configs/assets/panel_d_orientation_prefix512.npz",
        "configs/assets/panel_d_orientation_prefix512.json",
        "docs/diffraction_observables_a100_runbook.md",
        "scripts/run_diffraction_observables.py",
        "scripts/validate_diffraction_observables.py",
        "scripts/analyze_diffraction_observables.py",
        "scripts/package_diffraction_observables.py",
        "tests/test_precision_forward.py",
        "tests/test_precision_analysis.py",
    )]
    paths += list((root / "tests").glob("test_diffraction_*.py"))
    paths = sorted(set(paths))
    missing = [str(path.relative_to(root)) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required bundle files are missing: {missing}")
    if not any(path.name.startswith("test_diffraction_") for path in paths):
        raise ValueError("No diffraction tests found")
    for path in paths:
        if path.is_symlink() or root.resolve() not in path.resolve().parents:
            raise ValueError(f"Bundle source must be a regular in-project file: {path}")
    return paths


def verify_archive(path, *, check_sidecar=True):
    """Verify every member and the gzip trailer without extracting files."""
    path = Path(path)
    # Tar readers may stop at the tar EOF before consuming the gzip trailer.
    # Explicitly consuming gzip detects a truncated transfer even in that case.
    with gzip.open(path, "rb") as stream:
        while stream.read(1024 * 1024):
            pass
    found = {}
    manifest = None
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if (not member.isfile() or name.is_absolute() or ".." in name.parts or
                    not name.parts or name.parts[0] != TOP or len(name.parts) < 2):
                raise ValueError(f"Unexpected or unsafe archive member: {member.name}")
            if member.name in found:
                raise ValueError(f"Duplicate archive member: {member.name}")
            stream = archive.extractfile(member)
            digest = hashlib.sha256()
            count = 0
            payload = [] if member.name == MANIFEST_NAME else None
            with stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                    count += len(chunk)
                    if payload is not None:
                        payload.append(chunk)
            if count != member.size:
                raise ValueError(f"Incomplete archive member: {member.name}")
            found[member.name] = {"sha256": digest.hexdigest(), "bytes": count}
            if payload is not None:
                manifest = json.loads(b"".join(payload))
    if manifest is None or manifest.get("schema") != "diffraction_bundle_v1":
        raise ValueError("Missing or unsupported bundle manifest")
    expected = {f"{TOP}/{name}": info for name, info in manifest["files"].items()}
    payload_members = {name: info for name, info in found.items() if name != MANIFEST_NAME}
    if set(expected) != set(payload_members):
        raise ValueError("Manifest and archive member sets differ")
    failed = [name for name in expected if expected[name] != payload_members[name]]
    if failed:
        raise ValueError(f"Member size/hash mismatch: {failed}")
    checksum = sha256(path)
    sidecar = Path(str(path) + ".sha256")
    if check_sidecar and sidecar.exists():
        fields = sidecar.read_text().strip().split(maxsplit=1)
        if len(fields) != 2 or fields[0] != checksum or fields[1].lstrip("*") != path.name:
            raise ValueError("Archive SHA256 sidecar differs")
    return {"status": "passed", "archive": str(path.resolve()),
            "archive_sha256": checksum, "files": len(expected)}


def build_archive(output, *, root=ROOT):
    root, output = Path(root), Path(output)
    if output.name == "panel_d_precision_a100.tar.gz":
        raise ValueError("The previous panel-d bundle must not be overwritten")
    # Read a coherent snapshot first. No archive-provided code is executed.
    payloads = {str(path.relative_to(root)): path.read_bytes() for path in bundle_paths(root)}
    manifest = {"schema": "diffraction_bundle_v1", "top_directory": TOP,
                "scope": "Portable full/self code, configuration, tests and orientation prefix; no production outputs.",
                "files": {name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                          for name, data in payloads.items()}}
    payloads["bundle_manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".partial")
    try:
        with temp.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                    for name, data in sorted(payloads.items()):
                        info = tarfile.TarInfo(f"{TOP}/{name}")
                        info.size = len(data)
                        info.mode = 0o755 if name.startswith("scripts/") else 0o644
                        info.mtime = 0
                        archive.addfile(info, io.BytesIO(data))
            raw.flush()
            os.fsync(raw.fileno())
        verify_archive(temp, check_sidecar=False)
        os.replace(temp, output)
        sidecar = Path(str(output) + ".sha256")
        sidecar_temp = sidecar.with_name(sidecar.name + ".partial")
        sidecar_temp.write_text(f"{sha256(output)}  {output.name}\n")
        os.replace(sidecar_temp, sidecar)
    finally:
        temp.unlink(missing_ok=True)
    return verify_archive(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/diffraction_observables_a100.tar.gz")
    parser.add_argument("--verify", type=Path, metavar="ARCHIVE", help="Verify an existing bundle without extraction")
    args = parser.parse_args()
    result = verify_archive(args.verify) if args.verify else build_archive(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
