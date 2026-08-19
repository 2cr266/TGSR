#!/usr/bin/env python3
"""Create the fixed ImageNet-Test-100 subset used by TGSR.

The source can be either the released ImageNet-Test ZIP archive or an
extracted directory. Candidate HR filenames are sorted before sampling.
NumPy RandomState is used deliberately to make the seed-0 selection stable
across runs and independent of filesystem enumeration order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the reproducible 100-image TGSR subset from ImageNet-Test."
    )
    parser.add_argument("source", type=Path, help="ImageNet-Test ZIP archive or extracted directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/imagenet_test_100.txt"),
        help="output filename list",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/imagenet_test_100.json"),
        help="output reproducibility manifest",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--expected-candidates", type=int, default=3000)
    parser.add_argument(
        "--source-label",
        default="ResShift ImageNet-Test",
        help="public dataset name recorded in the manifest",
    )
    return parser.parse_args()


def is_hr_image(path: PurePosixPath) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES and path.parent.name.lower() in {"gt", "hr"}


def filenames_from_zip(source: Path) -> list[str]:
    with zipfile.ZipFile(source) as archive:
        paths = [PurePosixPath(info.filename) for info in archive.infolist() if not info.is_dir()]
    return [path.name for path in paths if is_hr_image(path)]


def filenames_from_directory(source: Path) -> list[str]:
    paths = [path for path in source.rglob("*") if path.is_file()]
    return [path.name for path in paths if is_hr_image(PurePosixPath(path.as_posix()))]


def collect_candidates(source: Path) -> list[str]:
    if source.is_file() and zipfile.is_zipfile(source):
        names = filenames_from_zip(source)
    elif source.is_dir():
        names = filenames_from_directory(source)
    else:
        raise FileNotFoundError(f"Source is not a readable ZIP archive or directory: {source}")

    candidates = sorted(names)
    if len(candidates) != len(set(candidates)):
        raise ValueError("Duplicate HR basenames were found in the source dataset.")
    return candidates


def write_outputs(
    source: Path,
    candidates: list[str],
    selected: list[str],
    output: Path,
    manifest_path: Path,
    seed: int,
    source_label: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(selected) + "\n", encoding="utf-8", newline="\n")

    candidate_bytes = "\n".join(candidates).encode("utf-8")
    manifest = {
        "dataset": "ImageNet-Test-100",
        "source_dataset": source_label,
        "candidate_subdirectory": "gt (or hr)",
        "candidate_count": len(candidates),
        "selection_count": len(selected),
        "candidate_order": "lexicographic filename order",
        "rng": "numpy.random.RandomState",
        "seed": seed,
        "sampling": "choice without replacement",
        "candidate_list_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "filenames": selected,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    args = parse_args()
    candidates = collect_candidates(args.source)
    if len(candidates) != args.expected_candidates:
        raise ValueError(
            f"Expected {args.expected_candidates} HR candidates, found {len(candidates)}. "
            "Check that the source is the released ResShift ImageNet-Test archive."
        )
    if not 0 < args.count <= len(candidates):
        raise ValueError("count must be between 1 and the number of candidates")

    rng = np.random.RandomState(args.seed)
    indices = rng.choice(len(candidates), size=args.count, replace=False)
    selected = [candidates[int(index)] for index in indices]
    write_outputs(
        args.source,
        candidates,
        selected,
        args.output,
        args.manifest,
        args.seed,
        args.source_label,
    )

    print(f"Selected {len(selected)} images from {len(candidates)} candidates.")
    print(f"Filename list: {args.output}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
