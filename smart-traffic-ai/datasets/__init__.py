# datasets/__init__.py
"""
datasets/
=========
Dataset preparation and loading utilities.

Structure:
    raw/         — original downloaded datasets (not committed to git)
    processed/   — cleaned, annotated, split into train/val/test
    annotations/ — YOLO-format label files
    augmented/   — augmented training copies

Run to prepare:
    python datasets/prepare_dataset.py --task all
"""
