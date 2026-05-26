"""
ml_models/training/augmentation.py
=====================================
Image augmentation pipeline for both YOLO and CNN training.
Wraps Albumentations transforms with sensible defaults for traffic imagery.

Usage::

    from ml_models.training.augmentation import get_train_transforms, get_val_transforms
    transform = get_train_transforms(img_size=640)
    augmented = transform(image=cv2_bgr_image)["image"]
"""

import numpy as np

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False

from torchvision import transforms as T
from PIL import Image


# ── Albumentations transforms (for YOLO / OpenCV pipelines) ───

def get_yolo_train_transforms(img_size: int = 640):
    """
    Heavy augmentation for YOLO training on traffic images.
    Returns an albumentations Compose transform with bbox support.
    """
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError("Install albumentations: pip install albumentations")

    return A.Compose([
        A.RandomResizedCrop(img_size, img_size, scale=(0.5, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.8),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.MotionBlur(blur_limit=5, p=0.2),
        A.RandomRain(p=0.15),
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.1),
        A.RandomShadow(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))


def get_yolo_val_transforms(img_size: int = 640):
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError("Install albumentations: pip install albumentations")

    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))


# ── Torchvision transforms (for CNN training) ──────────────────

def get_cnn_train_transforms(img_size: int = 224):
    """Strong augmentation for CNN classification training."""
    return T.Compose([
        T.Resize((int(img_size * 1.15), int(img_size * 1.15))),
        T.RandomCrop(img_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.05),
        T.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.1),
        T.RandomRotation(degrees=18),
        T.RandomGrayscale(p=0.05),
        T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2),
        T.RandomPerspective(distortion_scale=0.3, p=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.15, scale=(0.02, 0.15)),
    ])


def get_cnn_val_transforms(img_size: int = 224):
    """Minimal transforms for CNN validation / inference."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ── Numpy-based augmentation (no dependencies) ─────────────────

def random_horizontal_flip(image: np.ndarray, p: float = 0.5) -> np.ndarray:
    if np.random.random() < p:
        return image[:, ::-1, :].copy()
    return image


def random_brightness_contrast(
    image: np.ndarray,
    brightness_range: tuple = (-30, 30),
    contrast_range: tuple = (0.8, 1.2),
) -> np.ndarray:
    img = image.astype(np.float32)
    alpha = np.random.uniform(*contrast_range)
    beta  = np.random.uniform(*brightness_range)
    img = np.clip(alpha * img + beta, 0, 255)
    return img.astype(np.uint8)


def random_gaussian_noise(image: np.ndarray, std: float = 15.0) -> np.ndarray:
    noise = np.random.normal(0, std, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def augment_image_numpy(image: np.ndarray) -> np.ndarray:
    """
    Lightweight, dependency-free augmentation pipeline for quick experiments.
    Input/output: H×W×C uint8 numpy array (BGR or RGB).
    """
    image = random_horizontal_flip(image)
    image = random_brightness_contrast(image)
    if np.random.random() < 0.3:
        image = random_gaussian_noise(image)
    return image


# ── Mixup / CutMix (for CNN) ───────────────────────────────────

def mixup(
    x1: np.ndarray, y1: int,
    x2: np.ndarray, y2: int,
    alpha: float = 0.4,
    num_classes: int = 2,
) -> tuple:
    """
    Mixup augmentation.
    Returns: mixed image (float32), mixed one-hot label
    """
    lam = np.random.beta(alpha, alpha)
    mixed_x = lam * x1.astype(np.float32) + (1 - lam) * x2.astype(np.float32)
    label1 = np.eye(num_classes)[y1]
    label2 = np.eye(num_classes)[y2]
    mixed_y = lam * label1 + (1 - lam) * label2
    return mixed_x, mixed_y
