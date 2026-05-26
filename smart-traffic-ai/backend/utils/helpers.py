"""
backend/utils/helpers.py
=========================
Shared utility functions used across Flask routes.
"""

import os
import uuid
import base64
import numpy as np
from pathlib import Path


def save_upload(file_obj, upload_folder: str) -> str:
    """Save a werkzeug FileStorage to upload_folder. Returns the absolute path."""
    ext = Path(file_obj.filename).suffix or ".jpg"
    fname = f"{uuid.uuid4().hex}{ext}"
    dest = os.path.join(upload_folder, fname)
    file_obj.save(dest)
    return dest


def base64_to_numpy(b64_str: str) -> np.ndarray:
    """Decode a base64 image string to a BGR numpy array (OpenCV format)."""
    import cv2
    header, _, data = b64_str.partition(",")
    raw = base64.b64decode(data if data else b64_str)
    arr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def numpy_to_base64(img: np.ndarray, ext: str = ".jpg") -> str:
    """Encode a BGR numpy array to a base64 string."""
    import cv2
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError("cv2.imencode failed")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    return f"data:{mime};base64,{b64}"


def success(data: dict = None, status: int = 200):
    """Standard success JSON response."""
    from flask import jsonify
    payload = {"status": "ok"}
    if data:
        payload.update(data)
    return jsonify(payload), status


def error(message: str, status: int = 400):
    """Standard error JSON response."""
    from flask import jsonify
    return jsonify({"status": "error", "error": message}), status


def paginate(query, page: int = 1, per_page: int = 50):
    """Helper to paginate a SQLAlchemy query."""
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    return {
        "items": pagination.items,
        "total": pagination.total,
        "page": pagination.page,
        "pages": pagination.pages,
        "per_page": pagination.per_page,
    }
