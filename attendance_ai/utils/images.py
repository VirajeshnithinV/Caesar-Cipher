import os
from typing import Union

import cv2
import numpy as np


def load_image_bgr(image: Union[str, np.ndarray]) -> np.ndarray:
	"""Load an image as BGR numpy array.

	- If a path is provided, read via cv2.imread (BGR).
	- If a numpy array is provided, attempt to ensure BGR ordering.
	"""
	if isinstance(image, str):
		if not os.path.exists(image):
			raise FileNotFoundError(f"Image not found: {image}")
		img = cv2.imread(image)
		if img is None:
			raise ValueError(f"Failed to read image: {image}")
		return img
	elif isinstance(image, np.ndarray):
		arr = image
		if arr.ndim == 2:
			# grayscale -> stack
			arr = np.stack([arr, arr, arr], axis=-1)
		if arr.shape[-1] == 3:
			# Heuristic: if it's likely RGB (common in PIL / matplotlib), convert to BGR
			# We cannot reliably detect; assume RGB for safety
			return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
		return arr
	else:
		raise TypeError("image must be a filepath or numpy array")


def ensure_float32(arr: np.ndarray) -> np.ndarray:
	if arr.dtype != np.float32:
		return arr.astype(np.float32)
	return arr