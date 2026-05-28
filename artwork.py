import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Dict, Any
import os

@dataclass
class Metadata:
    object_id: str
    title: Optional[str] = None
    artist: Optional[str] = None
    date: Optional[str] = None
    medium: Optional[str] = None
    dimensions: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None


class Artwork:
    __slots__ = ('_image', '_metadata', '_image_path', '_index')
    
    def __init__(self, image: np.ndarray, metadata: Metadata, image_path: str = "", index: int = 0):
        self._image = image
        self._metadata = metadata
        self._image_path = image_path
        self._index = index
    
    @property
    def image(self) -> np.ndarray:
        return self._image
    
    @property
    def metadata(self) -> Metadata:
        return self._metadata
    
    @property
    def image_path(self) -> str:
        return self._image_path
    
    @property
    def index(self) -> int:
        return self._index
    
    def __add__(self, other: 'Artwork') -> 'Artwork':
        if self._image.shape != other._image.shape:
            raise ValueError("Изображения должны иметь одинаковую размерность для сложения")
        blended = cv2.addWeighted(self._image, 0.5, other._image, 0.5, 0)
        combined_metadata = Metadata(
            object_id=f"{self._metadata.object_id}_{other._metadata.object_id}",
            title=f"{self._metadata.title} + {other._metadata.title}",
            raw_data={"source1": self._metadata.raw_data, "source2": other._metadata.raw_data}
        )
        return Artwork(blended, combined_metadata)
    
    def __str__(self) -> str:
        return f"Artwork(ID: {self._metadata.object_id}, Title: {self._metadata.title}, Shape: {self._image.shape})"
    
    def grayscale(self) -> np.ndarray:
        height, width = self._image.shape[:2]
        result = np.zeros((height, width))
        eye_vector = np.array([0.299, 0.587, 0.114])
        result = np.sum(self._image * eye_vector, axis=2)
        return result.astype(np.uint8)
    
    def convolution(self, kernel: np.ndarray) -> np.ndarray:
        kernel_height, kernel_width = kernel.shape
        padded_height, padded_width = kernel_height // 2, kernel_width // 2
        
        pad_width = [(padded_height, padded_height), (padded_width, padded_width)]
        if self._image.ndim == 3:
            pad_width.append((0, 0))
        
        padded = np.pad(self._image, pad_width)
        
        windows = np.lib.stride_tricks.sliding_window_view(padded, kernel.shape, axis=(0, 1))
        result = np.clip(np.tensordot(windows, kernel), 0, 255)
        return result.astype(np.uint8)
    
    def gaussian_blur(self, kernel_size: int = 3) -> np.ndarray:
        kernel = self._get_gaussian_kernel(kernel_size, kernel_size / 6)
        return self.convolution(kernel).astype(np.uint8)
    
    def sobel(self) -> np.ndarray:
        kernel_x = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[1, 2, 1],
                             [0, 0, 0],
                             [-1, -2, -1]], dtype=np.float32)
        
        grad_x = self.convolution(kernel_x)
        grad_y = self.convolution(kernel_y)
        
        grad_x = grad_x.astype(np.float32)
        grad_y = grad_y.astype(np.float32)
        grad = np.sqrt(grad_x**2 + grad_y**2)
        
        grad = (grad / grad.max()) * 255
        return grad.astype(np.uint8)
    
    def gamma_correction(self, gamma: float = 1.0) -> np.ndarray:
        normalized = self._image.astype(np.float32) / 255.0
        corrected = np.power(normalized, gamma)
        result = (corrected * 255).astype(np.uint8)
        return result
    
    def histogram_equalization(self) -> np.ndarray:
        lab = cv2.cvtColor(self._image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        hist = np.zeros(256, dtype=np.int32)
        for i in range(l_channel.shape[0]):
            for j in range(l_channel.shape[1]):
                hist[l_channel[i, j]] += 1
        
        cdf = hist.cumsum()
        cdf_normalized = ((cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())).astype(np.uint8)
        l_equalized = cdf_normalized[l_channel]
        
        lab_equalized = cv2.merge([l_equalized, a_channel, b_channel])
        result = cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)
        return result
    
    @staticmethod
    def _get_gaussian_kernel(kernel_size: int, sigma: float) -> np.ndarray:
        ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
        gauss = np.exp(-((ax / sigma) ** 2) / 2)
        kernel = np.outer(gauss, gauss)
        return (kernel / kernel.sum()).astype(np.float32)