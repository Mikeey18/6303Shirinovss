import numpy as np, cv2
from dataclasses import dataclass
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

@dataclass
class Metadata:
    """Класс для хранения метаданных изображения"""
    object_id: str
    title: Optional[str] = None
    artist: Optional[str] = None
    date: Optional[str] = None
    medium: Optional[str] = None
    dimensions: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None


class Artwork(ABC):
    """Абстрактный базовый класс для всех видов изображений"""
    __slots__ = ('_image', '_metadata', '_image_path')

    def __init__(self, image: np.ndarray, metadata: Metadata, image_path: str = ""):
        self._image = image
        self._metadata = metadata
        self._image_path = image_path
        self._validate_image()
    
    @abstractmethod
    def _validate_image(self) -> None:
        """Абстрактный метод для валидации изображения"""
        pass
    
    @property
    def image(self) -> np.ndarray:
        return self._image
    
    @property
    def metadata(self) -> Metadata:
        return self._metadata
    
    @property
    def image_path(self) -> str:
        return self._image_path
    
    @abstractmethod
    def __add__(self, other: 'Artwork') -> 'Artwork':
        """Абстрактный метод сложения изображений"""
        pass
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(ID: {self._metadata.object_id}, Title: {self._metadata.title}, Shape: {self._image.shape})"
    
    @abstractmethod
    def grayscale(self) -> np.ndarray:
        """Преобразование в оттенки серого"""
        pass
    
    def convolution(self, kernel: np.ndarray) -> np.ndarray:
        """Применение свертки с заданным ядром"""
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
        """Применение размытия по Гауссу"""
        kernel = self._get_gaussian_kernel(kernel_size, kernel_size / 6)
        return self.convolution(kernel).astype(np.uint8)
    
    def sobel(self) -> np.ndarray:
        """Применение оператора Собеля для выделения границ"""
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
        """Применение гамма-коррекции"""
        normalized = self._image.astype(np.float32) / 255.0
        corrected = np.power(normalized, gamma)
        result = (corrected * 255).astype(np.uint8)
        return result
    
    def histogram_equalization(self) -> np.ndarray:
        """Выравнивание гистограммы"""
        if len(self._image.shape) == 3:
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
        else:
            # Для черно-белых изображений
            hist = np.zeros(256, dtype=np.int32)
            for i in range(self._image.shape[0]):
                for j in range(self._image.shape[1]):
                    hist[self._image[i, j]] += 1
            
            cdf = hist.cumsum()
            cdf_normalized = ((cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())).astype(np.uint8)
            return cdf_normalized[self._image]
    
    @staticmethod
    def _get_gaussian_kernel(kernel_size: int, sigma: float) -> np.ndarray:
        """Генерация ядра Гаусса"""
        ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
        gauss = np.exp(-((ax / sigma) ** 2) / 2)
        kernel = np.outer(gauss, gauss)
        return (kernel / kernel.sum()).astype(np.float32)


class ColorArtwork(Artwork):
    """Класс для цветных изображений"""
    __slots__ = ('_color_space',)
    
    def __init__(self, image: np.ndarray, metadata: Metadata, image_path: str = "", color_space: str = "BGR"):
        super().__init__(image, metadata, image_path)
        self._color_space = color_space
    
    def _validate_image(self) -> None:
        """Проверка, что изображение цветное (3 канала)"""
        if len(self._image.shape) != 3:
            raise ValueError(f"ColorArtwork ожидает цветное изображение (3 канала), получено: {self._image.shape}")
        if self._image.shape[2] != 3:
            raise ValueError(f"ColorArtwork ожидает 3 цветовых канала, получено: {self._image.shape[2]}")
    
    @property
    def color_space(self) -> str:
        return self._color_space
    
    def __add__(self, other: Artwork) -> 'ColorArtwork':
        """Сложение двух цветных изображений"""
        if not isinstance(other, ColorArtwork):
            raise TypeError("Можно складывать только цветные изображения")
        
        if self._image.shape != other._image.shape:
            raise ValueError("Изображения должны иметь одинаковую размерность для сложения")
        
        alpha = 0.5
        beta = 1 - alpha
        blended = cv2.addWeighted(self._image, alpha, other._image, beta, 0)
        
        combined_metadata = Metadata(
            object_id=f"{self._metadata.object_id}_{other._metadata.object_id}",
            title=f"{self._metadata.title} + {other._metadata.title}",
            raw_data={"source1": self._metadata.raw_data, "source2": other._metadata.raw_data}
        )
        return ColorArtwork(blended, combined_metadata, color_space=self._color_space)
    
    def grayscale(self) -> np.ndarray:
        """Преобразование цветного изображения в оттенки серого"""
        height, width = self._image.shape[:2]
        result = np.zeros((height, width))
        eye_vector = np.array([0.299, 0.587, 0.114])
        # Умножаем каждый канал на соответствующий коэффициент и суммируем по оси каналов
        result = np.sum(self._image * eye_vector, axis=2)
        return result.astype(np.uint8)
    
    def to_grayscale_artwork(self, metadata: Optional[Metadata] = None) -> 'GrayscaleArtwork':
        """Преобразование в GrayscaleArtwork"""
        gray_image = self.grayscale()
        if metadata is None:
            metadata = Metadata(
                object_id=f"{self._metadata.object_id}_gray",
                title=f"{self._metadata.title} (grayscale)",
                artist=self._metadata.artist,
                date=self._metadata.date
            )
        return GrayscaleArtwork(gray_image, metadata, self._image_path)


class GrayscaleArtwork(Artwork):
    """Класс для черно-белых изображений"""
    __slots__ = ('_bit_depth',)
    
    def __init__(self, image: np.ndarray, metadata: Metadata, image_path: str = "", bit_depth: int = 8):
        super().__init__(image, metadata, image_path)
        self._bit_depth = bit_depth
    
    def _validate_image(self) -> None:
        """Проверка, что изображение черно-белое (2D массив)"""
        if len(self._image.shape) != 2:
            raise ValueError(f"GrayscaleArtwork ожидает 2D изображение, получено: {self._image.shape}")
    
    @property
    def bit_depth(self) -> int:
        return self._bit_depth
    
    def __add__(self, other: Artwork) -> 'GrayscaleArtwork':
        """Сложение двух черно-белых изображений"""
        if not isinstance(other, GrayscaleArtwork):
            raise TypeError("Можно складывать только черно-белые изображения")
        
        if self._image.shape != other._image.shape:
            raise ValueError("Изображения должны иметь одинаковую размерность для сложения")
        
        alpha = 0.5
        beta = 1 - alpha
        blended = cv2.addWeighted(self._image, alpha, other._image, beta, 0)
        
        combined_metadata = Metadata(
            object_id=f"{self._metadata.object_id}_{other._metadata.object_id}",
            title=f"{self._metadata.title} + {other._metadata.title}",
            raw_data={"source1": self._metadata.raw_data, "source2": other._metadata.raw_data}
        )
        return GrayscaleArtwork(blended, combined_metadata, bit_depth=self._bit_depth)
    
    def grayscale(self) -> np.ndarray:
        """Для черно-белого изображения возвращает само себя"""
        return self._image
    
    def to_color_artwork(self) -> ColorArtwork:
        """Преобразование в ColorArtwork (дублирование каналов)"""
        color_image = cv2.cvtColor(self._image, cv2.COLOR_GRAY2BGR)
        metadata = Metadata(
            object_id=f"{self._metadata.object_id}_color",
            title=f"{self._metadata.title} (colorized)",
            artist=self._metadata.artist,
            date=self._metadata.date
        )
        return ColorArtwork(color_image, metadata, self._image_path)