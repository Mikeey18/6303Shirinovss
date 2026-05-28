import os, csv, json, random, requests, numpy as np, cv2, time
from typing import List, Dict, Any, Optional, Callable
from functools import wraps
from artwork import Artwork, Metadata


def timer_decorator(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000
        print(f"[TIMER] {func.__name__}: {execution_time:.4f} ms")
        return result
    return wrapper


class ImageProcessor:
    __slots__ = ('_paintings_dir', '_csv_path', '_artworks')
    
    def __init__(self, csv_path: str = './data/MetObjects.csv'):
        self._paintings_dir = self._setup_directories()
        self._csv_path = csv_path
        self._artworks: List[Artwork] = []
    
    @staticmethod
    def _setup_directories() -> str:
        paintings_dir = os.path.join(os.getcwd(), 'paintings')
        os.makedirs(paintings_dir, exist_ok=True)
        return paintings_dir
    
    @timer_decorator
    def download_random_painting(self) -> Optional[Artwork]:
        print("\n[PROCESS] Скачка случайного изображения...")
        
        all_objects = self._read_csv_file()
        print(f"[PROCESS] Загружено {len(all_objects)} объектов из CSV")
        
        paintings = self._filter_paintings(all_objects)
        print(f"[PROCESS] Найдено {len(paintings)} картин")
        
        if not paintings:
            print("[ERROR] Картины не найдены")
            return None
        
        random_painting = random.choice(paintings)
        object_id = random_painting.get('Object ID')
        if not object_id:
            print("[ERROR] ID объекта не получен")
            return None
        
        print(f"[PROCESS] Выбран объект с Object ID: {object_id}")
        
        details = self._get_object_details(object_id)
        if not details:
            print("[ERROR] Детали объекта не получены")
            return None
        
        image_url = details.get('primaryImage')
        if not image_url:
            print("[ERROR] У объекта нет изображения")
            return None
        
        print(f"[PROCESS] Ссылка на изображение: {image_url}")
        
        image_path = os.path.join(self._paintings_dir, f"{object_id}.jpg")
        json_path = os.path.join(self._paintings_dir, f"{object_id}.json")
        
        if not self._download_image(image_url, image_path):
            print("[ERROR] Ошибка при скачке изображения")
            return None
        
        print(f"[PROCESS] Изображение сохранено: {image_path}")
        
        if not self._save_json(details, json_path):
            print("[WARNING] Ошибка при сохранении JSON метаданных")
        
        # Load image as numpy array
        image_array = cv2.imread(image_path)
        if image_array is None:
            print("[ERROR] Ошибка при загрузке скаченного изображения")
            return None
        
        # Create metadata
        metadata = Metadata(
            object_id=object_id,
            title=details.get('title'),
            artist=details.get('artistDisplayName'),
            date=details.get('objectDate'),
            medium=details.get('medium'),
            dimensions=details.get('dimensions'),
            raw_data=details
        )
        
        artwork = Artwork(image_array, metadata, image_path)
        self._artworks.append(artwork)
        
        print("[PROCESS] Скачка завершена успешно")
        return artwork
    
    def _read_csv_file(self) -> List[Dict[str, str]]:
        try:
            with open(self._csv_path, mode='r', encoding='utf-8-sig') as file:
                reader = csv.DictReader(file)
                return list(reader)
        except Exception as e:
            print(f"[ERROR] Чтение CSV: {e}")
            return []
    
    @staticmethod
    def _filter_paintings(objects: List[Dict[str, str]]) -> List[Dict[str, str]]:
        return [obj for obj in objects if obj.get('Classification') == 'Paintings']
    
    @staticmethod
    def _get_object_details(object_id: str) -> Optional[Dict[str, Any]]:
        url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"
        try:
            response = requests.get(url)
            return response.json()
        except Exception as e:
            print(f"[ERROR] Ошибка получения деталей объекта: {e}")
            return None
    
    @timer_decorator
    def _download_image(self, image_url: str, save_path: str) -> bool:
        try:
            response = requests.get(image_url, stream=True)
            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=16384):
                    file.write(chunk)
            return True
        except Exception as e:
            print(f"[ERROR] Ошибка скачки изображения: {e}")
            return False
    
    @staticmethod
    def _save_json(data: Dict[str, Any], filepath: str) -> bool:
        try:
            with open(filepath, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"[ERROR] Ошибка сохранения JSON: {e}")
            return False
    
    @timer_decorator
    def process_artwork(self, artwork: Artwork, operation: str, **kwargs) -> Optional[np.ndarray]:
        print(f"\n[PROCESS] Применение операции {operation} к изображению {artwork.metadata.object_id}")
        
        operations = {
            'grayscale': artwork.grayscale,
            'convolution': lambda: artwork.convolution(kwargs.get('kernel')), # lambda "анонимная функция"
            'gaussian_blur': lambda: artwork.gaussian_blur(kwargs.get('kernel_size', 3)),
            'sobel': artwork.sobel,
            'gamma_correction': lambda: artwork.gamma_correction(kwargs.get('gamma', 1.0)),
            'histogram_equalization': artwork.histogram_equalization
        }
        
        if operation not in operations:
            print(f"[ERROR] Ошибка незивестная операция: {operation}")
            return None
        
        try:
            result = operations[operation]()
            print(f"[PROCESS] Операция {operation} завершена успешно")
            return result
        except Exception as e:
            print(f"[ERROR] Ошибка обработки изображения: {e}")
            return None
    
    def save_result(self, result: np.ndarray, artwork: Artwork, operation_suffix: str) -> str:
        filename = os.path.basename(artwork.image_path)
        name_without_ext = os.path.splitext(filename)[0]
        save_path = os.path.join(self._paintings_dir, f"{name_without_ext}_{operation_suffix}.jpg")
        
        cv2.imwrite(save_path, result)
        print(f"[PROCESS] Результат сохранен: {save_path}")
        return save_path
    
    def compare_with_library(self, artwork: Artwork, operation: str, **kwargs) -> Dict[str, Any]:
        print(f"\n[COMPARE] Сравнение операции {operation} с библиотечной")

        start_time = time.perf_counter()
        custom_result = self.process_artwork(artwork, operation, **kwargs)
        my_time = (time.perf_counter() - start_time) * 1000

        start_time = time.perf_counter()
        lib_result = self._library_implementation(artwork, operation, **kwargs)
        lib_time = (time.perf_counter() - start_time) * 1000
        
        if custom_result is None or lib_result is None:
            print("[ERROR] Ошибка сравнения - одна или обе функции обработки выдали ошибку")
            return {}
        
        custom_path = self.save_result(custom_result, artwork, f"{operation}_custom")
        lib_path = self.save_result(lib_result, artwork, f"{operation}_lib")
    
        ratio = lib_time / my_time
        if ratio > 1:
            print(f"[COMPARE] Библиотека медленнее в {ratio:.2f} раз")
        else:
            print(f"[COMPARE] Библиотека быстрее в {1/ratio:.2f} раз")
        
        return {
            'operation': operation,
            'custom_path': custom_path,
            'lib_path': lib_path
        }
    
    @timer_decorator
    def _library_implementation(self, artwork: Artwork, operation: str, **kwargs) -> Optional[np.ndarray]:
        image = artwork.image
        
        try:
            if operation == 'grayscale':
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif operation == 'convolution':
                return cv2.filter2D(image, -1, kwargs.get('kernel'))
            elif operation == 'gaussian_blur':
                kernel_size = kwargs.get('kernel_size', 3)
                return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            elif operation == 'sobel':
                grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
                grad = np.sqrt(grad_x**2 + grad_y**2)
                grad = (grad / grad.max()) * 255
                return grad.astype(np.uint8)
            elif operation == 'gamma_correction':
                gamma = kwargs.get('gamma', 1.0)
                look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
                return cv2.LUT(image, look_up_table)
            elif operation == 'histogram_equalization':
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l_channel, a_channel, b_channel = cv2.split(lab)
                l_equalized = cv2.equalizeHist(l_channel)
                lab_equalized = cv2.merge([l_equalized, a_channel, b_channel])
                return cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)
        except Exception as e:
            print(f"[ERROR] Ошибка при библиотечной обработки изображения: {e}")
            return None
    
    @property
    def artworks(self) -> List[Artwork]:
        return self._artworks