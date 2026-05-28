"""Асинхронная загрузка и параллельная обработка изображений."""

import os, csv, json, asyncio, aiohttp, aiofiles, numpy as np, cv2, time, multiprocessing as mp, concurrent, random
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from metetl.logging_config import get_logger
from metetl.decorators import timing, async_timing
from metetl.images.models import Artwork, Metadata

logger = get_logger(__name__)


def process_image_file(index: int, image_path: str, metadata_path: str, params: Dict) -> Tuple[int, Optional[str], str]:
    """Функция для параллельной обработки изображения из файла.
    
    Args:
        index: Индекс изображения
        image_path: Путь к оригинальному изображению
        metadata_path: Путь к JSON с метаданными
        params: Параметры обработки
            - operation: тип операции
            - kernel_size: размер ядра (для Gaussian blur)
            - gamma: значение гаммы (для gamma_correction)
    
    Returns:
        Tuple[int, Optional[str], str]: (index, путь_к_обработанному_файлу, object_id)
    """
    logger_proc = get_logger(__name__) # Создается внутри функции, которая выполняется в отдельном процессе. Обычный глобальный logger может работать некорректно при передаче между процессами.
    logger_proc.debug(f"Обработка изображения {index} начата (PID: {os.getpid()})")
    start_time = time.time()
    
    try:
        # Загружаем изображение
        image_array = cv2.imread(image_path)
        if image_array is None:
            logger_proc.error(f"Изображение {index}: не удалось загрузить")
            return (index, None, "")
        
        # Загружаем метаданные
        with open(metadata_path, 'r', encoding='utf-8') as f:
            details = json.load(f)
        
        # Создаем метаданные
        metadata = Metadata(
            object_id=details.get('objectID', str(index)),
            title=details.get('title'),
            artist=details.get('artistDisplayName'),
            date=details.get('objectDate'),
            medium=details.get('medium'),
            dimensions=details.get('dimensions'),
            raw_data=details
        )
        
        # Создаем объект Artwork
        artwork = Artwork(image_array, metadata, image_path, index)
        
        # Применяем выбранную операцию
        operation = params.get('operation', 'gaussian_blur')

        if operation == 'gaussian_blur':
            kernel_size = params.get('kernel_size', 5)
            # Убеждаемся, что kernel_size нечетный
            if kernel_size % 2 == 0:
                kernel_size += 1
                logger_proc.debug(f"  kernel_size скорректирован до {kernel_size} (должен быть нечетным, чтобы у матрицы был центр)")
            result = artwork.gaussian_blur(kernel_size)
            logger_proc.debug(f"  Применено Gaussian blur (kernel_size={kernel_size})")
        elif operation == 'sobel':
            result = artwork.sobel()
            logger_proc.debug(f"  Применен детектор Собеля")
        elif operation == 'gamma_correction':
            gamma = params.get('gamma', 1.0)
            result = artwork.gamma_correction(gamma)
            logger_proc.debug(f"  Применена гамма-коррекция (gamma={gamma})")
        elif operation == 'histogram_equalization':
            result = artwork.histogram_equalization()
            logger_proc.debug(f"  Применена эквализация гистограммы")
        else:
            # По умолчанию - Gaussian blur
            kernel_size = params.get('kernel_size', 5)
            result = artwork.gaussian_blur(kernel_size)
            logger_proc.debug(f"  Применен Gaussian blur по умолчанию (kernel_size={kernel_size})")
        
        elapsed_time = (time.time() - start_time) * 1000
        logger_proc.debug(f"Обработка изображения {index} завершена за {elapsed_time:.2f}ms")
        
        # Сохраняем результат
        save_path = os.path.join(os.path.dirname(image_path), f"{index}_{metadata.object_id}_processed.jpg")
        cv2.imwrite(save_path, result)
        
        return (index, save_path, metadata.object_id)
        
    except Exception as e:
        logger_proc.error(f"Ошибка обработки изображения {index}: {e}", exc_info=True)
        return (index, None, "")


class ImageProcessor:
    """Асинхронный процессор для загрузки и обработки изображений."""
    
    def __init__(self, num_images: int = 1, output_dir: str = "images", num_workers: int = None):
        self._output_dir = self._setup_directories(output_dir)
        self._artworks: List[Artwork] = []
        self._num_images = num_images
        self._num_workers = num_workers if num_workers else min(mp.cpu_count(), num_images)
        logger.info(f"Инициализация процессов: {num_images} изображений, {self._num_workers} процессов")
    
    @staticmethod
    def _setup_directories(output_dir: str) -> str:
        """Создание директории для изображений."""
        paintings_dir = Path(output_dir)
        paintings_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Директория для изображений: {paintings_dir}")
        return str(paintings_dir)
    
    @async_timing()
    async def download_images_async(self, paintings_list: List[Dict]) -> List[Tuple[int, str, str]]:
        """Асинхронная загрузка нескольких изображений."""
        if (self._num_images > len(paintings_list)):
            logger.info(f"В файле метаданных не хватает {self._num_images - len(paintings_list)} изображений")
            logger.info(f"Начинаем асинхронную загрузку {len(paintings_list)} изображений")
        else:
            logger.info(f"Начинаем асинхронную загрузку {min(len(paintings_list), self._num_images)} изображений")
        
        tasks = []
        for idx, painting in enumerate(paintings_list[:self._num_images], 1):
            logger.debug(f"Создана задача для изображения {idx} (ID: {painting['object_id']})")
            task = self._download_single_image(idx, painting)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        successful = [r for r in results if r is not None]
        logger.info(f"Загрузка завершена. Успешно: {len(successful)} из {self._num_images}")
        
        return successful
    
    @async_timing()
    async def _download_single_image(self, index: int, painting: Dict) -> Optional[Tuple[int, str, str]]:
        """Асинхронная загрузка одного изображения."""
        object_id = painting['object_id']
        logger.debug(f"Загрузка изображения {index} (ID: {object_id})")
        
        async with aiohttp.ClientSession() as session:
            url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"
            
            try:
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.warning(f"HTTP {response.status} для ID {object_id}")
                        return None
                    
                    details = await response.json()
                    
                    image_url = details.get('primaryImage')
                    if not image_url:
                        logger.warning(f"Изображение {index}: нет URL изображения")
                        json_path = os.path.join(self._output_dir, f"{index}_{object_id}_metadata.json")
                        async with aiofiles.open(json_path, 'w', encoding='utf-8') as f:
                            await f.write(json.dumps(details, indent=2, ensure_ascii=False))
                        return None
                    
                    image_path = os.path.join(self._output_dir, f"{index}_{object_id}_original.jpg")
                    json_path = os.path.join(self._output_dir, f"{index}_{object_id}_metadata.json")
                    
                    async with aiofiles.open(json_path, 'w', encoding='utf-8') as f:
                        await f.write(json.dumps(details, indent=2, ensure_ascii=False))
                    
                    async with session.get(image_url) as img_response:
                        if img_response.status == 200:
                            async with aiofiles.open(image_path, 'wb') as f:
                                await f.write(await img_response.read())
                            logger.debug(f"Загрузка изображения {index} завершена")
                            return (index, image_path, json_path)
                        else:
                            logger.warning(f"Изображение {index}: ошибка HTTP {img_response.status}")
                            return None
                            
            except Exception as e:
                logger.error(f"Изображение {index}: ошибка загрузки - {e}")
                return None
    
    @async_timing()
    async def process_images_parallel(self, images_data: List[Tuple[int, str, str]], processing_params: Dict[str, Any]) -> List[Tuple]:
        """Параллельная обработка изображений с помощью ProcessPoolExecutor.
        
        Args:
            images_data: Список кортежей (index, image_path, json_path)
            processing_params: Параметры обработки изображений
                - operation: тип операции ('gaussian_blur', 'sobel', 'gamma_correction', 'histogram_equalization')
                - kernel_size: размер ядра для Gaussian blur
                - gamma: значение гаммы для гамма-коррекции
        """
        logger.info(f"Начинаем параллельную обработку {len(images_data)} изображений")
        logger.info(f"Параметры обработки: {processing_params}")
        
        if not images_data:
            logger.error("Нет данных для обработки")
            return []
        
        # Используем параметры, переданные пользователем
        params = {
            'operation': processing_params.get('operation', 'histogram_equalization'),
            'kernel_size': processing_params.get('kernel_size', 5),
            'gamma': processing_params.get('gamma', 1.0)
        }
        
        args_list = []
        for index, image_path, json_path in images_data:
            args_list.append((index, image_path, json_path, params))

        # Запускаем синхронную функцию в отдельных процессах, чтобы не блокировать event loop
        with concurrent.futures.ProcessPoolExecutor(max_workers=self._num_workers) as executor:
            loop = asyncio.get_event_loop()
            task = []
            for i in args_list:
                future = loop.run_in_executor(
                    executor,
                    process_image_file, # Используемая фукнция
                    *i
                )
                task.append(future)
            results = await asyncio.gather(*task)
        
        successful_count = sum(1 for r in results if r[1] is not None)
        logger.info(f"Параллельная обработка завершена. Успешно: {successful_count} из {len(results)}")

        # Загружаем Artwork объекты для сохранения в self._artworks
        for (index, image_path, json_path), result in zip(images_data, results):
            try:
                # Если обработка была успешной, result содержит (index, save_path, object_id)
                if result[1] is not None:
                    image_array = cv2.imread(image_path)
                    with open(json_path, 'r', encoding='utf-8') as f:
                        details = json.load(f)
                    
                    metadata = Metadata(
                        object_id=details.get('objectID', str(index)),
                        title=details.get('title'),
                        artist=details.get('artistDisplayName'),
                        date=details.get('objectDate'),
                        medium=details.get('medium'),
                        dimensions=details.get('dimensions'),
                        raw_data=details
                    )
                    
                    artwork = Artwork(image_array, metadata, image_path, index)
                    self._artworks.append(artwork)
            except Exception as e:
                logger.error(f"Не удалось загрузить Artwork для {index}: {e}")
        
        return results
    
    @async_timing()
    async def run_pipeline(self, paintings_list: List[Dict], processing_params: Optional[Dict[str, Any]] = None) -> List[Artwork]:
        """Запуск полного пайплайна обработки.
        
        Args:
            paintings_list: Список картин для загрузки
            processing_params: Параметры обработки изображений
                - operation: тип операции
                - kernel_size: размер ядра для Gaussian blur
                - gamma: значение гаммы для gamma_correction
        """
        if processing_params is None:
            # Параметры по умолчанию
            processing_params = {
                'operation': 'histogram_equalization',
                'kernel_size': 5,
                'gamma': 1.0
            }
        
        random.shuffle(paintings_list) # Перемешка списка со словарями метадаты картинок

        total_start = time.time()
        
        # Шаг 1: Асинхронная загрузка изображений
        images_data = await self.download_images_async(paintings_list)
        
        if not images_data:
            logger.error("Нет загруженных изображений")
            return []
        
        # Шаг 2: Параллельная обработка с переданными параметрами
        processed_results = await self.process_images_parallel(images_data, processing_params)
        
        total_time = (time.time() - total_start) * 1000
        logger.info(f"Общее время выполнения: {total_time:.2f} ms")
        
        return self._artworks
        
       