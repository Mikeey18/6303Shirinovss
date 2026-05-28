import os, csv, json, asyncio, aiohttp, aiofiles, numpy as np, cv2, random, concurrent
from typing import List, Dict, Any, Optional, Tuple
import time
import multiprocessing as mp
from artwork import Artwork, Metadata
from config import Config


def process_image_file(index, image_path, metadata_path, params):
    """
    Функция для параллельной обработки изображения из файла
    Находится на верхнем уровне для возможности pickle
    """
    print(f"[PROCESS] Свертка для изображения {index} начата (PID: {os.getpid()})")
    start_time = time.time()
    
    try:
        # Загружаем изображение
        image_array = cv2.imread(image_path)
        if image_array is None:
            print(f"[ERROR] Изображение {index}: не удалось загрузить")
            return (index, None, "")
        
        # Загружаем метаданные из JSON
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
        
        operation = params.get('operation', 'gaussian_blur')
        kernel_size = params.get('kernel_size', 5)
        
        if operation == 'gaussian_blur':
            result = artwork.gaussian_blur(kernel_size)
        elif operation == 'sobel':
            result = artwork.sobel()
        elif operation == 'gamma_correction':
            result = artwork.gamma_correction(params.get('gamma', 1.0))
        elif operation == 'histogram_equalization':
            result = artwork.histogram_equalization()
        else:
            kernel = params.get('kernel')
            if kernel is None:
                kernel = artwork._get_gaussian_kernel(kernel_size, kernel_size/6)
            result = artwork.convolution(kernel)
        
        elapsed_time = (time.time() - start_time) * 1000
        print(f"[PROCESS] Свертка для изображения {index} завершена за {elapsed_time:.2f}ms (PID: {os.getpid()})")
        
        # Сохраняем результат
        save_path = os.path.join(os.path.dirname(image_path), f"{index}_{metadata.object_id}_processed.jpg")
        cv2.imwrite(save_path, result)
        
        return (index, save_path, metadata.object_id)
    except Exception as e:
        print(f"[ERROR] Ошибка обработки изображения {index}: {e}")
        import traceback
        traceback.print_exc()
        return (index, None, "")


class AsyncImageProcessor:
    def __init__(self, num_images: int = 1):
        self._paintings_dir = self._setup_directories()
        self._artworks: List[Artwork] = []
        self._num_images = num_images
        # Не создаем ProcessPoolExecutor здесь, так как он может вызвать проблемы
        self._num_workers = min(mp.cpu_count(), num_images)
        print(f"[INIT] Будет использовано {self._num_workers} процессов для параллельной обработки")
    
    @staticmethod
    def _setup_directories() -> str:
        paintings_dir = Config.PAINTINGS_DIR
        os.makedirs(paintings_dir, exist_ok=True)
        return paintings_dir
    
    async def download_images_async(self) -> List[Tuple[int, str, str]]: # индекс, путь к изображению, путь к json
        """Асинхронная загрузка нескольких изображений"""
        print(f"\n[ASYNC] Начинаем асинхронную загрузку {self._num_images} изображений")
        
        # Получаем список ID картин
        painting_ids = await self._get_painting_ids_async() # Приостанавливаем корутину пока выполняется _get_painting_ids_async
        if not painting_ids:
            print("[ERROR] Не удалось получить ID картин")
            return []
        
        # Создаем задачи для загрузки
        tasks = []
        for idx, obj_id in enumerate(painting_ids[:self._num_images], 1):
            print(f"[ASYNC] Создана задача для изображения {idx} (ID: {obj_id})")
            task = self._download_single_image(idx, obj_id) # Для каждого ID создаем корутину загрузки и добавляем в список
            tasks.append(task)
        
        # Выполняем все задачи асинхронно
        results = await asyncio.gather(*tasks) # Распаковываем список аргументов и запускаем параллельно задачи и ждем их завершения
        
        # Фильтруем успешные загрузки
        successful = [r for r in results if r is not None]
        print(f"[ASYNC] Загрузка завершена. Успешно: {len(successful)} из {self._num_images}")
        
        return successful
    
    async def _get_painting_ids_async(self) -> List[str]:
        """Асинхронное получение случайных ID картин из CSV"""
        try:
            # Проверяем существование файла CSV
            if not os.path.exists(Config.CSV_PATH):
                print(f"[ERROR] Файл CSV не найден: {Config.CSV_PATH}")
                return []

            # Собираем ВСЕ ID картин
            all_painting_ids = []

            async with aiofiles.open(Config.CSV_PATH, mode='r', encoding='utf-8-sig') as f:
                content = await f.read()
                lines = content.splitlines()
                reader = csv.DictReader(lines)

                for row in reader:
                    if row.get('Classification') == 'Paintings' and row.get('Object ID'):
                        all_painting_ids.append(row.get('Object ID'))

            print(f"[ASYNC] Всего найдено картин в CSV: {len(all_painting_ids)}")

            # Проверяем, что у нас достаточно картин
            if len(all_painting_ids) < self._num_images:
                print(f"[WARNING] В CSV только {len(all_painting_ids)} картин, а запрошено {self._num_images}")
                print(f"[WARNING] Будет загружено {len(all_painting_ids)} картин")
                self._num_images = len(all_painting_ids)

            # Выбираем случайные ID (без повторений)
            painting_ids = random.sample(all_painting_ids, self._num_images)

            print(f"[ASYNC] Выбрано {len(painting_ids)} случайных ID картин")
            for idx, pid in enumerate(painting_ids[:5], 1):  # Показываем первые 5 для информации
                print(f"  {idx}. {pid}")
            if len(painting_ids) > 5:
                print(f"  ... и еще {len(painting_ids) - 5}")

            return painting_ids
        
        except Exception as e:
            print(f"[ERROR] Чтение CSV: {e}")
            return []

    async def _get_object_details_async(self, session: aiohttp.ClientSession, object_id: str) -> Optional[Dict[str, Any]]: # Optional - вернем словарь или none
        """Асинхронное получение деталей объекта"""
        url = f"{Config.MET_API_URL}/objects/{object_id}"
        try:
            async with session.get(url) as response: # отправка get-запроса
                if response.status == 200:
                    return await response.json() # асинхронное чтение и парсинг json
                else:
                    print(f"[WARNING] HTTP {response.status} для ID {object_id}")
                    return None
        except Exception as e:
            print(f"[ERROR] Ошибка получения деталей для {object_id}: {e}")
            return None
    
    async def _download_single_image(self, index: int, object_id: str) -> Optional[Tuple[int, str, str]]:
        """Асинхронная загрузка одного изображения"""
        print(f"[DOWNLOAD] Загрузка изображения {index} начата (ID: {object_id})")
        
        async with aiohttp.ClientSession() as session: # Создаёт асинхронную HTTP сессию. Сессия переиспользуется для нескольких запросов (деталей объекта и изображения).
            # Получаем детали
            details = await self._get_object_details_async(session, object_id)
            if not details:
                print(f"[ERROR] Изображение {index}: не получены детали")
                return None
            
            image_url = details.get('primaryImage')
            if not image_url:
                print(f"[ERROR] Изображение {index}: нет URL изображения")
                # Сохраняем JSON даже без изображения для отладки
                json_path = os.path.join(self._paintings_dir, f"{index}_{object_id}_metadata.json")
                async with aiofiles.open(json_path, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(details, indent=2, ensure_ascii=False))
                return None
            
            image_path = os.path.join(self._paintings_dir, f"{index}_{object_id}_original.jpg")
            json_path = os.path.join(self._paintings_dir, f"{index}_{object_id}_metadata.json")
            
            # Сохраняем JSON с метаданными
            async with aiofiles.open(json_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(details, indent=2, ensure_ascii=False)) # indent=2 - красивое форматирование. ensure_ascii=False - поддержка Unicode, русские символы.
            
            # Скачиваем изображение
            try:
                async with session.get(image_url) as response:
                    if response.status == 200:
                        async with aiofiles.open(image_path, 'wb') as f: # Бинарный режим записи
                            await f.write(await response.read()) # Асинхронное чтение тела ответа и асинхронная запись
                        print(f"[DOWNLOAD] Загрузка изображения {index} завершена")
                        return (index, image_path, json_path)
                    else:
                        print(f"[ERROR] Изображение {index}: ошибка HTTP {response.status}")
                        return None
            except Exception as e:
                print(f"[ERROR] Изображение {index}: ошибка загрузки - {e}")
                return None

    # def _process_images_sync(self, images_data, params): # Этот метод будет вызван в отдельном потоке, чтобы не блокировать asyncio
    #     """Синхронная обработка изображений в пуле процессов"""
    #     # Подготавливаем аргументы для каждого изображения
    #     args_list = []
    #     for index, image_path, json_path in images_data:
    #         args_list.append((index, image_path, json_path, params))
        
    #     # Используем multiprocessing.Pool для параллельной обработки
    #     with mp.Pool(processes=self._num_workers) as pool: # with гарантирует закрытие пула
    #         # Создаёт процессы и ждёт их завершения - блокирующий код. Все это врем event loop заблокирован
    #         results = pool.starmap(process_image_file, args_list) # starmap - как map, но распаковывает кортежи в аргументы
        
    #     return results
    
    async def process_images_parallel(self, images_data: List[Tuple[int, str, str]]):
        """Параллельная обработка изображений с помощью multiprocessing.Pool"""
        print(f"\n[PARALLEL] Начинаем параллельную обработку {len(images_data)} изображений")
        
        if not images_data:
            print("[ERROR] Нет данных для обработки")
            return []
        
        # ВЫБОР ОБРАБОТКИ
        params = {
            'operation': 'histogram_equalization',
            'kernel_size': 5
        }

        args_list = []
        for index, image_path, json_path in images_data:
            args_list.append((index, image_path, json_path, params))

        # results = self._process_images_sync(images_data, params)

        # # Запускаем синхронную функцию в отдельном потоке, чтобы не блокировать event loop
        # loop = asyncio.get_event_loop()
        # results = await loop.run_in_executor(
        #     None,  # Используем стандартный ThreadPoolExecutor
        #     self._process_images_sync, # Используемая фукнция
        #     images_data, # Параметры 
        #     params
        # )

        # Запускаем синхронную функцию в отдельном процессе, чтобы не блокировать event loop
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
        
        # """Синхронная обработка изображений в пуле процессов"""
        # # Подготавливаем аргументы для каждого изображения
        # args_list = []
        # for index, image_path, json_path in images_data:
        #     args_list.append((index, image_path, json_path, params))
        
        # # Используем multiprocessing.Pool для параллельной обработки
        # with mp.Pool(processes=self._num_workers) as pool: # with гарантирует закрытие пула
        #     # Создаёт процессы и ждёт их завершения - блокирующий код. Все это врем event loop заблокирован
        #     results = pool.starmap(process_image_file, args_list) # starmap - как map, но распаковывает кортежи в аргументы

        print(f"[PARALLEL] Параллельная обработка завершена. Обработано: {len(results)} изображений")
        
        # Загружаем Artwork объекты для сохранения в self._artworks
        for index, image_path, json_path in images_data:
            try:
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
                print(f"[ERROR] Не удалось загрузить Artwork для {index}: {e}")
        
        return results















    async def run_pipeline(self):
        """Запуск полного пайплайна обработки"""
        total_start = time.time()
        
        # Шаг 1: Асинхронная загрузка изображений
        download_start = time.time()
        images_data = await self.download_images_async()
        download_time = (time.time() - download_start) * 1000
        print(f"\n[TIMER] Асинхронная загрузка: {download_time:.2f} ms")
        
        if not images_data:
            print("[ERROR] Нет загруженных изображений")
            return
        
        # Шаг 2: Параллельная обработка сверткой
        process_start = time.time()
        processed_results = await self.process_images_parallel(images_data)
        process_time = (time.time() - process_start) * 1000
        print(f"[TIMER] Параллельная обработка: {process_time:.2f} ms")
        
        total_time = (time.time() - total_start) * 1000
        print(f"\n[TIMER] Общее время выполнения: {total_time:.2f} ms")
        
        # Оценка ускорения
        if len(images_data) > 0:
            sequential_estimate = download_time + (process_time * len(images_data))
            speedup = sequential_estimate / total_time if total_time > 0 else 1
            print(f"[TIMER] Ориентировочное ускорение: {speedup:.2f}x (при {len(images_data)} изображениях)")
        
        # Вывод информации о результатах
        successful_results = [r for r in processed_results if r[1] is not None]
        print(f"\n[RESULT] Успешно обработано: {len(successful_results)} из {len(processed_results)} изображений")
        
        for result in successful_results:
            index, save_path, object_id = result
            if save_path:
                print(f"  - Изображение {index} (ID: {object_id}): {os.path.basename(save_path)}")
    
    @property
    def artworks(self) -> List[Artwork]:
        return self._artworks