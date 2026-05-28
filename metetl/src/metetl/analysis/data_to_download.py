"""Подготовка списка изображений для скачивания."""

import csv, json, random
from pathlib import Path
from typing import List, Dict, Any, Optional

from metetl.logging_config import get_logger
from metetl.decorators import timing

logger = get_logger(__name__)


@timing()
def prepare_download_list(csv_path: str, output_path: str, limit: int = None, seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """Подготовка JSON файла с метаданными изображений для скачивания.
    
    Args:
        csv_path: Путь к CSV файлу MetObjects.csv
        output_path: Путь для сохранения JSON файла
        limit: Максимальное количество записей (выбираются случайные)
        seed: Seed для генератора случайных чисел (для воспроизводимости)
        
    Returns:
        List[Dict]: Список метаданных для скачивания
    """
    logger.info(f"Подготовка списка для скачивания из {csv_path}")
    
    all_paintings = []
    csv_file = Path(csv_path)
    
    if not csv_file.exists():
        logger.error(f"CSV файл не найден: {csv_path}")
        raise FileNotFoundError(f"CSV файл не найден: {csv_path}")
    
    try:
        # Сначала собираем ВСЕ картины из CSV
        with open(csv_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                if row.get('Classification') == 'Paintings' and row.get('Object ID'):
                    painting = {
                        'object_id': row.get('Object ID'),
                        'title': row.get('Title'),
                        'artist': row.get('Artist Display Name'),
                        'date': row.get('Object Date'),
                        'medium': row.get('Medium'),
                        'dimensions': row.get('Dimensions'),
                        'department': row.get('Department'),
                        'culture': row.get('Culture')
                    }
                    all_paintings.append(painting)
        
        logger.info(f"Всего найдено картин в CSV: {len(all_paintings)}")
        
        # Выбираем случайные изображения
        if limit and limit < len(all_paintings):
            if seed is not None:
                random.seed(seed)
                logger.debug(f"Установлен seed = {seed} для воспроизводимости")
            
            paintings = random.sample(all_paintings, limit)
            logger.info(f"Выбрано {limit} случайных картин из {len(all_paintings)}")
        else:
            paintings = all_paintings
            if limit:
                logger.warning(f"Запрошено {limit} картин, но доступно только {len(all_paintings)}. Будут использованы все доступные.")
            else:
                logger.info(f"Будут использованы все {len(all_paintings)} картин")
        
        # Сохраняем JSON
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(paintings, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Метаданные сохранены в {output_path}")
        
        return paintings
        
    except Exception as e:
        logger.error(f"Ошибка при обработке CSV: {e}")
        raise