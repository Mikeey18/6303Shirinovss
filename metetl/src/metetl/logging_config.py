"""Конфигурация логирования для проекта."""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler


def setup_logging(log_dir: str = "logs", log_file: str = "app.log"):
    """Настройка логирования в файл и консоль."""
    # Создаем директорию для логов
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    full_log_path = log_path / log_file
    
    # Создаем корневой логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Ограничиваем DEBUG для сторонних библиотек
    # Устанавливаем для них уровень INFO или выше
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    
    # Форматы
    file_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(pathname)s:%(lineno)d | %(funcName)s() | %(message)s', # - выравнивание влево 8 ширина поля s строковый тип
        # %(pathname)s:%(lineno)d путь к файлу, номер строки, %(funcName)s() - имя функции, %(message)s - само сообщение
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Очищаем существующие обработчики
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Файловый обработчик
    file_handler = RotatingFileHandler(
        full_log_path, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8' # 10мб - макс размер, хранить 5 старых файлов
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Консольный обработчик
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    root_logger.info("=" * 60)
    root_logger.info("Логирование инициализировано")
    root_logger.debug(f"Лог-файл: {full_log_path.absolute()}")
    root_logger.info("=" * 60)

def get_logger(name: str) -> logging.Logger:
    """Получить логгер с указанным именем."""
    return logging.getLogger(name)