"""Декораторы для логирования и измерения времени."""

import time, functools
from metetl.logging_config import get_logger


def timing(logger_name: str = None):
    """Декоратор для измерения времени выполнения функции.
    
    Args:
        logger_name: Имя логгера (если None - используем логгер функции)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name or func.__module__)
            logger.debug(f"Запуск {func.__name__}")
            start = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = (time.time() - start) * 1000
                logger.debug(f"Завершение {func.__name__} за {elapsed:.2f}ms")
                return result
            except Exception as e:
                elapsed = (time.time() - start) * 1000
                logger.error(f"Ошибка в {func.__name__} после {elapsed:.2f}ms: {e}")
                raise
        return wrapper
    return decorator


def async_timing(logger_name: str = None):
    """Декоратор для измерения времени выполнения асинхронной функции."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            logger = get_logger(logger_name or func.__module__)
            logger.debug(f"Запуск async {func.__name__}")
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                elapsed = (time.time() - start) * 1000
                logger.debug(f"Завершение async {func.__name__} за {elapsed:.2f}ms")
                return result
            except Exception as e:
                elapsed = (time.time() - start) * 1000
                logger.error(f"Ошибка в async {func.__name__} после {elapsed:.2f}ms: {e}")
                raise
        return wrapper
    return decorator