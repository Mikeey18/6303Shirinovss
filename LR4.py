import asyncio
import sys
import os


async def main_async(num_images: int):
    """Асинхронная главная функция"""
    from async_image_processor import AsyncImageProcessor
    
    print(f"=== Запуск LR4 с {num_images} изображениями ===")
    print(f"Доступно ядер CPU: {os.cpu_count()}\n")
    
    processor = AsyncImageProcessor(num_images)
    await processor.run_pipeline()
    
    print(f"\n=== Обработка завершена ===")
    print(f"Всего загружено изображений: {len(processor.artworks)}")
    
    for artwork in processor.artworks:
        print(f"  - Изображение {artwork.index}: ID={artwork.metadata.object_id}, "f"размер={artwork.image.shape}")
        
        # Выводится не все из за того что, часть изображений обрабатывается позже, чем вызывается блок этой функции
        # # Проверяем наличие обработанных файлов
        # processed_path = os.path.join('paintings', f"{artwork.index}_{artwork.metadata.object_id}_processed.jpg")
        # if os.path.exists(processed_path):
        #     file_size = os.path.getsize(processed_path) / 1024  # KB
        #     print(f"    Обработанное изображение: {os.path.basename(processed_path)} ({file_size:.1f} KB)")


def main():
    """Основная функция с обработкой аргументов командной строки"""
    import os
    
    # Парсинг аргументов командной строки
    num_images = 1  # значение по умолчанию
    
    if len(sys.argv) > 1:
        try:
            num_images = int(sys.argv[1])
            if num_images < 1:
                print("Количество изображений должно быть больше 0. Используется значение по умолчанию (1)")
                num_images = 1
            elif num_images > 100:
                print(f"Предупреждение: загрузка {num_images} изображений может занять много времени")
        except ValueError:
            print("Неверный параметр. Используется значение по умолчанию (1)")
    
    print(f"Количество изображений для загрузки: {num_images}")
    
    # Запуск асинхронной программы
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main_async(num_images))


if __name__ == "__main__":
    main()