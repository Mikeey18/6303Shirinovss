from image_processor import ImageProcessor
from artwork import ColorArtwork

def main():
    processor = ImageProcessor()

    # Скачиваем цветное изображение
    print("\n[1] Скачивание цветного изображения...")
    color_artwork = processor.download_random_painting(as_grayscale=False)
    
    if color_artwork is None:
        print("Ошибка скачки изображения")
        return
    
    print(f"\n{color_artwork}")
    
    # Скачиваем черно-белое изображение (для демонстрации)
    print("\n[2] Скачивание черно-белого изображения...")
    gray_artwork = processor.download_random_painting(as_grayscale=True)
    
    if gray_artwork is None:
        print("Ошибка скачки изображения")
        return
    
    print(f"\n{gray_artwork}")
    
    # Тестирование операций для разных типов
    operations = [
        {'name': 'grayscale', 'params': {}},
        {'name': 'gaussian_blur', 'params': {'kernel_size': 5}},
        {'name': 'sobel', 'params': {}},
        {'name': 'gamma_correction', 'params': {'gamma': 0.5}},
        {'name': 'histogram_equalization', 'params': {}}
    ]
    
    for op in operations:
        print(f"\nПроверка операции {op['name']} на цветном изображении...")
        comparison = processor.compare_with_library(color_artwork, op['name'], **op['params'])
        
        if comparison:
            print(f"\nРезультат сохранен для {op['name']}:")
            print(f"Ручной: {comparison['custom_path']}")
            print(f"Библиотечный: {comparison['lib_path']}")
    
    for op in operations:
        print(f"\nПроверка операции {op['name']} на черно-белом изображении...")
        processor.compare_with_library(gray_artwork, op['name'], **op['params'])

    try:
        # Сложение двух цветных изображений
        if len(processor.artworks) >= 2 and isinstance(processor.artworks[0], ColorArtwork) and isinstance(processor.artworks[1], ColorArtwork):
            print("\nСложение двух цветных изображений...")
            combined_color = processor.artworks[0] + processor.artworks[1]
            print(f"Результат: {combined_color}")
            processor.save_result(combined_color.image, processor.artworks[0], "combined_color")
        
        # Сложение двух черно-белых изображений
        if len(processor.artworks) >= 2:
            # Создаем второе черно-белое изображение
            if isinstance(processor.artworks[1], ColorArtwork):
                gray1 = processor.artworks[1].to_grayscale_artwork()
            else:
                gray1 = processor.artworks[1]
            
            if isinstance(processor.artworks[0], ColorArtwork):
                gray2 = processor.artworks[0].to_grayscale_artwork()
            else:
                gray2 = processor.artworks[0]
            
            print("\nСложение двух черно-белых изображений...")
            combined_gray = gray1 + gray2
            print(f"Результат: {combined_gray}")
            processor.save_result(combined_gray.image, processor.artworks[0], "combined_gray")
            
    except Exception as e:
        print(f"Ошибка при сложении: {e}")


if __name__ == "__main__":
    main()