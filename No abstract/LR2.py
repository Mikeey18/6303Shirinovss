from image_processor import ImageProcessor

def main(): 
    processor = ImageProcessor()

    artwork = processor.download_random_painting()
    
    if artwork is None:
        print("Ошибка скачки изображения")
        return
    
    print(f"\n{artwork}")
    
    operations = [
        {'name': 'grayscale', 'params': {}},
        {'name': 'gaussian_blur', 'params': {'kernel_size': 5}},
        {'name': 'sobel', 'params': {}},
        {'name': 'gamma_correction', 'params': {'gamma': 0.5}},
        {'name': 'histogram_equalization', 'params': {}}
    ]
    
    for op in operations:
        print(f"Проверка операции {op['name']}...")
        
        processor.compare_with_library(artwork, op['name'], **op['params'])
    
    if len(processor.artworks) >= 2:
        print("Проверка сложения двух изображений...")
        
        combined = processor.artworks[0] + processor.artworks[1]
        print(f"Результат: {combined}")
        
        processor.save_result(combined.image, processor.artworks[0], "combined")

if __name__ == "__main__":
    main()