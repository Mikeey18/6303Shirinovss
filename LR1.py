import numpy as np, random, os, task1 as t1, task2 as t2

def main():
    paintings_dir = t1.setup_directories()
    
    csv_path = 'data/MetObjects.csv'
    
    all_objects = t1.read_csv_file(csv_path)
    print(f"Загружено {len(all_objects)} объектов")
    
    paintings = t1.filter_paintings(all_objects)
    print(f"Найдено {len(paintings)} картин")
    
    random_painting = random.choice(paintings)
    object_id = random_painting.get('Object ID')
    if not object_id:
        print("Не удалось получить object_id")
        return
    
    print(f"\nВыбрана картина:")
    print(f"Object ID: {object_id}\n")
    
    details = t1.get_object_details(object_id)
    if not details:
        print("Не удалось получить детали объекта")
        
        return
    
    image_url = details.get('primaryImage')
    if not image_url:
        print("У объекта нет изображения")
        return
    
    print(f"Ссылка на изображение: {image_url}")
    
    image_path = os.path.join(paintings_dir, f"{object_id}.jpg")
    json_path = os.path.join(paintings_dir, f"{object_id}.json")
    
    if t1.download_image(image_url, image_path):
        print(f"Изображение сохранено: {image_path}")
    else:
        print("Не удалось скачать изображение")
    
    if t1.save_json(details, json_path):
        print(f"Метаданные сохранены: {json_path}")
    else:
        print("Не удалось сохранить json")

    operations = [
        {
            'my_func': t2.my_grayscale,
            'lib_func': t2.library_grayscale,
            'suffix': 'gray',
            'kernel': 0
        },
        {
            'my_func': t2.my_convolution,
            'lib_func': t2.library_convolution,
            'suffix': 'convolution',
            'kernel': np.array([[1/9, 1/9, 1/9],
                                [1/9, 1/9, 1/9],
                                [1/9, 1/9, 1/9]])
        },
        {
            'my_func': t2.my_gaussian_blur,
            'lib_func': t2.library_gaussian_blur,
            'suffix': 'gauss',
            'kernel': 0
        },
        {
            'my_func': t2.my_sobel,
            'lib_func': t2.library_sobel,
            'suffix': 'sobel',
            'kernel': 0
        },
        {
            'my_func': t2.my_gamma_correction,
            'lib_func': t2.library_gamma_correction,
            'suffix': 'gamma',
            'kernel': 0
        },
        {
            'my_func': t2.my_color_histogram_equalization,
            'lib_func': t2.library_color_histogram_equalization,
            'suffix': 'histogram',
            'kernel': 0
        }
    ]

    for op in operations:
        t2.test_operation(image_path, op['my_func'], op['lib_func'], op['suffix'], op['kernel'])


if __name__ == "__main__":
    main()