import csv, requests, json, os

def setup_directories():
    paintings_dir = os.path.join(os.getcwd(), 'paintings') # Текущая директория + имя папки
    os.makedirs(paintings_dir, exist_ok=True) # exist_ok = true, чтобы не выдывало ошибку, если папка уже есть
    return paintings_dir

def read_csv_file(csv_path):
    try:
        with open(csv_path, mode='r', encoding='utf-8-sig') as file: # utf-8-sig вместо utf-8. Чтобы ключ был 'Object Number' без byte order mark \ufeff
            reader = csv.DictReader(file)
            a = list(reader)
            return a
    except Exception as e:
        print(f"Ошибка при чтении CSV: {e}")
        return []

def filter_paintings(objects):
    return [obj for obj in objects if obj.get('Classification') == 'Paintings']

def get_object_details(object_id):
    url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"
    
    try:
        response = requests.get(url)
        return response.json()
    except Exception as e:
        print(f"Ошибка при получении деталей: {e}")
        return None
    
def download_image(image_url, save_path):
    try:
        response = requests.get(image_url, stream=True) # Stream = True, чтобы можно было записать частями
        with open(save_path, 'wb') as file: # Бинарный режим записи чанками, на случай если изображение большое, чтобы не забить оперативку
            for chunk in response.iter_content(chunk_size=16384):
                file.write(chunk)
        return True
    except Exception as e:
        print(f"Ошибка при скачивании: {e}")
        return False

def save_json(data, filepath):
    try:
        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2, ensure_ascii=False) # Отступы indent = 2, чтобы не в 1 строку; ensure_ascii = False, чтобы сохранять русские буквы и др. символы
        return True
    except Exception as e:
        print(f"Ошибка сохранения JSON: {e}")
        return False