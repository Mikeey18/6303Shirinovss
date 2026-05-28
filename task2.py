import os, numpy as np, numpy.lib.stride_tricks, cv2, time

def my_grayscale(image):
    height, width = image.shape[:2]
    result = np.zeros((height, width))
    eye_vector = np.array([0.299, 0.587, 0.114])
    # Умножаем каждый канал на соответствующий коэффициент и суммируем по оси каналов
    result = np.sum(image * eye_vector, axis=2)
    return result.astype(np.uint8)

def library_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def my_convolution(image, kernel):
    kernel_height, kernel_width = kernel.shape
    padded_height, padded_width = kernel_height // 2, kernel_width // 2

    pad_width = [(padded_height, padded_height), (padded_width, padded_width)]
    if image.ndim == 3:
        pad_width.append((0, 0))

    padded = np.pad(image, pad_width)

    windows = numpy.lib.stride_tricks.sliding_window_view(padded, kernel.shape, axis=(0, 1))
    return np.clip(np.tensordot(windows, kernel), 0, 255) # result[i,j] = sum_k ( a[i,k] * b[k,j] )

def library_convolution(image, kernel):
    return cv2.filter2D(image, -1, kernel) # -1, чтобы не менять глубину изображения

def get_gaussian_kernel(kernel_size, sigma):
    ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size) # равномерное распредление от -центра до центра
    gauss = np.exp(-((ax / sigma) ** 2) / 2) # ф-ия гаусса
    kernel = np.outer(gauss, gauss) # kernel[i,j] = gauss[i] * gauss[j]
    return (kernel / kernel.sum()).astype(np.float32) # Нормируем для сохранения яркости

def my_gaussian_blur(image, kernel_size = 3):
    kernel = get_gaussian_kernel(kernel_size, sigma=kernel_size / 6)
    return my_convolution(image, kernel).astype(np.uint8)

def library_gaussian_blur(image, kernel_size=3):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def my_sobel(image):
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[ 1,  2,  1],
                         [ 0,  0,  0],
                         [-1, -2, -1]], dtype=np.float32)
    
    grad_x = my_convolution(image, kernel_x)
    grad_y = my_convolution(image, kernel_y)
    
    grad_x = grad_x.astype(np.float32)
    grad_y = grad_y.astype(np.float32)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    
    grad = (grad / grad.max()) * 255
    return grad.astype(np.uint8)

def library_sobel(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    grad = (grad / grad.max()) * 255
    return grad.astype(np.uint8)

def my_gamma_correction(image, gamma=1.0): # gamma < 1 - осветление, > 1 - затемнение
    normalized = image.astype(np.float32) / 255.0
    corrected = np.power(normalized, gamma)
    result = (corrected * 255).astype(np.uint8)
    
    return result

def library_gamma_correction(image, gamma=0.5):
    look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8) # Таблица, где для каждого входного значения от 0 до 255 записано соответствующее выходное значение после гамма-коррекции.
    return cv2.LUT(image, look_up_table) # Для каждого пикселя просто берется готовое значение из таблицы по индексу

# Выравнивание гистограммы - это метод улучшения контрастности изображения путем перераспределения интенсивностей пикселей так, чтобы они занимали весь доступный диапазон [0, 255]

def my_color_histogram_equalization(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Разделяем каналы, чтобы при коррекции не изменялось соотношение между каналами
    # L - яркость, A - от зеленого до пурпурного, B - от синего до желтого
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Выравниваем L-канал
    # Вычисляем гистограмму для L-канала
    hist = np.zeros(256, dtype=np.int32) # 256 элементов (по одному на каждое возможное значение яркости 0-255)
    for i in range(l_channel.shape[0]):
        for j in range(l_channel.shape[1]):
            hist[l_channel[i, j]] += 1
    
    # Вычисляем кумулятивную функцию распределения. Новые значения яркости. Операция, которая для каждого элемента массива возвращает сумму всех предыдущих элементов, включая текущий.
    cdf = hist.cumsum()
    
    # Нормализуем CDF к диапазону [0, 255]
    cdf_normalized = ((cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())).astype(np.uint8)
    
    # Применяем преобразование к L-каналу, заменяя старые значения яркости на новые из cdf_normalized
    l_equalized = cdf_normalized[l_channel] 
    
    # Объединяем каналы обратно
    lab_equalized = cv2.merge([l_equalized, a_channel, b_channel])
    
    # Конвертируем обратно в RGB
    result = cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)
    
    return result

def library_color_histogram_equalization(image):
    # Конвертируем RGB в LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Разделяем каналы
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Выравниваем L-канал
    l_equalized = cv2.equalizeHist(l_channel)
    
    # Объединяем каналы
    lab_equalized = cv2.merge([l_equalized, a_channel, b_channel])
    
    # Конвертируем обратно в RGB
    result = cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)
    
    return result

def test_operation(image_path, my_func, lib_func, save_suffix, kernel):
    match save_suffix:
        case 'gray':
            print("\nПриведение цветного изображения к полутоновому.")
        case 'convolution':
            print("\nCвёртка c использованием двумерной маски.")
        case 'gauss':
            print("\nCглаживание (применение оператора Гаусса).")
        case 'sobel':
            print("\nВыделение границ (применение оператора Собеля).")
        case 'gamma':
            print("\nГамма коррекция.")
        case 'histogram':
            print("\nВыравнивание гистограммы изображения.")

    img = cv2.imread(image_path)
    if img is None:
        print(f"Ошибка загрузки: {image_path}")
        return
    
    if save_suffix == 'convolution':
        save_dir = os.path.dirname(image_path)
        filename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(filename)[0]

        start = time.perf_counter()
        my_result = my_func(img, kernel)
        my_time = time.perf_counter() - start
        
        start = time.perf_counter()
        lib_result = lib_func(img, kernel)
        lib_time = time.perf_counter() - start
    else:
        save_dir = os.path.dirname(image_path)
        filename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(filename)[0]

        start = time.perf_counter()
        my_result = my_func(img)
        my_time = time.perf_counter() - start
        
        start = time.perf_counter()
        lib_result = lib_func(img)
        lib_time = time.perf_counter() - start

    # Вывод времени
    print(f"\nМоя реализация: {my_time*1000:.4f} мс")
    print(f"Библиотечная: {lib_time*1000:.4f} мс")
    
    ratio = lib_time / my_time
    if ratio > 1:
        print(f"Библиотека медленнее в {ratio:.2f} раз")
    else:
        print(f"Библиотека быстрее в {1/ratio:.2f} раз")
    
    my_path = os.path.join(save_dir, f"{name_without_ext}_{save_suffix}_my.jpg")
    lib_path = os.path.join(save_dir, f"{name_without_ext}_{save_suffix}_lib.jpg")
    
    cv2.imwrite(my_path, my_result)
    cv2.imwrite(lib_path, lib_result)
    
    print(f"\nСохранено:")
    print(f"Моя: {my_path}")
    print(f"Библиотечная: {lib_path}")