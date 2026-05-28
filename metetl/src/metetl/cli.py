"""Интерфейс командной строки для MET ETL Pipeline."""

import asyncio, sys, click
from pathlib import Path
from typing import Optional

from metetl.logging_config import setup_logging, get_logger
from metetl.analysis.data_to_download import prepare_download_list
from metetl.analysis.aggregations import analyze_dataset
from metetl.images.processing import ImageProcessor

logger = get_logger(__name__)

def should_suppress_logging():
    """Проверяет, нужно ли подавить логирование."""
    return any(arg in sys.argv for arg in ['--help', '-h', 'help'])

@click.group()
@click.version_option(version="1.0.0", prog_name="metetl")
def cli():
    """MET ETL Pipeline - загрузка и обработка изображений MET Museum."""
    if not should_suppress_logging():
        setup_logging(log_dir="logs", log_file="app.log")
        logger.info("MET ETL Pipeline запущен")


@cli.command()
@click.option('--csv', 'csv_path', default='data/MetObjects.csv', type=click.Path(exists=True), help='Путь к CSV файлу MetObjects.csv (по умолчанию: data/MetObjects.csv)')
@click.option('--output', 'output_path', default='data/to_download.json', type=click.Path(), help='Путь для сохранения JSON файла с метаданными (по умолчанию: data/to_download.json)')
@click.option('--limit', '-n', default=None, type=int, help='Максимальное количество изображений (выбираются случайные)')
@click.option('--seed', default=None, type=int, help='Seed для случайной выборки (для воспроизводимости)')
def prepare(csv_path: str, output_path: str, limit: Optional[int], seed: Optional[int]):
    """Подготовка JSON файла с метаданными выбранных изображений.
    
    Пример:
        metetl prepare --csv data/MetObjects.csv --output data/to_download.json --limit 10
    """
    click.echo(f"Подготовка метаданных из {csv_path}")
    
    try:
        paintings = prepare_download_list(csv_path, output_path, limit, seed)
        
        click.echo(click.style(f"✓ Успешно подготовлено {len(paintings)} записей", fg='green'))
        click.echo(f"  Файл сохранен: {output_path}")
        
        logger.info(f"Команда prepare завершена успешно")
        click.echo(click.style(f"\n✓ Обработка завершена", fg='green'))

    except Exception as e:
        click.echo(click.style(f"✗ Ошибка: {e}", fg='red'), err=True)


@cli.command()
@click.option('--input', 'input_path', default='data/to_download.json', type=click.Path(exists=True), help='Путь к JSON файлу с метаданными (созданный командой prepare, по умолчанию: data/to_download.json)')
@click.option('--output', 'output_dir', default='images', type=click.Path(), help='Директория для сохранения изображений (по умолчанию: images)')
@click.option('--num', '-n', 'num_images', default=5, type=int, help='Количество изображений для скачивания и обработки (по умолчанию: 5)')
@click.option('--operation', '-op', default='histogram_equalization', type=click.Choice(['gaussian_blur', 'sobel', 'gamma_correction', 'histogram_equalization']), help='Операция обработки изображений (по умолчанию: histogram_equalization)')
@click.option('--kernel-size', '-k', default=5, type=int, help='Размер ядра для Gaussian blur (по умолчанию: 5, только для gaussian_blur)')
@click.option('--gamma', '-g', default=1.0, type=float, help='Значение гаммы для гамма-коррекции (по умолчанию: 1.0, только для gamma_correction)')
@click.option('--workers', '-w', default=None, type=int, help='Количество процессов для параллельной обработки (по умолчанию: число CPU)')
def process(input_path: str, output_dir: str, num_images: int, operation: str, kernel_size: int, gamma: float, workers: Optional[int]):
    """Запуск пайплайна по скачиванию и обработке изображений.
    
    Примеры:
        metetl process --input data/to_download.json --output images --num 5
        
        metetl process --input data/to_download.json --num 10 --operation gaussian_blur --kernel-size 7
        
        metetl process --input data/to_download.json --num 20 --operation gamma_correction --gamma 1.5
        
        metetl process --input data/to_download.json --num 15 --operation sobel --workers 8
    """
    
    import json
    
    click.echo(f"  Запуск обработки {num_images} изображений")
    click.echo(f"  Входной файл: {input_path}")
    click.echo(f"  Выходная директория: {output_dir}")
    click.echo(f"  Операция: {operation}")
    
    # Проверяем соответствие параметров операции
    if operation == 'gaussian_blur':
        click.echo(f"  Размер ядра: {kernel_size}")
        if kernel_size < 1 or kernel_size % 2 == 0:
            click.echo(click.style(f"  ⚠ Предупреждение: kernel_size должен быть нечетным положительным числом. Используется {kernel_size}", fg='yellow'))
    elif operation == 'gamma_correction':
        click.echo(f"  Гамма: {gamma}")
    elif operation == 'sobel':
        click.echo(f"  (Операция Собеля не требует дополнительных параметров)")
    elif operation == 'histogram_equalization':
        click.echo(f"  (Эквализация гистограммы не требует дополнительных параметров)")
    
    if workers:
        click.echo(f"  Процессов: {workers}")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            paintings_list = json.load(f)
        
        if not paintings_list:
            click.echo(click.style("✗ Файл метаданных пуст", fg='red'), err=True)
            sys.exit(1)
        
        click.echo(f"  Доступно изображений в JSON: {len(paintings_list)}")
        
        from metetl.images.processing import ImageProcessor
        
        processor = ImageProcessor(
            num_images=num_images, 
            output_dir=output_dir,
            num_workers=workers
        )
        
        # Формируем параметры обработки в зависимости от операции
        processing_params = {
            'operation': operation,
            'kernel_size': kernel_size if operation == 'gaussian_blur' else 5,  # По умолчанию для других операций
            'gamma': gamma if operation == 'gamma_correction' else 1.0  # По умолчанию для других операций
        }
        
        artworks = asyncio.run(processor.run_pipeline(paintings_list, processing_params))
        
        logger.info(f"Команда process завершена успешно")
        click.echo(click.style(f"\n✓ Обработка завершена", fg='green'))
        click.echo(f"  Загружено и обработано: {len(artworks)} изображений")
        
        # Показываем информацию о примененной обработке
        if operation == 'gaussian_blur':
            click.echo(f"  Применено: Gaussian blur (kernel_size={kernel_size})")
        elif operation == 'gamma_correction':
            click.echo(f"  Применено: Гамма-коррекция (gamma={gamma})")
        elif operation == 'sobel':
            click.echo(f"  Применено: Детектор границ Собеля")
        elif operation == 'histogram_equalization':
            click.echo(f"  Применено: Эквализация гистограммы")
        
        # Показываем первые 5 результатов
        for i, artwork in enumerate(artworks[:5], 1):
            title = artwork.metadata.title or 'N/A'
            if len(title) > 50:
                title = title[:47] + "..."
            click.echo(f"    {i}. ID: {artwork.metadata.object_id}, Title: {title}")
        
        if len(artworks) > 5:
            click.echo(f"    ... и еще {len(artworks) - 5} изображений")
        
    except Exception as e:
        click.echo(click.style(f"✗ Ошибка: {e}", fg='red'), err=True)
        sys.exit(1)


@cli.command()
@click.option('--csv', 'csv_path', default='data/MetObjects.csv', type=click.Path(exists=True), help='Путь к CSV файлу MetObjects.csv (по умолчанию: data/MetObjects.csv)')
@click.option('--output-dir', 'output_dir', default='data/plots', type=click.Path(), help='Директория для сохранения графиков (по умолчанию: data/plots)')
@click.option('--top-n', default=10, type=click.IntRange(min=1, max=50, clamp=False), help='Количество элементов в топ-списках (по умолчанию: 10)')
def analyze(csv_path: str, output_dir: str, top_n: int):
    """Запуск анализа датасета из CSV файла.
    
    Анализирует продолжительность создания объектов MET Museum:
    - Топ материалов по количеству объектов
    - Среднее время создания с доверительными интервалами
    - Временные тренды для материала с максимальным средним временем
    - Распределение продолжительности создания
    
    Пример:
        metetl analyze --csv data/MetObjects.csv --output-dir data/plots --top-n 15
    """
    
    click.echo(f"  Анализ датасета: {csv_path}")
    click.echo(f"  Сохранение графиков в: {output_dir}")
    click.echo(f"  Топ-N материалов: {top_n}")
    
    try:
        from metetl.analysis.aggregations import analyze_dataset
        result = analyze_dataset(csv_path, output_dir, top_n)
        
        logger.info(f"Команда analyze завершена успешно")
        click.echo(click.style(f"\n✓ Обработка завершена", fg='green'))
        click.echo(f"  Всего записей в CSV: {result['total_objects']:,}")
        click.echo(f"  Уникальных материалов: {result['unique_materials']:,}")
        click.echo(f"\n  Топ-{top_n} материалов по количеству:")
        
        for i, material in enumerate(result['top_materials'], 1):
            click.echo(f"    {i}. {material['Medium'][:50]}: {int(material['count']):,} объектов, " f"среднее {material['mean']:.1f} лет")
        
        click.echo(f"\n  Результаты сохранены в директории: {output_dir}")
        click.echo(f"    - analysis_plots.png (основные графики)")
        click.echo(f"    - material_statistics.csv (статистика)")
        
        if result['trend_data']:
            click.echo(f"    - time_trend.csv (данные временного тренда)")
        
    except Exception as e:
        click.echo(click.style(f"✗ Ошибка: {e}", fg='red'), err=True)
        sys.exit(1)