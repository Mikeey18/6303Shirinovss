"""Агрегация и визуализация данных из CSV для анализа продолжительности создания объектов."""

import csv
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Generator, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from metetl.logging_config import get_logger
from metetl.decorators import timing

logger = get_logger(__name__)


class ArtworkAnalysisPipeline:
    """Анализ продолжительности создания объектов MET Museum."""
    
    def __init__(self, filepath: str, chunksize: int = 5000):
        self.filepath = filepath
        self.chunksize = chunksize
        self.output_dir = None
    
    def read_csv_chunks(self) -> Generator[pd.DataFrame, None, None]:
        """Чтение CSV файла по частям для экономии памяти."""
        usecols = ['Medium', 'Object Begin Date', 'Object End Date', 'Title', 'Object Number']
        
        try:
            for chunk in pd.read_csv(
                self.filepath,
                chunksize=self.chunksize,
                usecols=usecols,
                dtype={'Object Begin Date': 'Int64', 'Object End Date': 'Int64'}
            ):
                # Очистка данных
                chunk = chunk.dropna(subset=['Object Begin Date', 'Object End Date', 'Medium'])
                chunk = chunk[
                    (chunk['Object Begin Date'].between(0, 2026)) &
                    (chunk['Object End Date'].between(0, 2026)) &
                    (chunk['Object End Date'] >= chunk['Object Begin Date'])
                ]
                if not chunk.empty:
                    yield chunk
        except Exception as e:
            logger.error(f"Ошибка при чтении файла: {e}")
            raise
    
    def calculate_duration(self, chunks: Generator[pd.DataFrame, None, None]) -> Generator[pd.DataFrame, None, None]:
        """Вычисление продолжительности создания объектов."""
        for chunk in chunks:
            chunk['Duration'] = chunk['Object End Date'] - chunk['Object Begin Date']
            yield chunk
    
    def aggregate_material_stats(self, chunks: Generator[pd.DataFrame, None, None]) -> pd.DataFrame:
        """Агрегация статистики по материалам."""
        master_stats = pd.DataFrame(columns=['Medium', 'sum', 'sum_sq', 'count'])
        master_stats = master_stats.set_index('Medium')
        
        for chunk in chunks:
            chunk_agg = chunk.groupby('Medium')['Duration'].agg(
                sum='sum',
                sum_sq=lambda x: np.sum(x.astype(float)**2),
                count='count'
            )
            master_stats = master_stats.add(chunk_agg, fill_value=0)
        
        master_stats = master_stats.reset_index()
        
        # Преобразуем все значения в float для избежания проблем с типами
        master_stats['sum'] = master_stats['sum'].astype(float)
        master_stats['sum_sq'] = master_stats['sum_sq'].astype(float)
        master_stats['count'] = master_stats['count'].astype(float)
        
        master_stats['mean'] = master_stats['sum'] / master_stats['count']
        
        # Дисперсия: E[X^2] - (E[X])^2
        sigma = (master_stats['sum_sq'] / master_stats['count']) - (master_stats['mean']**2)
        # Защита от отрицательных значений из-за погрешностей
        sigma = sigma.clip(lower=0)
        master_stats['std'] = np.sqrt(sigma.values)
        
        # Доверительный интервал (95%)
        z_score = 1.96
        se = master_stats['std'] / np.sqrt(master_stats['count'].values)
        master_stats['ci_lower'] = master_stats['mean'] - z_score * se
        master_stats['ci_upper'] = master_stats['mean'] + z_score * se
        
        # Интервал рассеяния (95% данных)
        master_stats['scatter_lower'] = master_stats['mean'] - 2 * master_stats['std']
        master_stats['scatter_upper'] = master_stats['mean'] + 2 * master_stats['std']
        
        # Удаляем строки с NaN
        master_stats = master_stats.dropna()
        
        return master_stats
    
    def get_top_materials(self, material_stats: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """Получение топ-N материалов по количеству объектов."""
        return material_stats.sort_values(by='count', ascending=False).head(top_n)
    
    def analyze_time_trend(self, chunks: Generator[pd.DataFrame, None, None], target_material: str, window: int = 50) -> pd.DataFrame:
        """Анализ временного тренда для выбранного материала."""
        trend_df = pd.DataFrame()
        
        for chunk in chunks:
            filtered = chunk[chunk['Medium'] == target_material]
            if not filtered.empty:
                filtered = filtered.copy()
                filtered['MidYear'] = filtered['Object Begin Date'] + (filtered['Object End Date'] - filtered['Object Begin Date']) / 2
                trend_df = pd.concat([trend_df, filtered[['MidYear', 'Duration', 'Medium']]], ignore_index=True)
        
        if trend_df.empty:
            return pd.DataFrame()
        
        trend_df = trend_df.sort_values('MidYear')
        trend_df['RollingMean'] = trend_df['Duration'].rolling(window=window, center=True, min_periods=1).mean()
        
        return trend_df
    
    def run_pipeline(self, output_dir: str, top_n: int = 10) -> Dict:
        """Запуск полного пайплайна анализа."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Начало анализа продолжительности создания объектов MET Museum")
        
        # Агрегация статистики
        logger.debug("Агрегация статистики по материалам...")
        material_stats_df = self.aggregate_material_stats(
            self.calculate_duration(self.read_csv_chunks())
        )
        
        if material_stats_df.empty:
            logger.error("Нет данных для анализа")
            return {
                'top_materials': [],
                'total_objects': 0,
                'unique_materials': 0,
                'target_material': None,
                'trend_data': None
            }
        
        logger.info(f"Собрана статистика по {len(material_stats_df)} материалам")
        
        # Топ материалов
        top_materials_df = self.get_top_materials(material_stats_df, top_n)
        logger.debug(f"Топ состоит из {len(top_materials_df)} материалов")
        
        # Целевой материал для анализа тренда
        target_material = top_materials_df.loc[
            top_materials_df['mean'].idxmax(), 'Medium'
        ]
        logger.info(f"Целевой материал для анализа тренда: '{target_material}'")
        
        # Анализ тренда
        logger.debug("Выполняется анализ временного тренда...")
        time_trend_data = self.analyze_time_trend(
            self.calculate_duration(self.read_csv_chunks()),
            target_material
        )
        
        if not time_trend_data.empty:
            logger.info(f"Найдено {len(time_trend_data)} объектов для анализа тренда")
        else:
            logger.warning("Данные для анализа тренда не найдены")
        
        # Создание визуализаций
        self._create_visualizations(top_materials_df, time_trend_data, top_n)
        
        # Сохранение статистики в CSV
        self._save_statistics(top_materials_df, time_trend_data)
        
        # Подготовка результата
        result = {
            'top_materials': top_materials_df.to_dict('records'),
            'total_objects': int(material_stats_df['count'].sum()),
            'unique_materials': len(material_stats_df),
            'target_material': target_material,
            'trend_data': time_trend_data.to_dict('records') if not time_trend_data.empty else None
        }
        
        logger.info("Анализ успешно завершен")
        return result
    
    def _create_visualizations(self, top_materials_df: pd.DataFrame, time_trend_data: pd.DataFrame, top_n: int):
        """Создание визуализаций результатов анализа."""
        
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('Анализ продолжительности создания объектов\nMetropolitan Museum of Art', fontsize=16, fontweight='bold')
        
        # 1. Средняя продолжительность с доверительными интервалами
        ax1 = plt.subplot(2, 2, 1)
        means = top_materials_df['mean'].values.astype(float)
        materials = top_materials_df['Medium'].values
        ci_err_lower = means - top_materials_df['ci_lower'].values.astype(float)
        ci_err_upper = top_materials_df['ci_upper'].values.astype(float) - means
        
        y_pos = np.arange(len(materials))
        ax1.barh(y_pos, means, color='steelblue', alpha=0.7, label='Среднее')
        ax1.errorbar(means, y_pos, xerr=[ci_err_lower, ci_err_upper], fmt='none', capsize=5, color='red', linewidth=2)
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([m[:30] + '...' if len(m) > 30 else m for m in materials])
        ax1.set_xlabel('Продолжительность (лет)')
        ax1.set_title(f'Топ-{top_n} материалов: среднее время создания\n(с 95% доверительным интервалом)')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 2. Интервалы рассеяния
        ax2 = plt.subplot(2, 2, 2)
        for i, row in enumerate(top_materials_df.itertuples()):
            ax2.plot([row.scatter_lower, row.scatter_upper], [i, i], 'b-', linewidth=3, alpha=0.7)
            ax2.plot(row.mean, i, 'ro', markersize=8, label='Среднее' if i == 0 else '')
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([m[:30] + '...' if len(m) > 30 else m for m in materials])
        ax2.set_xlabel('Продолжительность (лет)')
        ax2.set_title('95% интервал рассеяния (разброс данных)')
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.legend(loc='lower right')
        
        # 3. Временной тренд
        ax3 = plt.subplot(2, 2, 3)
        if not time_trend_data.empty:
            ax3.scatter(time_trend_data['MidYear'], time_trend_data['Duration'], alpha=0.3, s=20, c='lightblue', label='Объекты')
            ax3.plot(time_trend_data['MidYear'], time_trend_data['RollingMean'], 'r-', linewidth=2, label='Скользящее среднее')
            
            avg = time_trend_data['Duration'].mean()
            ax3.axhline(y=avg, color='green', linestyle='--', label=f'Среднее: {avg:.1f} лет')
            
            ax3.set_title(f'Изменение времени создания во времени\nМатериал: {time_trend_data["Medium"].iloc[0]}')
            ax3.set_xlabel('Год (середина периода)')
            ax3.set_ylabel('Продолжительность (лет)')
            ax3.legend(loc='best')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Нет данных для отображения', ha='center', va='center')
            ax3.set_title('Временной тренд')
        
        # 4. Распределение
        ax4 = plt.subplot(2, 2, 4)
        if not time_trend_data.empty:
            ax4.hist(time_trend_data['Duration'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            ax4.axvline(time_trend_data['Duration'].mean(), color='red', linestyle='--', label='Среднее')
            ax4.axvline(time_trend_data['Duration'].median(), color='green', linestyle='--', label='Медиана')
            
            ax4.set_title('Распределение продолжительности создания')
            ax4.set_xlabel('Продолжительность (лет)')
            ax4.set_ylabel('Частота')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Нет данных для отображения', ha='center', va='center')
            ax4.set_title('Распределение продолжительности')
        
        plt.tight_layout()
        
        # Сохранение графика
        plot_path = self.output_dir / 'analysis_plots.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.debug(f"Графики сохранены в {plot_path}")
        plt.close()
    
    def _save_statistics(self, top_materials_df: pd.DataFrame, time_trend_data: pd.DataFrame):
        """Сохранение статистики в CSV файл."""
        
        if top_materials_df.empty:
            return
            
        # Сохраняем топ материалов
        stats_path = self.output_dir / 'material_statistics.csv'
        top_materials_df[['Medium', 'count', 'mean', 'std', 'ci_lower', 'ci_upper', 'scatter_lower', 'scatter_upper']].to_csv(stats_path, index=False, encoding='utf-8')
        logger.debug(f"Статистика материалов сохранена в {stats_path}")
        
        # Сохраняем данные тренда
        if not time_trend_data.empty:
            trend_path = self.output_dir / 'time_trend.csv'
            time_trend_data[['MidYear', 'Duration', 'RollingMean']].to_csv(trend_path, index=False, encoding='utf-8')
            logger.debug(f"Данные тренда сохранены в {trend_path}")
    
    def log_detailed_stats(self, result: Dict):
        """Логирование детальной статистики."""
        
        logger.info("=" * 60)
        logger.info("ДЕТАЛЬНЫЙ СТАТИСТИЧЕСКИЙ АНАЛИЗ")
        logger.info("=" * 60)
        
        # Общая информация
        logger.info(f"Общее количество объектов: {result['total_objects']:,}")
        logger.info(f"Уникальных материалов: {result['unique_materials']:,}")
        
        # Топ материалов
        if result['top_materials']:
            logger.info("=" * 60)
            logger.info("ТОП МАТЕРИАЛОВ ПО КОЛИЧЕСТВУ ОБЪЕКТОВ")
            logger.info("=" * 60)
            
            for i, material in enumerate(result['top_materials'], 1):
                logger.info(f"{i}. {material['Medium']}")
                logger.info(f"   Количество объектов: {int(material['count']):,}")
                logger.info(f"   Средняя продолжительность: {material['mean']:.2f} ± {material['std']:.2f} лет")
                logger.info(f"   95% доверительный интервал: [{material['ci_lower']:.2f}, {material['ci_upper']:.2f}]")
        
        # Анализ тренда
        if result['trend_data']:
            logger.info("=" * 60)
            logger.info(f"АНАЛИЗ ВРЕМЕННОГО ТРЕНДА ДЛЯ МАТЕРИАЛА: {result['target_material']}")
            logger.info("=" * 60)
            
            trend_df = pd.DataFrame(result['trend_data'])
            logger.info(f"Всего проанализировано объектов: {len(trend_df):,}")
            logger.info(f"Диапазон годов: {trend_df['MidYear'].min():.0f} - {trend_df['MidYear'].max():.0f}")
            logger.info(f"Средняя продолжительность: {trend_df['Duration'].mean():.2f} лет")
            logger.info(f"Медианная продолжительность: {trend_df['Duration'].median():.2f} лет")
            logger.info(f"Стандартное отклонение: {trend_df['Duration'].std():.2f} лет")
            
            # Простой анализ тренда
            if len(trend_df) > 10:
                x = trend_df['MidYear'].values
                y = trend_df['Duration'].values
                slope, intercept = np.polyfit(x, y, 1)
                trend_direction = "увеличение" if slope > 0 else "уменьшение"
                logger.info(f"Тренд: {slope:.4f} лет/год ({trend_direction} продолжительности со временем)")


@timing()
def analyze_dataset(csv_path: str, output_dir: str, top_n: int = 10) -> Dict:
    """Анализ датасета MET Objects.
    
    Args:
        csv_path: Путь к CSV файлу
        output_dir: Директория для сохранения графиков
        top_n: Количество топ-материалов для отображения
        
    Returns:
        Dict: Результаты анализа
    """
    logger.debug(f"Анализ датасета: {csv_path}")
    
    csv_file = Path(csv_path)
    if not csv_file.exists():
        logger.error(f"CSV файл не найден: {csv_path}")
        raise FileNotFoundError(f"CSV файл не найден: {csv_path}")
    
    # Создаем пайплайн и запускаем анализ
    pipeline = ArtworkAnalysisPipeline(str(csv_file), chunksize=5000)
    result = pipeline.run_pipeline(output_dir, top_n)
    
    # Логируем результаты
    pipeline.log_detailed_stats(result)
    
    logger.debug(f"Анализ завершен. Результаты сохранены в {output_dir}")
    
    return result