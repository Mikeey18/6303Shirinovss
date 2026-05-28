import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Generator, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class ArtworkAnalysisPipeline:
    def __init__(self, filepath: str, chunksize: int = 10000):
        self.filepath = filepath
        self.chunksize = chunksize
        
    def read_csv_chunks(self) -> Generator[pd.DataFrame, None, None]:
        usecols = ['Medium', 'Object Begin Date', 'Object End Date', 'Title', 'Object Number']
        try:
            for chunk in pd.read_csv(
                self.filepath,
                chunksize=self.chunksize,
                usecols=usecols,
                dtype={'Object Begin Date': 'Int64', 'Object End Date': 'Int64'}
            ):
                chunk = chunk.dropna(subset=['Object Begin Date', 'Object End Date', 'Medium'])
                chunk = chunk[
                    (chunk['Object Begin Date'].between(0, 2026)) &
                    (chunk['Object End Date'].between(0, 2026)) &
                    (chunk['Object End Date'] >= chunk['Object Begin Date'])
                ]
                if not chunk.empty:
                    yield chunk
        except Exception as e:
            print(f"Ошибка при чтении файла: {e}")
            raise
    
    def calculate_duration(self, chunks: Generator[pd.DataFrame, None, None]) -> Generator[pd.DataFrame, None, None]:
        for chunk in chunks:
            chunk['Duration'] = chunk['Object End Date'] - chunk['Object Begin Date']
            yield chunk

    def aggregate_material_stats(self, chunks: Generator[pd.DataFrame, None, None]) -> pd.DataFrame:
        # Создаем пустой DataFrame для накопления сумм и количеств
        # Мы храним суммы и суммы квадратов, чтобы вычислить среднее и std позже без хранения всех данных
        master_stats = pd.DataFrame(columns=['Medium', 'sum', 'sum_sq', 'count'])
        master_stats = master_stats.set_index('Medium') # Сделал medium первым индексом (индексом строк)

        for chunk in chunks:
            # Агрегируем текущий чанк
            chunk_agg = chunk.groupby('Medium')['Duration'].agg(
                sum='sum',
                sum_sq=lambda x: np.sum(x**2),
                count='count'
            )
            # Обновляем основной DataFrame: складываем значения для совпадающих материалов
            master_stats = master_stats.add(chunk_agg, fill_value=0)

        # Вычисляем финальные метрики на основе накопленных сумм
        master_stats = master_stats.reset_index()
        master_stats['mean'] = master_stats['sum'] / master_stats['count']
        
        # Дисперсия: E[X^2] - (E[X])^2
        sigma = (master_stats['sum_sq'] / master_stats['count']) - (master_stats['mean']**2)
        master_stats['std'] = (sigma)**0.5 # Среднеквадратичное отклонение
        
        # Расчет доверительного интервала (Z-статистика, так как мы не храним все данные для T-распределения)
        # 1.96 - коэффициент для 95% CI
        z_score = 1.96
        se = master_stats['std'] / (master_stats['count']**0.5)
        master_stats['ci_lower'] = master_stats['mean'] - z_score * se
        master_stats['ci_upper'] = master_stats['mean'] + z_score * se
        
        # Если их хранить нельзя, используем правило 2-х сигм (95% данных в нормальном распределении)
        master_stats['scatter_lower'] = master_stats['mean'] - 2 * master_stats['std']
        master_stats['scatter_upper'] = master_stats['mean'] + 2 * master_stats['std']

        return master_stats

    def get_top_materials(self, material_stats: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        # Сортируем и возвращаем верхние строки
        return material_stats.sort_values(by='count', ascending=False).head(top_n)
    
    def analyze_time_trend(self, chunks: Generator[pd.DataFrame, None, None], target_material: str, window: int = 50) -> pd.DataFrame:
        trend_df = pd.DataFrame()
        
        for chunk in chunks:
            filtered = chunk[chunk['Medium'] == target_material]
            if not filtered.empty:
                filtered['MidYear'] = filtered['Object Begin Date'] + (filtered['Object End Date'] - filtered['Object Begin Date']) / 2
                # Присоединяем новые строки к итоговому DataFrame материала
                trend_df = pd.concat([trend_df, filtered[['MidYear', 'Duration', 'Medium']]], ignore_index=True)
        
        if trend_df.empty:
            return pd.DataFrame()

        trend_df = trend_df.sort_values('MidYear')
        trend_df['RollingMean'] = trend_df['Duration'].rolling(window=window, center=True, min_periods=1).mean()
        
        return trend_df

    def run_pipeline(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print("АНАЛИЗ ПРОДОЛЖИТЕЛЬНОСТИ СОЗДАНИЯ ОБЪЕКТОВ MET MUSEUM")
        print("="*80)

        # Сбор общей статистики
        print("\n1. Агрегация статистики по материалам...")
        material_stats_df = self.aggregate_material_stats(
            self.calculate_duration(self.read_csv_chunks())
        ) 

        # Выбор ТОП-10
        top_materials_df = self.get_top_materials(material_stats_df)

        # Вывод ТОП-10
        print("\n2. ТОП-10 материалов по количеству объектов:")
        for i, row in enumerate(top_materials_df.itertuples(), 1):
            print(f"   {i}. {row.Medium[:40]} - {int(row.count):,} объектов")

        # Поиск целевого материала (с максимальным средним временем)
        target_material = top_materials_df.loc[
            top_materials_df['mean'].idxmax(), 'Medium'
        ]
        print(f"\n3. Целевой материал для анализа тренда: '{target_material}'")

        # Анализ тренда
        print("   Выполняется анализ временного тренда...")
        time_trend_data = self.analyze_time_trend(
            self.calculate_duration(self.read_csv_chunks()), 
            target_material
        )

        if not time_trend_data.empty:
            print(f"   Найдено {len(time_trend_data):,} объектов")
        else:
            print("   Данные для анализа не найдены")

        return top_materials_df, time_trend_data

    def visualize_results(self, top_materials_df: pd.DataFrame, time_trend_data: pd.DataFrame):
        """Визуализация результатов анализа"""
        # Настройка полотна для графиков
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('Анализ продолжительности создания объектов\nMetropolitan Museum of Art', fontsize=16, fontweight='bold')

        # 1. Средняя продолжительность с доверительными интервалами
        ax1 = plt.subplot(2, 2, 1)

        # В Pandas выборка данных происходит по названиям колонок
        means = top_materials_df['mean'].values
        materials = top_materials_df['Medium'].values
        # Рассчитываем отклонения для errorbar (разница между средним и границами ДИ)
        ci_err_lower = means - top_materials_df['ci_lower'].values
        ci_err_upper = top_materials_df['ci_upper'].values - means

        y_pos = np.arange(len(materials))
        ax1.barh(y_pos, means, color='steelblue', alpha=0.7, label='Среднее')

        # Отрисовка "усиков" (error bars)
        ax1.errorbar(means, y_pos, xerr=[ci_err_lower, ci_err_upper], 
                     fmt='none', capsize=5, color='red', linewidth=2)

        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([m[:30] + '...' if len(m) > 30 else m for m in materials])
        ax1.set_xlabel('Продолжительность (лет)')
        ax1.set_title('Топ-10 материалов: среднее время создания\n(с 95% доверительным интервалом)')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3, axis='x')

        # 2. Интервалы рассеяния (индивидуальные наблюдения)
        ax2 = plt.subplot(2, 2, 2)

        for i, row in enumerate(top_materials_df.itertuples()):
            # Рисуем линию от минимального до максимального перцентиля
            ax2.plot([row.scatter_lower, row.scatter_upper], [i, i], 'b-', linewidth=3, alpha=0.7)
            # Ставим точку на значении среднего
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
            # Диаграмма рассеяния всех точек
            ax3.scatter(time_trend_data['MidYear'], time_trend_data['Duration'], 
                        alpha=0.3, s=20, c='lightblue', label='Объекты')

            # Линия скользящего среднего
            ax3.plot(time_trend_data['MidYear'], time_trend_data['RollingMean'], 
                     'r-', linewidth=2, label='Скользящее среднее')

            # Горизонтальная линия общего среднего
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

        # 4. Распределение (Гистограмма)
        ax4 = plt.subplot(2, 2, 4)
        if not time_trend_data.empty:
            # Гистограмма по колонке Duration
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
        plt.show()

        # Вывод текстовой статистики
        print("\nДЕТАЛЬНАЯ СТАТИСТИКА (TOP-10):")
        print(top_materials_df[['Medium', 'count', 'mean', 'std', 'ci_lower', 'ci_upper']].to_string(index=False))

    def print_detailed_stats(self, top_materials_df: pd.DataFrame, time_trend_data: pd.DataFrame):
        """Вывод детальной статистики в консоль"""

        print("\n" + "="*80)
        print("ДЕТАЛЬНЫЙ СТАТИСТИЧЕСКИЙ АНАЛИЗ")
        print("="*80)

        # Вывод информации по топ-10 материалам
        for i, row in enumerate(top_materials_df.itertuples(), 1):
            print(f"\n{i}. {row.Medium}")
            print(f"   Количество объектов: {int(row.count):,}")
            print(f"   Средняя продолжительность: {row.mean:.2f} ± {row.std:.2f} лет")
            print(f"   95% доверительный интервал (для среднего): [{row.ci_lower:.2f}, {row.ci_upper:.2f}]")
            print(f"   95% интервал рассеяния (для отдельных): [{row.scatter_lower:.2f}, {row.scatter_upper:.2f}]")

        # Анализ временного тренда
        if not time_trend_data.empty:
            print("\n" + "="*80)
            print(f"АНАЛИЗ ВРЕМЕННОГО ТРЕНДА ДЛЯ МАТЕРИАЛА: {time_trend_data['Medium'].iloc[0]}")
            print("="*80)

            print(f"\nВсего проанализировано объектов: {len(time_trend_data):,}")
            print(f"Диапазон годов: {time_trend_data['MidYear'].min():.0f} - {time_trend_data['MidYear'].max():.0f}")
            print(f"Средняя продолжительность: {time_trend_data['Duration'].mean():.2f} лет")
            print(f"Медианная продолжительность: {time_trend_data['Duration'].median():.2f} лет")
            print(f"Стандартное отклонение: {time_trend_data['Duration'].std():.2f} лет")
            print(f"Минимальная продолжительность: {time_trend_data['Duration'].min():.0f} лет")
            print(f"Максимальная продолжительность: {time_trend_data['Duration'].max():.0f} лет")

            # Квартили
            q1 = time_trend_data['Duration'].quantile(0.25)
            q3 = time_trend_data['Duration'].quantile(0.75)
            print(f"Межквартильный размах (IQR): {q1:.2f} - {q3:.2f} лет")

            # Проверка на наличие тренда
            if len(time_trend_data) > 10:
                # Простая линейная регрессия для определения тренда
                x = time_trend_data['MidYear'].values
                y = time_trend_data['Duration'].values
                slope, intercept = np.polyfit(x, y, 1)

                # Расчет R-squared
                y_pred = slope * x + intercept
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)

                trend_direction = "увеличение" if slope > 0 else "уменьшение"
                print(f"\nТренд: {slope:.4f} лет/год ({trend_direction} продолжительности со временем)")
                print(f"Коэффициент детерминации R²: {r_squared:.4f}")

                # Проверка значимости тренда
                if abs(slope) > 0.01 and r_squared > 0.1:
                    print(f"Обнаружен статистически значимый тренд в данных")
                else:
                    print(f"Статистически значимый тренд не обнаружен")

        else:
            print("\n" + "="*80)
            print("НЕТ ДАННЫХ ДЛЯ АНАЛИЗА ВРЕМЕННОГО ТРЕНДА")
            print("="*80)

def main():
    filepath = "./data/MetObjects.csv"
    
    try:
        # Создание и запуск пайплайна
        pipeline = ArtworkAnalysisPipeline(filepath, chunksize=5000)
        
        # Выполнение обработки
        top_materials_df, time_trend_data = pipeline.run_pipeline()
        
        # Вывод детальной статистики
        pipeline.print_detailed_stats(top_materials_df, time_trend_data)
        
        # Визуализация результатов
        pipeline.visualize_results(top_materials_df, time_trend_data)
        
    except Exception as e:
        print(f"\nПроизошла ошибка: {e}")


if __name__ == "__main__":
    main()