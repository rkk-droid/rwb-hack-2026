import polars as pl

LAGS = [1, 2, 3, 4, 6, 8, 24, 48, 96, 144, 192, 240, 288, 336]
WINDOWS = [5, 10, 48, 336]

def generate_features(df, output_path: str = "data/features.parquet", filter_aug_sep: bool = True):

    # Переводим в LazyFrame, если это не так
    if not isinstance(df, pl.LazyFrame):
        df = df.lazy()

    # Переименование колонки
    df = df.rename({"target_1h": "sum_1h"})

    # Суммы статусов
    df = df.with_columns([
        (pl.col("status_1") + pl.col("status_2") + pl.col("status_3")).alias("status_1-3_sum"),
        (pl.col("status_4") + pl.col("status_5") + pl.col("status_6")).alias("status_4-6_sum"),
    ])

    # Пропорции для групп
    df = df.with_columns([
        (pl.col("status_1-3_sum") / (pl.col("status_1-3_sum") + pl.col("status_4-6_sum") + 1e-5)).alias("status_1-3_all_prop"),
        (pl.col("status_4-6_sum") / (pl.col("status_1-3_sum") + pl.col("status_4-6_sum") + 1e-5)).alias("status_4-6_all_prop"),
        (pl.col("status_1") / (pl.col("status_1-3_sum") + 1e-5)).alias("status_1_1-3_prop"),
        (pl.col("status_2") / (pl.col("status_1-3_sum") + 1e-5)).alias("status_2_1-3_prop"),
        (pl.col("status_3") / (pl.col("status_1-3_sum") + 1e-5)).alias("status_3_1-3_prop"),
        (pl.col("status_4") / (pl.col("status_4-6_sum") + 1e-5)).alias("status_4_4-6_prop"),
        (pl.col("status_5") / (pl.col("status_4-6_sum") + 1e-5)).alias("status_5_4-6_prop"),
        (pl.col("status_6") / (pl.col("status_4-6_sum") + 1e-5)).alias("status_6_4-6_prop"),
    ])

    # Средние по кратным лагам ±i, ±2i, ±3i, ... (недельная периодичность)
    # i=336 — ровно 1 неделя (336 × 30 мин = 168 ч)
    _MEAN_LAG_RANGE = range(327, 337)
    _SERIES_LEN = 4630

    for i in _MEAN_LAG_RANGE:
        tmp_cols = []
        k = 1
        while i * k <= _SERIES_LEN:
            for sign in [1, -1]:  # 1 = прошлое, -1 = будущее
                col = f"__tmp_mean_lag_{i}_{k}_{sign}"
                df = df.with_columns(
                    pl.col("sum_1h").shift(sign * i * k).over("route_id").alias(col)
                )
                tmp_cols.append(col)
            k += 1
        df = df.with_columns(
            pl.mean_horizontal(tmp_cols).alias(f"mean_sum_1h_lag_{i}")
        ).drop(tmp_cols)

    # Лаговые признаки
    for lag in LAGS:
        for feat in ["status_1", "status_2", "status_3", "status_4", "status_5", "status_6", "sum_1h"]:
            df = df.with_columns(
                pl.col(feat).shift(lag).over("route_id").alias(f"{feat}_lag_{lag}")
            )

    # Оконные признаки (скользящие статистики)
    for window in WINDOWS:
        for feat in ["status_1", "status_2", "status_3", "status_4", "status_5", "status_6", "sum_1h"]:
            df = df.with_columns([
                pl.col(feat)
                  .shift(1)
                  .rolling_mean(window_size=window, min_periods=1)
                  .over("route_id")
                  .alias(f"{feat}_roll_mean_{window}"),
                pl.col(feat)
                  .shift(1)
                  .rolling_std(window_size=window, min_periods=1)
                  .over("route_id")
                  .alias(f"{feat}_roll_std_{window}")
            ])

    # Временные признаки
    df = df.with_columns([
        pl.col("timestamp").dt.weekday().alias("_weekday"),
        pl.col("timestamp").dt.hour().alias("_hour"),
    ])
    df = df.with_columns([
        (pl.col("_weekday") >= 5).cast(pl.Int8).alias("is_weekend"),
    ])

    # One-hot кодировка часа (0–23)
    for h in range(24):
        df = df.with_columns(
            (pl.col("_hour") == h).cast(pl.Int8).alias(f"hour_{h}")
        )

    # One-hot кодировка дня недели (0=пн, 1=вт, ..., 6=вс)
    for d in range(7):
        df = df.with_columns(
            (pl.col("_weekday") == d).cast(pl.Int8).alias(f"weekday_{d}")
        )

    # Среднее sum_1h по маршруту
    df = df.with_columns(
        pl.col("sum_1h").mean().over("route_id").alias("route_mean")
    )

    # Target encoding: среднее sum_1h по (маршрут × час суток) и (маршрут × день недели)
    df = df.with_columns([
        pl.col("sum_1h").mean().over(["route_id", "_hour"]).alias("route_hour_mean"),
        pl.col("sum_1h").mean().over(["route_id", "_weekday"]).alias("route_weekday_mean"),
    ])

    df = df.drop(["_weekday", "_hour"])

    # Убираем август и сентябрь (опционально)
    if filter_aug_sep:
        df = df.filter(~pl.col("timestamp").dt.month().is_in([8, 9]))

    # Сохраняем в parquet
    result = df.collect()
    result.write_parquet(output_path)
    print(f"Сохранено {len(result):,} строк → {output_path}")

    return result
