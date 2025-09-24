"""テスト用データフィクスチャ"""

import pandas as pd
import numpy as np
from typing import Tuple


def create_sample_dataframe(n_rows: int = 100, n_features: int = 5, random_state: int = 42) -> pd.DataFrame:
    """テスト用サンプルDataFrameを生成"""
    np.random.seed(random_state)

    data = {}
    for i in range(n_features):
        if i % 3 == 0:  # 数値型
            data[f'numeric_{i}'] = np.random.randn(n_rows)
        elif i % 3 == 1:  # カテゴリ型
            data[f'category_{i}'] = np.random.choice(['A', 'B', 'C'], n_rows)
        else:  # 整数型
            data[f'integer_{i}'] = np.random.randint(0, 100, n_rows)

    return pd.DataFrame(data)


def create_binary_classification_data(n_rows: int = 100, n_features: int = 5, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """二値分類用テストデータ生成"""
    np.random.seed(random_state)

    # 特徴量生成
    X = create_sample_dataframe(n_rows, n_features, random_state)

    # ターゲット生成（二値分類）
    y = pd.Series(np.random.choice([0, 1], n_rows), name='target')

    return X, y


def create_regression_data(n_rows: int = 100, n_features: int = 5, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """回帰用テストデータ生成"""
    np.random.seed(random_state)

    # 特徴量生成
    X = create_sample_dataframe(n_rows, n_features, random_state)

    # ターゲット生成（連続値）
    y = pd.Series(np.random.randn(n_rows) * 10 + 50, name='target')

    return X, y


def create_missing_data_dataframe(n_rows: int = 100, missing_ratio: float = 0.3, random_state: int = 42) -> pd.DataFrame:
    """欠損値を含むテストデータ生成"""
    df = create_sample_dataframe(n_rows, 5, random_state)

    np.random.seed(random_state)
    for col in df.columns:
        mask = np.random.random(n_rows) < missing_ratio
        df.loc[mask, col] = np.nan

    return df


def create_high_correlation_dataframe(n_rows: int = 100, random_state: int = 42) -> pd.DataFrame:
    """高相関データを含むテストデータ生成"""
    np.random.seed(random_state)

    # ベースデータ
    base = np.random.randn(n_rows)

    data = {
        'feature_1': base + np.random.randn(n_rows) * 0.1,  # 高相関
        'feature_2': base + np.random.randn(n_rows) * 0.1,  # 高相関
        'feature_3': np.random.randn(n_rows),  # 独立
        'feature_4': np.random.randn(n_rows),  # 独立
        'constant_col': np.ones(n_rows),  # 定数列
    }

    return pd.DataFrame(data)


def create_outlier_dataframe(n_rows: int = 100, outlier_ratio: float = 0.05, random_state: int = 42) -> pd.DataFrame:
    """異常値を含むテストデータ生成"""
    df = create_sample_dataframe(n_rows, 5, random_state)

    np.random.seed(random_state)
    n_outliers = int(n_rows * outlier_ratio)
    outlier_indices = np.random.choice(n_rows, n_outliers, replace=False)

    for col in df.select_dtypes(include=[np.number]).columns:
        df.loc[outlier_indices, col] *= 100  # 異常値を作成

    return df


def create_ecommerce_sample_data(n_rows: int = 100, random_state: int = 42) -> pd.DataFrame:
    """Eコマース風サンプルデータ生成"""
    np.random.seed(random_state)

    data = {
        'amount': np.random.exponential(scale=50, size=n_rows),
        'transaction_time': np.random.randint(0, 24, n_rows),
        'customer_age': np.random.randint(18, 80, n_rows),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Books'], n_rows),
        'payment_method': np.random.choice(['Credit', 'Debit', 'PayPal'], n_rows),
    }

    return pd.DataFrame(data)