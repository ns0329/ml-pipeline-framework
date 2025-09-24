#!/usr/bin/env python3
"""
データ前処理系統の汎用クラス群
sklearn互換のパイプライン対応
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DataQualityValidator(BaseEstimator, TransformerMixin):
    """データ品質検証・修復"""

    def __init__(self, fix_dtypes=True, drop_high_missing=0.5, drop_zero_variance=True):
        self.fix_dtypes = fix_dtypes
        self.drop_high_missing = drop_high_missing
        self.drop_zero_variance = drop_zero_variance
        self.columns_to_drop = []

    def fit(self, X, y=None):
        self.columns_to_drop = []

        if self.drop_high_missing:
            missing_ratio = X.isnull().mean()
            high_missing_cols = missing_ratio[missing_ratio > self.drop_high_missing].index.tolist()
            self.columns_to_drop.extend(high_missing_cols)
            print(f"🗑️ 高欠損率カラム削除対象: {high_missing_cols}")

        if self.drop_zero_variance:
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in self.columns_to_drop and X[col].var() == 0:
                    self.columns_to_drop.append(col)
                    print(f"🗑️ 分散ゼロカラム削除対象: {col}")

        return self

    def transform(self, X):
        X_new = X.copy()

        # 高欠損・分散ゼロカラム削除
        if self.columns_to_drop:
            X_new = X_new.drop(columns=self.columns_to_drop, errors='ignore')
            print(f"📊 データ品質検証完了: {len(self.columns_to_drop)}カラム削除")

        return X_new


class DtypeOptimizer(BaseEstimator, TransformerMixin):
    """データ型最適化"""

    def __init__(self, model_name=None, force_dtype=None):
        self.model_name = model_name
        self.force_dtype = force_dtype
        self.dtype_map = {}

    def fit(self, X, y=None):
        self.dtype_map = {}

        # LightGBM等は元のdtypeを維持
        if self.model_name in ['LGBMClassifier', 'LGBMRegressor']:
            print(f"📊 {self.model_name}: 元データ型を維持")
            return self

        # 強制指定がある場合
        if self.force_dtype:
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                self.dtype_map[col] = self.force_dtype
            print(f"📊 データ型強制変換: {self.force_dtype}")
        else:
            # 自動最適化
            for col in X.select_dtypes(include=[np.number]).columns:
                if X[col].dtype == 'float64':
                    if (X[col] == X[col].astype('float32')).all():
                        self.dtype_map[col] = 'float32'

        return self

    def transform(self, X):
        X_new = X.copy()

        for col, dtype in self.dtype_map.items():
            if col in X_new.columns:
                X_new[col] = X_new[col].astype(dtype)

        if self.dtype_map:
            print(f"📊 データ型最適化完了: {len(self.dtype_map)}カラム変換")

        return X_new