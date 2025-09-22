"""データ品質検証Transformer（既存ライブラリ活用版）"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DataQualityValidator(BaseEstimator, TransformerMixin):
    """シンプルなデータ品質検証・修復"""

    def __init__(self, fix_dtypes=True, drop_high_missing=0.5, drop_zero_variance=True):
        self.fix_dtypes = fix_dtypes
        self.drop_high_missing = drop_high_missing
        self.drop_zero_variance = drop_zero_variance
        self._cols_to_drop = []

    def fit(self, X, y=None):
        self._cols_to_drop = []
        high_missing = []
        zero_var = []

        # 高欠損率列を検出
        if self.drop_high_missing:
            missing_rates = X.isnull().mean()
            high_missing = missing_rates[missing_rates > self.drop_high_missing].index.tolist()
            self._cols_to_drop.extend(high_missing)

        # 分散0列を検出
        if self.drop_zero_variance:
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            zero_var_mask = X[numeric_cols].var() == 0
            zero_var = zero_var_mask[zero_var_mask].index.tolist()
            self._cols_to_drop.extend(zero_var)

        print(f"✅ データ品質検証: 高欠損列{high_missing}, 分散0列{zero_var}")
        return self

    def transform(self, X):
        X_clean = X.copy()

        # 問題列を削除
        X_clean = X_clean.drop(columns=self._cols_to_drop, errors='ignore')

        # 数値型を自動修正
        if self.fix_dtypes:
            X_clean = X_clean.infer_objects()

        return X_clean