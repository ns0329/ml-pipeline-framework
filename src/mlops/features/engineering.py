#!/usr/bin/env python3
"""
特徴生成系統の汎用テンプレートクラス群
sklearn互換のパイプライン対応
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RatioFeatureGenerator(BaseEstimator, TransformerMixin):
    """汎用比率特徴量生成器"""

    def __init__(self, ratio_pairs=None):
        """
        Args:
            ratio_pairs: [(分子カラム, 分母カラム, 新特徴量名), ...] のリスト
        """
        self.ratio_pairs = ratio_pairs or []

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
        else:
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_new = X.copy()

            for numerator, denominator, feature_name in self.ratio_pairs:
                if numerator in X.columns and denominator in X.columns:
                    X_new[feature_name] = np.where(
                        X[denominator] != 0,
                        X[numerator] / X[denominator],
                        0
                    )
                    print(f"✨ 比率特徴量生成: {feature_name} = {numerator}/{denominator}")
            return X_new
        else:
            return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, 'feature_names_in_',
                                   [f"feature_{i}" for i in range(self.n_features_in_)])

        feature_names = list(input_features)
        for _, _, feature_name in self.ratio_pairs:
            feature_names.append(feature_name)

        return np.array(feature_names)


class InteractionFeatureGenerator(BaseEstimator, TransformerMixin):
    """汎用交互作用特徴量生成器"""

    def __init__(self, interaction_pairs=None):
        """
        Args:
            interaction_pairs: [(カラム1, カラム2, 新特徴量名), ...] のリスト
        """
        self.interaction_pairs = interaction_pairs or []

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
        else:
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_new = X.copy()

            for col1, col2, feature_name in self.interaction_pairs:
                if col1 in X.columns and col2 in X.columns:
                    X_new[feature_name] = X[col1] * X[col2]
                    print(f"✨ 交互作用特徴量生成: {feature_name} = {col1} × {col2}")
            return X_new
        else:
            return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, 'feature_names_in_',
                                   [f"feature_{i}" for i in range(self.n_features_in_)])

        feature_names = list(input_features)
        for _, _, feature_name in self.interaction_pairs:
            feature_names.append(feature_name)

        return np.array(feature_names)


class AggregateFeatureGenerator(BaseEstimator, TransformerMixin):
    """汎用集約特徴量生成器"""

    def __init__(self, feature_groups=None):
        """
        Args:
            feature_groups: {グループ名: [カラムリスト]} の辞書
        """
        self.feature_groups = feature_groups or {}

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
        else:
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_new = X.copy()

            for group_name, columns in self.feature_groups.items():
                available_cols = [col for col in columns if col in X.columns]
                if available_cols:
                    X_new[f'{group_name}_sum'] = X[available_cols].sum(axis=1)
                    X_new[f'{group_name}_mean'] = X[available_cols].mean(axis=1)
                    X_new[f'{group_name}_std'] = X[available_cols].std(axis=1)
                    print(f"✨ 集約特徴量生成: {group_name}系統 ({len(available_cols)}カラムから3特徴量)")
            return X_new
        else:
            return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, 'feature_names_in_',
                                   [f"feature_{i}" for i in range(self.n_features_in_)])

        feature_names = list(input_features)
        for group_name in self.feature_groups.keys():
            feature_names.extend([
                f'{group_name}_sum',
                f'{group_name}_mean',
                f'{group_name}_std'
            ])

        return np.array(feature_names)


class PolynomialFeatureGenerator(BaseEstimator, TransformerMixin):
    """汎用多項式特徴量生成器"""

    def __init__(self, target_columns=None, degree=2):
        """
        Args:
            target_columns: 多項式特徴量を生成する対象カラム
            degree: 多項式の次数
        """
        self.target_columns = target_columns or []
        self.degree = degree

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
        else:
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_new = X.copy()

            for col in self.target_columns:
                if col in X.columns:
                    for d in range(2, self.degree + 1):
                        feature_name = f"{col}_pow{d}"
                        X_new[feature_name] = X[col] ** d
                        print(f"✨ 多項式特徴量生成: {feature_name} = {col}^{d}")
            return X_new
        else:
            return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, 'feature_names_in_',
                                   [f"feature_{i}" for i in range(self.n_features_in_)])

        feature_names = list(input_features)
        for col in self.target_columns:
            for d in range(2, self.degree + 1):
                feature_names.append(f"{col}_pow{d}")

        return np.array(feature_names)