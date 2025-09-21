#!/usr/bin/env python3
"""
カスタム特徴量生成Transformer
sklearn互換のパイプライン対応
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class WineRatioFeatures(BaseEstimator, TransformerMixin):
    """ワインデータ用カスタム比率特徴量生成"""

    def __init__(self, ratio_pairs=None):
        """
        Args:
            ratio_pairs: [(分子カラム, 分母カラム, 新特徴量名), ...] のリスト
        """
        self.ratio_pairs = ratio_pairs or [
            ('alcohol', 'total_phenols', 'alcohol_phenols_ratio'),
            ('flavanoids', 'nonflavanoid_phenols', 'flavanoid_ratio'),
            ('color_intensity', 'hue', 'color_hue_ratio')
        ]

    def fit(self, X, y=None):
        """学習フェーズ（統計情報保存）"""
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """変換フェーズ（新特徴量生成）"""
        X_new = X.copy()

        for numerator, denominator, feature_name in self.ratio_pairs:
            if numerator in X.columns and denominator in X.columns:
                # ゼロ除算対応
                X_new[feature_name] = np.where(
                    X[denominator] != 0,
                    X[numerator] / X[denominator],
                    0
                )

        return X_new

    def get_feature_names_out(self, input_features=None):
        """出力特徴量名を返す（sklearn 1.0+対応）"""
        if input_features is None:
            # 推定値
            input_features = [f"x{i}" for i in range(13)]

        feature_names = list(input_features)
        for _, _, feature_name in self.ratio_pairs:
            feature_names.append(feature_name)

        return np.array(feature_names)


class WineInteractionFeatures(BaseEstimator, TransformerMixin):
    """ワインデータ用交互作用特徴量生成"""

    def __init__(self, interaction_pairs=None):
        """
        Args:
            interaction_pairs: [(カラム1, カラム2, 新特徴量名), ...] のリスト
        """
        self.interaction_pairs = interaction_pairs or [
            ('alcohol', 'color_intensity', 'alcohol_x_color'),
            ('total_phenols', 'flavanoids', 'phenols_x_flavanoids'),
            ('ash', 'alcalinity_of_ash', 'ash_x_alcalinity')
        ]

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X_new = X.copy()

        for col1, col2, feature_name in self.interaction_pairs:
            if col1 in X.columns and col2 in X.columns:
                X_new[feature_name] = X[col1] * X[col2]

        return X_new

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(16)]  # 前ステップで増加後

        feature_names = list(input_features)
        for _, _, feature_name in self.interaction_pairs:
            feature_names.append(feature_name)

        return np.array(feature_names)


class WineAggregateFeatures(BaseEstimator, TransformerMixin):
    """ワインデータ用集約特徴量生成"""

    def __init__(self, feature_groups=None):
        """
        Args:
            feature_groups: {グループ名: [カラムリスト]} の辞書
        """
        self.feature_groups = feature_groups or {
            'phenol_group': ['total_phenols', 'flavanoids', 'nonflavanoid_phenols'],
            'acid_group': ['malic_acid', 'ash', 'alcalinity_of_ash'],
            'color_group': ['color_intensity', 'hue', 'od280/od315_of_diluted_wines']
        }

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X_new = X.copy()

        for group_name, columns in self.feature_groups.items():
            available_cols = [col for col in columns if col in X.columns]
            if available_cols:
                # 合計・平均・標準偏差を生成
                X_new[f'{group_name}_sum'] = X[available_cols].sum(axis=1)
                X_new[f'{group_name}_mean'] = X[available_cols].mean(axis=1)
                X_new[f'{group_name}_std'] = X[available_cols].std(axis=1)

        return X_new

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(18)]  # 前ステップで増加後

        feature_names = list(input_features)
        for group_name in self.feature_groups.keys():
            feature_names.extend([
                f'{group_name}_sum',
                f'{group_name}_mean',
                f'{group_name}_std'
            ])

        return np.array(feature_names)