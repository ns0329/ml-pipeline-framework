#!/usr/bin/env python3
"""
ドメイン固有特徴量クラス群（プロジェクト毎に実装）
sklearn互換のパイプライン対応

このファイルはプロジェクト毎に具体的な特徴量生成クラスを実装する場所です。
例：Eコマース用特徴量、金融用特徴量、医療用特徴量など
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class EcommerceFeatureGenerator(BaseEstimator, TransformerMixin):
    """Eコマース固有特徴量生成（サンプル実装）"""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
        else:
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            return X

        X_new = X.copy()

        # Eコマース固有特徴量の例

        # 例：購入金額関連の特徴量
        if 'amount' in X.columns:
            # 金額の対数変換
            X_new['amount_log'] = np.log1p(X['amount'])
            print("✨ Eコマース特徴量生成: amount_log = log(1+amount)")

        # 例：時間関連の特徴量
        if 'transaction_time' in X.columns:
            # 時間帯別フラグ（例：深夜時間帯の取引）
            X_new['is_night_transaction'] = (
                (X['transaction_time'] >= 22) | (X['transaction_time'] <= 6)
            ).astype(int)
            print("✨ Eコマース特徴量生成: is_night_transaction (深夜取引フラグ)")

        return X_new

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, 'feature_names_in_', [])

        feature_names = list(input_features)

        # 生成される特徴量名を追加
        if 'amount' in input_features:
            feature_names.append('amount_log')
        if 'transaction_time' in input_features:
            feature_names.append('is_night_transaction')

        return np.array(feature_names)


# 他のドメイン用特徴量生成クラスもここに追加
# class FinanceFeatureGenerator(BaseEstimator, TransformerMixin):
#     """金融固有特徴量生成"""
#     pass

# class HealthcareFeatureGenerator(BaseEstimator, TransformerMixin):
#     """医療固有特徴量生成"""
#     pass