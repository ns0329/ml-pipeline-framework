#!/usr/bin/env python3
"""
汎用特徴量処理Transformer集
sklearn互換のパイプライン対応
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, f_regression


class LowContributionDropper(BaseEstimator, TransformerMixin):
    """予測に寄与しなさそうなカラム自動検出・ドロップ"""

    def __init__(self, variance_threshold=0.01, correlation_threshold=0.95,
                 missing_threshold=0.5, constant_threshold=0.95):
        """
        Args:
            variance_threshold: 分散が閾値以下の特徴量を削除
            correlation_threshold: 相関が閾値以上の特徴量ペアの片方を削除
            missing_threshold: 欠損率が閾値以上の特徴量を削除
            constant_threshold: 同一値の割合が閾値以上の特徴量を削除
        """
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.missing_threshold = missing_threshold
        self.constant_threshold = constant_threshold
        self.columns_to_drop_ = []
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        """学習フェーズ（削除対象カラムを特定）"""
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            X_numeric = X.select_dtypes(include=[np.number])
        else:
            X_numeric = pd.DataFrame(X)
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]

        self.columns_to_drop_ = []

        # 1. 欠損率チェック
        missing_ratios = X_numeric.isnull().mean()
        high_missing_cols = missing_ratios[missing_ratios > self.missing_threshold].index.tolist()
        self.columns_to_drop_.extend(high_missing_cols)
        print(f"🗑️ 高欠損率カラム削除: {high_missing_cols}")

        # 2. 定数値チェック（同一値の割合）
        for col in X_numeric.columns:
            if col not in self.columns_to_drop_:
                value_counts = X_numeric[col].value_counts()
                if len(value_counts) > 0:
                    most_frequent_ratio = value_counts.iloc[0] / len(X_numeric)
                    if most_frequent_ratio > self.constant_threshold:
                        self.columns_to_drop_.append(col)
                        print(f"🗑️ 定数値カラム削除: {col} (同一値率: {most_frequent_ratio:.3f})")

        # 3. 低分散チェック
        remaining_cols = [col for col in X_numeric.columns if col not in self.columns_to_drop_]
        if remaining_cols:
            X_remaining = X_numeric[remaining_cols].fillna(X_numeric[remaining_cols].mean())
            variances = X_remaining.var()
            low_variance_cols = variances[variances < self.variance_threshold].index.tolist()
            self.columns_to_drop_.extend(low_variance_cols)
            print(f"🗑️ 低分散カラム削除: {low_variance_cols}")

        # 4. 高相関チェック
        remaining_cols = [col for col in X_numeric.columns if col not in self.columns_to_drop_]
        if len(remaining_cols) > 1:
            X_remaining = X_numeric[remaining_cols].fillna(X_numeric[remaining_cols].mean())
            corr_matrix = X_remaining.corr().abs()

            # 上三角行列で高相関ペアを検索
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > self.correlation_threshold:
                        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        high_corr_pairs.append((col1, col2, corr_matrix.iloc[i, j]))

            # 高相関ペアの片方を削除（より多くのペアに含まれる方を削除）
            corr_counts = {}
            for col1, col2, corr_val in high_corr_pairs:
                corr_counts[col1] = corr_counts.get(col1, 0) + 1
                corr_counts[col2] = corr_counts.get(col2, 0) + 1

            for col1, col2, corr_val in high_corr_pairs:
                if corr_counts[col1] >= corr_counts[col2] and col1 not in self.columns_to_drop_:
                    self.columns_to_drop_.append(col1)
                    print(f"🗑️ 高相関カラム削除: {col1} (相関: {corr_val:.3f} with {col2})")
                elif col2 not in self.columns_to_drop_:
                    self.columns_to_drop_.append(col2)
                    print(f"🗑️ 高相関カラム削除: {col2} (相関: {corr_val:.3f} with {col1})")

        print(f"📊 削除対象カラム総数: {len(self.columns_to_drop_)} / {X.shape[1]}")
        return self

    def transform(self, X):
        """変換フェーズ（対象カラムを削除）"""
        if isinstance(X, pd.DataFrame):
            X_new = X.drop(columns=self.columns_to_drop_, errors='ignore')
        else:
            # numpy arrayの場合
            if self.feature_names_in_:
                drop_indices = [i for i, name in enumerate(self.feature_names_in_)
                              if name in self.columns_to_drop_]
                keep_indices = [i for i in range(X.shape[1]) if i not in drop_indices]
                X_new = X[:, keep_indices]
            else:
                X_new = X

        return X_new

    def get_feature_names_out(self, input_features=None):
        """出力特徴量名を返す"""
        if input_features is None:
            input_features = self.feature_names_in_

        if input_features:
            return np.array([name for name in input_features if name not in self.columns_to_drop_])
        else:
            return np.array([f"feature_{i}" for i in range(len(input_features) - len(self.columns_to_drop_))])


class CustomColumnDropper(BaseEstimator, TransformerMixin):
    """任意指定カラムドロップ"""

    def __init__(self, columns_to_drop=None):
        """
        Args:
            columns_to_drop: 削除対象カラムのリスト
        """
        self.columns_to_drop = columns_to_drop if columns_to_drop is not None else []

    def fit(self, X, y=None):
        """学習フェーズ"""
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
        else:
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]

        # 存在しないカラムを警告
        if isinstance(X, pd.DataFrame):
            missing_cols = [col for col in self.columns_to_drop if col not in X.columns]
            if missing_cols:
                print(f"⚠️ 削除対象カラムが存在しません: {missing_cols}")

        print(f"🗑️ 指定カラム削除: {self.columns_to_drop}")
        return self

    def transform(self, X):
        """変換フェーズ（指定カラムを削除）"""
        if isinstance(X, pd.DataFrame):
            X_new = X.drop(columns=self.columns_to_drop, errors='ignore')
        else:
            # numpy arrayの場合
            if self.feature_names_in_:
                drop_indices = [i for i, name in enumerate(self.feature_names_in_)
                              if name in self.columns_to_drop]
                keep_indices = [i for i in range(X.shape[1]) if i not in drop_indices]
                X_new = X[:, keep_indices]
            else:
                X_new = X

        return X_new

    def get_feature_names_out(self, input_features=None):
        """出力特徴量名を返す"""
        if input_features is None:
            input_features = self.feature_names_in_

        if input_features:
            return np.array([name for name in input_features if name not in self.columns_to_drop])
        else:
            return np.array([f"feature_{i}" for i in range(len(input_features) - len(self.columns_to_drop))])


class SmartFeatureSelector(BaseEstimator, TransformerMixin):
    """統計的手法による特徴量選択"""

    def __init__(self, k=10, score_func=None):
        """
        Args:
            k: 選択する特徴量数
            score_func: スコア関数（Noneの場合は自動判定）
        """
        self.k = k
        self.score_func = score_func
        self.selector_ = None

    def fit(self, X, y=None):
        """学習フェーズ"""
        if y is None:
            print("⚠️ ターゲット変数が提供されていません。特徴量選択をスキップします。")
            return self

        # スコア関数の自動判定
        if self.score_func is None:
            # ターゲット変数のタイプで判定
            if hasattr(y, 'dtype') and y.dtype.kind in 'iufc':
                unique_values = len(np.unique(y))
                total_values = len(y)
                if unique_values <= max(20, total_values * 0.05):
                    self.score_func = f_classif  # 分類
                    print("📊 分類タスクを検出: f_classif使用")
                else:
                    self.score_func = f_regression  # 回帰
                    print("📊 回帰タスクを検出: f_regression使用")
            else:
                self.score_func = f_classif  # デフォルト
                print("📊 分類タスクとして処理: f_classif使用")

        # 数値型データのみ選択
        if isinstance(X, pd.DataFrame):
            X_numeric = X.select_dtypes(include=[np.number])
        else:
            X_numeric = X

        # 欠損値処理
        if hasattr(X_numeric, 'fillna'):
            X_numeric = X_numeric.fillna(X_numeric.mean())
        else:
            # numpy array
            X_numeric = pd.DataFrame(X_numeric).fillna(pd.DataFrame(X_numeric).mean()).values

        # 特徴量選択器の学習
        actual_k = min(self.k, X_numeric.shape[1])
        self.selector_ = SelectKBest(score_func=self.score_func, k=actual_k)
        self.selector_.fit(X_numeric, y)

        selected_features = self.selector_.get_support()
        if isinstance(X, pd.DataFrame):
            selected_names = X_numeric.columns[selected_features].tolist()
            print(f"📊 統計的特徴量選択: {selected_names}")
        else:
            print(f"📊 統計的特徴量選択: {actual_k}個選択")

        return self

    def transform(self, X):
        """変換フェーズ"""
        if self.selector_ is None:
            return X

        if isinstance(X, pd.DataFrame):
            X_numeric = X.select_dtypes(include=[np.number])
            X_numeric_filled = X_numeric.fillna(X_numeric.mean())
            X_selected = self.selector_.transform(X_numeric_filled)

            # 選択された特徴量名を取得
            selected_features = self.selector_.get_support()
            selected_columns = X_numeric.columns[selected_features]

            return pd.DataFrame(X_selected, columns=selected_columns, index=X.index)
        else:
            X_filled = pd.DataFrame(X).fillna(pd.DataFrame(X).mean()).values
            return self.selector_.transform(X_filled)

    def get_feature_names_out(self, input_features=None):
        """出力特徴量名を返す"""
        if self.selector_ is None:
            return input_features

        if input_features is not None:
            selected_features = self.selector_.get_support()
            return np.array(input_features)[selected_features]
        else:
            return np.array([f"feature_{i}" for i in range(self.selector_.k)])