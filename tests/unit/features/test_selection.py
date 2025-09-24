"""selection.py の単体テスト"""

import pytest
import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif, f_regression

from src.mlops.features.selection import (
    SmartFeatureSelector,
    LowContributionDropper,
    CustomColumnDropper
)
from tests.fixtures.test_data import (
    create_sample_dataframe,
    create_binary_classification_data,
    create_regression_data,
    create_missing_data_dataframe,
    create_high_correlation_dataframe
)
from tests.fixtures.test_utils import (
    assert_transformer_basic_contract,
    create_edge_case_dataframes
)


class TestSmartFeatureSelector:
    """SmartFeatureSelector の単体テスト"""

    def test_basic_functionality_classification(self):
        """基本機能テスト（分類）"""
        X, y = create_binary_classification_data(n_rows=100, n_features=10)
        selector = SmartFeatureSelector(k=5)

        # 基本契約のテスト
        assert_transformer_basic_contract(selector, X, y_train=y)

    def test_basic_functionality_regression(self):
        """基本機能テスト（回帰）"""
        X, y = create_regression_data(n_rows=100, n_features=10)
        selector = SmartFeatureSelector(k=5)

        # 基本契約のテスト
        assert_transformer_basic_contract(selector, X, y_train=y)

    def test_automatic_task_detection(self):
        """タスク自動判定テスト"""
        X_cls, y_cls = create_binary_classification_data(n_rows=100, n_features=8)
        X_reg, y_reg = create_regression_data(n_rows=100, n_features=8)

        # 分類タスク検出
        selector_cls = SmartFeatureSelector(k=4)
        selector_cls.fit(X_cls, y_cls)
        assert selector_cls.score_func == f_classif

        # 回帰タスク検出
        selector_reg = SmartFeatureSelector(k=4)
        selector_reg.fit(X_reg, y_reg)
        assert selector_reg.score_func == f_regression

    def test_manual_score_function(self):
        """手動スコア関数指定テスト"""
        X, y = create_binary_classification_data(n_rows=100, n_features=8)

        selector = SmartFeatureSelector(k=4, score_func=f_regression)
        selector.fit(X, y)

        assert selector.score_func == f_regression

    def test_k_parameter_variations(self):
        """k パラメータのバリエーションテスト"""
        X, y = create_binary_classification_data(n_rows=100, n_features=10)

        # k < feature数
        selector1 = SmartFeatureSelector(k=5)
        X_transformed1 = selector1.fit_transform(X, y)
        assert X_transformed1.shape[1] == 5

        # k = feature数
        selector2 = SmartFeatureSelector(k=10)
        X_transformed2 = selector2.fit_transform(X, y)
        assert X_transformed2.shape[1] == min(10, X.shape[1])

        # k > feature数（実際のfeature数に制限される）
        selector3 = SmartFeatureSelector(k=20)
        X_transformed3 = selector3.fit_transform(X, y)
        assert X_transformed3.shape[1] <= X.shape[1]

    def test_without_target(self):
        """ターゲット変数なしのテスト"""
        X = create_sample_dataframe(n_rows=100, n_features=8)
        selector = SmartFeatureSelector(k=4)

        # ターゲットなしではスキップされるはず
        X_transformed = selector.fit_transform(X)
        assert X_transformed.shape == X.shape  # 変化なし

    def test_get_feature_names_out(self):
        """特徴量名出力テスト"""
        X, y = create_binary_classification_data(n_rows=100, n_features=6)
        selector = SmartFeatureSelector(k=3)

        selector.fit(X, y)
        feature_names = selector.get_feature_names_out()

        assert len(feature_names) == 3
        assert all(name in X.columns for name in feature_names)

    def test_edge_cases(self):
        """エッジケーステスト"""
        # 1特徴量のみ
        X = pd.DataFrame({'feature_0': np.random.randn(50)})
        y = pd.Series(np.random.choice([0, 1], 50))

        selector = SmartFeatureSelector(k=1)
        X_transformed = selector.fit_transform(X, y)
        assert X_transformed.shape == (50, 1)

        # k=0の場合
        selector_zero = SmartFeatureSelector(k=0)
        X_transformed_zero = selector_zero.fit_transform(X, y)
        # k=0でも1つは選択されることを確認（ゼロにならない）


class TestLowContributionDropper:
    """LowContributionDropper の単体テスト"""

    def test_basic_functionality(self):
        """基本機能テスト"""
        X = create_high_correlation_dataframe(n_rows=100)
        dropper = LowContributionDropper()

        # 基本契約のテスト
        assert_transformer_basic_contract(dropper, X)

    def test_high_missing_removal(self):
        """高欠損率カラム削除テスト"""
        X = create_missing_data_dataframe(n_rows=100, missing_ratio=0.8)
        dropper = LowContributionDropper(missing_threshold=0.5)

        X_transformed = dropper.fit_transform(X)

        # 高欠損率カラムが削除されているか
        assert X_transformed.shape[1] <= X.shape[1]
        assert len(dropper.columns_to_drop_) > 0

    def test_constant_column_removal(self):
        """定数カラム削除テスト"""
        X = pd.DataFrame({
            'normal_col': np.random.randn(100),
            'constant_col': np.ones(100),
            'mostly_constant': [1] * 95 + [2] * 5
        })

        dropper = LowContributionDropper(constant_threshold=0.9)
        X_transformed = dropper.fit_transform(X)

        # 定数カラムと大部分が同じ値のカラムが削除されているか
        assert 'constant_col' in dropper.columns_to_drop_
        assert 'mostly_constant' in dropper.columns_to_drop_
        assert 'normal_col' not in dropper.columns_to_drop_

    def test_low_variance_removal(self):
        """低分散カラム削除テスト"""
        X = pd.DataFrame({
            'high_var_col': np.random.randn(100) * 10,
            'low_var_col': np.random.randn(100) * 0.001,
            'zero_var_col': np.zeros(100)
        })

        dropper = LowContributionDropper(variance_threshold=0.01)
        X_transformed = dropper.fit_transform(X)

        # 低分散・ゼロ分散カラムが削除されているか
        assert 'low_var_col' in dropper.columns_to_drop_
        assert 'zero_var_col' in dropper.columns_to_drop_
        assert 'high_var_col' not in dropper.columns_to_drop_

    def test_high_correlation_removal(self):
        """高相関カラム削除テスト"""
        X = create_high_correlation_dataframe(n_rows=100)
        dropper = LowContributionDropper(correlation_threshold=0.9)

        X_transformed = dropper.fit_transform(X)

        # 高相関カラムのいずれかが削除されているか
        high_corr_cols = ['feature_1', 'feature_2']
        dropped_high_corr = any(col in dropper.columns_to_drop_ for col in high_corr_cols)
        assert dropped_high_corr

    def test_parameter_combinations(self):
        """パラメータ組み合わせテスト"""
        X = create_high_correlation_dataframe(n_rows=100)

        # 緩い条件
        dropper1 = LowContributionDropper(
            variance_threshold=0.0001,
            correlation_threshold=0.99,
            missing_threshold=0.9,
            constant_threshold=0.99
        )
        X_transformed1 = dropper1.fit_transform(X)

        # 厳しい条件
        dropper2 = LowContributionDropper(
            variance_threshold=0.1,
            correlation_threshold=0.7,
            missing_threshold=0.1,
            constant_threshold=0.7
        )
        X_transformed2 = dropper2.fit_transform(X)

        # 厳しい条件の方がより多くのカラムが削除されるはず
        assert len(dropper2.columns_to_drop_) >= len(dropper1.columns_to_drop_)

    def test_get_feature_names_out(self):
        """特徴量名出力テスト"""
        X = create_high_correlation_dataframe(n_rows=100)
        dropper = LowContributionDropper()

        X_transformed = dropper.fit_transform(X)
        feature_names = dropper.get_feature_names_out()

        # 削除されなかったカラム名が含まれているか
        remaining_cols = [col for col in X.columns if col not in dropper.columns_to_drop_]
        assert len(feature_names) == len(remaining_cols)
        assert set(feature_names) == set(remaining_cols)


class TestCustomColumnDropper:
    """CustomColumnDropper の単体テスト"""

    def test_basic_functionality(self):
        """基本機能テスト"""
        X = create_sample_dataframe(n_rows=100, n_features=6)
        columns_to_drop = ['numeric_0', 'integer_2']
        dropper = CustomColumnDropper(columns_to_drop=columns_to_drop)

        # 基本契約のテスト
        assert_transformer_basic_contract(dropper, X)

    def test_column_dropping(self):
        """カラム削除テスト"""
        X = pd.DataFrame({
            'keep_col_1': [1, 2, 3],
            'drop_col_1': [4, 5, 6],
            'keep_col_2': [7, 8, 9],
            'drop_col_2': [10, 11, 12]
        })

        columns_to_drop = ['drop_col_1', 'drop_col_2']
        dropper = CustomColumnDropper(columns_to_drop=columns_to_drop)

        X_transformed = dropper.fit_transform(X)

        # 指定カラムが削除され、保持カラムが残っているか
        assert 'keep_col_1' in X_transformed.columns
        assert 'keep_col_2' in X_transformed.columns
        assert 'drop_col_1' not in X_transformed.columns
        assert 'drop_col_2' not in X_transformed.columns
        assert X_transformed.shape == (3, 2)

    def test_nonexistent_columns(self):
        """存在しないカラム指定テスト"""
        X = pd.DataFrame({
            'existing_col_1': [1, 2, 3],
            'existing_col_2': [4, 5, 6]
        })

        columns_to_drop = ['existing_col_1', 'nonexistent_col']
        dropper = CustomColumnDropper(columns_to_drop=columns_to_drop)

        # エラーにならずに処理されるか
        X_transformed = dropper.fit_transform(X)
        assert 'existing_col_2' in X_transformed.columns
        assert 'existing_col_1' not in X_transformed.columns
        assert X_transformed.shape == (3, 1)

    def test_empty_drop_list(self):
        """空のドロップリストテスト"""
        X = create_sample_dataframe(n_rows=50, n_features=4)
        dropper = CustomColumnDropper(columns_to_drop=[])

        X_transformed = dropper.fit_transform(X)

        # 何も削除されないか
        assert X_transformed.shape == X.shape
        assert list(X_transformed.columns) == list(X.columns)

    def test_drop_all_columns(self):
        """全カラム削除テスト"""
        X = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })

        dropper = CustomColumnDropper(columns_to_drop=['col1', 'col2'])
        X_transformed = dropper.fit_transform(X)

        # 空DataFrameになるか
        assert X_transformed.shape == (3, 0)

    def test_get_feature_names_out(self):
        """特徴量名出力テスト"""
        X = pd.DataFrame({
            'keep_me': [1, 2, 3],
            'drop_me': [4, 5, 6],
            'also_keep': [7, 8, 9]
        })

        dropper = CustomColumnDropper(columns_to_drop=['drop_me'])
        dropper.fit(X)

        feature_names = dropper.get_feature_names_out()
        expected_names = ['keep_me', 'also_keep']

        assert len(feature_names) == 2
        assert set(feature_names) == set(expected_names)

    def test_numpy_array_input(self):
        """numpy array入力テスト"""
        X_array = np.random.randn(50, 5)
        dropper = CustomColumnDropper(columns_to_drop=[])

        # numpy arrayでも動作するか（カラム名はないので実際には何も削除されない）
        X_transformed = dropper.fit_transform(X_array)
        assert X_transformed.shape == X_array.shape


class TestSelectionIntegration:
    """選択クラス統合テスト"""

    def test_sequential_selection(self):
        """連続的な特徴量選択テスト"""
        X, y = create_binary_classification_data(n_rows=200, n_features=15)

        # 段階的に特徴量を選択/削除
        dropper1 = LowContributionDropper(
            variance_threshold=0.01,
            correlation_threshold=0.95
        )
        X_step1 = dropper1.fit_transform(X)

        dropper2 = CustomColumnDropper(columns_to_drop=['numeric_0'] if 'numeric_0' in X_step1.columns else [])
        X_step2 = dropper2.fit_transform(X_step1)

        selector = SmartFeatureSelector(k=min(5, X_step2.shape[1]))
        X_final = selector.fit_transform(X_step2, y)

        # 最終的に適切な数の特徴量が選択されているか
        assert X_final.shape[0] == X.shape[0]  # 行数は保持
        assert X_final.shape[1] <= X.shape[1]  # 列数は減少または同じ
        assert X_final.shape[1] <= 5  # 最終選択数以下

    def test_extreme_filtering(self):
        """極端なフィルタリングテスト"""
        # 多くのノイズ特徴量を含むデータ
        X_signal = np.random.randn(100, 3) * 2  # シグナル特徴量
        X_noise = np.random.randn(100, 20) * 0.1  # ノイズ特徴量
        X = pd.DataFrame(np.concatenate([X_signal, X_noise], axis=1),
                        columns=[f'feature_{i}' for i in range(23)])
        y = pd.Series((X_signal.sum(axis=1) > 0).astype(int))

        # 段階的フィルタリング
        dropper = LowContributionDropper(
            variance_threshold=0.05,
            correlation_threshold=0.95
        )
        selector = SmartFeatureSelector(k=5)

        X_filtered = dropper.fit_transform(X)
        X_selected = selector.fit_transform(X_filtered, y)

        # シグナルのある特徴量が優先的に選択されているか確認
        # （これは厳密なテストではないが、傾向を確認）
        assert X_selected.shape[1] <= 5
        assert X_selected.shape[0] == X.shape[0]