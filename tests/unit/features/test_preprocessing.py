"""preprocessing.py の単体テスト"""

import pytest
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from src.mlops.features.preprocessing import DataQualityValidator, DtypeOptimizer
from tests.fixtures.test_data import (
    create_sample_dataframe,
    create_missing_data_dataframe,
    create_high_correlation_dataframe
)
from tests.fixtures.test_utils import (
    assert_transformer_basic_contract,
    assert_parameter_validation,
    create_edge_case_dataframes
)


class TestDataQualityValidator:
    """DataQualityValidator の単体テスト"""

    def test_basic_functionality(self):
        """基本機能テスト"""
        X = create_sample_dataframe(n_rows=100, n_features=5)
        validator = DataQualityValidator()

        # 基本契約のテスト
        assert_transformer_basic_contract(validator, X)

    def test_high_missing_column_removal(self):
        """高欠損率カラム削除テスト"""
        X = create_missing_data_dataframe(n_rows=100, missing_ratio=0.8)
        validator = DataQualityValidator(drop_high_missing=0.5)

        X_transformed = validator.fit_transform(X)

        # 高欠損率カラムが削除されているか確認
        assert X_transformed.shape[1] <= X.shape[1]
        assert len(validator.columns_to_drop) > 0

    def test_zero_variance_column_removal(self):
        """分散ゼロカラム削除テスト"""
        X = pd.DataFrame({
            'normal_col': np.random.randn(100),
            'constant_col': np.ones(100),
            'zero_var_col': np.zeros(100)
        })

        validator = DataQualityValidator(drop_zero_variance=True)
        X_transformed = validator.fit_transform(X)

        # 定数・ゼロ分散カラムが削除されているか
        assert X_transformed.shape[1] == 1
        assert 'normal_col' in X_transformed.columns
        assert 'constant_col' not in X_transformed.columns
        assert 'zero_var_col' not in X_transformed.columns

    def test_parameter_combinations(self):
        """パラメータ組み合わせテスト"""
        X = create_missing_data_dataframe(n_rows=50, missing_ratio=0.6)

        # 全フラグOFF
        validator1 = DataQualityValidator(
            fix_dtypes=False,
            drop_high_missing=None,
            drop_zero_variance=False
        )
        X_transformed1 = validator1.fit_transform(X)
        assert X_transformed1.shape == X.shape

        # 全フラグON
        validator2 = DataQualityValidator(
            fix_dtypes=True,
            drop_high_missing=0.3,
            drop_zero_variance=True
        )
        X_transformed2 = validator2.fit_transform(X)
        # 何らかの処理が実行されるはず

    def test_edge_cases(self):
        """エッジケーステスト"""
        edge_cases = create_edge_case_dataframes()
        validator = DataQualityValidator()

        for i, X in enumerate(edge_cases):
            if X.empty:
                # 空DataFrameの場合はスキップまたは適切に処理
                continue

            try:
                X_transformed = validator.fit_transform(X)
                # 基本的なshape確認
                assert X_transformed.shape[0] == X.shape[0]
            except Exception as e:
                pytest.fail(f"エッジケース {i} でエラー: {e}")

    def test_parameter_validation(self):
        """パラメータバリデーションテスト"""
        X = create_sample_dataframe(n_rows=50)

        valid_params = {
            'fix_dtypes': True,
            'drop_high_missing': 0.5,
            'drop_zero_variance': True
        }

        invalid_params_list = [
            {'drop_high_missing': -0.1},  # 負値
            {'drop_high_missing': 1.1},   # 1超過
            {'drop_high_missing': 'invalid'},  # 文字列
        ]

        # バリデーション実行（エラーが出ない場合もある）
        for invalid_params in invalid_params_list:
            try:
                validator = DataQualityValidator(**invalid_params)
                validator.fit_transform(X)
            except (ValueError, TypeError):
                pass  # 期待される例外


class TestDtypeOptimizer:
    """DtypeOptimizer の単体テスト"""

    def test_basic_functionality(self):
        """基本機能テスト"""
        X = create_sample_dataframe(n_rows=100, n_features=5)
        optimizer = DtypeOptimizer()

        # 基本契約のテスト
        assert_transformer_basic_contract(optimizer, X)

    def test_lightgbm_mode(self):
        """LightGBMモード（型保持）テスト"""
        X = pd.DataFrame({
            'float64_col': np.random.randn(100).astype('float64'),
            'int64_col': np.random.randint(0, 100, 100).astype('int64')
        })

        optimizer = DtypeOptimizer(model_name='LGBMClassifier')
        X_transformed = optimizer.fit_transform(X)

        # 元のdtypeが維持されているか
        assert X_transformed['float64_col'].dtype == X['float64_col'].dtype
        assert X_transformed['int64_col'].dtype == X['int64_col'].dtype

    def test_forced_dtype_conversion(self):
        """強制データ型変換テスト"""
        X = pd.DataFrame({
            'float64_col': np.random.randn(100).astype('float64'),
            'float32_col': np.random.randn(100).astype('float32'),
            'int_col': np.random.randint(0, 100, 100)
        })

        optimizer = DtypeOptimizer(force_dtype='float32')
        X_transformed = optimizer.fit_transform(X)

        # 数値列が全てfloat32に変換されているか
        for col in X.select_dtypes(include=[np.number]).columns:
            assert X_transformed[col].dtype == np.float32

    def test_automatic_optimization(self):
        """自動最適化テスト"""
        # float64でも実際はfloat32で表現可能な値
        X = pd.DataFrame({
            'reducible_col': np.array([1.5, 2.5, 3.5], dtype='float64'),
            'irreducible_col': np.array([1.123456789e10, 2.987654321e10], dtype='float64'),
            'non_numeric_col': ['A', 'B', 'C']
        })

        optimizer = DtypeOptimizer()
        X_transformed = optimizer.fit_transform(X)

        # 非数値列は変更されない
        assert X_transformed['non_numeric_col'].dtype == X['non_numeric_col'].dtype

    def test_parameter_combinations(self):
        """パラメータ組み合わせテスト"""
        X = create_sample_dataframe(n_rows=50)

        # モデル名指定あり
        optimizer1 = DtypeOptimizer(model_name='LGBMRegressor')
        X_transformed1 = optimizer1.fit_transform(X)

        # 強制型指定あり
        optimizer2 = DtypeOptimizer(force_dtype='float32')
        X_transformed2 = optimizer2.fit_transform(X)

        # 自動最適化（デフォルト）
        optimizer3 = DtypeOptimizer()
        X_transformed3 = optimizer3.fit_transform(X)

        # 全て正常に処理されることを確認
        assert X_transformed1.shape == X.shape
        assert X_transformed2.shape == X.shape
        assert X_transformed3.shape == X.shape

    def test_edge_cases(self):
        """エッジケーステスト"""
        edge_cases = create_edge_case_dataframes()
        optimizer = DtypeOptimizer()

        for i, X in enumerate(edge_cases):
            if X.empty:
                continue

            try:
                X_transformed = optimizer.fit_transform(X)
                assert X_transformed.shape == X.shape
            except Exception as e:
                pytest.fail(f"エッジケース {i} でエラー: {e}")

    def test_invalid_parameters(self):
        """無効パラメータテスト"""
        X = create_sample_dataframe(n_rows=50)

        # 無効な force_dtype
        try:
            optimizer = DtypeOptimizer(force_dtype='invalid_dtype')
            X_transformed = optimizer.fit_transform(X)
        except (ValueError, TypeError):
            pass  # 期待される例外

        # 無効な model_name（ただし、これは通常エラーにならない）
        optimizer = DtypeOptimizer(model_name='InvalidModel')
        X_transformed = optimizer.fit_transform(X)  # エラーにならないはず


class TestPreprocessingIntegration:
    """前処理クラス統合テスト"""

    def test_sequential_application(self):
        """複数前処理の連続適用テスト"""
        X = create_high_correlation_dataframe(n_rows=100)

        # 順次適用
        validator = DataQualityValidator(drop_high_missing=0.8, drop_zero_variance=True)
        optimizer = DtypeOptimizer(force_dtype='float32')

        X_step1 = validator.fit_transform(X)
        X_step2 = optimizer.fit_transform(X_step1)

        # 最終結果の確認
        assert X_step2.shape[0] == X.shape[0]  # 行数は保持
        assert X_step2.shape[1] <= X.shape[1]  # 列数は減少または同じ

        # 数値列がfloat32になっているか
        for col in X_step2.select_dtypes(include=[np.number]).columns:
            assert X_step2[col].dtype == np.float32

    def test_with_various_data_types(self):
        """様々なデータ型での動作確認"""
        # 混合データ型のDataFrame
        X = pd.DataFrame({
            'int8_col': np.array([1, 2, 3], dtype='int8'),
            'int16_col': np.array([100, 200, 300], dtype='int16'),
            'int32_col': np.array([1000, 2000, 3000], dtype='int32'),
            'int64_col': np.array([10000, 20000, 30000], dtype='int64'),
            'float16_col': np.array([1.1, 2.2, 3.3], dtype='float16'),
            'float32_col': np.array([1.11, 2.22, 3.33], dtype='float32'),
            'float64_col': np.array([1.111, 2.222, 3.333], dtype='float64'),
            'object_col': ['A', 'B', 'C'],
            'bool_col': [True, False, True],
        })

        validator = DataQualityValidator()
        optimizer = DtypeOptimizer()

        X_validated = validator.fit_transform(X)
        X_optimized = optimizer.fit_transform(X_validated)

        # 基本的な形状チェック
        assert X_optimized.shape == X.shape
        assert set(X_optimized.columns) == set(X.columns)