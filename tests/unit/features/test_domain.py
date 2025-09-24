"""domain.py の単体テスト"""

import pytest
import pandas as pd
import numpy as np

from src.mlops.features.domain import EcommerceFeatureGenerator
from tests.fixtures.test_data import create_ecommerce_sample_data
from tests.fixtures.test_utils import (
    assert_transformer_basic_contract,
    create_edge_case_dataframes
)


class TestEcommerceFeatureGenerator:
    """EcommerceFeatureGenerator の単体テスト"""

    def test_basic_functionality(self):
        """基本機能テスト"""
        X = create_ecommerce_sample_data(n_rows=100)
        generator = EcommerceFeatureGenerator()

        # 基本契約のテスト
        assert_transformer_basic_contract(generator, X)

    def test_amount_log_transformation(self):
        """金額対数変換テスト"""
        X = pd.DataFrame({
            'amount': [1, 10, 100, 1000],
            'other_col': [1, 2, 3, 4]
        })

        generator = EcommerceFeatureGenerator()
        X_transformed = generator.fit_transform(X)

        # amount_log が追加されているか
        assert 'amount_log' in X_transformed.columns

        # log1p(amount) が正しく計算されているか
        expected_log = np.log1p([1, 10, 100, 1000])
        np.testing.assert_array_almost_equal(
            X_transformed['amount_log'].values,
            expected_log,
            decimal=6
        )

        # 元のカラムが保持されているか
        assert 'amount' in X_transformed.columns
        assert 'other_col' in X_transformed.columns

    def test_transaction_time_night_flag(self):
        """深夜取引フラグテスト"""
        X = pd.DataFrame({
            'transaction_time': [0, 6, 7, 21, 22, 23],  # 深夜: 0,6,22,23 昼間: 7,21
            'other_col': [1, 2, 3, 4, 5, 6]
        })

        generator = EcommerceFeatureGenerator()
        X_transformed = generator.fit_transform(X)

        # is_night_transaction が追加されているか
        assert 'is_night_transaction' in X_transformed.columns

        # 深夜時間帯が正しくフラグされているか
        expected_flags = [1, 1, 0, 0, 1, 1]  # 22時以降 or 6時以前 = 1
        np.testing.assert_array_equal(
            X_transformed['is_night_transaction'].values,
            expected_flags
        )

        # フラグは整数型か
        assert X_transformed['is_night_transaction'].dtype == int

    def test_both_features_together(self):
        """両方の特徴量が同時に生成されるテスト"""
        X = pd.DataFrame({
            'amount': [50, 100, 200],
            'transaction_time': [2, 12, 23],  # 深夜, 昼間, 深夜
            'customer_id': ['A', 'B', 'C']
        })

        generator = EcommerceFeatureGenerator()
        X_transformed = generator.fit_transform(X)

        # 両方の新特徴量が追加されているか
        assert 'amount_log' in X_transformed.columns
        assert 'is_night_transaction' in X_transformed.columns

        # 値が正しく設定されているか
        expected_log = np.log1p([50, 100, 200])
        expected_night_flags = [1, 0, 1]

        np.testing.assert_array_almost_equal(X_transformed['amount_log'].values, expected_log)
        np.testing.assert_array_equal(X_transformed['is_night_transaction'].values, expected_night_flags)

        # 元のカラムが保持されているか
        assert 'amount' in X_transformed.columns
        assert 'transaction_time' in X_transformed.columns
        assert 'customer_id' in X_transformed.columns

    def test_missing_target_columns(self):
        """対象カラムが存在しない場合のテスト"""
        # amountカラムなし
        X_no_amount = pd.DataFrame({
            'transaction_time': [10, 15, 20],
            'other_col': [1, 2, 3]
        })

        generator = EcommerceFeatureGenerator()
        X_transformed = generator.fit_transform(X_no_amount)

        # amount_log は追加されないが、is_night_transaction は追加される
        assert 'amount_log' not in X_transformed.columns
        assert 'is_night_transaction' in X_transformed.columns

        # transaction_timeカラムなし
        X_no_time = pd.DataFrame({
            'amount': [100, 200, 300],
            'other_col': [1, 2, 3]
        })

        X_transformed2 = generator.fit_transform(X_no_time)

        # amount_log は追加されるが、is_night_transaction は追加されない
        assert 'amount_log' in X_transformed2.columns
        assert 'is_night_transaction' not in X_transformed2.columns

    def test_both_target_columns_missing(self):
        """両方の対象カラムが存在しない場合のテスト"""
        X = pd.DataFrame({
            'unrelated_col_1': [1, 2, 3],
            'unrelated_col_2': ['A', 'B', 'C']
        })

        generator = EcommerceFeatureGenerator()
        X_transformed = generator.fit_transform(X)

        # 何も変更されないか
        assert X_transformed.shape == X.shape
        assert list(X_transformed.columns) == list(X.columns)
        assert 'amount_log' not in X_transformed.columns
        assert 'is_night_transaction' not in X_transformed.columns

    def test_boundary_transaction_times(self):
        """境界値の取引時間テスト"""
        X = pd.DataFrame({
            'transaction_time': [6, 7, 21, 22],  # 境界値を含む
            'amount': [100, 200, 300, 400]
        })

        generator = EcommerceFeatureGenerator()
        X_transformed = generator.fit_transform(X)

        # 境界値での動作確認
        # 6以下 or 22以上が深夜
        expected_flags = [1, 0, 0, 1]  # 6:深夜, 7:昼, 21:昼, 22:深夜
        np.testing.assert_array_equal(
            X_transformed['is_night_transaction'].values,
            expected_flags
        )

    def test_extreme_transaction_times(self):
        """極端な取引時間値のテスト"""
        X = pd.DataFrame({
            'transaction_time': [-1, 24, 25, 100],  # 範囲外の値
            'amount': [100, 200, 300, 400]
        })

        generator = EcommerceFeatureGenerator()

        # 範囲外の値でもエラーにならないか
        try:
            X_transformed = generator.fit_transform(X)
            # フラグは計算されるが、値は期待と異なる可能性がある（実装依存）
            assert 'is_night_transaction' in X_transformed.columns
        except Exception as e:
            pytest.fail(f"極端な値でエラーが発生: {e}")

    def test_zero_and_negative_amounts(self):
        """ゼロ・負の金額値のテスト"""
        X = pd.DataFrame({
            'amount': [0, -10, -100, 0.1],
            'transaction_time': [10, 11, 12, 13]
        })

        generator = EcommerceFeatureGenerator()
        X_transformed = generator.fit_transform(X)

        # log1p は負値でも計算されるが、ゼロやマイナスの場合の処理を確認
        assert 'amount_log' in X_transformed.columns

        # log1p(0) = 0, log1p(-10) = log(1-10) = log(-9) -> warning/error の可能性
        # 実装では np.log1p を使用しているので、負値では警告が出る可能性がある

    def test_get_feature_names_out(self):
        """特徴量名出力テスト"""
        X = pd.DataFrame({
            'amount': [100, 200, 300],
            'transaction_time': [10, 15, 20],
            'other_col': [1, 2, 3]
        })

        generator = EcommerceFeatureGenerator()
        generator.fit(X)

        feature_names = generator.get_feature_names_out()

        # 元の特徴量 + 生成特徴量
        expected_names = [
            'amount', 'transaction_time', 'other_col',
            'amount_log', 'is_night_transaction'
        ]

        assert len(feature_names) == 5
        assert set(feature_names) == set(expected_names)

    def test_get_feature_names_out_partial(self):
        """部分的な特徴量名出力テスト"""
        # amountのみ存在
        X_partial = pd.DataFrame({
            'amount': [100, 200],
            'other_col': [1, 2]
        })

        generator = EcommerceFeatureGenerator()
        generator.fit(X_partial)

        feature_names = generator.get_feature_names_out()

        # amount関連の特徴量のみ追加
        expected_names = ['amount', 'other_col', 'amount_log']

        assert len(feature_names) == 3
        assert set(feature_names) == set(expected_names)

    def test_edge_cases(self):
        """エッジケーステスト"""
        edge_cases = create_edge_case_dataframes()
        generator = EcommerceFeatureGenerator()

        for i, X in enumerate(edge_cases):
            if X.empty:
                # 空DataFrameの場合
                X_transformed = generator.fit_transform(X)
                assert X_transformed.empty
                continue

            try:
                X_transformed = generator.fit_transform(X)
                # 基本的なshape確認
                assert X_transformed.shape[0] == X.shape[0]
            except Exception as e:
                pytest.fail(f"エッジケース {i} でエラー: {e}")

    def test_realistic_ecommerce_data(self):
        """現実的なEコマースデータテスト"""
        X = create_ecommerce_sample_data(n_rows=200)
        generator = EcommerceFeatureGenerator()

        X_transformed = generator.fit_transform(X)

        # 現実的なデータでの動作確認
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] > X.shape[1]  # 特徴量が増加

        # 生成された特徴量の基本統計確認
        if 'amount_log' in X_transformed.columns:
            # 対数変換された値は元の値より小さくなるはず（amount > 1の場合）
            amount_log_values = X_transformed['amount_log'].values
            assert not np.any(np.isnan(amount_log_values)), "対数変換でNaNが生成されました"

        if 'is_night_transaction' in X_transformed.columns:
            # フラグは0または1のみ
            night_flags = X_transformed['is_night_transaction'].values
            assert set(night_flags).issubset({0, 1}), "夜間フラグが0/1以外の値を含んでいます"

    def test_numpy_array_input(self):
        """numpy array入力テスト"""
        # numpy arrayでは列名がないため、基本的に何も処理されない
        X_array = np.random.randn(50, 5)
        generator = EcommerceFeatureGenerator()

        X_transformed = generator.fit_transform(X_array)

        # numpy arrayの場合は変更されない（列名がないため）
        assert X_transformed.shape == X_array.shape
        np.testing.assert_array_equal(X_transformed, X_array)