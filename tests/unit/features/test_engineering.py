"""engineering.py の単体テスト"""

import pytest
import pandas as pd
import numpy as np

from src.mlops.features.engineering import (
    RatioFeatureGenerator,
    InteractionFeatureGenerator,
    AggregateFeatureGenerator,
    PolynomialFeatureGenerator
)
from tests.fixtures.test_data import create_sample_dataframe, create_ecommerce_sample_data
from tests.fixtures.test_utils import (
    assert_transformer_basic_contract,
    create_edge_case_dataframes
)


class TestRatioFeatureGenerator:
    """RatioFeatureGenerator の単体テスト"""

    def test_basic_functionality(self):
        """基本機能テスト"""
        X = pd.DataFrame({
            'numerator_1': [10, 20, 30],
            'denominator_1': [2, 4, 6],
            'numerator_2': [100, 200, 300],
            'denominator_2': [10, 20, 30],
            'other_col': [1, 2, 3]
        })

        ratio_pairs = [
            ('numerator_1', 'denominator_1', 'ratio_1'),
            ('numerator_2', 'denominator_2', 'ratio_2')
        ]

        generator = RatioFeatureGenerator(ratio_pairs=ratio_pairs)

        # 基本契約のテスト
        assert_transformer_basic_contract(generator, X)

    def test_ratio_calculation(self):
        """比率計算の正確性テスト"""
        X = pd.DataFrame({
            'num_col': [10, 20, 30],
            'den_col': [2, 4, 6],
            'other_col': [1, 2, 3]
        })

        ratio_pairs = [('num_col', 'den_col', 'calculated_ratio')]
        generator = RatioFeatureGenerator(ratio_pairs=ratio_pairs)

        X_transformed = generator.fit_transform(X)

        # 比率が正しく計算されているか
        expected_ratios = [5.0, 5.0, 5.0]  # 10/2, 20/4, 30/6
        assert 'calculated_ratio' in X_transformed.columns
        np.testing.assert_array_equal(X_transformed['calculated_ratio'].values, expected_ratios)

        # 元のカラムも保持されているか
        assert 'num_col' in X_transformed.columns
        assert 'den_col' in X_transformed.columns
        assert 'other_col' in X_transformed.columns

    def test_zero_division_handling(self):
        """ゼロ除算処理テスト"""
        X = pd.DataFrame({
            'numerator': [10, 20, 30],
            'denominator': [2, 0, 6],  # ゼロを含む
            'safe_col': [1, 2, 3]
        })

        ratio_pairs = [('numerator', 'denominator', 'safe_ratio')]
        generator = RatioFeatureGenerator(ratio_pairs=ratio_pairs)

        X_transformed = generator.fit_transform(X)

        # ゼロ除算は0になるか確認
        expected_ratios = [5.0, 0.0, 5.0]  # 10/2, 20/0->0, 30/6
        assert 'safe_ratio' in X_transformed.columns
        np.testing.assert_array_equal(X_transformed['safe_ratio'].values, expected_ratios)

    def test_nonexistent_columns(self):
        """存在しないカラム指定テスト"""
        X = pd.DataFrame({
            'existing_num': [10, 20, 30],
            'existing_den': [2, 4, 6]
        })

        # 存在しないカラムを指定
        ratio_pairs = [
            ('existing_num', 'existing_den', 'valid_ratio'),
            ('nonexistent_num', 'existing_den', 'invalid_ratio'),
            ('existing_num', 'nonexistent_den', 'invalid_ratio_2')
        ]

        generator = RatioFeatureGenerator(ratio_pairs=ratio_pairs)
        X_transformed = generator.fit_transform(X)

        # 有効なペアのみ処理されているか
        assert 'valid_ratio' in X_transformed.columns
        assert 'invalid_ratio' not in X_transformed.columns
        assert 'invalid_ratio_2' not in X_transformed.columns

    def test_empty_ratio_pairs(self):
        """空の比率ペアテスト"""
        X = create_sample_dataframe(n_rows=50, n_features=4)
        generator = RatioFeatureGenerator(ratio_pairs=[])

        X_transformed = generator.fit_transform(X)

        # 何も変更されないか
        assert X_transformed.shape == X.shape
        assert list(X_transformed.columns) == list(X.columns)

    def test_get_feature_names_out(self):
        """特徴量名出力テスト"""
        X = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [2, 4, 6],
            'c': [3, 6, 9]
        })

        ratio_pairs = [('a', 'b', 'a_div_b'), ('a', 'c', 'a_div_c')]
        generator = RatioFeatureGenerator(ratio_pairs=ratio_pairs)

        generator.fit(X)
        feature_names = generator.get_feature_names_out()

        expected_names = ['a', 'b', 'c', 'a_div_b', 'a_div_c']
        assert len(feature_names) == 5
        assert set(feature_names) == set(expected_names)


class TestInteractionFeatureGenerator:
    """InteractionFeatureGenerator の単体テスト"""

    def test_basic_functionality(self):
        """基本機能テスト"""
        X = pd.DataFrame({
            'feature_1': [2, 4, 6],
            'feature_2': [3, 5, 7],
            'feature_3': [1, 2, 3],
            'other_col': [10, 20, 30]
        })

        interaction_pairs = [
            ('feature_1', 'feature_2', 'interaction_1_2'),
            ('feature_1', 'feature_3', 'interaction_1_3')
        ]

        generator = InteractionFeatureGenerator(interaction_pairs=interaction_pairs)

        # 基本契約のテスト
        assert_transformer_basic_contract(generator, X)

    def test_interaction_calculation(self):
        """交互作用計算の正確性テスト"""
        X = pd.DataFrame({
            'col_a': [2, 3, 4],
            'col_b': [5, 6, 7],
            'other_col': [1, 2, 3]
        })

        interaction_pairs = [('col_a', 'col_b', 'a_times_b')]
        generator = InteractionFeatureGenerator(interaction_pairs=interaction_pairs)

        X_transformed = generator.fit_transform(X)

        # 交互作用が正しく計算されているか
        expected_interactions = [10, 18, 28]  # 2*5, 3*6, 4*7
        assert 'a_times_b' in X_transformed.columns
        np.testing.assert_array_equal(X_transformed['a_times_b'].values, expected_interactions)

    def test_negative_values_interaction(self):
        """負値を含む交互作用テスト"""
        X = pd.DataFrame({
            'positive': [2, 3, 4],
            'negative': [-1, -2, -3],
            'mixed': [-1, 2, -3]
        })

        interaction_pairs = [
            ('positive', 'negative', 'pos_neg'),
            ('positive', 'mixed', 'pos_mixed'),
            ('negative', 'mixed', 'neg_mixed')
        ]

        generator = InteractionFeatureGenerator(interaction_pairs=interaction_pairs)
        X_transformed = generator.fit_transform(X)

        # 正負を含む計算が正しいか
        expected_pos_neg = [-2, -6, -12]  # 2*(-1), 3*(-2), 4*(-3)
        expected_pos_mixed = [-2, 6, -12]  # 2*(-1), 3*2, 4*(-3)
        expected_neg_mixed = [1, -4, 9]  # (-1)*(-1), (-2)*2, (-3)*(-3)

        np.testing.assert_array_equal(X_transformed['pos_neg'].values, expected_pos_neg)
        np.testing.assert_array_equal(X_transformed['pos_mixed'].values, expected_pos_mixed)
        np.testing.assert_array_equal(X_transformed['neg_mixed'].values, expected_neg_mixed)

    def test_nonexistent_columns(self):
        """存在しないカラム指定テスト"""
        X = pd.DataFrame({
            'exists_1': [1, 2, 3],
            'exists_2': [4, 5, 6]
        })

        interaction_pairs = [
            ('exists_1', 'exists_2', 'valid_interaction'),
            ('exists_1', 'nonexistent', 'invalid_interaction'),
            ('nonexistent_1', 'nonexistent_2', 'invalid_interaction_2')
        ]

        generator = InteractionFeatureGenerator(interaction_pairs=interaction_pairs)
        X_transformed = generator.fit_transform(X)

        # 有効なペアのみ処理されているか
        assert 'valid_interaction' in X_transformed.columns
        assert 'invalid_interaction' not in X_transformed.columns
        assert 'invalid_interaction_2' not in X_transformed.columns

    def test_empty_interaction_pairs(self):
        """空の交互作用ペアテスト"""
        X = create_sample_dataframe(n_rows=50, n_features=4)
        generator = InteractionFeatureGenerator(interaction_pairs=[])

        X_transformed = generator.fit_transform(X)

        # 何も変更されないか
        assert X_transformed.shape == X.shape
        assert list(X_transformed.columns) == list(X.columns)


class TestAggregateFeatureGenerator:
    """AggregateFeatureGenerator の単体テスト"""

    def test_basic_functionality(self):
        """基本機能テスト"""
        X = pd.DataFrame({
            'group1_col1': [1, 2, 3],
            'group1_col2': [4, 5, 6],
            'group2_col1': [7, 8, 9],
            'other_col': [10, 11, 12]
        })

        feature_groups = {
            'group1': ['group1_col1', 'group1_col2'],
            'group2': ['group2_col1']
        }

        generator = AggregateFeatureGenerator(feature_groups=feature_groups)

        # 基本契約のテスト
        assert_transformer_basic_contract(generator, X)

    def test_aggregate_calculations(self):
        """集約計算の正確性テスト"""
        X = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })

        feature_groups = {'group_ab': ['a', 'b']}
        generator = AggregateFeatureGenerator(feature_groups=feature_groups)

        X_transformed = generator.fit_transform(X)

        # 集約統計が正しく計算されているか
        expected_sum = [5, 7, 9]  # 1+4, 2+5, 3+6
        expected_mean = [2.5, 3.5, 4.5]  # (1+4)/2, (2+5)/2, (3+6)/2
        expected_std = [2.121, 2.121, 2.121]  # np.std([1,4]), np.std([2,5]), np.std([3,6])

        assert 'group_ab_sum' in X_transformed.columns
        assert 'group_ab_mean' in X_transformed.columns
        assert 'group_ab_std' in X_transformed.columns

        np.testing.assert_array_equal(X_transformed['group_ab_sum'].values, expected_sum)
        np.testing.assert_array_equal(X_transformed['group_ab_mean'].values, expected_mean)
        np.testing.assert_array_almost_equal(X_transformed['group_ab_std'].values, expected_std, decimal=3)

    def test_partial_missing_columns(self):
        """一部カラムが存在しない場合のテスト"""
        X = pd.DataFrame({
            'exists_1': [1, 2, 3],
            'exists_2': [4, 5, 6],
            'other_col': [7, 8, 9]
        })

        feature_groups = {
            'mixed_group': ['exists_1', 'exists_2', 'nonexistent'],
            'missing_group': ['nonexistent_1', 'nonexistent_2']
        }

        generator = AggregateFeatureGenerator(feature_groups=feature_groups)
        X_transformed = generator.fit_transform(X)

        # 存在するカラムのみで集約されているか
        assert 'mixed_group_sum' in X_transformed.columns  # exists_1 + exists_2
        assert 'missing_group_sum' not in X_transformed.columns  # 全て存在しない

    def test_single_column_groups(self):
        """単一カラムグループのテスト"""
        X = pd.DataFrame({
            'single_col': [10, 20, 30],
            'other_col': [1, 2, 3]
        })

        feature_groups = {'single_group': ['single_col']}
        generator = AggregateFeatureGenerator(feature_groups=feature_groups)

        X_transformed = generator.fit_transform(X)

        # 単一カラムでも統計が計算されているか
        assert 'single_group_sum' in X_transformed.columns
        assert 'single_group_mean' in X_transformed.columns
        assert 'single_group_std' in X_transformed.columns

        # 単一カラムの場合、std は 0 になるはず
        np.testing.assert_array_equal(X_transformed['single_group_sum'].values, [10, 20, 30])
        np.testing.assert_array_equal(X_transformed['single_group_mean'].values, [10, 20, 30])
        np.testing.assert_array_equal(X_transformed['single_group_std'].values, [0, 0, 0])

    def test_empty_feature_groups(self):
        """空の特徴量グループテスト"""
        X = create_sample_dataframe(n_rows=50, n_features=4)
        generator = AggregateFeatureGenerator(feature_groups={})

        X_transformed = generator.fit_transform(X)

        # 何も変更されないか
        assert X_transformed.shape == X.shape
        assert list(X_transformed.columns) == list(X.columns)


class TestPolynomialFeatureGenerator:
    """PolynomialFeatureGenerator の単体テスト"""

    def test_basic_functionality(self):
        """基本機能テスト"""
        X = pd.DataFrame({
            'target_col': [2, 3, 4],
            'other_col': [1, 2, 3]
        })

        generator = PolynomialFeatureGenerator(
            target_columns=['target_col'],
            degree=3
        )

        # 基本契約のテスト
        assert_transformer_basic_contract(generator, X)

    def test_polynomial_calculation(self):
        """多項式計算の正確性テスト"""
        X = pd.DataFrame({
            'base_col': [2, 3, 4],
            'other_col': [10, 20, 30]
        })

        generator = PolynomialFeatureGenerator(
            target_columns=['base_col'],
            degree=3
        )

        X_transformed = generator.fit_transform(X)

        # 多項式特徴量が正しく計算されているか
        expected_pow2 = [4, 9, 16]  # 2^2, 3^2, 4^2
        expected_pow3 = [8, 27, 64]  # 2^3, 3^3, 4^3

        assert 'base_col_pow2' in X_transformed.columns
        assert 'base_col_pow3' in X_transformed.columns

        np.testing.assert_array_equal(X_transformed['base_col_pow2'].values, expected_pow2)
        np.testing.assert_array_equal(X_transformed['base_col_pow3'].values, expected_pow3)

        # 元のカラムは保持されているか
        assert 'base_col' in X_transformed.columns
        assert 'other_col' in X_transformed.columns

    def test_multiple_target_columns(self):
        """複数ターゲットカラムテスト"""
        X = pd.DataFrame({
            'col_a': [2, 3],
            'col_b': [4, 5],
            'col_c': [6, 7]
        })

        generator = PolynomialFeatureGenerator(
            target_columns=['col_a', 'col_b'],
            degree=2
        )

        X_transformed = generator.fit_transform(X)

        # 複数カラムの多項式特徴量が生成されているか
        assert 'col_a_pow2' in X_transformed.columns
        assert 'col_b_pow2' in X_transformed.columns
        assert 'col_c_pow2' not in X_transformed.columns  # 対象外

        # 値が正しいか
        np.testing.assert_array_equal(X_transformed['col_a_pow2'].values, [4, 9])
        np.testing.assert_array_equal(X_transformed['col_b_pow2'].values, [16, 25])

    def test_degree_variations(self):
        """次数バリエーションテスト"""
        X = pd.DataFrame({'col': [2, 3]})

        # degree=1 (実際にはpow1は生成されない、2から始まる)
        generator1 = PolynomialFeatureGenerator(target_columns=['col'], degree=1)
        X_transformed1 = generator1.fit_transform(X)
        assert X_transformed1.shape[1] == 1  # 元のカラムのみ

        # degree=2
        generator2 = PolynomialFeatureGenerator(target_columns=['col'], degree=2)
        X_transformed2 = generator2.fit_transform(X)
        assert 'col_pow2' in X_transformed2.columns

        # degree=4
        generator4 = PolynomialFeatureGenerator(target_columns=['col'], degree=4)
        X_transformed4 = generator4.fit_transform(X)
        assert 'col_pow2' in X_transformed4.columns
        assert 'col_pow3' in X_transformed4.columns
        assert 'col_pow4' in X_transformed4.columns

    def test_nonexistent_columns(self):
        """存在しないカラム指定テスト"""
        X = pd.DataFrame({
            'existing_col': [2, 3, 4]
        })

        generator = PolynomialFeatureGenerator(
            target_columns=['existing_col', 'nonexistent_col'],
            degree=2
        )

        X_transformed = generator.fit_transform(X)

        # 存在するカラムのみ処理されているか
        assert 'existing_col_pow2' in X_transformed.columns
        assert 'nonexistent_col_pow2' not in X_transformed.columns

    def test_negative_values(self):
        """負値を含む多項式テスト"""
        X = pd.DataFrame({
            'mixed_col': [-2, 3, -4]
        })

        generator = PolynomialFeatureGenerator(
            target_columns=['mixed_col'],
            degree=3
        )

        X_transformed = generator.fit_transform(X)

        # 負値の多項式が正しく計算されているか
        expected_pow2 = [4, 9, 16]  # (-2)^2, 3^2, (-4)^2
        expected_pow3 = [-8, 27, -64]  # (-2)^3, 3^3, (-4)^3

        np.testing.assert_array_equal(X_transformed['mixed_col_pow2'].values, expected_pow2)
        np.testing.assert_array_equal(X_transformed['mixed_col_pow3'].values, expected_pow3)

    def test_empty_target_columns(self):
        """空のターゲットカラムリストテスト"""
        X = create_sample_dataframe(n_rows=50, n_features=4)
        generator = PolynomialFeatureGenerator(target_columns=[], degree=2)

        X_transformed = generator.fit_transform(X)

        # 何も変更されないか
        assert X_transformed.shape == X.shape
        assert list(X_transformed.columns) == list(X.columns)


class TestEngineeringIntegration:
    """特徴量生成クラス統合テスト"""

    def test_sequential_feature_generation(self):
        """連続的な特徴量生成テスト"""
        X = pd.DataFrame({
            'amount': [100, 200, 300],
            'quantity': [2, 4, 6],
            'price': [50, 50, 50],
            'tax_rate': [0.1, 0.1, 0.1]
        })

        # 段階的に特徴量生成
        # 1. 比率特徴量
        ratio_gen = RatioFeatureGenerator(ratio_pairs=[
            ('amount', 'quantity', 'unit_price')
        ])
        X_step1 = ratio_gen.fit_transform(X)

        # 2. 交互作用特徴量
        interaction_gen = InteractionFeatureGenerator(interaction_pairs=[
            ('amount', 'tax_rate', 'total_tax')
        ])
        X_step2 = interaction_gen.fit_transform(X_step1)

        # 3. 集約特徴量
        agg_gen = AggregateFeatureGenerator(feature_groups={
            'financial': ['amount', 'total_tax', 'unit_price']
        })
        X_step3 = agg_gen.fit_transform(X_step2)

        # 4. 多項式特徴量
        poly_gen = PolynomialFeatureGenerator(
            target_columns=['amount'],
            degree=2
        )
        X_final = poly_gen.fit_transform(X_step3)

        # 段階的に特徴量が増加しているか確認
        assert X_final.shape[0] == X.shape[0]  # 行数は保持
        assert X_final.shape[1] > X.shape[1]  # 特徴量数は増加

        # 生成された特徴量が存在するか
        expected_features = [
            'unit_price', 'total_tax',
            'financial_sum', 'financial_mean', 'financial_std',
            'amount_pow2'
        ]
        for feature in expected_features:
            assert feature in X_final.columns

    def test_ecommerce_realistic_scenario(self):
        """Eコマース現実的シナリオテスト"""
        X = create_ecommerce_sample_data(n_rows=100)

        # 実用的な特徴量生成パイプライン
        ratio_gen = RatioFeatureGenerator(ratio_pairs=[
            ('amount', 'customer_age', 'amount_per_age')
        ])

        poly_gen = PolynomialFeatureGenerator(
            target_columns=['amount', 'customer_age'],
            degree=2
        )

        agg_gen = AggregateFeatureGenerator(feature_groups={
            'customer_profile': ['customer_age', 'amount']
        })

        # 連続適用
        X_ratios = ratio_gen.fit_transform(X)
        X_poly = poly_gen.fit_transform(X_ratios)
        X_final = agg_gen.fit_transform(X_poly)

        # 元の特徴量 + 生成特徴量が含まれているか
        assert X_final.shape[0] == X.shape[0]
        assert X_final.shape[1] > X.shape[1]

        # 数値以外のカラムは保持されているか
        assert 'product_category' in X_final.columns
        assert 'payment_method' in X_final.columns