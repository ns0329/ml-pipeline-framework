"""pipeline_utils.py の単体テスト"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from src.mlops.utils.pipeline_utils import (
    get_pipeline_feature_names,
    get_transformed_data_with_feature_names
)
from tests.fixtures.test_data import create_binary_classification_data


class TestGetPipelineFeatureNames:
    """get_pipeline_feature_names 関数の単体テスト"""

    def test_basic_feature_name_tracking(self):
        """基本的な特徴量名追跡テスト"""
        # Mock pipeline設定
        mock_transformer1 = Mock()
        mock_transformer1.get_feature_names_out.return_value = ['scaled_feature_0', 'scaled_feature_1']
        type(mock_transformer1).__name__ = "StandardScaler"

        mock_transformer2 = Mock()
        mock_transformer2.get_feature_names_out.return_value = ['pca_0', 'pca_1']
        type(mock_transformer2).__name__ = "PCA"

        mock_classifier = Mock()

        mock_pipeline = Mock()
        mock_pipeline.steps = [
            ("scaler", mock_transformer1),
            ("pca", mock_transformer2),
            ("classifier", mock_classifier)
        ]

        original_feature_names = ['feature_0', 'feature_1']
        result = get_pipeline_feature_names(mock_pipeline, original_feature_names)

        expected = ['pca_0', 'pca_1']
        assert result == expected

        # 呼び出し順序を確認
        mock_transformer1.get_feature_names_out.assert_called_once_with(['feature_0', 'feature_1'])
        mock_transformer2.get_feature_names_out.assert_called_once_with(['scaled_feature_0', 'scaled_feature_1'])

    def test_feature_names_with_sampling_step(self):
        """サンプリングステップ含む特徴量名追跡テスト"""
        # Mock設定
        mock_sampler = Mock()
        type(mock_sampler).__name__ = "SMOTE"

        mock_scaler = Mock()
        mock_scaler.get_feature_names_out.return_value = ['scaled_feature_0', 'scaled_feature_1', 'scaled_feature_2']
        type(mock_scaler).__name__ = "StandardScaler"

        mock_classifier = Mock()

        mock_pipeline = Mock()
        mock_pipeline.steps = [
            ("sampler", mock_sampler),
            ("scaler", mock_scaler),
            ("classifier", mock_classifier)
        ]

        original_feature_names = ['feature_0', 'feature_1', 'feature_2']
        result = get_pipeline_feature_names(mock_pipeline, original_feature_names)

        # サンプラーはスキップされ、スケーラーのみ適用される
        expected = ['scaled_feature_0', 'scaled_feature_1', 'scaled_feature_2']
        assert result == expected

        # サンプラーのget_feature_names_outは呼び出されない
        assert not hasattr(mock_sampler, 'get_feature_names_out') or not mock_sampler.get_feature_names_out.called
        mock_scaler.get_feature_names_out.assert_called_once_with(['feature_0', 'feature_1', 'feature_2'])

    def test_transformer_without_get_feature_names_out(self):
        """get_feature_names_outメソッドがないtransformerのテスト"""
        # Mock設定
        mock_transformer_no_method = Mock(spec=[])  # get_feature_names_outメソッドなし
        mock_transformer_no_method.feature_names_in_ = ['feature_0', 'feature_1']
        type(mock_transformer_no_method).__name__ = "SimpleImputer"

        mock_transformer_with_method = Mock()
        mock_transformer_with_method.get_feature_names_out.return_value = ['transformed_feature_0', 'transformed_feature_1']
        type(mock_transformer_with_method).__name__ = "StandardScaler"

        mock_classifier = Mock()

        mock_pipeline = Mock()
        mock_pipeline.steps = [
            ("imputer", mock_transformer_no_method),
            ("scaler", mock_transformer_with_method),
            ("classifier", mock_classifier)
        ]

        original_feature_names = ['feature_0', 'feature_1']
        result = get_pipeline_feature_names(mock_pipeline, original_feature_names)

        # imputer（メソッドなし）はスキップ、scalerのみ適用
        expected = ['transformed_feature_0', 'transformed_feature_1']
        assert result == expected

    def test_transformer_get_feature_names_out_error(self):
        """get_feature_names_outでエラーが発生する場合のテスト"""
        # Mock設定
        mock_transformer_error = Mock()
        mock_transformer_error.get_feature_names_out.side_effect = Exception("Feature names error")
        type(mock_transformer_error).__name__ = "ProblematicTransformer"

        mock_transformer_success = Mock()
        mock_transformer_success.get_feature_names_out.return_value = ['final_feature_0', 'final_feature_1']
        type(mock_transformer_success).__name__ = "StandardScaler"

        mock_classifier = Mock()

        mock_pipeline = Mock()
        mock_pipeline.steps = [
            ("error_transformer", mock_transformer_error),
            ("success_transformer", mock_transformer_success),
            ("classifier", mock_classifier)
        ]

        original_feature_names = ['feature_0', 'feature_1']

        with patch('builtins.print') as mock_print:
            result = get_pipeline_feature_names(mock_pipeline, original_feature_names)

        # エラーが発生したtransformerはスキップ、成功したtransformerのみ適用
        expected = ['final_feature_0', 'final_feature_1']
        assert result == expected

        # エラーメッセージが出力されることを確認
        mock_print.assert_called_once()
        assert "error_transformer" in mock_print.call_args[0][0]
        assert "Feature names error" in mock_print.call_args[0][0]

    def test_multiple_sampling_steps(self):
        """複数のサンプリングステップのテスト"""
        # Mock設定
        mock_over_sampler = Mock()
        type(mock_over_sampler).__name__ = "RandomOverSampler"

        mock_under_sampler = Mock()
        type(mock_under_sampler).__name__ = "RandomUnderSampler"

        mock_scaler = Mock()
        mock_scaler.get_feature_names_out.return_value = ['scaled_0', 'scaled_1']
        type(mock_scaler).__name__ = "StandardScaler"

        mock_classifier = Mock()

        mock_pipeline = Mock()
        mock_pipeline.steps = [
            ("over_sampler", mock_over_sampler),
            ("under_sampler", mock_under_sampler),
            ("scaler", mock_scaler),
            ("classifier", mock_classifier)
        ]

        original_feature_names = ['feature_0', 'feature_1']
        result = get_pipeline_feature_names(mock_pipeline, original_feature_names)

        # 両方のサンプラーはスキップされ、スケーラーのみ適用
        expected = ['scaled_0', 'scaled_1']
        assert result == expected
        mock_scaler.get_feature_names_out.assert_called_once_with(['feature_0', 'feature_1'])

    def test_pipeline_with_only_classifier(self):
        """分類器のみのパイプラインテスト"""
        # Mock設定
        mock_classifier = Mock()

        mock_pipeline = Mock()
        mock_pipeline.steps = [("classifier", mock_classifier)]

        original_feature_names = ['feature_0', 'feature_1', 'feature_2']
        result = get_pipeline_feature_names(mock_pipeline, original_feature_names)

        # 変換ステップがないため、元の特徴量名が返される
        expected = ['feature_0', 'feature_1', 'feature_2']
        assert result == expected

    def test_feature_names_with_array_output(self):
        """numpy arrayを返すget_feature_names_outのテスト"""
        # Mock設定
        mock_transformer = Mock()
        mock_transformer.get_feature_names_out.return_value = np.array(['array_feature_0', 'array_feature_1'])
        type(mock_transformer).__name__ = "CustomTransformer"

        mock_classifier = Mock()

        mock_pipeline = Mock()
        mock_pipeline.steps = [
            ("transformer", mock_transformer),
            ("classifier", mock_classifier)
        ]

        original_feature_names = ['feature_0', 'feature_1']
        result = get_pipeline_feature_names(mock_pipeline, original_feature_names)

        # numpy arrayがlistに変換されることを確認
        expected = ['array_feature_0', 'array_feature_1']
        assert result == expected
        assert isinstance(result, list)

    def test_unknown_transformer_handling(self):
        """不明なtransformerの処理テスト"""
        # Mock設定
        mock_unknown_transformer = Mock(spec=[])  # get_feature_names_outもfeature_names_in_もなし
        type(mock_unknown_transformer).__name__ = "UnknownTransformer"

        mock_classifier = Mock()

        mock_pipeline = Mock()
        mock_pipeline.steps = [
            ("unknown", mock_unknown_transformer),
            ("classifier", mock_classifier)
        ]

        original_feature_names = ['feature_0', 'feature_1']

        with patch('builtins.print') as mock_print:
            result = get_pipeline_feature_names(mock_pipeline, original_feature_names)

        # 不明なtransformerはスキップされ、元の特徴量名が返される
        expected = ['feature_0', 'feature_1']
        assert result == expected

        # 警告メッセージが出力されることを確認
        mock_print.assert_called_once()
        assert "unknown" in mock_print.call_args[0][0]
        assert "特徴量名の取得方法が不明" in mock_print.call_args[0][0]


class TestGetTransformedDataWithFeatureNames:
    """get_transformed_data_with_feature_names 関数の単体テスト"""

    def test_basic_data_transformation(self):
        """基本的なデータ変換テスト"""
        # Mock pipeline設定
        X_original = pd.DataFrame({
            'feature_0': [1, 2, 3],
            'feature_1': [4, 5, 6]
        })

        X_transformed = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])

        mock_pipeline = Mock()
        mock_pipeline.__getitem__.return_value.transform.return_value = X_transformed

        # get_pipeline_feature_namesをモック
        with patch('src.mlops.utils.pipeline_utils.get_pipeline_feature_names') as mock_get_names:
            mock_get_names.return_value = ['scaled_feature_0', 'scaled_feature_1']

            result_data, result_names = get_transformed_data_with_feature_names(
                mock_pipeline, X_original, ['feature_0', 'feature_1']
            )

        # アサーション
        np.testing.assert_array_equal(result_data, X_transformed)
        assert result_names == ['scaled_feature_0', 'scaled_feature_1']
        mock_pipeline.__getitem__.assert_called_once_with(slice(None, -1))
        mock_get_names.assert_called_once_with(mock_pipeline, ['feature_0', 'feature_1'])

    def test_data_transformation_with_sampling(self):
        """サンプリング含むデータ変換テスト"""
        # Mock設定
        X_original = pd.DataFrame({
            'feature_0': [1, 2, 3],
            'feature_1': [4, 5, 6]
        })

        # AttributeErrorを発生させてサンプリング処理をトリガー
        mock_pipeline = Mock()
        mock_pipeline.__getitem__.return_value.transform.side_effect = AttributeError("No transform method")

        # サンプリングステップを含む手動変換
        mock_sampler = Mock()
        type(mock_sampler).__name__ = "SMOTE"

        mock_scaler = Mock()
        def scaler_transform(X):
            return X * 2  # 簡単な変換
        mock_scaler.transform = scaler_transform
        type(mock_scaler).__name__ = "StandardScaler"

        mock_classifier = Mock()

        mock_pipeline.steps = [
            ("sampler", mock_sampler),
            ("scaler", mock_scaler),
            ("classifier", mock_classifier)
        ]

        with patch('src.mlops.utils.pipeline_utils.get_pipeline_feature_names') as mock_get_names:
            mock_get_names.return_value = ['processed_feature_0', 'processed_feature_1']

            result_data, result_names = get_transformed_data_with_feature_names(
                mock_pipeline, X_original, ['feature_0', 'feature_1']
            )

        # サンプラーはスキップされ、スケーラーのみ適用される
        expected_data = X_original * 2
        pd.testing.assert_frame_equal(result_data, expected_data)
        assert result_names == ['processed_feature_0', 'processed_feature_1']

    def test_with_numpy_array_input(self):
        """numpy array入力のテスト"""
        # Mock設定
        X_original = np.array([[1, 2], [3, 4], [5, 6]])
        X_transformed = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

        mock_pipeline = Mock()
        mock_pipeline.__getitem__.return_value.transform.return_value = X_transformed

        with patch('src.mlops.utils.pipeline_utils.get_pipeline_feature_names') as mock_get_names:
            mock_get_names.return_value = ['transformed_0', 'transformed_1']

            result_data, result_names = get_transformed_data_with_feature_names(
                mock_pipeline, X_original, ['original_0', 'original_1']
            )

        # アサーション
        np.testing.assert_array_equal(result_data, X_transformed)
        assert result_names == ['transformed_0', 'transformed_1']

    def test_multiple_sampling_classes_handling(self):
        """複数のサンプリングクラスの処理テスト"""
        # Mock設定
        X_original = pd.DataFrame({'feature_0': [1, 2], 'feature_1': [3, 4]})

        mock_pipeline = Mock()
        mock_pipeline.__getitem__.return_value.transform.side_effect = AttributeError("No transform method")

        # 複数のサンプリングクラスを含むステップ
        mock_smote = Mock()
        type(mock_smote).__name__ = "SMOTE"

        mock_adasyn = Mock()
        type(mock_adasyn).__name__ = "ADASYN"

        mock_borderline = Mock()
        type(mock_borderline).__name__ = "BorderlineSMOTE"

        mock_scaler = Mock()
        mock_scaler.transform = lambda X: X + 10  # 簡単な変換
        type(mock_scaler).__name__ = "StandardScaler"

        mock_classifier = Mock()

        mock_pipeline.steps = [
            ("smote", mock_smote),
            ("adasyn", mock_adasyn),
            ("borderline", mock_borderline),
            ("scaler", mock_scaler),
            ("classifier", mock_classifier)
        ]

        with patch('src.mlops.utils.pipeline_utils.get_pipeline_feature_names') as mock_get_names:
            mock_get_names.return_value = ['final_0', 'final_1']

            result_data, result_names = get_transformed_data_with_feature_names(
                mock_pipeline, X_original, ['feature_0', 'feature_1']
            )

        # 全てのサンプリングクラスがスキップされ、スケーラーのみ適用
        expected_data = X_original + 10
        pd.testing.assert_frame_equal(result_data, expected_data)
        assert result_names == ['final_0', 'final_1']

    def test_empty_dataframe_input(self):
        """空のDataFrame入力のテスト"""
        # Mock設定
        X_empty = pd.DataFrame()
        X_transformed_empty = np.array([]).reshape(0, 0)

        mock_pipeline = Mock()
        mock_pipeline.__getitem__.return_value.transform.return_value = X_transformed_empty

        with patch('src.mlops.utils.pipeline_utils.get_pipeline_feature_names') as mock_get_names:
            mock_get_names.return_value = []

            result_data, result_names = get_transformed_data_with_feature_names(
                mock_pipeline, X_empty, []
            )

        # 空のデータが適切に処理されることを確認
        np.testing.assert_array_equal(result_data, X_transformed_empty)
        assert result_names == []

    def test_single_column_dataframe(self):
        """単一列DataFrameのテスト"""
        # Mock設定
        X_single = pd.DataFrame({'single_feature': [1, 2, 3, 4, 5]})
        X_transformed_single = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]])

        mock_pipeline = Mock()
        mock_pipeline.__getitem__.return_value.transform.return_value = X_transformed_single

        with patch('src.mlops.utils.pipeline_utils.get_pipeline_feature_names') as mock_get_names:
            mock_get_names.return_value = ['transformed_single_feature']

            result_data, result_names = get_transformed_data_with_feature_names(
                mock_pipeline, X_single, ['single_feature']
            )

        # アサーション
        np.testing.assert_array_equal(result_data, X_transformed_single)
        assert result_names == ['transformed_single_feature']

    def test_pipeline_steps_copy_behavior(self):
        """パイプラインステップのコピー動作テスト"""
        # Mock設定
        X_original = pd.DataFrame({'feature_0': [1, 2]})

        mock_pipeline = Mock()
        mock_pipeline.__getitem__.return_value.transform.side_effect = AttributeError("No transform method")

        mock_transformer = Mock()
        mock_transformer.transform = lambda X: X * 3
        type(mock_transformer).__name__ = "CustomTransformer"

        mock_classifier = Mock()

        mock_pipeline.steps = [
            ("transformer", mock_transformer),
            ("classifier", mock_classifier)
        ]

        with patch('src.mlops.utils.pipeline_utils.get_pipeline_feature_names') as mock_get_names:
            mock_get_names.return_value = ['custom_feature_0']

            # 元のDataFrameをコピーして変換することを確認
            original_copy = X_original.copy()
            result_data, result_names = get_transformed_data_with_feature_names(
                mock_pipeline, X_original, ['feature_0']
            )

        # 元のDataFrameが変更されていないことを確認
        pd.testing.assert_frame_equal(X_original, original_copy)
        assert result_names == ['custom_feature_0']

    def test_integration_with_real_data(self):
        """実データでの統合テスト"""
        # 実際のDataFrameを作成
        X_real, _ = create_binary_classification_data(n_rows=50, n_features=3)

        # Mock pipeline（簡単な変換を実装）
        mock_pipeline = Mock()

        # 実際の変換を模擬
        def mock_transform(X):
            return X.values * 0.5  # DataFrameをnumpy arrayに変換し、0.5倍

        mock_pipeline.__getitem__.return_value.transform = mock_transform

        with patch('src.mlops.utils.pipeline_utils.get_pipeline_feature_names') as mock_get_names:
            mock_get_names.return_value = ['scaled_feature_0', 'scaled_feature_1', 'scaled_feature_2']

            result_data, result_names = get_transformed_data_with_feature_names(
                mock_pipeline, X_real, X_real.columns.tolist()
            )

        # データが正しく変換されることを確認
        expected_data = X_real.values * 0.5
        np.testing.assert_array_equal(result_data, expected_data)
        assert result_names == ['scaled_feature_0', 'scaled_feature_1', 'scaled_feature_2']
        assert result_data.shape == X_real.shape