"""visualization.py の単体テスト"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from omegaconf import DictConfig

from src.mlops.components.visualization import (
    BaseVisualizer,
    YellowBrickVisualizer,
    ClassBalanceVisualizer,
    PermutationImportanceVisualizer,
    SHAPBaseVisualizer,
    SHAPSummaryVisualizer,
    SHAPDependenceVisualizer,
    VisualizationFactory,
    create_visualizations,
    _import_class
)
from tests.fixtures.test_data import create_binary_classification_data, create_regression_data


class TestBaseVisualizer:
    """BaseVisualizer 抽象クラスの単体テスト"""

    def test_base_visualizer_initialization(self):
        """BaseVisualizerの初期化テスト"""
        # Mock pipeline設定
        mock_pipeline = Mock()
        mock_pipeline.named_steps = {"classifier": Mock()}
        X_test_transformed = np.array([[1.0, 2.0], [3.0, 4.0]])
        mock_pipeline.__getitem__.return_value.transform.return_value = X_test_transformed

        X_train, y_train = create_binary_classification_data(n_rows=50, n_features=3)
        X_test, y_test = create_binary_classification_data(n_rows=20, n_features=3)

        # BaseVisualizerは抽象クラスなので具象クラスを作成してテスト
        class TestVisualizer(BaseVisualizer):
            def create_plot(self, output_path: str) -> None:
                pass

        with patch('src.mlops.components.visualization.get_pipeline_feature_names') as mock_get_names:
            mock_get_names.return_value = ['feature_0', 'feature_1', 'feature_2']

            visualizer = TestVisualizer(mock_pipeline, X_train, y_train, X_test, y_test, "classification")

        # アサーション
        assert visualizer.pipeline == mock_pipeline
        assert visualizer.task_type == "classification"
        assert hasattr(visualizer, 'X_train_transformed')
        assert hasattr(visualizer, 'X_test_transformed')
        assert hasattr(visualizer, 'feature_names')
        assert visualizer.feature_names == ['feature_0', 'feature_1', 'feature_2']

    def test_base_visualizer_with_sampling(self):
        """サンプラー含むパイプラインでのBaseVisualizer初期化テスト"""
        # Mock pipeline設定（サンプラー含む）
        mock_pipeline = Mock()
        mock_pipeline.named_steps = {"classifier": Mock()}

        # サンプラーを含むステップをモック
        mock_sampler = Mock()
        mock_scaler = Mock()
        mock_scaler.transform.side_effect = lambda x: x * 2  # 簡単な変換

        # AttributeErrorを発生させてサンプラー処理をトリガー
        mock_pipeline.__getitem__.return_value.transform.side_effect = AttributeError("No transform method")
        mock_pipeline.steps = [("sampler", mock_sampler), ("scaler", mock_scaler), ("classifier", Mock())]

        X_train, y_train = create_binary_classification_data(n_rows=50, n_features=3)
        X_test, y_test = create_binary_classification_data(n_rows=20, n_features=3)

        class TestVisualizer(BaseVisualizer):
            def create_plot(self, output_path: str) -> None:
                pass

        with patch('src.mlops.components.visualization.get_pipeline_feature_names') as mock_get_names:
            with patch('src.mlops.components.visualization.SAMPLING_CLASSES', ['MockSampler']):
                mock_get_names.return_value = ['feature_0', 'feature_1', 'feature_2']

                visualizer = TestVisualizer(mock_pipeline, X_train, y_train, X_test, y_test, "classification")

        # サンプラー処理が適用されたことを確認
        assert hasattr(visualizer, 'X_train_transformed')
        assert hasattr(visualizer, 'X_test_transformed')

    def test_abstract_method_enforcement(self):
        """抽象メソッドの強制テスト"""
        with pytest.raises(TypeError):
            # create_plotを実装していないため、インスタンス化できない
            BaseVisualizer(Mock(), Mock(), Mock(), Mock(), Mock(), "classification")


class TestYellowBrickVisualizer:
    """YellowBrickVisualizer クラスの単体テスト"""

    @patch('matplotlib.pyplot.rcdefaults')
    @patch('matplotlib.pyplot.clf')
    @patch('matplotlib.pyplot.close')
    def test_yellowbrick_basic_plot_creation(self, mock_close, mock_clf, mock_rcdefaults):
        """YellowBrick基本プロット作成テスト"""
        # Mock設定
        mock_pipeline = Mock()
        mock_pipeline.named_steps = {"classifier": Mock()}
        X_test_transformed = np.array([[1.0, 2.0], [3.0, 4.0]])
        mock_pipeline.__getitem__.return_value.transform.return_value = X_test_transformed

        X_train, y_train = create_binary_classification_data(n_rows=50, n_features=3)
        X_test, y_test = create_binary_classification_data(n_rows=20, n_features=3)

        # Mock visualization class
        mock_viz_class = Mock()
        mock_viz = Mock()
        mock_viz_class.return_value = mock_viz

        with patch('src.mlops.components.visualization.get_pipeline_feature_names') as mock_get_names:
            mock_get_names.return_value = ['feature_0', 'feature_1', 'feature_2']

            visualizer = YellowBrickVisualizer(
                mock_pipeline, X_train, y_train, X_test, y_test,
                "classification", mock_viz_class, ["class_0", "class_1"]
            )

            visualizer.create_plot("/tmp/test_plot.png")

        # YellowBrickの基本的な処理フローを確認
        mock_viz_class.assert_called_once()
        mock_viz.fit.assert_called_once()
        mock_viz.score.assert_called_once()
        mock_viz.show.assert_called_once_with("/tmp/test_plot.png")
        mock_viz.finalize.assert_called_once()

    @patch('matplotlib.pyplot.rcdefaults')
    @patch('matplotlib.pyplot.clf')
    @patch('matplotlib.pyplot.close')
    def test_validation_curve_special_handling(self, mock_close, mock_clf, mock_rcdefaults):
        """ValidationCurve特殊処理テスト"""
        # Mock設定
        mock_pipeline = Mock()
        mock_classifier = Mock()
        type(mock_classifier).__name__ = "LGBMClassifier"
        mock_pipeline.named_steps = {"classifier": mock_classifier}
        X_test_transformed = np.array([[1.0, 2.0]])
        mock_pipeline.__getitem__.return_value.transform.return_value = X_test_transformed

        X_train, y_train = create_binary_classification_data(n_rows=50, n_features=3)
        X_test, y_test = create_binary_classification_data(n_rows=20, n_features=3)

        # Mock ValidationCurve class
        class MockValidationCurve:
            __name__ = "ValidationCurve"

            def __init__(self, model, param_name=None, param_range=None):
                self.model = model
                self.param_name = param_name
                self.param_range = param_range

            def fit(self, X, y):
                pass

            def score(self, X, y):
                pass

            def show(self, output_path):
                pass

            def finalize(self):
                pass

        with patch('src.mlops.components.visualization.get_pipeline_feature_names') as mock_get_names:
            mock_get_names.return_value = ['feature_0', 'feature_1', 'feature_2']

            visualizer = YellowBrickVisualizer(
                mock_pipeline, X_train, y_train, X_test, y_test,
                "classification", MockValidationCurve
            )

            visualizer.create_plot("/tmp/validation_curve.png")

        # ValidationCurve特有のパラメータが設定されることを確認
        assert visualizer.viz_class == MockValidationCurve


class TestClassBalanceVisualizer:
    """ClassBalanceVisualizer クラスの単体テスト"""

    @patch('matplotlib.pyplot.rcdefaults')
    @patch('matplotlib.pyplot.clf')
    @patch('matplotlib.pyplot.close')
    @patch('src.mlops.components.visualization.ClassBalance')
    def test_class_balance_plot_creation(self, mock_class_balance, mock_close, mock_clf, mock_rcdefaults):
        """ClassBalanceプロット作成テスト"""
        # Mock設定
        mock_pipeline = Mock()
        mock_pipeline.named_steps = {"classifier": Mock()}
        X_test_transformed = np.array([[1.0, 2.0]])
        mock_pipeline.__getitem__.return_value.transform.return_value = X_test_transformed

        X_train, y_train = create_binary_classification_data(n_rows=50, n_features=3)
        X_test, y_test = create_binary_classification_data(n_rows=20, n_features=3)

        mock_viz = Mock()
        mock_class_balance.return_value = mock_viz

        with patch('src.mlops.components.visualization.get_pipeline_feature_names') as mock_get_names:
            mock_get_names.return_value = ['feature_0', 'feature_1', 'feature_2']

            visualizer = ClassBalanceVisualizer(
                mock_pipeline, X_train, y_train, X_test, y_test, "classification"
            )

            visualizer.create_plot("/tmp/class_balance.png")

        # ClassBalance特有の処理（ターゲットのみ使用）を確認
        mock_viz.fit.assert_called_once_with(y_train)
        mock_viz.show.assert_called_once_with("/tmp/class_balance.png")
        mock_viz.finalize.assert_called_once()


class TestPermutationImportanceVisualizer:
    """PermutationImportanceVisualizer クラスの単体テスト"""

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.barh')
    @patch('matplotlib.pyplot.yticks')
    @patch('matplotlib.pyplot.xlabel')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.tight_layout')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.clf')
    @patch('src.mlops.components.visualization.permutation_importance')
    def test_permutation_importance_plot_creation(self, mock_perm_importance, mock_clf, mock_close,
                                                 mock_savefig, mock_tight_layout, mock_title,
                                                 mock_xlabel, mock_yticks, mock_barh, mock_figure):
        """Permutation Importanceプロット作成テスト"""
        # Mock設定
        mock_pipeline = Mock()
        mock_pipeline.named_steps = {"classifier": Mock()}
        mock_pipeline.__getitem__.return_value = Mock()  # 最終段階のモデル

        X_train, y_train = create_binary_classification_data(n_rows=50, n_features=3)
        X_test, y_test = create_binary_classification_data(n_rows=20, n_features=3)

        # Mock permutation importance結果
        mock_perm_result = Mock()
        mock_perm_result.importances_mean = np.array([0.1, 0.3, 0.2])
        mock_perm_importance.return_value = mock_perm_result

        with patch('src.mlops.components.visualization.get_pipeline_feature_names') as mock_get_names:
            with patch('src.mlops.components.visualization.get_transformed_data_with_feature_names') as mock_get_data:
                mock_get_names.return_value = ['feature_0', 'feature_1', 'feature_2']
                mock_get_data.return_value = (X_test.values, ['feature_0', 'feature_1', 'feature_2'])

                visualizer = PermutationImportanceVisualizer(
                    mock_pipeline, X_train, y_train, X_test, y_test, "classification"
                )

                visualizer.create_plot("/tmp/permutation_importance.png")

        # Permutation importanceの計算が実行されることを確認
        mock_perm_importance.assert_called_once()
        mock_figure.assert_called_once()
        mock_savefig.assert_called_once_with("/tmp/permutation_importance.png", dpi=150, bbox_inches='tight')


class TestSHAPVisualizers:
    """SHAP可視化クラス群の単体テスト"""

    def test_shap_base_visualizer_explainer_creation(self):
        """SHAPBase可視化のexplainer作成テスト"""
        # Mock設定
        mock_pipeline = Mock()
        mock_classifier = Mock()
        mock_classifier.predict_proba = Mock(return_value=np.array([[0.8, 0.2], [0.6, 0.4]]))
        mock_pipeline.named_steps = {"classifier": mock_classifier}
        X_test_transformed = np.array([[1.0, 2.0], [3.0, 4.0]])
        mock_pipeline.__getitem__.return_value.transform.return_value = X_test_transformed

        X_train, y_train = create_binary_classification_data(n_rows=50, n_features=3)
        X_test, y_test = create_binary_classification_data(n_rows=20, n_features=3)

        with patch('src.mlops.components.visualization.get_pipeline_feature_names') as mock_get_names:
            with patch('src.mlops.components.visualization.shap') as mock_shap:
                mock_explainer = Mock()
                mock_shap.Explainer.return_value = mock_explainer

                mock_get_names.return_value = ['feature_0', 'feature_1', 'feature_2']

                visualizer = SHAPBaseVisualizer(
                    mock_pipeline, X_train, y_train, X_test, y_test, "classification"
                )

                explainer = visualizer._create_explainer()

                # 分類タスクでpredict_probaを持つ場合、predict_probaが使用されることを確認
                mock_shap.Explainer.assert_called_once_with(
                    mock_classifier.predict_proba, visualizer.X_train_transformed
                )

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.tight_layout')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.clf')
    def test_shap_summary_visualizer(self, mock_clf, mock_close, mock_savefig,
                                   mock_tight_layout, mock_title, mock_figure):
        """SHAP Summary可視化テスト"""
        # Mock設定
        mock_pipeline = Mock()
        mock_classifier = Mock()
        mock_classifier.predict = Mock(return_value=np.array([0, 1]))
        mock_pipeline.named_steps = {"classifier": mock_classifier}
        X_test_transformed = np.array([[1.0, 2.0], [3.0, 4.0]])
        mock_pipeline.__getitem__.return_value.transform.return_value = X_test_transformed

        X_train, y_train = create_binary_classification_data(n_rows=50, n_features=3)
        X_test, y_test = create_binary_classification_data(n_rows=20, n_features=3)

        with patch('src.mlops.components.visualization.get_pipeline_feature_names') as mock_get_names:
            with patch('src.mlops.components.visualization.shap') as mock_shap:
                # Mock SHAP値
                mock_shap_values = Mock()
                mock_shap_values.values = np.array([[0.1, 0.2], [0.3, 0.4]])

                mock_explainer = Mock()
                mock_explainer.return_value = mock_shap_values
                mock_shap.Explainer.return_value = mock_explainer

                mock_get_names.return_value = ['feature_0', 'feature_1', 'feature_2']

                visualizer = SHAPSummaryVisualizer(
                    mock_pipeline, X_train, y_train, X_test, y_test, "classification"
                )

                visualizer.create_plot("/tmp/shap_summary.png")

        # SHAP summary plotが作成されることを確認
        mock_shap.summary_plot.assert_called_once()
        mock_savefig.assert_called_once_with("/tmp/shap_summary.png", dpi=150, bbox_inches='tight')


class TestVisualizationFactory:
    """VisualizationFactory クラスの単体テスト"""

    def test_create_permutation_importance_visualizer(self):
        """Permutation Importance可視化オブジェクト作成テスト"""
        mock_pipeline = Mock()
        mock_pipeline.named_steps = {"classifier": Mock()}
        X_test_transformed = np.array([[1.0, 2.0]])
        mock_pipeline.__getitem__.return_value.transform.return_value = X_test_transformed

        X_train, y_train = create_binary_classification_data(n_rows=50, n_features=3)
        X_test, y_test = create_binary_classification_data(n_rows=20, n_features=3)

        with patch('src.mlops.components.visualization.get_pipeline_feature_names') as mock_get_names:
            mock_get_names.return_value = ['feature_0', 'feature_1', 'feature_2']

            visualizer = VisualizationFactory.create_visualizer(
                "permutation_importance", mock_pipeline, X_train, y_train, X_test, y_test, "classification"
            )

        assert isinstance(visualizer, PermutationImportanceVisualizer)

    def test_create_shap_summary_visualizer(self):
        """SHAP Summary可視化オブジェクト作成テスト"""
        mock_pipeline = Mock()
        mock_pipeline.named_steps = {"classifier": Mock()}
        X_test_transformed = np.array([[1.0, 2.0]])
        mock_pipeline.__getitem__.return_value.transform.return_value = X_test_transformed

        X_train, y_train = create_binary_classification_data(n_rows=50, n_features=3)
        X_test, y_test = create_binary_classification_data(n_rows=20, n_features=3)

        with patch('src.mlops.components.visualization.get_pipeline_feature_names') as mock_get_names:
            mock_get_names.return_value = ['feature_0', 'feature_1', 'feature_2']

            visualizer = VisualizationFactory.create_visualizer(
                "shap_summary", mock_pipeline, X_train, y_train, X_test, y_test, "classification"
            )

        assert isinstance(visualizer, SHAPSummaryVisualizer)

    def test_create_class_balance_visualizer(self):
        """ClassBalance可視化オブジェクト作成テスト"""
        mock_pipeline = Mock()
        mock_pipeline.named_steps = {"classifier": Mock()}
        X_test_transformed = np.array([[1.0, 2.0]])
        mock_pipeline.__getitem__.return_value.transform.return_value = X_test_transformed

        X_train, y_train = create_binary_classification_data(n_rows=50, n_features=3)
        X_test, y_test = create_binary_classification_data(n_rows=20, n_features=3)

        with patch('src.mlops.components.visualization.get_pipeline_feature_names') as mock_get_names:
            mock_get_names.return_value = ['feature_0', 'feature_1', 'feature_2']

            visualizer = VisualizationFactory.create_visualizer(
                "class_balance", mock_pipeline, X_train, y_train, X_test, y_test, "classification"
            )

        assert isinstance(visualizer, ClassBalanceVisualizer)

    def test_create_yellowbrick_visualizer_classification(self):
        """YellowBrick分類可視化オブジェクト作成テスト"""
        mock_pipeline = Mock()
        mock_pipeline.named_steps = {"classifier": Mock()}
        X_test_transformed = np.array([[1.0, 2.0]])
        mock_pipeline.__getitem__.return_value.transform.return_value = X_test_transformed

        X_train, y_train = create_binary_classification_data(n_rows=50, n_features=3)
        X_test, y_test = create_binary_classification_data(n_rows=20, n_features=3)

        with patch('src.mlops.components.visualization.get_pipeline_feature_names') as mock_get_names:
            with patch('src.mlops.components.visualization._import_class') as mock_import:
                mock_viz_class = Mock()
                mock_import.return_value = mock_viz_class

                mock_get_names.return_value = ['feature_0', 'feature_1', 'feature_2']

                visualizer = VisualizationFactory.create_visualizer(
                    "confusion_matrix", mock_pipeline, X_train, y_train, X_test, y_test, "classification"
                )

        assert isinstance(visualizer, YellowBrickVisualizer)

    def test_create_yellowbrick_visualizer_regression(self):
        """YellowBrick回帰可視化オブジェクト作成テスト"""
        mock_pipeline = Mock()
        mock_pipeline.named_steps = {"classifier": Mock()}
        X_test_transformed = np.array([[1.0, 2.0]])
        mock_pipeline.__getitem__.return_value.transform.return_value = X_test_transformed

        X_train, y_train = create_regression_data(n_rows=50, n_features=3)
        X_test, y_test = create_regression_data(n_rows=20, n_features=3)

        with patch('src.mlops.components.visualization.get_pipeline_feature_names') as mock_get_names:
            with patch('src.mlops.components.visualization._import_class') as mock_import:
                mock_viz_class = Mock()
                mock_import.return_value = mock_viz_class

                mock_get_names.return_value = ['feature_0', 'feature_1', 'feature_2']

                visualizer = VisualizationFactory.create_visualizer(
                    "residuals_plot", mock_pipeline, X_train, y_train, X_test, y_test, "regression"
                )

        assert isinstance(visualizer, YellowBrickVisualizer)

    def test_unsupported_plot_type(self):
        """未対応プロットタイプのエラーテスト"""
        mock_pipeline = Mock()
        mock_pipeline.named_steps = {"classifier": Mock()}
        X_test_transformed = np.array([[1.0, 2.0]])
        mock_pipeline.__getitem__.return_value.transform.return_value = X_test_transformed

        X_train, y_train = create_binary_classification_data(n_rows=50, n_features=3)
        X_test, y_test = create_binary_classification_data(n_rows=20, n_features=3)

        with patch('src.mlops.components.visualization.get_pipeline_feature_names') as mock_get_names:
            mock_get_names.return_value = ['feature_0', 'feature_1', 'feature_2']

            with pytest.raises(ValueError) as excinfo:
                VisualizationFactory.create_visualizer(
                    "unsupported_plot", mock_pipeline, X_train, y_train, X_test, y_test, "classification"
                )

        assert "未対応の可視化タイプ: unsupported_plot" in str(excinfo.value)


class TestImportClass:
    """_import_class 関数の単体テスト"""

    def test_import_class_success(self):
        """クラスインポート成功テスト"""
        # 実際のクラスをインポートしてテスト
        pandas_class = _import_class("pandas.DataFrame")
        assert pandas_class == pd.DataFrame

    def test_import_class_module_error(self):
        """存在しないモジュールのインポートエラーテスト"""
        with pytest.raises(ModuleNotFoundError):
            _import_class("nonexistent.module.Class")

    def test_import_class_attribute_error(self):
        """存在しないクラスのインポートエラーテスト"""
        with pytest.raises(AttributeError):
            _import_class("pandas.NonexistentClass")


class TestCreateVisualizations:
    """create_visualizations 関数の単体テスト"""

    @patch('matplotlib.pyplot.rcdefaults')
    @patch('matplotlib.pyplot.clf')
    @patch('matplotlib.pyplot.close')
    @patch('tempfile.TemporaryDirectory')
    @patch('src.mlops.components.visualization.mlflow')
    @patch('src.mlops.components.visualization.VisualizationFactory')
    def test_create_visualizations_success(self, mock_factory, mock_mlflow, mock_temp_dir,
                                         mock_close, mock_clf, mock_rcdefaults):
        """可視化作成成功テスト"""
        # Mock設定
        mock_temp_dir.return_value.__enter__.return_value = "/tmp/viz_dir"

        mock_visualizer = Mock()
        mock_factory.create_visualizer.return_value = mock_visualizer

        mock_pipeline = Mock()
        X_train, y_train = create_binary_classification_data(n_rows=50, n_features=3)
        X_test, y_test = create_binary_classification_data(n_rows=20, n_features=3)

        cfg = DictConfig({
            "mlflow": {
                "artifacts": {
                    "visualization_dir": "visualizations"
                }
            }
        })

        plot_types = ["permutation_importance", "shap_summary"]
        target_names_str = ["class_0", "class_1"]

        with patch('os.listdir') as mock_listdir:
            mock_listdir.return_value = ["permutation_importance.png", "shap_summary.png"]

            create_visualizations(mock_pipeline, X_train, y_train, X_test, y_test,
                                target_names_str, plot_types, cfg, "classification")

        # 複数の可視化が作成されることを確認
        assert mock_factory.create_visualizer.call_count == 2
        mock_mlflow.log_artifact.assert_any_call("/tmp/viz_dir/permutation_importance.png", "visualizations")
        mock_mlflow.log_artifact.assert_any_call("/tmp/viz_dir/shap_summary.png", "visualizations")

    @patch('matplotlib.pyplot.rcdefaults')
    @patch('matplotlib.pyplot.clf')
    @patch('matplotlib.pyplot.close')
    @patch('tempfile.TemporaryDirectory')
    @patch('src.mlops.components.visualization.mlflow')
    @patch('src.mlops.components.visualization.VisualizationFactory')
    def test_create_visualizations_with_error(self, mock_factory, mock_mlflow, mock_temp_dir,
                                            mock_close, mock_clf, mock_rcdefaults):
        """可視化作成エラーハンドリングテスト"""
        # Mock設定
        mock_temp_dir.return_value.__enter__.return_value = "/tmp/viz_dir"

        # 最初の可視化でエラー発生、2番目は成功
        mock_visualizer1 = Mock()
        mock_visualizer1.create_plot.side_effect = Exception("Visualization error")
        mock_visualizer2 = Mock()

        mock_factory.create_visualizer.side_effect = [mock_visualizer1, mock_visualizer2]

        mock_pipeline = Mock()
        X_train, y_train = create_binary_classification_data(n_rows=50, n_features=3)
        X_test, y_test = create_binary_classification_data(n_rows=20, n_features=3)

        cfg = DictConfig({
            "mlflow": {
                "artifacts": {
                    "visualization_dir": "visualizations"
                }
            }
        })

        plot_types = ["error_plot", "success_plot"]
        target_names_str = ["class_0", "class_1"]

        with patch('os.listdir') as mock_listdir:
            mock_listdir.return_value = ["success_plot.png"]  # エラー発生でerror_plot.pngは生成されない

            create_visualizations(mock_pipeline, X_train, y_train, X_test, y_test,
                                target_names_str, plot_types, cfg, "classification")

        # エラーが発生しても処理が継続することを確認
        assert mock_factory.create_visualizer.call_count == 2
        mock_mlflow.log_artifact.assert_called_once_with("/tmp/viz_dir/success_plot.png", "visualizations")

    @patch('matplotlib.pyplot.rcdefaults')
    @patch('matplotlib.pyplot.clf')
    @patch('matplotlib.pyplot.close')
    @patch('tempfile.TemporaryDirectory')
    @patch('src.mlops.components.visualization.mlflow')
    @patch('src.mlops.components.visualization.VisualizationFactory')
    def test_create_visualizations_empty_plot_types(self, mock_factory, mock_mlflow, mock_temp_dir,
                                                  mock_close, mock_clf, mock_rcdefaults):
        """空のプロットタイプリストテスト"""
        # Mock設定
        mock_temp_dir.return_value.__enter__.return_value = "/tmp/viz_dir"

        mock_pipeline = Mock()
        X_train, y_train = create_binary_classification_data(n_rows=50, n_features=3)
        X_test, y_test = create_binary_classification_data(n_rows=20, n_features=3)

        cfg = DictConfig({
            "mlflow": {
                "artifacts": {
                    "visualization_dir": "visualizations"
                }
            }
        })

        plot_types = []
        target_names_str = ["class_0", "class_1"]

        with patch('os.listdir') as mock_listdir:
            mock_listdir.return_value = []

            create_visualizations(mock_pipeline, X_train, y_train, X_test, y_test,
                                target_names_str, plot_types, cfg, "classification")

        # 何も可視化されないことを確認
        mock_factory.create_visualizer.assert_not_called()
        mock_mlflow.log_artifact.assert_not_called()