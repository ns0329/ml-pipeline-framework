"""artifacts.py の単体テスト"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock, mock_open
from omegaconf import DictConfig

from src.mlops.components.artifacts import (
    save_model_artifacts,
    log_config_parameters,
    log_runtime_parameters,
    log_experiment_metrics,
    create_prediction_dataframe,
    save_prediction_results,
    setup_mlflow_experiment,
    set_mlflow_tags
)
from tests.fixtures.test_data import create_binary_classification_data, create_regression_data


class TestSaveModelArtifacts:
    """save_model_artifacts 関数の単体テスト"""

    @patch('src.mlops.components.artifacts.mlflow')
    @patch('src.mlops.components.artifacts.joblib')
    @patch('tempfile.TemporaryDirectory')
    def test_save_model_artifacts(self, mock_temp_dir, mock_joblib, mock_mlflow):
        """モデルアーティファクト保存テスト"""
        # Mock設定
        mock_temp_dir.return_value.__enter__.return_value = "/tmp/test_dir"

        mock_pipeline = Mock()
        mock_pipeline.named_steps = {"classifier": Mock()}
        type(mock_pipeline.named_steps["classifier"]).__name__ = "RandomForestClassifier"

        feature_names = ["feature_0", "feature_1", "feature_2"]
        target_names = [0, 1]
        cfg = DictConfig({
            "mlflow": {
                "artifacts": {
                    "model_filename": "model.pkl",
                    "info_filename": "model_info.json",
                    "model_dir": "models"
                }
            }
        })

        with patch('builtins.open', mock_open()) as mock_file:
            save_model_artifacts(mock_pipeline, feature_names, target_names, cfg)

        # アサーション
        mock_joblib.dump.assert_called_once()
        mock_mlflow.log_artifact.assert_any_call("/tmp/test_dir/model.pkl", "models")
        mock_mlflow.log_artifact.assert_any_call("/tmp/test_dir/model_info.json", "models")

    @patch('src.mlops.components.artifacts.mlflow')
    @patch('src.mlops.components.artifacts.joblib')
    @patch('tempfile.TemporaryDirectory')
    def test_save_model_info_content(self, mock_temp_dir, mock_joblib, mock_mlflow):
        """モデル情報内容のテスト"""
        # Mock設定
        mock_temp_dir.return_value.__enter__.return_value = "/tmp/test_dir"

        mock_pipeline = Mock()
        mock_classifier = Mock()
        type(mock_classifier).__name__ = "LGBMClassifier"
        mock_pipeline.named_steps = {"classifier": mock_classifier}

        feature_names = ["amount", "transaction_time", "customer_id"]
        target_names = ["legitimate", "fraud"]
        cfg = DictConfig({
            "mlflow": {
                "artifacts": {
                    "model_filename": "model.pkl",
                    "info_filename": "info.json",
                    "model_dir": "artifacts"
                }
            }
        })

        written_data = None

        def capture_write(data):
            nonlocal written_data
            written_data = data

        mock_file = mock_open()
        mock_file.return_value.write.side_effect = capture_write

        with patch('builtins.open', mock_file):
            save_model_artifacts(mock_pipeline, feature_names, target_names, cfg)

        # JSON内容を確認
        if written_data:
            info_data = json.loads(written_data)
            assert info_data["model_type"] == "LGBMClassifier"
            assert info_data["feature_names"] == feature_names
            assert info_data["target_names"] == ["legitimate", "fraud"]


class TestLogConfigParameters:
    """log_config_parameters 関数の単体テスト"""

    @patch('src.mlops.components.artifacts.mlflow')
    def test_log_config_parameters(self, mock_mlflow):
        """設定パラメータログのテスト"""
        cfg = DictConfig({
            "data": {"random_state": 42, "test_size": 0.2},
            "model": {"n_estimators": 100, "max_depth": 5},
            "nested": {"param": {"sub_param": "value"}}
        })

        log_config_parameters(cfg)

        # MLflowへの記録を確認
        expected_calls = [
            ("data", str({"random_state": 42, "test_size": 0.2})),
            ("model", str({"n_estimators": 100, "max_depth": 5})),
            ("nested", str({"param": {"sub_param": "value"}}))
        ]

        for key, value in expected_calls:
            mock_mlflow.log_param.assert_any_call(key, value)

    @patch('src.mlops.components.artifacts.mlflow')
    def test_log_empty_config(self, mock_mlflow):
        """空の設定のテスト"""
        cfg = DictConfig({})
        log_config_parameters(cfg)

        # 何も記録されないことを確認
        mock_mlflow.log_param.assert_not_called()


class TestLogRuntimeParameters:
    """log_runtime_parameters 関数の単体テスト"""

    @patch('src.mlops.components.artifacts.mlflow')
    def test_log_runtime_parameters_without_tuning(self, mock_mlflow):
        """チューニングなしの実行時パラメータログテスト"""
        # Mock pipeline設定
        mock_scaler = Mock()
        type(mock_scaler).__name__ = "StandardScaler"
        mock_classifier = Mock()
        type(mock_classifier).__name__ = "RandomForestClassifier"
        mock_classifier.get_params.return_value = {"n_estimators": 100, "random_state": 42}

        mock_pipeline = Mock()
        mock_pipeline.steps = [("scaler", mock_scaler), ("classifier", mock_classifier)]
        mock_pipeline.named_steps = {"classifier": mock_classifier}

        cfg = DictConfig({})

        log_runtime_parameters(mock_pipeline, cfg)

        # アサーション
        mock_mlflow.log_param.assert_any_call("pipeline_steps", 2)
        mock_mlflow.log_param.assert_any_call("step_0_name", "scaler")
        mock_mlflow.log_param.assert_any_call("step_0_class", "StandardScaler")
        mock_mlflow.log_param.assert_any_call("step_1_name", "classifier")
        mock_mlflow.log_param.assert_any_call("step_1_class", "RandomForestClassifier")
        mock_mlflow.log_param.assert_any_call("model_n_estimators", "100")
        mock_mlflow.log_param.assert_any_call("model_random_state", "42")
        mock_mlflow.log_param.assert_any_call("tuning_optimized", False)

    @patch('src.mlops.components.artifacts.mlflow')
    def test_log_runtime_parameters_with_tuning(self, mock_mlflow):
        """チューニングありの実行時パラメータログテスト"""
        # Mock pipeline設定
        mock_classifier = Mock()
        type(mock_classifier).__name__ = "LGBMClassifier"
        mock_classifier.get_params.return_value = {"n_estimators": 150, "learning_rate": 0.1}

        mock_pipeline = Mock()
        mock_pipeline.steps = [("classifier", mock_classifier)]
        mock_pipeline.named_steps = {"classifier": mock_classifier}

        cfg = DictConfig({})
        best_params = {"n_estimators": 150, "learning_rate": 0.1}

        log_runtime_parameters(mock_pipeline, cfg, best_params)

        # チューニング関連のログを確認
        mock_mlflow.log_param.assert_any_call("tuning_optimized", True)
        mock_mlflow.log_param.assert_any_call("tuned_n_estimators", "150")
        mock_mlflow.log_param.assert_any_call("tuned_learning_rate", "0.1")

    @patch('src.mlops.components.artifacts.mlflow')
    def test_log_runtime_parameters_no_classifier(self, mock_mlflow):
        """分類器なしの場合のテスト"""
        mock_pipeline = Mock()
        mock_pipeline.steps = [("scaler", Mock())]
        mock_pipeline.named_steps = {}

        cfg = DictConfig({})

        log_runtime_parameters(mock_pipeline, cfg)

        # パイプライン情報のみログされることを確認
        mock_mlflow.log_param.assert_any_call("pipeline_steps", 1)


class TestLogExperimentMetrics:
    """log_experiment_metrics 関数の単体テスト"""

    @patch('src.mlops.components.artifacts.mlflow')
    def test_log_classification_metrics_binary(self, mock_mlflow):
        """二値分類メトリクスログテスト"""
        # Mock設定
        mock_pipeline = Mock()
        mock_pipeline.predict.return_value = np.array([0, 1, 1, 0, 1])
        mock_pipeline.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7], [0.1, 0.9], [0.9, 0.1], [0.2, 0.8]])

        X_train, y_train = create_binary_classification_data(n_rows=20, n_features=3)
        X_test = pd.DataFrame(np.random.randn(5, 3), columns=['feature_0', 'feature_1', 'feature_2'])
        y_test = np.array([0, 1, 1, 0, 1])

        cfg = DictConfig({
            "optuna": {
                "scoring": {
                    "classification": "f1_binary"
                }
            }
        })
        cv_scores = np.array([0.85, 0.87, 0.84])

        # MLflow runをモック
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id_12345"
        mock_mlflow.active_run.return_value = mock_run

        log_experiment_metrics(mock_pipeline, X_train, y_train, X_test, y_test,
                             "classification", cv_scores, cfg)

        # 基本メトリクスの確認
        mock_mlflow.log_metric.assert_any_call("cv_mean", cv_scores.mean())
        mock_mlflow.log_metric.assert_any_call("cv_std", cv_scores.std())
        mock_mlflow.log_metric.assert_any_call("test_accuracy", pytest.approx(1.0))

    @patch('src.mlops.components.artifacts.mlflow')
    def test_log_classification_metrics_weighted(self, mock_mlflow):
        """重み付きF1スコアメトリクスログテスト"""
        # Mock設定
        mock_pipeline = Mock()
        mock_pipeline.predict.return_value = np.array([0, 1, 2, 0, 1])
        mock_pipeline.predict_proba.return_value = np.array([
            [0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7],
            [0.8, 0.1, 0.1], [0.2, 0.7, 0.1]
        ])

        X_train, y_train = create_binary_classification_data(n_rows=20, n_features=3)
        X_test = pd.DataFrame(np.random.randn(5, 3), columns=['feature_0', 'feature_1', 'feature_2'])
        y_test = np.array([0, 1, 2, 0, 1])  # 多クラス

        cfg = DictConfig({
            "optuna": {
                "scoring": {
                    "classification": "f1_weighted"
                }
            }
        })
        cv_scores = np.array([0.80, 0.82, 0.78])

        # MLflow runをモック
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id_12345"
        mock_mlflow.active_run.return_value = mock_run

        log_experiment_metrics(mock_pipeline, X_train, y_train, X_test, y_test,
                             "classification", cv_scores, cfg)

        # weighted F1スコアが記録されることを確認
        assert any(call.args[0] == "test_f1_weighted" for call in mock_mlflow.log_metric.call_args_list)

    @patch('src.mlops.components.artifacts.mlflow')
    def test_log_regression_metrics(self, mock_mlflow):
        """回帰メトリクスログテスト"""
        # Mock設定
        mock_pipeline = Mock()
        y_pred = np.array([1.5, 2.1, 3.2, 1.8, 2.7])
        mock_pipeline.predict.side_effect = [y_pred, np.array([1.2, 1.9, 2.8, 1.5, 2.5])]  # test, train

        X_train, y_train = create_regression_data(n_rows=20, n_features=3)
        X_test = pd.DataFrame(np.random.randn(5, 3), columns=['feature_0', 'feature_1', 'feature_2'])
        y_test = np.array([1.4, 2.0, 3.1, 1.7, 2.6])

        cfg = DictConfig({
            "optuna": {
                "scoring": {
                    "regression": "neg_mean_squared_error"
                }
            }
        })
        cv_scores = np.array([-1.2, -1.5, -1.1])

        # MLflow runをモック
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id_12345"
        mock_mlflow.active_run.return_value = mock_run

        log_experiment_metrics(mock_pipeline, X_train, y_train, X_test, y_test,
                             "regression", cv_scores, cfg)

        # 回帰メトリクスの確認
        mock_mlflow.log_metric.assert_any_call("cv_mean", cv_scores.mean())
        mock_mlflow.log_metric.assert_any_call("cv_std", cv_scores.std())
        assert any("test_mse" in call.args[0] for call in mock_mlflow.log_metric.call_args_list)
        assert any("test_rmse" in call.args[0] for call in mock_mlflow.log_metric.call_args_list)
        assert any("test_mae" in call.args[0] for call in mock_mlflow.log_metric.call_args_list)
        assert any("test_r2" in call.args[0] for call in mock_mlflow.log_metric.call_args_list)

    @patch('src.mlops.components.artifacts.mlflow')
    def test_log_experiment_metrics_with_provided_predictions(self, mock_mlflow):
        """予測値が提供された場合のテスト"""
        mock_pipeline = Mock()
        mock_pipeline.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7], [0.1, 0.9]])

        X_train, y_train = create_binary_classification_data(n_rows=20, n_features=3)
        X_test = pd.DataFrame(np.random.randn(3, 3), columns=['feature_0', 'feature_1', 'feature_2'])
        y_test = np.array([0, 1, 1])
        y_pred = np.array([0, 1, 1])  # 提供された予測値

        cfg = DictConfig({
            "optuna": {
                "scoring": {
                    "classification": "f1_binary"
                }
            }
        })
        cv_scores = np.array([0.85, 0.87, 0.84])

        # MLflow runをモック
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id_12345"
        mock_mlflow.active_run.return_value = mock_run

        log_experiment_metrics(mock_pipeline, X_train, y_train, X_test, y_test,
                             "classification", cv_scores, cfg, y_pred=y_pred)

        # pipeline.predictが呼ばれないことを確認
        mock_pipeline.predict.assert_not_called()


class TestCreatePredictionDataframe:
    """create_prediction_dataframe 関数の単体テスト"""

    def test_create_classification_dataframe(self):
        """分類予測DataFrameの作成テスト"""
        # Mock pipeline設定
        mock_pipeline = Mock()
        X_test_transformed = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        mock_pipeline.__getitem__.return_value.transform.return_value = X_test_transformed
        mock_pipeline.predict.return_value = np.array([0, 1, 1])
        mock_pipeline.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7], [0.1, 0.9]])

        X_test = pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6]]), columns=['feature_0', 'feature_1'])
        y_test = pd.Series([0, 1, 1])

        with patch('src.mlops.components.artifacts.get_pipeline_feature_names') as mock_get_names:
            mock_get_names.return_value = ['feature_0', 'feature_1']

            df_result = create_prediction_dataframe(mock_pipeline, X_test, y_test, "classification")

        # アサーション
        assert 'y_true' in df_result.columns
        assert 'y_pred' in df_result.columns
        assert 'proba_class_0' in df_result.columns
        assert 'proba_class_1' in df_result.columns
        assert 'prediction_confidence' in df_result.columns

        assert len(df_result) == 3
        assert df_result['y_pred'].tolist() == [0, 1, 1]

    def test_create_regression_dataframe(self):
        """回帰予測DataFrameの作成テスト"""
        # Mock pipeline設定
        mock_pipeline = Mock()
        X_test_transformed = np.array([[1.5, 2.5], [3.5, 4.5]])
        mock_pipeline.__getitem__.return_value.transform.return_value = X_test_transformed
        mock_pipeline.predict.return_value = np.array([10.5, 20.3])

        X_test = pd.DataFrame(np.array([[1, 2], [3, 4]]), columns=['feature_0', 'feature_1'])
        y_test = pd.Series([10.2, 20.1])

        with patch('src.mlops.components.artifacts.get_pipeline_feature_names') as mock_get_names:
            mock_get_names.return_value = ['feature_0', 'feature_1']

            df_result = create_prediction_dataframe(mock_pipeline, X_test, y_test, "regression")

        # アサーション
        assert 'y_true' in df_result.columns
        assert 'y_pred' in df_result.columns
        assert 'proba_class_0' not in df_result.columns  # 回帰では確率なし

        assert len(df_result) == 2
        assert df_result['y_pred'].tolist() == [10.5, 20.3]

    def test_create_dataframe_with_provided_predictions(self):
        """提供された予測値でのDataFrame作成テスト"""
        mock_pipeline = Mock()
        X_test_transformed = np.array([[1.0, 2.0]])
        mock_pipeline.__getitem__.return_value.transform.return_value = X_test_transformed

        X_test = pd.DataFrame(np.array([[1, 2]]), columns=['feature_0', 'feature_1'])
        y_test = pd.Series([1])
        y_pred = np.array([0])  # 提供された予測値

        with patch('src.mlops.components.artifacts.get_pipeline_feature_names') as mock_get_names:
            mock_get_names.return_value = ['feature_0', 'feature_1']

            df_result = create_prediction_dataframe(mock_pipeline, X_test, y_test,
                                                  "classification", y_pred=y_pred)

        # pipeline.predictが呼ばれないことを確認
        mock_pipeline.predict.assert_not_called()
        assert df_result['y_pred'].iloc[0] == 0


class TestSavePredictionResults:
    """save_prediction_results 関数の単体テスト"""

    @patch('src.mlops.components.artifacts.mlflow')
    @patch('tempfile.TemporaryDirectory')
    def test_save_prediction_results(self, mock_temp_dir, mock_mlflow):
        """予測結果保存テスト"""
        # Mock設定
        mock_temp_dir.return_value.__enter__.return_value = "/tmp/test_dir"

        df_predictions = pd.DataFrame({
            'feature_0': [1, 2, 3],
            'feature_1': [4, 5, 6],
            'y_true': [0, 1, 1],
            'y_pred': [0, 1, 0]
        })

        cfg = DictConfig({})

        with patch.object(pd.DataFrame, 'to_csv') as mock_to_csv:
            result_df = save_prediction_results(df_predictions, cfg)

        # アサーション
        mock_to_csv.assert_called_once_with("/tmp/test_dir/test_predictions.csv", index=False)
        mock_mlflow.log_artifact.assert_called_once_with("/tmp/test_dir/test_predictions.csv", "predictions")
        assert result_df.equals(df_predictions)


class TestSetupMlflowExperiment:
    """setup_mlflow_experiment 関数の単体テスト"""

    @patch('src.mlops.components.artifacts.mlflow')
    def test_setup_with_description(self, mock_mlflow):
        """説明付き実験セットアップテスト"""
        mock_experiment = Mock()
        mock_mlflow.set_experiment.return_value = mock_experiment

        cfg = DictConfig({
            "mlflow": {
                "experiment_name": "test_experiment",
                "description": "Test experiment description"
            }
        })

        setup_mlflow_experiment(cfg)

        mock_mlflow.set_experiment.assert_called_once_with("test_experiment")
        mock_mlflow.set_experiment_tag.assert_called_once_with("description", "Test experiment description")

    @patch('src.mlops.components.artifacts.mlflow')
    def test_setup_without_description(self, mock_mlflow):
        """説明なし実験セットアップテスト"""
        cfg = DictConfig({
            "mlflow": {
                "experiment_name": "test_experiment"
            }
        })

        setup_mlflow_experiment(cfg)

        mock_mlflow.set_experiment.assert_called_once_with("test_experiment")
        mock_mlflow.set_experiment_tag.assert_not_called()


class TestSetMlflowTags:
    """set_mlflow_tags 関数の単体テスト"""

    @patch('src.mlops.components.artifacts.mlflow')
    def test_set_mlflow_tags(self, mock_mlflow):
        """MLflowタグ設定テスト"""
        cfg = DictConfig({
            "mlflow": {
                "tags": {
                    "environment": "testing",
                    "version": "v1.0.0",
                    "team": "data_science"
                }
            }
        })

        set_mlflow_tags(cfg)

        mock_mlflow.set_tag.assert_any_call("environment", "testing")
        mock_mlflow.set_tag.assert_any_call("version", "v1.0.0")
        mock_mlflow.set_tag.assert_any_call("team", "data_science")

    @patch('src.mlops.components.artifacts.mlflow')
    def test_set_mlflow_tags_no_tags(self, mock_mlflow):
        """タグなしの場合のテスト"""
        cfg = DictConfig({
            "mlflow": {}
        })

        set_mlflow_tags(cfg)

        mock_mlflow.set_tag.assert_not_called()

    @patch('src.mlops.components.artifacts.mlflow')
    def test_set_mlflow_tags_empty_tags(self, mock_mlflow):
        """空のタグの場合のテスト"""
        cfg = DictConfig({
            "mlflow": {
                "tags": {}
            }
        })

        set_mlflow_tags(cfg)

        mock_mlflow.set_tag.assert_not_called()