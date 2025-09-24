"""pipeline.py の単体テスト"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from omegaconf import DictConfig, OmegaConf

from src.mlops.components.pipeline import (
    generate_optuna_params,
    create_pipeline_step,
    create_pipeline,
    _is_sampler_step,
    SAMPLING_MODULES,
    SAMPLING_CLASSES
)
from tests.fixtures.test_data import create_binary_classification_data


class TestGenerateOptunaParams:
    """generate_optuna_params 関数の単体テスト"""

    def test_int_parameter(self):
        """整数パラメータの生成テスト"""
        mock_trial = Mock()
        mock_trial.suggest_int.return_value = 50

        optuna_space = {"n_estimators": ("int", 10, 100)}
        params = generate_optuna_params(mock_trial, optuna_space)

        assert params == {"n_estimators": 50}
        mock_trial.suggest_int.assert_called_once_with("n_estimators", 10, 100)

    def test_float_parameter(self):
        """浮動小数点パラメータの生成テスト"""
        mock_trial = Mock()
        mock_trial.suggest_float.return_value = 0.5

        optuna_space = {"learning_rate": ("float", 0.01, 1.0)}
        params = generate_optuna_params(mock_trial, optuna_space)

        assert params == {"learning_rate": 0.5}
        mock_trial.suggest_float.assert_called_once_with("learning_rate", 0.01, 1.0)

    def test_categorical_parameter(self):
        """カテゴリカルパラメータの生成テスト"""
        mock_trial = Mock()
        mock_trial.suggest_categorical.return_value = "gini"

        optuna_space = {"criterion": ("categorical", ["gini", "entropy"])}
        params = generate_optuna_params(mock_trial, optuna_space)

        assert params == {"criterion": "gini"}
        mock_trial.suggest_categorical.assert_called_once_with("criterion", ["gini", "entropy"])

    def test_multiple_parameters(self):
        """複数パラメータの生成テスト"""
        mock_trial = Mock()
        mock_trial.suggest_int.return_value = 100
        mock_trial.suggest_float.return_value = 0.1
        mock_trial.suggest_categorical.return_value = "gini"

        optuna_space = {
            "n_estimators": ("int", 10, 200),
            "learning_rate": ("float", 0.01, 1.0),
            "criterion": ("categorical", ["gini", "entropy"])
        }

        params = generate_optuna_params(mock_trial, optuna_space)

        expected = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "criterion": "gini"
        }
        assert params == expected

    def test_empty_optuna_space(self):
        """空のOptuna設定のテスト"""
        mock_trial = Mock()
        params = generate_optuna_params(mock_trial, {})
        assert params == {}


class TestIsSamplerStep:
    """_is_sampler_step 関数の単体テスト"""

    def test_sampling_module_detection(self):
        """サンプリングモジュールの検出テスト"""
        step_config = DictConfig({
            "module": "imblearn.over_sampling",
            "class": "SMOTE"
        })
        assert _is_sampler_step(step_config) == True

    def test_sampling_class_detection(self):
        """サンプリングクラスの検出テスト"""
        step_config = DictConfig({
            "module": "other.module",
            "class": "SMOTE"
        })
        assert _is_sampler_step(step_config) == True

    def test_non_sampling_step(self):
        """非サンプリングステップの判定テスト"""
        step_config = DictConfig({
            "module": "sklearn.preprocessing",
            "class": "StandardScaler"
        })
        assert _is_sampler_step(step_config) == False

    def test_missing_class_key(self):
        """classキーが存在しない場合のテスト"""
        step_config = DictConfig({
            "module": "sklearn.preprocessing"
        })
        assert _is_sampler_step(step_config) == False


class TestCreatePipelineStep:
    """create_pipeline_step 関数の単体テスト"""

    @patch('src.mlops.components.pipeline.import_class')
    @patch('src.mlops.components.pipeline.resolve_function_references')
    def test_basic_step_creation(self, mock_resolve_func, mock_import_class):
        """基本的なステップ作成テスト"""
        # Mock設定
        mock_class = Mock()
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        mock_import_class.return_value = mock_class
        mock_resolve_func.return_value = {"param1": "value1"}

        step_config = DictConfig({
            "name": "scaler",
            "module": "sklearn.preprocessing",
            "class": "StandardScaler",
            "params": {"param1": "value1"}
        })

        result = create_pipeline_step(step_config)

        # アサーション
        mock_import_class.assert_called_once_with("sklearn.preprocessing", "StandardScaler")
        mock_resolve_func.assert_called_once_with({"param1": "value1"})
        mock_class.assert_called_once_with(param1="value1")
        assert result == mock_instance

    @patch('src.mlops.components.pipeline.import_class')
    @patch('src.mlops.components.pipeline.resolve_function_references')
    def test_step_with_columns(self, mock_resolve_func, mock_import_class):
        """カラム指定ありのステップ作成テスト"""
        from sklearn.compose import ColumnTransformer

        # Mock設定
        mock_class = Mock()
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        mock_import_class.return_value = mock_class
        mock_resolve_func.return_value = {}

        step_config = DictConfig({
            "name": "scaler",
            "module": "sklearn.preprocessing",
            "class": "StandardScaler",
            "columns": ["col1", "col2"]
        })

        result = create_pipeline_step(step_config)

        # ColumnTransformerでラップされることを確認
        assert isinstance(result, ColumnTransformer)
        assert len(result.transformers) == 1
        assert result.transformers[0][0] == "scaler"
        assert result.transformers[0][2] == ["col1", "col2"]

    @patch('src.mlops.components.pipeline.import_class')
    @patch('src.mlops.components.pipeline.resolve_function_references')
    def test_step_with_all_columns(self, mock_resolve_func, mock_import_class):
        """全カラム指定のステップ作成テスト"""
        # Mock設定
        mock_class = Mock()
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        mock_import_class.return_value = mock_class
        mock_resolve_func.return_value = {}

        step_config = DictConfig({
            "name": "scaler",
            "module": "sklearn.preprocessing",
            "class": "StandardScaler",
            "columns": "all"
        })

        result = create_pipeline_step(step_config)

        # ColumnTransformerでラップされないことを確認
        assert result == mock_instance

    @patch('src.mlops.components.pipeline.import_class')
    @patch('src.mlops.components.pipeline.resolve_function_references')
    def test_step_without_params(self, mock_resolve_func, mock_import_class):
        """パラメータなしのステップ作成テスト"""
        # Mock設定
        mock_class = Mock()
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        mock_import_class.return_value = mock_class
        mock_resolve_func.return_value = {}

        step_config = DictConfig({
            "name": "scaler",
            "module": "sklearn.preprocessing",
            "class": "StandardScaler"
        })

        result = create_pipeline_step(step_config)

        # パラメータなしでインスタンス化されることを確認
        mock_class.assert_called_once_with()
        assert result == mock_instance


class TestCreatePipeline:
    """create_pipeline 関数の単体テスト"""

    @patch('src.mlops.components.pipeline.create_pipeline_step')
    @patch('src.mlops.components.pipeline.import_class')
    @patch('src.mlops.components.pipeline._is_sampler_step')
    def test_basic_pipeline_creation(self, mock_is_sampler, mock_import_class, mock_create_step):
        """基本的なパイプライン作成テスト"""
        from sklearn.pipeline import Pipeline

        # Mock設定
        mock_is_sampler.return_value = False
        mock_step = Mock()
        mock_create_step.return_value = mock_step

        mock_model_class = Mock()
        mock_model_instance = Mock()
        mock_model_class.return_value = mock_model_instance
        mock_import_class.return_value = mock_model_class

        # パラメータでrandom_stateをサポート
        import inspect
        mock_signature = Mock()
        mock_signature.parameters = {"random_state": Mock()}
        with patch.object(inspect, 'signature', return_value=mock_signature):

            cfg = DictConfig({
                "pipeline": {
                    "steps": [{
                        "name": "scaler",
                        "module": "sklearn.preprocessing",
                        "class": "StandardScaler",
                        "enabled": True
                    }]
                },
                "model": {
                    "module": "sklearn.ensemble",
                    "class": "RandomForestClassifier",
                    "default_params": {"n_estimators": 100},
                    "optuna_space": {}
                },
                "data": {
                    "random_state": 42
                }
            })

            pipeline = create_pipeline(cfg)

        # アサーション
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0][0] == "scaler"
        assert pipeline.steps[1][0] == "classifier"
        mock_model_class.assert_called_once_with(n_estimators=100, random_state=42)

    @patch('src.mlops.components.pipeline.create_pipeline_step')
    @patch('src.mlops.components.pipeline.import_class')
    @patch('src.mlops.components.pipeline._is_sampler_step')
    def test_pipeline_with_sampler(self, mock_is_sampler, mock_import_class, mock_create_step):
        """サンプラー含むパイプライン作成テスト"""
        # Mock設定
        mock_is_sampler.side_effect = lambda x: x.name == "sampler"
        mock_step = Mock()
        mock_create_step.return_value = mock_step

        mock_model_class = Mock()
        mock_model_instance = Mock()
        mock_model_class.return_value = mock_model_instance
        mock_import_class.return_value = mock_model_class

        with patch('src.mlops.components.pipeline.IMBLEARN_AVAILABLE', True):
            with patch('src.mlops.components.pipeline.ImbPipeline') as mock_imbpipeline:
                mock_pipeline = Mock()
                mock_imbpipeline.return_value = mock_pipeline

                cfg = DictConfig({
                    "pipeline": {
                        "steps": [
                            {
                                "name": "sampler",
                                "module": "imblearn.over_sampling",
                                "class": "SMOTE",
                                "enabled": True
                            },
                            {
                                "name": "scaler",
                                "module": "sklearn.preprocessing",
                                "class": "StandardScaler",
                                "enabled": True
                            }
                        ]
                    },
                    "model": {
                        "module": "sklearn.ensemble",
                        "class": "RandomForestClassifier",
                        "default_params": {"n_estimators": 100},
                        "optuna_space": {}
                    },
                    "data": {
                        "random_state": 42
                    }
                })

                pipeline = create_pipeline(cfg)

        # ImbPipelineが使用されることを確認
        mock_imbpipeline.assert_called_once()

    @patch('src.mlops.components.pipeline.create_pipeline_step')
    @patch('src.mlops.components.pipeline.import_class')
    @patch('src.mlops.components.pipeline._is_sampler_step')
    def test_pipeline_with_disabled_step(self, mock_is_sampler, mock_import_class, mock_create_step):
        """無効化されたステップを含むパイプライン作成テスト"""
        from sklearn.pipeline import Pipeline

        # Mock設定
        mock_is_sampler.return_value = False
        mock_step = Mock()
        mock_create_step.return_value = mock_step

        mock_model_class = Mock()
        mock_model_instance = Mock()
        mock_model_class.return_value = mock_model_instance
        mock_import_class.return_value = mock_model_class

        cfg = DictConfig({
            "pipeline": {
                "steps": [
                    {
                        "name": "disabled_step",
                        "module": "sklearn.preprocessing",
                        "class": "StandardScaler",
                        "enabled": False
                    },
                    {
                        "name": "enabled_step",
                        "module": "sklearn.preprocessing",
                        "class": "MinMaxScaler",
                        "enabled": True
                    }
                ]
            },
            "model": {
                "module": "sklearn.ensemble",
                "class": "RandomForestClassifier",
                "default_params": {"n_estimators": 100},
                "optuna_space": {}
            },
            "data": {
                "random_state": 42
            }
        })

        pipeline = create_pipeline(cfg)

        # 無効化されたステップが除外されることを確認
        assert len(pipeline.steps) == 2  # enabled_step + classifier
        step_names = [name for name, _ in pipeline.steps]
        assert "disabled_step" not in step_names
        assert "enabled_step" in step_names
        assert "classifier" in step_names

    @patch('src.mlops.components.pipeline.generate_optuna_params')
    @patch('src.mlops.components.pipeline.create_pipeline_step')
    @patch('src.mlops.components.pipeline.import_class')
    @patch('src.mlops.components.pipeline._is_sampler_step')
    def test_pipeline_with_trial(self, mock_is_sampler, mock_import_class, mock_create_step, mock_generate_params):
        """Optunaトライアル付きパイプライン作成テスト"""
        # Mock設定
        mock_is_sampler.return_value = False
        mock_step = Mock()
        mock_create_step.return_value = mock_step

        mock_model_class = Mock()
        mock_model_instance = Mock()
        mock_model_class.return_value = mock_model_instance
        mock_import_class.return_value = mock_model_class

        mock_generate_params.return_value = {"n_estimators": 150}

        mock_trial = Mock()

        cfg = DictConfig({
            "pipeline": {
                "steps": [{
                    "name": "scaler",
                    "module": "sklearn.preprocessing",
                    "class": "StandardScaler",
                    "enabled": True
                }]
            },
            "model": {
                "module": "sklearn.ensemble",
                "class": "RandomForestClassifier",
                "default_params": {"n_estimators": 100},
                "optuna_space": {"n_estimators": ("int", 50, 200)}
            },
            "data": {
                "random_state": 42
            }
        })

        pipeline = create_pipeline(cfg, trial=mock_trial)

        # Optunaパラメータ生成が呼ばれることを確認
        mock_generate_params.assert_called_once()

    @patch('src.mlops.components.pipeline.create_pipeline_step')
    @patch('src.mlops.components.pipeline.import_class')
    @patch('src.mlops.components.pipeline._is_sampler_step')
    def test_pipeline_with_best_params(self, mock_is_sampler, mock_import_class, mock_create_step):
        """最適パラメータ付きパイプライン作成テスト"""
        # Mock設定
        mock_is_sampler.return_value = False
        mock_step = Mock()
        mock_create_step.return_value = mock_step

        mock_model_class = Mock()
        mock_model_instance = Mock()
        mock_model_class.return_value = mock_model_instance
        mock_import_class.return_value = mock_model_class

        best_params = {"n_estimators": 200, "max_depth": 10}

        cfg = DictConfig({
            "pipeline": {
                "steps": [{
                    "name": "scaler",
                    "module": "sklearn.preprocessing",
                    "class": "StandardScaler",
                    "enabled": True
                }]
            },
            "model": {
                "module": "sklearn.ensemble",
                "class": "RandomForestClassifier",
                "default_params": {"n_estimators": 100},
                "optuna_space": {}
            },
            "data": {
                "random_state": 42
            }
        })

        pipeline = create_pipeline(cfg, best_params=best_params)

        # best_paramsが使用されることを確認
        mock_model_class.assert_called_once_with(n_estimators=200, max_depth=10, random_state=42)

    def test_edge_cases(self):
        """エッジケーステスト"""
        # 空のステップリスト
        with patch('src.mlops.components.pipeline.import_class') as mock_import_class:
            mock_model_class = Mock()
            mock_model_instance = Mock()
            mock_model_class.return_value = mock_model_instance
            mock_import_class.return_value = mock_model_class

            cfg = DictConfig({
                "pipeline": {"steps": []},
                "model": {
                    "module": "sklearn.ensemble",
                    "class": "RandomForestClassifier",
                    "default_params": {},
                    "optuna_space": {}
                },
                "data": {"random_state": 42}
            })

            pipeline = create_pipeline(cfg)
            assert len(pipeline.steps) == 1  # classifierのみ


class TestPipelineIntegration:
    """パイプライン統合テスト"""

    def test_real_pipeline_creation(self):
        """実際のパイプライン作成統合テスト"""
        cfg = DictConfig({
            "pipeline": {
                "steps": [
                    {
                        "name": "scaler",
                        "module": "sklearn.preprocessing",
                        "class": "StandardScaler",
                        "enabled": True
                    }
                ]
            },
            "model": {
                "module": "sklearn.ensemble",
                "class": "RandomForestClassifier",
                "default_params": {"n_estimators": 10, "random_state": 42},
                "optuna_space": {}
            },
            "data": {
                "random_state": 42
            }
        })

        pipeline = create_pipeline(cfg)

        # 実際のパイプラインとして動作することを確認
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier

        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) == 2
        assert isinstance(pipeline.steps[0][1], StandardScaler)
        assert isinstance(pipeline.steps[1][1], RandomForestClassifier)

        # 簡単なデータでfit/predictテスト
        X, y = create_binary_classification_data(n_rows=50, n_features=5)
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)
        assert len(predictions) == len(y)