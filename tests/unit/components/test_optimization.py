"""optimization.py の単体テスト"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from omegaconf import DictConfig

from src.mlops.components.optimization import OptunaOptimizer
from tests.fixtures.test_data import create_binary_classification_data, create_regression_data


class TestOptunaOptimizer:
    """OptunaOptimizer クラスの単体テスト"""

    def test_init_classification(self):
        """分類タスクでの初期化テスト"""
        X, y = create_binary_classification_data(n_rows=50, n_features=5)
        cfg = DictConfig({
            "optuna": {
                "scoring": {
                    "classification": "f1_weighted",
                    "regression": "neg_mean_squared_error"
                },
                "n_trials": 10,
                "direction": "maximize",
                "study_name": "test_study"
            }
        })

        optimizer = OptunaOptimizer(cfg, X, y, "classification")

        assert optimizer.cfg == cfg
        assert optimizer.X_train is X
        assert optimizer.y_train is y
        assert optimizer.task_type == "classification"

    def test_init_regression(self):
        """回帰タスクでの初期化テスト"""
        X, y = create_regression_data(n_rows=50, n_features=5)
        cfg = DictConfig({
            "optuna": {
                "scoring": {
                    "classification": "f1_weighted",
                    "regression": "neg_mean_squared_error"
                },
                "n_trials": 10,
                "direction": "minimize",
                "study_name": "test_regression_study"
            }
        })

        optimizer = OptunaOptimizer(cfg, X, y, "regression")

        assert optimizer.cfg == cfg
        assert optimizer.X_train is X
        assert optimizer.y_train is y
        assert optimizer.task_type == "regression"

    @patch('src.mlops.components.optimization.create_pipeline')
    @patch('src.mlops.components.optimization.create_cv_strategy')
    @patch('src.mlops.components.optimization.cross_val_score')
    def test_objective_classification(self, mock_cv_score, mock_cv_strategy, mock_create_pipeline):
        """分類タスクでの目的関数テスト"""
        X, y = create_binary_classification_data(n_rows=50, n_features=5)
        cfg = DictConfig({
            "optuna": {
                "scoring": {
                    "classification": "f1_weighted",
                    "regression": "neg_mean_squared_error"
                },
                "direction": "maximize"
            }
        })

        # Mock設定
        mock_pipeline = Mock()
        mock_create_pipeline.return_value = mock_pipeline

        mock_cv_strategy_instance = Mock()
        mock_cv_strategy.return_value = mock_cv_strategy_instance

        mock_cv_score.return_value = np.array([0.8, 0.85, 0.82])

        optimizer = OptunaOptimizer(cfg, X, y, "classification")
        mock_trial = Mock()

        result = optimizer.objective(mock_trial)

        # アサーション
        mock_create_pipeline.assert_called_once_with(cfg, trial=mock_trial)
        mock_cv_strategy.assert_called_once_with(cfg)
        mock_cv_score.assert_called_once_with(
            mock_pipeline, X, y,
            cv=mock_cv_strategy_instance,
            scoring="f1_weighted",
            n_jobs=-1
        )

        expected_mean = np.array([0.8, 0.85, 0.82]).mean()
        assert result == expected_mean

    @patch('src.mlops.components.optimization.create_pipeline')
    @patch('src.mlops.components.optimization.create_cv_strategy')
    @patch('src.mlops.components.optimization.cross_val_score')
    def test_objective_regression(self, mock_cv_score, mock_cv_strategy, mock_create_pipeline):
        """回帰タスクでの目的関数テスト"""
        X, y = create_regression_data(n_rows=50, n_features=5)
        cfg = DictConfig({
            "optuna": {
                "scoring": {
                    "classification": "f1_weighted",
                    "regression": "neg_mean_squared_error"
                },
                "direction": "minimize"
            }
        })

        # Mock設定
        mock_pipeline = Mock()
        mock_create_pipeline.return_value = mock_pipeline

        mock_cv_strategy_instance = Mock()
        mock_cv_strategy.return_value = mock_cv_strategy_instance

        mock_cv_score.return_value = np.array([-1.2, -1.5, -1.1])

        optimizer = OptunaOptimizer(cfg, X, y, "regression")
        mock_trial = Mock()

        result = optimizer.objective(mock_trial)

        # アサーション
        mock_create_pipeline.assert_called_once_with(cfg, trial=mock_trial)
        mock_cv_strategy.assert_called_once_with(cfg)
        mock_cv_score.assert_called_once_with(
            mock_pipeline, X, y,
            cv=mock_cv_strategy_instance,
            scoring="neg_mean_squared_error",
            n_jobs=-1
        )

        expected_mean = np.array([-1.2, -1.5, -1.1]).mean()
        assert result == expected_mean

    @patch('src.mlops.components.optimization.create_pipeline')
    @patch('src.mlops.components.optimization.create_cv_strategy')
    @patch('src.mlops.components.optimization.cross_val_score')
    def test_objective_exception_handling_maximize(self, mock_cv_score, mock_cv_strategy, mock_create_pipeline):
        """例外処理テスト（maximize方向）"""
        X, y = create_binary_classification_data(n_rows=50, n_features=5)
        cfg = DictConfig({
            "optuna": {
                "scoring": {
                    "classification": "f1_weighted",
                    "regression": "neg_mean_squared_error"
                },
                "direction": "maximize"
            }
        })

        # Mock設定（例外発生）
        mock_create_pipeline.side_effect = Exception("Pipeline creation failed")

        optimizer = OptunaOptimizer(cfg, X, y, "classification")
        mock_trial = Mock()

        result = optimizer.objective(mock_trial)

        # maximize時は-infを返すことを確認
        assert result == float('-inf')

    @patch('src.mlops.components.optimization.create_pipeline')
    @patch('src.mlops.components.optimization.create_cv_strategy')
    @patch('src.mlops.components.optimization.cross_val_score')
    def test_objective_exception_handling_minimize(self, mock_cv_score, mock_cv_strategy, mock_create_pipeline):
        """例外処理テスト（minimize方向）"""
        X, y = create_regression_data(n_rows=50, n_features=5)
        cfg = DictConfig({
            "optuna": {
                "scoring": {
                    "classification": "f1_weighted",
                    "regression": "neg_mean_squared_error"
                },
                "direction": "minimize"
            }
        })

        # Mock設定（例外発生）
        mock_create_pipeline.side_effect = Exception("Pipeline creation failed")

        optimizer = OptunaOptimizer(cfg, X, y, "regression")
        mock_trial = Mock()

        result = optimizer.objective(mock_trial)

        # minimize時は+infを返すことを確認
        assert result == float('inf')

    @patch('optuna.create_study')
    def test_optimize(self, mock_create_study):
        """最適化実行テスト"""
        X, y = create_binary_classification_data(n_rows=50, n_features=5)
        cfg = DictConfig({
            "optuna": {
                "scoring": {
                    "classification": "f1_weighted",
                    "regression": "neg_mean_squared_error"
                },
                "n_trials": 5,
                "direction": "maximize",
                "study_name": "test_optimization"
            }
        })

        # Mock study設定
        mock_study = Mock()
        mock_study.best_params = {"n_estimators": 100, "max_depth": 5}
        mock_study.best_value = 0.85
        mock_create_study.return_value = mock_study

        optimizer = OptunaOptimizer(cfg, X, y, "classification")

        best_params, best_value = optimizer.optimize()

        # アサーション
        mock_create_study.assert_called_once_with(
            direction="maximize",
            study_name="test_optimization"
        )
        mock_study.optimize.assert_called_once_with(optimizer.objective, n_trials=5)

        assert best_params == {"n_estimators": 100, "max_depth": 5}
        assert best_value == 0.85

    @patch('src.mlops.components.optimization.create_pipeline')
    @patch('src.mlops.components.optimization.create_cv_strategy')
    @patch('src.mlops.components.optimization.cross_val_score')
    @patch('optuna.create_study')
    def test_full_optimization_workflow(self, mock_create_study, mock_cv_score,
                                      mock_cv_strategy, mock_create_pipeline):
        """完全な最適化ワークフローテスト"""
        X, y = create_binary_classification_data(n_rows=100, n_features=8)
        cfg = DictConfig({
            "optuna": {
                "scoring": {
                    "classification": "accuracy",
                    "regression": "neg_mean_squared_error"
                },
                "n_trials": 3,
                "direction": "maximize",
                "study_name": "full_test"
            }
        })

        # Mock設定
        mock_pipeline = Mock()
        mock_create_pipeline.return_value = mock_pipeline

        mock_cv_strategy_instance = Mock()
        mock_cv_strategy.return_value = mock_cv_strategy_instance

        # CV結果をランダムに設定
        mock_cv_score.side_effect = [
            np.array([0.8, 0.82, 0.85]),
            np.array([0.83, 0.81, 0.84]),
            np.array([0.86, 0.88, 0.87])
        ]

        mock_study = Mock()
        mock_study.best_params = {"max_depth": 8}
        mock_study.best_value = 0.87
        mock_create_study.return_value = mock_study

        optimizer = OptunaOptimizer(cfg, X, y, "classification")
        best_params, best_value = optimizer.optimize()

        # 複数回の最適化実行を確認
        mock_study.optimize.assert_called_once_with(optimizer.objective, n_trials=3)
        assert best_params == {"max_depth": 8}
        assert best_value == 0.87

    def test_parameter_validation(self):
        """パラメータバリデーションテスト"""
        X, y = create_binary_classification_data(n_rows=30, n_features=3)

        # 不正なタスクタイプ
        cfg = DictConfig({
            "optuna": {
                "scoring": {
                    "classification": "f1_weighted",
                    "regression": "neg_mean_squared_error"
                },
                "n_trials": 1,
                "direction": "maximize",
                "study_name": "test"
            }
        })

        # 無効なタスクタイプでも初期化はできるが、objective関数で適切に処理される
        optimizer = OptunaOptimizer(cfg, X, y, "invalid_task_type")
        assert optimizer.task_type == "invalid_task_type"

    @patch('src.mlops.components.optimization.create_pipeline')
    @patch('src.mlops.components.optimization.create_cv_strategy')
    @patch('src.mlops.components.optimization.cross_val_score')
    def test_edge_cases(self, mock_cv_score, mock_cv_strategy, mock_create_pipeline):
        """エッジケーステスト"""
        X, y = create_binary_classification_data(n_rows=10, n_features=2)
        cfg = DictConfig({
            "optuna": {
                "scoring": {
                    "classification": "f1_weighted",
                    "regression": "neg_mean_squared_error"
                },
                "direction": "maximize"
            }
        })

        # Mock設定
        mock_pipeline = Mock()
        mock_create_pipeline.return_value = mock_pipeline

        mock_cv_strategy_instance = Mock()
        mock_cv_strategy.return_value = mock_cv_strategy_instance

        optimizer = OptunaOptimizer(cfg, X, y, "classification")
        mock_trial = Mock()

        # 空のCV結果
        mock_cv_score.return_value = np.array([])
        result = optimizer.objective(mock_trial)
        assert np.isnan(result)  # 空配列の平均はNaN

        # 単一のCV結果
        mock_cv_score.return_value = np.array([0.75])
        result = optimizer.objective(mock_trial)
        assert result == 0.75

    def test_different_scoring_metrics(self):
        """異なるスコアリング指標でのテスト"""
        X, y = create_binary_classification_data(n_rows=50, n_features=5)

        # 異なるスコアリング指標を設定
        scoring_configs = [
            {"classification": "precision", "regression": "neg_mean_squared_error"},
            {"classification": "recall", "regression": "neg_mean_absolute_error"},
            {"classification": "roc_auc", "regression": "r2"}
        ]

        for scoring_cfg in scoring_configs:
            cfg = DictConfig({
                "optuna": {
                    "scoring": scoring_cfg,
                    "n_trials": 1,
                    "direction": "maximize",
                    "study_name": "scoring_test"
                }
            })

            optimizer = OptunaOptimizer(cfg, X, y, "classification")
            assert optimizer.cfg.optuna.scoring.classification == scoring_cfg["classification"]

            # 回帰でも同様にテスト
            X_reg, y_reg = create_regression_data(n_rows=50, n_features=5)
            optimizer_reg = OptunaOptimizer(cfg, X_reg, y_reg, "regression")
            assert optimizer_reg.cfg.optuna.scoring.regression == scoring_cfg["regression"]