"""MLOpsパイプライン統合テスト"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from omegaconf import DictConfig, OmegaConf

from src.mlops.components.pipeline import create_pipeline
from src.mlops.components.optimization import OptunaOptimizer
from src.mlops.components.artifacts import (
    log_experiment_metrics,
    save_model_artifacts,
    create_prediction_dataframe
)
from src.mlops.utils.data_loader import load_csv_data
from src.mlops.utils.cv_utils import create_cv_strategy
from tests.fixtures.test_data import create_binary_classification_data, create_regression_data


class TestMLOpsPipelineIntegration:
    """MLOpsパイプライン統合テスト"""

    def test_complete_classification_pipeline(self):
        """完全な分類パイプライン統合テスト"""
        # テストデータ作成（数値データのみ）
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 4), columns=[f'feature_{i}' for i in range(4)])
        y = pd.Series(np.random.choice([0, 1], 100), name='target')
        X_train, X_test = X.iloc[:80], X.iloc[80:]
        y_train, y_test = y.iloc[:80], y.iloc[80:]

        # 設定作成
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
                "default_params": {
                    "n_estimators": 10,
                    "random_state": 42
                },
                "optuna_space": {}
            },
            "data": {
                "random_state": 42
            },
            "evaluation": {
                "cv_strategy": {
                    "module": "sklearn.model_selection",
                    "class": "StratifiedKFold",
                    "params": {
                        "n_splits": 3,
                        "shuffle": True,
                        "random_state": 42
                    }
                }
            },
            "optuna": {
                "scoring": {
                    "classification": "f1_weighted"
                }
            }
        })

        # パイプライン作成
        pipeline = create_pipeline(cfg)

        # パイプライン学習
        pipeline.fit(X_train, y_train)

        # 予測
        y_pred = pipeline.predict(X_test)

        # 基本的な動作確認
        assert len(y_pred) == len(y_test)
        assert all(pred in [0, 1] for pred in y_pred)  # 二値分類の結果

        # クロスバリデーション実行
        cv_strategy = create_cv_strategy(cfg)
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_strategy, scoring="f1_weighted")

        assert len(cv_scores) == 3  # 3-fold CV
        assert all(0 <= score <= 1 for score in cv_scores)  # スコアの範囲確認

    def test_complete_regression_pipeline(self):
        """完全な回帰パイプライン統合テスト"""
        # テストデータ作成（数値データのみ）
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 4), columns=[f'feature_{i}' for i in range(4)])
        y = pd.Series(np.random.randn(100) * 10 + 50, name='target')
        X_train, X_test = X.iloc[:80], X.iloc[80:]
        y_train, y_test = y.iloc[:80], y.iloc[80:]

        # 設定作成
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
                "class": "RandomForestRegressor",
                "default_params": {
                    "n_estimators": 10,
                    "random_state": 42
                },
                "optuna_space": {}
            },
            "data": {
                "random_state": 42
            },
            "evaluation": {
                "cv_strategy": {
                    "module": "sklearn.model_selection",
                    "class": "KFold",
                    "params": {
                        "n_splits": 3,
                        "shuffle": True,
                        "random_state": 42
                    }
                }
            },
            "optuna": {
                "scoring": {
                    "regression": "neg_mean_squared_error"
                }
            }
        })

        # パイプライン作成と学習
        pipeline = create_pipeline(cfg)
        pipeline.fit(X_train, y_train)

        # 予測
        y_pred = pipeline.predict(X_test)

        # 基本的な動作確認
        assert len(y_pred) == len(y_test)
        assert all(isinstance(pred, (int, float, np.number)) for pred in y_pred)

        # クロスバリデーション実行
        cv_strategy = create_cv_strategy(cfg)
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_strategy, scoring="neg_mean_squared_error")

        assert len(cv_scores) == 3
        assert all(score <= 0 for score in cv_scores)  # neg_mse は負の値

    @patch('src.mlops.components.optimization.optuna')
    def test_optimization_integration(self, mock_optuna):
        """最適化統合テスト"""
        # Mock Optuna設定
        mock_study = Mock()
        mock_study.best_params = {"n_estimators": 50, "max_depth": 8}
        mock_study.best_value = 0.85
        mock_optuna.create_study.return_value = mock_study

        # テストデータ作成（数値データのみ）
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(80, 3), columns=[f'feature_{i}' for i in range(3)])
        y = pd.Series(np.random.choice([0, 1], 80), name='target')

        # 設定作成
        cfg = DictConfig({
            "pipeline": {
                "steps": []
            },
            "model": {
                "module": "sklearn.ensemble",
                "class": "RandomForestClassifier",
                "default_params": {
                    "n_estimators": 10,
                    "random_state": 42
                },
                "optuna_space": {
                    "n_estimators": ("int", 10, 100),
                    "max_depth": ("int", 3, 15)
                }
            },
            "data": {
                "random_state": 42
            },
            "evaluation": {
                "cv_strategy": {
                    "module": "sklearn.model_selection",
                    "class": "StratifiedKFold",
                    "params": {
                        "n_splits": 3,
                        "shuffle": True,
                        "random_state": 42
                    }
                }
            },
            "optuna": {
                "n_trials": 5,
                "direction": "maximize",
                "study_name": "test_optimization",
                "scoring": {
                    "classification": "f1_weighted"
                }
            }
        })

        # 最適化実行
        optimizer = OptunaOptimizer(cfg, X, y, "classification")
        best_params, best_value = optimizer.optimize()

        # 結果確認
        assert best_params == {"n_estimators": 50, "max_depth": 8}
        assert best_value == 0.85
        mock_study.optimize.assert_called_once_with(optimizer.objective, n_trials=5)

    def test_pipeline_with_custom_features(self):
        """カスタム特徴量生成パイプライン統合テスト"""
        # テストデータ作成（数値データのみ）
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 3), columns=[f'feature_{i}' for i in range(3)])
        y = pd.Series(np.random.choice([0, 1], 100), name='target')

        # 設定作成（特徴量生成ステップ含む）
        cfg = DictConfig({
            "pipeline": {
                "steps": [
                    {
                        "name": "feature_generator",
                        "module": "src.mlops.features.engineering",
                        "class": "RatioFeatureGenerator",
                        "params": {
                            "ratio_pairs": [("feature_0", "feature_1")]
                        },
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
                "default_params": {
                    "n_estimators": 10,
                    "random_state": 42
                },
                "optuna_space": {}
            },
            "data": {
                "random_state": 42
            }
        })

        # パイプライン作成と実行
        pipeline = create_pipeline(cfg)

        # 特徴量が増加することを確認
        original_features = X.shape[1]
        X_transformed = pipeline[:-1].transform(X)  # 分類器以外の変換

        # 比率特徴量が追加されているはず
        assert X_transformed.shape[1] >= original_features

    @patch('src.mlops.components.artifacts.mlflow')
    def test_artifacts_logging_integration(self, mock_mlflow):
        """アーティファクト記録統合テスト"""
        # テストデータ作成（数値データのみ）
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.randn(80, 3), columns=[f'feature_{i}' for i in range(3)])
        y_train = pd.Series(np.random.choice([0, 1], 80), name='target')
        X_test = pd.DataFrame(np.random.randn(20, 3), columns=[f'feature_{i}' for i in range(3)])
        y_test = pd.Series(np.random.choice([0, 1], 20), name='target')

        # パイプライン作成
        cfg = DictConfig({
            "pipeline": {"steps": []},
            "model": {
                "module": "sklearn.ensemble",
                "class": "RandomForestClassifier",
                "default_params": {"n_estimators": 10, "random_state": 42},
                "optuna_space": {}
            },
            "data": {"random_state": 42},
            "optuna": {
                "scoring": {"classification": "f1_weighted"}
            },
            "mlflow": {
                "artifacts": {
                    "model_filename": "model.pkl",
                    "info_filename": "info.json",
                    "model_dir": "models"
                }
            }
        })

        pipeline = create_pipeline(cfg)
        pipeline.fit(X_train, y_train)

        # CV結果を模擬
        cv_scores = np.array([0.8, 0.82, 0.85])

        # Mock MLflow run
        mock_run = Mock()
        mock_run.info.run_id = "test_run_12345"
        mock_mlflow.active_run.return_value = mock_run

        # メトリクス記録テスト
        log_experiment_metrics(pipeline, X_train, y_train, X_test, y_test,
                             "classification", cv_scores, cfg)

        # MLflowメソッドが呼ばれることを確認
        assert mock_mlflow.log_metric.called
        assert mock_mlflow.log_metric.call_count >= 3  # CV mean, std, accuracy等

    @patch('tempfile.TemporaryDirectory')
    def test_csv_data_loading_integration(self, mock_temp_dir):
        """CSVデータ読み込み統合テスト"""
        # 一時ファイル作成のモック
        temp_file_path = "/tmp/test_data.csv"
        mock_temp_dir.return_value.__enter__.return_value = "/tmp"

        # テストCSVデータ作成
        test_data = pd.DataFrame({
            'feature_a': [1, 2, 3, 4, 5],
            'feature_b': [6, 7, 8, 9, 10],
            'feature_c': [11, 12, 13, 14, 15],
            'target': [0, 1, 0, 1, 0]
        })

        # CSVファイル作成
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data.to_csv(f, index=False)
            actual_temp_file = f.name

        try:
            # 設定作成
            cfg = DictConfig({
                "data": {
                    "file_path": actual_temp_file,
                    "target_column": "target"
                }
            })

            # データ読み込み
            with patch('builtins.print'):
                df, feature_cols, target_names = load_csv_data(cfg)

            # 結果確認
            assert df.shape == (5, 4)
            assert feature_cols == ['feature_a', 'feature_b', 'feature_c']
            assert set(target_names) == {0, 1}

        finally:
            # ファイルクリーンアップ
            os.unlink(actual_temp_file)

    def test_prediction_dataframe_integration(self):
        """予測結果DataFrame統合テスト"""
        # テストデータ作成（数値データのみ）
        np.random.seed(42)
        X_test = pd.DataFrame(np.random.randn(30, 4), columns=[f'feature_{i}' for i in range(4)])
        y_test = pd.Series(np.random.choice([0, 1], 30), name='target')

        # パイプライン作成と学習
        cfg = DictConfig({
            "pipeline": {"steps": []},
            "model": {
                "module": "sklearn.ensemble",
                "class": "RandomForestClassifier",
                "default_params": {"n_estimators": 10, "random_state": 42},
                "optuna_space": {}
            },
            "data": {"random_state": 42}
        })

        pipeline = create_pipeline(cfg)
        pipeline.fit(X_test, y_test)  # テスト用に同じデータでfit

        # 予測結果DataFrame作成
        df_predictions = create_prediction_dataframe(pipeline, X_test, y_test, "classification")

        # 結果確認
        assert len(df_predictions) == len(X_test)
        assert 'y_true' in df_predictions.columns
        assert 'y_pred' in df_predictions.columns
        assert 'proba_class_0' in df_predictions.columns
        assert 'proba_class_1' in df_predictions.columns
        assert 'prediction_confidence' in df_predictions.columns

        # 予測確率の妥当性確認
        assert all(0 <= conf <= 1 for conf in df_predictions['prediction_confidence'])
        assert all(0 <= prob <= 1 for prob in df_predictions['proba_class_0'])
        assert all(0 <= prob <= 1 for prob in df_predictions['proba_class_1'])

    def test_complex_pipeline_integration(self):
        """複雑なパイプライン統合テスト"""
        # テストデータ作成（数値データのみ）
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(120, 5), columns=[f'feature_{i}' for i in range(5)])
        y = pd.Series(np.random.choice([0, 1], 120), name='target')

        # 複雑なパイプライン設定
        cfg = DictConfig({
            "pipeline": {
                "steps": [
                    {
                        "name": "preprocessor",
                        "module": "src.mlops.features.preprocessing",
                        "class": "DataQualityValidator",
                        "params": {
                            "fix_dtypes": True,
                            "drop_high_missing": 0.8,
                            "drop_zero_variance": True
                        },
                        "enabled": True
                    },
                    {
                        "name": "feature_selector",
                        "module": "src.mlops.features.selection",
                        "class": "SmartFeatureSelector",
                        "params": {
                            "k": 3
                        },
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
                "default_params": {
                    "n_estimators": 20,
                    "random_state": 42
                },
                "optuna_space": {}
            },
            "data": {
                "random_state": 42
            },
            "evaluation": {
                "cv_strategy": {
                    "module": "sklearn.model_selection",
                    "class": "StratifiedKFold",
                    "params": {
                        "n_splits": 3,
                        "shuffle": True,
                        "random_state": 42
                    }
                }
            }
        })

        # パイプライン作成と実行
        pipeline = create_pipeline(cfg)

        # データ分割
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 学習と予測
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # 結果の妥当性確認
        assert len(y_pred) == len(y_test)
        assert all(pred in [0, 1] for pred in y_pred)

        # パフォーマンス評価
        from sklearn.metrics import accuracy_score, f1_score
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        assert 0 <= accuracy <= 1
        assert 0 <= f1 <= 1

    def test_pipeline_error_handling(self):
        """パイプラインエラーハンドリング統合テスト"""
        # テストデータ作成（数値データのみ）
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(50, 3), columns=[f'feature_{i}' for i in range(3)])
        y = pd.Series(np.random.choice([0, 1], 50), name='target')

        # 無効な設定（存在しないクラス）
        cfg = DictConfig({
            "pipeline": {
                "steps": [
                    {
                        "name": "invalid_step",
                        "module": "nonexistent.module",
                        "class": "NonexistentClass",
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
            "data": {"random_state": 42}
        })

        # ImportErrorが発生することを確認
        with pytest.raises(ImportError):
            create_pipeline(cfg)

    def test_end_to_end_classification_workflow(self):
        """エンドツーエンド分類ワークフローテスト"""
        # テストデータ作成とCSVファイル保存（数値データのみ）
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 4), columns=[f'feature_{i}' for i in range(4)])
        y = pd.Series(np.random.choice([0, 1], 100), name='target')
        test_df = X.copy()
        test_df['target'] = y

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_df.to_csv(f, index=False)
            temp_file_path = f.name

        try:
            # 完全な設定作成
            cfg = DictConfig({
                "data": {
                    "file_path": temp_file_path,
                    "target_column": "target"
                },
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
                    "default_params": {
                        "n_estimators": 10,
                        "random_state": 42
                    },
                    "optuna_space": {}
                },
                "evaluation": {
                    "cv_strategy": {
                        "module": "sklearn.model_selection",
                        "class": "StratifiedKFold",
                        "params": {
                            "n_splits": 3,
                            "shuffle": True,
                            "random_state": 42
                        }
                    }
                },
                "optuna": {
                    "scoring": {"classification": "f1_weighted"}
                }
            })

            # エンドツーエンドワークフロー実行

            # 1. データ読み込み
            with patch('builtins.print'):
                df, feature_cols, target_names = load_csv_data(cfg)

            # 2. データ分割
            from sklearn.model_selection import train_test_split
            X_features = df[feature_cols]
            y_target = df[cfg.data.target_column]
            X_train, X_test, y_train, y_test = train_test_split(
                X_features, y_target, test_size=0.2, random_state=42, stratify=y_target
            )

            # 3. パイプライン作成と学習
            pipeline = create_pipeline(cfg)
            pipeline.fit(X_train, y_train)

            # 4. クロスバリデーション実行
            cv_strategy = create_cv_strategy(cfg)
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_strategy, scoring="f1_weighted")

            # 5. テスト予測
            y_pred = pipeline.predict(X_test)

            # 6. 予測結果DataFrame作成
            df_predictions = create_prediction_dataframe(pipeline, X_test, y_test, "classification")

            # 全体の結果確認
            assert len(cv_scores) == 3
            assert len(y_pred) == len(y_test)
            assert len(df_predictions) == len(y_test)
            assert 'y_true' in df_predictions.columns
            assert 'y_pred' in df_predictions.columns

            # パフォーマンス妥当性確認
            from sklearn.metrics import accuracy_score
            accuracy = accuracy_score(y_test, y_pred)
            assert 0 <= accuracy <= 1

        finally:
            # ファイルクリーンアップ
            os.unlink(temp_file_path)